"""
CATalyze FastAPI server — orchestrates inspections, reports, and part ID.

Run from project root:
    uvicorn backend.main:app --reload
"""

import base64
import hashlib
import json
import os
import random
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()

from backend.modal_functions import analyze_image, analyze_live, analyze_voice, identify_part
from backend.supermemory import (
    get_all_inspection_results,
    get_inspection_history,
    save_inspection_result,
)
from prompts.inspection import MACHINES

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="CATalyze",
    version="0.1.0",
    description="AI-powered field inspection tool for heavy equipment",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class VoiceRequest(BaseModel):
    machine: str
    component: str
    voice_text: str
    inspector_id: str | None = None


class LiveRequest(BaseModel):
    image_base64: str


class RegisterRequest(BaseModel):
    email: str
    username: str
    password: str
    confirm_password: str


class LoginRequest(BaseModel):
    email: str
    password: str


class VerifyEmailRequest(BaseModel):
    email: str
    code: str


class ForgotPasswordRequest(BaseModel):
    email: str


class ResetPasswordRequest(BaseModel):
    email: str
    code: str
    new_password: str
    confirm_password: str


class ReportRequest(BaseModel):
    results: list[dict[str, Any]]
    machine: str
    inspector: str | None = None


class SaveInspectionRequest(BaseModel):
    machine: str
    component: str
    status: str
    observation: str | None = None
    confidence: float | None = None
    recommended_action: str | None = None
    maintenance_steps: list[str] | None = None
    timestamp: str | None = None
    inspector_id: str | None = None


# ---------------------------------------------------------------------------
# Auth — username + password (in-memory store, hashed with SHA-256)
# ---------------------------------------------------------------------------

USERS_FILE = Path(__file__).parent.parent / "users.json"


def _load_users() -> dict:
    if USERS_FILE.exists():
        return json.loads(USERS_FILE.read_text())
    return {}


def _save_users(users: dict) -> None:
    USERS_FILE.write_text(json.dumps(users, indent=2))


_users = _load_users()


def _hash_pw(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


_WELCOME_HTML = (
    '<div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; background-color: #121212; color: #ffffff; padding: 40px; border-radius: 12px;">'
    '  <div style="text-align: center; margin-bottom: 30px;">'
    '    <h1 style="color: #FFCD11; font-size: 36px; margin: 0;"><span style="color: #ffffff;">CAT</span>alyze</h1>'
    '    <p style="color: #888; font-size: 14px; margin-top: 5px;">AI-Powered Heavy Equipment Inspection</p>'
    '  </div>'
    '  <div style="background-color: #1e1e1e; border-radius: 8px; padding: 30px; margin-bottom: 20px;">'
    '    <h2 style="color: #FFCD11; margin-top: 0;">Welcome aboard! &#127881;</h2>'
    '    <p style="color: #cccccc; line-height: 1.6;">Your account is ready. Start inspecting your Caterpillar equipment with AI-powered precision &mdash; catch issues before they become costly failures.</p>'
    '    <div style="margin: 25px 0;">'
    '      <div style="display: flex; margin-bottom: 15px;"><span style="color: #FFCD11; margin-right: 10px;">&#128247;</span><span style="color: #cccccc;">Instant AI photo analysis</span></div>'
    '      <div style="display: flex; margin-bottom: 15px;"><span style="color: #FFCD11; margin-right: 10px;">&#127908;&#65039;</span><span style="color: #cccccc;">Voice-powered inspections</span></div>'
    '      <div style="display: flex; margin-bottom: 15px;"><span style="color: #FFCD11; margin-right: 10px;">&#128202;</span><span style="color: #cccccc;">Health analytics &amp; trends</span></div>'
    '      <div style="display: flex;"><span style="color: #FFCD11; margin-right: 10px;">&#128308;</span><span style="color: #cccccc;">Live inspection mode</span></div>'
    '    </div>'
    '    <a href="http://127.0.0.1:8000" style="display: inline-block; background-color: #FFCD11; color: #121212; padding: 12px 30px; border-radius: 6px; text-decoration: none; font-weight: bold; font-size: 16px;">Start Inspecting &rarr;</a>'
    '  </div>'
    '  <p style="color: #555; font-size: 12px; text-align: center;">&copy; 2025 CATalyze. Built for Caterpillar equipment inspectors.</p>'
    '</div>'
)


_VERIFY_EMAIL_HTML = (
    '<div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; background-color: #121212; color: #ffffff; padding: 40px; border-radius: 12px;">'
    '  <div style="text-align: center; margin-bottom: 30px;">'
    '    <h1 style="color: #FFCD11; font-size: 36px; margin: 0;"><span style="color: #ffffff;">CAT</span>alyze</h1>'
    '    <p style="color: #888; font-size: 14px; margin-top: 5px;">AI-Powered Heavy Equipment Inspection</p>'
    '  </div>'
    '  <div style="background-color: #1e1e1e; border-radius: 8px; padding: 30px; margin-bottom: 20px;">'
    '    <h2 style="color: #FFCD11; margin-top: 0;">Verify Your Email</h2>'
    '    <p style="color: #cccccc; line-height: 1.6;">Use the code below to verify your CATalyze account. This code expires in 10 minutes.</p>'
    '    <div style="text-align: center; margin: 30px 0;">'
    '      <span style="display: inline-block; background-color: #FFCD11; color: #121212; padding: 16px 32px; border-radius: 8px; font-size: 32px; font-weight: bold; letter-spacing: 8px;">{code}</span>'
    '    </div>'
    '    <p style="color: #888; font-size: 13px; text-align: center;">If you didn&rsquo;t create this account, you can safely ignore this email.</p>'
    '  </div>'
    '  <p style="color: #555; font-size: 12px; text-align: center;">&copy; 2025 CATalyze. Built for Caterpillar equipment inspectors.</p>'
    '</div>'
)

_pending_users: dict[str, dict] = {}
_VERIFY_TTL = 600  # 10 minutes


def _send_resend_email(to: str, subject: str, html: str) -> None:
    """Send an email via Resend. Fails silently."""
    api_key = os.getenv("RESEND_API_KEY")
    if not api_key:
        return
    try:
        import resend
        resend.api_key = api_key
        resend.Emails.send({
            "from": "onboarding@resend.dev",
            "to": to,
            "subject": subject,
            "html": html,
        })
    except Exception:
        pass


@app.post("/auth/register")
async def auth_register(req: RegisterRequest):
    """Validate registration and send a verification code (account stays pending)."""
    email = req.email.strip().lower()
    if email in _users:
        raise HTTPException(
            status_code=400,
            detail={"success": False, "message": "An account with this email already exists."},
        )
    if req.password != req.confirm_password:
        return {"success": False, "error": "Passwords do not match."}
    if len(req.password) < 4:
        return {"success": False, "error": "Password must be at least 4 characters."}

    code = f"{random.randint(0, 999999):06d}"
    _pending_users[email] = {
        "email": email,
        "username": req.username,
        "password_hash": _hash_pw(req.password),
        "code": code,
        "expiry": time.time() + _VERIFY_TTL,
    }

    _send_resend_email(email, "CATalyze — Verify Your Email", _VERIFY_EMAIL_HTML.replace("{code}", code))

    return {"success": True, "pending": True}


@app.post("/auth/verify-email")
async def auth_verify_email(req: VerifyEmailRequest):
    """Verify the email code, activate the account, and send the welcome email."""
    email = req.email.strip().lower()
    entry = _pending_users.get(email)
    if not entry:
        return {"success": False, "error": "No pending registration found for this email."}
    if time.time() > entry["expiry"]:
        _pending_users.pop(email, None)
        return {"success": False, "error": "Code expired. Please register again."}
    if req.code != entry["code"]:
        return {"success": False, "error": "Invalid code."}

    _users[email] = {
        "email": entry["email"],
        "username": entry["username"],
        "password_hash": entry["password_hash"],
    }
    _save_users(_users)
    _pending_users.pop(email, None)

    _send_resend_email(email, "Welcome to CATalyze!", _WELCOME_HTML)

    return {"success": True, "user_id": email, "username": entry["username"]}


@app.post("/auth/login")
async def auth_login(req: LoginRequest):
    """Verify email + password and return user info."""
    user = _users.get(req.email.strip().lower())
    if not user:
        return {"success": False, "error": "No account found with this email."}
    if user["password_hash"] != _hash_pw(req.password):
        return {"success": False, "error": "Incorrect password."}
    return {"success": True, "user_id": user["email"], "username": user["username"]}


# ---- Forgot / Reset password ---------------------------------------------

_reset_codes: dict[str, tuple[str, float]] = {}
_RESET_TTL = 600  # 10 minutes

_RESET_EMAIL_HTML = (
    '<div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; background-color: #121212; color: #ffffff; padding: 40px; border-radius: 12px;">'
    '  <div style="text-align: center; margin-bottom: 30px;">'
    '    <h1 style="color: #FFCD11; font-size: 36px; margin: 0;"><span style="color: #ffffff;">CAT</span>alyze</h1>'
    '    <p style="color: #888; font-size: 14px; margin-top: 5px;">AI-Powered Heavy Equipment Inspection</p>'
    '  </div>'
    '  <div style="background-color: #1e1e1e; border-radius: 8px; padding: 30px; margin-bottom: 20px;">'
    '    <h2 style="color: #FFCD11; margin-top: 0;">Password Reset</h2>'
    '    <p style="color: #cccccc; line-height: 1.6;">Use the code below to reset your CATalyze password. This code expires in 10 minutes.</p>'
    '    <div style="text-align: center; margin: 30px 0;">'
    '      <span style="display: inline-block; background-color: #FFCD11; color: #121212; padding: 16px 32px; border-radius: 8px; font-size: 32px; font-weight: bold; letter-spacing: 8px;">{code}</span>'
    '    </div>'
    '    <p style="color: #888; font-size: 13px; text-align: center;">If you didn&rsquo;t request this, you can safely ignore this email.</p>'
    '  </div>'
    '  <p style="color: #555; font-size: 12px; text-align: center;">&copy; 2025 CATalyze. Built for Caterpillar equipment inspectors.</p>'
    '</div>'
)


@app.post("/auth/forgot-password")
async def auth_forgot_password(req: ForgotPasswordRequest):
    """Generate a 6-digit reset code, store it, and email it via Resend."""
    email = req.email.strip().lower()
    if email not in _users:
        return {"success": False, "error": "No account found with this email."}

    code = f"{random.randint(0, 999999):06d}"
    _reset_codes[email] = (code, time.time() + _RESET_TTL)

    _send_resend_email(email, "CATalyze — Password Reset Code", _RESET_EMAIL_HTML.replace("{code}", code))

    return {"success": True}


@app.post("/auth/reset-password")
async def auth_reset_password(req: ResetPasswordRequest):
    """Verify the reset code and update the user's password."""
    email = req.email.strip().lower()
    entry = _reset_codes.get(email)
    if not entry:
        return {"success": False, "error": "No reset code found. Please request a new one."}

    stored_code, expiry = entry
    if time.time() > expiry:
        _reset_codes.pop(email, None)
        return {"success": False, "error": "Code expired. Please request a new one."}
    if req.code != stored_code:
        return {"success": False, "error": "Invalid code."}
    if req.new_password != req.confirm_password:
        return {"success": False, "error": "Passwords do not match."}
    if len(req.new_password) < 4:
        return {"success": False, "error": "Password must be at least 4 characters."}

    _users[email]["password_hash"] = _hash_pw(req.new_password)
    _save_users(_users)
    _reset_codes.pop(email, None)
    return {"success": True}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ELEVENLABS_TTS_URL = "https://api.elevenlabs.io/v1/text-to-speech"


async def _elevenlabs_tts(text: str) -> bytes | None:
    """Generate speech audio via ElevenLabs. Returns MP3 bytes or None."""
    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        return None

    voice_id = os.environ.get("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")

    async with httpx.AsyncClient(timeout=60) as client:
        try:
            resp = await client.post(
                f"{ELEVENLABS_TTS_URL}/{voice_id}",
                headers={
                    "xi-api-key": api_key,
                    "Content-Type": "application/json",
                },
                json={
                    "text": text,
                    "model_id": "eleven_turbo_v2",
                    "voice_settings": {
                        "stability": 0.5,
                        "similarity_boost": 0.75,
                    },
                },
            )
            if resp.status_code == 200:
                return resp.content
        except httpx.HTTPError:
            pass

    return None


def _build_report_text(
    machine: str,
    inspector: str | None,
    results: list[dict[str, Any]],
) -> str:
    """Compile inspection results into a human-readable report."""
    lines = [f"Inspection Report — {machine}"]
    if inspector:
        lines.append(f"Inspector: {inspector}")
    lines.append(
        f"Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
    )
    lines.append("")

    pass_ct = sum(1 for r in results if r.get("status") == "PASS")
    monitor_ct = sum(1 for r in results if r.get("status") == "MONITOR")
    fail_ct = sum(1 for r in results if r.get("status") == "FAIL")

    lines.append(
        f"Summary: {pass_ct} PASS · {monitor_ct} MONITOR · {fail_ct} FAIL "
        f"out of {len(results)} components"
    )
    lines.append("")

    for r in results:
        comp = r.get("component", "Unknown")
        status = r.get("status", "N/A")
        conf = r.get("confidence", 0)
        lines.append(f"  {comp}: {status} ({conf:.0%} confidence)")
        lines.append(f"    Observation: {r.get('observation', '—')}")
        if status != "PASS":
            lines.append(
                f"    Action: {r.get('recommended_action', '—')}"
            )

    return "\n".join(lines)


def _build_voice_summary(
    machine: str,
    inspector: str | None,
    results: list[dict[str, Any]],
) -> str:
    """Build a natural-language summary suitable for text-to-speech."""
    pass_ct = sum(1 for r in results if r.get("status") == "PASS")
    monitor_ct = sum(1 for r in results if r.get("status") == "MONITOR")
    fail_ct = sum(1 for r in results if r.get("status") == "FAIL")

    parts = [f"Inspection report for {machine}."]
    if inspector:
        parts.append(f"Inspector: {inspector}.")
    parts.append(
        f"Out of {len(results)} components inspected, "
        f"{pass_ct} passed, {monitor_ct} need monitoring, and {fail_ct} failed."
    )

    attention = [
        r for r in results if r.get("status") in ("MONITOR", "FAIL")
    ]
    if attention:
        parts.append("Items requiring attention:")
        for r in attention:
            parts.append(
                f"{r.get('component', 'Unknown')}: {r.get('status')}. "
                f"{r.get('observation', '')} "
                f"Recommended action: {r.get('recommended_action', 'none')}."
            )
    else:
        parts.append("All components passed inspection.")

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/")
async def root():
    """Serve the PWA frontend."""
    return FileResponse("frontend/index.html")


@app.get("/machines")
async def get_machines():
    """Return all machines and their components for the frontend selector."""
    return MACHINES


# ---- History -------------------------------------------------------------


@app.get("/history")
async def get_history(
    machine: str = Query(...),
    component: str = Query(...),
    inspector_id: str | None = Query(None),
    limit: int = Query(20),
):
    """Return past inspection records for a machine/component (for timeline)."""
    records = await get_inspection_history(
        machine, component, inspector_id=inspector_id, limit=limit
    )
    return {"machine": machine, "component": component, "records": records}


# ---- Inspection ----------------------------------------------------------


def _extract_middle_frame(video_bytes: bytes, filename: str = "video.mp4") -> bytes | None:
    """Extract the middle frame from a video as JPEG bytes. Returns None on failure."""
    ext = Path(filename).suffix or ".mp4"
    if ext.lower() not in (".mp4", ".webm", ".mov", ".avi", ".mkv"):
        ext = ".mp4"
    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
            f.write(video_bytes)
            path = f.name
        try:
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                return None
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total < 1:
                total = 1
            mid = (total - 1) // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
            ret, frame = cap.read()
            cap.release()
            if not ret or frame is None:
                return None
            _, jpeg = cv2.imencode(".jpg", frame)
            return jpeg.tobytes()
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
    except Exception:
        return None


@app.post("/inspect/image")
async def inspect_image(
    image: UploadFile = File(...),
    machine: str = Form(...),
    component: str = Form(...),
    inspector_id: str = Form(""),
):
    """Analyze a photo of a machine component via Gemini Vision."""
    image_bytes = await image.read()
    iid = inspector_id or None

    history = await get_inspection_history(machine, component, inspector_id=iid)

    result = analyze_image(image_bytes, machine, component, history)

    result["machine"] = machine
    result["component"] = component
    result["timestamp"] = datetime.now(timezone.utc).isoformat()
    if iid:
        result["inspector_id"] = iid

    return result


@app.post("/inspect/video")
async def inspect_video(
    video: UploadFile = File(...),
    machine: str = Form(...),
    component: str = Form(...),
    inspector_id: str = Form(""),
):
    """Analyze a video by extracting the middle frame and running image analysis."""
    video_bytes = await video.read()
    frame_bytes = _extract_middle_frame(video_bytes, filename=video.filename or "video.mp4")
    if not frame_bytes:
        raise HTTPException(
            status_code=400,
            detail="Could not extract a frame from the video. Ensure the file is a valid video.",
        )
    iid = inspector_id or None
    history = await get_inspection_history(machine, component, inspector_id=iid)
    result = analyze_image(frame_bytes, machine, component, history)
    result["machine"] = machine
    result["component"] = component
    result["timestamp"] = datetime.now(timezone.utc).isoformat()
    if iid:
        result["inspector_id"] = iid
    return result


@app.post("/inspect/voice")
async def inspect_voice(req: VoiceRequest):
    """Analyze a transcribed voice observation."""
    iid = req.inspector_id or None
    history = await get_inspection_history(
        req.machine, req.component, inspector_id=iid
    )

    result = analyze_voice(req.machine, req.component, req.voice_text, history)

    result["machine"] = req.machine
    result["component"] = req.component
    result["timestamp"] = datetime.now(timezone.utc).isoformat()
    if iid:
        result["inspector_id"] = iid

    return result


@app.post("/inspect/save")
async def inspect_save(req: SaveInspectionRequest):
    """Save an inspection result to Supermemory (called by frontend after each success)."""
    iid = req.inspector_id or None
    result = {
        "status": req.status,
        "observation": req.observation or "",
        "confidence": req.confidence,
        "recommended_action": req.recommended_action,
        "maintenance_steps": req.maintenance_steps or [],
        "timestamp": req.timestamp,
    }
    ok = await save_inspection_result(
        req.machine, req.component, result, inspector_id=iid
    )
    return {"success": ok}


@app.get("/inspect/results")
async def get_inspect_results(
    inspector_id: str = Query(..., description="Logged-in user email or ID"),
):
    """Return all past inspection results for the given inspector from Supermemory."""
    records = await get_all_inspection_results(inspector_id)
    return {"results": records}


# ---- Report --------------------------------------------------------------


@app.post("/report/generate")
async def generate_report(req: ReportRequest):
    """Compile a full inspection report and generate an audio summary."""
    report_text = _build_report_text(req.machine, req.inspector, req.results)
    voice_summary = _build_voice_summary(
        req.machine, req.inspector, req.results
    )

    pass_ct = sum(1 for r in req.results if r.get("status") == "PASS")
    monitor_ct = sum(1 for r in req.results if r.get("status") == "MONITOR")
    fail_ct = sum(1 for r in req.results if r.get("status") == "FAIL")

    audio_bytes = await _elevenlabs_tts(voice_summary)
    audio_base64 = (
        base64.b64encode(audio_bytes).decode() if audio_bytes else None
    )

    return {
        "report_text": report_text,
        "summary": {
            "machine": req.machine,
            "inspector": req.inspector,
            "total": len(req.results),
            "pass": pass_ct,
            "monitor": monitor_ct,
            "fail": fail_ct,
        },
        "results": req.results,
        "audio_base64": audio_base64,
    }


# ---- Live inspection -----------------------------------------------------


@app.post("/inspect/live")
async def inspect_live(req: LiveRequest):
    """Analyze a single frame from a live camera feed."""
    image_bytes = base64.b64decode(req.image_base64)
    result = analyze_live(image_bytes)
    return result


# ---- Parts identification ------------------------------------------------


@app.post("/parts/identify")
async def identify_part_route(image: UploadFile = File(...)):
    """Identify an unknown part from a photo with ranked candidates."""
    image_bytes = await image.read()

    result = identify_part(image_bytes)
    return result


# ---------------------------------------------------------------------------
# Static files (frontend assets like JS, CSS, icons)
# ---------------------------------------------------------------------------

if os.path.isdir("frontend"):
    app.mount("/static", StaticFiles(directory="frontend"), name="static")
