"""FastAPI boundary for the trust-traced research agent."""

from __future__ import annotations

import os
from pathlib import Path
from threading import Lock
from typing import Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agent.gmail_agent import (
    ResearchAgentError,
    build_gmail_service,
    resume_research_agent,
    run_research_agent,
)
from agent.youtube_tools import build_youtube_service
from agent.voice_commands import parse_voice_command
from tracer.run_store import RunStore
from voice.stt import transcribe_with_confidence

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@asynccontextmanager
async def lifespan(_app: FastAPI):
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
    yield


app = FastAPI(title="Agent Trust Tracer API", version="1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get(
        "DASHBOARD_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_runs: dict[str, Any] = {}
_runs_lock = Lock()


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    source: str = Field(default="Both sources", pattern="^(Both sources|Gmail|YouTube)$")


class ApprovalRequest(BaseModel):
    approved: bool


class TranscriptionResponse(BaseModel):
    text: str
    confidence: float | None = None


def _services(source: str) -> tuple[Any, Any]:
    gmail = None
    youtube = None
    client_file = Path(
        os.environ.get(
            "GOOGLE_OAUTH_CLIENT_FILE",
            str(PROJECT_ROOT / "client_secret.json"),
        )
    ).expanduser()
    token_file = Path(
        os.environ.get("GMAIL_TOKEN_FILE", str(PROJECT_ROOT / ".gmail-token.json"))
    ).expanduser()
    if source in {"Both sources", "Gmail"}:
        gmail = build_gmail_service(client_file, token_file)
    if source in {"Both sources", "YouTube"}:
        youtube = build_youtube_service()
    return gmail, youtube


def _allowed_tools(source: str, question: str = "") -> list[str] | None:
    if source == "Gmail":
        return [
            "search_gmail",
            "read_gmail_message",
            "create_gmail_draft",
            "send_gmail_message",
        ]
    if source == "YouTube":
        tools = [
            "search_youtube",
            "youtube_search_latest",
            "get_youtube_channel_details",
        ]
        if parse_voice_command(question).media_playback:
            tools.append("youtube_search_and_play")
        return tools
    return None


def _run_payload(run: Any) -> dict[str, Any]:
    report = run.trust_report.as_dict() if run.trust_report else {}
    return {
        "run_id": run.run_id,
        "status": "review" if run.pending_approval else "completed",
        "answer": run.answer,
        "pending_approval": run.pending_approval,
        "task_type": run.task_type,
        "trust_report": report,
    }


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "agent-trust-tracer"}


@app.get("/api/overview")
def overview() -> dict[str, Any]:
    store = RunStore(
        os.environ.get("AGENT_TRUST_DB", str(PROJECT_ROOT / ".agent-trust.sqlite"))
    )
    return {
        "active_agents": 1,
        "new_critical_alerts": 0,
        "posture_score": 95,
        "resolution_rate": 100,
        "latest_run": store.get(next(iter(_runs), "")) if _runs else None,
    }


@app.post("/api/transcribe", response_model=TranscriptionResponse)
async def transcribe_voice(audio: UploadFile = File(...)) -> TranscriptionResponse:
    content = await audio.read()
    try:
        text, confidence = transcribe_with_confidence(content)
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return TranscriptionResponse(text=text, confidence=confidence)


@app.post("/api/runs")
def create_run(request: QueryRequest) -> dict[str, Any]:
    intent = parse_voice_command(request.question)
    effective_source = (
        intent.source
        if request.source == "Both sources" and intent.source
        else request.source
    )
    if (
        intent.destination
        and intent.url
        and not intent.media_playback
    ):
        return {
            "run_id": "",
            "status": "completed",
            "answer": (
                f"Opening {intent.destination.capitalize()} in a new browser tab."
            ),
            "pending_approval": None,
            "task_type": "browser_navigation",
            "browser_action": {
                "destination": intent.destination,
                "url": intent.url,
            },
            "trust_report": {},
        }
    try:
        gmail, youtube = _services(effective_source)
        run = run_research_agent(
            request.question,
            gmail,
            youtube,
            allowed_tools=_allowed_tools(effective_source, request.question),
        )
    except (ResearchAgentError, RuntimeError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    with _runs_lock:
        _runs[run.run_id] = run
    return _run_payload(run)


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    with _runs_lock:
        run = _runs.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found.")
    return _run_payload(run)


@app.post("/api/runs/{run_id}/approval")
def approve_run(run_id: str, request: ApprovalRequest) -> dict[str, Any]:
    with _runs_lock:
        run = _runs.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found.")
    try:
        resumed = resume_research_agent(run, request.approved)
    except (ResearchAgentError, RuntimeError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    with _runs_lock:
        _runs[run_id] = resumed
    return _run_payload(resumed)
