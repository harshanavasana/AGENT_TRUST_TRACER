"""Groq Whisper speech-to-text with bounded, explicit failures."""

from __future__ import annotations

import os
from typing import Any


class SpeechToTextError(RuntimeError):
    """Raised when audio cannot be transcribed."""


def transcribe_with_confidence(
    audio_bytes: bytes,
    language: str | None = None,
) -> tuple[str, float | None]:
    """Transcribe audio and return an optional average segment confidence."""
    if not audio_bytes or len(audio_bytes) < 100:
        raise SpeechToTextError("Audio was empty or too short to transcribe.")
    try:
        from groq import Groq

        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        response: Any = client.audio.transcriptions.create(
            file=("voice.wav", audio_bytes),
            model=os.environ.get("GROQ_STT_MODEL", "whisper-large-v3-turbo"),
            language=language or os.environ.get("GROQ_STT_LANGUAGE", "en"),
            prompt=(
                "Gmail, YouTube, email, inbox, latest song, Taylor Swift, "
                "play, open, search"
            ),
            temperature=0,
        )
        text = str(getattr(response, "text", "") or "").strip()
        segments = getattr(response, "segments", None) or []
        confidence_values = [
            1 - float(item.get("no_speech_prob", 0))
            for item in segments
            if isinstance(item, dict) and "no_speech_prob" in item
        ]
    except Exception as error:
        raise SpeechToTextError(
            f"Speech transcription failed ({type(error).__name__})."
        ) from error
    if not text:
        raise SpeechToTextError("Speech transcription returned no text.")
    confidence = (
        sum(confidence_values) / len(confidence_values)
        if confidence_values
        else None
    )
    return text, confidence


def transcribe(audio_bytes: bytes, language: str | None = None) -> str:
    """Transcribe audio bytes with Groq Whisper."""
    return transcribe_with_confidence(audio_bytes, language)[0]
