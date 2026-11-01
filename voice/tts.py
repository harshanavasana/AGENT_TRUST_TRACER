"""Best-effort spoken replies with Groq Orpheus and local fallbacks."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import tempfile

MAX_SPOKEN_CHARS = 400


def _read_and_remove(path: str | Path) -> bytes:
    file_path = Path(path)
    try:
        return file_path.read_bytes()
    finally:
        file_path.unlink(missing_ok=True)


def _groq_speak(text: str) -> bytes:
    from groq import Groq

    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    response = client.audio.speech.create(
        model=os.environ.get("GROQ_TTS_MODEL", "canopylabs/orpheus-v1-english"),
        voice=os.environ.get("GROQ_TTS_VOICE", "hannah"),
        input=text[:MAX_SPOKEN_CHARS],
        response_format="wav",
    )
    if hasattr(response, "read"):
        return response.read()
    content = getattr(response, "content", None)
    if content:
        return bytes(content)
    raise RuntimeError("Groq TTS returned no audio.")


def _edge_speak(text: str) -> bytes:
    import edge_tts

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
        path = handle.name
    try:
        asyncio.run(
            edge_tts.Communicate(text[:MAX_SPOKEN_CHARS], "en-US-AriaNeural").save(path)
        )
        return _read_and_remove(path)
    except Exception:
        Path(path).unlink(missing_ok=True)
        raise


def _pyttsx3_speak(text: str) -> bytes:
    import pyttsx3

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
        path = handle.name
    try:
        engine = pyttsx3.init()
        engine.save_to_file(text[:MAX_SPOKEN_CHARS], path)
        engine.runAndWait()
        return _read_and_remove(path)
    except Exception:
        Path(path).unlink(missing_ok=True)
        raise


def speak(text: str) -> bytes | None:
    """Return WAV bytes, trying cloud then local speech engines."""
    short_text = " ".join(str(text).split())[:MAX_SPOKEN_CHARS].strip()
    if not short_text:
        return None
    for synthesizer in (_groq_speak, _edge_speak, _pyttsx3_speak):
        try:
            audio = synthesizer(short_text)
            if audio:
                return audio
        except Exception:
            continue
    return None
