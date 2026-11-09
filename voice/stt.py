"""Speech-to-text using hosted Hugging Face Whisper, with Groq fallback."""

from __future__ import annotations

from functools import lru_cache
import io
import os
from typing import Any, Callable


class SpeechToTextError(RuntimeError):
    """Raised when audio cannot be transcribed."""


DEFAULT_HF_MODEL = "openai/whisper-large-v3"
SUPPORTED_PROVIDERS = {"huggingface", "huggingface-local", "groq"}


def _provider() -> str:
    return os.environ.get(
        "VOICE_STT_PROVIDER",
        os.environ.get("STT_PROVIDER", "huggingface"),
    ).strip().casefold()


def _huggingface_token() -> str:
    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        or ""
    ).strip()
    if not token:
        raise SpeechToTextError(
            "Set HF_TOKEN in .env to use Hugging Face ASR inference."
        )
    return token


def _huggingface_model() -> str:
    """Return the hosted ASR model, accepting the legacy setting name."""
    return (
        os.environ.get("HF_ASR_MODEL")
        or os.environ.get("HF_WHISPER_MODEL")
        or DEFAULT_HF_MODEL
    ).strip()


def _transcribe_huggingface(
    audio_bytes: bytes,
    language: str | None,
) -> tuple[str, float | None]:
    """Transcribe through Hugging Face hosted inference."""
    try:
        from huggingface_hub import InferenceClient
    except ImportError as error:
        raise SpeechToTextError(
            "Hugging Face inference needs huggingface_hub. "
            "Install requirements.txt."
        ) from error

    model_name = _huggingface_model()
    try:
        client = InferenceClient(
            provider=os.environ.get("HF_INFERENCE_PROVIDER", "hf-inference"),
            api_key=_huggingface_token(),
        )
        result = client.automatic_speech_recognition(
            audio_bytes,
            model=model_name,
        )
    except SpeechToTextError:
        raise
    except Exception as error:
        raise SpeechToTextError(
            f"Hugging Face ASR inference failed for {model_name!r} "
            f"({type(error).__name__})."
        ) from error

    if isinstance(result, dict):
        text = str(result.get("text", "") or "").strip()
    else:
        text = str(result or "").strip()
    if not text:
        raise SpeechToTextError("Hugging Face ASR returned no text.")
    return text, None


@lru_cache(maxsize=2)
def _load_huggingface_pipeline(model_name: str, device: str) -> Callable[..., Any]:
    try:
        import torch
        from transformers import pipeline
    except (ImportError, OSError) as error:
        raise SpeechToTextError(
            "Local ASR needs the transformers, torch, and soundfile packages. "
            "Use VOICE_STT_PROVIDER=huggingface for hosted ASR."
        ) from error

    try:
        resolved_device = (
            0
            if device == "auto" and torch.cuda.is_available()
            else -1
            if device == "auto"
            else int(device)
        )
        return pipeline(
            "automatic-speech-recognition",
            model=model_name,
            device=resolved_device,
            torch_dtype=torch.float16
            if resolved_device >= 0
            else torch.float32,
        )
    except Exception as error:
        raise SpeechToTextError(
            f"Unable to load local ASR model {model_name!r} "
            f"({type(error).__name__})."
        ) from error


def _transcribe_huggingface_local(
    audio_bytes: bytes,
    language: str | None,
) -> tuple[str, float | None]:
    try:
        import numpy as np
        import soundfile as sf

        audio, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        raise SpeechToTextError(
            f"Could not decode the recording for local ASR "
            f"({type(error).__name__})."
        ) from error

    if getattr(audio, "ndim", 1) > 1:
        audio = np.mean(audio, axis=1)
    if not len(audio):
        raise SpeechToTextError("Audio was empty after decoding.")

    model_name = _huggingface_model()
    device = os.environ.get("HF_WHISPER_DEVICE", "auto").strip().casefold()
    whisper = _load_huggingface_pipeline(model_name, device)
    generate_kwargs = {
        "task": "transcribe",
        "language": language or os.environ.get("GROQ_STT_LANGUAGE", "en"),
    }
    try:
        result = whisper(
            {"raw": audio, "sampling_rate": sample_rate},
            generate_kwargs=generate_kwargs,
        )
    except Exception as error:
        raise SpeechToTextError(
            f"Local ASR transcription failed ({type(error).__name__})."
        ) from error
    text = str(result.get("text", "") if isinstance(result, dict) else "").strip()
    if not text:
        raise SpeechToTextError("Local ASR returned no text.")
    return text, None


def _transcribe_groq(
    audio_bytes: bytes,
    language: str | None,
) -> tuple[str, float | None]:
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


def transcribe_with_confidence(
    audio_bytes: bytes,
    language: str | None = None,
) -> tuple[str, float | None]:
    """Transcribe audio and return text plus optional confidence.

    Hugging Face hosted OpenAI Whisper Large v3 is the default. Set
    ``VOICE_STT_PROVIDER=groq`` for Groq or enable
    ``VOICE_STT_FALLBACK_TO_GROQ`` to recover from Hugging Face failures.
    """
    if not audio_bytes or len(audio_bytes) < 100:
        raise SpeechToTextError("Audio was empty or too short to transcribe.")
    provider = _provider()
    if provider not in SUPPORTED_PROVIDERS:
        raise SpeechToTextError(
            f"Unsupported VOICE_STT_PROVIDER={provider!r}. "
            "Use 'huggingface', 'huggingface-local', or 'groq'."
        )
    transcriber = {
        "groq": _transcribe_groq,
        "huggingface": _transcribe_huggingface,
        "huggingface-local": _transcribe_huggingface_local,
    }[provider]
    try:
        return transcriber(audio_bytes, language)
    except SpeechToTextError:
        if (
            provider in {"huggingface", "huggingface-local"}
            and os.environ.get("VOICE_STT_FALLBACK_TO_GROQ", "").casefold()
            in {"1", "true", "yes"}
        ):
            return _transcribe_groq(audio_bytes, language)
        raise


def transcribe(audio_bytes: bytes, language: str | None = None) -> str:
    """Transcribe audio bytes with the configured ASR backend."""
    return transcribe_with_confidence(audio_bytes, language)[0]
