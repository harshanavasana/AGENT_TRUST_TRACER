"""Small, deterministic helpers for voice-command routing."""

from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class VoiceCommand:
    """The safe browser intent extracted from a spoken request."""

    text: str
    destination: str | None
    url: str | None
    source: str | None


_DESTINATIONS = {
    "youtube": "https://www.youtube.com/",
    "gmail": "https://mail.google.com/",
}
_BROWSER_ACTION_PATTERNS = (
    r"\bopen\b",
    r"\blaunch\b",
    r"\bvisit\b",
    r"\bgo\s+to\b",
    r"\btake\s+me\s+to\b",
    r"\bnavigate\s+to\b",
)


def parse_voice_command(text: str) -> VoiceCommand:
    """Normalize a transcript and recognize only supported browser destinations."""
    normalized = re.sub(r"\s+", " ", text).strip()
    lowered = normalized.casefold()
    has_browser_action = any(
        re.search(pattern, lowered) for pattern in _BROWSER_ACTION_PATTERNS
    )
    destination = (
        next(
            (name for name in _DESTINATIONS if re.search(rf"\b{name}\b", lowered)),
            None,
        )
        if has_browser_action
        else None
    )
    source = (
        "Gmail"
        if re.search(r"\b(gmail|email|emails|inbox|mail)\b", lowered)
        else "YouTube"
        if re.search(r"\b(youtube|video|videos|channel)\b", lowered)
        else None
    )
    return VoiceCommand(
        text=normalized,
        destination=destination,
        url=_DESTINATIONS.get(destination) if destination else None,
        source=source,
    )
