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
    media_playback: bool


_DESTINATIONS = {
    "youtube": "https://www.youtube.com/",
    "gmail": "https://mail.google.com/",
}
_DESTINATION_ALIASES = {
    "youtube": (
        r"\byoutube(?:\.com)?\b",
        r"\byou\s*tube\b",
    ),
    "gmail": (
        r"\bgmail(?:\.com)?\b",
        r"\bgoogle\s+mail\b",
        r"\bmail\.google\.com\b",
        r"\b(?:my\s+)?(?:email|emails|inbox)\b",
    ),
}
_BROWSER_ACTION_PATTERNS = (
    r"\bopen\b",
    r"\blaunch\b",
    r"\bvisit\b",
    r"\bgo\s+to\b",
    r"\btake\s+me\s+to\b",
    r"\bnavigate\s+to\b",
    r"\bbring\s+(?:up|me\s+to)\b",
    r"\bpull\s+up\b",
    r"\bshow\s+me\b",
)
_MEDIA_PLAYBACK_PATTERNS = (
    r"\bplay\b",
    r"\bwatch\b",
    r"\blisten\b",
    r"\bstream\b",
    r"\b(?:open|launch|show)\s+(?:the\s+)?(?:video|audio|song|music)\b",
)


def parse_voice_command(text: str) -> VoiceCommand:
    """Normalize a transcript and recognize only supported browser destinations."""
    normalized = re.sub(r"\s+", " ", text).strip()
    lowered = normalized.casefold()
    has_browser_action = any(
        re.search(pattern, lowered) for pattern in _BROWSER_ACTION_PATTERNS
    )
    destination = None
    if has_browser_action:
        destination = next(
            (
                name
                for name, aliases in _DESTINATION_ALIASES.items()
                if any(re.search(alias, lowered) for alias in aliases)
            ),
            None,
        )
    source = (
        "Gmail"
        if re.search(r"\b(gmail|email|emails|inbox|mail)\b", lowered)
        else "YouTube"
        if re.search(r"\b(youtube|video|videos|channel)\b", lowered)
        else None
    )
    media_playback = any(
        re.search(pattern, lowered) for pattern in _MEDIA_PLAYBACK_PATTERNS
    )
    return VoiceCommand(
        text=normalized,
        destination=destination,
        url=_DESTINATIONS.get(destination) if destination else None,
        source=source,
        media_playback=media_playback,
    )
