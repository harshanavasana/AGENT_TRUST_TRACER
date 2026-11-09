"""Bounded browser actions for fixed, auditable Gmail and YouTube destinations."""

from __future__ import annotations

import os
import re
import time
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal

from tracer.run_store import RunStore

ALLOWED_SITES = {
    "youtube": "https://www.youtube.com/",
    "gmail": "https://mail.google.com/",
}
VIDEO_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{11}$")
MAX_BROWSER_STEPS = 8
DEFAULT_BROWSER_TIMEOUT_MS = 30_000
_LIVE_CONTEXTS: dict[str, tuple[Any, Any, Any]] = {}
_BROWSER_EXECUTOR = ThreadPoolExecutor(
    max_workers=2,
    thread_name_prefix="source-desk-browser",
)


def _run_browser(operation: Any) -> Any:
    """Run sync Playwright outside Streamlit's asyncio event-loop thread."""
    return _BROWSER_EXECUTOR.submit(operation).result()


def _context() -> Any:
    from playwright.sync_api import sync_playwright

    playwright = sync_playwright().start()
    launch_options = {"headless": False}
    executable_path = os.environ.get("BROWSER_EXECUTABLE_PATH")
    if executable_path:
        launch_options["executable_path"] = executable_path
    browser = playwright.chromium.launch_persistent_context(
        os.environ.get("BROWSER_PROFILE_DIR", ".browser_profile"),
        **launch_options,
    )
    return playwright, browser, browser.pages[0] if browser.pages else browser.new_page()


def _close(context: tuple[Any, Any, Any]) -> None:
    playwright, browser, _page = context
    try:
        try:
            browser.close()
        except Exception:
            pass
    finally:
        try:
            playwright.stop()
        except Exception:
            pass


def _play_video(url: str, run_id: str = "") -> dict[str, Any]:
    def play() -> dict[str, Any]:
        context = _context()
        _playwright, _browser, page = context
        try:
            page.set_default_timeout(DEFAULT_BROWSER_TIMEOUT_MS)
            page.goto(url, wait_until="domcontentloaded")
            for selector in (
                "button[aria-label*='Accept']",
                "button:has-text('Accept all')",
            ):
                try:
                    page.locator(selector).first.click(timeout=2000)
                    break
                except Exception:
                    continue
            try:
                page.locator(".ytp-play-button").click(timeout=5000)
            except Exception:
                pass
            result = {"status": "playing", "url": page.url, "title": page.title()}
            if run_id:
                _LIVE_CONTEXTS[run_id] = context
            else:
                _close(context)
            return result
        except Exception:
            _close(context)
            raise

    try:
        return _run_browser(play)
    except Exception as error:
        # Do not open a second browser after Playwright has already launched.
        # The caller must retry explicitly if the controlled browser action fails.
        raise RuntimeError(
            "YouTube browser playback failed after the controlled browser "
            f"operation started ({type(error).__name__}: {error}); "
            "no second browser was opened."
        ) from error


def build_browser_tools(
    youtube_service: Any,
    *,
    run_id: str = "",
    run_store: RunStore | None = None,
    browser_enabled: bool = True,
) -> list[Any]:
    """Build only fixed-destination browser tools."""
    from langchain_core.tools import tool

    searched_video_ids: set[str] = set()
    steps = 0
    playback_started = False

    def record(name: str, status: str, **detail: Any) -> None:
        if run_store and run_id:
            run_store.event(run_id, name, status, **detail)

    def step() -> None:
        nonlocal steps
        steps += 1
        if steps > MAX_BROWSER_STEPS:
            raise RuntimeError("The browser action limit for this run was reached.")

    @tool
    def open_site(site: Literal["youtube", "gmail"]) -> str:
        """Open one fixed, allowlisted public site."""
        step()
        if site not in ALLOWED_SITES:
            raise ValueError("Only YouTube and Gmail are allowlisted destinations.")
        record("Opening site", "started", site=site)
        if not browser_enabled:
            return f"Browser action disabled in this consistency pass: {site}."
        def open_page() -> str:
            context = _context()
            try:
                context[2].set_default_timeout(DEFAULT_BROWSER_TIMEOUT_MS)
                context[2].goto(ALLOWED_SITES[site], wait_until="domcontentloaded")
                result = f"Opened {site}."
                record("Opening site", "finished", site=site)
                if run_id:
                    _LIVE_CONTEXTS[run_id] = context
                    return result
                return result
            finally:
                if context not in _LIVE_CONTEXTS.values():
                    _close(context)

        try:
            return _run_browser(open_page)
        except Exception as error:
            record("Opening site", "failed", site=site, error=type(error).__name__)
            raise RuntimeError(
                f"Opening the controlled {site} browser destination failed; "
                "no system-browser fallback was used."
            ) from error

    @tool
    def youtube_search_and_play(
        query: str,
        prefer_latest: bool = True,
        official_only: bool = True,
    ) -> str:
        """Find and play a YouTube video using API metadata and bounded browser control."""
        nonlocal playback_started
        step()
        if playback_started:
            raise RuntimeError(
                "Only one YouTube playback action is allowed per run; "
                "no additional browser tab was opened."
            )
        if not query.strip() or len(query) > 200:
            raise ValueError("YouTube queries must be 1-200 characters.")
        record("Searching YouTube", "started", query_length=len(query))
        try:
            response = (
                youtube_service.search()
                .list(
                    part="snippet",
                    q=query,
                    order="date" if prefer_latest else "relevance",
                    type="video",
                    maxResults=5,
                )
                .execute()
            )
        except Exception as error:
            raise RuntimeError(
                "YouTube API search failed; no browser-page fallback was used "
                "because it can select an unrelated result."
            ) from error
        items = response.get("items", [])
        rejected = re.compile(r"\b(cover|reaction|lyric|fan[\s-]?made)\b", re.I)
        candidates = [
            item for item in items
            if not official_only
            or not rejected.search(str(item.get("snippet", {}).get("title", "")))
        ]
        search_terms = {
            term.casefold()
            for term in re.findall(r"[A-Za-z0-9]+", query)
            if term.casefold()
            not in {
                "play",
                "open",
                "watch",
                "show",
                "launch",
                "latest",
                "newest",
                "new",
                "song",
                "video",
                "music",
            }
        }
        ranked = []
        for candidate in candidates or items:
            snippet = candidate.get("snippet", {})
            searchable = " ".join(
                (
                    str(snippet.get("title", "")),
                    str(snippet.get("channelTitle", "")),
                    str(snippet.get("description", "")),
                )
            ).casefold()
            score = sum(term in searchable for term in search_terms)
            ranked.append((score, candidate))
        if search_terms and ranked and max(score for score, _ in ranked) == 0:
            return "No YouTube result matched the requested title or channel."
        item = max(
            enumerate(ranked),
            key=lambda entry: (entry[1][0], -entry[0]),
        )[1][1] if ranked else None
        if item is None:
            return "No YouTube results were found."
        video_id = item.get("id", {}).get("videoId")
        snippet = item.get("snippet", {})
        if not isinstance(video_id, str) or not VIDEO_ID_PATTERN.fullmatch(video_id):
            raise ValueError("YouTube returned an invalid video ID.")
        searched_video_ids.add(video_id)
        playback_started = True
        url = f"https://www.youtube.com/watch?v={video_id}"
        record("Searching YouTube", "finished", video_id=video_id)
        if browser_enabled:
            result = _play_video(url, run_id)
            status = result["status"]
        else:
            status = "ready_for_in_app_player"
        return {
            "title": snippet.get("title", "(untitled)"),
            "channel": snippet.get("channelTitle", "(unknown)"),
            "video_id": video_id,
            "url": url,
            "published_at": snippet.get("publishedAt", "(unknown)"),
            "status": status,
        }

    @tool
    def gmail_open_inbox() -> str:
        """Open the fixed Gmail inbox destination without entering credentials."""
        step()
        if not browser_enabled:
            return "Browser action disabled in this consistency pass."
        def open_inbox() -> str:
            context = _context()
            try:
                context[2].goto(ALLOWED_SITES["gmail"], wait_until="domcontentloaded")
                if run_id:
                    _LIVE_CONTEXTS[run_id] = context
                    return "Opened Gmail inbox."
                return "Opened Gmail inbox."
            finally:
                if context not in _LIVE_CONTEXTS.values():
                    _close(context)

        try:
            return _run_browser(open_inbox)
        except Exception as error:
            record(
                "Opening Gmail inbox",
                "failed",
                error=type(error).__name__,
            )
            raise RuntimeError(
                "Opening the controlled Gmail browser destination failed; "
                "no system-browser fallback was used."
            ) from error

    @tool
    def return_to_dashboard(run_id: str) -> str:
        """Raise the metrics dashboard for a completed run."""
        step()
        if not re.fullmatch(r"[A-Za-z0-9_-]{8,128}", run_id):
            raise ValueError("Invalid run ID.")
        dashboard_url = os.environ.get("DASHBOARD_URL", "http://localhost:8501")
        url = f"{dashboard_url}/?run={run_id}&view=metrics"
        return f"Dashboard metrics are available for run {run_id}."

    return [open_site, youtube_search_and_play, gmail_open_inbox, return_to_dashboard]
