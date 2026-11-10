from api.server import _allowed_tools, _services, _run_payload
from agent.gmail_agent import ResearchRun
from tracer.trust_checks import RunTrace, TrustReport


def test_research_youtube_does_not_enable_playback() -> None:
    tools = _allowed_tools("YouTube", "Find the latest music video")

    assert "youtube_search_and_play" not in tools


def test_explicit_youtube_playback_enables_only_controlled_playback_tool() -> None:
    tools = _allowed_tools("YouTube", "Play the latest music video")

    assert "youtube_search_and_play" in tools
    assert "open_site" not in tools


def test_gmail_source_keeps_research_tools_scoped() -> None:
    tools = _allowed_tools("Gmail", "Find my latest project email")

    assert tools == [
        "search_gmail",
        "read_gmail_message",
        "create_gmail_draft",
        "send_gmail_message",
    ]


def test_source_services_are_initialized_selectively(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        "api.server.build_gmail_service",
        lambda *_args: calls.append("gmail") or object(),
    )
    monkeypatch.setattr(
        "api.server.build_youtube_service",
        lambda: calls.append("youtube") or object(),
    )

    gmail, youtube = _services("YouTube")

    assert gmail is None
    assert youtube is not None
    assert calls == ["youtube"]


def test_run_payload_exposes_play_media_action() -> None:
    trace = RunTrace()
    trace.add_tool_call(
        "youtube_search_and_play",
        "youtube",
        {"query": "music"},
        "success",
        1,
        result=str(
            {
                "video_id": "abcdefghijk",
                "title": "Requested song",
                "url": "https://www.youtube.com/watch?v=abcdefghijk",
                "status": "ready_for_in_app_player",
            }
        ),
        evidence_id="E1",
    )
    run = ResearchRun(
        answer="The video is ready in the player. [E1]",
        trust_report=TrustReport([], trace, None, None),
        run_id="run-12345678",
    )

    payload = _run_payload(run)

    assert payload["media_action"] == {
        "type": "PLAY_MEDIA",
        "video_id": "abcdefghijk",
        "title": "Requested song",
        "url": "https://www.youtube.com/watch?v=abcdefghijk",
        "status": "ready_for_in_app_player",
    }
