import base64
from email import policy
from email.parser import BytesParser

from langchain_core.messages import AIMessage, ToolMessage
import pytest

from tracer.reasoner import (
    CausalSource,
    ScoredReason,
    StepJudgment,
)
from tracer.adaptive import AdaptiveStore
from agent.gmail_agent import (
    DEFAULT_GROQ_MODEL,
    MAX_AGENT_TURNS,
    MAX_MESSAGE_CHARS,
    MAX_SEARCH_RESULTS,
    MAX_SNIPPET_CHARS,
    MAX_TOTAL_READ_CHARS,
    build_gmail_tools,
    extract_message_body,
    run_research_agent,
    resume_research_agent,
)
from agent.youtube_tools import (
    MAX_YOUTUBE_QUERY_CHARS,
    MAX_YOUTUBE_RESULTS,
    build_youtube_tools,
)


def test_default_groq_model_supports_tool_calls() -> None:
    assert DEFAULT_GROQ_MODEL == "openai/gpt-oss-120b"


def test_groq_request_context_is_bounded() -> None:
    assert MAX_SEARCH_RESULTS == 5
    assert MAX_SNIPPET_CHARS == 180
    assert MAX_MESSAGE_CHARS == 3_000
    assert MAX_TOTAL_READ_CHARS == 5_000


def _encoded(text: str) -> str:
    return base64.urlsafe_b64encode(text.encode()).decode()


class _PassingJudge:
    def with_structured_output(self, _schema):
        return self

    def invoke(self, _messages):
        return StepJudgment(
            intent_alignment=ScoredReason(score=1, reason="Matches the request."),
            causal_source=CausalSource(
                source="user",
                earlier_step=0,
                reason="The user asked for the research.",
            ),
            paraphrase_check=ScoredReason(
                score=1,
                reason="No untrusted instruction is being followed.",
            ),
            verdict="PASS",
            confidence=0.98,
            evidence_steps=[0],
            reason="The read-only query is aligned with the user's request.",
        )


def test_extract_message_body_prefers_plain_text() -> None:
    payload = {
        "mimeType": "multipart/alternative",
        "parts": [
            {"mimeType": "text/html", "body": {"data": _encoded("<p>HTML</p>")}},
            {"mimeType": "text/plain", "body": {"data": _encoded("Plain text")}},
        ],
    }

    assert extract_message_body(payload) == "Plain text"


def test_extract_message_body_reads_nested_plain_text_part() -> None:
    payload = {
        "mimeType": "multipart/mixed",
        "parts": [
            {
                "mimeType": "multipart/alternative",
                "parts": [
                    {
                        "mimeType": "text/plain",
                        "body": {"data": _encoded("Nested message")},
                    }
                ],
            }
        ],
    }

    assert extract_message_body(payload) == "Nested message"


def test_read_tool_rejects_ids_not_returned_by_search() -> None:
    class FakeService:
        def users(self):
            return self

        def messages(self):
            return self

    read_tool = build_gmail_tools(FakeService())[1]
    with pytest.raises(ValueError, match="Search for a message"):
        read_tool.invoke({"message_id": "unseen"})


class _Request:
    def __init__(self, value):
        self.value = value

    def execute(self):
        return self.value


class _GmailMessages:
    def __init__(self):
        self.searches = 0
        self.sent_messages = []

    def list(self, **_kwargs):
        self.searches += 1
        return _Request({"messages": [{"id": "message-1"}]})

    def get(self, *, id, format, **_kwargs):
        return _Request(
            {
                "payload": {
                    "headers": [
                        {"name": "From", "value": "sender@example.test"},
                        {"name": "Subject", "value": "Project update"},
                    ]
                },
                "snippet": "The project is on track.",
            }
        )

    def send(self, *, userId, body):
        self.sent_messages.append(body["raw"])
        return _Request({"id": "sent-message-1"})


class _GmailDrafts:
    def __init__(self):
        self.created_drafts = []

    def create(self, *, userId, body):
        self.created_drafts.append(body["message"]["raw"])
        return _Request({"id": "draft-1"})


class _GmailService:
    def __init__(self):
        self.message_api = _GmailMessages()
        self.draft_api = _GmailDrafts()

    def users(self):
        return self

    def messages(self):
        return self.message_api

    def drafts(self):
        return self.draft_api


def test_gmail_draft_tool_encodes_message_without_sending() -> None:
    service = _GmailService()
    draft_tool = build_gmail_tools(service)[2]

    result = draft_tool.invoke(
        {
            "to": "recipient@example.test",
            "subject": "Project update",
            "body": "The project is ready.",
        }
    )

    raw_message = base64.urlsafe_b64decode(
        service.draft_api.created_drafts[0]
    )
    message = BytesParser(policy=policy.default).parsebytes(raw_message)
    assert result == "Draft created (draft ID: draft-1). It has not been sent."
    assert message["To"] == "recipient@example.test"
    assert message["Subject"] == "Project update"
    assert message.get_content().strip() == "The project is ready."
    assert service.message_api.sent_messages == []


def test_gmail_send_is_approved_before_execution_and_not_repeated(tmp_path) -> None:
    class SendModel:
        def __init__(self):
            self.available_tools = set()

        def bind_tools(self, tools):
            self.available_tools = {item.name for item in tools}
            return self

        def with_structured_output(self, _schema):
            return _PassingJudge()

        def invoke(self, messages):
            if isinstance(messages[-1], ToolMessage):
                return AIMessage(content="The email was sent. [E1]")
            if "send_gmail_message" in self.available_tools:
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "send_gmail_message",
                            "args": {
                                "to": "recipient@example.test",
                                "subject": "Project update",
                                "body": "The project is ready.",
                            },
                            "id": "send-1",
                            "type": "tool_call",
                        }
                    ],
                )
            return AIMessage(content="The email was sent. [E1]")

    service = _GmailService()
    result = run_research_agent(
        "Send an email to recipient@example.test about the project.",
        service,
        _YouTubeService(),
        model=SendModel(),
        adaptive_store=AdaptiveStore(tmp_path / "adaptive.sqlite"),
    )

    assert result.pending_approval["tool"] == "send_gmail_message"
    assert result.pending_approval["arguments"]["to"] == "recipient@example.test"
    assert result.pending_approval["arguments"]["subject"] == "Project update"
    assert result.pending_approval["arguments"]["body"] == "The project is ready."
    assert service.message_api.sent_messages == []

    result = resume_research_agent(result, approved=True)

    assert result.pending_approval is None
    assert len(service.message_api.sent_messages) == 1
    assert "The email was sent" in result.answer
    consistency_tools = [
        event
        for event in result.trust_report.consistency_trace.records
        if event.kind == "tool"
    ]
    assert all(
        event.name not in {"create_gmail_draft", "send_gmail_message"}
        for event in consistency_tools
    )

class _SearchThenAnswerModel:
    def with_structured_output(self, _schema):
        return _PassingJudge()

    def bind_tools(self, _tools):
        return self

    def invoke(self, messages):
        if isinstance(messages[-1], ToolMessage):
            return AIMessage(content="The project is on track. [E1]")
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search_gmail",
                    "args": {"query": "subject:project"},
                    "id": "search-1",
                    "type": "tool_call",
                }
            ],
        )


def test_langgraph_runs_agent_tool_agent_flow(tmp_path) -> None:
    service = _GmailService()
    answer = run_research_agent(
        "What is the project update?",
        service,
        _YouTubeService(),
        model=_SearchThenAnswerModel(),
        adaptive_store=AdaptiveStore(tmp_path / "adaptive.sqlite"),
    )

    assert answer.answer == "The project is on track. [E1]"
    assert service.message_api.searches == 2
    assert len(answer.trust_report.checks) == 6
    assert answer.trust_report.checks[-1].status == "pass"
    assert len(answer.trust_report.primary_trace.records) == 8
    assert len(answer.trust_report.consistency_trace.records) == 8


class _AlwaysSearchModel:
    def __init__(self):
        self.calls = 0

    def bind_tools(self, _tools):
        return self

    def with_structured_output(self, _schema):
        return _PassingJudge()

    def invoke(self, _messages):
        self.calls += 1
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search_gmail",
                    "args": {"query": "subject:project"},
                    "id": f"search-{self.calls}",
                    "type": "tool_call",
                }
            ],
        )


def test_langgraph_returns_notice_at_agent_turn_limit(tmp_path) -> None:
    service = _GmailService()
    model = _AlwaysSearchModel()
    answer = run_research_agent(
        "Keep searching",
        service,
        _YouTubeService(),
        model=model,
        adaptive_store=AdaptiveStore(tmp_path / "adaptive.sqlite"),
    )

    assert "step limit" in answer.answer
    assert model.calls == MAX_AGENT_TURNS * 2
    assert service.message_api.searches == (MAX_AGENT_TURNS - 1) * 2


class _UnknownJudge:
    def with_structured_output(self, _schema):
        return self

    def invoke(self, _messages):
        return StepJudgment(
            intent_alignment=ScoredReason(score=0.5, reason="Insufficient context."),
            causal_source=CausalSource(
                source="unknown",
                earlier_step=None,
                reason="The judge cannot determine the action source.",
            ),
            paraphrase_check=ScoredReason(
                score=0.5,
                reason="Insufficient context to check for paraphrase.",
            ),
            verdict="UNKNOWN",
            confidence=0,
            evidence_steps=[],
            reason="Mock judge has insufficient evidence.",
        )


class _ReviewRequiredModel:
    def bind_tools(self, _tools):
        return self

    def with_structured_output(self, _schema, **_kwargs):
        return _UnknownJudge()

    def invoke(self, messages):
        if isinstance(messages[-1], ToolMessage):
            if "human reviewer rejected" in messages[-1].content:
                return AIMessage(content="I will not perform the rejected read.")
            return AIMessage(content="The search completed.")
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search_gmail",
                    "args": {"query": "subject:project"},
                    "id": f"search-{id(messages)}",
                    "type": "tool_call",
                }
            ],
        )


def test_unknown_high_risk_action_pauses_until_human_decision(tmp_path) -> None:
    service = _GmailService()
    model = _ReviewRequiredModel()
    result = run_research_agent(
        "Search Gmail for the project",
        service,
        _YouTubeService(),
        model=model,
        adaptive_store=AdaptiveStore(tmp_path / "adaptive.sqlite"),
    )

    assert result.pending_approval["tool"] == "search_gmail"
    assert service.message_api.searches == 0
    result = resume_research_agent(result, approved=False)
    assert result.pending_approval["tool"] == "search_gmail"
    assert service.message_api.searches == 0
    result = resume_research_agent(result, approved=False)

    assert result.pending_approval is None
    assert "rejected read" in result.answer
    assert service.message_api.searches == 0
    assert any(
        record.kind == "human_review" and record.status == "rejected"
        for trace in (
            result.trust_report.primary_trace,
            result.trust_report.consistency_trace,
        )
        for record in trace.records
    )


def test_approved_high_risk_review_executes_only_after_resume(tmp_path) -> None:
    service = _GmailService()
    result = run_research_agent(
        "Search Gmail for the project",
        service,
        _YouTubeService(),
        model=_ReviewRequiredModel(),
        adaptive_store=AdaptiveStore(tmp_path / "adaptive.sqlite"),
    )

    assert service.message_api.searches == 0
    result = resume_research_agent(result, approved=True)
    assert service.message_api.searches == 1
    assert result.pending_approval["tool"] == "search_gmail"
    result = resume_research_agent(result, approved=True)

    assert service.message_api.searches == 2
    assert result.pending_approval is None
    assert result.answer == "The search completed."


class _YouTubeRequest:
    def __init__(self, value):
        self.value = value

    def execute(self):
        return self.value


class _YouTubeService:
    def __init__(self):
        self.search_limit = None
        self.video_ids = None
        self.channel_ids = None

    def search(self):
        return self

    def videos(self):
        return self

    def channels(self):
        return self

    def list(self, **kwargs):
        if kwargs.get("type") == "video":
            self.search_limit = kwargs["maxResults"]
            return _YouTubeRequest(
                {
                    "items": [
                        {
                            "id": {"videoId": "video-1"},
                            "snippet": {
                                "title": "Public update",
                                "description": "A public summary.",
                                "channelId": "channel-1",
                                "channelTitle": "Example Channel",
                                "publishedAt": "2026-01-01T00:00:00Z",
                            },
                        }
                    ]
                }
            )
        if kwargs.get("part") == "contentDetails,statistics":
            self.video_ids = kwargs["id"]
            return _YouTubeRequest(
                {
                    "items": [
                        {
                            "id": "video-1",
                            "contentDetails": {"duration": "PT2M"},
                            "statistics": {"viewCount": "42"},
                        }
                    ]
                }
            )
        self.channel_ids = kwargs["id"]
        return _YouTubeRequest(
            {
                "items": [
                    {
                        "snippet": {
                            "title": "Example Channel",
                            "publishedAt": "2020-01-01T00:00:00Z",
                            "description": "Channel information.",
                        },
                        "statistics": {
                            "subscriberCount": "10",
                            "videoCount": "2",
                            "viewCount": "50",
                        },
                    }
                ]
            }
        )


def test_youtube_tools_search_video_details_and_channel_details() -> None:
    service = _YouTubeService()
    search_tool, channel_tool = build_youtube_tools(service)

    search_result = search_tool.invoke({"query": "public project update"})
    channel_result = channel_tool.invoke({"channel_id": "channel-1"})

    assert service.search_limit == MAX_YOUTUBE_RESULTS
    assert service.video_ids == "video-1"
    assert "Title: Public update" in search_result
    assert "Duration: PT2M" in search_result
    assert "Views: 42" in search_result
    assert "Subscribers: 10" in channel_result
    assert service.channel_ids == "channel-1"


def test_youtube_channel_tool_requires_channel_from_search() -> None:
    _, channel_tool = build_youtube_tools(_YouTubeService())

    with pytest.raises(ValueError, match="Search YouTube first"):
        channel_tool.invoke({"channel_id": "unseen-channel"})


def test_youtube_search_rejects_empty_or_oversized_queries() -> None:
    search_tool, _ = build_youtube_tools(_YouTubeService())

    with pytest.raises(ValueError, match="non-empty"):
        search_tool.invoke({"query": "  "})
    with pytest.raises(ValueError, match="at most"):
        search_tool.invoke({"query": "x" * (MAX_YOUTUBE_QUERY_CHARS + 1)})


class _YouTubeSearchThenAnswerModel:
    def __init__(self):
        self.tool_names = []

    def bind_tools(self, tools):
        self.tool_names = [tool.name for tool in tools]
        return self

    def invoke(self, messages):
        if isinstance(messages[-1], ToolMessage):
            return AIMessage(content="A public video update was found. [E1]")
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search_youtube",
                    "args": {"query": "public project update"},
                    "id": "youtube-search-1",
                    "type": "tool_call",
                }
            ],
        )


def test_single_langgraph_agent_can_route_to_youtube_tools(tmp_path) -> None:
    model = _YouTubeSearchThenAnswerModel()
    answer = run_research_agent(
        "Find public project videos.",
        _GmailService(),
        _YouTubeService(),
        model=model,
        adaptive_store=AdaptiveStore(tmp_path / "adaptive.sqlite"),
    )

    assert answer.answer == "A public video update was found. [E1]"
    assert "search_gmail" in model.tool_names
    assert "read_gmail_message" in model.tool_names


def test_selected_source_limits_tools_bound_to_the_agent(tmp_path) -> None:
    model = _YouTubeSearchThenAnswerModel()
    run_research_agent(
        "Find public project videos.",
        _GmailService(),
        _YouTubeService(),
        model=model,
        adaptive_store=AdaptiveStore(tmp_path / "adaptive.sqlite"),
        allowed_tools=["search_youtube", "get_youtube_channel_details"],
    )

    assert model.tool_names == ["search_youtube", "get_youtube_channel_details"]
    assert "search_youtube" in model.tool_names
    assert "get_youtube_channel_details" in model.tool_names


def test_gmail_tools_bound_search_and_email_text() -> None:
    class Request:
        def __init__(self, value):
            self.value = value

        def execute(self):
            return self.value

    class Messages:
        def __init__(self):
            self.search_limit = None

        def list(self, **kwargs):
            self.search_limit = kwargs["maxResults"]
            return Request({"messages": [{"id": "one"}, {"id": "two"}]})

        def get(self, *, id, format, **kwargs):
            headers = [
                {"name": "From", "value": "sender@example.test"},
                {"name": "Subject", "value": "Test"},
            ]
            if format == "metadata":
                return Request(
                    {
                        "payload": {"headers": headers},
                        "snippet": "s" * (MAX_SNIPPET_CHARS + 100),
                    }
                )
            return Request(
                {
                    "payload": {
                        "mimeType": "text/plain",
                        "body": {"data": _encoded("b" * 10_000)},
                        "headers": headers,
                    }
                }
            )

    class Service:
        def __init__(self):
            self.message_api = Messages()

        def users(self):
            return self

        def messages(self):
            return self.message_api

    service = Service()
    search_tool, read_tool = build_gmail_tools(service)[:2]
    search_result = search_tool.invoke({"query": "from:sender"})
    first_read = read_tool.invoke({"message_id": "one"})
    second_read = read_tool.invoke({"message_id": "two"})
    exhausted_read = read_tool.invoke({"message_id": "one"})

    assert service.message_api.search_limit == MAX_SEARCH_RESULTS
    assert len(search_result) < MAX_SEARCH_RESULTS * (MAX_SNIPPET_CHARS + 200)
    assert "[Email text truncated" in first_read
    assert len(first_read) < MAX_MESSAGE_CHARS + 200
    assert len(second_read) < MAX_TOTAL_READ_CHARS - MAX_MESSAGE_CHARS + 200
    assert exhausted_read == "The per-question email reading limit was reached."
