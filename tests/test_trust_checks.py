from tracer.trust_checks import RunTrace, build_trust_report


def _trace_with_evidence(
    result: str = "The Project status is active.",
    *,
    name: str = "search_gmail",
    source: str = "gmail",
    arguments: dict | None = None,
) -> RunTrace:
    trace = RunTrace()
    trace.add_tool_call(
        name,
        source,
        arguments or {"query": "project"},
        "success",
        3.2,
        result=result,
        evidence_id="E1",
    )
    return trace


def test_grounding_requires_valid_evidence_citations() -> None:
    trace = _trace_with_evidence()

    report = build_trust_report(
        "Search Gmail for the project.",
        "The Project status is active. [E1]",
        trace,
        "The Project status is active. [E1]",
        trace,
        None,
    )

    assert report.checks[0].status == "pass"
    assert report.checks[0].score == 1


def test_grounding_flags_missing_citations_and_contradiction_check_flags_opposites() -> None:
    trace = _trace_with_evidence("The Project status is active.")

    report = build_trust_report(
        "Search Gmail for the project.",
        "The Project status is inactive. [E1]",
        trace,
        "The Project status is inactive. [E1]",
        trace,
        None,
    )

    assert report.checks[0].status == "pass"
    assert report.checks[1].status == "warning"


def test_grounding_rejects_unknown_evidence_ids() -> None:
    report = build_trust_report(
        "Search Gmail for the project.",
        "The Project status is active. [E9]",
        _trace_with_evidence(),
        None,
        None,
        "Second pass was not run.",
    )

    assert report.checks[0].status == "warning"
    assert "E9" in report.checks[0].summary


def test_scope_and_repeat_checks_flag_unrequested_repeated_tool() -> None:
    trace = RunTrace()
    for _ in range(2):
        trace.add_tool_call(
            "search_youtube",
            "youtube",
            {"query": "project"},
            "success",
            2,
            result="Public video.",
            evidence_id=f"E{_ + 1}",
        )

    report = build_trust_report(
        "Search Gmail for my project.",
        "No Gmail evidence was returned.",
        trace,
        "No Gmail evidence was returned.",
        RunTrace(),
        None,
    )

    assert report.checks[3].status == "warning"
    assert report.checks[4].status == "warning"


def test_read_only_precondition_check_is_explicitly_not_applicable() -> None:
    report = build_trust_report(
        "Search YouTube for project updates.",
        "Found a video. [E1]",
        _trace_with_evidence(source="youtube", name="search_youtube"),
        "Found a video. [E1]",
        _trace_with_evidence(source="youtube", name="search_youtube"),
        None,
    )

    check = report.checks[2]
    assert check.status == "not_applicable"
    assert "Read-only" in check.summary


def test_gmail_write_reports_api_acceptance_without_claiming_delivery() -> None:
    trace = RunTrace()
    trace.add_tool_call(
        "send_gmail_message",
        "gmail",
        {
            "to": "recipient@example.test",
            "subject": "Project update",
            "body": "[redacted from trace]",
        },
        "success",
        3,
        result="Email sent (message ID: sent-1).",
    )

    report = build_trust_report(
        "Send an email about the project.",
        "The email was sent. [E1]",
        trace,
        None,
        None,
        None,
    )

    check = report.checks[2]
    assert check.status == "not_applicable"
    assert "API boundary" in check.summary
    assert "does not independently" in check.summary


def test_precondition_check_surfaces_failed_source_calls() -> None:
    trace = RunTrace()
    trace.add_tool_call(
        "search_youtube",
        "youtube",
        {"query": "project"},
        "error",
        3,
        error="HttpError HTTP 403",
    )
    report = build_trust_report(
        "Search YouTube for project updates.",
        "",
        trace,
        None,
        None,
        "YouTube search failed.",
    )

    assert report.checks[2].status == "warning"
    assert "HTTP 403" in report.checks[2].evidence[0]


def test_self_consistency_flags_disagreeing_answers() -> None:
    report = build_trust_report(
        "Find project status.",
        "Project status is active.",
        _trace_with_evidence(),
        "Project status is inactive.",
        _trace_with_evidence(),
        None,
    )

    assert report.checks[5].status == "warning"
    assert report.checks[5].score is not None
    assert report.checks[5].score < 0.8
