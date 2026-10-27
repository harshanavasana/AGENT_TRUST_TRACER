"""In-memory trace records and evidence-aware checks for research-agent runs."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from typing import Any


@dataclass
class TraceRecord:
    event_id: int
    kind: str
    name: str
    status: str
    duration_ms: float
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunTrace:
    records: list[TraceRecord] = field(default_factory=list)
    _next_event_id: int = 1
    _next_evidence_id: int = 1

    def add(
        self,
        kind: str,
        name: str,
        status: str,
        duration_ms: float,
        **detail: Any,
    ) -> TraceRecord:
        record = TraceRecord(
            event_id=self._next_event_id,
            kind=kind,
            name=name,
            status=status,
            duration_ms=round(duration_ms, 2),
            detail=detail,
        )
        self._next_event_id += 1
        self.records.append(record)
        return record

    def add_model_turn(
        self,
        turn: int,
        tool_names: list[str],
        duration_ms: float,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        self.add(
            "model",
            f"Model turn {turn}",
            "tool_request" if tool_names else "final_answer",
            duration_ms,
            requested_tools=tool_names,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    def add_tool_call(
        self,
        name: str,
        source: str,
        arguments: dict[str, Any],
        status: str,
        duration_ms: float,
        *,
        result: str = "",
        error: str = "",
        evidence_id: str = "",
    ) -> None:
        self.add(
            "tool",
            name,
            status,
            duration_ms,
            source=source,
            arguments=arguments,
            result=result,
            error=error,
            evidence_id=evidence_id,
        )

    def new_evidence_id(self) -> str:
        evidence_id = f"E{self._next_evidence_id}"
        self._next_evidence_id += 1
        return evidence_id

    def as_dict(self) -> dict[str, Any]:
        return {"records": [asdict(record) for record in self.records]}


@dataclass
class TrustCheck:
    check_id: str
    name: str
    status: str
    score: float | None
    confidence: str
    summary: str
    evidence: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrustReport:
    checks: list[TrustCheck]
    primary_trace: RunTrace
    consistency_trace: RunTrace | None
    consistency_answer: str | None
    hybrid_decisions: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "checks": [check.as_dict() for check in self.checks],
            "primary_trace": self.primary_trace.as_dict(),
            "consistency_trace": (
                self.consistency_trace.as_dict()
                if self.consistency_trace is not None
                else None
            ),
            "consistency_answer": self.consistency_answer,
            "hybrid_decisions": self.hybrid_decisions,
        }


@dataclass
class ResearchRun:
    answer: str
    trust_report: TrustReport
    pending_approval: dict[str, Any] | None = None
    continuation: Any | None = field(default=None, repr=False)
    run_id: str = ""
    task_type: str = "general_research"


def _tool_records(trace: RunTrace) -> list[TraceRecord]:
    return [record for record in trace.records if record.kind == "tool"]


def _evidence_records(trace: RunTrace) -> list[TraceRecord]:
    return [
        record
        for record in _tool_records(trace)
        if record.status == "success" and record.detail.get("evidence_id")
    ]


def _grounding_check(answer: str, trace: RunTrace) -> TrustCheck:
    evidence = _evidence_records(trace)
    valid_ids = {str(record.detail["evidence_id"]) for record in evidence}
    citations = set(re.findall(r"\[(E\d+)\]", answer))
    invalid = sorted(citations - valid_ids)
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z])|\n+", answer)
        if sentence.strip()
    ]
    cited_sentences = [
        sentence
        for sentence in sentences
        if any(f"[{evidence_id}]" in sentence for evidence_id in valid_ids)
    ]
    if not evidence:
        return TrustCheck(
            "grounding",
            "Grounding",
            "not_assessable",
            None,
            "high",
            "The agent produced no successful source evidence, so factual grounding cannot be checked.",
        )
    coverage = len(cited_sentences) / max(len(sentences), 1)
    if invalid:
        status = "warning"
        summary = f"The answer cited unknown evidence IDs: {', '.join(invalid)}."
    elif coverage == 1:
        status = "pass"
        summary = (
            "Every answer sentence cites evidence recorded from a successful tool call. "
            "This verifies citation coverage, not semantic entailment."
        )
    elif coverage > 0:
        status = "warning"
        summary = (
            f"{len(cited_sentences)} of {len(sentences)} answer sentences cite recorded evidence; "
            "uncited claims may be unsupported."
        )
    else:
        status = "warning"
        summary = (
            "Source evidence exists, but the answer did not cite any of its evidence IDs."
        )
    return TrustCheck(
        "grounding",
        "Grounding",
        status,
        round(coverage, 2),
        "high",
        summary,
        [f"{record.detail['evidence_id']}: {record.name}" for record in evidence],
    )


_CONTRADICTORY_TERMS = (
    ("active", "inactive"),
    ("enabled", "disabled"),
    ("sent", "unsent"),
    ("sent", "draft"),
    ("succeeded", "failed"),
    ("success", "failure"),
    ("completed", "incomplete"),
    ("available", "unavailable"),
    ("present", "missing"),
)
_STOP_WORDS = {
    "about", "after", "again", "also", "and", "are", "before", "been",
    "being", "but", "for", "from", "has", "have", "into", "its", "more",
    "not", "only", "that", "the", "their", "then", "there", "these",
    "they", "this", "was", "were", "what", "when", "where", "which",
    "with", "you",
}


def _sentence_terms(sentence: str) -> set[str]:
    return {
        word.casefold()
        for word in re.findall(r"[A-Za-z0-9_@.-]+", sentence)
        if len(word) > 2 and word.casefold() not in _STOP_WORDS
    }


def _contradiction_check(answer: str, trace: RunTrace) -> TrustCheck:
    evidence_sentences = []
    for record in _evidence_records(trace):
        evidence_sentences.extend(
            re.split(
                r"(?<=[.!?])\s+(?=[A-Z])|\n+",
                str(record.detail.get("result", "")),
            )
        )
    answer_sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])|\n+", answer)
    conflicts = []
    for answer_sentence in answer_sentences:
        answer_terms = _sentence_terms(answer_sentence)
        for positive, negative in _CONTRADICTORY_TERMS:
            answer_value = negative if re.search(rf"\b{negative}\b", answer_sentence, re.I) else positive
            opposite = positive if answer_value == negative else negative
            if not re.search(rf"\b{answer_value}\b", answer_sentence, re.I):
                continue
            for evidence_sentence in evidence_sentences:
                evidence_terms = _sentence_terms(evidence_sentence)
                if not re.search(rf"\b{opposite}\b", evidence_sentence, re.I):
                    continue
                shared_subject = answer_terms & evidence_terms - {
                    positive,
                    negative,
                }
                if shared_subject:
                    conflicts.append(
                        f"Answer says “{answer_value}” while evidence says "
                        f"“{opposite}” about {', '.join(sorted(shared_subject)[:4])}."
                    )
                    break
    if conflicts:
        return TrustCheck(
            "contradiction",
            "Contradiction",
            "warning",
            0.25,
            "medium",
            "Possible explicit status conflict found. This is a conservative lexical check, not a general semantic verifier.",
            conflicts[:5],
        )
    return TrustCheck(
        "contradiction",
        "Contradiction",
        "pass" if evidence_sentences else "not_assessable",
        1.0 if evidence_sentences else None,
        "low",
        (
            "No configured opposite-status terms were found between the answer and evidence. "
            "Subtle semantic contradictions may not be detected."
            if evidence_sentences
            else "No successful source evidence is available for comparison."
        ),
    )


def _precondition_check(trace: RunTrace) -> TrustCheck:
    tools = _tool_records(trace)
    failures = [
        record for record in tools if record.status == "error"
    ]
    if failures:
        return TrustCheck(
            "preconditions",
            "Precondition / Postcondition",
            "warning",
            0.0,
            "high",
            f"{len(failures)} source operation(s) failed. Gmail compose calls report API acceptance only; the app does not independently verify resulting draft or delivery state.",
            [f"{record.name}: {record.detail.get('error', 'operation failed')}" for record in failures],
        )
    successful_writes = [
        record
        for record in tools
        if record.status == "success"
        and record.name in {"create_gmail_draft", "send_gmail_message"}
    ]
    return TrustCheck(
        "preconditions",
        "Precondition / Postcondition",
        "not_applicable",
        None,
        "high",
        (
            "Gmail compose calls succeeded at the API boundary, but this app does not independently read back the draft or verify message delivery."
            if successful_writes
            else "No Gmail draft or send action ran. Read-only calls do not change remote state, so before/after state verification is not applicable."
        ),
        [
            (
                f"{record.name}: Gmail write API accepted the request; "
                "resulting state is not independently verified"
                if record.name in {"create_gmail_draft", "send_gmail_message"}
                else f"{record.name}: successful read operation"
            )
            for record in tools
        ][:8],
    )


def _requested_sources(question: str) -> set[str]:
    normalized = question.casefold()
    gmail_requested = bool(re.search(r"\b(gmail|email|emails|inbox)\b", normalized))
    youtube_requested = bool(
        re.search(r"\b(youtube|video|videos|channel|channels)\b", normalized)
    )
    if gmail_requested or youtube_requested:
        return {
            source
            for source, requested in (
                ("gmail", gmail_requested),
                ("youtube", youtube_requested),
            )
            if requested
        }
    return {"gmail", "youtube"}


def _scope_check(question: str, trace: RunTrace) -> TrustCheck:
    tools = _tool_records(trace)
    requested_sources = _requested_sources(question)
    out_of_scope = [
        record
        for record in tools
        if record.detail.get("source") not in {"gmail", "youtube"}
        or record.detail.get("source") not in requested_sources
    ]
    if out_of_scope:
        return TrustCheck(
            "scope",
            "Scope creep",
            "warning",
            0.0,
            "medium",
            "The agent called a source not named in the request, or an unapproved tool.",
            [
                f"{record.name} used source {record.detail.get('source', 'unknown')}"
                for record in out_of_scope
            ],
        )
    return TrustCheck(
        "scope",
        "Scope creep",
        "pass",
        1.0,
        "medium",
        "All recorded calls used connected Gmail or YouTube tools from a source requested or permitted by the request; semantic action alignment is assessed separately.",
        [f"{record.name} ({record.detail.get('source')})" for record in tools],
    )


def _trajectory_check(trace: RunTrace) -> TrustCheck:
    seen: dict[tuple[str, str], int] = {}
    repeats = []
    for record in _tool_records(trace):
        key = (record.name, repr(sorted(record.detail.get("arguments", {}).items())))
        seen[key] = seen.get(key, 0) + 1
        if seen[key] == 2:
            repeats.append(f"{record.name} was repeated with identical arguments.")
    if repeats:
        return TrustCheck(
            "trajectory",
            "Trajectory anomaly",
            "warning",
            0.0,
            "high",
            "Repeated identical tool calls may indicate a loop; review the trace.",
            repeats,
        )
    return TrustCheck(
        "trajectory",
        "Trajectory anomaly",
        "pass",
        1.0,
        "high",
        "No identical tool call was repeated during this run.",
        [f"{record.name} ({record.status})" for record in _tool_records(trace)],
    )


def _self_consistency_check(
    answer: str,
    consistency_answer: str | None,
    consistency_error: str | None,
    threshold: float = 0.85,
) -> TrustCheck:
    if consistency_error:
        return TrustCheck(
            "self_consistency",
            "Self-consistency",
            "unavailable",
            None,
            "high",
            f"The second run could not complete: {consistency_error}",
        )
    if consistency_answer is None:
        return TrustCheck(
            "self_consistency",
            "Self-consistency",
            "unavailable",
            None,
            "high",
            "No second answer was produced for comparison.",
        )
    normalize = lambda text: " ".join(text.casefold().split())
    similarity = SequenceMatcher(
        None,
        normalize(answer).split(),
        normalize(consistency_answer).split(),
    ).ratio()
    status = "pass" if similarity >= threshold else "warning"
    return TrustCheck(
        "self_consistency",
        "Self-consistency",
        status,
        round(similarity, 2),
        "medium",
        f"Two independent runs have {similarity:.0%} normalized text similarity. "
        f"This is a lexical comparison (pass threshold {threshold:.0%}); "
        "paraphrases can score lower despite agreeing.",
        [f"First answer: {answer[:400]}", f"Second answer: {consistency_answer[:400]}"],
    )


def build_trust_report(
    question: str,
    answer: str,
    primary_trace: RunTrace,
    consistency_answer: str | None,
    consistency_trace: RunTrace | None,
    consistency_error: str | None,
    self_consistency_threshold: float = 0.85,
) -> TrustReport:
    """Evaluate the six checks and retain ephemeral run traces for UI inspection."""
    checks = [
        _grounding_check(answer, primary_trace),
        _contradiction_check(answer, primary_trace),
        _precondition_check(primary_trace),
        _scope_check(question, primary_trace),
        _trajectory_check(primary_trace),
        _self_consistency_check(
            answer,
            consistency_answer,
            consistency_error,
            self_consistency_threshold,
        ),
    ]
    return TrustReport(checks, primary_trace, consistency_trace, consistency_answer)
