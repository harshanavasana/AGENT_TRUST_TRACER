"""Streamlit interface for the Gmail + YouTube research agent."""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from agent.gmail_agent import (
    ResearchAgentError,
    build_gmail_service,
    run_research_agent,
    resume_research_agent,
)
from agent.youtube_tools import build_youtube_service
from tracer.adaptive import AdaptiveStore
from tracer.fusion import load_fusion_config

load_dotenv()

st.set_page_config(page_title="Source Desk", page_icon="✳", layout="wide")
st.markdown(
    """
    <style>
    :root {
        --jet-black: #0B0B0B;
        --hot-pink: #FF20AB;
        --chartreuse: #D4FF3F;
        --panel: #151515;
        --muted: #A7A7A7;
    }
    [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background:
            radial-gradient(ellipse at 2% 5%, rgba(255, 32, 171, .26), transparent 34%),
            radial-gradient(ellipse at 98% 90%, rgba(212, 255, 63, .18), transparent 32%),
            radial-gradient(ellipse at 74% 18%, rgba(255, 32, 171, .08), transparent 28%),
            linear-gradient(140deg, #0B0B0B 0%, #11100F 48%, #0B0B0B 100%);
        color: #F7F7F7;
        background-attachment: fixed;
        background-size: 140% 140%;
        animation: mesh-drift 24s ease-in-out infinite alternate;
    }
    @keyframes mesh-drift {
        from { background-position: 0% 0%; }
        to { background-position: 100% 100%; }
    }
    [data-testid="stMainBlockContainer"] {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }
    h1, h2, h3, p, label, [data-testid="stMarkdownContainer"] {
        color: #F7F7F7;
    }
    h1 { letter-spacing: -0.045em; font-weight: 750; }
    h2 { letter-spacing: -0.025em; }
    [data-testid="stTextArea"] textarea, [data-testid="stTextInput"] input {
        background: #171717;
        color: #F7F7F7;
        border: 1px solid #383838;
        border-radius: 12px;
    }
    [data-testid="stTextArea"] textarea:focus,
    [data-testid="stTextInput"] input:focus {
        border-color: var(--hot-pink);
        box-shadow: 0 0 0 1px var(--hot-pink);
    }
    [data-testid="stButton"] button {
        background: var(--hot-pink);
        color: #0B0B0B;
        border: 0;
        border-radius: 999px;
        font-weight: 750;
        min-height: 2.7rem;
        padding-inline: 1.25rem;
    }
    [data-testid="stButton"] button:hover {
        background: var(--chartreuse);
        color: #0B0B0B;
    }
    [data-testid="stMetric"] {
        background: var(--panel);
        border: 1px solid #2B2B2B;
        border-radius: 14px;
        padding: 1rem 1.15rem;
    }
    [data-testid="stMetricLabel"] { color: var(--muted); }
    [data-testid="stMetricValue"] { color: var(--chartreuse); }
    [data-testid="stExpander"] {
        border: 1px solid #303030;
        border-radius: 12px;
        background: var(--panel);
    }
    [data-testid="stAlert"] { border-radius: 12px; }
    hr { border-color: #303030; }
    @media (prefers-reduced-motion: reduce) {
        [data-testid="stAppViewContainer"] { animation: none; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    '<div style="color:#FF20AB;font-size:.78rem;font-weight:800;'
    'letter-spacing:.18em;text-transform:uppercase;margin-bottom:.6rem">'
    "SOURCE DESK&nbsp; / &nbsp;PRIVATE RESEARCH</div>",
    unsafe_allow_html=True,
)
st.markdown("# Ask your sources.")
st.caption(
    "Search Gmail and public YouTube. Drafting and sending always require "
    "your approval; routine read checks stay with the agent tracer."
)

client_file = os.environ.get("GOOGLE_OAUTH_CLIENT_FILE", "client_secret.json")
token_file = os.environ.get("GMAIL_TOKEN_FILE", ".gmail-token.json")

if not os.environ.get("GROQ_API_KEY"):
    st.error("Set GROQ_API_KEY in your local .env file before starting.")
    st.stop()
if not os.environ.get("YOUTUBE_API_KEY"):
    st.error("Set YOUTUBE_API_KEY in your local .env file before starting.")
    st.stop()
if not Path(client_file).expanduser().is_file():
    st.error(
        "Set GOOGLE_OAUTH_CLIENT_FILE to the downloaded OAuth Desktop app "
        "client JSON file. Do not share or commit that file."
    )
    st.stop()

st.sidebar.markdown("### Your workspace")
st.sidebar.caption("Manage source connections and your local session.")
if st.sidebar.button("Connect / refresh Gmail", type="primary"):
    st.session_state["gmail_service"] = build_gmail_service(client_file, token_file)
    st.sidebar.success("Gmail connected with read and compose access.")

service = st.session_state.get("gmail_service")
if service is None:
    st.info("Use **Connect / refresh Gmail** in the sidebar to begin.")
    st.stop()
st.sidebar.markdown("**Connected sources**")
st.sidebar.markdown("- 🟢 Gmail · private messages")
st.sidebar.markdown("- 🟢 YouTube · public videos")

st.session_state.setdefault("research_run", None)
st.session_state.setdefault("trust_report", None)
st.session_state.setdefault("query_draft", "")
st.session_state.setdefault("query_history", [])
adaptive_db = os.environ.get(
    "AGENT_TRUST_DB",
    str(Path(__file__).parents[1] / ".agent-trust.sqlite"),
)
adaptive_store = AdaptiveStore(adaptive_db)

def set_query(value: str) -> None:
    st.session_state["query_draft"] = value


st.markdown("### What are you looking for?")
source_mode = st.radio(
    "Search in",
    ["Both sources", "Gmail", "YouTube"],
    horizontal=True,
    label_visibility="collapsed",
    key="source_mode",
)
prompt_cols = st.columns(3)
quick_prompts = [
    "Find the most important email I received today.",
    "Find the latest public video from a YouTube channel.",
    "Compare a recent email with a public video update.",
]
for column, prompt, key in zip(
    prompt_cols,
    quick_prompts,
    ("quick_email", "quick_video", "quick_compare"),
):
    column.button(
        prompt,
        key=key,
        on_click=set_query,
        args=(prompt,),
        use_container_width=True,
    )

history = st.session_state["query_history"]
if history:
    with st.expander("Recent searches"):
        selected_search = st.selectbox(
            "Choose a previous question",
            [""] + history[-5:][::-1],
            label_visibility="collapsed",
            key="selected_search",
        )
        if st.button("Use this search", disabled=not selected_search):
            set_query(selected_search)
            st.rerun()

with st.form("source_search", border=True):
    question = st.text_area(
        "Your question",
        placeholder="For example: Find the most important project email from last week.",
        key="query_draft",
        height=120,
    )
    submit_search = st.form_submit_button(
        "Search sources",
        type="primary",
        disabled=not question.strip(),
        use_container_width=True,
    )

if submit_search:
    scoped_question = {
        "Gmail": f"Use Gmail only. User request: {question}",
        "YouTube": f"Use YouTube only. User request: {question}",
    }.get(source_mode, question)
    try:
        with st.spinner("Searching the connected sources..."):
            youtube_service = build_youtube_service()
            st.session_state["research_run"] = run_research_agent(
                scoped_question,
                service,
                youtube_service,
            )
            st.session_state["query_history"] = (
                st.session_state["query_history"] + [question]
            )[-10:]
            st.session_state["trust_report"] = st.session_state[
                "research_run"
            ].trust_report
    except ResearchAgentError as error:
        st.error(str(error))
        st.session_state["research_run"] = None
        st.session_state["trust_report"] = error.trace_report
    except RuntimeError as error:
        st.error(str(error))
        st.session_state["research_run"] = None
        st.session_state["trust_report"] = None

research_run = st.session_state.get("research_run")
if research_run is not None and research_run.pending_approval:
    pending = research_run.pending_approval
    decision = pending.get("decision", {})
    st.warning(
        f"Paused before executing **{pending.get('tool')}**. "
        "The tracer found a significant warning; this action has not run yet."
    )
    pending_args = pending.get("arguments", {})
    if pending.get("tool") in {"create_gmail_draft", "send_gmail_message"}:
        st.markdown(f"**To:** {pending_args.get('to', '')}")
        st.markdown(f"**Subject:** {pending_args.get('subject', '')}")
        st.text_area(
            "Exact message body for approval",
            value=pending_args.get("body", ""),
            height=180,
            disabled=True,
        )
    st.json(
        {
            "tool": pending.get("tool"),
            "verdict": decision.get("verdict"),
            "confidence": decision.get("confidence"),
            "caught_by": decision.get("contributing_layers"),
            "anomaly_score": decision.get("anomaly_score"),
            "reasons": decision.get("reasons"),
        }
    )
    approve_col, reject_col = st.columns(2)
    approve_label = (
        "Approve sending this exact email"
        if pending.get("tool") == "send_gmail_message"
        else "Approve creating this exact draft"
    )
    if approve_col.button(approve_label, type="primary"):
        with st.spinner("Resuming the agent..."):
            resumed = resume_research_agent(research_run, approved=True)
            st.session_state["research_run"] = resumed
            st.session_state["trust_report"] = resumed.trust_report
        st.rerun()
    if reject_col.button("Reject this action"):
        with st.spinner("Resuming without executing the rejected action..."):
            resumed = resume_research_agent(research_run, approved=False)
            st.session_state["research_run"] = resumed
            st.session_state["trust_report"] = resumed.trust_report
        st.rerun()

if research_run is not None:
    st.markdown("## Agent answer")
    if research_run.answer:
        st.markdown(research_run.answer)
    elif research_run.pending_approval:
        st.caption("The answer will be produced after the pending review decision.")

report = st.session_state.get("trust_report")
if report is not None:
    primary_tool_calls = [
        record
        for record in report.primary_trace.records
        if record.kind == "tool"
    ]
    passed_checks = sum(check.status == "pass" for check in report.checks)
    review_count = sum(
        item.get("decision") == "REVIEW"
        for item in report.hybrid_decisions
        if item.get("mode") != "full_trace_rules"
    )
    human_review_count = sum(
        record.kind == "human_review"
        for record in report.primary_trace.records
    )
    st.markdown("---")
    st.markdown("## Run metrics")
    model_turns = [
        record
        for record in report.primary_trace.records
        if record.kind == "model"
    ]
    input_tokens = sum(
        int(record.detail.get("input_tokens", 0)) for record in model_turns
    )
    output_tokens = sum(
        int(record.detail.get("output_tokens", 0)) for record in model_turns
    )
    metric_cols = st.columns(6)
    metric_cols[0].metric("Tool calls", len(primary_tool_calls))
    metric_cols[1].metric(
        "Checks passed",
        f"{passed_checks}/{len(report.checks)}",
    )
    metric_cols[2].metric("Review signals", review_count)
    metric_cols[3].metric("Human escalations", human_review_count)
    metric_cols[4].metric("Model tokens", f"{input_tokens} in / {output_tokens} out")
    metric_cols[5].metric(
        "Recorded latency",
        f"{sum(record.duration_ms for record in report.primary_trace.records) / 1000:.1f}s",
    )

    st.markdown("## Hybrid supervisor decisions")
    if report.hybrid_decisions:
        for item in report.hybrid_decisions:
            if item.get("mode") == "full_trace_rules":
                st.caption(
                    f"Full-trace audit · {item.get('trace')} step "
                    f"{item.get('step_number')}: {item.get('decision')} "
                    f"({item.get('rule')})"
                )
                continue
            st.markdown(
                f"**Step {item.get('root_cause', {}).get('manifestation_step', '?')} · "
                f"{item.get('tool')} · {item.get('decision')}**"
            )
            st.write(
                {
                    "caught_by": item.get("layers", []),
                    "confidence": item.get("confidence"),
                    "anomaly_score": item.get("anomaly_score"),
                    "reduced_assurance": item.get("reduced_assurance", False),
                }
            )
            if item.get("root_cause"):
                with st.expander("Root cause and suggested fix"):
                    st.json(item["root_cause"])
            if (
                research_run is not None
                and item.get("decision") in {"PASS", "REVIEW", "FAIL"}
                and item.get("event_id") is not None
            ):
                feedback_cols = st.columns(2)
                step_key = f"{research_run.run_id}-{item['event_id']}"
                if feedback_cols[0].button(
                    "Step verdict correct",
                    key=f"step-correct-{step_key}",
                ):
                    adaptive_store.record_feedback(
                        run_id=research_run.run_id,
                        task_type=research_run.task_type,
                        check_id="semantic_judge",
                        predicted_status=item["decision"],
                        label="correct",
                    )
                    st.success("Step feedback saved without source text.")
                if feedback_cols[1].button(
                    "Step verdict incorrect",
                    key=f"step-incorrect-{step_key}",
                ):
                    adaptive_store.record_feedback(
                        run_id=research_run.run_id,
                        task_type=research_run.task_type,
                        check_id="semantic_judge",
                        predicted_status=item["decision"],
                        label="incorrect",
                    )
                    st.success("Step feedback saved; threshold adjustment is reversible.")
    else:
        st.caption("No tool action has reached the hybrid gate yet.")

    st.markdown("## Six trust checks")
    st.caption(
        "Checks report evidence and limits: citation coverage is not semantic "
        "proof; contradiction detection is lexical; before/after state checks "
        "cannot independently verify Gmail draft creation or message delivery."
    )
    labels = {
        "pass": "✅ PASS",
        "warning": "⚠️ REVIEW",
        "not_applicable": "➖ NOT APPLICABLE",
        "not_assessable": "➖ NOT ASSESSABLE",
        "unavailable": "⚠️ UNAVAILABLE",
    }
    for start in range(0, len(report.checks), 2):
        columns = st.columns(2)
        for column, check in zip(columns, report.checks[start : start + 2]):
            with column:
                score = (
                    f" · score {check.score:.0%}"
                    if check.score is not None
                    else ""
                )
                st.markdown(
                    f"### {check.name} · {labels.get(check.status, check.status)}{score}"
                )
                st.write(check.summary)
                st.caption(f"Confidence: {check.confidence}")
                if check.evidence:
                    with st.expander("Check evidence"):
                        for item in check.evidence:
                            st.write(f"- {item}")
                if research_run is not None:
                    feedback_cols = st.columns(2)
                    if feedback_cols[0].button(
                        "Verdict correct",
                        key=f"feedback-correct-{research_run.run_id}-{check.check_id}",
                    ):
                        adaptive_store.record_feedback(
                            run_id=research_run.run_id,
                            task_type=research_run.task_type,
                            check_id=check.check_id,
                            predicted_status=check.status,
                            label="correct",
                        )
                        st.success("Feedback saved without storing source text.")
                    if feedback_cols[1].button(
                        "Verdict incorrect",
                        key=f"feedback-incorrect-{research_run.run_id}-{check.check_id}",
                    ):
                        adaptive_store.record_feedback(
                            run_id=research_run.run_id,
                            task_type=research_run.task_type,
                            check_id=check.check_id,
                            predicted_status=check.status,
                            label="incorrect",
                        )
                        st.success("Feedback saved; any threshold change is reversible.")

    if research_run is not None and research_run.answer and st.button(
        "Confirm this as a good baseline run"
    ):
        primary_tool_calls = [
            record
            for record in report.primary_trace.records
            if record.kind == "tool" and record.status == "success"
        ]
        tool_names = [record.name for record in primary_tool_calls]
        query_lengths = {
            name: [
                len(str(record.detail.get("arguments", {}).get("query", "")))
                for record in primary_tool_calls
                if record.name == name
                and isinstance(record.detail.get("arguments", {}).get("query"), str)
            ]
            for name in set(tool_names)
        }
        adaptive_store.learn_confirmed_good_run(
            research_run.task_type,
            tool_names,
            query_lengths=query_lengths,
        )
        st.success("Aggregate behavior baseline updated from this confirmed run.")

    st.markdown("## Adaptive learning")
    fusion_config = load_fusion_config()
    st.write(
        {
            "anomaly_review_threshold": adaptive_store.current_threshold(
                "T_anom",
                fusion_config.anomaly_review,
            ),
            "judge_block_threshold": adaptive_store.current_threshold(
                "T_high",
                fusion_config.high_confidence,
            ),
            "drift_summary": adaptive_store.drift_summary(),
        }
    )
    if st.button("Undo last learned threshold change"):
        if adaptive_store.undo_last_threshold_change():
            st.success("Last threshold change reverted.")
        else:
            st.info("There is no unreverted threshold change.")
    if st.button("Undo last feedback label"):
        if adaptive_store.undo_last_feedback():
            st.success("Last feedback example and its linked threshold update were removed.")
        else:
            st.info("There is no feedback label to undo.")
    if st.button("Undo last baseline update"):
        if adaptive_store.undo_last_baseline_update():
            st.success("Last confirmed-good baseline update was reverted.")
        else:
            st.info("There is no baseline update to undo.")

    st.markdown("## Agent trace")
    st.caption(
        "Trace data is kept in this Streamlit session only. Expanding a successful "
        "tool call can display the source result, including email text."
    )
    for run_name, trace in (
        ("Primary run", report.primary_trace),
        ("Self-consistency run", report.consistency_trace),
    ):
        if trace is None:
            continue
        with st.expander(f"{run_name} · {len(trace.records)} events"):
            for record in trace.records:
                st.markdown(
                    f"**#{record.event_id} {record.name}** · "
                    f"{record.kind} · {record.status} · "
                    f"{record.duration_ms:.0f} ms"
                )
                if record.kind == "tool":
                    st.caption(
                        f"Source: {record.detail.get('source')} · "
                        f"Evidence: {record.detail.get('evidence_id') or 'none'}"
                    )
                    st.json(record.detail.get("arguments", {}))
                    if record.detail.get("error"):
                        st.error(record.detail["error"])
                    if record.detail.get("result"):
                        with st.expander("View tool result"):
                            st.text(record.detail["result"])
                elif record.detail.get("requested_tools"):
                    st.write(
                        "Requested tools: "
                        + ", ".join(record.detail["requested_tools"])
                    )
                    st.caption(
                        f"Tokens: {record.detail.get('input_tokens', 0)} input / "
                        f"{record.detail.get('output_tokens', 0)} output"
                    )
                elif record.kind in {"gate_decision", "layer", "human_review"}:
                    st.json(record.detail)
