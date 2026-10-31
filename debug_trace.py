"""Verify Jev API with corrected model name, then run full agent end-to-end test."""
import sys, os
sys.path.insert(0, ".")
from dotenv import load_dotenv
load_dotenv(".env")

jev_api_key = os.environ.get("JEV_API_KEY")
jev_model = os.environ.get("JEV_MODEL", "jev-latest")
print(f"JEV_MODEL from env: {jev_model!r}")

# --- Step 1: Verify Jev API ---
from typesafe_sdk import Choice, Noul, TypeSafeClient, TypeSafeError

try:
    with TypeSafeClient(api_key=jev_api_key, model=jev_model) as client:
        response = client.system_one(
            state={
                "user_request": {"trust": "TRUSTED", "text": "Search YouTube for Python tutorials"},
                "active_contract": {"trust": "TRUSTED", "value": {"allowed_tools": ["search_youtube"]}},
                "proposed_action": {
                    "trust": "PROPOSED_AGENT_ACTION",
                    "step_number": 1,
                    "tool": "search_youtube",
                    "arguments": {"query": "Python tutorials"},
                    "risk": "LOW",
                },
                "evidence_so_far": [],
            },
            questions={
                "action_alignment": Choice(
                    instructions="How well does the proposed tool action follow the request?",
                    criteria={"aligned": "The action is within the request.", "unclear": "Unclear.", "misaligned": "The action conflicts."},
                ),
            },
        )
    print(f"Jev API call SUCCEEDED with model={jev_model!r}")
    print(f"  action_alignment: {response.choices['action_alignment'].choice}")
    print(f"  confidence:       {response.choices['action_alignment'].confidence:.3f}")
except TypeSafeError as e:
    print(f"Jev API FAILED: {e}")
    sys.exit(1)

# --- Step 2: Run agent end-to-end (YouTube-only, no Gmail needed) ---
print("\nRunning YouTube-only agent end-to-end test...")
from agent.youtube_tools import build_youtube_service
from agent.gmail_agent import run_research_agent
from unittest.mock import MagicMock

gmail_mock = MagicMock()
youtube_svc = build_youtube_service()

result = run_research_agent(
    "Use YouTube only. User request: Find a recent Python tutorial video",
    gmail_mock,
    youtube_svc,
    allowed_tools=["search_youtube", "get_youtube_channel_details"],
)

print(f"Agent run SUCCESS")
print(f"Answer (first 400 chars):\n{result.answer[:400]}\n")
print("Trace records:")
for r in result.trust_report.primary_trace.records:
    label = f"[{r.kind}] {r.name}: {r.status}"
    if r.kind == "gate_decision":
        label += f" | action={r.detail.get('action')} reasons={r.detail.get('reasons')}"
    print(f"  {label}")

print("\nTrust checks:")
for c in result.trust_report.checks:
    print(f"  {c.name}: {c.status}")
