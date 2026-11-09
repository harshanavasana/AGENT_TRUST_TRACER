# Runtime Validation Report

**Generated**: 2026-11-09T14:08:00+05:30
**Target**: `C:\Users\Harsha\Downloads\agent_trust_tracer`

## Summary

| Step | Status | Exit Code | Details |
|------|--------|-----------|---------|
| Startup | PASS | 0 | `python -m streamlit run dashboard/app.py --server.headless true --server.port 8512`; `/_stcore/health` returned HTTP 200 and `ok` |
| Integration tests | PASS | 0 | 79 pytest tests passed |
| Python compilation | PASS | 0 | `compileall` completed without errors |
| Streamlit AppTest | PASS | 0 | App ran with zero Streamlit exceptions |
| Browser E2E | PASS | 0 | Playwright with installed Chrome loaded the dashboard and verified YouTube mode, voice controls, recorder, and command box |
| Live service checks | PASS | 0 | Groq model listing, YouTube search, and Gmail OAuth profile checks passed; Hugging Face model authorization passed previously |

## Environment

- Docker: available after starting Docker Desktop.
- Node.js: available (v26.13.0).
- npm: available (v11.20.0).
- Chrome: available and used with Playwright.
- Python virtual environment: available.
- Browser tier: primary Playwright validation completed.

## Coverage

The automated suite covers Gmail and YouTube tool validation, exact video URL
selection, trust-gate behavior,
bounded execution and observation-only consistency passes,
approval flows, run persistence, voice routing, hosted Hugging Face Whisper
and explicit Streamlit voice-resume error handling,
configuration, Groq fallback behavior, and tracer checks.

The Streamlit startup smoke test confirms that the application initializes and
serves its health endpoint. The AppTest harness confirms that the dashboard
script executes without exceptions in a non-browser runtime. Playwright also
loaded the dashboard in Chrome and verified the YouTube-mode voice interface.

## Known gaps

1. Actual microphone capture and browser permission behavior were not exercised
   with physical audio input.
2. Gmail requires the user to complete OAuth again because the saved token was
   expired/revoked. The application now starts that flow automatically.
3. The Hugging Face transcription endpoint itself was not invoked with audio;
   model authorization and availability passed.
4. YouTube API failure now stops playback rather than scraping a personalized
   results page; this avoids opening an unrelated video.
5. Direct `open_youtube_video` is excluded whenever controlled
   `youtube_search_and_play` is available, so one request has one browser owner.

## Verdict

```text
environment:
  docker: AVAILABLE — Docker Desktop started and docker info succeeded
  node: AVAILABLE — v26.13.0
  playwright: AVAILABLE — Chromium installed; Chrome browser smoke flow passed
  infra-tier: PRIMARY(Docker-based) — daemon available; application has no containerized dependency tier
  browser-tier: PRIMARY(Playwright) — dashboard voice interface rendered and controls were verified
startup: PASS — Streamlit startup, health endpoint HTTP 200, exit_code: 0
integration: PASS — exit_code: 0, passed: 79, failed: 0, skipped: 0
e2e: PASS — exit_code: 0, dashboard browser flow verified in Chrome
overall: NEEDS_SIGNOFF — executable checks pass; physical microphone input still requires user interaction
```
