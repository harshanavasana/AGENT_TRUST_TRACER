import React from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  AlertTriangle,
  Bell,
  CheckCircle2,
  ChevronRight,
  CircleHelp,
  FileSearch,
  Gauge,
  Github,
  Info,
  LayoutDashboard,
  Menu,
  Mic,
  Moon,
  Play,
  Search,
  Settings,
  ShieldCheck,
  Sparkles,
  Terminal,
  ExternalLink,
  XCircle,
} from "lucide-react";
import "./styles.css";

const initialTimeline = [
  { title: "Voice command received", detail: "Waiting for a trusted request", status: "idle", icon: Mic },
  { title: "Transcription and intent", detail: "Whisper Large v3", status: "idle", icon: Sparkles },
  { title: "Trust policy evaluation", detail: "Layer A → Jev → semantic judge", status: "idle", icon: ShieldCheck },
  { title: "Agent execution", detail: "No active tool call", status: "idle", icon: Terminal },
];
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "";

function MetricCard({ label, value, note, tone = "blue", icon: Icon }) {
  return (
    <article className={`metric-card ${tone}`}>
      <div className="metric-header"><span>{label}</span><Icon size={17} /></div>
      <strong>{value}</strong>
      <small>{note}</small>
    </article>
  );
}

function StatusBadge({ status }) {
  const labels = { idle: "Ready", running: "Running", completed: "Completed", review: "Review", failed: "Blocked" };
  return <span className={`status-badge ${status}`}>{labels[status] || status}</span>;
}

function YouTubePlayer({ videoId, title, onStatus }) {
  const containerRef = React.useRef(null);

  React.useEffect(() => {
    let player;
    let cancelled = false;
    const createPlayer = () => {
      if (cancelled || !window.YT || !containerRef.current) return;
      player = new window.YT.Player(containerRef.current, {
        videoId,
        playerVars: { autoplay: 0, modestbranding: 1, rel: 0 },
        events: {
          onStateChange: (event) => {
            const states = {
              [window.YT.PlayerState.PLAYING]: "PLAYING",
              [window.YT.PlayerState.PAUSED]: "PAUSED",
              [window.YT.PlayerState.ENDED]: "ENDED",
            };
            if (states[event.data]) onStatus(states[event.data]);
          },
          onError: (event) => onStatus("ERROR", event.data),
        },
      });
    };
    if (window.YT) {
      createPlayer();
    } else {
      const existing = document.querySelector('script[src="https://www.youtube.com/iframe_api"]');
      if (!existing) {
        const script = document.createElement("script");
        script.src = "https://www.youtube.com/iframe_api";
        document.body.appendChild(script);
      }
      const previousReady = window.onYouTubeIframeAPIReady;
      window.onYouTubeIframeAPIReady = () => {
        previousReady?.();
        createPlayer();
      };
    }
    return () => {
      cancelled = true;
      player?.destroy();
    };
  }, [videoId, onStatus]);

  return <div ref={containerRef} aria-label={`YouTube player: ${title}`} />;
}

function App() {
  const [question, setQuestion] = React.useState("");
  const [source, setSource] = React.useState("Both sources");
  const [timeline, setTimeline] = React.useState(initialTimeline);
  const [run, setRun] = React.useState(null);
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState("");
  const [recording, setRecording] = React.useState(false);
  const [activeView, setActiveView] = React.useState("overview");
  const [overview, setOverview] = React.useState(null);
  const [apiConnected, setApiConnected] = React.useState(false);
  const [mediaStatus, setMediaStatus] = React.useState("IDLE");
  const [trustMetrics, setTrustMetrics] = React.useState(null);
  const [gmailStatus, setGmailStatus] = React.useState(null);
  const recorderRef = React.useRef(null);
  const chunksRef = React.useRef([]);
  const stopTimerRef = React.useRef(null);

  const refreshOverview = React.useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/overview`);
      if (!response.ok) throw new Error("Overview request failed.");
      setOverview(await response.json());
      setApiConnected(true);
    } catch {
      setApiConnected(false);
    }
  }, []);

  const refreshTrustMetrics = React.useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/metrics/trust`);
      if (response.ok) setTrustMetrics(await response.json());
    } catch {
      setTrustMetrics(null);
    }
  }, []);

  const refreshGmailStatus = React.useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/gmail/status`);
      if (response.ok) setGmailStatus(await response.json());
    } catch {
      setGmailStatus(null);
    }
  }, []);

  React.useEffect(() => {
    refreshOverview();
    refreshTrustMetrics();
    refreshGmailStatus();
  }, [refreshOverview, refreshTrustMetrics, refreshGmailStatus]);

  async function toggleRecording() {
    if (recording) {
      recorderRef.current?.stop();
      setRecording(false);
      return;
    }
    if (!navigator.mediaDevices?.getUserMedia) {
      setError("Microphone recording is not supported by this browser.");
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      chunksRef.current = [];
      recorder.ondataavailable = (event) => {
        if (event.data.size) chunksRef.current.push(event.data);
      };
      recorder.onstop = async () => {
        if (stopTimerRef.current) window.clearTimeout(stopTimerRef.current);
        stream.getTracks().forEach((track) => track.stop());
        const formData = new FormData();
        formData.append("audio", new Blob(chunksRef.current, { type: recorder.mimeType }), "voice.webm");
        try {
              const response = await fetch(`${API_BASE_URL}/api/transcribe`, { method: "POST", body: formData });
          const payload = await response.json();
          if (!response.ok) throw new Error(payload.detail || "Transcription failed.");
          setQuestion(payload.text);
          setError("");
          await submitQuestion(payload.text);
        } catch (requestError) {
          setError(requestError.message);
        }
      };
      recorderRef.current = recorder;
      recorder.start();
      setRecording(true);
      stopTimerRef.current = window.setTimeout(() => {
        if (recorder.state === "recording") recorder.stop();
      }, 5000);
    } catch (requestError) {
      setError(requestError.message || "Microphone permission was denied.");
    }
  }

  async function submitQuestion(command) {
    if (!command.trim() || loading) return;
    setError("");
    setLoading(true);
    setTimeline(initialTimeline.map((item, index) => index === 0 ? { ...item, status: "completed" } : index === 1 ? { ...item, status: "running" } : item));
    try {
      const response = await fetch(`${API_BASE_URL}/api/runs`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: command, source }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "The agent could not start.");
      if (payload.browser_action?.url) {
        window.open(payload.browser_action.url, "_blank", "noopener,noreferrer");
      }
      setRun(payload);
      refreshOverview();
      refreshTrustMetrics();
      setMediaStatus(payload.media_action ? "READY" : "IDLE");
      setTimeline(initialTimeline.map((item, index) => ({
        ...item,
        status: payload.status === "review" && index === 2 ? "review" : index < 3 ? "completed" : payload.status === "completed" ? "completed" : "running",
      })));
    } catch (requestError) {
      setError(requestError.message);
      setTimeline(initialTimeline.map((item, index) => index < 2 ? { ...item, status: "completed" } : index === 2 ? { ...item, status: "failed" } : item));
    } finally {
      setLoading(false);
    }
  }

  async function submit(event) {
    event.preventDefault();
    await submitQuestion(question);
  }

  async function decide(approved) {
    if (!run?.run_id) return;
    const response = await fetch(`${API_BASE_URL}/api/runs/${run.run_id}/approval`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ approved }),
    });
    const payload = await response.json();
    if (!response.ok) {
      setError(payload.detail || "Approval failed.");
      return;
    }
    setRun(payload);
    refreshOverview();
    refreshTrustMetrics();
  }

  const reportMediaStatus = React.useCallback(async (status, errorCode = null) => {
    if (!run?.run_id) return;
    setMediaStatus(status);
    await fetch(`${API_BASE_URL}/api/media/status`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ run_id: run.run_id, status, error_code: errorCode }),
    });
    refreshOverview();
    refreshTrustMetrics();
  }, [run?.run_id, refreshOverview, refreshTrustMetrics]);

  const report = run?.trust_report || {};
  const checks = report.checks || [];
  const records = report.primary_trace?.records || [];
  const stepExplanations = records.filter((record) => record.kind !== "model");
  const postureScore = overview?.posture_score ?? "--";
  const resolutionRate = overview?.resolution_rate == null
    ? "--"
    : `${overview.resolution_rate}%`;
  const mediaAction = run?.media_action;
  const integrity = trustMetrics?.trust_score ?? postureScore;
  const statusState = run?.pending_approval
    ? "ATTENTION NEEDED"
    : run?.status === "failed"
      ? "BLOCKED"
      : "VERIFIED";

  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand"><div className="brand-mark"><ShieldCheck size={19} /></div><div><b>Agent Trust</b><span>TRACER CONSOLE</span></div></div>
        <nav>
          <a className={activeView === "overview" ? "active" : ""} onClick={() => setActiveView("overview")}><LayoutDashboard size={18} />Overview</a>
          <a className={activeView === "execution" ? "active" : ""} onClick={() => setActiveView("execution")}><Activity size={18} />Live execution</a>
          <a className={activeView === "forensics" ? "active" : ""} onClick={() => setActiveView("forensics")}><FileSearch size={18} />Forensics</a>
          <a className={activeView === "analytics" ? "active" : ""} onClick={() => setActiveView("analytics")}><Gauge size={18} />Trust analytics</a>
          <a className={activeView === "settings" ? "active" : ""} onClick={() => setActiveView("settings")}><Settings size={18} />Settings</a>
        </nav>
        <div className="sidebar-bottom"><a><CircleHelp size={18} />Documentation</a><div className="system-state"><i />All systems operational</div></div>
      </aside>

      <main className="main">
        <header className="topbar">
          <div className="mobile-menu"><Menu size={20} /></div>
          <div className="searchbox"><Search size={16} /><span>Search runs, agents, policies...</span><kbd>⌘ K</kbd></div>
          <div className="top-actions"><span className={`connected ${apiConnected ? "" : "disconnected"}`}><i /> {apiConnected ? "API connected" : "API disconnected"}</span><Github size={17} /><Bell size={17} /><Moon size={17} /><div className="avatar">H</div></div>
        </header>

        <section className="content">
          <section className={`status-banner ${statusState.toLowerCase().replaceAll(" ", "-")}`}>
            <div className="status-icon"><CheckCircle2 size={26} /></div>
            <div><p className="eyebrow">SYSTEM STATUS</p><h2>STATUS: <strong>{statusState}</strong></h2><p>System integrity confirmed. Agent actions are traced and policy validated.</p></div>
            <span className="updated-at">Last updated: {new Date().toLocaleString()} <ChevronRight size={14} /></span>
          </section>
          <div className="page-heading"><div><p className="eyebrow">TRUST OPERATIONS / {activeView.toUpperCase()}</p><h1>{activeView === "overview" ? "Agent command center" : activeView === "execution" ? "Live execution" : activeView === "forensics" ? "Run forensics" : activeView === "analytics" ? "Trust analytics" : "Console settings"}</h1><p className="muted">Observe voice requests, safety decisions, and tool execution in one place.</p></div><button className="outline-button" onClick={() => setActiveView("execution")}><Play size={14} /> Replay last run</button></div>

          <div className="metrics-grid">
            <MetricCard label="ACTIVE AGENTS" value={overview?.active_agents ?? "--"} note="Live API status" tone="blue" icon={Activity} />
            <MetricCard label="TRUST POSTURE" value={integrity} note="Calculated from recorded decisions" tone="green" icon={ShieldCheck} />
            <MetricCard label="OPEN REVIEWS" value={run?.status === "review" ? "01" : "00"} note="Current run approval state" tone="amber" icon={AlertTriangle} />
            <MetricCard label="RESOLUTION RATE" value={resolutionRate} note="Calculated from current session runs" tone="violet" icon={CheckCircle2} />
          </div>

          {activeView === "forensics" && <section className="panel detail-panel"><div className="panel-heading"><div><p className="eyebrow">FORENSIC TRACE</p><h2>Step-by-step explanation</h2></div><FileSearch size={18} /></div><div className="explanation-list">{stepExplanations.length ? stepExplanations.map((record, index) => <article className="explanation-item" key={`${record.name}-${index}`}><div className="explanation-index">{index + 1}</div><div><b>{record.name}</b><p>Status: {record.status}</p><small>{record.detail?.reason || record.detail?.rule || record.detail?.decision || "The agent recorded this execution step."}</small></div></article>) : <p className="muted">Run a command to see each agent and trust step explained here.</p>}</div></section>}
          {activeView === "analytics" && <section className="panel detail-panel"><div className="panel-heading"><div><p className="eyebrow">TRUST ANALYTICS</p><h2>Decision breakdown</h2></div><Gauge size={18} /></div><div className="analytics-grid">{checks.length ? checks.map((check) => <div className="analytics-card" key={check.check_id || check.name}><span>{check.name}</span><b className={check.status}>{check.status.toUpperCase()}</b><p>{check.explanation || check.reason || "Trust check completed."}</p></div>) : <p className="muted">Trust results appear after the first command.</p>}</div></section>}
          {activeView === "settings" && <section className="panel detail-panel"><div className="panel-heading"><div><p className="eyebrow">CONSOLE SETTINGS</p><h2>Active execution policy</h2></div><Settings size={18} /></div><div className="settings-list"><div><span>Gmail connection</span><b className={gmailStatus?.connected ? "green-text" : ""}>{gmailStatus?.connected ? "Connected" : "Reconnect required"}</b></div><div><span>Gmail scopes</span><b>{gmailStatus?.scopes?.length ? gmailStatus.scopes.map((scope) => scope.split(".").pop()).join(", ") : "Not granted"}</b></div><div><span>Speech model</span><b>openai/whisper-large-v3</b></div><div><span>Voice auto-run</span><b className="green-text">Enabled</b></div><div><span>Playback policy</span><b>Explicit play/watch/listen only</b></div><div><span>Approval policy</span><b>Required for Gmail writes</b></div></div></section>}
          {mediaAction && <section className="panel media-panel"><div className="panel-heading"><div><p className="eyebrow">MEDIA PLAYER</p><h2>{mediaAction.title}</h2></div><span className={`status-badge ${mediaStatus === "PLAYING" ? "completed" : "running"}`}>{mediaStatus}</span></div><div className="player-frame"><YouTubePlayer videoId={mediaAction.video_id} title={mediaAction.title} onStatus={reportMediaStatus} /></div><div className="player-actions"><span><Play size={14} /> Use the player controls to start playback.</span><a href={mediaAction.url} target="_blank" rel="noreferrer"><ExternalLink size={14} /> Open on YouTube</a><span><Info size={14} /> {mediaStatus === "PLAYING" ? "Playback verified by YouTube." : "Click play if the browser blocks autoplay."}</span></div></section>}
          <div className="workspace-grid">
            <section className="panel execution-panel">
              <div className="panel-heading"><div><p className="eyebrow">LIVE EXECUTION</p><h2>Voice agent trace</h2></div><StatusBadge status={run ? run.status : "idle"} /></div>
              <form className="command-form" onSubmit={submit}>
                <div className="command-input"><button type="button" className={`mic-button ${recording ? "recording" : ""}`} onClick={toggleRecording} aria-label={recording ? "Stop recording" : "Start recording"}><Mic size={18} /></button><input value={question} onChange={(event) => setQuestion(event.target.value)} placeholder={recording ? "Listening..." : "Speak or type a trusted command..."} /><select value={source} onChange={(event) => setSource(event.target.value)}><option>Both sources</option><option>Gmail</option><option>YouTube</option></select><button type="submit" disabled={loading}><Sparkles size={16} />{loading ? "Running" : "Run agent"}</button></div>
                <div className="form-hint">Voice commands use Whisper Large v3. Every tool call passes Layer A, Jev, and trust fusion before execution.</div>
              </form>
              {error && <div className="alert error"><XCircle size={16} />{error}</div>}
              {run?.pending_approval && <div className="approval-card"><div className="approval-header"><AlertTriangle size={16} /><div><b>Email preview — approval required</b><span>{run.pending_approval.decision?.reasons?.join(" ") || `${run.pending_approval.tool} requires a policy decision.`}</span></div></div><div className="email-preview"><div><b>To</b><span>{run.pending_approval.arguments?.to || "Not specified"}</span></div><div><b>Subject</b><span>{run.pending_approval.arguments?.subject || "Not specified"}</span></div><div><b>Body</b><p>{run.pending_approval.arguments?.body || "No body supplied."}</p></div></div><div className="approval-actions"><button onClick={() => decide(true)}>Approve &amp; send</button><button className="danger-button" onClick={() => decide(false)}>Reject</button></div></div>}
              <div className="timeline">
                {timeline.map((item, index) => { const Icon = item.icon; return <div className={`timeline-row ${item.status}`} key={item.title}><div className="timeline-marker"><Icon size={15} /></div><div className="timeline-line" /><div className="timeline-body"><div className="timeline-title"><span>{index + 1}. {item.title}</span><StatusBadge status={item.status} /></div><p>{item.detail}</p>{index === 2 && <div className="trace-chip"><ShieldCheck size={13} /> {checks.length ? `${checks.length} trust checks recorded` : "Deterministic policy gate ready"} <ChevronRight size={13} /></div>}</div></div>; })}
              </div>
              {run?.answer && <div className="result-card"><p className="eyebrow">AGENT RESPONSE</p><p>{run.answer}</p></div>}
              {run?.trust_report?.hybrid_decisions?.length > 0 && <div className="decision-proof"><div className="panel-heading"><div><p className="eyebrow">DECISION PROOF</p><h3>Why did the agent do this?</h3></div><ShieldCheck size={16} /></div>{run.trust_report.hybrid_decisions.slice(0, 4).map((decision, index) => <div className="proof-row" key={`${decision.tool || decision.rule}-${index}`}><span className={`proof-dot ${String(decision.decision || "").toLowerCase()}`} /><div><b>{decision.tool || decision.rule || "Policy evaluation"}</b><p>{decision.reason || "The action passed the active policy checks."}</p><small>Decision: {decision.decision || "PASS"} · Confidence: {decision.confidence ?? "--"}</small></div></div>)}</div>}
            </section>

            <aside className="right-column">
              <section className="panel risk-panel"><div className="panel-heading"><div><p className="eyebrow">RISK OVERVIEW</p><h2>Trust posture</h2></div><ShieldCheck size={18} className="green-icon" /></div><div className="score-wrap"><div className="score-ring"><strong>{postureScore}</strong><span>/100</span></div><div><b>{postureScore === "--" ? "Awaiting run data" : postureScore >= 85 ? "Highly trusted" : "Review recommended"}</b><p>Calculated from trust checks</p></div></div><div className="risk-bars"><div><span>Recorded checks</span><b>{checks.length}</b><i><em style={{ width: `${checks.length ? 100 : 0}%` }} /></i></div><div><span>Critical alerts</span><b>{overview?.new_critical_alerts ?? "--"}</b><i><em style={{ width: `${overview?.new_critical_alerts ? 100 : 0}%` }} /></i></div><div><span>Resolution rate</span><b>{resolutionRate}</b><i><em style={{ width: `${overview?.resolution_rate ?? 0}%` }} /></i></div></div></section>
              <section className="panel logs-panel"><div className="panel-heading"><div><p className="eyebrow">LIVE OUTPUT</p><h2>Execution logs</h2></div><span className="live-dot">LIVE</span></div><div className="logs">{records.length ? records.slice(-6).map((record, index) => <div className="log-row" key={`${record.name}-${index}`}><time>{record.status}</time><span className={`log-dot ${record.status === "success" ? "green" : "blue"}`} /><div><b>{record.name}</b><p>{record.detail?.decision || record.detail?.rule || "Trace event recorded"}</p></div></div>) : <div className="empty-logs"><Terminal size={20} /><span>Execution events will appear here.</span></div>}</div></section>
            </aside>
          </div>
        </section>
      </main>
    </div>
  );
}

createRoot(document.getElementById("root")).render(<App />);
