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
  XCircle,
} from "lucide-react";
import "./styles.css";

const initialTimeline = [
  { title: "Voice command received", detail: "Waiting for a trusted request", status: "idle", icon: Mic },
  { title: "Transcription and intent", detail: "Whisper Large v3", status: "idle", icon: Sparkles },
  { title: "Trust policy evaluation", detail: "Layer A → Jev → semantic judge", status: "idle", icon: ShieldCheck },
  { title: "Agent execution", detail: "No active tool call", status: "idle", icon: Terminal },
];

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

function App() {
  const [question, setQuestion] = React.useState("");
  const [source, setSource] = React.useState("Both sources");
  const [timeline, setTimeline] = React.useState(initialTimeline);
  const [run, setRun] = React.useState(null);
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState("");
  const [recording, setRecording] = React.useState(false);
  const [activeView, setActiveView] = React.useState("overview");
  const recorderRef = React.useRef(null);
  const chunksRef = React.useRef([]);
  const stopTimerRef = React.useRef(null);

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
          const response = await fetch("/api/transcribe", { method: "POST", body: formData });
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
      const response = await fetch("/api/runs", {
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
    const response = await fetch(`/api/runs/${run.run_id}/approval`, {
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
  }

  const report = run?.trust_report || {};
  const checks = report.checks || [];
  const records = report.primary_trace?.records || [];
  const stepExplanations = records.filter((record) => record.kind !== "model");

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
          <div className="top-actions"><span className="connected"><i /> API connected</span><Github size={17} /><Bell size={17} /><Moon size={17} /><div className="avatar">H</div></div>
        </header>

        <section className="content">
          <div className="page-heading"><div><p className="eyebrow">TRUST OPERATIONS / {activeView.toUpperCase()}</p><h1>{activeView === "overview" ? "Agent command center" : activeView === "execution" ? "Live execution" : activeView === "forensics" ? "Run forensics" : activeView === "analytics" ? "Trust analytics" : "Console settings"}</h1><p className="muted">Observe voice requests, safety decisions, and tool execution in one place.</p></div><button className="outline-button" onClick={() => setActiveView("execution")}><Play size={14} /> Replay last run</button></div>

          <div className="metrics-grid">
            <MetricCard label="ACTIVE AGENTS" value="01" note="● Ready for input" tone="blue" icon={Activity} />
            <MetricCard label="TRUST POSTURE" value="95" note="↑ 4 points this week" tone="green" icon={ShieldCheck} />
            <MetricCard label="OPEN REVIEWS" value={run?.status === "review" ? "01" : "00"} note="Human decisions required" tone="amber" icon={AlertTriangle} />
            <MetricCard label="RESOLUTION RATE" value="100%" note="All recent runs resolved" tone="violet" icon={CheckCircle2} />
          </div>

          {activeView === "forensics" && <section className="panel detail-panel"><div className="panel-heading"><div><p className="eyebrow">FORENSIC TRACE</p><h2>Step-by-step explanation</h2></div><FileSearch size={18} /></div><div className="explanation-list">{stepExplanations.length ? stepExplanations.map((record, index) => <article className="explanation-item" key={`${record.name}-${index}`}><div className="explanation-index">{index + 1}</div><div><b>{record.name}</b><p>Status: {record.status}</p><small>{record.detail?.reason || record.detail?.rule || record.detail?.decision || "The agent recorded this execution step."}</small></div></article>) : <p className="muted">Run a command to see each agent and trust step explained here.</p>}</div></section>}
          {activeView === "analytics" && <section className="panel detail-panel"><div className="panel-heading"><div><p className="eyebrow">TRUST ANALYTICS</p><h2>Decision breakdown</h2></div><Gauge size={18} /></div><div className="analytics-grid">{checks.length ? checks.map((check) => <div className="analytics-card" key={check.check_id || check.name}><span>{check.name}</span><b className={check.status}>{check.status.toUpperCase()}</b><p>{check.explanation || check.reason || "Trust check completed."}</p></div>) : <p className="muted">Trust results appear after the first command.</p>}</div></section>}
          {activeView === "settings" && <section className="panel detail-panel"><div className="panel-heading"><div><p className="eyebrow">CONSOLE SETTINGS</p><h2>Active execution policy</h2></div><Settings size={18} /></div><div className="settings-list"><div><span>Speech model</span><b>openai/whisper-large-v3</b></div><div><span>Voice auto-run</span><b className="green-text">Enabled</b></div><div><span>Playback policy</span><b>Explicit play/watch/listen only</b></div><div><span>Approval policy</span><b>Required for email writes</b></div></div></section>}
          <div className="workspace-grid">
            <section className="panel execution-panel">
              <div className="panel-heading"><div><p className="eyebrow">LIVE EXECUTION</p><h2>Voice agent trace</h2></div><StatusBadge status={run ? run.status : "idle"} /></div>
              <form className="command-form" onSubmit={submit}>
                <div className="command-input"><button type="button" className={`mic-button ${recording ? "recording" : ""}`} onClick={toggleRecording} aria-label={recording ? "Stop recording" : "Start recording"}><Mic size={18} /></button><input value={question} onChange={(event) => setQuestion(event.target.value)} placeholder={recording ? "Listening..." : "Speak or type a trusted command..."} /><select value={source} onChange={(event) => setSource(event.target.value)}><option>Both sources</option><option>Gmail</option><option>YouTube</option></select><button type="submit" disabled={loading}><Sparkles size={16} />{loading ? "Running" : "Run agent"}</button></div>
                <div className="form-hint">Voice commands use Whisper Large v3. Every tool call passes Layer A, Jev, and trust fusion before execution.</div>
              </form>
              {error && <div className="alert error"><XCircle size={16} />{error}</div>}
              {run?.pending_approval && <div className="alert review"><AlertTriangle size={16} /><div><b>Human review required</b><span>{run.pending_approval.tool} is waiting for your decision.</span></div><button onClick={() => decide(true)}>Approve</button><button className="danger-button" onClick={() => decide(false)}>Reject</button></div>}
              <div className="timeline">
                {timeline.map((item, index) => { const Icon = item.icon; return <div className={`timeline-row ${item.status}`} key={item.title}><div className="timeline-marker"><Icon size={15} /></div><div className="timeline-line" /><div className="timeline-body"><div className="timeline-title"><span>{index + 1}. {item.title}</span><StatusBadge status={item.status} /></div><p>{item.detail}</p>{index === 2 && <div className="trace-chip"><ShieldCheck size={13} /> {checks.length ? `${checks.length} trust checks recorded` : "Deterministic policy gate ready"} <ChevronRight size={13} /></div>}</div></div>; })}
              </div>
              {run?.answer && <div className="result-card"><p className="eyebrow">AGENT RESPONSE</p><p>{run.answer}</p></div>}
            </section>

            <aside className="right-column">
              <section className="panel risk-panel"><div className="panel-heading"><div><p className="eyebrow">RISK OVERVIEW</p><h2>Trust posture</h2></div><ShieldCheck size={18} className="green-icon" /></div><div className="score-wrap"><div className="score-ring"><strong>95</strong><span>/100</span></div><div><b>Highly trusted</b><p>Current session posture</p></div></div><div className="risk-bars"><div><span>Intent alignment</span><b>98%</b><i><em style={{ width: "98%" }} /></i></div><div><span>Source boundaries</span><b>100%</b><i><em style={{ width: "100%" }} /></i></div><div><span>Execution safety</span><b>95%</b><i><em style={{ width: "95%" }} /></i></div></div></section>
              <section className="panel logs-panel"><div className="panel-heading"><div><p className="eyebrow">LIVE OUTPUT</p><h2>Execution logs</h2></div><span className="live-dot">LIVE</span></div><div className="logs">{records.length ? records.slice(-6).map((record, index) => <div className="log-row" key={`${record.name}-${index}`}><time>{record.status}</time><span className={`log-dot ${record.status === "success" ? "green" : "blue"}`} /><div><b>{record.name}</b><p>{record.detail?.decision || record.detail?.rule || "Trace event recorded"}</p></div></div>) : <div className="empty-logs"><Terminal size={20} /><span>Execution events will appear here.</span></div>}</div></section>
            </aside>
          </div>
        </section>
      </main>
    </div>
  );
}

createRoot(document.getElementById("root")).render(<App />);
