import { useEffect, useMemo, useState } from "react";
import { buildPredictionRecords, demoSeries } from "./data";
import { requestJson } from "./api";

const defaultApiUrl = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

function StatCard({ label, value, hint }) {
  return (
    <div className="card stat-card">
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value}</div>
      <div className="stat-hint">{hint}</div>
    </div>
  );
}

function Sparkline({ data, stroke = "#06b6d4" }) {
  const points = useMemo(() => {
    const values = data.map((d) => d.score ?? d.anomaly_score ?? 0);
    const min = Math.min(...values);
    const max = Math.max(...values);
    const width = 100;
    const height = 28;
    return values
      .map((value, index) => {
        const x = (index / Math.max(values.length - 1, 1)) * width;
        const y = height - ((value - min) / Math.max(max - min, 0.001)) * height;
        return `${x},${y}`;
      })
      .join(" ");
  }, [data]);

  return (
    <svg viewBox="0 0 100 28" preserveAspectRatio="none" className="sparkline">
      <defs>
        <filter id="glow-filter" x="-20%" y="-20%" width="140%" height="140%">
          <feGaussianBlur stdDeviation="1.5" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>
      <polyline points={points} fill="none" stroke={stroke} strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" filter="url(#glow-filter)" />
    </svg>
  );
}

export default function App() {
  const [apiUrl, setApiUrl] = useState(localStorage.getItem("apiUrl") || defaultApiUrl);
  const [health, setHealth] = useState("checking");
  const [modelInfo, setModelInfo] = useState(null);
  const [predictions, setPredictions] = useState([]);
  const [message, setMessage] = useState("Connect the backend to unlock live predictions.");
  const [isWorking, setIsWorking] = useState(false);
  const [tab, setTab] = useState("overview");
  const [requestError, setRequestError] = useState("");

  const anomalyCount = predictions.filter((p) => p.is_anomaly).length;
  const avgScore = predictions.length
    ? (predictions.reduce((sum, row) => sum + Number(row.anomaly_score || 0), 0) / predictions.length).toFixed(4)
    : "0.0000";

  async function refreshModel() {
    setRequestError("");
    try {
      const [healthPayload, infoPayload] = await Promise.all([
        requestJson(apiUrl, "/health"),
        requestJson(apiUrl, "/model/info"),
      ]);
      setHealth(healthPayload.status || "ok");
      setModelInfo(infoPayload);
      setMessage("Backend connected and model is ready.");
    } catch (error) {
      setHealth("offline");
      setModelInfo(null);
      setMessage("Demo mode active. Point the UI to your backend to enable live actions.");
      setRequestError(error.message);
    }
  }

  useEffect(() => {
    refreshModel();
  }, [apiUrl]);

  async function handlePredict() {
    setIsWorking(true);
    setRequestError("");
    try {
      const payload = { records: buildPredictionRecords() };
      const result = await requestJson(apiUrl, "/predict", {
        method: "POST",
        body: JSON.stringify(payload),
      });
      setPredictions(result.predictions || []);
      setTab("results");
      setMessage(`Prediction complete: ${result.predictions?.length || 0} scored points.`);
    } catch (error) {
      setRequestError(error.message);
    } finally {
      setIsWorking(false);
    }
  }

  async function handleRetrain() {
    setIsWorking(true);
    setRequestError("");
    try {
      const result = await requestJson(apiUrl, "/retrain", { method: "POST", body: "{}" });
      setMessage(`Retrain complete. Saved to ${result.model_dir}.`);
      await refreshModel();
    } catch (error) {
      setRequestError(error.message);
    } finally {
      setIsWorking(false);
    }
  }

  async function saveApiUrl() {
    localStorage.setItem("apiUrl", apiUrl);
    await refreshModel();
  }

  const threshold = modelInfo?.threshold?.toFixed ? modelInfo.threshold.toFixed(4) : modelInfo?.threshold ?? "-";
  const featureCount = modelInfo?.feature_count ?? "-";
  const seqLength = modelInfo?.seq_length ?? "-";

  return (
    <div className="page-shell">
      <header className="hero">
        <div className="hero-copy card">
          <div className="eyebrow"><span className="pulse-dot"></span> Italian Grid Intelligence System</div>
          <h1>Detect electrical grid anomalies before faults happen.</h1>
          <p>
            A production-style anomaly detection platform for TERNA/GSE time-series data with hybrid LSTM + Isolation Forest modeling,
            fast inference, drift monitoring, and retraining workflows.
          </p>
          <div className="hero-actions">
            <button className="btn primary" onClick={handlePredict} disabled={isWorking}>
              Predict demo
            </button>
            <button className="btn secondary" onClick={handleRetrain} disabled={isWorking}>
              Retrain now
            </button>
          </div>
          <div className="hero-message">{message}</div>
        </div>
        <div className="hero-panel card">
          <div className={`status-dot ${health === "ok" ? "green" : "amber"}`}></div>
          <div>
            <div className="muted">Backend</div>
            <strong>{health === "ok" ? "Online" : "Demo mode"}</strong>
          </div>
          <div className="hero-panel-grid">
            <div>
              <span className="muted">Sequence length</span>
              <strong>{seqLength}</strong>
            </div>
            <div>
              <span className="muted">Threshold</span>
              <strong>{threshold}</strong>
            </div>
            <div>
              <span className="muted">Features</span>
              <strong>{featureCount}</strong>
            </div>
            <div>
              <span className="muted">Anomalies</span>
              <strong>{anomalyCount}</strong>
            </div>
          </div>
        </div>
      </header>

      <section className="stats-grid">
        <StatCard label="Avg anomaly score" value={avgScore} hint="Scores from the latest prediction run" />
        <StatCard label="Demo windows" value={demoSeries.length} hint="TERNA-style 2-hour sample series" />
        <StatCard label="Monitoring" value="Evidently" hint="Drift reports + retraining feedback loop" />
        <StatCard label="Deploy target" value="Vercel" hint="Static React UI + hosted API backend" />
      </section>

      <section className="main-grid">
        <div className="card panel">
          <div className="panel-head">
            <div>
              <div className="section-kicker">Score trend</div>
              <h2>Anomaly heatmap sparkline</h2>
            </div>
            <Sparkline data={predictions.length ? predictions : demoSeries} />
          </div>
          <div className="trend-bars">
            {(predictions.length ? predictions : demoSeries).map((item, index) => (
              <div className="trend-row" key={`${item.timestamp || item.label}-${index}`}>
                <span>{item.timestamp ? item.timestamp.slice(11, 16) : item.label}</span>
                <div className="trend-track">
                  <div className="trend-fill" style={{ width: `${Math.min(100, Math.max(8, Number(item.anomaly_score || item.score || 0) * 220))}%` }} />
                </div>
                <strong>{Number(item.anomaly_score || item.score || 0).toFixed(3)}</strong>
              </div>
            ))}
          </div>
        </div>

        <aside className="card panel">
          <div className="section-kicker">Connection</div>
          <h2>Point the frontend at your backend</h2>
          <div className="field">
            <label>API URL</label>
            <input value={apiUrl} onChange={(e) => setApiUrl(e.target.value)} placeholder="http://127.0.0.1:8000" />
          </div>
          <div className="button-row">
            <button className="btn primary" onClick={saveApiUrl}>Save</button>
            <button className="btn secondary" onClick={refreshModel}>Refresh</button>
          </div>
          <div className="log-box">
            <div className="section-kicker">Status</div>
            <p>{message}</p>
            {requestError ? <p className="error">{requestError}</p> : null}
          </div>
        </aside>
      </section>

      <section className="tabs card">
        {["overview", "predict", "retrain", "results"].map((name) => (
          <button key={name} className={`tab ${tab === name ? "active" : ""}`} onClick={() => setTab(name)}>
            {name}
          </button>
        ))}
      </section>

      {tab === "overview" ? (
        <section className="card content-card">
          <h2>What the system does</h2>
          <div className="feature-grid">
            <div>
              <strong>Predict</strong>
              <p>Scores TERNA-style grid telemetry for anomalies before faults appear.</p>
            </div>
            <div>
              <strong>Retrain</strong>
              <p>Runs the hybrid training workflow and stores updated model artifacts.</p>
            </div>
            <div>
              <strong>Monitor</strong>
              <p>Uses drift detection to keep model behavior aligned with live data.</p>
            </div>
            <div>
              <strong>Present</strong>
              <p>Clean, interview-ready UI that looks like a real operations tool.</p>
            </div>
          </div>
        </section>
      ) : null}

      {tab === "predict" ? (
        <section className="card content-card">
          <h2>Prediction payload</h2>
          <p>Click predict to send a demo TERNA payload to your FastAPI backend.</p>
          <pre>{JSON.stringify(buildPredictionRecords(), null, 2)}</pre>
        </section>
      ) : null}

      {tab === "retrain" ? (
        <section className="card content-card">
          <h2>Retraining workflow</h2>
          <p>Trigger a retrain from the UI. The backend reuses the latest raw TERNA data, trains, saves artifacts, and updates model info.</p>
          <div className="button-row">
            <button className="btn primary" onClick={handleRetrain} disabled={isWorking}>Run retrain</button>
            <button className="btn secondary" onClick={refreshModel}>Reload model info</button>
          </div>
        </section>
      ) : null}

      {tab === "results" ? (
        <section className="card content-card">
          <h2>Latest predictions</h2>
          <div className="results-grid">
            {(predictions.length ? predictions : buildPredictionRecords()).slice(0, 8).map((row, index) => (
              <div className="result-row" key={index}>
                <div>
                  <strong>{row.timestamp ? row.timestamp.slice(11, 16) : row.label}</strong>
                  <p>{row.is_anomaly ? "Anomaly flagged" : "Normal operating window"}</p>
                </div>
                <span>{Number(row.anomaly_score || row.score || 0).toFixed(3)}</span>
              </div>
            ))}
          </div>
        </section>
      ) : null}
    </div>
  );
}
