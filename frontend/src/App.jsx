import { useEffect, useMemo, useState } from "react";
import { buildPredictionRecords, demoSeries } from "./data";
import { requestJson } from "./api";

const defaultApiUrl = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

const translations = {
  en: {
    eyebrow: "Italian Grid Intelligence System",
    title: "Detect electrical grid anomalies before faults happen.",
    desc: "A production-style anomaly detection platform for TERNA/GSE time-series data with hybrid LSTM + Isolation Forest modeling, fast inference, drift monitoring, and retraining workflows.",
    btn_predict: "Predict demo",
    btn_retrain: "Retrain now",
    card_backend: "Backend",
    card_online: "Online",
    card_demo: "Demo mode",
    card_seq: "Sequence length",
    card_thresh: "Threshold",
    card_features: "Features",
    card_anomalies: "Anomalies",
    stat_avg_score: "Avg anomaly score",
    stat_avg_hint: "Scores from the latest prediction run",
    stat_windows: "Demo windows",
    stat_windows_hint: "TERNA-style 2-hour sample series",
    stat_monitoring: "Monitoring",
    stat_monitoring_hint: "Drift reports + retraining feedback loop",
    stat_target: "Deploy target",
    stat_target_hint: "Static React UI + hosted API backend",
    panel_trend: "Score trend",
    panel_heatmap: "Anomaly heatmap sparkline",
    panel_connection: "Connection",
    panel_conn_desc: "Point the frontend at your backend",
    panel_api_url: "API URL",
    panel_save: "Save",
    panel_refresh: "Refresh",
    panel_status: "Status",
    tab_overview: "overview",
    tab_predict: "predict",
    tab_retrain: "retrain",
    tab_results: "results",
    over_title: "What the system does",
    over_pred: "Predict",
    over_pred_desc: "Scores TERNA-style grid telemetry for anomalies before faults appear.",
    over_retrain: "Retrain",
    over_retrain_desc: "Runs the hybrid training workflow and stores updated model artifacts.",
    over_monitor: "Monitor",
    over_monitor_desc: "Uses drift detection to keep model behavior aligned with live data.",
    over_present: "Present",
    over_present_desc: "Clean, interview-ready UI that looks like a real operations tool.",
    pred_title: "Prediction payload",
    pred_desc: "Click predict to send a demo TERNA payload to your FastAPI backend.",
    retrain_title: "Retraining workflow",
    retrain_desc: "Trigger a retrain from the UI. The backend reuses the latest raw TERNA data, trains, saves artifacts, and updates model info.",
    retrain_btn: "Run retrain",
    retrain_reload: "Reload model info",
    results_title: "Latest predictions",
    results_anomaly: "Anomaly flagged",
    results_normal: "Normal operating window"
  },
  it: {
    eyebrow: "Sistema di Intelligenza della Rete Italiana",
    title: "Rileva anomalie nella rete elettrica prima che si verifichino guasti.",
    desc: "Una piattaforma di rilevamento anomalie di livello industriale per dati di serie temporali TERNA/GSE con modellazione ibrida LSTM + Isolation Forest, inferenza rapida, monitoraggio del drift e workflow di riaddestramento.",
    btn_predict: "Prevedi demo",
    btn_retrain: "Riaddestra ora",
    card_backend: "Backend",
    card_online: "Online",
    card_demo: "Modalità demo",
    card_seq: "Lunghezza sequenza",
    card_thresh: "Soglia",
    card_features: "Caratteristiche",
    card_anomalies: "Anomalie",
    stat_avg_score: "Punteggio medio anomalie",
    stat_avg_hint: "Punteggi dell'ultimo ciclo di previsione",
    stat_windows: "Finestre demo",
    stat_windows_hint: "Serie di campioni di 2 ore stile TERNA",
    stat_monitoring: "Monitoraggio",
    stat_monitoring_hint: "Report di drift + loop di feedback di riaddestramento",
    stat_target: "Target di deployment",
    stat_target_hint: "UI React statica + API backend ospitata",
    panel_trend: "Andamento del punteggio",
    panel_heatmap: "Sparkline mappa di calore anomalie",
    panel_connection: "Connessione",
    panel_conn_desc: "Indirizza il frontend al tuo backend",
    panel_api_url: "URL dell'API",
    panel_save: "Salva",
    panel_refresh: "Aggiorna",
    panel_status: "Stato",
    tab_overview: "panoramica",
    tab_predict: "previsione",
    tab_retrain: "riaddestramento",
    tab_results: "risultati",
    over_title: "Cosa fa il sistema",
    over_pred: "Prevedi",
    over_pred_desc: "Valuta la telemetria della rete in stile TERNA per anomalie prima che si presentino guasti.",
    over_retrain: "Riaddestra",
    over_retrain_desc: "Esegue il workflow di addestramento ibrido e memorizza gli artefatti del modello aggiornati.",
    over_monitor: "Monitora",
    over_monitor_desc: "Utilizza il rilevamento del drift per mantenere il comportamento del modello allineato con i dati live.",
    over_present: "Presenta",
    over_present_desc: "Interfaccia utente pulita e pronta per colloqui che sembra un vero strumento operativo.",
    pred_title: "Payload di previsione",
    pred_desc: "Clicca su prevedi per inviare un payload demo in stile TERNA al tuo backend FastAPI.",
    retrain_title: "Workflow di riaddestramento",
    retrain_desc: "Attiva un riaddestramento dall'interfaccia utente. Il backend riutilizza gli ultimi dati grezzi TERNA, addestra, salva gli artefatti e aggiorna le info del modello.",
    retrain_btn: "Esegui riaddestramento",
    retrain_reload: "Ricarica info modello",
    results_title: "Ultime previsioni",
    results_anomaly: "Anomalia segnalata",
    results_normal: "Finestra operativa normale"
  }
};

function translateMessage(msg, lang) {
  if (!msg) return "";
  if (msg.includes("Connect the backend")) {
    return lang === "en" ? "Connect the backend to unlock live predictions." : "Connetti il backend per sbloccare le previsioni live.";
  }
  if (msg.includes("Backend connected")) {
    return lang === "en" ? "Backend connected and model is ready." : "Backend connesso e modello pronto.";
  }
  if (msg.includes("Demo mode active")) {
    return lang === "en" ? "Demo mode active. Point the UI to your backend to enable live actions." : "Modalità demo attiva. Indirizza l'UI al tuo backend per abilitare le azioni live.";
  }
  if (msg.includes("Prediction complete")) {
    const match = msg.match(/\d+/);
    const count = match ? match[0] : 0;
    return lang === "en" ? `Prediction complete: ${count} scored points.` : `Previsione completata: ${count} punti valutati.`;
  }
  if (msg.includes("Retrain complete")) {
    const dir = msg.split("Saved to ")[1] || "";
    return lang === "en" ? `Retrain complete. Saved to ${dir}.` : `Riaddestramento completato. Salvato in ${dir}.`;
  }
  return msg;
}

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
  const [lang, setLang] = useState(localStorage.getItem("lang") || "en");

  const t = (key) => translations[lang][key] || key;

  const handleSetLang = (newLang) => {
    setLang(newLang);
    localStorage.setItem("lang", newLang);
  };

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
      <div className="lang-selector">
        <button className="lang-btn" onClick={() => handleSetLang(lang === "en" ? "it" : "en")}>
          <svg className="globe-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="12" cy="12" r="10" />
            <line x1="2" y1="12" x2="22" y2="12" />
            <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
          </svg>
          <span>{lang === "en" ? "IT" : "EN"}</span>
        </button>
      </div>

      <header className="hero">
        <div className="hero-copy card">
          <div className="eyebrow"><span className="pulse-dot"></span> {t("eyebrow")}</div>
          <h1>{t("title")}</h1>
          <p>{t("desc")}</p>
          <div className="hero-actions">
            <button className="btn primary" onClick={handlePredict} disabled={isWorking}>
              {t("btn_predict")}
            </button>
            <button className="btn secondary" onClick={handleRetrain} disabled={isWorking}>
              {t("btn_retrain")}
            </button>
          </div>
          <div className="hero-message">{translateMessage(message, lang)}</div>
        </div>
        <div className="hero-panel card">
          <div className={`status-dot ${health === "ok" ? "green" : "amber"}`}></div>
          <div>
            <div className="muted">{t("card_backend")}</div>
            <strong>{health === "ok" ? t("card_online") : t("card_demo")}</strong>
          </div>
          <div className="hero-panel-grid">
            <div>
              <span className="muted">{t("card_seq")}</span>
              <strong>{seqLength}</strong>
            </div>
            <div>
              <span className="muted">{t("card_thresh")}</span>
              <strong>{threshold}</strong>
            </div>
            <div>
              <span className="muted">{t("card_features")}</span>
              <strong>{featureCount}</strong>
            </div>
            <div>
              <span className="muted">{t("card_anomalies")}</span>
              <strong>{anomalyCount}</strong>
            </div>
          </div>
        </div>
      </header>

      <section className="stats-grid">
        <StatCard label={t("stat_avg_score")} value={avgScore} hint={t("stat_avg_hint")} />
        <StatCard label={t("stat_windows")} value={demoSeries.length} hint={t("stat_windows_hint")} />
        <StatCard label={t("stat_monitoring")} value="Evidently" hint={t("stat_monitoring_hint")} />
        <StatCard label={t("stat_target")} value="Vercel" hint={t("stat_target_hint")} />
      </section>

      <section className="main-grid">
        <div className="card panel">
          <div className="panel-head">
            <div>
              <div className="section-kicker">{t("panel_trend")}</div>
              <h2>{t("panel_heatmap")}</h2>
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
          <div className="section-kicker">{t("panel_connection")}</div>
          <h2>{t("panel_conn_desc")}</h2>
          <div className="field">
            <label>{t("panel_api_url")}</label>
            <input value={apiUrl} onChange={(e) => setApiUrl(e.target.value)} placeholder="http://127.0.0.1:8000" />
          </div>
          <div className="button-row">
            <button className="btn primary" onClick={saveApiUrl}>{t("panel_save")}</button>
            <button className="btn secondary" onClick={refreshModel}>{t("panel_refresh")}</button>
          </div>
          <div className="log-box">
            <div className="section-kicker">{t("panel_status")}</div>
            <p>{translateMessage(message, lang)}</p>
            {requestError ? <p className="error">{requestError}</p> : null}
          </div>
        </aside>
      </section>

      <section className="tabs card">
        {["overview", "predict", "retrain", "results"].map((name) => (
          <button key={name} className={`tab ${tab === name ? "active" : ""}`} onClick={() => setTab(name)}>
            {t(`tab_${name}`)}
          </button>
        ))}
      </section>

      {tab === "overview" ? (
        <section className="card content-card">
          <h2>{t("over_title")}</h2>
          <div className="feature-grid">
            <div>
              <strong>{t("over_pred")}</strong>
              <p>{t("over_pred_desc")}</p>
            </div>
            <div>
              <strong>{t("over_retrain")}</strong>
              <p>{t("over_retrain_desc")}</p>
            </div>
            <div>
              <strong>{t("over_monitor")}</strong>
              <p>{t("over_monitor_desc")}</p>
            </div>
            <div>
              <strong>{t("over_present")}</strong>
              <p>{t("over_present_desc")}</p>
            </div>
          </div>
        </section>
      ) : null}

      {tab === "predict" ? (
        <section className="card content-card">
          <h2>{t("pred_title")}</h2>
          <p>{t("pred_desc")}</p>
          <pre>{JSON.stringify(buildPredictionRecords(), null, 2)}</pre>
        </section>
      ) : null}

      {tab === "retrain" ? (
        <section className="card content-card">
          <h2>{t("retrain_title")}</h2>
          <p>{t("retrain_desc")}</p>
          <div className="button-row">
            <button className="btn primary" onClick={handleRetrain} disabled={isWorking}>{t("retrain_btn")}</button>
            <button className="btn secondary" onClick={refreshModel}>{t("retrain_reload")}</button>
          </div>
        </section>
      ) : null}

      {tab === "results" ? (
        <section className="card content-card">
          <h2>{t("results_title")}</h2>
          <div className="results-grid">
            {(predictions.length ? predictions : buildPredictionRecords()).slice(0, 8).map((row, index) => (
              <div className="result-row" key={index}>
                <div>
                  <strong>{row.timestamp ? row.timestamp.slice(11, 16) : row.label}</strong>
                  <p>{row.is_anomaly ? t("results_anomaly") : t("results_normal")}</p>
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
