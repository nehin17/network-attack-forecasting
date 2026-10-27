import { useEffect, useState } from "react";
import {
  AreaChart,
  Area,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
} from "recharts";

import {
  attackHistory,
  attackDictionary,
} from "./data/mockData";

import "./index.css";


function formatTime(seconds) {
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;

  return `${String(mins).padStart(2, "0")}:${String(secs).padStart(
    2,
    "0"
  )}`;
}


function App() {
  const [activePage, setActivePage] = useState("monitor");
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [authMode, setAuthMode] = useState("login");
  const [monitoring, setMonitoring] = useState(true);
  const [defenseMode, setDefenseMode] = useState(false);
  const [darkMode, setDarkMode] = useState(true);
  const [sensitivity, setSensitivity] = useState("medium");
  const [selectedAttack, setSelectedAttack] = useState(null);
  const [selectedDictionaryAttack, setSelectedDictionaryAttack] =
    useState(null);

  const [currentForecast, setCurrentForecast] = useState(null);
  const [currentDetection, setCurrentDetection] = useState(null);
  const [trafficSnapshot, setTrafficSnapshot] = useState(null);
  const [trafficError, setTrafficError] = useState("");
  const [apiLoading, setApiLoading] = useState(true);
  const [apiError, setApiError] = useState("");

  useEffect(() => {
    let cancelled = false;
    fetch("/api/forecast/sample")
      .then(async (response) => {
        if (!response.ok) {
          const body = await response.json().catch(() => ({}));
          throw new Error(body.detail || `Forecast API HTTP ${response.status}`);
        }
        return response.json();
      })
      .then((data) => {
        if (cancelled) return;
        setCurrentForecast({
          attackType: data.prediction,
          category: "Binary next-minute forecast · CICIDS2017",
          severity: data.prediction === "ATTACK" ? "ELEVATED" : "LOW",
          confidence: Math.round(data.confidence * 100),
          riskScore: Math.round(data.attack_probability * 100),
          leadTimeSeconds: 60,
          evidence: [
            { label: "Attack probability", value: `${(data.attack_probability * 100).toFixed(1)}%`, score: Math.round(data.attack_probability * 100) },
            { label: "Decision threshold", value: `${(data.threshold * 100).toFixed(1)}%`, score: Math.round(data.threshold * 100) },
            { label: "History window", value: `${data.history_minutes} consecutive minutes`, score: 100 },
            { label: "Actual next-minute label", value: data.actual_next_minute, score: data.actual_next_minute === data.prediction ? 100 : 0 },
          ],
          interpretation: `The saved ${data.model} model predicted ${data.prediction} for the next one-minute bin. Sample source: ${data.capture}. Ground truth: ${data.actual_next_minute}.`,
          recommendations: data.prediction === "ATTACK"
            ? ["Review the flagged test window", "Inspect false-positive behaviour before operational use", "Validate on additional captures"]
            : ["Compare this forecast with the ground-truth label", "Review false negatives across the full test set", "Continue evaluation across additional captures"],
          timestamp: data.target_time,
          actual: data.actual_next_minute,
          capture: data.capture,
          model: data.model,
          probability: data.attack_probability,
        });
        setApiLoading(false);
      })
      .catch((error) => {
        if (!cancelled) {
          setApiError(error.message || "Could not reach backend");
          setApiLoading(false);
        }
      });
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
  let cancelled = false;

  fetch("/api/detection/sample")
    .then((response) => {
      if (!response.ok) throw new Error(`Detection API HTTP ${response.status}`);
      return response.json();
    })
    .then((data) => {
      if (!cancelled) setCurrentDetection(data);
    })
    .catch((error) => console.error("Detection API:", error));

  return () => {
    cancelled = true;
  };
}, []);

  useEffect(() => {
    let cancelled = false;
    fetch("/api/traffic/sample")
      .then(async (response) => {
        if (!response.ok) {
          const body = await response.json().catch(() => ({}));
          throw new Error(body.detail || `Traffic API HTTP ${response.status}`);
        }
        return response.json();
      })
      .then((data) => { if (!cancelled) setTrafficSnapshot(data); })
      .catch((error) => { if (!cancelled) setTrafficError(error.message || "Traffic data unavailable"); });
    return () => { cancelled = true; };
  }, []);

  const [leadTime, setLeadTime] = useState(60);




  const themeClass = darkMode ? "dark-theme" : "light-theme";


  return (
    <div className={`app ${themeClass} ${sidebarOpen ? "sidebar-is-open" : "sidebar-is-closed"}`}>

      {/* ================= HEADER ================= */}

      <header className="header">

        <div className="header-left">

          <button
            className="menu-toggle"
            onClick={() => setSidebarOpen(!sidebarOpen)}
            aria-label="Toggle navigation"
          >
            ☰
          </button>

          <div className="brand">

            <div className="brand-icon">
              ◉
            </div>

            <div>
              <h1>NetGuard</h1>

              <span>
                Network Attack Forecasting
              </span>
            </div>

          </div>

        </div>


        <button
          className={`status ${
            monitoring ? "active" : "inactive"
          }`}
          onClick={() => setMonitoring(!monitoring)}
        >

          <span className="status-dot"></span>

          {monitoring ? "MONITORING" : "OFF"}

        </button>

      </header>

      {/* =====================================================
            SIDEBAR NAVIGATION
        ===================================================== */}

        <aside className={`sidebar ${sidebarOpen ? "open" : ""}`}>

          <div className="sidebar-top">

            <button
              className={`sidebar-item ${
                activePage === "monitor" ? "selected" : ""
              }`}
              onClick={() => {
                setActivePage("monitor");
              }}
            >
              <span>◉</span>
              <label>Monitor</label>
            </button>


            <button
              className={`sidebar-item ${
                activePage === "history" ||
                activePage === "report"
                  ? "selected"
                  : ""
              }`}
              onClick={() => {
                setActivePage("history");
              }}
            >
              <span>◷</span>
              <label>History</label>
            </button>


            <button
              className={`sidebar-item ${
                activePage === "dictionary" ||
                activePage === "dictionary-detail"
                  ? "selected"
                  : ""
              }`}
              onClick={() => {
                setActivePage("dictionary");
              }}
            >
              <span>▣</span>
              <label>Attack Dictionary</label>
            </button>


            <button
              className={`sidebar-item ${
                activePage === "system"
                  ? "selected"
                  : ""
              }`}
              onClick={() => {
                setActivePage("system");
              }}
            >
              <span>⚙</span>
              <label>Settings</label>
            </button>

          </div>


          <div className="sidebar-bottom">

            <button
              className={`sidebar-item login-item ${
                activePage === "login" ? "selected" : ""
              }`}
              onClick={() => {
                setActivePage("login");
              }}
            >
              <span>♙</span>
              <label>Login</label>
            </button>

          </div>

        </aside>

                    


      {/* ================= CONTENT ================= */}

      <main className="content">


        {/* =====================================================
            MONITOR
        ===================================================== */}

        {activePage === "monitor" && (

          <div>

            <div className="page-heading">

              <h2>Network Monitor</h2>

              <p>
                Continuous activity and threat forecasting
              </p>

            </div>


            <div style={{ marginBottom: "1rem" }}>
              {apiLoading && "Loading real CICIDS2017 test-split inference from backend…"}
              {!apiLoading && apiError && `Forecast API unavailable: ${apiError}`}
              {!apiLoading && !apiError && currentForecast && (
                <span>Backend connected · {currentForecast.capture} · ground truth: {currentForecast.actual}</span>
              )}
            </div>

            {/* ACTUAL MODEL DETECTION */}
<section className="active-threat" style={{ marginBottom: "1rem" }}>
  <div className="threat-label">TON_IoT MODEL DETECTION</div>

  {currentDetection ? (
    <>
      <h2 className="threat-name">
        {currentDetection.predicted_class.toUpperCase()}
      </h2>

      <div className="threat-metrics">
        <div>
          <span>Confidence</span>
          <strong>
            {(currentDetection.confidence * 100).toFixed(2)}%
          </strong>
        </div>

        <div>
          <span>Risk Score</span>
          <strong>{currentDetection.risk_score}</strong>
        </div>

        <div>
          <span>Severity</span>
          <strong>{currentDetection.severity}</strong>
        </div>
      </div>

      <p>
        Actual label: {currentDetection.actual_class} ·
        {" "}Prediction {currentDetection.correct ? "correct" : "incorrect"}
      </p>

      <small>
        Offline dataset replay · Sample {currentDetection.sample_index}
      </small>
    </>
  ) : (
    <p>Loading detection model…</p>
  )}
</section>

            {/* ACTIVE THREAT */}

            {currentForecast && monitoring ? (

              <section className="active-threat">

                <div className="threat-header">

                  <div>

                    <div className="threat-label">
                      ⚠ ACTIVE FORECAST
                    </div>

                    <div className="threat-category">
                      {currentForecast.category}
                    </div>

                  </div>

                  <div className="severity-badge high">
                    {currentForecast.severity}
                  </div>

                </div>


                <h2 className="threat-name">
                  {currentForecast.attackType}
                </h2>


                <div className="countdown-container">

                  <span>
                    Predicted attack window
                  </span>

                  <strong>
                    {formatTime(leadTime)}
                  </strong>

                  <small>
                    estimated lead time remaining
                  </small>

                </div>


                <div className="threat-metrics">

                    <div>
                      <span>Confidence</span>
                      <strong>
                        {currentForecast.confidence}%
                      </strong>
                    </div>

                    <div>
                      <span>Risk Score</span>
                      <strong>
                        {currentForecast.riskScore}
                      </strong>
                    </div>

                    <div>
                      <span>Lead Time</span>
                      <strong>
                        {formatTime(leadTime)}
                      </strong>
                    </div>

                  </div>


                {/* EXPLAINABILITY */}

                <div className="evidence-section">

                  <div className="subheading">
                    WHY THIS WAS PREDICTED
                  </div>

                  {currentForecast.evidence.map(
                    (item) => (

                      <div
                        className="evidence-row"
                        key={item.label}
                      >

                        <div className="evidence-info">

                          <span>
                            {item.label}
                          </span>

                          <strong>
                            {item.value}
                          </strong>

                        </div>

                        <div className="evidence-bar">

                          <div
                            style={{
                              width: `${item.score}%`,
                            }}
                          />

                        </div>

                      </div>

                    )
                  )}

                  <p className="interpretation">
                    {currentForecast.interpretation}
                  </p>

                </div>


                {/* DEFENSE */}

                <div className="defense-section">

                  <div className="defense-heading">

                    <div>
                      <div className="subheading">
                        DEFENSE MODE
                      </div>

                      <span>
                        Automatic response is currently
                        experimental.
                      </span>
                    </div>


                    <button
                      className={`toggle ${
                        defenseMode ? "on" : ""
                      }`}
                      onClick={() =>
                        setDefenseMode(!defenseMode)
                      }
                    >
                      <span />
                    </button>

                  </div>


                  <div className="defense-status">

                    {defenseMode
                      ? "Defense Mode enabled — automatic response will be available in a future release."
                      : "Recommendation mode — NetGuard will suggest actions but will not execute them."}

                  </div>


                  <div className="recommendations">

                    {currentForecast.recommendations.map(
                      (recommendation) => (

                        <div
                          className="recommendation"
                          key={recommendation}
                        >
                          ✓ {recommendation}
                        </div>

                      )
                    )}

                  </div>

                </div>

              </section>

            ) : (

              <section className="no-threat">

                <div className="safe-icon">
                  ✓
                </div>

                <h2>
                  No Active Threats
                </h2>

                <p>
                  Network activity is currently within
                  normal operating patterns.
                </p>

              </section>

            )}
            {/* NETWORK ACTIVITY */}

          <section className="network-monitor">

            <div className="network-header">

              <div>
                <span className="subheading">
                  NETWORK MONITORING
                </span>

                <h3>
                  Dataset Traffic Replay
                </h3>
              </div>

              <span className="live-indicator">
                ● OFFLINE REPLAY
              </span>

            </div>


            <p className="dataset-note" style={{ margin: "0.35rem 0 0.75rem", opacity: 0.75, fontSize: "0.75rem" }}>
              {trafficSnapshot ? `${trafficSnapshot.source} · ${trafficSnapshot.capture}` : (trafficError || "Loading dataset traffic replay…")}
            </p>

            {/* TRAFFIC GRAPH */}

            <div className="traffic-chart-card">

              <div className="chart-title">

                <div>
                  <strong>Traffic Flow</strong>
                  <span>Inbound vs outbound throughput</span>
                </div>

                <div className="chart-legend">

                  <span>
                    <i className="legend-inbound"></i>
                    Inbound
                  </span>

                  <span>
                    <i className="legend-outbound"></i>
                    Outbound
                  </span>

                </div>

              </div>


              <div className="chart">

                <ResponsiveContainer
                  width="100%"
                  height={190}
                >

                  <AreaChart data={trafficSnapshot?.minutes || []}>

                    <defs>

                      <linearGradient
                        id="inboundGradient"
                        x1="0"
                        y1="0"
                        x2="0"
                        y2="1"
                      >

                        <stop
                          offset="0%"
                          stopColor="#69d596"
                          stopOpacity={0.28}
                        />

                        <stop
                          offset="100%"
                          stopColor="#69d596"
                          stopOpacity={0}
                        />

                      </linearGradient>


                      <linearGradient
                        id="outboundGradient"
                        x1="0"
                        y1="0"
                        x2="0"
                        y2="1"
                      >

                        <stop
                          offset="0%"
                          stopColor="#78aef5"
                          stopOpacity={0.22}
                        />

                        <stop
                          offset="100%"
                          stopColor="#78aef5"
                          stopOpacity={0}
                        />

                      </linearGradient>

                    </defs>


                    <CartesianGrid
                      vertical={false}
                      stroke="currentColor"
                      opacity={0.08}
                    />

                    <XAxis
                      dataKey="time"
                      tick={{ fontSize: 8 }}
                      axisLine={false}
                      tickLine={false}
                    />

                    <YAxis
                      tick={{ fontSize: 8 }}
                      axisLine={false}
                      tickLine={false}
                      width={25}
                    />

                    <Tooltip />


                    <Area
                      type="monotone"
                      dataKey="inbound"
                      stroke="#69d596"
                      strokeWidth={2}
                      fill="url(#inboundGradient)"
                    />

                    <Area
                      type="monotone"
                      dataKey="outbound"
                      stroke="#78aef5"
                      strokeWidth={2}
                      fill="url(#outboundGradient)"
                    />

                  </AreaChart>

                </ResponsiveContainer>

              </div>

            </div>


            {/* METRICS */}

            <div className="network-metrics">

              <div className="network-metric">

                <span>AVERAGE FLOWS / MIN</span>

                <strong>{trafficSnapshot ? Math.round(trafficSnapshot.total_flows / Math.max(trafficSnapshot.minutes.length, 1)) : "—"}</strong>

                <small>Average flows per minute in replay window</small>

              </div>


              <div className="network-metric">

                <span>RECORDED BYTES / MIN</span>

                <strong>{trafficSnapshot ? `${(trafficSnapshot.minutes.reduce((sum, row) => sum + row.inbound + row.outbound, 0) / Math.max(trafficSnapshot.minutes.length, 1)).toFixed(1)} KB/min` : "—"}</strong>

                <small>Average recorded packet bytes per minute (KB)</small>

              </div>


              <div className="network-metric">

                <span>TOTAL FLOWS</span>

                <strong>{trafficSnapshot ? trafficSnapshot.total_flows : "—"}</strong>

                <small>Total flows in selected replay window</small>

              </div>


              <div className="network-metric">

                <span>ATTACK-LABELLED FLOWS</span>

                <strong>{trafficSnapshot ? trafficSnapshot.attack_flows : "—"}</strong>

                <small>Flows labelled as attacks</small>

              </div>

            </div>


            {/* LOWER VISUALS */}

            <div className="network-visual-grid">


              {/* PROTOCOL DISTRIBUTION */}

              <div className="mini-chart-card">

                <div className="mini-chart-header">

                  <strong>Protocol Distribution</strong>

                  <span>Current</span>

                </div>


                <div className="protocol-chart">

                  <ResponsiveContainer
                    width="100%"
                    height={145}
                  >

                    <PieChart>

                      <Pie
                        data={trafficSnapshot?.protocols || []}
                        dataKey="value"
                        nameKey="name"
                        cx="50%"
                        cy="50%"
                        innerRadius={38}
                        outerRadius={58}
                        paddingAngle={3}
                      >

                        <Cell fill="#69d596" />
                        <Cell fill="#78aef5" />
                        <Cell fill="#e6bf69" />
                        <Cell fill="#8b96a3" />

                      </Pie>

                      <Tooltip />

                    </PieChart>

                  </ResponsiveContainer>


                  <div className="protocol-center">
                    <strong>{trafficSnapshot?.protocols?.[0]?.value ?? "—"}%</strong>
                    <span>{trafficSnapshot?.protocols?.[0]?.name ?? "Protocol"}</span>
                  </div>

                </div>


                <div className="protocol-list">

                  {(trafficSnapshot?.protocols || []).map((protocol, index) => (

                    <div
                      key={protocol.name}
                      className="protocol-item"
                    >

                      <span>
                        <i className={`protocol-dot dot-${index}`} />
                        {protocol.name}
                      </span>

                      <strong>
                        {protocol.value}%
                      </strong>

                    </div>

                  ))}

                </div>

              </div>


              {/* ACTIVITY / RISK */}

              <div className="mini-chart-card">

                <div className="mini-chart-header">

                  <strong>Activity Risk</strong>

                  <span>
                    Offline capture window
                  </span>

                </div>


                <div className="risk-chart">

                  <ResponsiveContainer
                    width="100%"
                    height={180}
                  >

                    <BarChart data={trafficSnapshot?.minutes || []}>

                      <XAxis
                        dataKey="time"
                        hide
                      />

                      <YAxis
                        hide
                        domain={[0, 100]}
                      />

                      <Tooltip />

                      <Bar
                        dataKey="risk"
                        radius={[4, 4, 0, 0]}
                        fill="#c63d45"
                      />

                    </BarChart>

                  </ResponsiveContainer>

                </div>


                <div className="risk-summary">

                  <span>
                    Current risk
                  </span>

                  <strong>
                    {trafficSnapshot ? `${trafficSnapshot.risk_score} / 100` : "—"}
                  </strong>

                </div>

              </div>

            </div>

          </section>

          </div>

        )}


        {/* =====================================================
            HISTORY
        ===================================================== */}

        {activePage === "history" && (

          <div>

            <div className="page-heading">

              <h2>Attack History</h2>

              <p>
                Previous forecasts and security events
              </p>

            </div>


            <section className="card history-list">

              {attackHistory.map((attack) => (

                <button
                  className="history-item"
                  key={attack.id}
                  onClick={() => {
                    setSelectedAttack(attack);
                    setActivePage("report");
                  }}
                >

                  <div className="history-main">

                    <div className="history-icon">
                      ⚠
                    </div>

                    <div>

                      <strong>
                        {attack.attackType}
                      </strong>

                      <span>
                        {attack.category}
                      </span>

                      <small>
                        {attack.date} • {attack.time}
                      </small>

                    </div>

                  </div>


                  <div className="history-right">

                    <span
                      className={`severity-text ${
                        attack.severity.toLowerCase()
                      }`}
                    >
                      {attack.severity}
                    </span>

                    <span className="arrow">
                      →
                    </span>

                  </div>

                </button>

              ))}

            </section>

          </div>

        )}


        {/* =====================================================
            ATTACK REPORT
        ===================================================== */}

        {activePage === "report" &&
          selectedAttack && (

            <div>

              <button
                className="back-button"
                onClick={() => {
                  setSelectedAttack(null);
                  setActivePage("history");
                }}
              >
                ← Back to History
              </button>


              <div className="page-heading">

                <h2>
                  Attack Report
                </h2>

                <p>
                  Detailed security event information
                </p>

              </div>


              <section className="card report-card">

                <div className="report-title">

                  <div>
                    <span className="subheading">
                      {selectedAttack.category}
                    </span>

                    <h2>
                      {selectedAttack.attackType}
                    </h2>
                  </div>

                  <span
                    className={`severity-badge ${
                      selectedAttack.severity.toLowerCase()
                    }`}
                  >
                    {selectedAttack.severity}
                  </span>

                </div>


                <div className="report-grid">

                  <div>
                    <span>Date</span>
                    <strong>
                      {selectedAttack.date}
                    </strong>
                  </div>

                  <div>
                    <span>Time</span>
                    <strong>
                      {selectedAttack.time}
                    </strong>
                  </div>

                  <div>
                    <span>Confidence</span>
                    <strong>
                      {selectedAttack.confidence}%
                    </strong>
                  </div>

                  <div>
                    <span>Risk Score</span>
                    <strong>
                      {selectedAttack.riskScore}
                    </strong>
                  </div>

                  <div>
                    <span>Lead Time</span>
                    <strong>
                      {selectedAttack.leadTime}
                    </strong>
                  </div>

                  <div>
                    <span>Status</span>
                    <strong>
                      {selectedAttack.status}
                    </strong>
                  </div>

                </div>


                <div className="report-section">

                  <span className="subheading">
                    ACTION TAKEN
                  </span>

                  <p>
                    {selectedAttack.action}
                  </p>

                </div>

              </section>

            </div>

          )}


        {/* =====================================================
            DICTIONARY
        ===================================================== */}

        {activePage === "dictionary" && (

          <div>

            <div className="page-heading">

              <div className="dictionary-heading">

                <div className="book-icon">
                  ▣
                </div>

                <div>

                  <h2>
                    Attack Dictionary
                  </h2>

                  <p>
                    Understand common network threats
                  </p>

                </div>

              </div>

            </div>


            <div className="dictionary-grid">

              {attackDictionary.map((attack) => (

                <button
                  className="dictionary-card"
                  key={attack.type}
                  onClick={() => {
                    setSelectedDictionaryAttack(attack);
                    setActivePage("dictionary-detail");
                  }}
                >

                  <div className="dictionary-icon">
                    {attack.icon}
                  </div>

                  <span className="dictionary-category">
                    {attack.category}
                  </span>

                  <h3>
                    {attack.type}
                  </h3>

                  <p>
                    {attack.description}
                  </p>

                  <span className="view-details">
                    View details →
                  </span>

                </button>

              ))}

            </div>

          </div>

        )}


        {/* =====================================================
            DICTIONARY DETAIL
        ===================================================== */}

        {activePage === "dictionary-detail" &&
          selectedDictionaryAttack && (

            <div>

              <button
                className="back-button"
                onClick={() => {
                  setSelectedDictionaryAttack(null);
                  setActivePage("dictionary");
                }}
              >
                ← Back to Dictionary
              </button>


              <div className="page-heading">

                <span className="dictionary-category">
                  {selectedDictionaryAttack.category}
                </span>

                <h2>
                  {selectedDictionaryAttack.type}
                </h2>

                <p>
                  {selectedDictionaryAttack.description}
                </p>

              </div>


              <section className="card dictionary-detail">

                <div className="detail-section">

                  <span className="subheading">
                    COMMON INDICATORS
                  </span>

                  {selectedDictionaryAttack.indicators.map(
                    (indicator) => (

                      <div
                        className="detail-row"
                        key={indicator}
                      >
                        <span>•</span>
                        {indicator}
                      </div>

                    )
                  )}

                </div>


                <div className="detail-section">

                  <span className="subheading">
                    POSSIBLE DEFENSES
                  </span>

                  {selectedDictionaryAttack.defenses.map(
                    (defense) => (

                      <div
                        className="detail-row"
                        key={defense}
                      >
                        <span>✓</span>
                        {defense}
                      </div>

                    )
                  )}

                </div>

              </section>

            </div>

          )}
          {/* =====================================================
            LOGIN
        ===================================================== */}

        
        {activePage === "login" && (

          <div className="auth-page">

            <div className="auth-card">

              <div className="auth-logo">
                ◉
              </div>

              {authMode === "login" ? (

                <>
                  <h2>Welcome back</h2>

                  <p className="auth-subtitle">
                    Sign in to access your NetGuard dashboard.
                  </p>

                  <form
                    className="auth-form"
                    onSubmit={(e) => e.preventDefault()}
                  >

                    <label>Email</label>

                    <input
                      type="email"
                      placeholder="Enter your email"
                    />

                    <label>Password</label>

                    <input
                      type="password"
                      placeholder="Enter your password"
                    />

                    <button
                      type="submit"
                      className="auth-submit"
                    >
                      Sign In
                    </button>

                  </form>

                  <div className="auth-switch">

                    <span>
                      Don't have an account?
                    </span>

                    <button
                      onClick={() => setAuthMode("signup")}
                    >
                      Create one
                    </button>

                  </div>
                </>

              ) : (

                <>
                  <h2>Create your account</h2>

                  <p className="auth-subtitle">
                    Set up your NetGuard security profile.
                  </p>

                  <form
                    className="auth-form"
                    onSubmit={(e) => e.preventDefault()}
                  >

                    <label>Name</label>

                    <input
                      type="text"
                      placeholder="Enter your name"
                    />

                    <label>Email</label>

                    <input
                      type="email"
                      placeholder="Enter your email"
                    />

                    <label>Password</label>

                    <input
                      type="password"
                      placeholder="Create a password"
                    />

                    <label>Confirm Password</label>

                    <input
                      type="password"
                      placeholder="Confirm your password"
                    />

                    <button
                      type="submit"
                      className="auth-submit"
                    >
                      Create Account
                    </button>

                  </form>

                  <div className="auth-switch">

                    <span>
                      Already have an account?
                    </span>

                    <button
                      onClick={() => setAuthMode("login")}
                    >
                      Sign in
                    </button>

                  </div>
                </>

              )}

              <div className="auth-demo-note">
                Authentication will be connected to FastAPI later.
              </div>

            </div>

          </div>

        )}

        {/* =====================================================
            SETTINGS
        ===================================================== */}

        {activePage === "system" && (

          <div>

            <div className="page-heading">

              <h2>
                Settings
              </h2>

              <p>
                Configure how NetGuard behaves
              </p>

            </div>


            <section className="card settings-card">

              {/* Monitoring */}

              <div className="setting">

                <div>

                  <strong>
                    Network Monitoring
                  </strong>

                  <span>
                    Run monitoring in the background
                  </span>

                </div>

                <button
                  className={`toggle ${
                    monitoring ? "on" : ""
                  }`}
                  onClick={() =>
                    setMonitoring(!monitoring)
                  }
                >
                  <span />
                </button>

              </div>


              {/* Notifications */}

              <div className="setting">

                <div>

                  <strong>
                    Threat Notifications
                  </strong>

                  <span>
                    Receive alerts for relevant forecasts
                  </span>

                </div>

                <button className="toggle on">
                  <span />
                </button>

              </div>


              {/* Theme */}

              <div className="setting">

                <div>

                  <strong>
                    Dark Mode
                  </strong>

                  <span>
                    Switch between light and dark appearance
                  </span>

                </div>

                <button
                  className={`toggle ${
                    darkMode ? "on" : ""
                  }`}
                  onClick={() =>
                    setDarkMode(!darkMode)
                  }
                >
                  <span />
                </button>

              </div>


              {/* Sensitivity */}

              <div className="sensitivity-setting">

                <div>

                  <strong>
                    Threat Sensitivity
                  </strong>

                  <span>
                    Choose which threat levels should
                    trigger attention
                  </span>

                </div>


                <div className="sensitivity-options">

                  {[
                    ["critical", "Critical only"],
                    ["high", "High & Critical"],
                    ["medium", "Medium & above"],
                    ["all", "All activity"],
                  ].map(([value, label]) => (

                    <button
                      key={value}
                      className={
                        sensitivity === value
                          ? "selected"
                          : ""
                      }
                      onClick={() =>
                        setSensitivity(value)
                      }
                    >
                      {label}
                    </button>

                  ))}

                </div>

              </div>


              {/* Defense */}

              <div className="setting">

                <div>

                  <strong>
                    Defense Mode
                  </strong>

                  <span>
                    Future automatic response capability
                  </span>

                </div>

                <button
                  className={`toggle ${
                    defenseMode ? "on" : ""
                  }`}
                  onClick={() =>
                    setDefenseMode(!defenseMode)
                  }
                >
                  <span />
                </button>

              </div>


              {/* Status */}

              <div className="setting">

                <div>

                  <strong>
                    Backend Status
                  </strong>

                  <span>
                    Local simulation environment
                  </span>

                </div>

                <span className="connected">
                  CONNECTED
                </span>

              </div>

            </section>

          </div>

        )}

      </main>

    </div>

    

  );
}


export default App;