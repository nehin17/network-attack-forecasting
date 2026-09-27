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
  networkData,
  currentForecast,
  attackHistory,
  attackDictionary,
  protocolData,
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
  const [monitoring, setMonitoring] = useState(true);
  const [defenseMode, setDefenseMode] = useState(false);
  const [darkMode, setDarkMode] = useState(true);
  const [sensitivity, setSensitivity] = useState("medium");
  const [selectedAttack, setSelectedAttack] = useState(null);
  const [selectedDictionaryAttack, setSelectedDictionaryAttack] =
    useState(null);

  const [leadTime, setLeadTime] = useState(
    currentForecast.leadTimeSeconds
  );

  useEffect(() => {
    if (leadTime <= 0) return;

    const timer = setInterval(() => {
      setLeadTime((current) => Math.max(current - 1, 0));
    }, 1000);

    return () => clearInterval(timer);
  }, [leadTime]);


  const themeClass = darkMode ? "dark-theme" : "light-theme";


  return (
    <div className={`app ${themeClass}`}>

      {/* ================= HEADER ================= */}

      <header className="header">

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
                  Live Traffic Activity
                </h3>
              </div>

              <span className="live-indicator">
                ● LIVE
              </span>

            </div>


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

                  <AreaChart data={networkData}>

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

                <span>PACKETS / SEC</span>

                <strong>850</strong>

                <small>↑ 18% vs baseline</small>

              </div>


              <div className="network-metric">

                <span>THROUGHPUT</span>

                <strong>2.4 MB/s</strong>

                <small>↑ 12% vs baseline</small>

              </div>


              <div className="network-metric">

                <span>FLOWS / MIN</span>

                <strong>120</strong>

                <small>↑ 27% vs baseline</small>

              </div>


              <div className="network-metric">

                <span>CONNECTIONS</span>

                <strong>34</strong>

                <small>↑ 9% vs baseline</small>

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
                        data={protocolData}
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
                    <strong>68%</strong>
                    <span>TCP</span>
                  </div>

                </div>


                <div className="protocol-list">

                  {protocolData.map((protocol, index) => (

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
                    Last 9 min
                  </span>

                </div>


                <div className="risk-chart">

                  <ResponsiveContainer
                    width="100%"
                    height={180}
                  >

                    <BarChart data={networkData}>

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
                    87 / 100
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


      {/* =====================================================
          BOTTOM NAVIGATION
      ===================================================== */}

      <nav className="navigation">

        <button
          className={
            activePage === "monitor"
              ? "selected"
              : ""
          }
          onClick={() =>
            setActivePage("monitor")
          }
        >

          <span>◉</span>
          Monitor

        </button>


        <button
          className={
            activePage === "history" ||
            activePage === "report"
              ? "selected"
              : ""
          }
          onClick={() =>
            setActivePage("history")
          }
        >

          <span>◷</span>
          History

        </button>


        <button
          className={
            activePage === "dictionary" ||
            activePage === "dictionary-detail"
              ? "selected"
              : ""
          }
          onClick={() =>
            setActivePage("dictionary")
          }
        >

          <span>▣</span>
          Dictionary

        </button>


        <button
          className={
            activePage === "system"
              ? "selected"
              : ""
          }
          onClick={() =>
            setActivePage("system")
          }
        >

          <span>⚙</span>
          Settings

        </button>

      </nav>

    </div>
  );
}

export default App;