import { useState, useRef, useEffect } from "react";

const API_URL = process.env.REACT_APP_API_URL || "http://localhost:8000";

function Sparkline({ data, width = 140, height = 45 }) {
  if (!data || data.length < 2) return null;
  const vals = data.map((d) => d.Close || 0).filter(Boolean);
  const min = Math.min(...vals), max = Math.max(...vals);
  const range = max - min || 1;
  const pts = vals.map((v, i) => {
    const x = (i / (vals.length - 1)) * width;
    const y = height - ((v - min) / range) * (height - 8) - 4;
    return `${x},${y}`;
  }).join(" ");
  const trend = vals[vals.length - 1] >= vals[0];
  return (
    <svg width={width} height={height}>
      <defs>
        <linearGradient id="sg" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={trend ? "#2a7a4b" : "#b94040"} stopOpacity="0.18" />
          <stop offset="100%" stopColor={trend ? "#2a7a4b" : "#b94040"} stopOpacity="0" />
        </linearGradient>
      </defs>
      <polygon points={`0,${height} ${pts} ${width},${height}`} fill="url(#sg)" />
      <polyline points={pts} fill="none" stroke={trend ? "#2a7a4b" : "#b94040"} strokeWidth="2" strokeLinejoin="round" />
    </svg>
  );
}

function MomentumMeter({ value, label }) {
  const pct = Math.min(Math.max(value / 100, 0), 1);
  const color = pct > 0.65 ? "#2a7a4b" : pct > 0.35 ? "#c07a2a" : "#b94040";
  return (
    <div style={{ marginBottom: 12 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 5 }}>
        <span style={{ fontSize: 12, color: "#6b6660", fontFamily: "Georgia, serif" }}>{label}</span>
        <span style={{ fontSize: 13, fontWeight: "bold", color, fontFamily: "Georgia, serif" }}>{Math.round(value)}</span>
      </div>
      <div style={{ height: 6, background: "#e8e4de", borderRadius: 3, overflow: "hidden" }}>
        <div style={{ height: "100%", width: `${pct * 100}%`, background: color, borderRadius: 3, transition: "width 0.8s ease" }} />
      </div>
    </div>
  );
}

function PriceChart({ data, forecast }) {
  if (!data || data.length === 0) return null;
  const W = 680, H = 160, PL = 50, PR = 20, PT = 10, PB = 28;
  const chartW = W - PL - PR, chartH = H - PT - PB;
  const allPrices = data.map((d) => d.Close || 0).filter(Boolean);
  const fcData = forecast?.forecast_data?.slice(-30) || [];
  const fcPrices = fcData.map((d) => d.yhat);
  const allVals = [...allPrices, ...fcPrices].filter(Boolean);
  const minV = Math.min(...allVals) * 0.985;
  const maxV = Math.max(...allVals) * 1.015;
  const range = maxV - minV || 1;
  const hx = (i, len) => PL + (i / (len - 1)) * chartW;
  const hy = (v) => PT + chartH - ((v - minV) / range) * chartH;
  const histPts = allPrices.map((v, i) => `${hx(i, allPrices.length)},${hy(v)}`).join(" ");
  const lastX = hx(allPrices.length - 1, allPrices.length);
  const fcPts = fcData.map((d, i) => `${lastX + (i / (fcData.length - 1 || 1)) * (PR + 60)},${hy(d.yhat)}`).join(" ");
  const trend = allPrices[allPrices.length - 1] >= allPrices[0];
  const lineColor = trend ? "#2a7a4b" : "#b94040";
  const yTicks = [0.2, 0.5, 0.8].map(f => ({ y: PT + chartH * (1 - f), val: (minV + range * f).toFixed(0) }));

  return (
    <svg width="100%" viewBox={`0 0 ${W + 60} ${H}`} style={{ overflow: "visible" }}>
      <defs>
        <linearGradient id="cg" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={lineColor} stopOpacity="0.12" />
          <stop offset="100%" stopColor={lineColor} stopOpacity="0" />
        </linearGradient>
      </defs>
      {yTicks.map((t, i) => (
        <g key={i}>
          <line x1={PL} y1={t.y} x2={W + 40} y2={t.y} stroke="#e8e4de" strokeWidth={1} />
          <text x={PL - 6} y={t.y + 4} textAnchor="end" fill="#a09890" fontSize={10} fontFamily="Georgia, serif">${t.val}</text>
        </g>
      ))}
      <polygon points={`${PL},${PT + chartH} ${histPts} ${lastX},${PT + chartH}`} fill="url(#cg)" />
      <polyline points={histPts} fill="none" stroke={lineColor} strokeWidth={2} strokeLinejoin="round" />
      <line x1={lastX} y1={PT} x2={lastX} y2={PT + chartH} stroke="#c8c0b8" strokeWidth={1} strokeDasharray="4,3" />
      {fcPts && fcData.length > 1 && (
        <polyline points={`${allPrices.map((v, i) => `${hx(i, allPrices.length)},${hy(v)}`).slice(-1)[0]} ${fcPts}`}
          fill="none" stroke="#8a7a6a" strokeWidth={1.5} strokeDasharray="6,3" opacity={0.7} />
      )}
      <text x={lastX + 4} y={PT + 10} fill="#8a7a6a" fontSize={9} fontFamily="Georgia, serif">forecast →</text>
    </svg>
  );
}

function ChatBubble({ msg }) {
  const isUser = msg.role === "user";
  return (
    <div style={{ display: "flex", justifyContent: isUser ? "flex-end" : "flex-start", marginBottom: 10 }}>
      {!isUser && (
        <div style={{ width: 28, height: 28, borderRadius: "50%", background: "#e8e4de", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 11, color: "#6b6660", fontWeight: "bold", marginRight: 8, flexShrink: 0, fontFamily: "Georgia, serif" }}>A</div>
      )}
      <div style={{
        maxWidth: "82%", padding: "10px 14px", borderRadius: isUser ? "14px 14px 4px 14px" : "14px 14px 14px 4px",
        background: isUser ? "#3a5a3a" : "#f5f1ec",
        color: isUser ? "#e8f0e8" : "#2a2420",
        fontSize: 13, lineHeight: 1.65, fontFamily: "Georgia, serif",
        boxShadow: "0 1px 3px rgba(0,0,0,0.06)",
        whiteSpace: "pre-wrap"
      }}>{msg.content}</div>
      {isUser && (
        <div style={{ width: 28, height: 28, borderRadius: "50%", background: "#3a5a3a", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 11, color: "#e8f0e8", fontWeight: "bold", marginLeft: 8, flexShrink: 0, fontFamily: "Georgia, serif" }}>D</div>
      )}
    </div>
  );
}

export default function App() {
  const [ticker, setTicker] = useState("");
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [messages, setMessages] = useState([]);
  const [chatInput, setChatInput] = useState("");
  const [chatLoading, setChatLoading] = useState(false);
  const [tab, setTab] = useState("overview");
  const chatEndRef = useRef(null);

  useEffect(() => { chatEndRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages]);

  const handleAnalyze = async () => {
    if (!ticker.trim()) return;
    setLoading(true); setError(""); setAnalysis(null); setMessages([]);
    try {
      const res = await fetch(`${API_URL}/analyze`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ticker: ticker.trim().toUpperCase() }),
      });
      if (!res.ok) { const e = await res.json(); throw new Error(e.detail || "Analysis failed"); }
      const data = await res.json();
      setAnalysis(data);
      setMessages([{ role: "assistant", content: `I've pulled up ${data.data.name} (${ticker.toUpperCase()}). The ML models have run and the commentary is ready. What would you like to dig into?` }]);
    } catch (e) { setError(e.message); }
    finally { setLoading(false); }
  };

  const handleChat = async () => {
    if (!chatInput.trim() || !analysis) return;
    const question = chatInput.trim();
    setChatInput("");
    const newMessages = [...messages, { role: "user", content: question }];
    setMessages(newMessages);
    setChatLoading(true);
    try {
      const res = await fetch(`${API_URL}/chat`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ticker: analysis.data.ticker, question, history: newMessages.slice(-8) }),
      });
      if (!res.ok) throw new Error("Chat failed");
      const data = await res.json();
      setMessages([...newMessages, { role: "assistant", content: data.response }]);
    } catch (e) { setMessages([...newMessages, { role: "assistant", content: "Having trouble connecting. Please try again." }]); }
    finally { setChatLoading(false); }
  };

  const d = analysis?.data;
  const f = analysis?.forecast;
  const m = analysis?.momentum;
  const priceUp = d?.price_change >= 0;
  const fmt = (n, dec = 2) => n != null ? Number(n).toFixed(dec) : "—";
  const fmtM = (n) => !n ? "—" : n >= 1e12 ? `$${(n / 1e12).toFixed(2)}T` : n >= 1e9 ? `$${(n / 1e9).toFixed(2)}B` : n >= 1e6 ? `$${(n / 1e6).toFixed(2)}M` : `$${n}`;

  const suggestedQuestions = [
    "How does the Sharpe ratio compare to the market?",
    "What does the momentum score indicate?",
    "Is the current valuation reasonable?",
    "What are the main risk factors?",
  ];

  return (
    <div style={{ minHeight: "100vh", background: "#faf7f4", color: "#2a2420", fontFamily: "Georgia, serif" }}>
      <style>{`
        * { box-sizing: border-box; margin: 0; padding: 0; }
        ::-webkit-scrollbar { width: 5px; }
        ::-webkit-scrollbar-track { background: #f0ece6; }
        ::-webkit-scrollbar-thumb { background: #c8c0b8; border-radius: 3px; }
        input::placeholder { color: #b0a898; }
        textarea::placeholder { color: #b0a898; }
        .tab { background: none; border: none; cursor: pointer; padding: 10px 18px; font-family: Georgia, serif; font-size: 13px; color: #8a7a6a; border-bottom: 2px solid transparent; transition: all 0.2s; }
        .tab.active { color: #3a5a3a; border-bottom-color: #3a5a3a; font-weight: bold; }
        .tab:hover { color: #4a6a4a; }
        .card { background: #fff; border: 1px solid #e8e4de; border-radius: 10px; padding: 20px; }
        .metric { display: flex; justify-content: space-between; align-items: center; padding: 9px 0; border-bottom: 1px solid #f0ece6; font-size: 13px; }
        .metric:last-child { border-bottom: none; }
        .kpi { background: #fff; border: 1px solid #e8e4de; border-radius: 8px; padding: 14px 16px; }
        .analyze-btn { background: #3a5a3a; color: #fff; border: none; padding: 13px 28px; border-radius: 8px; cursor: pointer; font-family: Georgia, serif; font-size: 14px; font-weight: bold; transition: background 0.2s; letter-spacing: 0.3px; }
        .analyze-btn:hover { background: #2a4a2a; }
        .analyze-btn:disabled { background: #c8c0b8; cursor: not-allowed; }
        .send-btn { background: #3a5a3a; color: #fff; border: none; padding: 10px 18px; border-radius: 6px; cursor: pointer; font-family: Georgia, serif; font-size: 13px; font-weight: bold; transition: background 0.2s; }
        .send-btn:hover { background: #2a4a2a; }
        .send-btn:disabled { background: #c8c0b8; cursor: not-allowed; }
        .suggestion { background: #f5f1ec; border: 1px solid #e0dbd4; border-radius: 20px; padding: 5px 12px; font-size: 11px; color: #6b6660; cursor: pointer; font-family: Georgia, serif; transition: all 0.15s; white-space: nowrap; }
        .suggestion:hover { background: #ede8e0; color: #3a5a3a; border-color: #3a5a3a; }
        @keyframes fadeUp { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }
        .fade-up { animation: fadeUp 0.4s ease forwards; }
        @keyframes spin { to { transform: rotate(360deg); } }
        .spinner { width: 18px; height: 18px; border: 2px solid #e8e4de; border-top-color: #3a5a3a; border-radius: 50%; animation: spin 0.8s linear infinite; display: inline-block; }
      `}</style>

      {/* Header */}
      <div style={{ background: "#fff", borderBottom: "1px solid #e8e4de", padding: "16px 40px", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <div>
          <div style={{ fontSize: 18, fontWeight: "bold", color: "#2a2420", letterSpacing: "-0.3px" }}>Meridian Analytics</div>
          <div style={{ fontSize: 11, color: "#8a7a6a", marginTop: 1 }}>Institutional-grade financial intelligence</div>
        </div>
        <div style={{ fontSize: 10, color: "#b0a898", fontStyle: "italic" }}>For research purposes only · Not investment advice</div>
      </div>

      <div style={{ maxWidth: 1080, margin: "0 auto", padding: "32px 24px" }}>

        {/* Search */}
        <div style={{ marginBottom: 32 }}>
          <h1 style={{ fontSize: 26, fontWeight: "bold", color: "#2a2420", marginBottom: 6, letterSpacing: "-0.5px" }}>
            Financial Intelligence Platform
          </h1>
          <p style={{ fontSize: 14, color: "#8a7a6a", marginBottom: 20, lineHeight: 1.6 }}>
            Enter any stock, ETF, or mutual fund ticker to generate ML-powered analysis, price forecasting, and an AI research commentary.
          </p>
          <div style={{ display: "flex", gap: 12, maxWidth: 560 }}>
            <input value={ticker} onChange={(e) => setTicker(e.target.value.toUpperCase())}
              onKeyDown={(e) => e.key === "Enter" && handleAnalyze()}
              placeholder="AAPL, JPM, SPY, VFIAX..."
              maxLength={10}
              style={{ flex: 1, background: "#fff", border: "1px solid #d0c8c0", color: "#2a2420", padding: "13px 16px", borderRadius: 8, fontSize: 15, fontFamily: "Georgia, serif", outline: "none", letterSpacing: 1 }} />
            <button className="analyze-btn" onClick={handleAnalyze} disabled={loading || !ticker.trim()}>
              {loading ? <span className="spinner" /> : "Analyze →"}
            </button>
          </div>
          {/* Quick tickers */}
          <div style={{ display: "flex", gap: 8, marginTop: 12, flexWrap: "wrap" }}>
            {["AAPL", "JPM", "SPY", "BRK-B", "VFIAX", "QQQ"].map((t) => (
              <button key={t} className="suggestion" onClick={() => { setTicker(t); }}>{t}</button>
            ))}
          </div>
        </div>

        {error && (
          <div style={{ background: "#fff5f5", border: "1px solid #e8c0c0", color: "#8a3030", padding: "12px 16px", borderRadius: 8, marginBottom: 20, fontSize: 13 }}>
            {error}
          </div>
        )}

        {analysis && (
          <div className="fade-up">

            {/* Security header */}
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 20, flexWrap: "wrap", gap: 12 }}>
              <div>
                <h2 style={{ fontSize: 22, fontWeight: "bold", color: "#2a2420", marginBottom: 4 }}>{d.name}</h2>
                <div style={{ fontSize: 12, color: "#8a7a6a" }}>{d.ticker} · {d.sector} · {d.industry}</div>
              </div>
              <div style={{ textAlign: "right" }}>
                <div style={{ fontSize: 30, fontWeight: "bold", color: "#2a2420", letterSpacing: "-1px" }}>${fmt(d.current_price)}</div>
                <div style={{ fontSize: 13, color: priceUp ? "#2a7a4b" : "#b94040", fontWeight: "bold" }}>
                  {priceUp ? "▲" : "▼"} {Math.abs(d.price_change).toFixed(2)}% today
                </div>
              </div>
            </div>

            {/* KPI row */}
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))", gap: 10, marginBottom: 20 }}>
              {[
                ["Market Cap",    fmtM(d.market_cap)],
                ["52W High",      `$${fmt(d.week_52_high)}`],
                ["52W Low",       `$${fmt(d.week_52_low)}`],
                ["P/E Ratio",     d.pe_ratio ? fmt(d.pe_ratio, 1) : "—"],
                ["Beta",          d.beta ? fmt(d.beta, 2) : "—"],
                ["Dividend Yield", d.dividend_yield ? `${(d.dividend_yield * 100).toFixed(2)}%` : "—"],
              ].map(([k, v]) => (
                <div key={k} className="kpi">
                  <div style={{ fontSize: 10, color: "#a09890", marginBottom: 5, textTransform: "uppercase", letterSpacing: 1 }}>{k}</div>
                  <div style={{ fontSize: 15, fontWeight: "bold", color: "#2a2420" }}>{v}</div>
                </div>
              ))}
            </div>

            {/* Main grid */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 320px", gap: 20 }}>

              {/* Left column */}
              <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>

                {/* Chart card */}
                <div className="card">
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
                    <div style={{ fontSize: 13, fontWeight: "bold", color: "#4a4040" }}>Price History · 1 Year + 30-Day Forecast</div>
                    <div style={{ fontSize: 12, color: f?.direction === "bullish" ? "#2a7a4b" : "#b94040", fontWeight: "bold" }}>
                      ML forecast: {f?.direction} ({f?.forecast_change > 0 ? "+" : ""}{fmt(f?.forecast_change)}%)
                    </div>
                  </div>
                  <PriceChart data={d.price_data?.slice(-60)} forecast={f} />
                </div>

                {/* Tabs */}
                <div className="card" style={{ padding: 0, overflow: "hidden" }}>
                  <div style={{ display: "flex", borderBottom: "1px solid #e8e4de" }}>
                    {["overview", "risk & momentum", "commentary"].map((t) => (
                      <button key={t} className={`tab ${tab === t ? "active" : ""}`} onClick={() => setTab(t)}>{t.charAt(0).toUpperCase() + t.slice(1)}</button>
                    ))}
                  </div>
                  <div style={{ padding: 20 }}>
                    {tab === "overview" && (
                      <div>
                        {[
                          ["Annual Return (1Y)",    `${d.annual_return >= 0 ? "+" : ""}${fmt(d.annual_return)}%`, d.annual_return >= 0 ? "#2a7a4b" : "#b94040"],
                          ["Annual Volatility",     `${fmt(d.annual_vol)}%`, "#2a2420"],
                          ["Sharpe Ratio",          fmt(d.sharpe_ratio, 3), d.sharpe_ratio > 1 ? "#2a7a4b" : d.sharpe_ratio > 0 ? "#c07a2a" : "#b94040"],
                          ["Max Drawdown",          `${fmt(d.max_drawdown)}%`, "#b94040"],
                          ["Value at Risk (95%)",   `${fmt(d.var_95)}%`, "#2a2420"],
                        ].map(([k, v, c]) => (
                          <div key={k} className="metric">
                            <span style={{ color: "#6b6660" }}>{k}</span>
                            <span style={{ fontWeight: "bold", color: c || "#2a2420" }}>{v}</span>
                          </div>
                        ))}
                        {d.description && <p style={{ fontSize: 12, color: "#8a7a6a", marginTop: 16, lineHeight: 1.7 }}>{d.description}</p>}
                      </div>
                    )}
                    {tab === "risk & momentum" && (
                      <div>
                        <MomentumMeter value={m?.momentum_score || 0} label="Momentum Score (0–100)" />
                        <MomentumMeter value={m?.rsi || 50} label="RSI — Relative Strength Index" />
                        <div style={{ marginTop: 16 }}>
                          {[
                            ["20-Day Moving Average", m?.ma_20 ? `$${fmt(m.ma_20)}` : "—"],
                            ["50-Day Moving Average", m?.ma_50 ? `$${fmt(m.ma_50)}` : "—"],
                            ["200-Day Moving Average", m?.ma_200 ? `$${fmt(m.ma_200)}` : "—"],
                          ].map(([k, v]) => (
                            <div key={k} className="metric">
                              <span style={{ color: "#6b6660" }}>{k}</span>
                              <span style={{ fontWeight: "bold" }}>{v}</span>
                            </div>
                          ))}
                        </div>
                        {m?.signals?.length > 0 && (
                          <div style={{ marginTop: 14 }}>
                            <div style={{ fontSize: 11, color: "#8a7a6a", marginBottom: 8, textTransform: "uppercase", letterSpacing: 1 }}>Active Signals</div>
                            {m.signals.map((s, i) => (
                              <div key={i} style={{ fontSize: 12, color: "#4a6a4a", padding: "5px 0", borderBottom: "1px solid #f0ece6" }}>✓ {s}</div>
                            ))}
                          </div>
                        )}
                      </div>
                    )}
                    {tab === "commentary" && (
                      <div style={{ fontSize: 13, color: "#4a4040", lineHeight: 1.85, whiteSpace: "pre-wrap" }}>{analysis.commentary}</div>
                    )}
                  </div>
                </div>
              </div>

              {/* Right: Chat */}
              <div className="card" style={{ padding: 0, display: "flex", flexDirection: "column", height: 560 }}>
                <div style={{ padding: "14px 18px", borderBottom: "1px solid #e8e4de" }}>
                  <div style={{ fontSize: 14, fontWeight: "bold", color: "#2a2420" }}>Research Assistant</div>
                  <div style={{ fontSize: 11, color: "#8a7a6a", marginTop: 2 }}>Ask questions about {d?.name}</div>
                </div>
                <div style={{ flex: 1, overflowY: "auto", padding: "14px 16px", display: "flex", flexDirection: "column" }}>
                  {messages.map((msg, i) => <ChatBubble key={i} msg={msg} />)}
                  {chatLoading && (
                    <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "8px 0" }}>
                      <div style={{ width: 28, height: 28, borderRadius: "50%", background: "#e8e4de", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 11, color: "#6b6660", fontWeight: "bold", fontFamily: "Georgia, serif" }}>A</div>
                      <div style={{ display: "flex", gap: 4 }}>
                        {[0, 1, 2].map((i) => (
                          <div key={i} style={{ width: 6, height: 6, borderRadius: "50%", background: "#b0a898", animation: `spin 1s ease-in-out ${i * 0.15}s infinite` }} />
                        ))}
                      </div>
                    </div>
                  )}
                  <div ref={chatEndRef} />
                </div>

                {/* Suggested questions */}
                {messages.length <= 1 && (
                  <div style={{ padding: "0 16px 10px", display: "flex", flexWrap: "wrap", gap: 6 }}>
                    {suggestedQuestions.map((q) => (
                      <button key={q} className="suggestion" onClick={() => { setChatInput(q); }}>{q}</button>
                    ))}
                  </div>
                )}

                <div style={{ padding: "12px 16px", borderTop: "1px solid #e8e4de", display: "flex", gap: 8 }}>
                  <input value={chatInput} onChange={(e) => setChatInput(e.target.value)}
                    onKeyDown={(e) => e.key === "Enter" && handleChat()}
                    placeholder="Ask a question..."
                    style={{ flex: 1, background: "#faf7f4", border: "1px solid #d0c8c0", color: "#2a2420", padding: "9px 12px", borderRadius: 6, fontSize: 13, fontFamily: "Georgia, serif", outline: "none" }} />
                  <button className="send-btn" onClick={handleChat} disabled={chatLoading || !chatInput.trim()}>Send</button>
                </div>
              </div>
            </div>
          </div>
        )}

        {!analysis && !loading && (
          <div style={{ textAlign: "center", padding: "60px 0", color: "#b0a898" }}>
            <div style={{ fontSize: 36, marginBottom: 16 }}>◇</div>
            <div style={{ fontSize: 15, color: "#8a7a6a", fontStyle: "italic" }}>Enter a ticker above to begin your research</div>
            <div style={{ fontSize: 12, marginTop: 8, color: "#c0b8b0" }}>Stocks · ETFs · Mutual Funds</div>
          </div>
        )}

        {/* Footer */}
        <div style={{ marginTop: 48, paddingTop: 20, borderTop: "1px solid #e8e4de", display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 8 }}>
          <div style={{ fontSize: 11, color: "#b0a898" }}>Meridian Analytics · Built by Dilip Chennam</div>
          <div style={{ fontSize: 11, color: "#b0a898", fontStyle: "italic" }}>ML models: Prophet forecasting · Random Forest · Momentum scoring · Claude AI commentary</div>
        </div>
      </div>
    </div>
  );
}
