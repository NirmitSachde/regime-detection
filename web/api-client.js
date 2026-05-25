/* ============================================================
   api-client.js — fetch wrapper that talks to the FastAPI service.
   Falls back to baked sample data (sample_data.js) on any failure.
   ============================================================ */
(function () {
  "use strict";

  const cfg = window.SITE_CONFIG || {};
  const API_BASE = (cfg.API_BASE || "").replace(/\/+$/, "");

  const TIMEOUT_MS = 9000;       // longer than Render cold-start would normally take
  const CACHE_TTL_MS = 60_000;   // in-memory cache window per path
  const _cache = new Map();      // path → { at, body }

  // Observable status — pages render a banner from this
  // mode:        whether the API call succeeded ('live') or fell back ('sample')
  // dataSource:  what the API itself reports — 'warehouse' (real pipeline data)
  //              or 'synthetic' (baked-in fallback). null until first call.
  const status = {
    mode: "unknown",          // "live" | "sample" | "loading"
    dataSource: null,         // "warehouse" | "synthetic" | null
    apiBase: API_BASE,
    lastError: null,
    listeners: new Set(),
    subscribe(fn) { this.listeners.add(fn); return () => this.listeners.delete(fn); },
    set(mode, err, dataSource) {
      this.mode = mode;
      this.lastError = err || null;
      if (dataSource !== undefined) this.dataSource = dataSource;
      this.listeners.forEach((fn) => { try { fn(this); } catch {} });
    },
  };
  window.API_STATUS = status;

  // ---------- Fetch with timeout ----------
  async function _fetch(url, opts = {}) {
    const ctrl = new AbortController();
    const t = setTimeout(() => ctrl.abort(), opts.timeout || TIMEOUT_MS);
    try {
      const res = await fetch(url, { ...opts, signal: ctrl.signal });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return await res.json();
    } finally {
      clearTimeout(t);
    }
  }

  // ---------- Public: call(path) ----------
  async function call(path) {
    const cached = _cache.get(path);
    if (cached && Date.now() - cached.at < CACHE_TTL_MS) {
      return { ...cached.body, _source: cached.source };
    }

    if (!API_BASE) {
      const body = _sampleFor(path);
      _cache.set(path, { at: Date.now(), body, source: "sample" });
      status.set("sample", null, "synthetic");
      return { ...body, _source: "sample" };
    }

    try {
      const body = await _fetch(`${API_BASE}${path}`);
      _cache.set(path, { at: Date.now(), body, source: "live" });
      // Trust the data_source field from the response when present
      const ds = body && body.data_source ? body.data_source : null;
      status.set("live", null, ds);
      return { ...body, _source: "live" };
    } catch (err) {
      const body = _sampleFor(path);
      _cache.set(path, { at: Date.now(), body, source: "sample" });
      status.set("sample", err.message || String(err), "synthetic");
      return { ...body, _source: "sample" };
    }
  }

  // ---------- Sample fallback (mirrors the FastAPI shape) ----------
  function _sampleFor(path) {
    const d = window.SAMPLE_DATA || {};
    const dates = d.dates || [];
    const regimes = d.regimes || [];
    const prices = d.prices || [];
    const stats = d.stats || {};
    const dist = d.regime_summary || {};
    const labels = d.regime_labels || {0: "Bull / low-vol", 1: "Neutral / chop", 2: "Bear / high-vol"};

    const lastIdx = Math.max(0, dates.length - 1);

    if (path === "/health") {
      return { status: "ok (sample)", version: "0.1.0", docs_url: "/docs" };
    }

    if (path === "/regime/latest" || path.startsWith("/regime/latest")) {
      return _sampleRegime(dates[lastIdx], regimes[lastIdx], prices[lastIdx], labels);
    }

    if (path === "/regime/distribution") {
      const total = regimes.length;
      return {
        as_of: dates[lastIdx],
        total_days: total,
        states: [0, 1, 2].map((s) => ({
          state: s,
          label: labels[s],
          n_days: regimes.filter((r) => r === s).length,
          pct: total ? +(100 * regimes.filter((r) => r === s).length / total).toFixed(1) : 0,
        })),
        data_source: "synthetic",
      };
    }

    if (path.startsWith("/regime/history")) {
      const params = new URLSearchParams(path.split("?")[1] || "");
      const limit = Math.min(parseInt(params.get("limit") || "365", 10), dates.length);
      const start = Math.max(0, dates.length - limit);
      const items = [];
      for (let i = start; i < dates.length; i++) {
        items.push(_sampleRegime(dates[i], regimes[i], prices[i], labels));
      }
      return { n: items.length, items, data_source: "synthetic" };
    }

    if (path.startsWith("/regime/implications/latest") || path === "/regime/implications/latest") {
      return _sampleImplications(regimes[lastIdx], dates[lastIdx], labels, dist);
    }

    if (path.startsWith("/regime/implications/")) {
      // /regime/implications/{date} — find that date in the sample
      const day = path.split("/").pop();
      const idx = dates.indexOf(day);
      if (idx < 0) return { detail: "not found", _status: 404 };
      return _sampleImplications(regimes[idx], dates[idx], labels, dist);
    }

    if (path.startsWith("/regime/")) {
      // /regime/{date}
      const day = path.split("/").pop();
      const idx = dates.indexOf(day);
      if (idx < 0) return { detail: "not found", _status: 404 };
      return _sampleRegime(dates[idx], regimes[idx], prices[idx], labels);
    }

    if (path === "/backtest/summary") {
      return {
        as_of: dates[lastIdx],
        strategies: [
          {name: "buy_hold", label: "Buy & Hold",
           sharpe: stats.buy_hold?.sharpe ?? 1.07,
           sortino: stats.buy_hold?.sortino ?? 1.47,
           cagr_pct: stats.buy_hold?.cagr_pct ?? 22.7,
           max_dd_pct: stats.buy_hold?.max_dd_pct ?? -37.3,
           calmar: stats.buy_hold?.calmar ?? 0.61,
           final_equity: stats.buy_hold?.final ?? 448751},
          {name: "baseline_trend", label: "Trend (unconditional)",
           sharpe: stats.baseline_trend?.sharpe ?? 0.27,
           sortino: stats.baseline_trend?.sortino ?? 0.30,
           cagr_pct: stats.baseline_trend?.cagr_pct ?? 3.3,
           max_dd_pct: stats.baseline_trend?.max_dd_pct ?? -44.0,
           calmar: stats.baseline_trend?.calmar ?? 0.07,
           final_equity: stats.baseline_trend?.final ?? 126531},
          {name: "regime_conditioned", label: "Trend + Regime Overlay",
           sharpe: stats.regime_conditioned?.sharpe ?? 0.62,
           sortino: stats.regime_conditioned?.sortino ?? 0.80,
           cagr_pct: stats.regime_conditioned?.cagr_pct ?? 7.5,
           max_dd_pct: stats.regime_conditioned?.max_dd_pct ?? -28.0,
           calmar: stats.regime_conditioned?.calmar ?? 0.27,
           final_equity: stats.regime_conditioned?.final ?? 169912},
        ],
        sharpe_improvement: 2.30,
        note: "Sample-data response. Deploy the FastAPI and set API_BASE in config.js to see live numbers.",
        data_source: "synthetic",
      };
    }

    return { detail: "no sample for path", path, _status: 404 };
  }

  function _sampleRegime(date, regime, price, labels) {
    const probs = {0: 0.05, 1: 0.05, 2: 0.05};
    probs[regime] = 0.85;
    if (regime === 0) probs[1] += 0.05;
    else if (regime === 1) { probs[0] += 0.025; probs[2] += 0.025; }
    else probs[1] += 0.05;
    const total = probs[0] + probs[1] + probs[2];
    return {
      date,
      regime,
      regime_label: labels[regime],
      probabilities: {
        "0": +(probs[0] / total).toFixed(4),
        "1": +(probs[1] / total).toFixed(4),
        "2": +(probs[2] / total).toFixed(4),
      },
      price,
      data_source: "synthetic",
    };
  }

  function _sampleImplications(regime, date, labels, dist) {
    const RISK = ["Risk-On", "Neutral", "Risk-Off"];
    const tiltCards = {
      0: [
        {asset_class: "Equity",    tilt: "Overweight",  magnitude: "Moderate", bps:  800, rationale: "Trend is your friend; realised vol below long-run average historically associates with positive forward 1-3 month equity returns."},
        {asset_class: "Duration",  tilt: "Underweight", magnitude: "Light",    bps: -300, rationale: "Risk-on episodes tend to coincide with mild rate drift higher; duration provides less diversification when equity-bond correlation turns positive."},
        {asset_class: "Credit",    tilt: "Overweight",  magnitude: "Moderate", bps:  400, rationale: "HY OAS compression is the textbook bull-regime carry trade; spread duration earns its keep here."},
        {asset_class: "Cash",      tilt: "Underweight", magnitude: "Light",    bps: -500, rationale: "Holding cash in a low-vol bull is opportunity cost."},
        {asset_class: "Vol hedge", tilt: "Underweight", magnitude: "Light",    bps: -400, rationale: "Long-vol carry is expensive when realised vol stays low; harvest only minimum tail protection."},
      ],
      1: [
        {asset_class: "Equity",    tilt: "Neutral",     magnitude: "Light",    bps:    0, rationale: "No edge in either direction; preserve dry powder for the next regime change."},
        {asset_class: "Duration",  tilt: "Neutral",     magnitude: "Light",    bps:  100, rationale: "Slight tilt to duration for diversification, but no view; watch the curve for the next signal."},
        {asset_class: "Credit",    tilt: "Neutral",     magnitude: "Light",    bps:    0, rationale: "Spread carry roughly fair; avoid concentration."},
        {asset_class: "Cash",      tilt: "Overweight",  magnitude: "Light",    bps:  300, rationale: "Hold a buffer to deploy decisively when the regime resolves; money-market yields make this less costly than usual."},
        {asset_class: "Vol hedge", tilt: "Neutral",     magnitude: "Light",    bps:  200, rationale: "Add a modest long-vol position — convexity is most valuable right before a regime shift, not during one."},
      ],
      2: [
        {asset_class: "Equity",    tilt: "Underweight", magnitude: "Strong",   bps: -1200, rationale: "High-vol bear regimes historically deliver negative median equity returns; the asymmetry of large drawdowns argues for reducing exposure decisively, not partially."},
        {asset_class: "Duration",  tilt: "Overweight",  magnitude: "Strong",   bps:  700, rationale: "Flight-to-quality bid for US Treasuries; equity-bond correlation typically reverts to negative in risk-off, restoring duration's diversification value."},
        {asset_class: "Credit",    tilt: "Underweight", magnitude: "Strong",   bps: -600, rationale: "Spreads widen and default risk repricing accelerates; reduce HY and lower-quality IG exposure first."},
        {asset_class: "Cash",      tilt: "Overweight",  magnitude: "Moderate", bps:  600, rationale: "Optionality to deploy into dislocations is worth more in a high-vol regime than at any other time."},
        {asset_class: "Vol hedge", tilt: "Overweight",  magnitude: "Moderate", bps:  500, rationale: "VIX term structure may be in backwardation; out-of-the-money put spreads or VIX call calendars to hedge the left tail."},
      ],
    };
    const descriptions = {
      0: "Low realised volatility, positive trend, compressing credit spreads. Macro is supportive: curve term-premia stable, dollar range-bound.",
      1: "Mixed signals. Volatility elevated relative to bull but not extreme; trend is unclear. Macro readings disagree across indicators.",
      2: "High realised and implied volatility, negative momentum, widening credit spreads. Flight-to-quality flows into Treasuries. Dollar typically bid.",
    };
    const headlines = {
      0: "Current regime: Risk-On (bull / low-vol), 92% model confidence. Suggested tilt: overweight equity and credit, underweight duration and cash.",
      1: "Current regime: Neutral (chop), 90% model confidence. Suggested tilt: stay close to benchmark, hold a cash buffer, add a modest vol hedge.",
      2: "Current regime: Risk-Off (bear / high-vol), 92% model confidence. Suggested tilt: reduce equity and credit, increase duration and cash, add vol protection.",
    };
    const histStats = {
      0: {n_episodes: 4, total_days: 1185, avg_duration_days: 296.2, median_daily_return_pct: 0.078, annualized_return_pct: 21.4, annualized_vol_pct: 11.9, hit_rate_pct: 56.8, max_drawdown_pct: -9.4, sample_basis: "SPY adj close (synthetic)"},
      1: {n_episodes: 3, total_days: 369,  avg_duration_days: 123.0, median_daily_return_pct: 0.012, annualized_return_pct: 3.6,  annualized_vol_pct: 19.8, hit_rate_pct: 51.2, max_drawdown_pct: -14.7, sample_basis: "SPY adj close (synthetic)"},
      2: {n_episodes: 3, total_days: 296,  avg_duration_days: 98.7,  median_daily_return_pct: -0.085, annualized_return_pct: -19.2, annualized_vol_pct: 41.2, hit_rate_pct: 44.3, max_drawdown_pct: -34.1, sample_basis: "SPY adj close (synthetic)"},
    };
    const runDays = {0: 38, 1: 12, 2: 21}[regime];
    const probs = {0: 0.05, 1: 0.05, 2: 0.05};
    probs[regime] = 0.85;
    if (regime === 0) probs[1] += 0.05;
    else if (regime === 1) { probs[0] += 0.025; probs[2] += 0.025; }
    else probs[1] += 0.05;
    const total = probs[0] + probs[1] + probs[2];

    return {
      as_of: date,
      regime,
      regime_label: labels[regime],
      risk_profile: RISK[regime],
      description: descriptions[regime],
      confidence: 0.92,
      confidence_label: "High",
      probabilities: {
        "0": +(probs[0] / total).toFixed(4),
        "1": +(probs[1] / total).toFixed(4),
        "2": +(probs[2] / total).toFixed(4),
      },
      days_in_current_run: runDays,
      historical: histStats[regime],
      allocation: tiltCards[regime],
      headline: headlines[regime],
      alternative: null,
      caveats: [
        "Regime classifications are a model output, not a forecast. They describe what regime the data currently resembles, not what regime will hold next week.",
        "Historical stats summarise past instances of this regime in the training window; future occurrences may differ.",
        "Allocation tilts are illustrative and assume a 60/30/10 (equity/duration/credit) benchmark with cash and vol-hedge sleeves available.",
      ],
      data_source: "synthetic",
    };
  }

  // ---------- Public API ----------
  window.APIClient = { call, status };
})();
