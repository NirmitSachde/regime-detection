/* ============================================================
   Adaptive Market Regime Detection — demo charts + interactions
   Reads window.SAMPLE_DATA injected by sample_data.js.
   ============================================================ */
(function () {
  "use strict";

  const data = window.SAMPLE_DATA;
  if (!data) {
    console.error("SAMPLE_DATA missing");
    return;
  }

  // ---------- Color tokens (mirror styles.css :root) ----------
  const C = {
    bg: "#07091a",
    surface: "#131938",
    border: "#1e264f",
    text: "#e6e9f5",
    muted: "#8892b8",
    cyan: "#00d9ff",
    coral: "#ff7a59",
    mint: "#00ff9c",
    amber: "#ffc857",
    violet: "#b794ff",
    red: "#ff4757",
    grey: "#5b6488",
  };
  const REGIME_C = {
    0: C.mint,   // bull
    1: C.amber,  // neutral
    2: C.red,    // bear
  };
  const REGIME_LABEL = data.regime_labels;

  // ---------- Stat cards ----------
  function fmtPct(x) { return (x > 0 ? "+" : "") + x.toFixed(1) + "%"; }

  function fillStats() {
    const s = data.stats;
    document.getElementById("stat-bh-sharpe").textContent      = s.buy_hold.sharpe;
    document.getElementById("stat-bh-mdd").textContent         = s.buy_hold.max_dd_pct + "%";
    document.getElementById("stat-base-sharpe").textContent    = s.baseline_trend.sharpe;
    document.getElementById("stat-base-mdd").textContent       = s.baseline_trend.max_dd_pct + "%";
    document.getElementById("stat-regime-sharpe").textContent  = s.regime_conditioned.sharpe;
    document.getElementById("stat-regime-mdd").textContent     = s.regime_conditioned.max_dd_pct + "%";
    const improvement = s.regime_conditioned.sharpe / s.baseline_trend.sharpe;
    document.getElementById("stat-improvement").textContent =
      isFinite(improvement) ? improvement.toFixed(2) + "×" : "n/a";
  }

  // ---------- Plotly layout helpers ----------
  function baseLayout(extra) {
    return Object.assign({
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor:  "rgba(0,0,0,0)",
      font: { family: "Inter, system-ui", color: C.text, size: 12 },
      margin: { l: 56, r: 24, t: 24, b: 40 },
      xaxis: {
        gridcolor: C.border,
        zerolinecolor: C.border,
        linecolor: C.border,
        tickfont: { color: C.muted, family: "JetBrains Mono, monospace", size: 11 },
      },
      yaxis: {
        gridcolor: C.border,
        zerolinecolor: C.border,
        linecolor: C.border,
        tickfont: { color: C.muted, family: "JetBrains Mono, monospace", size: 11 },
      },
      legend: {
        orientation: "h",
        yanchor: "bottom", y: 1.02,
        xanchor: "right", x: 1,
        font: { color: C.muted, size: 11 },
        bgcolor: "rgba(0,0,0,0)",
      },
      hoverlabel: {
        bgcolor: C.surface,
        bordercolor: C.border,
        font: { color: C.text, family: "JetBrains Mono, monospace", size: 12 },
      },
      hovermode: "x unified",
      showlegend: true,
    }, extra || {});
  }

  const config = { displayModeBar: false, responsive: true };

  // ---------- Build regime shapes (vertical bands behind the price line) ----------
  function regimeShapes() {
    const shapes = [];
    let curState = data.regimes[0];
    let runStart = data.dates[0];
    for (let i = 1; i < data.regimes.length; i++) {
      if (data.regimes[i] !== curState || i === data.regimes.length - 1) {
        shapes.push({
          type: "rect",
          xref: "x",
          yref: "paper",
          x0: runStart,
          x1: data.dates[i],
          y0: 0, y1: 1,
          fillcolor: REGIME_C[curState],
          opacity: 0.10,
          line: { width: 0 },
          layer: "below",
        });
        curState = data.regimes[i];
        runStart = data.dates[i];
      }
    }
    return shapes;
  }

  // ---------- Chart 1: Price + regime ----------
  function renderPriceChart(overlay = "regime") {
    const traces = [
      {
        x: data.dates,
        y: data.prices,
        type: "scatter",
        mode: "lines",
        name: "SPY (synthetic)",
        line: { color: C.cyan, width: 2.4 },
        hovertemplate: "%{x|%b %Y} · $%{y:.2f}<extra></extra>",
      },
    ];

    if (overlay === "ma" || overlay === "both") {
      traces.push({
        x: data.dates,
        y: data.ma50,
        type: "scatter",
        mode: "lines",
        name: "MA(50)",
        line: { color: C.amber, width: 1.4, dash: "dot" },
        hovertemplate: "MA50 $%{y:.2f}<extra></extra>",
      });
      traces.push({
        x: data.dates,
        y: data.ma200,
        type: "scatter",
        mode: "lines",
        name: "MA(200)",
        line: { color: C.coral, width: 1.4, dash: "dot" },
        hovertemplate: "MA200 $%{y:.2f}<extra></extra>",
      });
    }

    const layout = baseLayout({
      shapes: (overlay === "regime" || overlay === "both") ? regimeShapes() : [],
      height: 380,
      yaxis: Object.assign({}, baseLayout().yaxis, {
        tickprefix: "$",
        tickformat: ",.0f",
      }),
    });

    Plotly.react("chart-price", traces, layout, config);
  }

  // ---------- Chart 2: Equity curves ----------
  function renderEquityChart(strat = "all") {
    const traces = [];
    if (strat === "all" || strat === "compare") {
      traces.push({
        x: data.dates, y: data.baseline_eq,
        type: "scatter", mode: "lines",
        name: "trend (unconditional)",
        line: { color: C.coral, width: 2 },
        hovertemplate: "%{x|%b %Y} · $%{y:,.0f}<extra>trend</extra>",
      });
    }
    if (strat === "all" || strat === "compare" || strat === "regime") {
      traces.push({
        x: data.dates, y: data.regime_eq,
        type: "scatter", mode: "lines",
        name: "trend + regime overlay",
        line: { color: C.mint, width: 2.6 },
        fill: strat === "regime" ? "tozeroy" : undefined,
        fillcolor: strat === "regime" ? "rgba(0,255,156,0.08)" : undefined,
        hovertemplate: "%{x|%b %Y} · $%{y:,.0f}<extra>regime</extra>",
      });
    }
    if (strat === "all") {
      traces.unshift({
        x: data.dates, y: data.buy_hold_eq,
        type: "scatter", mode: "lines",
        name: "buy &amp; hold",
        line: { color: C.muted, width: 1.5, dash: "dash" },
        hovertemplate: "%{x|%b %Y} · $%{y:,.0f}<extra>B&amp;H</extra>",
      });
    }

    const layout = baseLayout({
      height: 380,
      yaxis: Object.assign({}, baseLayout().yaxis, {
        tickprefix: "$",
        tickformat: ",.0f",
      }),
    });

    Plotly.react("chart-equity", traces, layout, config);
  }

  // ---------- Toggle wiring ----------
  function wireToggle(groupId, attr, onChange) {
    const group = document.getElementById(groupId);
    if (!group) return;
    group.addEventListener("click", (e) => {
      const btn = e.target.closest("button");
      if (!btn) return;
      group.querySelectorAll("button").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      onChange(btn.dataset[attr]);
    });
  }

  // ---------- Reveal on scroll ----------
  function wireReveal() {
    const els = document.querySelectorAll(".reveal");
    if (!("IntersectionObserver" in window)) {
      els.forEach(el => el.classList.add("in"));
      return;
    }
    const io = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add("in");
          io.unobserve(entry.target);
        }
      });
    }, { threshold: 0.08, rootMargin: "0px 0px -40px 0px" });
    els.forEach(el => io.observe(el));
  }

  // ---------- Init ----------
  function init() {
    fillStats();
    renderPriceChart("regime");
    renderEquityChart("all");
    wireToggle("overlay-toggle",  "overlay", renderPriceChart);
    wireToggle("strategy-toggle", "strat",   renderEquityChart);
    wireReveal();

    // Re-render on resize (Plotly responsive flag handles most cases, but
    // we force a relayout on orientation change for mobile).
    window.addEventListener("resize", () => {
      Plotly.Plots.resize("chart-price");
      Plotly.Plots.resize("chart-equity");
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
