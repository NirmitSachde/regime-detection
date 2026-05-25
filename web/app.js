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
    bg:        "#0a0a0a",
    surface:   "#141414",
    border:    "#232323",
    border_2:  "#2e2e2e",
    text:      "#ededed",
    text_2:    "#d4d4d4",
    muted:     "#8a8a8a",
    dim:       "#5a5a5a",
    accent:    "#d4a017",
    positive:  "#4a9d6e",
    negative:  "#c14040",
    neutral:   "#a8a29e",
    info:      "#5c8df7",
  };
  const REGIME_C = {
    0: C.positive,  // bull
    1: C.neutral,   // chop
    2: C.negative,  // bear
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
      font: { family: "Inter, system-ui", color: C.text_2, size: 11.5 },
      margin: { l: 60, r: 28, t: 24, b: 44 },
      xaxis: {
        gridcolor: C.border,
        zerolinecolor: C.border,
        linecolor: C.border_2,
        tickfont: { color: C.muted, family: "JetBrains Mono, monospace", size: 10.5 },
      },
      yaxis: {
        gridcolor: C.border,
        zerolinecolor: C.border,
        linecolor: C.border_2,
        tickfont: { color: C.muted, family: "JetBrains Mono, monospace", size: 10.5 },
      },
      legend: {
        orientation: "h",
        yanchor: "bottom", y: 1.02,
        xanchor: "right", x: 1,
        font: { color: C.muted, size: 10.5, family: "JetBrains Mono, monospace" },
        bgcolor: "rgba(0,0,0,0)",
      },
      hoverlabel: {
        bgcolor: C.surface,
        bordercolor: C.border_2,
        font: { color: C.text, family: "JetBrains Mono, monospace", size: 11.5 },
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
        line: { color: C.text, width: 1.8 },
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
        line: { color: C.accent, width: 1.2, dash: "dot" },
        hovertemplate: "MA50 $%{y:.2f}<extra></extra>",
      });
      traces.push({
        x: data.dates,
        y: data.ma200,
        type: "scatter",
        mode: "lines",
        name: "MA(200)",
        line: { color: C.info, width: 1.2, dash: "dot" },
        hovertemplate: "MA200 $%{y:.2f}<extra></extra>",
      });
    }

    const layout = baseLayout({
      shapes: (overlay === "regime" || overlay === "both") ? regimeShapes() : [],
      height: 400,
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
        line: { color: C.negative, width: 1.6 },
        hovertemplate: "%{x|%b %Y} · $%{y:,.0f}<extra>trend</extra>",
      });
    }
    if (strat === "all" || strat === "compare" || strat === "regime") {
      traces.push({
        x: data.dates, y: data.regime_eq,
        type: "scatter", mode: "lines",
        name: "trend + regime overlay",
        line: { color: C.positive, width: 2 },
        fill: strat === "regime" ? "tozeroy" : undefined,
        fillcolor: strat === "regime" ? "rgba(74,157,110,0.06)" : undefined,
        hovertemplate: "%{x|%b %Y} · $%{y:,.0f}<extra>regime</extra>",
      });
    }
    if (strat === "all") {
      traces.unshift({
        x: data.dates, y: data.buy_hold_eq,
        type: "scatter", mode: "lines",
        name: "buy &amp; hold",
        line: { color: C.muted, width: 1.2, dash: "dash" },
        hovertemplate: "%{x|%b %Y} · $%{y:,.0f}<extra>B&amp;H</extra>",
      });
    }

    const layout = baseLayout({
      height: 400,
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
    // Config-driven links + nav active state handled by nav.js.
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
