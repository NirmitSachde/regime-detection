# Research Memo — Adaptive Market Regime Detection

> **Status:** Methodology + scaffolding in place. Results section populated
> after the first end-to-end pipeline run with live ingestion.

## 1. Problem

Static trading rules over-fit to whichever regime dominated the training data.
A 50/200 MA crossover earns money in trending markets and bleeds in chop. Risk
parameters (position size, stop distance, holding period) should be a function
of the prevailing regime, not constants.

## 2. Hypothesis

A regime classifier trained on macro + price features can identify distinct,
persistent market states. Conditioning a trend-following strategy on those
states meaningfully improves risk-adjusted returns out-of-sample.

**Success criterion:** OOS Sharpe of the regime-aware strategy ≥ 1.3× the same
strategy run unconditionally, after realistic transaction costs and across a
walk-forward evaluation.

## 3. Data

| Source | What | Access |
|---|---|---|
| yfinance | Daily OHLCV for SPY, sector ETFs, VIX | Python lib, no key |
| FRED | DGS10, DGS2, T10Y2Y, VIXCLS, DTWEXBGS, CPIAUCSL, UNRATE, FEDFUNDS, BAMLH0A0HYM2, DCOILWTICO | REST, free key |

History: 2010-01-01 onward (~3,800 trading days × 16 tickers ≈ 60k rows).
Macro panel forward-filled per series.

## 4. Methodology

### 4.1 Feature engineering
- **Price-derived (per ticker, per day)**: log returns at 1/5/21d horizons; RSI(14); MA(20, 50, 200); ATR(14); Bollinger %B(20, 2); volume Z-score(21d); realized vol(21d, 63d, annualised).
- **Macro (per day)**: VIX level + 5d change; DXY 5d change; yield curve slope + 21d change; HY OAS + 21d change; CPI; unemployment; Fed funds rate.

All features lagged by one day to avoid look-ahead. Enforced by
`tests/unit/test_look_ahead_bias.py` (empirical perturbation test).

### 4.2 Regime model — Gaussian HMM
- **Input:** macro-only feature subset (VIX, VIX_chg_5d, DXY_chg_5d, yield curve slope, HY OAS).
- **K selection:** fit K ∈ {3, 4}, pick by BIC.
- **Covariance:** diagonal (avoids overfitting with limited training data).
- **Persistence:** transition matrix + Gaussian params pickled, logged to MLflow.

### 4.3 Supervised classifier — LightGBM
- **Input:** full `mart_features` (per-ticker price + macro features).
- **Labels:** HMM-assigned regime for each date, joined onto every ticker-day.
- **CV:** `TimeSeriesSplit(n_splits=5)`, expanding window, early stopping on val fold.
- **Metric:** macro-F1 (handles regime imbalance better than accuracy).

### 4.4 Backtest
- **Engine:** vectorbt.
- **Strategies:**
  - A — baseline 50/200 MA crossover, long-only
  - B — same crossover with position size = `base × regime_multiplier`
  - C — Bollinger %B mean-reversion, only enabled in low-vol / range regimes
- **Costs:** linear slippage (base 1 bp + size impact), $0 commission (Alpaca paper), 50 bp annualised borrow on shorts.
- **Risk:** Sharpe, Sortino, max DD, Calmar, hit rate, exposure, turnover.
- **Significance:** 1000-resample bootstrap CI on OOS Sharpe; deflated Sharpe per López de Prado as a stretch.

## 5. Results

*Populated after first end-to-end run. Template:*

| Strategy | Sharpe | Sortino | CAGR | Max DD | Calmar | Hit | Turnover |
|---|---|---|---|---|---|---|---|
| Baseline trend | – | – | – | – | – | – | – |
| Regime-cond trend | – | – | – | – | – | – | – |
| Regime-cond mean-rev | – | – | – | – | – | – | – |

OOS bootstrap Sharpe 95% CI: `[lo, hi]`.

Cost sensitivity:

| Slippage (bps) | 0 | 1 | 5 | 10 |
|---|---|---|---|---|
| Strategy B Sharpe | – | – | – | – |

## 6. Limitations + what didn't work

- HMM state ordering is non-deterministic across re-fits — handled by mapping
  states by mean realized vol post-fit, but downstream multipliers must be
  re-validated each retraining.
- Daily-only horizon. Intraday regime shifts are invisible.
- No transaction-cost calibration to actual fills — slippage params are
  conservative defaults, not measured.
- Universe is US-equity-only.

## 7. Reproducibility

```bash
git clone <repo>
cd regime-detection
echo "FRED_API_KEY=your_key" > .env
make setup && make up
make ingest && make dbt-build && make train && make backtest
```

Random seed is fixed in `config.py` (`RANDOM_SEED=42`); all stochastic
components (HMM, LightGBM, bootstrap) consume it.
