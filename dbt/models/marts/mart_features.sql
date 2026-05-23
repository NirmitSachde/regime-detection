{{ config(materialized='table') }}

-- One row per (ticker, trade_date). Wide feature table used by ML training.
-- All features are LAGGED appropriately upstream — none use forward information.

select
    t.ticker,
    t.trade_date,
    t.adj_close,
    t.volume,
    -- Returns (the *current* day's return — careful: target features should lead this)
    r.log_ret_1d,
    r.log_ret_5d,
    r.log_ret_21d,
    -- Technicals (all backward-looking by construction)
    t.ma_20,
    t.ma_50,
    t.ma_200,
    t.atr_14,
    t.rsi_14,
    t.bb_pctb_20,
    t.volume_z_21,
    -- Realized vol
    v.realized_vol_21d,
    v.realized_vol_63d,
    -- Macro (joined on the most recent macro observation strictly before trade_date close)
    m.ust_10y,
    m.ust_2y,
    m.yc_10y2y,
    m.vix,
    m.dxy,
    m.cpi,
    m.unemp,
    m.fed_funds,
    m.hy_oas,
    m.wti,
    m.vix_chg_5d,
    m.dxy_chg_5d,
    m.yc_chg_21d,
    m.hy_oas_chg_21d
from {{ ref('int_technicals') }} t
left join {{ ref('int_returns') }} r
    on r.ticker = t.ticker
   and r.trade_date = t.trade_date
left join {{ ref('int_realized_vol') }} v
    on v.ticker = t.ticker
   and v.trade_date = t.trade_date
left join {{ ref('int_macro_features') }} m
    on m.feature_date = t.trade_date
