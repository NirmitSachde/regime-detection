{{ config(materialized='table') }}

-- Rolling technicals computed in SQL. RSI uses a simple-MA gain/loss variant
-- which is a fine first cut; refine to Wilder smoothing in a later iteration.

with px as (
    select
        ticker,
        trade_date,
        open, high, low, close, adj_close, volume,
        lag(adj_close) over w as prev_close
    from {{ ref('stg_prices') }}
    window w as (partition by ticker order by trade_date)
),

ranged as (
    select
        ticker,
        trade_date,
        adj_close,
        volume,
        prev_close,
        greatest(high - low,
                 abs(high - prev_close),
                 abs(low - prev_close)) as tr,
        case when adj_close - prev_close > 0
             then adj_close - prev_close else 0 end as gain,
        case when adj_close - prev_close < 0
             then prev_close - adj_close else 0 end as loss,
        avg(adj_close) over (
            partition by ticker order by trade_date
            rows between 49 preceding and current row
        ) as ma_50,
        avg(adj_close) over (
            partition by ticker order by trade_date
            rows between 199 preceding and current row
        ) as ma_200,
        avg(adj_close) over (
            partition by ticker order by trade_date
            rows between 19 preceding and current row
        ) as ma_20,
        stddev_samp(adj_close) over (
            partition by ticker order by trade_date
            rows between 19 preceding and current row
        ) as sd_20
    from px
)

select
    ticker,
    trade_date,
    adj_close,
    volume,
    ma_20,
    ma_50,
    ma_200,
    -- ATR(14)
    avg(tr) over (
        partition by ticker order by trade_date
        rows between 13 preceding and current row
    ) as atr_14,
    -- RSI(14)
    case
        when avg(loss) over (
                partition by ticker order by trade_date
                rows between 13 preceding and current row) = 0
        then 100.0
        else 100.0 - 100.0 / (1.0 + (
            avg(gain) over (
                partition by ticker order by trade_date
                rows between 13 preceding and current row)
            /
            nullif(avg(loss) over (
                partition by ticker order by trade_date
                rows between 13 preceding and current row), 0)
        ))
    end as rsi_14,
    -- Bollinger %B (20, 2)
    case when sd_20 is not null and sd_20 > 0
         then (adj_close - (ma_20 - 2 * sd_20))
              / nullif((ma_20 + 2 * sd_20) - (ma_20 - 2 * sd_20), 0)
    end as bb_pctb_20,
    -- Volume Z-score (21d)
    case
        when stddev_samp(volume) over (
                partition by ticker order by trade_date
                rows between 20 preceding and current row) > 0
        then (volume - avg(volume) over (
                partition by ticker order by trade_date
                rows between 20 preceding and current row))
             / stddev_samp(volume) over (
                partition by ticker order by trade_date
                rows between 20 preceding and current row)
    end as volume_z_21
from ranged
