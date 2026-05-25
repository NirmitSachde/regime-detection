{{ config(materialized='table') }}

-- Macro features joined with SPY return signals so the HMM can distinguish
-- "high-vol going up" from "high-vol going down". Without this, the model
-- can't see direction — it groups COVID-crash and post-COVID-recovery into
-- the same "high vol" cluster.

with macro as (
    select * from {{ ref('int_macro_features') }}
),

spy as (
    select
        trade_date,
        adj_close as spy_close,
        log_ret_1d as spy_ret_1d
    from {{ ref('int_returns') }}
    where ticker = 'SPY'
),

spy_aug as (
    select
        trade_date,
        spy_close,
        spy_ret_1d,
        -- 21d cumulative log-return: trend signal, captures "going up vs down"
        ln(spy_close / nullif(lag(spy_close, 21) over (order by trade_date), 0)) as spy_ret_21d,
        -- 63d cumulative log-return: medium-term momentum
        ln(spy_close / nullif(lag(spy_close, 63) over (order by trade_date), 0)) as spy_ret_63d,
        -- 21d realized vol on SPY itself
        stddev_pop(spy_ret_1d) over (
            order by trade_date rows between 20 preceding and current row
        ) * sqrt(252) as spy_rv_21d
    from spy
),

joined as (
    select
        m.*,
        s.spy_ret_21d,
        s.spy_ret_63d,
        s.spy_rv_21d
    from macro m
    left join spy_aug s on s.trade_date = m.feature_date
)

select * from joined
where spy_ret_21d is not null
  and not isnan(spy_ret_21d)
  and spy_rv_21d is not null
  and not isnan(spy_rv_21d)
