{{ config(materialized='table') }}

select
    ticker,
    trade_date,
    log_ret_1d,
    stddev_pop(log_ret_1d) over (
        partition by ticker order by trade_date
        rows between 20 preceding and current row
    ) * sqrt(252) as realized_vol_21d,
    stddev_pop(log_ret_1d) over (
        partition by ticker order by trade_date
        rows between 62 preceding and current row
    ) * sqrt(252) as realized_vol_63d
from {{ ref('int_returns') }}
