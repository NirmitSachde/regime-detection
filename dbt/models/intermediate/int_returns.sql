{{ config(materialized='table') }}

with px as (
    select
        ticker,
        trade_date,
        adj_close,
        lag(adj_close, 1)  over w as lag_1,
        lag(adj_close, 5)  over w as lag_5,
        lag(adj_close, 21) over w as lag_21
    from {{ ref('stg_prices') }}
    window w as (partition by ticker order by trade_date)
)
select
    ticker,
    trade_date,
    adj_close,
    case when lag_1  is not null and lag_1  > 0
         then ln(adj_close / lag_1)  end as log_ret_1d,
    case when lag_5  is not null and lag_5  > 0
         then ln(adj_close / lag_5)  end as log_ret_5d,
    case when lag_21 is not null and lag_21 > 0
         then ln(adj_close / lag_21) end as log_ret_21d
from px
