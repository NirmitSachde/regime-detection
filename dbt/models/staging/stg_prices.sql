{{ config(materialized='view') }}

with src as (
    select
        upper(ticker)            as ticker,
        cast(date as date)       as trade_date,
        cast(open as double)     as open,
        cast(high as double)     as high,
        cast(low as double)      as low,
        cast(close as double)    as close,
        cast(adj_close as double) as adj_close,
        cast(volume as double)   as volume
    from {{ source('raw', 'prices') }}
)
select *
from src
where open > 0
  and high > 0
  and low > 0
  and close > 0
  and adj_close > 0
  and volume >= 0
