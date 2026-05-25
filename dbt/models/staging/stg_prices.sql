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
-- Filter out NaN rows explicitly. DuckDB treats NaN > 0 as TRUE, so a
-- bare `adj_close > 0` filter would let NaN through and poison every
-- downstream rolling window. yfinance returns NaN for pre-listing dates
-- of younger ETFs (XLC launched 2018-06; SPDR sector ETFs vary).
select *
from src
where open > 0     and not isnan(open)
  and high > 0     and not isnan(high)
  and low > 0      and not isnan(low)
  and close > 0    and not isnan(close)
  and adj_close > 0 and not isnan(adj_close)
  and volume >= 0  and not isnan(volume)
