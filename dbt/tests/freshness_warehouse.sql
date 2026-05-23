-- Singular test: warehouse must be no more than N business days behind today.
-- Returns rows when the test FAILS.

with latest as (
    select max(trade_date) as max_dt from {{ ref('stg_prices') }}
)
select
    max_dt,
    current_date as today,
    date_diff('day', max_dt, current_date) as days_behind
from latest
where date_diff('day', max_dt, current_date) > 5  -- generous; 2 biz days = ~5 calendar in long weekends
