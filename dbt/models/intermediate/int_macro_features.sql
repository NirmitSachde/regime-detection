{{ config(materialized='table') }}

with wide as (
    select
        observation_date as feature_date,
        max(case when series_id = 'DGS10'        then value end) as ust_10y,
        max(case when series_id = 'DGS2'         then value end) as ust_2y,
        max(case when series_id = 'T10Y2Y'      then value end) as yc_10y2y,
        max(case when series_id = 'VIXCLS'       then value end) as vix,
        max(case when series_id = 'DTWEXBGS'     then value end) as dxy,
        max(case when series_id = 'CPIAUCSL'     then value end) as cpi,
        max(case when series_id = 'UNRATE'       then value end) as unemp,
        max(case when series_id = 'FEDFUNDS'     then value end) as fed_funds,
        max(case when series_id = 'BAMLH0A0HYM2' then value end) as hy_oas,
        max(case when series_id = 'DCOILWTICO'  then value end) as wti
    from {{ ref('stg_macro') }}
    group by 1
),

-- Forward-fill macro values onto every calendar date (most series are weekly/monthly)
ffilled as (
    select
        feature_date,
        last_value(ust_10y    ignore nulls) over w as ust_10y,
        last_value(ust_2y     ignore nulls) over w as ust_2y,
        last_value(yc_10y2y   ignore nulls) over w as yc_10y2y,
        last_value(vix        ignore nulls) over w as vix,
        last_value(dxy        ignore nulls) over w as dxy,
        last_value(cpi        ignore nulls) over w as cpi,
        last_value(unemp      ignore nulls) over w as unemp,
        last_value(fed_funds  ignore nulls) over w as fed_funds,
        last_value(hy_oas     ignore nulls) over w as hy_oas,
        last_value(wti        ignore nulls) over w as wti
    from wide
    window w as (order by feature_date rows between unbounded preceding and current row)
)

select
    feature_date,
    ust_10y,
    ust_2y,
    yc_10y2y,
    vix,
    dxy,
    cpi,
    unemp,
    fed_funds,
    hy_oas,
    wti,
    vix - lag(vix, 5)  over (order by feature_date) as vix_chg_5d,
    dxy - lag(dxy, 5)  over (order by feature_date) as dxy_chg_5d,
    yc_10y2y - lag(yc_10y2y, 21) over (order by feature_date) as yc_chg_21d,
    case when lag(hy_oas, 21) over (order by feature_date) is not null
         then hy_oas - lag(hy_oas, 21) over (order by feature_date)
    end as hy_oas_chg_21d
from ffilled
