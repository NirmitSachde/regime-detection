{{ config(materialized='view') }}

select
    upper(series_id)        as series_id,
    cast(date as date)      as observation_date,
    cast(value as double)   as value
from {{ source('raw', 'macro') }}
where value is not null
