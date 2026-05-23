{{ config(materialized='table') }}

-- One row per trade_date. Macro panel for HMM input (no ticker dimension).

select *
from {{ ref('int_macro_features') }}
