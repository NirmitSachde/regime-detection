{{ config(materialized='table') }}

-- One row per trade_date. Macro panel for HMM input (no ticker dimension).
-- Joins SPY return + realized-vol features so the HMM can distinguish
-- direction (high-vol going up vs high-vol going down).

select *
from {{ ref('int_macro_with_spy') }}
