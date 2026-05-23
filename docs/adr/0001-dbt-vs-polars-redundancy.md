# ADR-0001 — Why dbt and Polars build features in parallel

**Status:** Accepted
**Date:** 2026-05-23

## Context

Two consumers need feature tables:

1. The **analytics surface** (dbt docs, Streamlit research page, ad-hoc SQL) wants
   a versioned, tested, lineage-tracked warehouse model.
2. The **training loop** wants a fast, columnar, in-memory frame it can hand
   directly to hmmlearn / LightGBM without round-tripping through SQL or pickle.

Doing both from one place forces a compromise. SQL is verbose for math-heavy
feature engineering and slow to iterate on outside the warehouse. Polars is
poor at materializing audited tables for BI consumers.

## Decision

Build the same logic twice:

- **`dbt/models/`** is the source of truth for the **warehouse**. Tests, docs,
  freshness, accepted-range checks, and downstream BI run here.
- **`src/regime/transform/features.py`** is the source of truth for **training**.
  Identical math, expressed in Polars; consumed by `models/train.py`.

The look-ahead-bias guard test in `tests/unit/test_look_ahead_bias.py`
exercises the Polars path. Equivalence between the two paths is checked by
an integration test (Phase 5+) that joins the dbt mart against the Polars
output and asserts they agree on a sampled date range.

## Consequences

**Cost:** Logic exists in two places. Drift is the risk.

**Mitigation:** A single integration test gates them. When a feature changes
in one, the test fails until the other is updated. The redundancy is cheap
because the features are simple windowed math; we are not duplicating a
complex DAG.

**Alternative considered:** Run dbt during training and read the materialized
table back into Polars. Rejected because training feedback loops should not
require a 30s dbt-build invalidation cycle.
