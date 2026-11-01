# Feast Feature Store for PaySim

A feature store built with [Feast](https://feast.dev) that serves the same fraud-detection features consistently for model training (offline) and live lookups (online), using the PaySim mobile-money dataset.

**Dataset:** [PaySim Mobile Money Fraud Dataset (Kaggle)](https://www.kaggle.com/datasets/ealaxi/paysim1)

## What it demonstrates

- **Feature store setup:** entity, source and feature view definitions, with a Parquet offline store and a SQLite online store.
- **Feature definition:** four per-account features computed from the past 24 hours, with the reasoning documented.
- **Point-in-time retrieval:** a training dataset built with `get_historical_features`, with no leakage from future data.
- **Online/offline consistency:** features materialized to SQLite, and online lookups checked against the offline values.

## Repository structure

```
.
├── feast_feature_store_paysim.ipynb   # End-to-end notebook
├── note.md                            # Design choices, event time conversion, trade-offs
├── README.md
└── feature_repo/
    ├── feature_store.yaml             # Offline (file) and online (SQLite) store config
    ├── entities.py                    # Entity: account (join key: nameDest)
    ├── features.py                    # Parquet source and feature view
    └── data/
        └── customer_features.parquet  # Precomputed features (offline store)
```

## Design summary

| Item | Choice |
|---|---|
| Entity | `account`, join key `nameDest` (recipient account) |
| Event time | `step` (hours) converted to `event_timestamp` from a 2025-01-01 base date |
| Features | `txn_count_24h`, `avg_amount_24h`, `transfer_count_24h`, `cashout_count_24h` |
| Leakage prevention | Rolling window with `closed="left"`, label and post-transaction balance columns excluded |
| Offline store | Parquet |
| Online store | SQLite |
| Data subset | 20,169 accounts (all 8,169 accounts that received fraud, plus 12,000 sampled normal accounts), 218,922 rows |

See [`note.md`](note.md) for the full reasoning, results and limitations.

## Key results

- Point-in-time training dataset: 199,937 rows; zero difference from the independently computed features.
- Online vs offline: 20,169 accounts compared; zero differences across all 4 features.
- Independent leakage test on 1,000 sampled rows: no mismatches.

## How to run

1. Open the notebook in Kaggle with the PaySim dataset attached (or adjust the CSV path in the first cells).
2. Install the dependencies: `pip install feast pandas numpy pyarrow scikit-learn matplotlib`.
3. Run the notebook from top to bottom. It computes the features, writes the Parquet file, and runs `feast apply`, historical retrieval, materialization and the checks.

To use the committed repo directly, run `feast apply` inside `feature_repo/`.
