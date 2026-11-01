# Feast Feature Store for PaySim

This document explains the work done in the notebook (`feast_feature_store_paysim.ipynb`), the decisions made, and the reasoning behind them. All numbers are taken from the notebook outputs.

## 1. Goal

Build a feature store for fraud detection that serves the same features consistently for model training (offline) and live lookups (online). Two things needed to be demonstrated:

1. **Point-in-time correctness:** no feature in the training set is computed from data after the prediction timestamp (no leakage).
2. **Online/offline consistency:** an online lookup returns the same values as the offline path.

## 2. Data

PaySim is a synthetic mobile-money transaction log: 6,362,620 rows and 11 columns, with no missing values. `step` is an hour counter from 1 to 743 (~31 days). There are 5 transaction types: `CASH_OUT`, `PAYMENT`, `CASH_IN`, `TRANSFER`, `DEBIT`. Fraud is very rare: 8,213 rows (~0.13%).

## 3. Entity selection: `nameDest`

Features answer the question "what has this entity done in the past?", so the tracked entity must appear multiple times in the data. I measured both candidates:

| Metric | `nameOrig` (sender) | `nameDest` (recipient) |
|---|---|---|
| Unique values | 6,353,307 | 2,722,362 |
| With more than 1 transaction | 9,298 | 459,658 |
| With more than 10 transactions | 0 (max 3) | 130,286 |
| Maximum transactions | 3 | 113 |

Almost every sender appears only once, so history-based features would always be 0 for them. I chose **`nameDest`** (the recipient account) as the entity. In Feast the entity is named `account`, with join key `nameDest`.

**Alternative considered:** merging sender and recipient roles into one "account" entity. Since the sender side has no history, it would add little value and more complexity, so I kept it simple.

**Limitation:** the entity is a recipient account, not a "customer".

## 4. Fraud distribution

The subset and the features are based on measured distributions, not assumptions.

| Transaction type | Total | Fraud | Ratio |
|---|---|---|---|
| `TRANSFER` | 532,909 | 4,097 | 0.77% |
| `CASH_OUT` | 2,237,500 | 4,116 | 0.18% |
| `CASH_IN`, `DEBIT`, `PAYMENT` | - | 0 | 0% |

By recipient account type: accounts starting with `C` received 8,213 fraud rows out of 4,211,125; `M` (merchant) accounts, with 2,151,495 rows, received none. The number of `M` rows equals the number of `PAYMENT` rows, so `M` accounts only receive `PAYMENT`.

## 5. Event time conversion

`step` is an integer, not a real timestamp. Since Feast is time-based, the conversion is required:

```python
base = pd.Timestamp("2025-01-01")
df["event_timestamp"] = base + pd.to_timedelta(df["step"], unit="h")
```

- Resulting range: 2025-01-01 01:00 to 2025-01-31 23:00.
- The base date is arbitrary because the data is synthetic. Real weekday or holiday effects are not meaningful here.
- Because `step` is hourly, all transactions within the same hour share one timestamp (see section 7).

## 6. Subset construction (`df_small`)

6.36 million rows are heavy for Feast and pandas. Sampling rows at random would cut accounts' histories, so I sampled at the **account level**: every selected account keeps all of its transactions.

- Only `C` accounts (571,961 accounts), since `M` accounts have no fraud.
- **All 8,169 accounts that received fraud** were included. These accounts received the 8,213 fraud transactions. Half of them have 4 or fewer transactions (median 4, max 89), so a "5+ transactions" filter would have dropped about half of them.
- Normal accounts: 280,200 accounts have at least 5 transactions. From those without fraud, **12,000** were sampled at random (`seed=42`, reproducible).
- Result: **20,169 accounts and 218,922 rows**, with 8,213 fraud rows (none lost).

**Trade-off:** the fraud share in the subset (~3.8%) is much higher than in the real data (0.13%). This is a deliberate choice, and it means model results on the subset do not reflect real-world performance.

## 7. Feature design

Four features computed from the recipient account's **past 24 hours** for each transaction:

| Feature | Meaning |
|---|---|
| `txn_count_24h` | Number of transactions received in the last 24 hours |
| `avg_amount_24h` | Average amount of those transactions (0 if there are none) |
| `transfer_count_24h` | Number of `TRANSFER` transactions received in the last 24 hours |
| `cashout_count_24h` | Number of `CASH_OUT` transactions received in the last 24 hours |

**Why these:** `transfer_count_24h` and `cashout_count_24h` follow a measured fact (100% of fraud is in these two types). The count and average amount reflect account activity and come from the general fraud-engineering assumption that fraudsters concentrate money in a few accounts. I did **not** verify this assumption on this data.

**Leakage prevention:**
- `rolling("24h", closed="left")` is used, which excludes the right edge (the current moment). An account's first transaction therefore has count 0.
- `isFraud` is the label, so it is not used as a feature.
- `oldbalance*` / `newbalance*` columns are not used, because they can reflect the post-transaction balance and carry leakage risk.

**Limitations:**
- Transactions within the same hour (same `step`) do not count as each other's past. This is a conservative choice.
- Feast requires one row per (account, timestamp), so duplicates were dropped: 218,922 rows became 199,937.
- **Window choice:** 24 hours is a design choice. In 40.3% of rows `txn_count_24h` is 0 (mean 1.64, median 1, max 86). Since about 60% of rows carry meaningful values, I kept the window. For sparse accounts (days between transactions) the features stay 0.

## 8. Feast repository and store configuration

```
feature_repo/
├── feature_store.yaml
├── entities.py
├── features.py
└── data/customer_features.parquet
```

- **Offline store:** a Parquet file (`customer_features.parquet`). Column types were aligned with the Feast schema before writing: count columns as `int64`, `avg_amount_24h` as `float32`.
- **Online store:** SQLite (`data/online_store.db`), with the registry in `data/registry.db`.
- **Entity:** `account`, `join_keys=["nameDest"]`.
- **Feature view:** `account_features`, 4 features, `ttl = 7 days`, Parquet source with `event_timestamp` as the timestamp field.
- `feast apply` ran without errors. Feast only warned that `value_type` for entities will become mandatory in a future release (not yet set in this repo).

## 9. Point-in-time training dataset

The entity dataframe (`entity_df`) contains the (account, timestamp) pairs from `df_small` plus `isFraud` (199,937 rows). `store.get_historical_features(...)` joined the 4 features, giving 199,937 rows × 7 columns.

**Validation:** the Feast output was compared with the pandas-computed table `res`, joined on (account, timestamp). Across all 199,937 rows the difference is **0** for all 4 features.

Note: Feast returns timestamps in UTC (`+00:00`); this part was removed for the comparison.

**Limitation:** this comparison shows that Feast joins the feature rows correctly, but it does not by itself prove that `res` is leak-free; the independent test in section 11 covers that. Also, the query timestamps coincide with the feature timestamps, so the case where a query falls between two feature timestamps was not tested separately.

## 10. Materialization and online/offline consistency

`store.materialize(...)` loaded the range 2025-01-01 to 2025-02-01 into the SQLite online store. The online store keeps only the **latest** value per account, so the comparison used each account's latest offline row.

| Metric | Result |
|---|---|
| Accounts compared | 20,169 (all) |
| Differences in `txn_count_24h`, `transfer_count_24h`, `cashout_count_24h`, `avg_amount_24h` | 0 |
| Accounts with `txn_count_24h > 0` | 5,101 |

Strength of the check: 5,101 accounts have a non-zero latest value, so the match does not rely only on "0 equals 0". In my first test with two accounts the online value was 0 because their latest offline row was also 0 (no other transaction in the 24 hours before their last one).

**Limitation:** the online check covers only the latest value, not historical values.

## 11. Leakage test (independent check)

1,000 random rows (700 with non-zero features, 300 with zero) were recomputed from scratch with a plain pandas filter, independently of `rolling`, and compared with the Feast output. The same rows were also computed with a "leaky" rule that includes the current moment.

- Rows that do not match the correct rule (past 24 hours, current moment excluded): **0**.
- Rows that match only the "leaky" rule: **0**.

**Limitation:** this is a sample check of 1,000 rows, not all 199,937.

## 12. TTL and freshness

The age of each account's online value was computed as the gap between its latest feature row and the end of the data. The median age is **337 hours (~14 days)**, the mean is ~270 hours, and the maximum is 721 hours.

| TTL | Share of stale accounts |
|---|---|
| 1 day | 98.5% |
| 3 days | 88.6% |
| 7 days | 74.6% |
| 14 days | 50.3% |

**Conclusion and trade-off:** because the data is sparse, a short TTL leaves features empty, while a longer TTL increases coverage. The feature view uses **TTL = 7 days**. The measurement suggests that a longer TTL (for example 14-30 days) would be more sensible for this data. I did not change the value and record this as a known limitation.

**Note:** this analysis is a **simulation** on the `res` table. I did not separately test how Feast applies TTL in the online store.

## 13. Fraud model

A `LogisticRegression(class_weight="balanced")` was trained on the offline features. The data was split **by time** (not randomly) so the model does not learn from the future:

- Train: before 2025-01-24, 193,312 rows (6,003 fraud)
- Test: 2025-01-24 and later, 6,625 rows (2,130 fraud)

**Result:** ROC-AUC = **0.52**, which is close to random guessing. For the fraud class recall is 0.99 and precision 0.33; for the normal class recall is 0.04. The model labels almost everything as fraud, and the ROC curve shows the same.

**Why it is weak:** the features only look at the recipient account's past 24 hours and do not carry a strong fraud signal. Also, because the subset was built at the account level, the fraud share in the test period (32%) differs a lot from training (3.1%), which distorts the evaluation.

**Live prediction:** the same model produced fraud probabilities (0.17 to 0.64) for 5 accounts using features from the online lookup. The values themselves are not meaningful, but the flow (train on offline features, predict with online features) works. The goal of this part is to demonstrate that flow, not model quality.

## 14. Architectural trade-offs

- **SQLite as the online store:** it is simple, local, and needs no extra service. The cost is that it cannot match the scale and resilience of a distributed store such as Redis.
- **Parquet as the offline store:** it is simple and fits the task. It does not offer the capabilities of a full data warehouse at larger scale.
- **Batch features:** features are precomputed in pandas, written to a file, and materialized. A streaming source was not built, so features are not updated in real time.
- **Recipient account as the entity:** it gives features real history, but the entity is not a "customer", and sender-side history is not captured.
- **Account-level subset:** it keeps histories intact and preserves all fraud, but the fraud share is much higher than in reality.
- **24-hour window:** it is simple, but 40.3% of rows end up with a zero count; a longer window could reduce that.
- **7-day TTL:** it is a conservative choice, but the measurements suggest a longer TTL would give better coverage on this sparse data.
- **Same-hour transactions:** they do not see each other and duplicates are dropped, because the data has no finer time resolution.
