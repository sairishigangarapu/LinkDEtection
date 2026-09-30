# DATASET REPORT — LCORE-D (local inspection, no assumptions)

Source: `LCORE-D A Benchmark Dataset for Core Network Analy/` (GNS3-emulated ISP,
ICMP + SNMP polling, fault-annotated; Singh & Gupta, Mendeley Data).
Fault log: 70 injected events, 12-10-2025 → 28-11-2025.

## Files and size

| File | Rows | Cols | Device | H | T | TH | F |
|------|-----:|-----:|--------|---:|---:|---:|---:|
| LCORE-D R1.csv | 24,240 | 42 | CORE-R1 | 23,811 | 0 | 0 | 429 |
| LCORE-D R2.csv | 24,229 | 32 | CORE-R2 | 5,699 | 18,135 | 1 | 394 |
| LCORE-D R3.csv | 24,246 | 51 | CORE-R3 | 5,006 | 18,883 | 0 | 357 |
| LCORE-D R4.csv | 24,254 | 47 | CORE-R4 | 5,143 | 18,773 | 0 | 338 |
| LCORE-D R5.csv | 24,234 | 42 | CORE-R5 | 4,973 | 18,926 | 0 | 335 |
| LCORE-D R6.csv | 24,239 | 51 | CORE-R6 | 5,150 | 18,749 | 0 | 340 |
| LCORE-D R7.csv | 24,270 | 42 | EDGE-MS | 23,811 | 0 | 0 | 429 |
| Fault Log/FAULT_LOG.csv | 70 events | 6 | — | — | — | — | — |

Unified table (`data/unified.parquet`): **169,682 rows** (30 R7 rows dropped:
missing `class`), 232 columns (union; interface columns differ per router).

## Label definition (verified, not assumed)

* `class == 'F'` marks induced-fault intervals: spot-checked F01
  (single link failure, 12-10 10:58–11:04) against first `F` rows
  (epoch 1760266680 = 10:58:00) — exact match.
* 1,545 / 2,622 `F` rows (59%) fall inside logged fault windows; **all 69
  parsed fault events contain `F` rows**. The remaining 41% of `F` rows lie
  outside logged windows — consistent with the dataset paper's preserved
  transient/natural faults (reconvergence, control-plane instability).
* `class == 'T'` (transient/troubled) dominates CORE-R2…R6 after 12-10 evening
  (ICMP loss ≈ 99.98%, i.e. worse-looking than `F`). The release does not
  define `H/T/F` precisely. A first F-vs-rest run gave val PR-AUC ≈ 0.01
  (chance ≈ 0.01): keeping `T` as "non-failure" makes the task unlearnable,
  because `T` outnumbers `F` 60:1 among loss≈100 rows. Final documented
  choice: **binary task H-vs-F; `T` + the single `TH` row excluded from
  train/val/test** (rolling history may still use `T`-row telemetry values —
  labels never leak). Supplementary check: the final models score held-out
  `T` rows ≈ 0 failure probability, i.e. they learn subtler signatures
  (response time, ping, rolling context) rather than "loss = 100 ⇒ failure".

## Class balance

Failure samples: 2,622. Non-failure: 167,060. **Failure rate 1.545%**
(ratio ≈ 1:64). Per-device failure %: 1.38–1.77 (evenly spread — every
router sees failures, which enables node-level localisation).

## Temporal coverage

Unix `timestamp`, Δ = 60 s sampling, 2025-10-12 10:16 → 2025-11-28 14:14 UTC
(~47 days with collection gaps; no duplicate rows or timestamps).
Temporal information: YES → global 70/15/15 temporal split, no shuffling.

## Topology information: PARTIAL (preserved for the GNN phase)

Available: `Device_name` (7 nodes: CORE-R1…R6, EDGE-MS), `Hop_to_server`,
`Hop_to_core`, `downstream_dependents`, per-interface SNMP counters
(U/B1…B14/P1…P4/C1…C3 prefixes). **Missing: explicit link_id / source /
destination columns; fault log gives windows + domains, not per-link IDs.**
Consequence: genuine link-level localisation is NOT supported — node-level
(router) ranking is used instead, stated as a limitation.

## Features

* Numerical: ICMP loss / ping / response time, per-interface Bits
  received/sent, duplex, discards, errors, interface type/speed,
  SNMP availability, uptime. Categorical: `class`, `Device_name`
  (+ `router_name` variant in R4–R6, harmonised on load).
* Sentinel `-1` = "unavailable" (ICMP loss, Bits counters, uptime) → NaN +
  missing indicators, median-imputed (train medians only).
* Dirty data: one `` `1 `` typo in R1 `ICMP ping` (row 12900) → coerced to NaN.
* Constant (dropped): `path_up` (always 1), all operational statuses
  (always 1), never-varying interface counters.
* Signal check: R1/R7 interface discard/error cells are non-zero
  (8,642 / 967); R2/R3/R5 sparse; R4/R6 all-zero — interface errors alone
  cannot explain failures; ICMP + rolling context carry the signal.

## Leakage note

`F` intervals show ICMP loss ≈ 96–100%, so contemporaneous ICMP features are
strongly indicative — this is a *detection* baseline
(P(failed now | telemetry up to now)), stated openly. True forecasting
(P(failure at t+Δ)) is listed as future work. No future rows are used in any
feature (causal shift(1) rolling); split is strictly temporal.
