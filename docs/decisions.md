# Engineering Decisions

### 1. Chunked processing
**Problem.** The merged long table has 58.3M rows (7.9 GB as CSV) and the single-DataFrame approach hit memory limits (reported by the author).
**Decision.** Melt and merge 3,000 series at a time and write each result straight to disk.
**Trade-off.** Peak RAM is bounded by one chunk; but chunk boundaries follow the `id` order, so a chunk is a *subset of series*, not a time slice. Later notebooks that use only some chunks therefore see only some stores/departments (see Decision 6).

### 2. Parquet for intermediate and processed data
**Problem.** Repeated analytical reads of multi-GB CSV.
**Decision.** Store every stage as Parquet.
**Evidence.** 99.4 MB for the 11 master chunks vs 7.9 GB CSV; column-selective reads (`date`, `id` only) used throughout.

### 3. Memory optimization
Techniques actually used (notebook 04): drop duplicate column `d`; `category` for 11 text columns; integer downcasting to `int8/int16`; `float32` for price and engineered features. Result on chunk 1: −28.7 % (1,086 → 775 MB). Not used: column pruning, sparse types, out-of-core training.

### 4. Native categorical handling in LightGBM
**Decision.** No one-hot encoding; pass categorical columns to LightGBM.
**Reason.** `item_id` has thousands of levels; one-hot would be memory-prohibitive.
**Trade-off.** `item_id` is the single most-used feature in the baseline; with the Day 7 split the validation series are, judging from the split code, mostly item_ids never seen in training (not measured).

### 5. Leakage-aware lag/rolling features
Rolling windows are computed on `sales.shift(1)` and all group operations are per `id`. Two exceptions are documented in [feature_engineering.md](feature_engineering.md) (`price_max`, price back-fill).

### 6. Validation strategy — an unresolved inconsistency
Day 6 used a time-based split; Day 7 switched to a row-based 80/20 split on four chunks. This was **not** a deliberate improvement: the Day 7 comment says "same split as Day 6", which is not what the code does. Results from the two are not comparable ([modeling.md](modeling.md)). Resolving this is the first engineering task before any accuracy claim.

### 7. Refactoring decisions (this repository)
* Code extracted verbatim from notebooks into `src/`; no logic changed, so known issues (`is_weekend`, `price_max`) are preserved and documented rather than silently fixed.
* The Day 7 row-based split was **not** promoted to `src/`.
* Data and models moved from `dayN/` to `data/` and `models/`; notebook paths now point there.
* Deep-learning experiments (LSTM) were removed from the repository by request.
