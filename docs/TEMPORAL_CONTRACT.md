# Temporal Contract

This document is the authoritative time-semantics contract for feature code.
The implementation of its interval rules lives in `src/features/temporal.py`.

## 1. Prediction target and observation

The project is a binary classification task, consistent with
`PROJECT_CONTEXT.md`:

```text
target = 1 if rating >= 4, otherwise 0
```

For each rating event, predict whether user `userId` will give movie `movieId`
a high rating at that event's recorded `timestamp`. The prediction unit is one
row of `ratings.parquet`; its conceptual grain is:

```text
(userId, movieId, timestamp) -> target
```

The user, movie, and recorded timestamp are the entity keys and prediction
context. The rating row is the event being predicted. Its `rating`, derived
target, and every other field learned from that row are unavailable to its own
predictors. The current data has no repeated `(userId, movieId)` pair or full
row, so this conceptual grain is unique in this extract. Future code must not
use that empirical fact to weaken the event semantics.

This contract would also support raw-rating regression, but that is not the
current target. The current supervised task predicts the outcome of an observed
rating event. It does not define observations or candidate generation for
arbitrary unrated `(userId, movieId)` pairs.

## 2. Available-information rule

For an observation with prediction timestamp `t`, an event `e` is eligible
historical information if and only if:

```text
e.timestamp < t
```

Equivalently, `history(t) = {e | e.timestamp < t}`. This rule applies before
filtering by user, movie, or any other entity. It excludes the target event,
every simultaneous event, and every future event from user histories, movie
histories, counts, averages, popularity, interactions, and all other derived
features.

Using `<= t` leaks outcomes recorded at the prediction time. It would expose
the target itself and, when several rows share a timestamp, allow outcomes with
no known earlier ordering to affect one another.

If a future dataset contains repeated ratings by the same user for the same
movie, each row remains a distinct prediction event. An earlier interaction is
eligible for a later one exactly when its timestamp is strictly earlier; an
interaction at the same timestamp remains ineligible.

## 3. Timestamp resolution and ordering

Ratings and tags are the source tables containing user-generated events. Both
have `userId`, `movieId`, and `timestamp`; ratings carry `rating`, while tags
carry tag text. Their CSV timestamps have whole-second resolution and no UTC
offset. Phase 0 stores them as timezone-naive `datetime64[us]`/Parquet
`timestamp[us]`; the microsecond storage unit does **not** create microsecond
source precision.

The project assumes that the recorded event timestamp approximates the time at
which the event became available to the system. MovieLens supplies no separate
ingestion or arrival timestamp, so Phase 1 cannot verify delayed availability.

The source provides no sequence number or other tie-breaker. Therefore:

- chronological order is defined only between distinct timestamps;
- file order, DataFrame index, `userId`, and `movieId` must never fabricate an
  order within a timestamp;
- a stable row identifier may make output reproducible, but cannot affect
  feature eligibility.

## 4. Simultaneous-event policy

All events sharing timestamp `t` are predicted from the same historical state
`H(t)`, containing only events before `t`. No event in the group can see another
event in the group. After predictions/features for the complete group are
produced, the group may be applied to state for timestamps later than `t`.

Future streaming or stateful feature code must use this batch order:

```text
read state before t -> compute every prediction at t -> apply all events at t
```

## 5. Window and missing-history conventions

A fixed lookback window of positive width `W` at prediction time `t` is:

```text
[t - W, t)
```

Thus an event exactly at `t - W` is included, an event one second before `t` is
included, an event exactly at `t` is excluded, and an event before `t - W` is
excluded. `rolling_window_mask` implements this left-closed, right-open rule.

Cumulative/expanding history has no left boundary and still ends strictly
before `t`. Every future feature must declare its minimum prior-event count.
Counts for empty history are `0`; undefined aggregates (such as a mean) are
missing, not silently zero. Any cold-start fallback must be explicit, must be
derived without future data, and should be accompanied by a support count when
useful.

Timestamps remain timezone-naive because the source timezone is unknown. Code
must not label them UTC or convert them between zones. Window widths represent
elapsed durations against these recorded values, and conversions must preserve
the source's one-second resolution.

## 6. Other MovieLens tables

`tags` is event data and is subject to the same strict cutoff if tag-derived
features are introduced. `movies`, `links`, `genome_scores`, and `genome_tags`
have no event/availability timestamp. Movie identity, title, genres, and link
IDs are provisionally treated as static catalog attributes. Genome values are
an undated snapshot with unknown historical availability. Being static in this
extract does not make them temporally valid for past predictions: any future
use as a static or dynamic predictor requires an explicit temporal justification
or restriction.

No exception to `< t` currently exists for timestamp-bearing data.

## 7. Leakage examples and implementation invariant

At a prediction time of `10:00`, events at `09:59`, `10:00`, and `10:01` yield
only the `09:59` event as eligible history. If two or 600 target events occur at
`10:00`, none is visible to any other. Shuffling physical rows cannot change
either result.

Feature code must use `is_strictly_prior`, `historical_events`, or
`rolling_window_mask` from `src/features/temporal.py` instead of introducing
local cutoff comparisons. Tests explicitly fail if the right boundary changes
from `< t` to `<= t`.

## 8. Implications for later phases

- Historical feature builders must process complete equal-timestamp batches.
- Offline and online paths must share the same cutoff and state-update order.
- Temporal train/validation/test boundaries must be based on timestamps, never
  random row assignment; equal-timestamp groups must not be split across a
  boundary.
- Fitted transforms and fallback statistics must use only the training period.
- Training-partition fitting and observation-level history answer different
  questions. Model preprocessing may be fitted on an appropriate training
  partition, but a historical aggregate used as a feature or fallback for an
  observation must still obey that observation's `< t` boundary. For example,
  attaching one mean computed from the entire training period to earlier
  training observations can leak later training-period outcomes.
- The exact split dates and evaluation update protocol belong to the later
  temporal-splitting phase and are intentionally not chosen here.

The temporal calculations in `notebooks/01_temporal_high_rate_analysis.ipynb`
are descriptive EDA. Some order ties by `movieId`, include the current outcome
in a rolling value, or use full-history quantities. They do not necessarily
satisfy this production contract and must not be copied into predictive feature
engineering.

## 9. Reproducible ratings audit

Run:

```bash
python -m src.data.audit_temporal
```

Results for the current canonical `data/processed/ratings.parquet` are:

| Metric | Value |
|---|---:|
| Rows | 20,000,263 |
| Unique users | 138,493 |
| Unique movies | 26,744 |
| Minimum timestamp | 1995-01-09 11:46:44 |
| Maximum timestamp | 2015-03-31 06:40:02 |
| Declared source resolution | 1 second |
| Parquet storage unit | microseconds (`timestamp[us]`) |
| Full duplicate rows beyond first occurrence | 0 |
| Repeated `(userId, movieId)` pairs | 0 |
| Rows in repeated `(userId, movieId)` pairs | 0 |
| Distinct timestamps containing more than one event | 2,281,728 |
| Maximum events at one timestamp | 643 |
| Users with more than one event at the same timestamp | 79,297 |

All observed rating timestamps were validated as aligned to the declared
whole-second resolution. Duplicate counts do not trigger any mutation or
deduplication; the audit is read-only.
The high prevalence of ties is why arbitrary within-second ordering is unsafe.

For context, the processed tags table has 465,564 rows, spans
2005-12-24 13:00:10 through 2015-03-31 03:09:12, and also has whole-second,
timezone-naive timestamps. It contains 21,828 shared timestamps, with at most
22 tag events at one timestamp. These events follow the same policy if used.
