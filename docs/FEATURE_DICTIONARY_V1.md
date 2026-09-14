# Feature Dictionary v1 and Genome addendum

This document is the authoritative contract for the original 17 Phase 3
predictors and the eight-predictor Genome addendum. The combined 25-predictor
contract is the input to the current PRD model; the original 17-predictor
contract remains the historical Phase 5 baseline. This document specifies
features; the canonical model pointer and split/training contract live in
`models/prd_model_manifest.json`, and the corresponding executable contract is
`src/training/prd_config.py`.

## 1. Inherited contracts

Feature rows have the grain of one canonical `rating_event` and are computed
immediately before that event's recorded `timestamp`. The target event's
`rating` and `highRating` are never inputs. All timestamp-bearing history is
selected with the Phase 1 rule `event.timestamp < prediction_timestamp`; a
fixed window is `[prediction_timestamp - window, prediction_timestamp)`.

All events at one timestamp are simultaneous. Offline and online builders must
therefore use this order:

```text
read state before t -> score every event at t -> apply the complete t batch
```

`ratingEventId` is immutable provenance, not temporal order. Each canonical
event updates global, user, and movie state once. A movie-genre bridge join may
produce one update per `(ratingEventId, genreId)`, but must not create another
canonical event or change canonical event counts. Checkpoints represent complete
`H(checkpointCutoff)` and replay only complete batches in
`[checkpointCutoff, t)`; Phase 4E replay must enforce idempotency at the Phase 2
provenance grain.

## 2. Notation and shared policies

For a target event `(u, m, t)`, let

- `H(t) = {i | t_i < t}` be all legal prior canonical rating events;
- `H_u(t)` and `H_m(t)` be the subsets for user `u` and movie `m`;
- `H_m,30(t) = {i | movieId_i=m, t-30 days <= t_i < t}`;
- `r_i` be the raw rating in `[0.5, 5.0]`;
- `G(m)` be the set of distinct `genreId` values in the validated
  `movie_genre` bridge for `m`;
- `H_ug(t) = {i | userId_i=u, g in G(movieId_i), t_i<t}`;
- `n(A)=|A|`, `s(A)=sum_{i in A} r_i`, and
  `mean(A)=s(A)/n(A)` for nonempty `A`;
- `pop_sd(A) = sqrt(sum_{i in A}(r_i-mean(A))^2/n(A))` for nonempty
  `A`. This is population, not sample, standard deviation.

Counts are exact event or relationship-update counts and never count a missing
fallback as an observation. Rating sums and sum-of-squares must be accumulated
in `float64`; exposed means and standard deviations are `float32`.

### Fallback root

`global_mean_rating(t)` is the mean of `H(t)`. When `H(t)` is empty, it is the
fixed baseline prior `3.5` chosen by this feature contract. This constant is
configuration, not fitted data. No full-dataset or later-training aggregate is
legal. User means fall back to this point-in-time global mean;
movie means do the same; target-genre user means fall back to the resolved user
mean. Standard deviations fall back to `0.0` only when support is zero; for one
observation population standard deviation is exactly `0.0`. Support features
make these fallback cases distinguishable.

Historical validation/test replay must update state after each complete scored
timestamp batch so the features remain the exact `H(t)` quantities defined
here. A frozen-at-split evaluation would measure stale-state variants and must
not reuse these feature names. Online serving can produce the exact values when
prior rating outcomes have arrived by their recorded timestamps; delayed or
missing outcomes are a serving-state limitation that must be monitored rather
than backfilled from future data.

### Static catalog assumption

MovieLens supplies no availability timestamps for `movies` or its genre
labels. V1 adopts a declared, versioned **frozen catalog snapshot** assumption:
the canonical `movies` row and validated `movie_genre` memberships in the
processed snapshot are treated as intrinsic movie metadata available whenever
that movie can be scored. This gives offline/online parity when serving loads
the same catalog snapshot, but historical availability cannot be proven from
the source. Results using genre/year features must disclose this limitation and
should be compared with a dynamic-history-only baseline.

In the current implementation, “versioned” means that Phase 4D computes a
deterministic content identity and Phase 4E persists and enforces it in
checkpoint and materialized-feature metadata.

`releaseYear` is the conservative terminal-title parse defined in Phase 2. It
is not a release date. The terminal token `(0000)` is missing because the
Gregorian calendar and movie-age definition have no year zero. Missing static
values remain missing rather than being filled from later data; an explicit
missingness feature accompanies release year and age.

## 3. Accepted v1 features

Every feature in this table is available to the preferred offline/online v1
model under the policies above. “After event” in an update rule always means
after every event in the timestamp batch has been scored.

| `feature_name` | `feature_family` / `entity` / `state_key` | `mathematical_definition` | `source_data` | `temporal_window` | `point_in_time_rule` | `online_availability` | `update_rule` | `cold_start_fallback` | `expected_type` |
|---|---|---|---|---|---|---|---|---|---|
| `global_rating_count` | Global history; singleton state | `n(H(t))` | `rating_events.ratingEventId`, `timestamp` | Expanding | Count distinct canonical events with `timestamp < t`; never genre-expanded rows. | Yes; scalar state | After event, increment once per previously unapplied `ratingEventId`. | `0` | `uint64`, non-null |
| `global_mean_rating` | Global history; singleton state | `mean(H(t))` when nonempty | `rating_events.ratingEventId`, `rating`, `timestamp` | Expanding | Same legal set `H(t)`; the target and its entire simultaneous batch are excluded. | Yes; scalar count and `float64` sum | After event, add its rating once and increment count once. | Fixed `3.5` only when global count is zero | `float32`, non-null |
| `user_rating_count` | User history; `user_state(userId)` | `n(H_u(t))` | `rating_events.ratingEventId`, `userId`, `timestamp` | Expanding | Only canonical events for `u` with `timestamp < t`. | Yes | Increment once for `(userId, ratingEventId)`. | `0` | `uint64`, non-null |
| `user_mean_rating` | User history; `user_state(userId)` | `mean(H_u(t))` when nonempty | `rating_events.ratingEventId`, `userId`, `rating`, `timestamp` | Expanding | Same `H_u(t)` as the support count. | Yes | Add rating to `float64` user sum and increment user count once. | `global_mean_rating(t)` | `float32`, non-null |
| `user_rating_std_pop` | User history; `user_state(userId)` | `pop_sd(H_u(t))` | `rating_events.ratingEventId`, `userId`, `rating`, `timestamp` | Expanding | Same `H_u(t)`; population divisor is `n`, including at `n=1`. | Yes | Update count, mean, and `M2` with a numerically stable population-variance recurrence once per event. | `0.0` when count is zero; exactly `0.0` at count one | `float32`, non-null |
| `user_seconds_since_last_rating` | User history; `user_state(userId)` | `(t - max_{i in H_u(t)} t_i)` in elapsed seconds | `rating_events.userId`, `timestamp` | Expanding last observation | Maximum timestamp must be strictly less than `t`; equal-time rows cannot become “last.” | Yes | After the batch, set last timestamp to the batch timestamp for each touched user. | Missing | nullable `float64` seconds, nonnegative |
| `movie_rating_count` | Movie history; `movie_state(movieId)` | `n(H_m(t))` | `rating_events.ratingEventId`, `movieId`, `timestamp` | Expanding | Only canonical events for `m` with `timestamp < t`. | Yes | Increment once for `(movieId, ratingEventId)`. | `0` | `uint64`, non-null |
| `movie_mean_rating` | Movie history; `movie_state(movieId)` | `mean(H_m(t))` when nonempty | `rating_events.ratingEventId`, `movieId`, `rating`, `timestamp` | Expanding | Same `H_m(t)` as the support count. | Yes | Add rating to `float64` movie sum and increment movie count once. | `global_mean_rating(t)` | `float32`, non-null |
| `movie_rating_std_pop` | Movie history; `movie_state(movieId)` | `pop_sd(H_m(t))` | `rating_events.ratingEventId`, `movieId`, `rating`, `timestamp` | Expanding | Same `H_m(t)`; population divisor is `n`. | Yes | Update count, mean, and `M2` once per event. | `0.0` when count is zero; exactly `0.0` at count one | `float32`, non-null |
| `movie_rating_count_30d` | Movie activity; keyed by `movieId` | `n(H_m,30(t))` | `rating_events.ratingEventId`, `movieId`, `timestamp` | Trailing 30 elapsed days | Includes the left boundary `t-30 days`; excludes `t` and all simultaneous events. | Yes, with a timestamped queue/buckets that preserve second-level boundary semantics | After the batch, add each event once; before reads, evict only timestamps `< t-30 days`, not the inclusive boundary. | `0` | `uint64`, non-null |
| `user_target_genre_rating_count` | User × target genres; `user_genre_state(userId, genreId)` | `sum_{g in G(m)} n(H_ug(t))` | `rating_events`; validated `movie_genre` | Expanding relationship associations | Build every `H_ug(t)` only from events before `t`, then sum target-genre states. A prior event sharing two target genres contributes two associations by definition, but remains one canonical event. | Yes under frozen catalog assumption | For each distinct bridge membership of an event's movie, increment `(userId, genreId)` once with the same `ratingEventId`. | `0`; also `0` when `G(m)` is empty | `uint64`, non-null |
| `user_target_genre_mean_rating` | User × target genres; `user_genre_state(userId, genreId)` | `sum_{g in G(m)} s(H_ug(t)) / sum_{g in G(m)} n(H_ug(t))` when denominator is positive | `rating_events`; validated `movie_genre` | Expanding relationship associations | Numerator and denominator use identical strict-prior user-genre states. Overlap weighting is intentional and documented by the support count. | Yes under frozen catalog assumption | Add rating and count once per distinct `(ratingEventId, genreId)` after scoring the batch. | `user_mean_rating(t)` when association count is zero or `G(m)` is empty | `float32`, non-null |
| `user_target_genre_mean_delta` | User × genre relative preference | `user_target_genre_mean_rating - user_mean_rating` | The two resolved v1 features above | Expanding derived feature | Both operands come from the same pre-`t` state and their specified fallbacks. | Yes; derived at read time | No independent state update. | `0.0` follows from the target-genre mean fallback | `float32`, non-null |
| `movie_genre_count` | Static movie context; `movie_genre(movieId, genreId)` | `|G(m)|` | Validated `movie_genre` | Static snapshot | Use only the versioned frozen catalog snapshot; count distinct bridge keys. | Yes under frozen catalog assumption | Changes only when deploying a new catalog snapshot, never from a rating event. | `0` for `genreStatus=missing_in_source` | `uint8`, non-null |
| `movie_release_year` | Static movie context; `movie(movieId)` | Conservative Phase 2 `releaseYear` parse | `movies.title` via canonical movie builder | Static snapshot | Use only the versioned frozen catalog snapshot; do not derive from future events or external APIs. | Yes under frozen catalog assumption | Changes only with a versioned catalog correction/redeployment. | Missing | nullable `int16` |
| `movie_release_year_missing` | Static movie context; `movie(movieId)` | `1` iff canonical `releaseYear` is missing, else `0` | Canonical `movie.releaseYear` | Static snapshot | Same snapshot as `movie_release_year`. | Yes under frozen catalog assumption | Recompute only with a versioned catalog deployment. | `true` when year is missing | `bool`, non-null |
| `movie_age_years` | Movie × prediction context | `(t - Timestamp(releaseYear, Jan 1, 00:00:00)) / (365.2425 days)` | Canonical `movie.releaseYear`; prediction `timestamp` | Static attribute plus current time | Use the target timestamp and frozen release year only. Do not clamp negative values: they reveal source inconsistency rather than silently changing semantics. | Yes under frozen catalog assumption | No rating-state update; compute at read time. | Missing when release year is missing | nullable `float32` years |

The feature output must retain `ratingEventId` for audit joins, but it is not a
predictor. `userId`, `movieId`, and prediction `timestamp` are observation
context/keys, not numeric model features unless a later contract explicitly
introduces a valid encoding.

### Genome addendum: current PRD predictors (8 predictors)

This controlled addendum extends materialization from 17 to 25 predictors
without changing any baseline feature. `genome_scores` is classified as
**static_external_metadata**: its undated relevance snapshot is assumed to be
available for a movie at every scoring time. This is an explicit modeling
assumption and source limitation, not a claim of historical availability.
User-dependent Genome features still read only rating events in `H_u(t)` and
exclude the target row and every same-timestamp row.

Let `G_m` be the complete relevance vector ordered by `tagId`. A movie is
Genome-valid only when all snapshot tags are present and its vector has a
finite nonzero norm. `L_u(t)` contains valid-Genome movies rated at least 4 by
`u` strictly before `t`; `N_u(t)` is defined analogously for ratings below 4.
Cosine is the ordinary dot product divided by both vector norms.

| `feature_name` | entity level | source / classification | definition | temporal behavior | missing-value behavior |
|---|---|---|---|---|---|
| `genome_user_positive_cosine` | User × target movie | `genome_scores` static_external_metadata + historical ratings | `cosine(G_m, mean_{j in L_u(t)} G_j)` | Historical; only ratings `< t`; update positive sum after the complete timestamp batch | Missing if target vector is unavailable, positive history is empty, or a required norm is zero |
| `genome_user_negative_cosine` | User × target movie | `genome_scores` static_external_metadata + historical ratings | `cosine(G_m, mean_{j in N_u(t)} G_j)` | Historical; only ratings `< t`; update negative sum after the complete timestamp batch | Missing if target vector is unavailable, negative history is empty, or a required norm is zero |
| `genome_preference_margin` | User × target movie | Derived from the two historical cosine features | Positive cosine minus negative cosine | Historical; both operands use the same pre-`t` state | Missing unless both cosine operands are present |
| `genome_nearest_liked_similarity` | User × target movie | `genome_scores` static_external_metadata + historical liked ratings | `max_{j in L_u(t)} cosine(G_m, G_j)` | Historical; only liked ratings `< t` | Missing if target vector is unavailable or no valid liked vector exists |
| `genome_top5_liked_similarity` | User × target movie | `genome_scores` static_external_metadata + historical liked ratings | Mean of the five largest valid `cosine(G_m, G_j)` values; use all when fewer than five exist | Historical; only liked ratings `< t` | Missing if target vector is unavailable or no valid liked vector exists |
| `genome_movie_relevance_mean` | Target movie | `genome_scores`; static_external_metadata | Population mean of all values in `G_m` | Static snapshot; no rating history used | Missing when the target has no complete, finite, nonzero Genome vector |
| `genome_movie_relevance_std` | Target movie | `genome_scores`; static_external_metadata | Population standard deviation (`ddof=0`) of `G_m` | Static snapshot; no rating history used | Missing when the target has no complete, finite, nonzero Genome vector |
| `genome_movie_top10_mean` | Target movie | `genome_scores`; static_external_metadata | Mean of the ten largest relevance values in `G_m` | Static snapshot; no rating history used | Missing when the target has no complete, finite, nonzero Genome vector |

All eight outputs are nullable `float32`. Historical movies without valid
Genome vectors are ignored rather than imputed. No global, future, target-row,
or label-derived fallback is used. The top-10 definition requires at least ten
snapshot tags; MovieLens 20M supplies 1,128.

The controlled same-protocol experiment promoted this addendum together with
the unchanged 17 predictors as `xgboost_genome_prd_v1`. The manifest identifies
the exact model artifact and locks the ordered names, dtypes, target, temporal
splits, and training parameters; the promotion record documents the evidence
and limitations.

## 4. Candidate disposition

### Accept for v1

The 17 features above are the smallest useful mix found in the current
repository: two global context/fallback features, four user-history features,
four movie-history/activity features, three compact genre-preference features,
and four static/context features. Support and missingness features are retained
because they let the model distinguish evidence from fallback values.

### Defer

| Candidate | Reason |
|---|---|
| User/movie/global high-rating count or rate | Closely related to mean raw rating and directly derived from the target threshold; test incremental value after the baseline. |
| Genre population count/mean/rate | Adds another overlapping relationship state and fallback layer; first measure whether user-target-genre history adds value. |
| Multiple recent popularity windows | Correlated feature expansion and additional eviction state; 30 days is one explicit first baseline. |
| Recent user means, slopes, or first-vs-last drift | EDA motivates drift, but a stable online trend estimator and minimum-support policy need separate design. |
| Per-genre one-hot model columns | Encoding belongs to the Phase 5 model pipeline; the Phase 3 state contract retains canonical genre keys without committing to a changing column vocabulary. |
| Raw title or title-derived tokens | High-dimensional text processing is outside the interpretable baseline. |
| Tag event features | Tags obey `< t` but require text normalization, user-generated availability semantics, and online tag ingestion. |
| Raw Genome score/tag columns | The eight-feature Genome addendum is accepted; raw 1,128-dimensional expansion remains out of scope. |
| Director/actor and user-person features | No current stable person IDs or mappings; Phase 2 explicitly blocks materialization. |
| External IDs (`imdbId`, `tmdbId`) | Join keys, not ordinal predictors; retain for future versioned enrichment only. |

### Reject

| Candidate | Reason |
|---|---|
| Current rating or `highRating` | The outcome being predicted; direct leakage. |
| Full-dataset or full-training-period aggregate attached to earlier rows | Contains outcomes later than those rows even if called a prior or imputation value. |
| File-order, DataFrame-index, movie-ID, user-ID, or `ratingEventId` ordering | Fabricates chronology within equal timestamps; IDs are identity, not time. |
| Raw IDs as continuous numeric predictors | Numeric magnitude has no contracted predictive meaning and does not generalize to unseen entities. |
| User × movie history | No repeated pairs in the current extract, so it has no demonstrated baseline signal and adds state without purpose. |
| Random-split target aggregates | Violates the event-time training contract. |
| Counting exploded genre rows as rating events | Breaks canonical event grain and biases global/user/movie counts. |

## 5. Leakage and parity audit

- **Simultaneous events:** every dynamic feature reads one immutable pre-batch
  snapshot. Only after all rows at `t` are emitted may the batch update state.
- **Cold start:** unseen keys have zero support. All rating means descend through
  point-in-time state to the fixed `3.5` root; no future aggregate is used.
  Recency and unavailable release metadata remain genuinely missing.
- **Global priors:** global count/mean are historical online state, not fitted
  constants. The only fallback constant is the contracted `3.5` baseline prior.
  Any later scaler, encoder, or imputer must be fit on the training partition
  and frozen for validation/test/serving, without changing historical feature
  values.
- **Genre expansion:** bridge keys are unique and every expanded update retains
  `ratingEventId`. Global, user, and movie counts use canonical events only.
  User-target-genre counts intentionally count associations and are named as
  such; overlap cannot masquerade as canonical event support.
- **State updates:** serializable state primitives, deterministic restoration,
  and source-bound replay enforce idempotency at the canonical event grain.
  Window eviction preserves the inclusive left boundary.
- **Source verification:** all dynamic columns exist in canonical
  `ratings.parquet`; static `title`/`genres` inputs exist in `movies.parquet`;
  `releaseYear`, `genreStatus`, `ratingEventId`, `highRating`, and bridge tables
  are canonical Phase 2 derived fields rather than Phase 0 Parquet columns.
- **Online feasibility:** expanding features require small count/sum/M2/last-time
  state; the 30-day count additionally requires timestamped eviction state.
  Genre and year features require the identical versioned catalog snapshot.

## 6. Catalog identity and persisted provenance

The feature semantics are closed. Phase 4D added deterministic, order-invariant
SHA-256 `catalog_snapshot_id` generation over canonical `movies` plus
`movie_genre`. Phase 4E persists and enforces that identity on feature outputs
and checkpoints so offline and online code can prove that they used the same
frozen static snapshot.
