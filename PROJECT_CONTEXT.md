# Movie Rating Predictor — Project Context

## 1. Project Goal

The long-term goal is to build a binary classification system that predicts whether a user will give a movie a high rating.

Define the target as:

\[
Y_{u,i,t} = \mathbf{1}(rating_{u,i,t} \ge 4)
\]

The modeling objective is:

\[
P(Y_{u,i,t}=1 \mid \mathcal{F}_{t^-})
\]

where \(\mathcal{F}_{t^-}\) represents all information that was available strictly before the rating event at time \(t\).

The model must be designed as if it will eventually be used for online predictions in a production environment.

That means:

- all features must be computable from information available before prediction time;
- feature definitions used during training and serving must be identical;
- temporal leakage must be explicitly prevented;
- features should be cheap enough to compute or retrieve online;
- dynamic features should be maintained as state whenever possible instead of scanning the full event history at prediction time.

---

## 2. Dataset

The project uses the MovieLens 20M dataset.

Current raw files:

```text
data/
├── genome_scores.csv
├── genome_tags.csv
├── link.csv
├── movie.csv
├── rating.csv
└── tag.csv
```

The main rating file contains approximately 20 million events.

Expected rating event schema:

```text
userId
movieId
rating
timestamp
```

The raw CSV files must remain immutable.

---

## 3. Repository Language

The entire repository should be written in English:

- filenames
- variable names
- function names
- comments
- docstrings
- README
- notebooks
- chart titles
- axis labels
- tests
- API contracts

---

## 4. Preferred Stack

Keep the implementation intentionally simple.

Preferred stack:

- Python
- pandas
- NumPy
- PyArrow / Parquet
- scikit-learn
- matplotlib
- FastAPI
- uvicorn
- pytest
- Jupyter for exploratory analysis only

Do not introduce Spark, Polars, DuckDB, databases, orchestration frameworks, feature-store products, or unnecessary infrastructure unless the pandas-based implementation becomes insufficient.

Prefer small functions over complex object-oriented abstractions.

Avoid premature framework design.

---

## 5. Data Layer

Because the ratings dataset contains roughly 20 million rows, processed data should use Parquet instead of CSV.

Recommended structure:

```text
data/
├── raw/
│   ├── rating.csv
│   ├── movie.csv
│   ├── tag.csv
│   ├── genome_scores.csv
│   ├── genome_tags.csv
│   └── link.csv
│
├── processed/
│   ├── ratings.parquet
│   ├── movies.parquet
│   ├── links.parquet
│   └── ...
│
└── features/
    ├── user_state.parquet
    ├── movie_state.parquet
    ├── user_genre_state.parquet
    ├── user_director_state.parquet
    └── user_actor_state.parquet
```

The first data task should be:

```text
raw CSV
   ↓
schema validation
   ↓
dtype optimization
   ↓
Parquet
```

After that, notebooks and feature pipelines should prefer Parquet reads.

Use column pruning when possible.

Example principle:

```python
pd.read_parquet(
    path,
    columns=["userId", "movieId", "rating", "timestamp"],
)
```

Do not partition Parquet by year unless measurements show that it improves the workflow enough to justify the added complexity.

---

## 6. Temporal Contract

The model is event-based.

The fundamental training observation is:

```text
(userId, movieId, timestamp) -> target
```

not a daily user-movie snapshot.

For every feature:

\[
history(t) = \{events: timestamp < t\}
\]

Never use the target event itself or any event after prediction time.

Use the original timestamp for ordering events, even if some analytical summaries use day-level windows.

A date dimension may be created for convenience, with daily granularity, covering the available data period and optionally extending through the end of 2016 for simulation purposes.

However, the date table must not become the main unit of modeling.

---

## 7. Exploratory Findings

Previous exploratory analysis concluded that the temporal dimension matters and should be considered in the model design.

The project therefore follows a point-in-time-correct feature engineering framework.

One key distinction:

- calendar-time changes can be affected by changing user composition;
- within-user chronological changes are more relevant to user behavior drift.

---

## 8. Core Entities

Recommended entities:

### User

Primary key:

```text
userId
```

Dynamic state examples:

- first rating timestamp
- last rating timestamp
- number of prior ratings
- cumulative rating sum
- cumulative high-rating count

### Movie

Primary key:

```text
movieId
```

Static information:

- title
- genres
- TMDb ID
- IMDb ID
- directors
- actors

Dynamic information:

- recent rating statistics
- recent high-rating rate
- recent rating counts

### Genre

Key:

```text
genre
```

Used mainly as a cross dimension with users.

### Director

Recommended stable key:

```text
directorId
```

Prefer external person IDs from TMDb rather than names when available.

### Actor

Recommended stable key:

```text
actorId
```

Prefer external person IDs from TMDb rather than names.

### Rating Event

Represents one user rating one movie at one point in time.

Fields include:

```text
userId
movieId
rating
timestamp
high_rating
```

### Cross-Entity States

Important dynamic entities:

```text
(userId, genre)
(userId, directorId)
(userId, actorId)
```

These should summarize only prior events.

---

## 9. Static Movie Enrichment

This remains a planned extension after the compact leakage-safe baseline. It is
not a prerequisite for Feature Dictionary v1, Phase 4 feature implementation,
or Phase 4E materialization.

Use `link.csv` to connect MovieLens movies to external metadata.

Expected identifiers:

```text
movieId
imdbId
tmdbId
```

Preferred external source: TMDb.

Enrich movies with:

- director IDs
- director names
- actor IDs
- actor names
- cast order
- possibly additional stable metadata if useful later

Recommended normalized structures:

```text
movie_director
movieId | directorId
```

```text
movie_actor
movieId | actorId | cast_order
```

Do not use wide columns such as `actor_1`, `actor_2`, `actor_3` as the primary storage model.

For modeling, it may later be reasonable to restrict actor-based features to the top 3–5 billed actors.

---

## 10. Initial Feature Families

Do not implement all features immediately.

First formalize them in a feature dictionary.

### 10.1 User Features

Do not interpret "user age" as biological age because MovieLens 20M does not contain demographic age.

Use user tenure instead.

Candidate variables:

```text
user_tenure_days
user_total_ratings_before_t
user_avg_rating_before_t
user_high_rate_before_t
user_days_since_last_rating
```

Possible later extensions:

```text
user_avg_rating_30d
user_avg_rating_90d
user_high_rate_30d
user_high_rate_90d
```

### 10.2 Movie Dynamic Features

For rolling windows such as:

```text
1 month
3 months
6 months
12 months
```

define at least:

```text
movie_avg_rating_<window>
movie_rating_count_<window>
movie_high_rate_<window>
```

Example:

\[
movie\_high\_rate_{90d}(t)
=
\frac{\#\{rating \ge 4\}}{\#\{ratings\}}
\]

using only ratings before \(t\).

Counts are mandatory because averages without support size are misleading.

### 10.3 User × Genre Features

For each `(userId, genre)` state, maintain point-in-time statistics such as:

```text
count
rating_sum
high_count
avg_rating
high_rate
```

At prediction time, a movie may belong to several genres.

Therefore derive movie-level features from the user's genre states, for example:

```text
user_target_genre_avg_rating_mean
user_target_genre_high_rate_mean
user_target_genre_max_affinity
user_target_genre_history_count
```

Avoid creating a giant sparse column set unless there is a clear modeling reason.

### 10.4 User × Director Features

The proper entity is:

```text
(userId, directorId)
```

not `(userId, directorId, movieId)`.

Candidate statistics:

```text
user_director_rating_count
user_director_avg_rating
user_director_high_rate
```

These represent how the user historically rated other movies by the target movie's director.

Expect substantial sparsity.

Later, use smoothing or fallback logic when counts are small.

### 10.5 User × Actor Features

Because movie enrichment will already retrieve cast information, consider analogous features:

```text
user_actor_rating_count
user_actor_avg_rating
user_actor_high_rate
```

For movies with several main actors, aggregate actor affinity using statistics such as:

```text
mean
max
count-weighted mean
```

Do not overcomplicate this in the first version.

---

## 11. Cold Start and Fallbacks

Every feature definition must specify what happens when no historical state exists.

Examples:

```text
new user
new movie
user has never seen this genre
user has never seen this director
user has never seen this actor
```

Possible fallbacks include:

- global mean
- movie-level prior
- genre-level prior
- director-level prior
- smoothed estimate

Do not silently fill unknown historical rates with zero, because zero has semantic meaning.

---

## 12. Online Prediction Requirement

The intended production-like interface should not require the caller to manually provide engineered variables.

Preferred prediction request:

```text
POST /predict
```

Input:

```json
{
  "userId": 123,
  "movieId": 456,
  "timestamp": "2015-01-10T18:30:00"
}
```

Conceptual flow:

```text
userId + movieId + timestamp
          ↓
feature service / state lookup
          ↓
user features
movie features
user-genre features
user-director features
user-actor features
          ↓
model
          ↓
P(rating >= 4)
```

Possible response:

```json
{
  "probability_high_rating": 0.82
}
```

An optional debugging endpoint may expose computed features, but the main contract should stay entity-based.

---

## 13. Rating Ingestion Requirement

Create a second production-like workflow:

```text
POST /ratings
```

Input:

```json
{
  "userId": 123,
  "movieId": 456,
  "rating": 5.0,
  "timestamp": "2015-01-10T18:30:00"
}
```

Conceptual update flow:

```text
new rating event
      ↓
validate event
      ↓
append raw event
      ↓
update user state
      ↓
update movie state
      ↓
update user-genre state
      ↓
update user-director state
      ↓
update user-actor state
```

This endpoint simulates a local streaming feature update.

The implementation does not need real streaming infrastructure.

---

## 14. Feature State Principle

Do not scan the full 20M-row history during every prediction request.

Maintain compact state tables.

Example:

```text
user_genre_state

userId | genre  | count | rating_sum | high_count
123    | Sci-Fi | 52    | 224        | 41
```

Then:

\[
avg\_rating = rating\_sum / count
\]

and:

\[
high\_rate = high\_count / count
\]

can be retrieved cheaply.

This distinction is central:

```text
raw event history
vs.
maintained feature state
```

---

## 15. Offline / Online Parity

This is one of the most important engineering requirements.

The same feature definition must be used for training and serving.

Conceptually:

\[
FeatureDefinition_{offline}
=
FeatureDefinition_{online}
\]

The training dataset must reproduce what the online system would have known at each historical prediction timestamp.

Avoid training-serving skew.

---

## 16. Training Dataset Construction

Each historical rating event becomes one labeled observation.

Example:

```text
userId = 123
movieId = 456
timestamp = t

features = state available strictly before t
target   = 1 if rating >= 4 else 0
```

Training rows must be created using point-in-time-correct joins or equivalent cumulative state logic.

Do not compute full-history aggregates and join them back to past events.

That would cause leakage.

---

## 17. Validation Strategy

Do not use a purely random train/test split as the main validation strategy.

Use temporal validation.

Conceptually:

```text
|---------------- train ----------------|--- validation ---|--- test --->
```

The exact cutoffs should be chosen after inspecting the timestamp coverage and sample sizes.

Metrics can be finalized later.

The first baseline should remain simple.

---

## 18. Repository Design Principles

Recommended high-level structure:

```text
movie_rating_predictor/
├── data/
├── notebooks/
├── src/
│   ├── data/
│   ├── features/
│   ├── training/
│   └── serving/
├── tests/
├── requirements.txt
├── .gitignore
└── README.md
```

Possible feature modules:

```text
src/features/
├── user.py
├── movie.py
├── user_genre.py
├── user_director.py
└── user_actor.py
```

Keep modules small.

Prefer functions such as:

```text
build_user_features(...)
build_movie_features(...)
build_user_genre_features(...)
```

Do not create unnecessary inheritance hierarchies, factories, registries, YAML-driven frameworks, or configuration layers.

---

## 19. Notebook Policy

Use notebooks for:

- EDA
- validating hypotheses
- inspecting feature behavior
- checking temporal assumptions
- comparing model results

Do not leave production feature logic only inside notebooks.

Reusable feature logic should move to `src/`.

---

## 20. Code Style

The desired implementation style is:

- concise
- readable
- explicit
- pandas-first
- vectorized where practical
- easy to test
- low abstraction
- low dependency count

Avoid:

- very large functions
- row-wise loops when vectorization is practical
- hidden temporal assumptions
- duplicated feature definitions
- unnecessary classes
- premature optimization
- infrastructure for infrastructure's sake

---

## 21. Roadmap

### Original roadmap

The original plan placed TMDb director/actor enrichment before point-in-time
feature builders, then separated training dataset construction, temporal
validation, the baseline model, current online state, `/predict`, `/ratings`,
and final parity tests into successive phases. That sequence established the
long-term direction, but it was refined after the available MovieLens data and
the first feature contract were examined.

In that historical numbering, enrichment was Phase 4, feature builders were
Phase 5, and training through final parity occupied Phases 6–12. References in
older commits may therefore use those numbers; the sequence below is current.

The compact v1 baseline deliberately uses MovieLens-derived rating, genre, and
catalog information first. TMDb director/actor enrichment remains planned, but
it is not required for Feature Dictionary v1 or the current Phase 4. Stable,
source-qualified person identifiers and a versioned external snapshot are still
required before any people-derived state or predictor can be enabled.

### Current engineering roadmap

Follow this sequence for current work:

#### Phase 0 — Data foundation — COMPLETE

Validate schemas and values, optimize dtypes, convert all six immutable raw
MovieLens inputs to typed Parquet, and verify each round trip.

#### Phase 1 — Temporal contract — COMPLETE

Define the target, exclusive `timestamp < t` history, equal-timestamp atomic
batches, and fixed windows `[t - W, t)`.

#### Phase 2 — Entity model — COMPLETE

Define canonical users, movies, genres, rating events, relationship bridges,
source-event provenance, and sparse state identity. Director and actor entities
remain contractual placeholders pending stable mappings.

#### Phase 3 — Feature Dictionary v1 — COMPLETE

Specify the 17-feature compact baseline, including formulas, sources, temporal
rules, update behavior, fallbacks, and dtypes, before broad implementation.
Identifiers, timestamps, and provenance fields are observation context rather
than predictors.

#### Phase 4 — Feature implementation — IN PROGRESS

- **4A — COMPLETE:** expanding global, user, and movie history.
- **4B — COMPLETE:** user recency and rolling 30-day movie activity.
- **4C — COMPLETE:** user × target-genre history.
- **4D — COMPLETE:** static catalog/context and deterministic catalog content
  identity.
- **4E — NEXT:** checkpoint/replay orchestration and full feature
  materialization from canonical Parquet inputs.

Phase 4E must persist and restore complete state at an explicit exclusive
cutoff, replay complete timestamp batches, prevent duplicate application at the
canonical provenance grain, persist and enforce `catalog_snapshot_id`, prove
uninterrupted/replay equivalence and event conservation, and record enough
provenance to reproduce a materialized feature dataset. Checkpoint cadence,
physical formats, and storage layout remain implementation choices.

Phase 4E does not add predictor families, external enrichment, tags/genome
features, embeddings, collaborative filtering, modeling, serving APIs, or
feature-store infrastructure.

#### Phase 5 — Training dataset and temporal modeling — NOT STARTED

Progress from full feature materialization to a labeled event-level dataset,
temporal train/validation/test splits, a simple baseline classifier, and
evaluation. Exact split dates and fitted preprocessing belong here.

#### Phase 6 — Current online state — NOT STARTED

Materialize the latest state required for online prediction.

#### Phase 7 — FastAPI `/predict` — NOT STARTED

Generate an online score from `userId`, `movieId`, and `timestamp`.

#### Phase 8 — FastAPI `/ratings` — NOT STARTED

Ingest a new rating event and update the relevant feature state.

#### Phase 9 — End-to-end parity and temporal tests — NOT STARTED

Complete serving-level leakage, boundary, cold-start, replay, and offline/online
parity coverage. Foundational tests are already developed alongside earlier
phases; this phase closes the integrated system contract.

### Deferred enrichment

TMDb movie-director and movie-actor mappings remain a planned extension after
the compact leakage-safe baseline. Cache normalized mappings with stable
external person identifiers and documented snapshot availability before adding
director-, actor-, or user-person features.

---

## 22. What Codex Should Do Next

Proceed to Phase 4E. Complete checkpoint/replay and provenance guarantees
before materializing the full 17-feature dataset. Do not begin modeling until
the uninterrupted and restored/replayed feature paths are equivalent.

Keep all implementation decisions consistent with the principles in this document.

When uncertain, prefer the simpler design.
