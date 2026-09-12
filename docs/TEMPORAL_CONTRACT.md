# Temporal Contract

## Observation and target

One observation is a rating event identified by `(userId, movieId, timestamp)`.
Its binary target is `1` when `rating >= 4` and `0` otherwise.

## Information boundary

For an event at timestamp `t`, every feature may use only events whose original
timestamp is strictly less than `t`. The target event and all later events are
excluded. Calendar dates may support analysis, but the original timestamp is
the ordering field and a rating event remains the modeling unit.

## Equal timestamps

Events with identical timestamps must not provide information to one another.
All features for that timestamp are computed from the state before the group;
the full group is applied to state only after its feature values are produced.
A stable row identifier may make processing deterministic but must not weaken
the strict `< t` rule.

## Rolling windows

A lookback window of length `w` at time `t` includes events in `[t - w, t)`.
The lower bound is inclusive and the prediction timestamp is exclusive. Window
sizes and timezone handling must be explicit in each feature definition.

## Offline/online parity

Training and serving must call the same feature definitions and state-update
logic. Online prediction reads state before an event; rating ingestion validates
and records the event, then updates every affected state. Neither path may scan
the full history for each prediction.

## Missing history

Each rate or average must have an explicit cold-start fallback and an associated
support count where useful. Missing history is not silently represented as zero.
