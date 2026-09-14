"""Genome metadata lookups and strict-prior user preference features.

Genome relevance vectors are an undated, frozen external snapshot.  Static
movie statistics read that snapshot directly.  User-dependent values are
resolved before each complete timestamp batch is applied, so only ratings with
``event.timestamp < prediction_timestamp`` contribute.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype, is_numeric_dtype


CONTEXT_COLUMNS = ("ratingEventId", "userId", "movieId", "timestamp")
GENOME_FEATURE_COLUMNS = (
    "genome_user_positive_cosine",
    "genome_user_negative_cosine",
    "genome_preference_margin",
    "genome_nearest_liked_similarity",
    "genome_top5_liked_similarity",
    "genome_movie_relevance_mean",
    "genome_movie_relevance_std",
    "genome_movie_top10_mean",
)


@dataclass(frozen=True, slots=True)
class GenomeLookup:
    """Dense, normalized vectors and static statistics for complete movies."""

    movie_to_row: dict[int, int]
    normalized_vectors: np.ndarray
    raw_vectors: np.ndarray
    static_statistics: np.ndarray
    tag_ids: tuple[int, ...]


def build_genome_lookup(genome_scores: pd.DataFrame) -> GenomeLookup:
    """Validate long-form scores and index only complete, nonzero movie vectors.

    A movie missing any tag in the snapshot has no valid Genome vector.  This
    preserves missingness instead of inventing relevance values.
    """
    required = ("movieId", "tagId", "relevance")
    if missing := set(required) - set(genome_scores.columns):
        raise ValueError(f"genome_scores: missing required columns: {sorted(missing)}")
    values = genome_scores.loc[:, required]
    if values.isna().any().any():
        raise ValueError("genome_scores: required columns must not contain missing values")
    for column in ("movieId", "tagId"):
        if not is_integer_dtype(values[column].dtype):
            raise TypeError(f"genome_scores: {column} must have an integer dtype")
        if (values[column] <= 0).any():
            raise ValueError(f"genome_scores: {column} must contain only positive values")
    if not is_numeric_dtype(values["relevance"].dtype):
        raise TypeError("genome_scores: relevance must have a numeric dtype")
    relevance = values["relevance"].to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(relevance).all() or not ((relevance >= 0) & (relevance <= 1)).all():
        raise ValueError("genome_scores: relevance must be finite and between 0 and 1")
    if values.duplicated(["movieId", "tagId"]).any():
        raise ValueError("genome_scores: (movieId, tagId) must be unique")

    tag_ids = tuple(sorted(int(value) for value in values["tagId"].unique()))
    if not tag_ids:
        empty = np.empty((0, 0), dtype=np.float32)
        return GenomeLookup({}, empty, empty, np.empty((0, 3), dtype=np.float32), ())
    if len(tag_ids) < 10:
        raise ValueError("genome_scores: at least 10 distinct tags are required")

    counts = values.groupby("movieId", sort=False)["tagId"].size()
    complete_ids = sorted(
        int(movie_id)
        for movie_id, count in counts.items()
        if count == len(tag_ids)
    )
    if not complete_ids:
        empty = np.empty((0, len(tag_ids)), dtype=np.float32)
        return GenomeLookup({}, empty, empty, np.empty((0, 3), dtype=np.float32), tag_ids)

    complete = values[values["movieId"].isin(complete_ids)]
    pivot = complete.pivot(index="movieId", columns="tagId", values="relevance").reindex(
        index=complete_ids, columns=tag_ids
    )
    valid = ~pivot.isna().any(axis=1)
    raw = pivot.loc[valid].to_numpy(dtype=np.float32, copy=True)
    movie_ids = [int(value) for value in pivot.index[valid]]
    norms = np.linalg.norm(raw.astype(np.float64), axis=1)
    nonzero = np.isfinite(norms) & (norms > 0)
    raw = raw[nonzero]
    norms = norms[nonzero]
    movie_ids = [movie_id for movie_id, keep in zip(movie_ids, nonzero, strict=True) if keep]
    normalized = (raw.astype(np.float64) / norms[:, None]).astype(np.float32)
    top_mean = np.partition(raw, raw.shape[1] - 10, axis=1)[:, -10:].mean(axis=1)
    statistics = np.column_stack(
        (raw.mean(axis=1), raw.std(axis=1, ddof=0), top_mean)
    ).astype(np.float32)
    return GenomeLookup(
        {movie_id: row for row, movie_id in enumerate(movie_ids)},
        normalized,
        raw,
        statistics,
        tag_ids,
    )


@dataclass(slots=True)
class GenomeHistoryState:
    """Sparse per-user sufficient state for Genome preference features."""

    lookup: GenomeLookup
    positive_sums: dict[int, np.ndarray] = field(default_factory=dict)
    negative_sums: dict[int, np.ndarray] = field(default_factory=dict)
    liked_rows: dict[int, list[int]] = field(default_factory=dict)

    def resolve(self, user_id: int, movie_id: int) -> tuple[float, ...]:
        row = self.lookup.movie_to_row.get(movie_id)
        if row is None:
            return (np.nan,) * len(GENOME_FEATURE_COLUMNS)

        target = self.lookup.normalized_vectors[row].astype(np.float64, copy=False)

        def centroid_cosine(sums: dict[int, np.ndarray]) -> float:
            vector_sum = sums.get(user_id)
            if vector_sum is None:
                return np.nan
            norm = float(np.linalg.norm(vector_sum))
            return float(np.dot(target, vector_sum) / norm) if norm > 0 else np.nan

        positive = centroid_cosine(self.positive_sums)
        negative = centroid_cosine(self.negative_sums)
        margin = positive - negative if np.isfinite(positive) and np.isfinite(negative) else np.nan
        liked = self.liked_rows.get(user_id)
        if liked:
            similarities = (
                self.lookup.normalized_vectors[np.asarray(liked, dtype=np.intp)]
                @ target
            )
            nearest = float(np.max(similarities))
            count = min(5, similarities.size)
            top5 = float(np.partition(similarities, similarities.size - count)[-count:].mean())
        else:
            nearest = np.nan
            top5 = np.nan
        static = self.lookup.static_statistics[row]
        return positive, negative, margin, nearest, top5, *(float(value) for value in static)

    def apply_timestamp_batch(self, ratings: list[tuple[int, int, float]]) -> None:
        """Apply a complete simultaneous batch after all rows have been scored."""
        width = len(self.lookup.tag_ids)
        # Sorting is a numeric reduction detail only. It makes simultaneous
        # updates independent of physical row order without using event IDs.
        for user_id, movie_id, rating in sorted(ratings):
            row = self.lookup.movie_to_row.get(movie_id)
            if row is None:
                continue
            raw = self.lookup.raw_vectors[row]
            destination = self.positive_sums if rating >= 4.0 else self.negative_sums
            if user_id not in destination:
                destination[user_id] = np.zeros(width, dtype=np.float64)
            destination[user_id] += raw
            if rating >= 4.0:
                self.liked_rows.setdefault(user_id, []).append(row)


def _empty_values(event_count: int) -> dict[str, np.ndarray]:
    return {
        column: np.full(event_count, np.nan, dtype=np.float32)
        for column in GENOME_FEATURE_COLUMNS
    }


def _feature_frame(context: pd.DataFrame, values: dict[str, np.ndarray]) -> pd.DataFrame:
    return pd.concat(
        [context.loc[:, CONTEXT_COLUMNS].reset_index(drop=True), pd.DataFrame(values)],
        axis=1,
    )


def _validate_events(rating_events: pd.DataFrame) -> None:
    required = {*CONTEXT_COLUMNS, "rating"}
    if missing := required - set(rating_events.columns):
        raise ValueError(f"rating_events: missing required columns: {sorted(missing)}")
    if rating_events[list(required)].isna().any().any():
        raise ValueError("rating_events: required columns must not contain missing values")
    if rating_events["ratingEventId"].duplicated().any():
        raise ValueError("rating_events: ratingEventId must be unique")


def iter_genome_features(
    rating_events: pd.DataFrame,
    genome_scores: pd.DataFrame,
    *,
    chunk_size: int = 250_000,
) -> Iterator[pd.DataFrame]:
    """Yield Genome rows in the same chronological chunks as the core builder."""
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    _validate_events(rating_events)
    lookup = build_genome_lookup(genome_scores)
    event_count = len(rating_events)
    if event_count == 0:
        yield _feature_frame(rating_events, _empty_values(0))
        return

    timestamps = rating_events["timestamp"].to_numpy(copy=False)
    order = np.argsort(timestamps, kind="stable")
    ordered_timestamps = timestamps[order]
    state = GenomeHistoryState(lookup)
    start = 0
    while start < event_count:
        end = min(start + chunk_size, event_count)
        while end < event_count and ordered_timestamps[end] == ordered_timestamps[end - 1]:
            end += 1
        chunk = (
            rating_events.iloc[order[start:end]]
            .loc[:, (*CONTEXT_COLUMNS, "rating")]
            .reset_index(drop=True)
        )
        values = _empty_values(len(chunk))
        chunk_timestamps = chunk["timestamp"].to_numpy(copy=False)
        boundaries = np.flatnonzero(chunk_timestamps[1:] != chunk_timestamps[:-1]) + 1
        starts = np.concatenate(([0], boundaries))
        ends = np.concatenate((boundaries, [len(chunk)]))
        for batch_start, batch_end in zip(starts, ends, strict=True):
            batch_ratings: list[tuple[int, int, float]] = []
            for position in range(int(batch_start), int(batch_end)):
                row = chunk.iloc[position]
                resolved = state.resolve(int(row.userId), int(row.movieId))
                for column, value in zip(GENOME_FEATURE_COLUMNS, resolved, strict=True):
                    values[column][position] = value
                batch_ratings.append((int(row.userId), int(row.movieId), float(row.rating)))
            state.apply_timestamp_batch(batch_ratings)
        yield _feature_frame(chunk, values)
        start = end


def build_genome_features(
    rating_events: pd.DataFrame, genome_scores: pd.DataFrame
) -> pd.DataFrame:
    """Build all eight Genome features and restore physical input row order."""
    chunks = list(
        iter_genome_features(
            rating_events, genome_scores, chunk_size=max(len(rating_events), 1)
        )
    )
    chronological = pd.concat(chunks, ignore_index=True)
    positions = pd.Series(
        np.arange(len(rating_events)),
        index=rating_events["ratingEventId"].astype("uint64"),
    )
    chronological["inputPosition"] = chronological["ratingEventId"].astype("uint64").map(positions)
    return (
        chronological.sort_values("inputPosition", kind="stable")
        .drop(columns="inputPosition")
        .reset_index(drop=True)
    )
