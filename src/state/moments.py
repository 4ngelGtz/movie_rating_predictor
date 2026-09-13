"""Serializable sufficient statistics for expanding rating histories.

``M2`` is the sum of squared deviations from ``mean``.  For ``count > 0``,
population variance is exactly ``M2 / count`` (never ``M2 / (count - 1)``).
The empty-state mean is intentionally internal; feature fallbacks are resolved
by the feature builder rather than encoded as observations in this state.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
import math
from typing import Any

import pandas as pd


MOVIE_ACTIVITY_WINDOW = pd.Timedelta(days=30)


def _timestamp(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError("timestamp must not be missing")
    if timestamp.tz is not None:
        raise ValueError("timestamp must be timezone-naive")
    return timestamp


@dataclass(slots=True)
class RunningMoments:
    """Mergeable Welford-style ``count``, ``mean``, and ``M2`` state."""

    count: int = 0
    mean: float = 0.0
    M2: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.count, bool) or not isinstance(self.count, int):
            raise TypeError("count must be an integer")
        if self.count < 0:
            raise ValueError("count must be nonnegative")
        if not math.isfinite(self.mean) or not math.isfinite(self.M2):
            raise ValueError("mean and M2 must be finite")
        if self.M2 < 0.0:
            raise ValueError("M2 must be nonnegative")
        if self.count == 0 and (self.mean != 0.0 or self.M2 != 0.0):
            raise ValueError("empty moments must have mean=0.0 and M2=0.0")

    @classmethod
    def from_values(cls, values: Iterable[float]) -> RunningMoments:
        """Build deterministic batch moments from a multiset of values.

        Numeric sorting is a reduction detail, not a temporal ordering rule.
        It makes merging a simultaneous batch independent of its physical row
        order.  MovieLens half-star ratings are represented exactly in binary.
        """
        ordered = sorted(float(value) for value in values)
        if not ordered:
            return cls()
        if not all(math.isfinite(value) for value in ordered):
            raise ValueError("moment values must be finite")
        count = len(ordered)
        mean = math.fsum(ordered) / count
        M2 = math.fsum((value - mean) ** 2 for value in ordered)
        return cls(count=count, mean=mean, M2=M2)

    def merge(self, other: RunningMoments) -> None:
        """Merge disjoint observations using the parallel Welford recurrence."""
        if other.count == 0:
            return
        if self.count == 0:
            self.count = other.count
            self.mean = other.mean
            self.M2 = other.M2
            return

        combined_count = self.count + other.count
        delta = other.mean - self.mean
        self.M2 += other.M2 + delta * delta * self.count * other.count / combined_count
        self.mean += delta * other.count / combined_count
        self.count = combined_count

    @property
    def population_variance(self) -> float:
        """Return ``M2 / N``; empty and singleton states resolve to zero."""
        if self.count < 2:
            return 0.0
        # Guard only against a tiny negative caused by floating-point roundoff.
        return max(self.M2 / self.count, 0.0)

    @property
    def population_std(self) -> float:
        return math.sqrt(self.population_variance)

    def to_dict(self) -> dict[str, int | float]:
        """Return a JSON-compatible representation with stable field order."""
        return {"count": self.count, "mean": self.mean, "M2": self.M2}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RunningMoments:
        """Restore state produced by :meth:`to_dict`."""
        if set(payload) != {"count", "mean", "M2"}:
            raise ValueError("moments payload must contain exactly count, mean, and M2")
        return cls(
            count=int(payload["count"]),
            mean=float(payload["mean"]),
            M2=float(payload["M2"]),
        )


@dataclass(slots=True)
class HistoricalRatingState:
    """Sparse rating moments and temporal state for complete timestamp batches."""

    global_moments: RunningMoments = field(default_factory=RunningMoments)
    user_moments: dict[int, RunningMoments] = field(default_factory=dict)
    movie_moments: dict[int, RunningMoments] = field(default_factory=dict)
    user_genre_moments: dict[tuple[int, str], RunningMoments] = field(
        default_factory=dict
    )
    last_user_rating: dict[int, pd.Timestamp] = field(default_factory=dict)
    movie_rating_count_30d: dict[int, int] = field(default_factory=dict)
    _movie_rating_batches_30d: deque[
        tuple[pd.Timestamp, tuple[tuple[int, int], ...]]
    ] = field(default_factory=deque)

    def expire_movie_activity(self, prediction_timestamp: Any) -> None:
        """Expire events strictly older than ``t - 30 days``.

        Events exactly on the left boundary remain eligible. Timestamp batches
        are stored once, so each batch enters and leaves the queue once.
        """
        boundary = _timestamp(prediction_timestamp) - MOVIE_ACTIVITY_WINDOW
        while (
            self._movie_rating_batches_30d
            and self._movie_rating_batches_30d[0][0] < boundary
        ):
            _, movie_counts = self._movie_rating_batches_30d.popleft()
            for movie_id, count in movie_counts:
                remaining = self.movie_rating_count_30d[movie_id] - count
                if remaining:
                    self.movie_rating_count_30d[movie_id] = remaining
                else:
                    del self.movie_rating_count_30d[movie_id]

    def apply_timestamp_batch(
        self,
        ratings: Iterable[tuple[int, int, float]],
        *,
        timestamp: Any | None = None,
        movie_genres: Mapping[int, tuple[str, ...]] | None = None,
    ) -> None:
        """Apply one already-scored timestamp batch atomically to all states.

        Each tuple is ``(userId, movieId, rating)`` for one canonical event.
        The caller is responsible for passing each event exactly once.  Batch
        moments are reduced independently of input order before being merged.
        When ``timestamp`` is supplied, recency and rolling state are updated
        for the complete batch as well. When ``movie_genres`` is supplied,
        each canonical event additionally updates one sparse user-genre state
        per distinct genre associated with its movie. These relationship
        updates do not alter canonical global, user, or movie counts.
        """
        if movie_genres is not None:
            for movie_id, genre_ids in movie_genres.items():
                if len(genre_ids) != len(set(genre_ids)):
                    raise ValueError(
                        "movie_genres memberships must be distinct for each movie; "
                        f"duplicate found for movieId {movie_id}"
                    )

        batch = [
            (int(user_id), int(movie_id), float(rating))
            for user_id, movie_id, rating in ratings
        ]
        if not batch:
            return
        batch_timestamp = _timestamp(timestamp) if timestamp is not None else None

        self.global_moments.merge(
            RunningMoments.from_values(rating for _, _, rating in batch)
        )

        users: dict[int, list[float]] = {}
        movies: dict[int, list[float]] = {}
        user_genres: dict[tuple[int, str], list[float]] = {}
        for user_id, movie_id, rating in batch:
            users.setdefault(user_id, []).append(rating)
            movies.setdefault(movie_id, []).append(rating)
            if movie_genres is not None:
                for genre_id in movie_genres.get(movie_id, ()):
                    user_genres.setdefault((user_id, genre_id), []).append(rating)

        for user_id in sorted(users):
            self.user_moments.setdefault(user_id, RunningMoments()).merge(
                RunningMoments.from_values(users[user_id])
            )
        for movie_id in sorted(movies):
            self.movie_moments.setdefault(movie_id, RunningMoments()).merge(
                RunningMoments.from_values(movies[movie_id])
            )
        for key in sorted(user_genres):
            self.user_genre_moments.setdefault(key, RunningMoments()).merge(
                RunningMoments.from_values(user_genres[key])
            )

        if batch_timestamp is not None:
            for user_id in sorted(users):
                self.last_user_rating[user_id] = batch_timestamp
            movie_counts = tuple(
                (movie_id, len(movies[movie_id])) for movie_id in sorted(movies)
            )
            self._movie_rating_batches_30d.append((batch_timestamp, movie_counts))
            for movie_id, count in movie_counts:
                self.movie_rating_count_30d[movie_id] = (
                    self.movie_rating_count_30d.get(movie_id, 0) + count
                )

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic, JSON-compatible checkpoint content."""
        return {
            "global": self.global_moments.to_dict(),
            "users": [
                {"userId": key, **self.user_moments[key].to_dict()}
                for key in sorted(self.user_moments)
            ],
            "movies": [
                {"movieId": key, **self.movie_moments[key].to_dict()}
                for key in sorted(self.movie_moments)
            ],
            "userGenres": [
                {
                    "userId": user_id,
                    "genreId": genre_id,
                    **self.user_genre_moments[(user_id, genre_id)].to_dict(),
                }
                for user_id, genre_id in sorted(self.user_genre_moments)
            ],
            "lastUserRatings": [
                {"userId": key, "timestamp": self.last_user_rating[key].isoformat()}
                for key in sorted(self.last_user_rating)
            ],
            "movieRatingBatches30d": [
                {
                    "timestamp": timestamp.isoformat(),
                    "movies": [
                        {"movieId": movie_id, "count": count}
                        for movie_id, count in movie_counts
                    ],
                }
                for timestamp, movie_counts in self._movie_rating_batches_30d
            ],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> HistoricalRatingState:
        """Restore a deterministic checkpoint representation."""
        phase_4a_keys = {"global", "users", "movies"}
        phase_4b_keys = phase_4a_keys | {
            "lastUserRatings",
            "movieRatingBatches30d",
        }
        phase_4c_keys = phase_4b_keys | {"userGenres"}
        if set(payload) not in (phase_4a_keys, phase_4b_keys, phase_4c_keys):
            raise ValueError("state payload has unexpected or missing fields")

        state = cls(global_moments=RunningMoments.from_dict(payload["global"]))
        for row in payload["users"]:
            user_id = int(row["userId"])
            if user_id in state.user_moments:
                raise ValueError(f"duplicate userId in state payload: {user_id}")
            state.user_moments[user_id] = RunningMoments.from_dict(
                {key: row[key] for key in ("count", "mean", "M2")}
            )
        for row in payload["movies"]:
            movie_id = int(row["movieId"])
            if movie_id in state.movie_moments:
                raise ValueError(f"duplicate movieId in state payload: {movie_id}")
            state.movie_moments[movie_id] = RunningMoments.from_dict(
                {key: row[key] for key in ("count", "mean", "M2")}
            )

        for row in payload.get("userGenres", []):
            user_id = int(row["userId"])
            genre_id = str(row["genreId"])
            key = (user_id, genre_id)
            if not genre_id or key in state.user_genre_moments:
                raise ValueError("user-genre state keys must be unique and nonempty")
            state.user_genre_moments[key] = RunningMoments.from_dict(
                {name: row[name] for name in ("count", "mean", "M2")}
            )

        for row in payload.get("lastUserRatings", []):
            user_id = int(row["userId"])
            if user_id in state.last_user_rating:
                raise ValueError(f"duplicate userId in recency payload: {user_id}")
            state.last_user_rating[user_id] = _timestamp(row["timestamp"])

        previous_timestamp: pd.Timestamp | None = None
        for row in payload.get("movieRatingBatches30d", []):
            timestamp = _timestamp(row["timestamp"])
            if previous_timestamp is not None and timestamp <= previous_timestamp:
                raise ValueError("rolling timestamp batches must be strictly increasing")
            previous_timestamp = timestamp
            seen_movies: set[int] = set()
            movie_counts: list[tuple[int, int]] = []
            for movie in row["movies"]:
                movie_id = int(movie["movieId"])
                count = int(movie["count"])
                if movie_id in seen_movies or count <= 0:
                    raise ValueError("rolling batch movie keys must be unique and positive")
                seen_movies.add(movie_id)
                movie_counts.append((movie_id, count))
                state.movie_rating_count_30d[movie_id] = (
                    state.movie_rating_count_30d.get(movie_id, 0) + count
                )
            state._movie_rating_batches_30d.append((timestamp, tuple(movie_counts)))
        return state
