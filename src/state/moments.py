"""Serializable sufficient statistics for expanding rating histories.

``M2`` is the sum of squared deviations from ``mean``.  For ``count > 0``,
population variance is exactly ``M2 / count`` (never ``M2 / (count - 1)``).
The empty-state mean is intentionally internal; feature fallbacks are resolved
by the feature builder rather than encoded as observations in this state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Iterable, Mapping


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
    """Sparse global, user, and movie moments for complete timestamp batches."""

    global_moments: RunningMoments = field(default_factory=RunningMoments)
    user_moments: dict[int, RunningMoments] = field(default_factory=dict)
    movie_moments: dict[int, RunningMoments] = field(default_factory=dict)

    def apply_timestamp_batch(
        self,
        ratings: Iterable[tuple[int, int, float]],
    ) -> None:
        """Apply one already-scored timestamp batch atomically to all states.

        Each tuple is ``(userId, movieId, rating)`` for one canonical event.
        The caller is responsible for passing each event exactly once.  Batch
        moments are reduced independently of input order before being merged.
        """
        batch = [
            (int(user_id), int(movie_id), float(rating))
            for user_id, movie_id, rating in ratings
        ]
        if not batch:
            return

        self.global_moments.merge(
            RunningMoments.from_values(rating for _, _, rating in batch)
        )

        users: dict[int, list[float]] = {}
        movies: dict[int, list[float]] = {}
        for user_id, movie_id, rating in batch:
            users.setdefault(user_id, []).append(rating)
            movies.setdefault(movie_id, []).append(rating)

        for user_id in sorted(users):
            self.user_moments.setdefault(user_id, RunningMoments()).merge(
                RunningMoments.from_values(users[user_id])
            )
        for movie_id in sorted(movies):
            self.movie_moments.setdefault(movie_id, RunningMoments()).merge(
                RunningMoments.from_values(movies[movie_id])
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
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> HistoricalRatingState:
        """Restore a deterministic checkpoint representation."""
        if set(payload) != {"global", "users", "movies"}:
            raise ValueError("state payload must contain exactly global, users, and movies")

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
        return state
