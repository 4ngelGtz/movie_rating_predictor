"""Versioned predictor contracts. Historical rows are frozen literal definitions."""

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureContract:
    version: str
    # name, dtype, nullable, information role; tuple order is model input order.
    rows: tuple[tuple[str, str, bool, str], ...]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(row[0] for row in self.rows)

    @property
    def dtypes(self) -> dict[str, str]:
        return {row[0]: row[1] for row in self.rows}

    @property
    def nullable(self) -> tuple[str, ...]:
        return tuple(row[0] for row in self.rows if row[2])

    @property
    def classes(self) -> dict[str, tuple[str, ...]]:
        return {
            role: tuple(row[0] for row in self.rows if row[3] == role)
            for role in dict.fromkeys(row[3] for row in self.rows)
        }


_BASELINE_ROWS = (
    ("global_rating_count", "uint64", False, "strict_prior_historical"),
    ("global_mean_rating", "float32", False, "strict_prior_historical"),
    ("user_rating_count", "uint64", False, "strict_prior_historical"),
    ("user_mean_rating", "float32", False, "strict_prior_historical"),
    ("user_rating_std_pop", "float32", False, "strict_prior_historical"),
    ("user_seconds_since_last_rating", "float64", True, "strict_prior_historical"),
    ("movie_rating_count", "uint64", False, "strict_prior_historical"),
    ("movie_mean_rating", "float32", False, "strict_prior_historical"),
    ("movie_rating_std_pop", "float32", False, "strict_prior_historical"),
    ("movie_rating_count_30d", "uint64", False, "strict_prior_historical"),
    ("user_target_genre_rating_count", "uint64", False, "strict_prior_historical"),
    ("user_target_genre_mean_rating", "float32", False, "strict_prior_historical"),
    ("user_target_genre_mean_delta", "float32", False, "strict_prior_historical"),
    ("movie_genre_count", "uint8", False, "static_catalog"),
    ("movie_release_year", "Int16", True, "static_catalog"),
    ("movie_release_year_missing", "bool", False, "static_catalog"),
    ("movie_age_years", "float32", True, "event_context_from_static_catalog"),
)
_GENOME_ROWS = (
    ("genome_user_positive_cosine", "float32", True, "strict_prior_historical"),
    ("genome_user_negative_cosine", "float32", True, "strict_prior_historical"),
    ("genome_preference_margin", "float32", True, "strict_prior_historical"),
    ("genome_nearest_liked_similarity", "float32", True, "strict_prior_historical"),
    ("genome_top5_liked_similarity", "float32", True, "strict_prior_historical"),
    ("genome_movie_relevance_mean", "float32", True, "static_external_metadata"),
    ("genome_movie_relevance_std", "float32", True, "static_external_metadata"),
    ("genome_movie_top10_mean", "float32", True, "static_external_metadata"),
)
_ROLLING_ROWS = (
    ("movie_rating_count_60d", "uint64", False, "strict_prior_historical"),
    ("movie_rating_avg_30d", "float32", True, "strict_prior_historical"),
    ("movie_rating_avg_60d", "float32", True, "strict_prior_historical"),
    ("movie_rating_avg_30d_over_60d", "float32", True, "strict_prior_historical"),
    ("movie_rating_count_30d_over_60d", "float32", True, "strict_prior_historical"),
)

BASELINE_17 = FeatureContract("FEATURE_DICTIONARY_V1", _BASELINE_ROWS)
GENOME_25 = FeatureContract(
    "FEATURE_DICTIONARY_V1_GENOME_ADDENDUM", (*_BASELINE_ROWS, *_GENOME_ROWS)
)
PRD_30 = FeatureContract(
    "FEATURE_DICTIONARY_V1_PRD_30",
    (*_BASELINE_ROWS, *_ROLLING_ROWS, *_GENOME_ROWS),
)
CONTRACTS = (BASELINE_17, GENOME_25, PRD_30)


def contract_for_version(version: str) -> FeatureContract:
    for contract in CONTRACTS:
        if contract.version == version:
            return contract
    raise ValueError(f"unsupported feature contract version: {version}")


def contract_for_names(names: tuple[str, ...]) -> FeatureContract:
    for contract in CONTRACTS:
        if contract.names == names:
            return contract
    raise ValueError("model input feature names or order do not match a contract")
