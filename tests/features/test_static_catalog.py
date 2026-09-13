import re

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from src.features.catalog import catalog_snapshot_id
from src.features.expanding import build_expanding_rating_features


T = pd.Timestamp("2020-01-01 00:00:00")


def events(rows: list[tuple[int, int, int, float, pd.Timestamp]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ratingEventId": pd.Series([row[0] for row in rows], dtype="UInt64"),
            "userId": pd.Series([row[1] for row in rows], dtype="uint32"),
            "movieId": pd.Series([row[2] for row in rows], dtype="uint32"),
            "rating": pd.Series([row[3] for row in rows], dtype="float32"),
            "timestamp": pd.to_datetime([row[4] for row in rows]),
        }
    )


def movies(
    rows: list[tuple[int, str, int | None, str]],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "movieId": pd.Series([row[0] for row in rows], dtype="uint32"),
            "title": pd.Series([row[1] for row in rows], dtype="string"),
            "releaseYear": pd.Series([row[2] for row in rows], dtype="UInt16"),
            "genreStatus": pd.Series([row[3] for row in rows], dtype="string"),
            "imdbId": pd.Series([pd.NA] * len(rows), dtype="UInt32"),
            "tmdbId": pd.Series([pd.NA] * len(rows), dtype="UInt32"),
        }
    )


def bridge(rows: list[tuple[int, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "movieId": pd.Series([row[0] for row in rows], dtype="uint32"),
            "genreId": pd.Series([row[1] for row in rows], dtype="string"),
        }
    )


def aligned(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values("ratingEventId").reset_index(drop=True)


def test_static_features_follow_canonical_genre_and_year_semantics() -> None:
    source = events(
        [
            (1, 1, 10, 4.0, T),
            (2, 1, 20, 3.0, T),
            (3, 2, 30, 2.0, T),
        ]
    )
    catalog = movies(
        [
            (10, "Single (2000)", 2000, "listed"),
            (20, "Multiple (1995)", 1995, "listed"),
            (30, "Malformed (20xx)", None, "missing_in_source"),
        ]
    )
    result = build_expanding_rating_features(
        source,
        bridge([(10, "Drama"), (20, "Romance"), (20, "Comedy")]),
        catalog,
    )

    assert result["movie_genre_count"].tolist() == [1, 2, 0]
    assert result["movie_release_year"].tolist() == [2000, 1995, pd.NA]
    assert result["movie_release_year_missing"].tolist() == [False, False, True]
    assert result.loc[0, "movie_age_years"] == pytest.approx(
        (T - pd.Timestamp("2000-01-01")) / pd.Timedelta(days=365.2425)
    )
    assert np.isnan(result.loc[2, "movie_age_years"])


def test_movie_age_uses_scoring_time_and_preserves_negative_inconsistency() -> None:
    source = events(
        [
            (1, 1, 10, 4.0, pd.Timestamp("2010-01-01")),
            (2, 2, 10, 4.0, pd.Timestamp("2011-01-01")),
            (3, 3, 20, 4.0, pd.Timestamp("2010-01-01")),
            (4, 4, 20, 4.0, pd.Timestamp("2010-01-01")),
        ]
    )
    catalog = movies(
        [
            (10, "Past (2000)", 2000, "listed"),
            (20, "Future (2012)", 2012, "listed"),
        ]
    )
    result = build_expanding_rating_features(
        source, bridge([(10, "Drama"), (20, "Drama")]), catalog
    )

    expected = [
        (timestamp - pd.Timestamp(f"{year}-01-01")) / pd.Timedelta(days=365.2425)
        for timestamp, year in zip(source["timestamp"], [2000, 2000, 2012, 2012])
    ]
    assert result["movie_age_years"].tolist() == pytest.approx(expected, abs=1e-6)
    assert result.loc[0, "movie_age_years"] != result.loc[1, "movie_age_years"]
    assert result.loc[2, "movie_age_years"] == result.loc[3, "movie_age_years"]
    assert result.loc[2, "movie_age_years"] < 0


def test_boundary_year_age_is_independent_of_datetime_storage_resolution() -> None:
    rows = [
        (1, 1, 10, 4.0, pd.Timestamp("2020-01-01")),
        (2, 1, 20, 3.0, pd.Timestamp("2020-01-02")),
        (3, 1, 30, 2.0, pd.Timestamp("2020-01-03")),
    ]
    source_us = events(rows)
    source_us["timestamp"] = source_us["timestamp"].astype("datetime64[us]")
    source_ns = events(rows)
    source_ns["timestamp"] = source_ns["timestamp"].astype("datetime64[ns]")
    catalog = movies(
        [
            (10, "Ordinary (2000)", 2000, "listed"),
            (20, "Early (0001)", 1, "listed"),
            (30, "Late (9999)", 9999, "listed"),
        ]
    )
    memberships = bridge([(10, "Drama"), (20, "Drama"), (30, "Drama")])

    result_us = build_expanding_rating_features(source_us, memberships, catalog)
    result_ns = build_expanding_rating_features(source_ns, memberships, catalog)

    assert result_ns["movie_release_year"].tolist() == [2000, 1, 9999]
    assert result_ns["movie_age_years"].tolist() == result_us[
        "movie_age_years"
    ].tolist()
    pdt.assert_frame_equal(
        result_ns.drop(columns="timestamp"),
        result_us.drop(columns="timestamp"),
    )


def test_phase_2_year_zero_missingness_is_accepted_consistently() -> None:
    source = events([(1, 1, 10, 4.0, T)])
    catalog = movies([(10, "Invalid calendar year (0000)", None, "listed")])
    result = build_expanding_rating_features(
        source, bridge([(10, "Drama")]), catalog
    )

    assert pd.isna(result.loc[0, "movie_release_year"])
    assert result.loc[0, "movie_release_year_missing"]
    assert np.isnan(result.loc[0, "movie_age_years"])


def test_phase_4d_dtypes_and_empty_output_match_dictionary() -> None:
    source = events([(1, 1, 10, 4.0, T)])
    catalog = movies([(10, "Unknown year", None, "missing_in_source")])
    result = build_expanding_rating_features(source, bridge([]), catalog)
    empty = build_expanding_rating_features(events([]), bridge([]), movies([]))

    for frame in (result, empty):
        assert str(frame["movie_genre_count"].dtype) == "uint8"
        assert str(frame["movie_release_year"].dtype) == "Int16"
        assert str(frame["movie_release_year_missing"].dtype) == "bool"
        assert str(frame["movie_age_years"].dtype) == "float32"


def test_all_feature_families_integrate_without_changing_dynamic_values() -> None:
    source = events(
        [
            (1, 1, 10, 2.0, T),
            (2, 1, 20, 4.0, T + pd.Timedelta(seconds=10)),
        ]
    )
    catalog = movies(
        [
            (10, "First (2000)", 2000, "listed"),
            (20, "Second (2010)", 2010, "listed"),
        ]
    )
    result = build_expanding_rating_features(
        source, bridge([(10, "Drama"), (20, "Drama")]), catalog
    )

    second = result.iloc[1]
    assert second["global_rating_count"] == 1
    assert second["user_rating_count"] == 1
    assert second["movie_rating_count"] == 0
    assert second["user_target_genre_rating_count"] == 1
    assert second["movie_genre_count"] == 1
    assert second["movie_release_year"] == 2010


def test_catalog_and_rating_order_are_invariant_and_events_are_conserved() -> None:
    source = events(
        [
            (1, 1, 10, 2.0, T),
            (2, 1, 20, 4.0, T + pd.Timedelta(seconds=1)),
            (3, 2, 10, 5.0, T + pd.Timedelta(seconds=1)),
        ]
    )
    catalog = movies(
        [
            (10, "First (2000)", 2000, "listed"),
            (20, "Second (2001)", 2001, "listed"),
        ]
    )
    memberships = bridge([(10, "Drama"), (10, "Comedy"), (20, "Drama")])
    expected = aligned(build_expanding_rating_features(source, memberships, catalog))
    shuffled = build_expanding_rating_features(
        source.sample(frac=1, random_state=1).reset_index(drop=True),
        memberships.sample(frac=1, random_state=2).reset_index(drop=True),
        catalog.sample(frac=1, random_state=3).reset_index(drop=True),
    )

    pdt.assert_frame_equal(expected, aligned(shuffled))
    assert len(shuffled) == len(source)
    assert set(shuffled["ratingEventId"]) == set(source["ratingEventId"])


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda frame: pd.concat([frame, frame.iloc[[0]]], ignore_index=True),
            "movieId must be unique",
        ),
        (
            lambda frame: frame.assign(
                movieId=pd.Series([pd.NA], dtype="UInt32")
            ),
            "movieId must not contain",
        ),
        (
            lambda frame: frame.assign(
                releaseYear=frame["releaseYear"].astype("string")
            ),
            "nullable integer dtype",
        ),
        (
            lambda frame: frame.assign(
                releaseYear=pd.Series([0], dtype="UInt16")
            ),
            "four-digit parsed value",
        ),
    ],
)
def test_malformed_canonical_movies_fail_loudly(
    mutate, message: str, monkeypatch
) -> None:
    source = events([(1, 1, 10, 4.0, T)])
    catalog = movies([(10, "Movie (2000)", 2000, "listed")])
    monkeypatch.setattr(
        "src.features.expanding.HistoricalRatingState",
        lambda: pytest.fail("dynamic state created before catalog validation"),
    )
    with pytest.raises((TypeError, ValueError), match=message):
        build_expanding_rating_features(source, bridge([(10, "Drama")]), mutate(catalog))


def test_catalog_omission_unknown_target_and_invalid_bridge_fail_loudly() -> None:
    source = events([(1, 1, 10, 4.0, T)])
    catalog = movies([(10, "Movie (2000)", 2000, "listed")])
    with pytest.raises(ValueError, match="canonical_movies must be explicitly supplied"):
        build_expanding_rating_features(source, bridge([(10, "Drama")]))
    with pytest.raises(ValueError, match="do not resolve to canonical_movies"):
        build_expanding_rating_features(
            source,
            bridge([(20, "Drama")]),
            movies([(20, "Other (2000)", 2000, "listed")]),
        )
    with pytest.raises(ValueError, match="do not resolve to canonical_movies"):
        build_expanding_rating_features(
            source, bridge([(10, "Drama"), (20, "Comedy")]), catalog
        )


def test_genre_status_and_memberships_must_be_consistent() -> None:
    source = events([(1, 1, 10, 4.0, T)])
    with pytest.raises(ValueError, match="listed movie must have"):
        build_expanding_rating_features(
            source, bridge([]), movies([(10, "Movie (2000)", 2000, "listed")])
        )
    with pytest.raises(ValueError, match="missing_in_source movie must have no"):
        build_expanding_rating_features(
            source,
            bridge([(10, "Drama")]),
            movies([(10, "Movie (2000)", 2000, "missing_in_source")]),
        )


def test_compound_genre_fails_before_dynamic_state_creation(monkeypatch) -> None:
    source = events(
        [
            (1, 1, 10, 4.0, T),
            (2, 1, 10, 3.0, T + pd.Timedelta(seconds=1)),
        ]
    )
    catalog = movies([(10, "Movie (2000)", 2000, "listed")])

    def state_must_not_be_created():
        raise AssertionError("dynamic state was created before catalog validation")

    monkeypatch.setattr(
        "src.features.expanding.HistoricalRatingState",
        state_must_not_be_created,
    )
    with pytest.raises(ValueError, match="one normalized genre token"):
        build_expanding_rating_features(
            source,
            bridge([(10, "Drama|Action")]),
            catalog,
        )


def test_snapshot_identity_is_order_invariant_and_content_sensitive() -> None:
    catalog = movies(
        [
            (10, "First (2000)", 2000, "listed"),
            (20, "Second (2001)", 2001, "listed"),
        ]
    )
    memberships = bridge([(10, "Drama"), (10, "Comedy"), (20, "Drama")])
    digest = catalog_snapshot_id(catalog, memberships)

    assert re.fullmatch(r"[0-9a-f]{64}", digest)
    assert digest == catalog_snapshot_id(
        catalog.iloc[::-1].reset_index(drop=True),
        memberships.iloc[::-1].reset_index(drop=True),
    )
    changed_metadata = catalog.copy()
    changed_metadata.loc[0, "title"] = "Corrected title (2000)"
    assert digest != catalog_snapshot_id(changed_metadata, memberships)
    changed_memberships = memberships.copy()
    changed_memberships.loc[0, "genreId"] = "Action"
    assert digest != catalog_snapshot_id(catalog, changed_memberships)
