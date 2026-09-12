import pandas as pd

from src.data.audit_temporal import audit_ratings


def test_audit_metrics_have_explicit_duplicate_definitions() -> None:
    ratings = pd.DataFrame(
        {
            "userId": pd.Series([1, 1, 1, 2, 2], dtype="uint32"),
            "movieId": pd.Series([10, 11, 10, 10, 10], dtype="uint32"),
            "rating": pd.Series([4, 3, 4, 5, 4], dtype="float32"),
            "timestamp": pd.to_datetime(
                [
                    "2020-01-01 10:00:00",
                    "2020-01-01 10:00:00",
                    "2020-01-01 10:00:00",
                    "2020-01-02 10:00:00",
                    "2020-01-03 10:00:00",
                ]
            ),
        }
    )
    result = audit_ratings(ratings, parquet_storage_unit="timestamp[us]")
    assert result["row_count"] == 5
    assert result["duplicate_full_rows_beyond_first"] == 1
    assert result["repeated_user_movie_pairs"] == 2
    assert result["rows_in_repeated_user_movie_pairs"] == 4
    assert result["timestamps_with_multiple_events"] == 1
    assert result["maximum_events_at_one_timestamp"] == 3
    assert result["users_with_multiple_events_at_same_timestamp"] == 1
    assert result["declared_source_timestamp_resolution"] == "1 second"
    assert result["timestamps_aligned_to_declared_resolution"] is True


def test_audit_does_not_claim_subsecond_values_match_declared_resolution() -> None:
    ratings = pd.DataFrame(
        {
            "userId": pd.Series([1], dtype="uint32"),
            "movieId": pd.Series([10], dtype="uint32"),
            "rating": pd.Series([4], dtype="float32"),
            "timestamp": pd.to_datetime(["2020-01-01 10:00:00.123"]),
        }
    )

    result = audit_ratings(ratings)

    assert result["declared_source_timestamp_resolution"] == "1 second"
    assert result["timestamps_aligned_to_declared_resolution"] is False
