from pathlib import Path

import pandas as pd
import pytest

from src.data.build_parquet import read_and_validate_csv, validate_dataframe, verify_round_trip
from src.data.schemas import SCHEMAS


def write_csv(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def valid_ratings() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "userId": pd.Series([1, 2], dtype="uint32"),
            "movieId": pd.Series([10, 20], dtype="uint32"),
            "rating": pd.Series([0.5, 5.0], dtype="float32"),
            "timestamp": pd.to_datetime(["2005-01-01 00:00:00", "2005-01-02 00:00:00"]),
        }
    )


def rating_csv(user_id: str = "1", rating: str = "5.0") -> str:
    return f"userId,movieId,rating,timestamp\n{user_id},10,{rating},2005-01-01 00:00:00\n"


def test_valid_fixture_passes() -> None:
    validate_dataframe(valid_ratings(), "ratings", SCHEMAS["ratings"])


def test_literal_na_tag_is_preserved(tmp_path: Path) -> None:
    text = "userId,movieId,tag,timestamp\n1,10,NA,2005-01-01 00:00:00\n"
    frame = read_and_validate_csv(write_csv(tmp_path, "tag.csv", text), "tags", SCHEMAS["tags"])
    assert frame.loc[0, "tag"] == "NA"
    assert not pd.isna(frame.loc[0, "tag"])


def test_empty_nullable_tmdb_id_is_preserved(tmp_path: Path) -> None:
    path = write_csv(tmp_path, "link.csv", "movieId,imdbId,tmdbId\n1,114709,\n")
    frame = read_and_validate_csv(path, "links", SCHEMAS["links"])
    assert frame["tmdbId"].dtype == pd.UInt32Dtype()
    assert pd.isna(frame.loc[0, "tmdbId"])


@pytest.mark.parametrize("user_id", ["-1", "4294967297"])
def test_invalid_uint32_id_is_rejected(tmp_path: Path, user_id: str) -> None:
    path = write_csv(tmp_path, "rating.csv", rating_csv(user_id=user_id))
    with pytest.raises(ValueError, match="positive|exceeds maximum"):
        read_and_validate_csv(path, "ratings", SCHEMAS["ratings"])


@pytest.mark.parametrize("rating", ["4.2", "5.00000001", "0.499999999"])
def test_invalid_and_near_boundary_ratings_are_rejected(tmp_path: Path, rating: str) -> None:
    path = write_csv(tmp_path, "rating.csv", rating_csv(rating=rating))
    with pytest.raises(ValueError, match="invalid rating values"):
        read_and_validate_csv(path, "ratings", SCHEMAS["ratings"])


def test_relevance_range_is_checked_before_downcast(tmp_path: Path) -> None:
    text = "movieId,tagId,relevance\n1,2,1.00000001\n"
    path = write_csv(tmp_path, "genome_scores.csv", text)
    with pytest.raises(ValueError, match="relevance must be between"):
        read_and_validate_csv(path, "genome_scores", SCHEMAS["genome_scores"])


def test_invalid_timestamp_is_rejected(tmp_path: Path) -> None:
    text = "userId,movieId,rating,timestamp\n1,10,5.0,not-a-time\n"
    with pytest.raises(ValueError, match="invalid timestamps"):
        read_and_validate_csv(write_csv(tmp_path, "rating.csv", text), "ratings", SCHEMAS["ratings"])


def test_reordered_columns_are_accepted(tmp_path: Path) -> None:
    text = "timestamp,rating,movieId,userId\n2005-01-01 00:00:00,5.0,10,1\n"
    frame = read_and_validate_csv(write_csv(tmp_path, "rating.csv", text), "ratings", SCHEMAS["ratings"])
    assert list(frame.columns) == ["timestamp", "rating", "movieId", "userId"]


def test_missing_required_column_is_detected(tmp_path: Path) -> None:
    text = "userId,rating,timestamp\n1,5.0,2005-01-01 00:00:00\n"
    with pytest.raises(ValueError, match="missing required columns"):
        read_and_validate_csv(write_csv(tmp_path, "rating.csv", text), "ratings", SCHEMAS["ratings"])


def test_unexpected_column_is_detected(tmp_path: Path) -> None:
    text = "userId,movieId,rating,timestamp,extra\n1,10,5.0,2005-01-01 00:00:00,x\n"
    with pytest.raises(ValueError, match="unexpected columns"):
        read_and_validate_csv(write_csv(tmp_path, "rating.csv", text), "ratings", SCHEMAS["ratings"])


def test_composite_genome_score_key_is_validated() -> None:
    frame = pd.DataFrame(
        {
            "movieId": pd.Series([1, 1], dtype="uint32"),
            "tagId": pd.Series([2, 2], dtype="uint16"),
            "relevance": pd.Series([0.1, 0.2], dtype="float32"),
        }
    )
    with pytest.raises(ValueError, match="violate unique key"):
        validate_dataframe(frame, "genome_scores", SCHEMAS["genome_scores"])


def test_nullable_link_and_unique_key_behavior() -> None:
    frame = pd.DataFrame(
        {
            "movieId": pd.Series([1, 2], dtype="uint32"),
            "imdbId": pd.Series([10, 20], dtype="uint32"),
            "tmdbId": pd.Series([pd.NA, 30], dtype="UInt32"),
        }
    )
    validate_dataframe(frame, "links", SCHEMAS["links"])
    frame.loc[1, "movieId"] = 1
    with pytest.raises(ValueError, match="violate unique key"):
        validate_dataframe(frame, "links", SCHEMAS["links"])


def test_timezone_aware_timestamp_is_rejected() -> None:
    frame = valid_ratings()
    frame["timestamp"] = frame["timestamp"].dt.tz_localize("UTC")
    with pytest.raises(ValueError, match="timezone-naive"):
        validate_dataframe(frame, "ratings", SCHEMAS["ratings"])


def test_round_trip_preserves_schema_rows_and_values(tmp_path: Path) -> None:
    source = valid_ratings()
    path = tmp_path / "ratings.parquet"
    source.to_parquet(path, engine="pyarrow", compression="snappy", index=False)
    restored = verify_round_trip(source, path, "ratings", SCHEMAS["ratings"])
    pd.testing.assert_frame_equal(restored, source)


@pytest.mark.parametrize("change", ["value", "row_order"])
def test_round_trip_rejects_changed_values_or_row_order(tmp_path: Path, change: str) -> None:
    source = valid_ratings()
    changed = source.copy()
    if change == "value":
        changed.loc[0, "timestamp"] += pd.Timedelta(seconds=1)
    else:
        changed = changed.iloc[::-1].reset_index(drop=True)
    path = tmp_path / "ratings.parquet"
    changed.to_parquet(path, engine="pyarrow", compression="snappy", index=False)
    with pytest.raises(ValueError, match="values or row order changed"):
        verify_round_trip(source, path, "ratings", SCHEMAS["ratings"])
