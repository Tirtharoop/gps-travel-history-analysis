import numpy as np
import pandas as pd

from gps_travel.clustering import compare_clustering, detect_places
from gps_travel.models import TravelData
from gps_travel.parser import parse_timeline
from gps_travel.pipeline import analyze
from gps_travel.sample_data import generate_classic_timeline, generate_sample_timeline
from gps_travel.statistics import (
    daily_stats,
    mode_summary,
    monthly_stats,
    overview_stats,
    repeated_routes,
    weekday_weekend_stats,
    weekly_stats,
)


def _stay_frame():
    rows = []
    base = pd.Timestamp("2026-01-01T08:00:00Z")
    clusters = [
        (12.9716, 77.5946, 5, 3600),
        (12.9352, 77.6245, 4, 7200),
        (12.9780, 77.6000, 2, 1800),
    ]
    index = 0
    for lat, lon, count, duration in clusters:
        for offset in range(count):
            start = base + pd.Timedelta(days=index, hours=offset)
            rows.append(
                {
                    "segment_id": index,
                    "segment_type": "stay",
                    "start_ts": start,
                    "end_ts": start + pd.Timedelta(seconds=duration),
                    "mode": "unknown",
                    "distance_m": 0.0,
                    "duration_s": float(duration),
                    "start_lat": lat + offset * 0.00002,
                    "start_lon": lon + offset * 0.00002,
                    "end_lat": lat,
                    "end_lon": lon,
                    "place_id": "",
                    "place_name": "",
                    "semantic_type": "",
                    "probability": np.nan,
                    "source": "test",
                }
            )
            index += 1
    return pd.DataFrame(rows)


def test_dbscan_and_kmeans_detect_places():
    stays = _stay_frame()
    places, clustered = detect_places(stays, eps_m=200, min_samples=2, algorithm="dbscan")
    assert len(places) == 3
    assert places["visit_count"].sum() == 11
    assert clustered["cluster"].ge(0).sum() == 11
    kmeans_places, _ = detect_places(stays, algorithm="kmeans", n_clusters=3)
    assert len(kmeans_places) == 3
    comparison = compare_clustering(stays, eps_m=200, min_samples=2, n_clusters=3)
    assert set(comparison["algorithm"]) == {"dbscan", "kmeans"}
    assert comparison["clusters"].between(1, 3).all()


def _segment_frame():
    rows = []
    base = pd.Timestamp("2026-01-01T08:00:00Z")
    specs = [
        (0, base, 600, 3000, "car"),
        (1, base + pd.Timedelta(hours=1), 900, 1500, "walking"),
        (2, base + pd.Timedelta(days=1, hours=2), 1200, 8000, "bus"),
        (3, base + pd.Timedelta(days=40), 300, 500, "walking"),
    ]
    for segment_id, start, duration, distance, mode in specs:
        rows.append(
            {
                "segment_id": segment_id,
                "segment_type": "move",
                "start_ts": start,
                "end_ts": start + pd.Timedelta(seconds=duration),
                "mode": mode,
                "distance_m": float(distance),
                "duration_s": float(duration),
                "start_lat": 12.9716,
                "start_lon": 77.5946,
                "end_lat": 12.9352,
                "end_lon": 77.6245,
                "place_id": "",
                "place_name": "",
                "semantic_type": "",
                "probability": np.nan,
                "source": "test",
            }
        )
    return pd.DataFrame(rows)


def test_statistics_aggregations():
    segments = _segment_frame()
    daily = daily_stats(segments)
    assert daily["distance_km"].sum() == 13.0
    assert daily["journeys"].sum() == 4
    weekly = weekly_stats(segments)
    assert len(weekly) == 2
    monthly = monthly_stats(segments)
    assert len(monthly) == 2
    modes = mode_summary(segments)
    assert modes["journeys"].sum() == 4
    assert abs(modes["share_distance"].sum() - 1.0) < 1e-9
    weekend = weekday_weekend_stats(segments)
    assert set(weekend["period"]).issubset({"Weekday", "Weekend"})
    routes = repeated_routes(segments)
    assert not routes.empty
    data = TravelData(segments=segments)
    summary = overview_stats(data, pd.DataFrame(), daily=daily, modes=modes)
    assert summary["journeys"] == 4
    assert summary["total_distance_km"] == 13.0


def test_pipeline_on_sample_data():
    payload = generate_sample_timeline(days=21, seed=4)
    data = parse_timeline(payload)
    result = analyze(data, eps_m=200, min_samples=2, predict_modes=True)
    assert result.overview["journeys"] > 0
    assert len(result.places) >= 2
    assert not result.daily.empty
    assert not result.mode_summary.empty
    assert result.classifier.trained
    assert 0.0 <= result.classifier.metrics["accuracy"] <= 1.0
    assert "mode_final" in result.data.segments.columns
    assert (result.data.segments["mode_source"] == "timeline").any()


def test_pipeline_on_classic_sample_data():
    payload = generate_classic_timeline(days=10, seed=5)
    data = parse_timeline(payload)
    assert data.meta["source_format"] == "classic_locations"
    assert not data.points.empty
    result = analyze(data, eps_m=220, min_samples=2)
    assert result.overview["journeys"] > 0
    assert len(result.places) >= 2
    assert result.classifier.trained
    assert "mode_final" in result.data.segments.columns


def test_filtered_travel_data():
    payload = generate_sample_timeline(days=5, seed=9)
    data = parse_timeline(payload)
    start = data.start_time.date()
    filtered = data.filtered(start, start)
    assert not filtered.segments.empty
    assert filtered.segments["start_ts"].dt.date.min() == start
    assert filtered.segments["start_ts"].dt.date.max() == start
