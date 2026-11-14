import pandas as pd

from gps_travel.geo import haversine, parse_latlng, robust_path_distance
from gps_travel.parser import to_timestamp


def test_parse_latlng_variants():
    assert parse_latlng("geo:12.9716,77.5946") == (12.9716, 77.5946)
    assert parse_latlng("12.9716°, 77.5946°") == (12.9716, 77.5946)
    assert parse_latlng({"latitudeE7": 129716000, "longitudeE7": 775946000}) == (12.9716, 77.5946)
    assert parse_latlng({"LatLng": "geo:12.9716,77.5946"}) == (12.9716, 77.5946)
    assert parse_latlng("not a coordinate") is None
    assert parse_latlng({"lat": 120.0, "lng": 77.0}) is None


def test_haversine_is_reasonable():
    distance = haversine(12.9716, 77.5946, 12.9352, 77.6245)
    assert 4_000 < distance < 7_000


def test_timestamp_conversion():
    assert to_timestamp("2026-01-01T00:00:00+00:00") == pd.Timestamp("2026-01-01T00:00:00Z")
    assert to_timestamp(1767225600000) == pd.Timestamp("2026-01-01T00:00:00Z")
    assert to_timestamp(None) is None


def test_robust_distance_skips_implausible_jump():
    coords = [(12.9716, 77.5946), (12.9720, 77.5950), (13.5000, 78.5000), (13.5005, 78.5005)]
    stamps = [
        pd.Timestamp("2026-01-01T00:00:00Z"),
        pd.Timestamp("2026-01-01T00:01:00Z"),
        pd.Timestamp("2026-01-01T00:02:00Z"),
        pd.Timestamp("2026-01-01T00:03:00Z"),
    ]
    clean = robust_path_distance(coords[:2], stamps[:2])
    filtered = robust_path_distance(coords, stamps)
    assert filtered < 1_000
    assert clean > 0
