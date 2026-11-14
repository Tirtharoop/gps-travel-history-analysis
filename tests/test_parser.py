import pandas as pd

from gps_travel.parser import parse_timeline


def test_semantic_parser_builds_move_and_stay():
    payload = {
        "semanticSegments": [
            {
                "startTime": "2026-01-01T08:00:00+00:00",
                "endTime": "2026-01-01T08:30:00+00:00",
                "timelinePath": [
                    {"point": "geo:12.9000,77.5000", "durationMinutesOffsetFromStartTime": "0"},
                    {"point": "geo:12.9050,77.5050", "durationMinutesOffsetFromStartTime": "15"},
                    {"point": "geo:12.9100,77.5100", "durationMinutesOffsetFromStartTime": "30"},
                ],
                "activity": {
                    "start": {"latLng": "geo:12.9000,77.5000"},
                    "end": {"latLng": "geo:12.9100,77.5100"},
                    "distanceMeters": 1500,
                    "topCandidate": {"type": "IN_PASSENGER_VEHICLE", "probability": 0.9},
                },
            },
            {
                "startTime": "2026-01-01T08:30:00+00:00",
                "endTime": "2026-01-01T09:30:00+00:00",
                "visit": {
                    "probability": 0.9,
                    "topCandidate": {
                        "placeId": "work-place",
                        "semanticType": "WORK",
                        "probability": 0.95,
                        "placeLocation": "geo:12.9100,77.5100",
                    },
                },
            },
        ]
    }
    data = parse_timeline(payload)
    assert data.meta["source_format"] == "semantic_segments"
    assert len(data.segments) == 2
    move = data.moves.iloc[0]
    stay = data.stays.iloc[0]
    assert move["mode"] == "car"
    assert move["distance_m"] == 1500
    assert move["duration_s"] == 1800
    assert stay["place_name"] == "Work"
    assert stay["place_id"] == "work-place"
    assert stay["duration_s"] == 3600
    assert not data.points.empty
    assert set(data.paths) == set(data.segments["segment_id"].tolist())


def test_semantic_unlabelled_activity_is_unknown():
    payload = {
        "semanticSegments": [
            {
                "startTime": "2026-01-01T08:00:00+00:00",
                "endTime": "2026-01-01T08:20:00+00:00",
                "activity": {
                    "start": {"latLng": "geo:12.9000,77.5000"},
                    "end": {"latLng": "geo:12.9200,77.5200"},
                    "distanceMeters": 2500,
                },
            }
        ]
    }
    data = parse_timeline(payload)
    assert data.moves.iloc[0]["mode"] == "unknown"


def test_semantic_separate_timeline_path_attaches_to_activity():
    payload = {
        "semanticSegments": [
            {
                "startTime": "2026-01-01T08:00:00+00:00",
                "endTime": "2026-01-01T08:30:00+00:00",
                "activity": {
                    "start": {"latLng": "geo:12.9000,77.5000"},
                    "end": {"latLng": "geo:12.9100,77.5100"},
                    "topCandidate": {"type": "IN_PASSENGER_VEHICLE", "probability": 0.9},
                },
            },
            {
                "startTime": "2026-01-01T08:00:00+00:00",
                "endTime": "2026-01-01T08:30:00+00:00",
                "timelinePath": [
                    {"point": "geo:12.9000,77.5000", "durationMinutesOffsetFromStartTime": "0"},
                    {"point": "geo:12.9050,77.5050", "durationMinutesOffsetFromStartTime": "15"},
                    {"point": "geo:12.9100,77.5100", "durationMinutesOffsetFromStartTime": "30"},
                ],
            },
        ]
    }
    data = parse_timeline(payload)
    assert len(data.segments) == 1
    move = data.moves.iloc[0]
    assert move["mode"] == "car"
    assert len(data.paths[int(move["segment_id"])]) == 3


def test_semantic_path_only_becomes_unknown_move():
    payload = {
        "semanticSegments": [
            {
                "startTime": "2026-01-01T08:00:00+00:00",
                "endTime": "2026-01-01T08:20:00+00:00",
                "timelinePath": [
                    {"point": "geo:12.9000,77.5000", "durationMinutesOffsetFromStartTime": "0"},
                    {"point": "geo:12.9100,77.5100", "durationMinutesOffsetFromStartTime": "20"},
                ],
            }
        ]
    }
    data = parse_timeline(payload)
    assert len(data.moves) == 1
    assert data.moves.iloc[0]["mode"] == "unknown"
    assert data.moves.iloc[0]["duration_s"] == 1200


def test_timeline_objects_parser():
    payload = {
        "timelineObjects": [
            {
                "placeVisit": {
                    "location": {
                        "latitudeE7": 129000000,
                        "longitudeE7": 775000000,
                        "name": "Home",
                        "placeId": "home",
                        "semanticType": "HOME",
                    },
                    "duration": {
                        "startTimestamp": "2026-01-01T00:00:00+00:00",
                        "endTimestamp": "2026-01-01T08:00:00+00:00",
                    },
                    "visitConfidence": 90,
                }
            },
            {
                "activitySegment": {
                    "startLocation": {"latitudeE7": 129000000, "longitudeE7": 775000000},
                    "endLocation": {"latitudeE7": 129100000, "longitudeE7": 775100000},
                    "duration": {
                        "startTimestamp": "2026-01-01T08:00:00+00:00",
                        "endTimestamp": "2026-01-01T08:20:00+00:00",
                    },
                    "distance": 3000,
                    "activityType": "IN_BUS",
                    "confidence": 85,
                    "waypointPath": {
                        "waypoints": [
                            {"latE7": 129000000, "lngE7": 775000000},
                            {"latE7": 129050000, "lngE7": 775050000},
                            {"latE7": 129100000, "lngE7": 775100000},
                        ]
                    },
                }
            },
            {
                "placeVisit": {
                    "location": {
                        "latitudeE7": 129100000,
                        "longitudeE7": 775100000,
                        "name": "Office",
                        "semanticType": "WORK",
                    },
                    "duration": {
                        "startTimestamp": "2026-01-01T08:20:00+00:00",
                        "endTimestamp": "2026-01-01T17:00:00+00:00",
                    },
                }
            },
        ]
    }
    data = parse_timeline(payload)
    assert len(data.stays) == 2
    assert len(data.moves) == 1
    move = data.moves.iloc[0]
    assert move["mode"] == "bus"
    assert move["distance_m"] == 3000
    assert len(data.paths[int(move["segment_id"])]) == 3
    assert set(data.stays["place_name"]) == {"Home", "Office"}


def test_classic_locations_derives_stays_and_moves():
    locations = []
    for minute, lat, lon in [
        (0, 12.9716, 77.5946),
        (5, 12.9717, 77.5947),
        (10, 12.9716, 77.5945),
        (15, 12.9800, 77.6100),
        (20, 12.9900, 77.6200),
        (25, 13.0000, 77.6300),
        (30, 12.9716, 77.5946),
        (35, 12.9717, 77.5946),
        (40, 12.9716, 77.5945),
    ]:
        record = {
            "latitudeE7": int(lat * 1e7),
            "longitudeE7": int(lon * 1e7),
            "timestamp": (pd.Timestamp("2026-01-01T08:00:00Z") + pd.Timedelta(minutes=minute)).isoformat(),
            "accuracy": 10,
            "activity": [{"type": "STILL" if minute <= 10 or minute >= 30 else "IN_PASSENGER_VEHICLE", "confidence": 90}],
        }
        locations.append(record)
    data = parse_timeline({"locations": locations})
    assert data.meta["source_format"] == "classic_locations"
    assert len(data.stays) >= 2
    assert len(data.moves) >= 1
    assert data.moves["distance_m"].sum() > 0
    assert "car" in data.points["mode"].unique()


def test_raw_signals_parser():
    payload = {
        "rawSignals": [
            {
                "position": {
                    "LatLng": "geo:12.9716,77.5946",
                    "accuracyMeters": 12,
                    "timestamp": "2026-01-01T08:00:00+00:00",
                }
            },
            {
                "position": {
                    "LatLng": "geo:12.9720,77.5950",
                    "accuracyMeters": 12,
                    "timestamp": "2026-01-01T08:05:00+00:00",
                }
            },
        ]
    }
    data = parse_timeline(payload)
    assert data.meta["source_format"] == "raw_signals"
    assert len(data.points) == 2
