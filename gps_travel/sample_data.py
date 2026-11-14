from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from typing import Any

import numpy as np

from gps_travel.geo import haversine

PLACES: dict[str, dict[str, Any]] = {
    "home": {"lat": 12.9716, "lon": 77.5946, "type": "HOME", "name": "Home"},
    "work": {"lat": 12.9352, "lon": 77.6245, "type": "WORK", "name": "Work"},
    "gym": {"lat": 12.9610, "lon": 77.6380, "type": "GYM", "name": "Gym"},
    "cafe": {"lat": 12.9780, "lon": 77.6000, "type": "CAFE", "name": "Cafe"},
    "mall": {"lat": 12.9165, "lon": 77.6101, "type": "SHOPPING", "name": "Mall"},
    "park": {"lat": 12.9500, "lon": 77.5850, "type": "PARK", "name": "Park"},
}

MODE_SPEEDS = {
    "walking": 1.35,
    "cycling": 4.2,
    "motorcycle": 10.0,
    "car": 8.5,
    "bus": 5.5,
    "train": 13.0,
    "metro": 11.0,
}

GOOGLE_TYPES = {
    "walking": "WALKING",
    "cycling": "CYCLING",
    "motorcycle": "MOTORCYCLING",
    "car": "IN_PASSENGER_VEHICLE",
    "bus": "IN_BUS",
    "train": "IN_TRAIN",
    "metro": "IN_SUBWAY",
}


def _at(day: datetime, hour: int, minute: int = 0) -> datetime:
    return day.replace(hour=hour, minute=minute, second=0, microsecond=0)


def _offset(coord: tuple[float, float], north_m: float, east_m: float) -> tuple[float, float]:
    lat, lon = coord
    lat_offset = north_m / 111320.0
    lon_offset = east_m / (111320.0 * max(0.2, math.cos(math.radians(lat))))
    return lat + lat_offset, lon + lon_offset


def _interpolate_path(
    start: tuple[float, float],
    end: tuple[float, float],
    rng: np.random.Generator,
    jitter_m: float,
    points: int = 14,
) -> list[tuple[float, float]]:
    lat0, lon0 = start
    lat1, lon1 = end
    dlat = (lat1 - lat0) * 111320.0
    dlon = (lon1 - lon0) * 111320.0 * max(0.2, math.cos(math.radians(lat0)))
    length = max(1.0, math.hypot(dlat, dlon))
    perp_north = -dlon / length
    perp_east = dlat / length
    path = []
    for index in range(points):
        fraction = index / (points - 1)
        lateral = rng.normal(0.0, jitter_m) * math.sin(math.pi * fraction)
        north = dlat * fraction + perp_north * lateral
        east = dlon * fraction + perp_east * lateral
        path.append(_offset(start, north, east))
    path[0] = start
    path[-1] = end
    return path


def _path_distance(path: list[tuple[float, float]]) -> float:
    return sum(haversine(lat1, lon1, lat2, lon2) for (lat1, lon1), (lat2, lon2) in pairwise(path))


def _stay_event(place_key: str, start: datetime, end: datetime, rng: np.random.Generator) -> dict[str, Any]:
    base = PLACES[place_key]
    coordinate = _offset(
        (float(base["lat"]), float(base["lon"])),
        rng.normal(0.0, 12.0),
        rng.normal(0.0, 12.0),
    )
    return {
        "kind": "stay",
        "start": start,
        "end": end,
        "place": place_key,
        "coord": coordinate,
        "mode": None,
        "distance_m": 0.0,
        "path": [coordinate],
    }


def _move_event(
    origin_key: str,
    destination_key: str,
    start: datetime,
    mode: str,
    rng: np.random.Generator,
    labelled: bool = True,
) -> dict[str, Any]:
    origin = (float(PLACES[origin_key]["lat"]), float(PLACES[origin_key]["lon"]))
    destination = (float(PLACES[destination_key]["lat"]), float(PLACES[destination_key]["lon"]))
    jitter = {"walking": 12.0, "cycling": 18.0, "car": 30.0, "bus": 30.0, "train": 20.0, "metro": 20.0, "motorcycle": 25.0}
    path = _interpolate_path(origin, destination, rng, jitter.get(mode, 25.0))
    distance = _path_distance(path)
    speed = MODE_SPEEDS.get(mode, 8.0)
    duration = max(60.0, distance / speed)
    return {
        "kind": "move",
        "start": start,
        "end": start + timedelta(seconds=duration),
        "origin": origin_key,
        "destination": destination_key,
        "coord": destination,
        "mode": mode,
        "distance_m": distance,
        "path": path,
        "labelled": labelled,
    }


def _commute_mode(rng: np.random.Generator) -> str:
    options = ["car", "bus", "metro", "train", "cycling", "walking", "motorcycle"]
    weights = np.array([0.28, 0.18, 0.16, 0.10, 0.14, 0.08, 0.06])
    return str(rng.choice(options, p=weights / weights.sum()))


def _maybe_label(rng: np.random.Generator) -> bool:
    return bool(rng.random() > 0.08)


def _build_events(days: int, seed: int, start: datetime) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    events: list[dict[str, Any]] = []
    day_zero = start.replace(hour=0, minute=0, second=0, microsecond=0)

    for day_index in range(days):
        day = day_zero + timedelta(days=day_index)
        next_day = day + timedelta(days=1)
        weekday = day.weekday()
        cursor = day

        if weekday < 5:
            morning_end = _at(day, 8, 0)
            events.append(_stay_event("home", cursor, morning_end, rng))
            mode = _commute_mode(rng)
            move = _move_event("home", "work", morning_end, mode, rng, labelled=_maybe_label(rng))
            events.append(move)
            cursor = move["end"]
            work_end = _at(day, 17, 30)
            if rng.random() < 0.45 and cursor < _at(day, 12, 20):
                lunch_start = _at(day, 12, 30)
                events.append(_stay_event("work", cursor, lunch_start, rng))
                walk_out = _move_event("work", "cafe", lunch_start, "walking", rng, labelled=_maybe_label(rng))
                events.append(walk_out)
                lunch_end = max(walk_out["end"] + timedelta(minutes=35), _at(day, 13, 5))
                events.append(_stay_event("cafe", walk_out["end"], lunch_end, rng))
                walk_back = _move_event("cafe", "work", lunch_end, "walking", rng, labelled=_maybe_label(rng))
                events.append(walk_back)
                cursor = walk_back["end"]
            if cursor < work_end:
                events.append(_stay_event("work", cursor, work_end, rng))
            cursor = work_end

            roll = rng.random()
            if roll < 0.30 and cursor < _at(day, 20, 0):
                destination = "gym" if rng.random() < 0.65 else "cafe"
                out_mode = rng.choice(["car", "bus", "cycling", "walking", "motorcycle"])
                outbound = _move_event("work", destination, cursor, str(out_mode), rng, labelled=_maybe_label(rng))
                events.append(outbound)
                stay_minutes = int(rng.integers(45, 95))
                departure = outbound["end"] + timedelta(minutes=stay_minutes)
                events.append(_stay_event(destination, outbound["end"], departure, rng))
                back_mode = rng.choice(["car", "bus", "cycling", "walking", "motorcycle"])
                inbound = _move_event(destination, "home", departure, str(back_mode), rng, labelled=_maybe_label(rng))
                events.append(inbound)
                cursor = inbound["end"]
            else:
                home_mode = "walking" if rng.random() < 0.15 else _commute_mode(rng)
                inbound = _move_event("work", "home", cursor, home_mode, rng, labelled=_maybe_label(rng))
                events.append(inbound)
                cursor = inbound["end"]
        else:
            morning_end = _at(day, 9, 30)
            events.append(_stay_event("home", cursor, morning_end, rng))
            cursor = morning_end
            if rng.random() < 0.85 and cursor < _at(day, 19, 0):
                destination = str(rng.choice(["park", "mall", "cafe", "mall", "park"]))
                out_mode = str(rng.choice(["car", "bus", "cycling", "walking"]))
                outbound = _move_event("home", destination, cursor, out_mode, rng, labelled=_maybe_label(rng))
                events.append(outbound)
                stay_minutes = int(rng.integers(60, 150))
                departure = outbound["end"] + timedelta(minutes=stay_minutes)
                events.append(_stay_event(destination, outbound["end"], departure, rng))
                if rng.random() < 0.35 and destination in {"mall", "park"}:
                    second = "cafe" if destination == "mall" else "mall"
                    second_mode = str(rng.choice(["walking", "car", "bus", "cycling"]))
                    second_move = _move_event(destination, second, departure, second_mode, rng, labelled=_maybe_label(rng))
                    events.append(second_move)
                    second_stay = int(rng.integers(30, 80))
                    second_departure = second_move["end"] + timedelta(minutes=second_stay)
                    events.append(_stay_event(second, second_move["end"], second_departure, rng))
                    back = _move_event(second, "home", second_departure, str(rng.choice(["car", "bus", "walking"])), rng, labelled=_maybe_label(rng))
                    events.append(back)
                    cursor = back["end"]
                else:
                    back = _move_event(destination, "home", departure, str(rng.choice(["car", "bus", "walking", "cycling"])), rng, labelled=_maybe_label(rng))
                    events.append(back)
                    cursor = back["end"]

        if cursor < next_day:
            events.append(_stay_event("home", cursor, next_day, rng))
    return events


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="seconds")


def _semantic_payload(events: list[dict[str, Any]], rng: np.random.Generator) -> dict[str, Any]:
    segments: list[dict[str, Any]] = []
    for event in events:
        start = _iso(event["start"])
        end = _iso(event["end"])
        if event["kind"] == "stay":
            place = PLACES[event["place"]]
            segments.append(
                {
                    "startTime": start,
                    "endTime": end,
                    "startTimeTimezoneUtcOffsetMinutes": 0,
                    "visit": {
                        "hierarchyLevel": 0,
                        "probability": round(float(rng.uniform(0.82, 0.99)), 2),
                        "topCandidate": {
                            "placeId": f"place_{event['place']}",
                            "semanticType": place["type"],
                            "probability": round(float(rng.uniform(0.80, 0.99)), 2),
                            "placeLocation": f"geo:{event['coord'][0]:.6f},{event['coord'][1]:.6f}",
                        },
                    },
                }
            )
        else:
            path = []
            duration = (event["end"] - event["start"]).total_seconds() / 60.0
            for index, (lat, lon) in enumerate(event["path"]):
                offset = duration * index / max(1, len(event["path"]) - 1)
                path.append(
                    {
                        "point": f"geo:{lat:.6f},{lon:.6f}",
                        "durationMinutesOffsetFromStartTime": f"{offset:.2f}",
                    }
                )
            record: dict[str, Any] = {
                "startTime": start,
                "endTime": end,
                "startTimeTimezoneUtcOffsetMinutes": 0,
                "timelinePath": path,
                "activity": {
                    "start": {"latLng": f"geo:{event['path'][0][0]:.6f},{event['path'][0][1]:.6f}"},
                    "end": {"latLng": f"geo:{event['path'][-1][0]:.6f},{event['path'][-1][1]:.6f}"},
                    "distanceMeters": round(float(event["distance_m"]), 1),
                },
            }
            if event.get("labelled", True):
                record["activity"]["topCandidate"] = {
                    "type": GOOGLE_TYPES.get(str(event["mode"]), "UNKNOWN"),
                    "probability": round(float(rng.uniform(0.72, 0.98)), 2),
                }
            segments.append(record)
    return {
        "semanticSegments": segments,
        "metadata": {"generator": "gps_travel.sample_data", "version": 1},
    }


def _classic_payload(events: list[dict[str, Any]], rng: np.random.Generator) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for event in events:
        if event["kind"] == "stay":
            start = event["start"]
            end = event["end"]
            step = timedelta(minutes=20)
            moments = [start]
            cursor = start + step
            while cursor < end:
                moments.append(cursor)
                cursor += step
            moments.append(end)
            for moment in moments:
                lat, lon = _offset(event["coord"], rng.normal(0.0, 6.0), rng.normal(0.0, 6.0))
                rows.append(
                    {
                        "latitudeE7": round(lat * 1e7),
                        "longitudeE7": round(lon * 1e7),
                        "timestamp": moment.astimezone(UTC),
                        "accuracy": int(rng.integers(6, 30)),
                        "activity": [{"type": "STILL", "confidence": 100}],
                    }
                )
        else:
            total_seconds = max(1.0, (event["end"] - event["start"]).total_seconds())
            count = len(event["path"])
            for index, (lat, lon) in enumerate(event["path"]):
                moment = event["start"] + timedelta(seconds=total_seconds * index / max(1, count - 1))
                row: dict[str, Any] = {
                    "latitudeE7": round(lat * 1e7),
                    "longitudeE7": round(lon * 1e7),
                    "timestamp": moment.astimezone(UTC),
                    "accuracy": int(rng.integers(5, 20)),
                }
                if event.get("labelled", True):
                    row["activity"] = [
                        {"type": GOOGLE_TYPES.get(str(event["mode"]), "UNKNOWN"), "confidence": int(rng.integers(72, 99))}
                    ]
                rows.append(row)
    rows.sort(key=lambda item: item["timestamp"])
    for row in rows:
        row["timestamp"] = row["timestamp"].astimezone(UTC).isoformat(timespec="seconds")
    return {"locations": rows, "metadata": {"generator": "gps_travel.sample_data", "version": 1}}


def generate_sample_timeline(
    days: int = 60,
    seed: int = 7,
    start: datetime | None = None,
    fmt: str = "semantic",
) -> dict[str, Any]:
    origin = start or datetime(2026, 8, 1, 0, 0, tzinfo=UTC)
    rng = np.random.default_rng(seed)
    events = _build_events(days, seed, origin)
    if fmt == "classic":
        return _classic_payload(events, rng)
    return _semantic_payload(events, rng)


def generate_classic_timeline(days: int = 30, seed: int = 11, start: datetime | None = None) -> dict[str, Any]:
    return generate_sample_timeline(days=days, seed=seed, start=start, fmt="classic")
