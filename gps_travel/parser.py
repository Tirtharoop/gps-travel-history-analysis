from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gps_travel.geo import (
    haversine,
    parse_latlng,
    robust_path_distance,
)
from gps_travel.models import (
    TravelData,
    empty_points,
    empty_segments,
)
from gps_travel.modes import normalize_mode
from gps_travel.segmentation import points_from_segments, segment_points


def parse_timeline(source: Any) -> TravelData:
    payload = _load_payload(source)
    if isinstance(payload, list):
        payload = {"locations": payload}
    if not isinstance(payload, dict):
        raise TypeError("Unsupported Timeline payload. Expected a JSON object or a list of location records.")

    if "semanticSegments" in payload:
        return _parse_semantic(payload)
    if "timelineObjects" in payload:
        return _parse_timeline_objects(payload)
    if "locations" in payload:
        return _parse_classic(payload)
    if "rawSignals" in payload:
        return _parse_raw_signals(payload)
    raise ValueError(
        "Unrecognised Timeline schema. Supported keys: semanticSegments, timelineObjects, locations, rawSignals."
    )


def _load_payload(source: Any) -> Any:
    if isinstance(source, (dict, list)):
        return source
    if isinstance(source, (bytes, bytearray)):
        return json.loads(bytes(source).decode("utf-8-sig"))
    if hasattr(source, "read"):
        content = source.read()
        if isinstance(content, bytes):
            content = content.decode("utf-8-sig")
        return json.loads(content)
    if isinstance(source, Path):
        return json.loads(source.read_text(encoding="utf-8-sig"))
    if isinstance(source, str):
        text = source.strip()
        if text.startswith(("{", "[")):
            return json.loads(text)
        return json.loads(Path(source).read_text(encoding="utf-8-sig"))
    raise ValueError("Unable to load Timeline JSON from the supplied source.")


def to_timestamp(value: Any) -> pd.Timestamp | None:
    if value is None:
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        if not np.isfinite(number) or number <= 0:
            return None
        unit = "ms" if abs(number) > 1e11 else "s"
        try:
            return pd.Timestamp(number, unit=unit, tz="UTC")
        except (ValueError, OverflowError):
            return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.isdigit():
            return to_timestamp(int(text))
        try:
            stamp = pd.to_datetime(text, utc=True, errors="coerce")
        except (ValueError, TypeError):
            return None
        if stamp is None or pd.isna(stamp):
            return None
        return stamp
    try:
        stamp = pd.Timestamp(value)
    except (ValueError, TypeError):
        return None
    if pd.isna(stamp):
        return None
    if stamp.tzinfo is None:
        return stamp.tz_localize("UTC")
    return stamp.tz_convert("UTC")


def _probability(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if not np.isfinite(number) or number < 0:
        return float("nan")
    if number > 1.0:
        number = number / 100.0
    return float(min(number, 1.0))


def _activity_mode(activities: Any) -> tuple[str, float]:
    if not isinstance(activities, list) or not activities:
        return "unknown", float("nan")
    best_mode = "unknown"
    best_confidence = -1.0
    for item in activities:
        if isinstance(item, str):
            mode = normalize_mode(item)
            confidence = 0.0
        elif isinstance(item, dict):
            mode = normalize_mode(item.get("type") or item.get("activityType") or item.get("mode"))
            raw = item.get("confidence", 0.0)
            try:
                confidence = float(raw)
            except (TypeError, ValueError):
                confidence = 0.0
        else:
            continue
        if confidence > best_confidence:
            best_mode = mode
            best_confidence = confidence
    if best_confidence < 0:
        best_confidence = float("nan")
    return best_mode, _probability(best_confidence) if best_confidence > 1 else best_confidence


def _points_frame(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return empty_points()
    frame = pd.DataFrame.from_records(rows)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame["lat"] = pd.to_numeric(frame["lat"], errors="coerce")
    frame["lon"] = pd.to_numeric(frame["lon"], errors="coerce")
    frame = frame.dropna(subset=["timestamp", "lat", "lon"])
    frame = frame[
        (frame["lat"].between(-90, 90))
        & (frame["lon"].between(-180, 180))
        & ~((frame["lat"].abs() < 1e-9) & (frame["lon"].abs() < 1e-9))
    ]
    if frame.empty:
        return empty_points()
    if "mode" not in frame:
        frame["mode"] = "unknown"
    if "source" not in frame:
        frame["source"] = "timeline"
    frame["mode"] = frame["mode"].fillna("unknown").map(normalize_mode)
    frame["source"] = frame["source"].fillna("timeline")
    if "accuracy" not in frame:
        frame["accuracy"] = np.nan
    frame = frame.sort_values("timestamp").drop_duplicates(subset=["timestamp", "lat", "lon"]).reset_index(drop=True)
    return frame[["timestamp", "lat", "lon", "accuracy", "mode", "source"]]


def _parse_classic(payload: dict) -> TravelData:
    records = payload.get("locations") or []
    rows: list[dict] = []
    warnings: list[str] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        coordinate = parse_latlng(record)
        if coordinate is None:
            continue
        stamp = to_timestamp(
            record.get("timestamp")
            or record.get("timestampMs")
            or record.get("time")
            or record.get("timestampISO")
        )
        if stamp is None:
            continue
        mode, _ = _activity_mode(record.get("activity"))
        if mode == "unknown":
            mode = normalize_mode(record.get("activityType") or record.get("mode"))
        accuracy = record.get("accuracy")
        rows.append(
            {
                "timestamp": stamp,
                "lat": coordinate[0],
                "lon": coordinate[1],
                "accuracy": accuracy,
                "mode": mode,
                "source": str(record.get("source") or "classic"),
            }
        )
    points = _points_frame(rows)
    if points.empty:
        warnings.append("No valid timestamped location records were found in the locations array.")
        return TravelData(meta={"source_format": "classic_locations", "warnings": warnings, "raw_counts": {"locations": len(records)}})
    segments, paths = segment_points(points)
    if segments.empty:
        warnings.append("Location records did not contain enough consecutive points to form journeys or stays.")
    return _finalize(
        records=[],
        points=points,
        segments=segments,
        paths=paths,
        source_format="classic_locations",
        warnings=warnings,
        raw_counts={"locations": len(records), "valid_points": len(points)},
    )


def _parse_raw_signals(payload: dict) -> TravelData:
    signals = payload.get("rawSignals") or []
    rows: list[dict] = []
    for item in signals:
        if not isinstance(item, dict):
            continue
        position = item.get("position") or item.get("positionRecord") or item.get("location")
        if not isinstance(position, dict):
            continue
        coordinate = parse_latlng(position)
        if coordinate is None:
            continue
        stamp = to_timestamp(
            position.get("timestamp") or position.get("timestampMs") or item.get("timestamp")
        )
        if stamp is None:
            continue
        accuracy = position.get("accuracyMeters") or position.get("accuracy")
        rows.append(
            {
                "timestamp": stamp,
                "lat": coordinate[0],
                "lon": coordinate[1],
                "accuracy": accuracy,
                "mode": "unknown",
                "source": "raw_signals",
            }
        )
    points = _points_frame(rows)
    warnings: list[str] = []
    if points.empty:
        warnings.append("rawSignals contained no usable position records.")
        return TravelData(meta={"source_format": "raw_signals", "warnings": warnings, "raw_counts": {"rawSignals": len(signals)}})
    segments, paths = segment_points(points)
    return _finalize(
        records=[],
        points=points,
        segments=segments,
        paths=paths,
        source_format="raw_signals",
        warnings=warnings,
        raw_counts={"rawSignals": len(signals), "valid_points": len(points)},
    )


def _path_from_timeline_path(entries: Any, start: pd.Timestamp | None) -> tuple[list[tuple[float, float]], list[pd.Timestamp]]:
    coords: list[tuple[float, float]] = []
    stamps: list[pd.Timestamp] = []
    if not isinstance(entries, list):
        return coords, stamps
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        coordinate = parse_latlng(entry.get("point") or entry.get("latLng") or entry.get("location"))
        if coordinate is None:
            continue
        coords.append(coordinate)
        offset = entry.get("durationMinutesOffsetFromStartTime")
        try:
            minutes = float(offset) if offset is not None else None
        except (TypeError, ValueError):
            minutes = None
        if start is not None and minutes is not None:
            stamps.append(start + pd.Timedelta(minutes=minutes))
        elif start is not None:
            stamps.append(start)
        else:
            stamps.append(None)
    return coords, stamps


def _parse_semantic(payload: dict) -> TravelData:
    semantic_segments = payload.get("semanticSegments") or []
    records: list[dict] = []
    warnings: list[str] = []
    raw_counts = {"semanticSegments": len(semantic_segments)}
    path_only: list[dict] = []

    for segment in semantic_segments:
        if not isinstance(segment, dict):
            continue
        start = to_timestamp(segment.get("startTime"))
        end = to_timestamp(segment.get("endTime"))
        path_coords, path_stamps = _path_from_timeline_path(segment.get("timelinePath"), start)
        visit = segment.get("visit")
        activity = segment.get("activity")

        if isinstance(visit, dict):
            candidate = visit.get("topCandidate") or {}
            coordinate = (
                parse_latlng(candidate.get("placeLocation"))
                or parse_latlng(visit.get("placeLocation"))
                or (path_coords[0] if path_coords else None)
            )
            if coordinate is None or start is None:
                continue
            semantic_type = str(candidate.get("semanticType") or "").strip()
            name = str(candidate.get("name") or "").strip()
            if not name and semantic_type and semantic_type.upper() != "UNKNOWN":
                name = semantic_type.replace("_", " ").title()
            records.append(
                {
                    "segment_type": "stay",
                    "start_ts": start,
                    "end_ts": end or start,
                    "mode": "unknown",
                    "distance_m": 0.0,
                    "duration_s": (end - start).total_seconds() if end is not None else 0.0,
                    "start_lat": coordinate[0],
                    "start_lon": coordinate[1],
                    "end_lat": coordinate[0],
                    "end_lon": coordinate[1],
                    "place_id": str(candidate.get("placeId") or ""),
                    "place_name": name,
                    "semantic_type": semantic_type,
                    "probability": _probability(candidate.get("probability") or visit.get("probability")),
                    "source": "semantic_visit",
                    "path": path_coords,
                    "_stamps": path_stamps,
                }
            )
            continue

        if isinstance(activity, dict):
            start_coord = parse_latlng(activity.get("start")) or (path_coords[0] if path_coords else None)
            end_coord = parse_latlng(activity.get("end")) or (path_coords[-1] if path_coords else None)
            if start_coord is None and path_coords:
                start_coord = path_coords[0]
            if end_coord is None and path_coords:
                end_coord = path_coords[-1]
            if start is None and end is None:
                continue
            if start is None:
                start = end
            if end is None:
                end = start
            candidate = activity.get("topCandidate") or {}
            mode = normalize_mode(candidate.get("type") or activity.get("type") or activity.get("activityType"))
            distance = activity.get("distanceMeters")
            try:
                distance = float(distance)
            except (TypeError, ValueError):
                distance = None
            if distance is None or not np.isfinite(distance) or distance <= 0:
                if path_coords:
                    distance = robust_path_distance(path_coords, path_stamps if all(path_stamps) else None)
                elif start_coord and end_coord:
                    distance = haversine(start_coord[0], start_coord[1], end_coord[0], end_coord[1])
                else:
                    distance = 0.0
            records.append(
                {
                    "segment_type": "move",
                    "start_ts": start,
                    "end_ts": end,
                    "mode": mode,
                    "distance_m": float(distance),
                    "duration_s": max(0.0, (end - start).total_seconds()),
                    "start_lat": start_coord[0] if start_coord else np.nan,
                    "start_lon": start_coord[1] if start_coord else np.nan,
                    "end_lat": end_coord[0] if end_coord else np.nan,
                    "end_lon": end_coord[1] if end_coord else np.nan,
                    "place_id": "",
                    "place_name": "",
                    "semantic_type": "",
                    "probability": _probability(candidate.get("probability") or activity.get("probability")),
                    "source": "semantic_activity",
                    "path": path_coords,
                    "_stamps": path_stamps,
                }
            )
            continue

        if path_coords:
            path_only.append(
                {
                    "start_ts": start,
                    "end_ts": end,
                    "coords": path_coords,
                    "stamps": path_stamps,
                }
            )

    records = _attach_path_only(records, path_only)
    if not records:
        warnings.append(
            "semanticSegments did not include visit or activity records. Attempting to derive journeys from any raw signals."
        )
        if payload.get("rawSignals"):
            return _parse_raw_signals(payload)
        return TravelData(
            meta={"source_format": "semantic_segments", "warnings": warnings, "raw_counts": raw_counts}
        )

    return _finalize(
        records=records,
        source_format="semantic_segments",
        warnings=warnings,
        raw_counts=raw_counts,
    )


def _overlap_seconds(start_a: pd.Timestamp, end_a: pd.Timestamp, start_b: pd.Timestamp, end_b: pd.Timestamp) -> float:
    if start_a is None or end_a is None or start_b is None or end_b is None:
        return 0.0
    latest = max(start_a, start_b)
    earliest = min(end_a, end_b)
    return max(0.0, (earliest - latest).total_seconds())


def _attach_path_only(records: list[dict], path_only: list[dict]) -> list[dict]:
    if not path_only:
        return records
    if not records:
        for item in path_only:
            coords = item["coords"]
            start = item["start_ts"]
            end = item["end_ts"]
            if start is None and end is None:
                continue
            if start is None:
                start = end
            if end is None:
                end = start
            stamps = item.get("stamps") or []
            distance = robust_path_distance(coords, stamps if stamps and all(stamps) else None)
            records.append(
                {
                    "segment_type": "move",
                    "start_ts": start,
                    "end_ts": end,
                    "mode": "unknown",
                    "distance_m": float(distance),
                    "duration_s": max(0.0, (end - start).total_seconds()),
                    "start_lat": coords[0][0],
                    "start_lon": coords[0][1],
                    "end_lat": coords[-1][0],
                    "end_lon": coords[-1][1],
                    "place_id": "",
                    "place_name": "",
                    "semantic_type": "",
                    "probability": float("nan"),
                    "source": "semantic_timeline_path",
                    "path": coords,
                }
            )
        return records

    for item in path_only:
        best_index = None
        best_overlap = 0.0
        for index, record in enumerate(records):
            overlap = _overlap_seconds(item["start_ts"], item["end_ts"], record["start_ts"], record["end_ts"])
            if overlap > best_overlap:
                best_overlap = overlap
                best_index = index
        if best_index is not None and best_overlap > 0:
            record = records[best_index]
            if not record.get("path"):
                record["path"] = item["coords"]
                if record["segment_type"] == "move":
                    if not np.isfinite(record.get("start_lat", np.nan)):
                        record["start_lat"], record["start_lon"] = item["coords"][0]
                    if not np.isfinite(record.get("end_lat", np.nan)):
                        record["end_lat"], record["end_lon"] = item["coords"][-1]
        else:
            coords = item["coords"]
            start = item["start_ts"]
            end = item["end_ts"]
            if start is None and end is None:
                continue
            if start is None:
                start = end
            if end is None:
                end = start
            stamps = item.get("stamps") or []
            distance = robust_path_distance(coords, stamps if stamps and all(stamps) else None)
            records.append(
                {
                    "segment_type": "move",
                    "start_ts": start,
                    "end_ts": end,
                    "mode": "unknown",
                    "distance_m": float(distance),
                    "duration_s": max(0.0, (end - start).total_seconds()),
                    "start_lat": coords[0][0],
                    "start_lon": coords[0][1],
                    "end_lat": coords[-1][0],
                    "end_lon": coords[-1][1],
                    "place_id": "",
                    "place_name": "",
                    "semantic_type": "",
                    "probability": float("nan"),
                    "source": "semantic_timeline_path",
                    "path": coords,
                }
            )
    return records


def _parse_timeline_objects(payload: dict) -> TravelData:
    objects = payload.get("timelineObjects") or []
    records: list[dict] = []
    warnings: list[str] = []
    for item in objects:
        if not isinstance(item, dict):
            continue
        place = item.get("placeVisit")
        if isinstance(place, dict):
            location = place.get("location") or {}
            coordinate = parse_latlng(location)
            duration = place.get("duration") or {}
            start = to_timestamp(duration.get("startTimestamp") or duration.get("startTimestampMs"))
            end = to_timestamp(duration.get("endTimestamp") or duration.get("endTimestampMs"))
            if coordinate is None or start is None:
                continue
            semantic_type = str(location.get("semanticType") or "").strip()
            name = str(location.get("name") or "").strip()
            if not name and semantic_type and semantic_type.upper() != "UNKNOWN":
                name = semantic_type.replace("_", " ").title()
            records.append(
                {
                    "segment_type": "stay",
                    "start_ts": start,
                    "end_ts": end or start,
                    "mode": "unknown",
                    "distance_m": 0.0,
                    "duration_s": (end - start).total_seconds() if end is not None else 0.0,
                    "start_lat": coordinate[0],
                    "start_lon": coordinate[1],
                    "end_lat": coordinate[0],
                    "end_lon": coordinate[1],
                    "place_id": str(location.get("placeId") or ""),
                    "place_name": name,
                    "semantic_type": semantic_type,
                    "probability": _probability(place.get("visitConfidence") or place.get("placeConfidence")),
                    "source": "timeline_place",
                    "path": [],
                }
            )
            continue

        activity = item.get("activitySegment")
        if isinstance(activity, dict):
            start_coord = parse_latlng(activity.get("startLocation"))
            end_coord = parse_latlng(activity.get("endLocation"))
            duration = activity.get("duration") or {}
            start = to_timestamp(duration.get("startTimestamp") or duration.get("startTimestampMs"))
            end = to_timestamp(duration.get("endTimestamp") or duration.get("endTimestampMs"))
            if start is None and end is None:
                continue
            if start is None:
                start = end
            if end is None:
                end = start
            path_coords = _waypoint_coordinates(activity)
            if start_coord is None and path_coords:
                start_coord = path_coords[0]
            if end_coord is None and path_coords:
                end_coord = path_coords[-1]
            distance = activity.get("distance")
            try:
                distance = float(distance)
            except (TypeError, ValueError):
                distance = None
            if distance is None or distance <= 0:
                if path_coords:
                    distance = robust_path_distance(path_coords)
                elif start_coord and end_coord:
                    distance = haversine(start_coord[0], start_coord[1], end_coord[0], end_coord[1])
                else:
                    distance = 0.0
            mode = normalize_mode(
                activity.get("activityType")
                or activity.get("topCandidate", {}).get("type") if isinstance(activity.get("topCandidate"), dict) else activity.get("activityType")
            )
            if mode == "unknown" and isinstance(activity.get("activities"), list) and activity["activities"]:
                mode, _ = _activity_mode(activity["activities"])
            records.append(
                {
                    "segment_type": "move",
                    "start_ts": start,
                    "end_ts": end,
                    "mode": mode,
                    "distance_m": float(distance),
                    "duration_s": max(0.0, (end - start).total_seconds()),
                    "start_lat": start_coord[0] if start_coord else (path_coords[0][0] if path_coords else np.nan),
                    "start_lon": start_coord[1] if start_coord else (path_coords[0][1] if path_coords else np.nan),
                    "end_lat": end_coord[0] if end_coord else (path_coords[-1][0] if path_coords else np.nan),
                    "end_lon": end_coord[1] if end_coord else (path_coords[-1][1] if path_coords else np.nan),
                    "place_id": "",
                    "place_name": "",
                    "semantic_type": "",
                    "probability": _probability(activity.get("confidence")),
                    "source": "timeline_activity",
                    "path": path_coords,
                }
            )

    if not records:
        warnings.append("timelineObjects did not contain any usable place visits or activity segments.")
        return TravelData(
            meta={
                "source_format": "timeline_objects",
                "warnings": warnings,
                "raw_counts": {"timelineObjects": len(objects)},
            }
        )
    return _finalize(
        records=records,
        source_format="timeline_objects",
        warnings=warnings,
        raw_counts={"timelineObjects": len(objects)},
    )


def _waypoint_coordinates(activity: dict) -> list[tuple[float, float]]:
    coords: list[tuple[float, float]] = []
    waypoint_path = activity.get("waypointPath")
    if isinstance(waypoint_path, dict):
        for waypoint in waypoint_path.get("waypoints") or []:
            coordinate = parse_latlng(waypoint)
            if coordinate is not None:
                coords.append(coordinate)
    simplified = activity.get("simplifiedRawPath")
    if isinstance(simplified, dict):
        points = simplified.get("points")
        if isinstance(points, str):
            for pair in points.replace(";", " ").split():
                coordinate = parse_latlng(pair)
                if coordinate is not None:
                    coords.append(coordinate)
    return coords


def _finalize(
    records: list[dict],
    source_format: str,
    warnings: list[str],
    raw_counts: dict,
    points: pd.DataFrame | None = None,
    segments: pd.DataFrame | None = None,
    paths: dict[int, list[tuple[float, float]]] | None = None,
) -> TravelData:
    if segments is None or segments.empty:
        if not records:
            return TravelData(
                points=points if points is not None else empty_points(),
                segments=empty_segments(),
                meta={"source_format": source_format, "warnings": warnings, "raw_counts": raw_counts},
            )
        records = sorted(records, key=lambda item: item.get("start_ts") or pd.Timestamp.min.tz_localize("UTC"))
        path_map: dict[int, list[tuple[float, float]]] = {}
        clean_records: list[dict] = []
        for index, record in enumerate(records):
            path = record.pop("path", []) or []
            record.pop("_stamps", None)
            record["segment_id"] = index
            if record.get("start_ts") is None:
                continue
            if record.get("end_ts") is None:
                record["end_ts"] = record["start_ts"]
            if record.get("duration_s") is None or not np.isfinite(record.get("duration_s", np.nan)):
                record["duration_s"] = max(0.0, (record["end_ts"] - record["start_ts"]).total_seconds())
            if record.get("distance_m") is None or not np.isfinite(record.get("distance_m", np.nan)):
                record["distance_m"] = robust_path_distance(path) if path else 0.0
            if record.get("probability") is None:
                record["probability"] = float("nan")
            record.setdefault("place_id", "")
            record.setdefault("place_name", "")
            record.setdefault("semantic_type", "")
            record.setdefault("source", source_format)
            if record["segment_type"] == "move":
                if not np.isfinite(record.get("start_lat", np.nan)) and path:
                    record["start_lat"], record["start_lon"] = path[0]
                if not np.isfinite(record.get("end_lat", np.nan)) and path:
                    record["end_lat"], record["end_lon"] = path[-1]
            path_map[record["segment_id"]] = path
            clean_records.append(record)
        if not clean_records:
            return TravelData(
                points=points if points is not None else empty_points(),
                meta={"source_format": source_format, "warnings": warnings, "raw_counts": raw_counts},
            )
        segments = pd.DataFrame.from_records(clean_records)
        paths = path_map
    else:
        paths = paths or {}

    for column, default in (
        ("place_id", ""),
        ("place_name", ""),
        ("semantic_type", ""),
        ("probability", float("nan")),
        ("source", source_format),
    ):
        if column not in segments:
            segments[column] = default
    if "mode" not in segments:
        segments["mode"] = "unknown"
    segments["mode"] = segments["mode"].map(normalize_mode)
    segments["segment_type"] = segments["segment_type"].fillna("move")
    segments["start_ts"] = pd.to_datetime(segments["start_ts"], utc=True, errors="coerce")
    segments["end_ts"] = pd.to_datetime(segments["end_ts"], utc=True, errors="coerce")
    segments = segments.dropna(subset=["start_ts"]).copy()
    segments["end_ts"] = segments["end_ts"].fillna(segments["start_ts"])
    segments["distance_m"] = pd.to_numeric(segments["distance_m"], errors="coerce").fillna(0.0)
    segments["duration_s"] = pd.to_numeric(segments["duration_s"], errors="coerce")
    computed = (segments["end_ts"] - segments["start_ts"]).dt.total_seconds()
    segments["duration_s"] = segments["duration_s"].fillna(computed).clip(lower=0.0)
    segments["start_lat"] = pd.to_numeric(segments["start_lat"], errors="coerce")
    segments["start_lon"] = pd.to_numeric(segments["start_lon"], errors="coerce")
    segments["end_lat"] = pd.to_numeric(segments["end_lat"], errors="coerce")
    segments["end_lon"] = pd.to_numeric(segments["end_lon"], errors="coerce")
    if "segment_id" not in segments:
        segments["segment_id"] = range(len(segments))
    segments["segment_id"] = segments["segment_id"].astype(int)
    segments = segments.sort_values(["start_ts", "segment_id"]).reset_index(drop=True)

    if points is None or points.empty:
        points = points_from_segments(segments, paths)
        extra_rows: list[dict] = []
        for record in segments[segments["segment_type"] == "stay"].itertuples(index=False):
            if np.isfinite(record.start_lat) and np.isfinite(record.start_lon):
                extra_rows.append(
                    {
                        "timestamp": record.start_ts,
                        "lat": float(record.start_lat),
                        "lon": float(record.start_lon),
                        "accuracy": np.nan,
                        "mode": "unknown",
                        "source": "place",
                    }
                )
        if extra_rows:
            extra = pd.DataFrame.from_records(extra_rows)
            points = pd.concat([points, extra], ignore_index=True)
            points = _points_frame(points.to_dict("records"))

    segments["path_point_count"] = [
        len(paths.get(int(segment_id), [])) for segment_id in segments["segment_id"]
    ]
    return TravelData(
        points=points,
        segments=segments,
        paths=paths,
        meta={
            "source_format": source_format,
            "warnings": warnings,
            "raw_counts": raw_counts,
            "moves": int((segments["segment_type"] == "move").sum()),
            "stays": int((segments["segment_type"] == "stay").sum()),
            "point_count": len(points),
        },
    )
