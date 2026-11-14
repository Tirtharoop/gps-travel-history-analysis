from __future__ import annotations

import numpy as np
import pandas as pd

from gps_travel.config import (
    DEFAULT_MAX_GAP_S,
    DEFAULT_MIN_STAY_S,
    DEFAULT_STAY_RADIUS_M,
    MAX_PLAUSIBLE_SPEED_MS,
)
from gps_travel.config import (
    DEFAULT_STAY_RADIUS_M as STAY_RADIUS,
)
from gps_travel.geo import Coordinate, haversine, robust_path_distance
from gps_travel.models import empty_points, empty_segments


def segment_points(
    points: pd.DataFrame,
    stay_radius_m: float = DEFAULT_STAY_RADIUS_M,
    min_stay_s: int = DEFAULT_MIN_STAY_S,
    max_gap_s: int = DEFAULT_MAX_GAP_S,
) -> tuple[pd.DataFrame, dict[int, list[Coordinate]]]:
    if points is None or points.empty:
        return empty_segments(), {}

    frame = points.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["timestamp", "lat", "lon"]).sort_values("timestamp").reset_index(drop=True)
    if len(frame) < 2:
        return empty_segments(), {}

    lats = frame["lat"].to_numpy(dtype=float)
    lons = frame["lon"].to_numpy(dtype=float)
    stamps = frame["timestamp"].tolist()
    modes = frame["mode"].fillna("unknown").astype(str).tolist()
    total = len(frame)

    runs: list[tuple[int, int]] = []
    index = 0
    while index < total:
        anchor_lat, anchor_lon = lats[index], lons[index]
        cursor = index
        while cursor + 1 < total:
            gap = (stamps[cursor + 1] - stamps[cursor]).total_seconds()
            if gap > max_gap_s:
                break
            distance = haversine(anchor_lat, anchor_lon, lats[cursor + 1], lons[cursor + 1])
            if distance > stay_radius_m:
                break
            cursor += 1
        duration = (stamps[cursor] - stamps[index]).total_seconds()
        if cursor > index and duration >= min_stay_s:
            runs.append((index, cursor))
            index = cursor + 1
        else:
            index += 1

    records: list[dict] = []
    paths: dict[int, list[Coordinate]] = {}
    path_key = 0

    def add_move(start_index: int, end_index: int) -> None:
        nonlocal path_key
        if end_index - start_index < 1:
            return
        coords: list[Coordinate] = list(zip(lats[start_index : end_index + 1], lons[start_index : end_index + 1]))
        move_stamps = stamps[start_index : end_index + 1]
        duration_s = max(0.0, (move_stamps[-1] - move_stamps[0]).total_seconds())
        distance_m = robust_path_distance(coords, move_stamps, max_speed_ms=MAX_PLAUSIBLE_SPEED_MS)
        move_modes = [mode for mode in modes[start_index : end_index + 1] if mode != "unknown"]
        mode = pd.Series(move_modes).mode().iloc[0] if move_modes else "unknown"
        records.append(
            {
                "segment_id": path_key,
                "segment_type": "move",
                "start_ts": move_stamps[0],
                "end_ts": move_stamps[-1],
                "mode": mode,
                "distance_m": float(distance_m),
                "duration_s": float(duration_s),
                "start_lat": float(coords[0][0]),
                "start_lon": float(coords[0][1]),
                "end_lat": float(coords[-1][0]),
                "end_lon": float(coords[-1][1]),
                "place_id": "",
                "place_name": "",
                "semantic_type": "",
                "probability": float("nan"),
                "source": "derived",
            }
        )
        paths[path_key] = coords
        path_key += 1

    previous_end = 0
    for run_start, run_end in runs:
        add_move(previous_end, run_start)
        stay_stamps = stamps[run_start : run_end + 1]
        stay_lats = lats[run_start : run_end + 1]
        stay_lons = lons[run_start : run_end + 1]
        duration_s = max(0.0, (stay_stamps[-1] - stay_stamps[0]).total_seconds())
        records.append(
            {
                "segment_id": path_key,
                "segment_type": "stay",
                "start_ts": stay_stamps[0],
                "end_ts": stay_stamps[-1],
                "mode": "unknown",
                "distance_m": 0.0,
                "duration_s": float(duration_s),
                "start_lat": float(np.mean(stay_lats)),
                "start_lon": float(np.mean(stay_lons)),
                "end_lat": float(np.mean(stay_lats)),
                "end_lon": float(np.mean(stay_lons)),
                "place_id": "",
                "place_name": "",
                "semantic_type": "",
                "probability": float("nan"),
                "source": "derived",
            }
        )
        paths[path_key] = list(zip(stay_lats.tolist(), stay_lons.tolist()))
        path_key += 1
        previous_end = run_end

    add_move(previous_end, total - 1)

    if not records:
        return empty_segments(), {}
    return pd.DataFrame.from_records(records), paths


def derive_stays_from_points(
    points: pd.DataFrame,
    stay_radius_m: float = STAY_RADIUS,
    min_stay_s: int = DEFAULT_MIN_STAY_S,
) -> pd.DataFrame:
    segments, _ = segment_points(points, stay_radius_m=stay_radius_m, min_stay_s=min_stay_s)
    if segments.empty:
        return empty_segments()
    return segments[segments["segment_type"] == "stay"].reset_index(drop=True)


def points_from_segments(segments: pd.DataFrame, paths: dict[int, list[Coordinate]]) -> pd.DataFrame:
    rows: list[dict] = []
    for record in segments.itertuples(index=False):
        path = paths.get(int(record.segment_id), [])
        if not path:
            continue
        start = record.start_ts
        end = record.end_ts
        duration = (end - start).total_seconds() if start is not None and end is not None else 0.0
        for position, (lat, lon) in enumerate(path):
            fraction = position / max(1, len(path) - 1)
            stamp = start + pd.Timedelta(seconds=duration * fraction)
            rows.append(
                {
                    "timestamp": stamp,
                    "lat": float(lat),
                    "lon": float(lon),
                    "accuracy": float("nan"),
                    "mode": record.mode if record.segment_type == "move" else "unknown",
                    "source": record.source,
                }
            )
    if not rows:
        return empty_points()
    return pd.DataFrame.from_records(rows)
