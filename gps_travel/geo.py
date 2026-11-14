from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from itertools import pairwise
from typing import Any

import numpy as np
import pandas as pd

from gps_travel.config import EARTH_RADIUS_M

Coordinate = tuple[float, float]


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(a)))


def haversine_arrays(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_M * np.arcsin(np.minimum(1.0, np.sqrt(a)))


def path_distance(coords: Sequence[Coordinate]) -> float:
    if len(coords) < 2:
        return 0.0
    total = 0.0
    for (lat1, lon1), (lat2, lon2) in pairwise(coords):
        total += haversine(lat1, lon1, lat2, lon2)
    return total


def robust_path_distance(
    coords: Sequence[Coordinate],
    timestamps: Sequence[pd.Timestamp] | None = None,
    max_speed_ms: float = 350.0,
) -> float:
    if len(coords) < 2:
        return 0.0
    total = 0.0
    for index in range(1, len(coords)):
        lat1, lon1 = coords[index - 1]
        lat2, lon2 = coords[index]
        step = haversine(lat1, lon1, lat2, lon2)
        if timestamps is not None:
            try:
                seconds = (timestamps[index] - timestamps[index - 1]).total_seconds()
            except (TypeError, AttributeError):
                seconds = None
            if seconds is None or seconds <= 0:
                continue
            if step / seconds > max_speed_ms:
                continue
        total += step
    return total


def valid_coordinate(lat: float, lon: float) -> bool:
    return (
        isinstance(lat, (int, float))
        and isinstance(lon, (int, float))
        and math.isfinite(lat)
        and math.isfinite(lon)
        and -90.0 <= lat <= 90.0
        and -180.0 <= lon <= 180.0
        and not (abs(lat) < 1e-9 and abs(lon) < 1e-9)
    )


def parse_latlng(value: Any) -> Coordinate | None:
    if value is None:
        return None
    if isinstance(value, dict):
        for key in ("latLng", "LatLng", "latlng"):
            if key in value:
                return parse_latlng(value[key])
        if "location" in value and isinstance(value["location"], dict):
            parsed = parse_latlng(value["location"])
            if parsed is not None:
                return parsed
        if "latitudeE7" in value and "longitudeE7" in value:
            try:
                return _validated(float(value["latitudeE7"]) / 1e7, float(value["longitudeE7"]) / 1e7)
            except (TypeError, ValueError):
                return None
        if "latE7" in value and "lngE7" in value:
            try:
                return _validated(float(value["latE7"]) / 1e7, float(value["lngE7"]) / 1e7)
            except (TypeError, ValueError):
                return None
        for lat_key, lon_key in (("latitude", "longitude"), ("lat", "lon"), ("lat", "lng")):
            if lat_key in value and lon_key in value:
                try:
                    return _validated(float(value[lat_key]), float(value[lon_key]))
                except (TypeError, ValueError):
                    return None
        return None
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return _validated(float(value[0]), float(value[1]))
        except (TypeError, ValueError):
            return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.lower().startswith("geo:"):
            text = text[4:]
        text = text.replace("\u00b0", "").replace("\u2032", "").replace("\u2033", "")
        parts = [part for part in text.replace(";", ",").split(",") if part.strip()]
        if len(parts) < 2:
            return None
        try:
            return _validated(float(parts[0].strip()), float(parts[1].strip()))
        except (TypeError, ValueError):
            return None
    return None


def _validated(lat: float, lon: float) -> Coordinate | None:
    if valid_coordinate(lat, lon):
        return lat, lon
    return None


def dataframe_path_distance(
    lats: Iterable[float], lons: Iterable[float], timestamps: Iterable[pd.Timestamp] | None = None
) -> float:
    lat_values = np.asarray(list(lats), dtype=float)
    lon_values = np.asarray(list(lons), dtype=float)
    if len(lat_values) < 2:
        return 0.0
    steps = haversine_arrays(lat_values[:-1], lon_values[:-1], lat_values[1:], lon_values[1:])
    if timestamps is None:
        return float(np.nansum(steps))
    ts = list(timestamps)
    keep = np.ones(len(steps), dtype=bool)
    for index in range(len(steps)):
        try:
            seconds = (ts[index + 1] - ts[index]).total_seconds()
        except (TypeError, AttributeError, IndexError):
            keep[index] = False
            continue
        if seconds <= 0 or steps[index] / seconds > 350.0:
            keep[index] = False
    return float(np.nansum(steps[keep]))


def coordinate_center(coords: Sequence[Coordinate]) -> Coordinate:
    if not coords:
        return 0.0, 0.0
    lat = sum(item[0] for item in coords) / len(coords)
    lon = sum(item[1] for item in coords) / len(coords)
    return lat, lon


def format_coordinate(lat: float, lon: float) -> str:
    return f"{lat:.5f}, {lon:.5f}"
