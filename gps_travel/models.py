from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from gps_travel.geo import Coordinate

POINT_COLUMNS = ["timestamp", "lat", "lon", "accuracy", "mode", "source"]
SEGMENT_COLUMNS = [
    "segment_id",
    "segment_type",
    "start_ts",
    "end_ts",
    "mode",
    "distance_m",
    "duration_s",
    "start_lat",
    "start_lon",
    "end_lat",
    "end_lon",
    "place_id",
    "place_name",
    "semantic_type",
    "probability",
    "source",
]
PLACE_COLUMNS = [
    "place_id",
    "cluster_id",
    "lat",
    "lon",
    "visit_count",
    "total_stay_s",
    "avg_stay_s",
    "first_visit",
    "last_visit",
    "radius_m",
    "name",
    "semantic_type",
    "color",
]


def _datetime_series() -> pd.Series:
    return pd.Series(dtype="datetime64[ns, UTC]")


def _float_series() -> pd.Series:
    return pd.Series(dtype="float64")


def _text_series() -> pd.Series:
    return pd.Series(dtype="object")


def empty_points() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": _datetime_series(),
            "lat": _float_series(),
            "lon": _float_series(),
            "accuracy": _float_series(),
            "mode": _text_series(),
            "source": _text_series(),
        }
    )


def empty_segments() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "segment_id": pd.Series(dtype="int64"),
            "segment_type": _text_series(),
            "start_ts": _datetime_series(),
            "end_ts": _datetime_series(),
            "mode": _text_series(),
            "distance_m": _float_series(),
            "duration_s": _float_series(),
            "start_lat": _float_series(),
            "start_lon": _float_series(),
            "end_lat": _float_series(),
            "end_lon": _float_series(),
            "place_id": _text_series(),
            "place_name": _text_series(),
            "semantic_type": _text_series(),
            "probability": _float_series(),
            "source": _text_series(),
        }
    )


def empty_places() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "place_id": _text_series(),
            "cluster_id": pd.Series(dtype="int64"),
            "lat": _float_series(),
            "lon": _float_series(),
            "visit_count": pd.Series(dtype="int64"),
            "total_stay_s": _float_series(),
            "avg_stay_s": _float_series(),
            "first_visit": _datetime_series(),
            "last_visit": _datetime_series(),
            "radius_m": _float_series(),
            "name": _text_series(),
            "semantic_type": _text_series(),
            "color": _text_series(),
        }
    )


@dataclass
class TravelData:
    points: pd.DataFrame = field(default_factory=empty_points)
    segments: pd.DataFrame = field(default_factory=empty_segments)
    paths: dict[int, list[Coordinate]] = field(default_factory=dict)
    places: pd.DataFrame = field(default_factory=empty_places)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def stays(self) -> pd.DataFrame:
        if self.segments.empty:
            return empty_segments()
        return self.segments[self.segments["segment_type"] == "stay"]

    @property
    def moves(self) -> pd.DataFrame:
        if self.segments.empty:
            return empty_segments()
        return self.segments[self.segments["segment_type"] == "move"]

    @property
    def start_time(self) -> pd.Timestamp | None:
        stamps = self._segment_stamps()
        if not stamps:
            return None
        return min(stamps)

    @property
    def end_time(self) -> pd.Timestamp | None:
        stamps = self._segment_stamps()
        if not stamps:
            return None
        return max(stamps)

    def _segment_stamps(self) -> list[pd.Timestamp]:
        if self.segments.empty:
            return []
        values: list[pd.Timestamp] = []
        for column in ("start_ts", "end_ts"):
            values.extend(pd.to_datetime(self.segments[column], utc=True, errors="coerce").dropna().tolist())
        return values

    def filtered(
        self,
        start: Any | None = None,
        end: Any | None = None,
        modes: list[str] | None = None,
    ) -> TravelData:
        return filter_travel_data(self, start=start, end=end, modes=modes)


def _bounds(start: Any | None, end: Any | None) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    lower = None
    upper = None
    if start is not None:
        lower = pd.Timestamp(start)
        if lower.tzinfo is None:
            lower = lower.tz_localize("UTC")
        else:
            lower = lower.tz_convert("UTC")
        lower = lower.normalize()
    if end is not None:
        upper = pd.Timestamp(end)
        if upper.tzinfo is None:
            upper = upper.tz_localize("UTC")
        else:
            upper = upper.tz_convert("UTC")
        upper = upper.normalize() + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    return lower, upper


def filter_travel_data(
    data: TravelData,
    start: Any | None = None,
    end: Any | None = None,
    modes: list[str] | None = None,
) -> TravelData:
    lower, upper = _bounds(start, end)
    segments = data.segments.copy()
    if segments.empty:
        return data

    mask = pd.Series(True, index=segments.index)
    start_ts = pd.to_datetime(segments["start_ts"], utc=True, errors="coerce")
    end_ts = pd.to_datetime(segments["end_ts"], utc=True, errors="coerce")
    if lower is not None:
        mask &= end_ts >= lower
    if upper is not None:
        mask &= start_ts <= upper
    if modes:
        selected = {mode.lower() for mode in modes}
        movement = segments["segment_type"] == "move"
        mask &= (~movement) | segments["mode"].isin(selected)
    filtered_segments = segments[mask].copy()

    if filtered_segments.empty:
        return TravelData(
            points=empty_points(),
            segments=empty_segments(),
            paths={},
            places=data.places,
            meta=data.meta,
        )

    points = data.points.copy()
    if not points.empty:
        point_mask = pd.Series(True, index=points.index)
        point_ts = pd.to_datetime(points["timestamp"], utc=True, errors="coerce")
        if lower is not None:
            point_mask &= point_ts >= lower
        if upper is not None:
            point_mask &= point_ts <= upper
        points = points[point_mask].copy()

    segment_ids = set(filtered_segments["segment_id"].astype(int).tolist())
    paths = {key: value for key, value in data.paths.items() if key in segment_ids}

    return TravelData(
        points=points,
        segments=filtered_segments,
        paths=paths,
        places=data.places,
        meta=data.meta,
    )


def date_limits(data: TravelData) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    if data.segments.empty:
        if not data.points.empty:
            stamps = pd.to_datetime(data.points["timestamp"], utc=True, errors="coerce").dropna()
            if not stamps.empty:
                return stamps.min(), stamps.max()
        return None, None
    start_ts = pd.to_datetime(data.segments["start_ts"], utc=True, errors="coerce").dropna()
    end_ts = pd.to_datetime(data.segments["end_ts"], utc=True, errors="coerce").dropna()
    if start_ts.empty or end_ts.empty:
        return None, None
    return start_ts.min(), end_ts.max()
