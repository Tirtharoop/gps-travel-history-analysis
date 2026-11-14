from __future__ import annotations

import numpy as np
import pandas as pd

from gps_travel.models import TravelData, empty_segments
from gps_travel.modes import MODE_COLORS, MODE_ORDER


def _moves(segments: pd.DataFrame, effective: bool = True) -> pd.DataFrame:
    if segments is None or segments.empty:
        return empty_segments()
    moves = segments[segments["segment_type"] == "move"].copy()
    if moves.empty:
        return empty_segments()
    if effective and "mode_final" in moves:
        moves["mode_used"] = moves["mode_final"]
    else:
        moves["mode_used"] = moves["mode"]
    moves["start_ts"] = pd.to_datetime(moves["start_ts"], utc=True, errors="coerce")
    moves["end_ts"] = pd.to_datetime(moves["end_ts"], utc=True, errors="coerce")
    moves["distance_m"] = pd.to_numeric(moves["distance_m"], errors="coerce").fillna(0.0)
    moves["duration_s"] = pd.to_numeric(moves["duration_s"], errors="coerce").fillna(0.0)
    return moves.dropna(subset=["start_ts"])


def _naive_time(series: pd.Series) -> pd.Series:
    stamps = pd.to_datetime(series, utc=True, errors="coerce")
    return stamps.dt.tz_convert("UTC").dt.tz_localize(None)


def mode_summary(segments: pd.DataFrame, effective: bool = True) -> pd.DataFrame:
    moves = _moves(segments, effective=effective)
    if moves.empty:
        return pd.DataFrame(
            columns=[
                "mode",
                "label",
                "color",
                "journeys",
                "distance_km",
                "duration_h",
                "share_distance",
                "share_duration",
                "share_journeys",
            ]
        )
    grouped = (
        moves.groupby("mode_used")
        .agg(journeys=("segment_id", "count"), distance_m=("distance_m", "sum"), duration_s=("duration_s", "sum"))
        .reset_index()
        .rename(columns={"mode_used": "mode"})
    )
    grouped["distance_km"] = grouped["distance_m"] / 1000.0
    grouped["duration_h"] = grouped["duration_s"] / 3600.0
    total_distance = grouped["distance_km"].sum()
    total_duration = grouped["duration_h"].sum()
    total_journeys = grouped["journeys"].sum()
    grouped["share_distance"] = grouped["distance_km"] / total_distance if total_distance else 0.0
    grouped["share_duration"] = grouped["duration_h"] / total_duration if total_duration else 0.0
    grouped["share_journeys"] = grouped["journeys"] / total_journeys if total_journeys else 0.0
    grouped["label"] = grouped["mode"].str.title()
    grouped["color"] = grouped["mode"].map(MODE_COLORS).fillna(MODE_COLORS["unknown"])
    order = {mode: index for index, mode in enumerate(MODE_ORDER)}
    grouped["_order"] = grouped["mode"].map(order).fillna(len(MODE_ORDER))
    grouped = grouped.sort_values(["distance_km", "_order"], ascending=[False, True]).drop(columns="_order")
    return grouped[
        [
            "mode",
            "label",
            "color",
            "journeys",
            "distance_km",
            "duration_h",
            "share_distance",
            "share_duration",
            "share_journeys",
        ]
    ].reset_index(drop=True)


def daily_stats(segments: pd.DataFrame, fill_missing: bool = True) -> pd.DataFrame:
    moves = _moves(segments)
    if moves.empty:
        return pd.DataFrame(columns=["date", "distance_km", "duration_min", "journeys"])
    moves = moves.copy()
    moves["date"] = _naive_time(moves["start_ts"]).dt.normalize()
    grouped = (
        moves.groupby("date")
        .agg(
            distance_km=("distance_m", lambda values: float(values.sum()) / 1000.0),
            duration_min=("duration_s", lambda values: float(values.sum()) / 60.0),
            journeys=("segment_id", "count"),
        )
        .reset_index()
    )
    if fill_missing and len(grouped) > 1:
        full_range = pd.date_range(grouped["date"].min(), grouped["date"].max(), freq="D")
        grouped = grouped.set_index("date").reindex(full_range, fill_value=0).rename_axis("date").reset_index()
    grouped["date"] = pd.to_datetime(grouped["date"])
    return grouped


def weekly_stats(segments: pd.DataFrame) -> pd.DataFrame:
    moves = _moves(segments)
    if moves.empty:
        return pd.DataFrame(columns=["week_start", "week", "distance_km", "duration_min", "journeys"])
    moves = moves.copy()
    naive = _naive_time(moves["start_ts"])
    period = naive.dt.to_period("W-SUN")
    moves["week_start"] = period.dt.start_time
    moves["week"] = period.astype(str)
    grouped = (
        moves.groupby(["week_start", "week"])
        .agg(
            distance_km=("distance_m", lambda values: float(values.sum()) / 1000.0),
            duration_min=("duration_s", lambda values: float(values.sum()) / 60.0),
            journeys=("segment_id", "count"),
        )
        .reset_index()
        .sort_values("week_start")
    )
    return grouped


def monthly_stats(segments: pd.DataFrame) -> pd.DataFrame:
    moves = _moves(segments)
    if moves.empty:
        return pd.DataFrame(columns=["month", "month_start", "distance_km", "duration_min", "journeys"])
    moves = moves.copy()
    naive = _naive_time(moves["start_ts"])
    period = naive.dt.to_period("M")
    moves["month"] = period.astype(str)
    moves["month_start"] = period.dt.start_time
    grouped = (
        moves.groupby(["month", "month_start"])
        .agg(
            distance_km=("distance_m", lambda values: float(values.sum()) / 1000.0),
            duration_min=("duration_s", lambda values: float(values.sum()) / 60.0),
            journeys=("segment_id", "count"),
        )
        .reset_index()
        .sort_values("month_start")
    )
    return grouped


def hourly_stats(segments: pd.DataFrame) -> pd.DataFrame:
    moves = _moves(segments)
    if moves.empty:
        return pd.DataFrame(columns=["hour", "journeys", "distance_km"])
    moves = moves.copy()
    moves["hour"] = _naive_time(moves["start_ts"]).dt.hour
    grouped = moves.groupby("hour").agg(
        journeys=("segment_id", "count"),
        distance_km=("distance_m", lambda values: float(values.sum()) / 1000.0),
    )
    grouped = grouped.reindex(range(24), fill_value=0).reset_index().rename(columns={"index": "hour"})
    grouped["hour"] = grouped["hour"].astype(int)
    return grouped


def weekday_stats(segments: pd.DataFrame) -> pd.DataFrame:
    names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    moves = _moves(segments)
    if moves.empty:
        return pd.DataFrame(columns=["weekday", "weekday_name", "distance_km", "journeys"])
    moves = moves.copy()
    moves["weekday"] = _naive_time(moves["start_ts"]).dt.weekday
    grouped = moves.groupby("weekday").agg(
        distance_km=("distance_m", lambda values: float(values.sum()) / 1000.0),
        journeys=("segment_id", "count"),
    )
    grouped = grouped.reindex(range(7), fill_value=0).reset_index()
    grouped["weekday_name"] = grouped["weekday"].map(dict(enumerate(names)))
    return grouped


def weekday_weekend_stats(segments: pd.DataFrame) -> pd.DataFrame:
    moves = _moves(segments)
    if moves.empty:
        return pd.DataFrame(columns=["period", "distance_km", "duration_min", "journeys"])
    moves = moves.copy()
    moves["is_weekend"] = _naive_time(moves["start_ts"]).dt.weekday >= 5
    moves["period"] = np.where(moves["is_weekend"], "Weekend", "Weekday")
    grouped = moves.groupby("period").agg(
        distance_km=("distance_m", lambda values: float(values.sum()) / 1000.0),
        duration_min=("duration_s", lambda values: float(values.sum()) / 60.0),
        journeys=("segment_id", "count"),
    )
    return grouped.reset_index()


def repeated_routes(segments: pd.DataFrame, top_n: int = 15, decimals: int = 3) -> pd.DataFrame:
    moves = _moves(segments)
    columns = [
        "route",
        "mode",
        "trips",
        "distance_km",
        "total_distance_km",
        "average_duration_min",
        "first_trip",
        "last_trip",
    ]
    if moves.empty:
        return pd.DataFrame(columns=columns)
    frame = moves.dropna(subset=["start_lat", "start_lon", "end_lat", "end_lon"]).copy()
    frame = frame[frame["distance_m"] > 0]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    frame["origin"] = [
        f"{lat:.{decimals}f}, {lon:.{decimals}f}" for lat, lon in zip(frame["start_lat"], frame["start_lon"])
    ]
    frame["destination"] = [
        f"{lat:.{decimals}f}, {lon:.{decimals}f}" for lat, lon in zip(frame["end_lat"], frame["end_lon"])
    ]
    frame["route"] = frame["origin"] + " -> " + frame["destination"]
    grouped = (
        frame.groupby(["route", "mode_used"])
        .agg(
            trips=("segment_id", "count"),
            distance_km=("distance_m", lambda values: float(values.mean()) / 1000.0),
            total_distance_km=("distance_m", lambda values: float(values.sum()) / 1000.0),
            average_duration_min=("duration_s", lambda values: float(values.mean()) / 60.0),
            first_trip=("start_ts", "min"),
            last_trip=("start_ts", "max"),
        )
        .reset_index()
        .rename(columns={"mode_used": "mode"})
    )
    grouped = grouped.sort_values(["trips", "total_distance_km"], ascending=[False, False])
    return grouped.head(top_n)[columns].reset_index(drop=True)


def overview_stats(
    data: TravelData,
    places: pd.DataFrame,
    daily: pd.DataFrame | None = None,
    modes: pd.DataFrame | None = None,
) -> dict:
    moves = _moves(data.segments)
    stays = data.stays
    total_distance_km = float(moves["distance_m"].sum() / 1000.0) if not moves.empty else 0.0
    total_duration_h = float(moves["duration_s"].sum() / 3600.0) if not moves.empty else 0.0
    journeys = len(moves)
    stay_hours = float(stays["duration_s"].sum() / 3600.0) if not stays.empty else 0.0
    if daily is None:
        daily = daily_stats(data.segments)
    active_days = int((daily["journeys"] > 0).sum()) if not daily.empty else 0
    average_daily = float(daily.loc[daily["journeys"] > 0, "distance_km"].mean()) if active_days else 0.0
    max_daily = float(daily["distance_km"].max()) if not daily.empty else 0.0
    max_daily_date = None
    if not daily.empty and max_daily > 0:
        max_daily_date = str(daily.loc[daily["distance_km"].idxmax(), "date"].date())
    if modes is None:
        modes = mode_summary(data.segments)
    top_mode = ""
    top_mode_share = 0.0
    if not modes.empty:
        top_mode = str(modes.iloc[0]["label"])
        top_mode_share = float(modes.iloc[0]["share_distance"])
    start = data.start_time
    end = data.end_time
    return {
        "total_distance_km": total_distance_km,
        "total_duration_h": total_duration_h,
        "total_stay_hours": stay_hours,
        "journeys": journeys,
        "stay_segments": len(stays),
        "visited_places": len(places),
        "total_points": len(data.points),
        "active_days": active_days,
        "analysis_start": start,
        "analysis_end": end,
        "average_daily_distance_km": average_daily,
        "max_daily_distance_km": max_daily,
        "max_daily_distance_date": max_daily_date,
        "top_mode": top_mode,
        "top_mode_share": top_mode_share,
        "modes_used": len(modes),
    }
