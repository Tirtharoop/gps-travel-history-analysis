from __future__ import annotations

from typing import Any

import folium
import numpy as np
import pandas as pd
from folium.plugins import HeatMap, MiniMap

from gps_travel.models import TravelData
from gps_travel.modes import mode_color, mode_label


def _html(value: Any) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _base_map(data: TravelData, places: pd.DataFrame | None) -> folium.Map:
    coordinates: list[tuple[float, float]] = []
    if places is not None and not places.empty:
        coordinates.extend(zip(places["lat"].astype(float), places["lon"].astype(float)))
    if not data.points.empty:
        sample = data.points.iloc[:: max(1, len(data.points) // 2000)]
        coordinates.extend(zip(sample["lat"].astype(float), sample["lon"].astype(float)))
    if not coordinates:
        return folium.Map(location=[20.0, 0.0], zoom_start=2, tiles="OpenStreetMap")
    lats = [item[0] for item in coordinates]
    lons = [item[1] for item in coordinates]
    center = (float(np.mean(lats)), float(np.mean(lons)))
    result = folium.Map(location=center, zoom_start=12, tiles="OpenStreetMap", control_scale=True)
    if len(coordinates) == 1:
        result.location = center
        result.options["zoom"] = 13
    else:
        result.fit_bounds(
            [[min(lats), min(lons)], [max(lats), max(lons)]],
            padding=(30, 30),
        )
    MiniMap(toggle_display=True, position="bottomleft").add_to(result)
    return result


def _mode_series(segments: pd.DataFrame) -> pd.Series:
    if "mode_final" in segments:
        return segments["mode_final"]
    return segments["mode"]


def _segment_popup(segment: Any, mode: str) -> str:
    stamp = pd.Timestamp(segment.start_ts).strftime("%d %b %Y %H:%M")
    distance = float(segment.distance_m) / 1000.0
    duration = float(segment.duration_s) / 60.0
    return (
        f"<b>{_html(mode_label(mode))}</b><br>"
        f"{_html(stamp)}<br>"
        f"{distance:.2f} km &middot; {duration:.0f} min"
    )


def build_route_map(
    data: TravelData,
    places: pd.DataFrame | None = None,
    modes: list[str] | None = None,
    show_places: bool = True,
    show_stays: bool = False,
    show_endpoints: bool = False,
    show_routes: bool = True,
) -> folium.Map:
    result = _base_map(data, places)
    if data.segments.empty:
        return result

    segments = data.segments.copy()
    segments["mode_used"] = _mode_series(segments)
    moves = segments[segments["segment_type"] == "move"].copy()
    if modes is not None:
        moves = moves[moves["mode_used"].isin(modes)]

    route_layer = folium.FeatureGroup(name="Journeys")
    endpoint_layer = folium.FeatureGroup(name="Start and end points")
    for segment in (moves.itertuples(index=False) if show_routes else []):
        path = data.paths.get(int(segment.segment_id), [])
        if not path:
            path = [
                (float(segment.start_lat), float(segment.start_lon)),
                (float(segment.end_lat), float(segment.end_lon)),
            ]
        path = [
            (float(lat), float(lon))
            for lat, lon in path
            if np.isfinite(lat) and np.isfinite(lon)
        ]
        if len(path) < 2:
            continue
        mode = str(segment.mode_used)
        folium.PolyLine(
            locations=path,
            color=mode_color(mode),
            weight=4,
            opacity=0.78,
            tooltip=f"{mode_label(mode)} | {float(segment.distance_m) / 1000.0:.2f} km",
            popup=folium.Popup(_segment_popup(segment, mode), max_width=250),
        ).add_to(route_layer)
        if show_endpoints:
            folium.CircleMarker(
                location=path[0],
                radius=3,
                color="#1b5e20",
                fill=True,
                fill_opacity=0.9,
                tooltip="Journey start",
            ).add_to(endpoint_layer)
            folium.CircleMarker(
                location=path[-1],
                radius=3,
                color="#b71c1c",
                fill=True,
                fill_opacity=0.9,
                tooltip="Journey end",
            ).add_to(endpoint_layer)
    route_layer.add_to(result)
    if show_endpoints:
        endpoint_layer.add_to(result)

    if show_stays and not data.segments.empty:
        stay_layer = folium.FeatureGroup(name="Stays")
        stays = segments[segments["segment_type"] == "stay"]
        for stay in stays.itertuples(index=False):
            if not np.isfinite(stay.start_lat) or not np.isfinite(stay.start_lon):
                continue
            hours = float(stay.duration_s) / 3600.0
            folium.CircleMarker(
                location=(float(stay.start_lat), float(stay.start_lon)),
                radius=4,
                color="#546e7a",
                fill=True,
                fill_opacity=0.35,
                tooltip=f"Stay | {hours:.1f} h",
            ).add_to(stay_layer)
        stay_layer.add_to(result)

    if show_places and places is not None and not places.empty:
        place_layer = folium.FeatureGroup(name="Frequently visited places")
        for index, place in enumerate(places.itertuples(index=False), start=1):
            visits = int(place.visit_count)
            hours = float(place.total_stay_s) / 3600.0
            first = pd.Timestamp(place.first_visit).strftime("%d %b %Y") if pd.notna(place.first_visit) else "n/a"
            last = pd.Timestamp(place.last_visit).strftime("%d %b %Y") if pd.notna(place.last_visit) else "n/a"
            popup = (
                f"<b>{_html(place.name)}</b><br>"
                f"Visits: {visits}<br>"
                f"Total stay: {hours:.1f} h<br>"
                f"First visit: {_html(first)}<br>"
                f"Last visit: {_html(last)}"
            )
            folium.CircleMarker(
                location=(float(place.lat), float(place.lon)),
                radius=9,
                color=str(place.color),
                fill=True,
                fill_color=str(place.color),
                fill_opacity=0.75,
                weight=2,
                tooltip=f"{index}. {_html(place.name)} ({visits})",
                popup=folium.Popup(popup, max_width=280),
            ).add_to(place_layer)
        place_layer.add_to(result)

    folium.LayerControl(collapsed=False).add_to(result)
    return result


def build_heatmap(
    data: TravelData,
    places: pd.DataFrame | None = None,
    weight_by: str = "travel_density",
    modes: list[str] | None = None,
) -> folium.Map:
    result = _base_map(data, places if weight_by != "travel_density" else None)
    weights: list[list[float]] = []

    if weight_by == "travel_density":
        if modes and not data.segments.empty:
            segments = data.segments.copy()
            segments["mode_used"] = _mode_series(segments)
            allowed = segments[(segments["segment_type"] == "move") & segments["mode_used"].isin(modes)]
            for row in allowed.itertuples(index=False):
                for lat, lon in data.paths.get(int(row.segment_id), []):
                    if np.isfinite(lat) and np.isfinite(lon):
                        weights.append([float(lat), float(lon), 1.0])
        else:
            points = data.points
            sample = points.iloc[:: max(1, len(points) // 20000)] if len(points) > 20000 else points
            weights = [
                [float(row.lat), float(row.lon), 1.0]
                for row in sample.itertuples(index=False)
                if np.isfinite(row.lat) and np.isfinite(row.lon)
            ]
    elif weight_by == "visit_frequency":
        if places is not None and not places.empty:
            weights = [
                [float(row.lat), float(row.lon), float(row.visit_count)]
                for row in places.itertuples(index=False)
            ]
        elif not data.segments.empty:
            stays = data.segments[data.segments["segment_type"] == "stay"]
            weights = [
                [float(row.start_lat), float(row.start_lon), 1.0]
                for row in stays.itertuples(index=False)
                if np.isfinite(row.start_lat) and np.isfinite(row.start_lon)
            ]
    else:
        if places is not None and not places.empty:
            weights = [
                [float(row.lat), float(row.lon), float(row.total_stay_s) / 3600.0]
                for row in places.itertuples(index=False)
            ]
        elif not data.segments.empty:
            stays = data.segments[data.segments["segment_type"] == "stay"]
            weights = [
                [float(row.start_lat), float(row.start_lon), max(float(row.duration_s) / 3600.0, 0.01)]
                for row in stays.itertuples(index=False)
                if np.isfinite(row.start_lat) and np.isfinite(row.start_lon)
            ]

    if weights:
        HeatMap(
            weights,
            name="Heat intensity",
            radius=18,
            blur=22,
            min_opacity=0.25,
            max_zoom=17,
            gradient={0.2: "#2c7bb6", 0.4: "#abd9e9", 0.6: "#ffffbf", 0.8: "#fdae61", 1.0: "#d7191c"},
        ).add_to(result)
    folium.LayerControl(collapsed=False).add_to(result)
    return result


def route_distance_summary(data: TravelData) -> dict[str, float]:
    if data.segments.empty:
        return {"distance_km": 0.0, "duration_h": 0.0, "journeys": 0}
    moves = data.segments[data.segments["segment_type"] == "move"]
    return {
        "distance_km": float(moves["distance_m"].sum() / 1000.0),
        "duration_h": float(moves["duration_s"].sum() / 3600.0),
        "journeys": len(moves),
    }
