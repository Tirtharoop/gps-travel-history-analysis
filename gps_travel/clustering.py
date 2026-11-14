from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score

from gps_travel.config import (
    CLUSTER_PALETTE,
    DEFAULT_EPS_M,
    DEFAULT_KMEANS_CLUSTERS,
    DEFAULT_MIN_SAMPLES,
    EARTH_RADIUS_M,
)
from gps_travel.geo import haversine_arrays
from gps_travel.models import empty_places, empty_segments


def cluster_coordinates(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    algorithm: str = "dbscan",
    eps_m: float = DEFAULT_EPS_M,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    n_clusters: int = DEFAULT_KMEANS_CLUSTERS,
) -> np.ndarray:
    count = len(latitudes)
    if count == 0:
        return np.array([], dtype=int)
    coords = np.radians(np.column_stack([latitudes, longitudes]))
    if algorithm.lower() == "kmeans":
        k = max(1, min(int(n_clusters), count))
        if k == 1:
            return np.zeros(count, dtype=int)
        model = KMeans(n_clusters=k, n_init=10, random_state=42)
        return model.fit_predict(coords)
    model = DBSCAN(eps=float(eps_m) / EARTH_RADIUS_M, min_samples=int(min_samples), metric="haversine")
    return model.fit_predict(coords)


def _dominant_label(frame: pd.DataFrame) -> str:
    values: list[str] = []
    for column in ("place_name", "semantic_type"):
        if column not in frame:
            continue
        for value in frame[column].dropna().astype(str):
            text = value.strip()
            if text and text.lower() not in {"unknown", "none", "nan", ""}:
                values.append(text)
    if not values:
        return ""
    series = pd.Series(values)
    return str(series.mode().iloc[0])


def summarize_clusters(stays: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    if stays is None or stays.empty or len(labels) != len(stays):
        return empty_places()

    frame = stays.copy().reset_index(drop=True)
    frame["cluster"] = labels
    frame["duration_s"] = pd.to_numeric(frame.get("duration_s"), errors="coerce").fillna(0.0)
    frame["start_ts"] = pd.to_datetime(frame["start_ts"], utc=True, errors="coerce")
    frame["end_ts"] = pd.to_datetime(frame["end_ts"], utc=True, errors="coerce")

    records: list[dict] = []
    for cluster_id, group in frame[frame["cluster"] >= 0].groupby("cluster"):
        latitudes = group["start_lat"].to_numpy(dtype=float)
        longitudes = group["start_lon"].to_numpy(dtype=float)
        center_lat = float(np.mean(latitudes))
        center_lon = float(np.mean(longitudes))
        distances = haversine_arrays(latitudes, longitudes, np.full_like(latitudes, center_lat), np.full_like(longitudes, center_lon))
        label = _dominant_label(group)
        records.append(
            {
                "cluster_id": int(cluster_id),
                "lat": center_lat,
                "lon": center_lon,
                "visit_count": len(group),
                "total_stay_s": float(group["duration_s"].sum()),
                "avg_stay_s": float(group["duration_s"].mean()),
                "first_visit": group["start_ts"].min(),
                "last_visit": group["end_ts"].max(),
                "radius_m": float(np.nanmax(distances)) if len(distances) else 0.0,
                "name": label,
                "semantic_type": label,
            }
        )

    if not records:
        return empty_places()

    places = pd.DataFrame.from_records(records)
    places = places.sort_values(["visit_count", "total_stay_s"], ascending=[False, False]).reset_index(drop=True)
    places["place_id"] = [f"P{index + 1:03d}" for index in range(len(places))]
    places["name"] = [
        row["name"] if row["name"] else f"Place {index + 1}" for index, row in places.iterrows()
    ]
    places["color"] = [CLUSTER_PALETTE[index % len(CLUSTER_PALETTE)] for index in range(len(places))]
    places = places[
        [
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
    ]
    return places


def detect_places(
    stays: pd.DataFrame,
    eps_m: float = DEFAULT_EPS_M,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    algorithm: str = "dbscan",
    n_clusters: int = DEFAULT_KMEANS_CLUSTERS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if stays is None or stays.empty:
        return empty_places(), empty_segments()

    frame = stays.copy().reset_index(drop=True)
    labels = cluster_coordinates(
        frame["start_lat"].to_numpy(dtype=float),
        frame["start_lon"].to_numpy(dtype=float),
        algorithm=algorithm,
        eps_m=eps_m,
        min_samples=min_samples,
        n_clusters=n_clusters,
    )
    places = summarize_clusters(frame, labels)
    frame["cluster"] = labels
    if not places.empty:
        lookup = dict(zip(places["cluster_id"].astype(int), places["place_id"]))
        names = dict(zip(places["cluster_id"].astype(int), places["name"]))
        frame["place_id"] = frame["place_id"].fillna("")
        frame["place_id"] = [
            existing if str(existing).strip() else lookup.get(int(cluster), "")
            for existing, cluster in zip(frame["place_id"], frame["cluster"])
        ]
        frame["place_name"] = frame["place_name"].fillna("")
        frame["place_name"] = [
            existing if str(existing).strip() else names.get(int(cluster), "")
            for existing, cluster in zip(frame["place_name"], frame["cluster"])
        ]
    return places, frame


def compare_clustering(
    stays: pd.DataFrame,
    eps_m: float = DEFAULT_EPS_M,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    n_clusters: int = DEFAULT_KMEANS_CLUSTERS,
) -> pd.DataFrame:
    if stays is None or stays.empty or len(stays) < 3:
        return pd.DataFrame(
            columns=["algorithm", "clusters", "noise_points", "noise_ratio", "silhouette", "avg_radius_m"]
        )

    lats = stays["start_lat"].to_numpy(dtype=float)
    lons = stays["start_lon"].to_numpy(dtype=float)
    records: list[dict] = []
    for algorithm in ("dbscan", "kmeans"):
        labels = cluster_coordinates(
            lats,
            lons,
            algorithm=algorithm,
            eps_m=eps_m,
            min_samples=min_samples,
            n_clusters=n_clusters,
        )
        valid = labels[labels >= 0]
        unique = np.unique(valid)
        noise = int(np.sum(labels < 0))
        silhouette = None
        if len(unique) >= 2 and len(valid) >= 3:
            if len(valid) > 2000:
                rng = np.random.default_rng(42)
                sample_idx = rng.choice(len(valid), size=2000, replace=False)
                sample_labels = valid[sample_idx]
                sample_coords = np.radians(np.column_stack([lats[labels >= 0], lons[labels >= 0]]))[sample_idx]
                if len(np.unique(sample_labels)) >= 2:
                    silhouette = float(silhouette_score(sample_coords, sample_labels, metric="haversine"))
            else:
                coords = np.radians(np.column_stack([lats[labels >= 0], lons[labels >= 0]]))
                silhouette = float(silhouette_score(coords, valid, metric="haversine"))
        radii: list[float] = []
        for cluster_id in unique:
            mask = labels == cluster_id
            center_lat = float(np.mean(lats[mask]))
            center_lon = float(np.mean(lons[mask]))
            distances = haversine_arrays(
                lats[mask],
                lons[mask],
                np.full(int(mask.sum()), center_lat),
                np.full(int(mask.sum()), center_lon),
            )
            radii.append(float(np.nanmax(distances)))
        records.append(
            {
                "algorithm": algorithm,
                "clusters": len(unique),
                "noise_points": noise,
                "noise_ratio": float(noise / len(labels)),
                "silhouette": silhouette,
                "avg_radius_m": float(np.mean(radii)) if radii else 0.0,
            }
        )
    return pd.DataFrame.from_records(records)
