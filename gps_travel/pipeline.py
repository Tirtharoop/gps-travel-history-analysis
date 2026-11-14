from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from gps_travel.clustering import compare_clustering, detect_places
from gps_travel.config import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_EPS_M,
    DEFAULT_KMEANS_CLUSTERS,
    DEFAULT_MIN_SAMPLES,
)
from gps_travel.models import TravelData
from gps_travel.statistics import (
    daily_stats,
    hourly_stats,
    mode_summary,
    monthly_stats,
    overview_stats,
    repeated_routes,
    weekday_stats,
    weekday_weekend_stats,
    weekly_stats,
)
from gps_travel.transport import (
    TransportModeClassifier,
    enrich_segments,
    train_and_enrich,
)


@dataclass
class AnalysisResult:
    data: TravelData
    places: pd.DataFrame
    cluster_comparison: pd.DataFrame
    mode_summary: pd.DataFrame
    daily: pd.DataFrame
    weekly: pd.DataFrame
    monthly: pd.DataFrame
    hourly: pd.DataFrame
    weekday: pd.DataFrame
    weekday_weekend: pd.DataFrame
    repeated_routes: pd.DataFrame
    overview: dict[str, Any]
    classifier: TransportModeClassifier = field(default_factory=TransportModeClassifier)
    settings: dict[str, Any] = field(default_factory=dict)


def analyze(
    data: TravelData,
    eps_m: float = DEFAULT_EPS_M,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    algorithm: str = "dbscan",
    n_clusters: int = DEFAULT_KMEANS_CLUSTERS,
    predict_modes: bool = True,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> AnalysisResult:
    segments = data.segments.copy()
    if not segments.empty and "path_point_count" not in segments:
        segments["path_point_count"] = [
            len(data.paths.get(int(segment_id), [])) for segment_id in segments["segment_id"]
        ]

    stays = segments[segments["segment_type"] == "stay"] if not segments.empty else segments
    places, stays_with_clusters = detect_places(
        stays,
        eps_m=eps_m,
        min_samples=min_samples,
        algorithm=algorithm,
        n_clusters=n_clusters,
    )

    if not stays_with_clusters.empty:
        cluster_info = stays_with_clusters[["segment_id", "cluster", "place_id", "place_name"]].rename(
            columns={
                "cluster": "place_cluster",
                "place_id": "cluster_place_id",
                "place_name": "cluster_place_name",
            }
        )
        segments = segments.merge(cluster_info, on="segment_id", how="left")
        segments["place_cluster"] = segments["place_cluster"].fillna(-1).astype(int)
        segments["cluster_place_id"] = segments["cluster_place_id"].fillna("")
        segments["cluster_place_name"] = segments["cluster_place_name"].fillna("")
    else:
        segments["place_cluster"] = -1
        segments["cluster_place_id"] = ""
        segments["cluster_place_name"] = ""

    if predict_modes:
        classifier, enriched = train_and_enrich(segments, threshold=confidence_threshold)
    else:
        classifier = TransportModeClassifier(threshold=confidence_threshold)
        enriched = enrich_segments(segments, None, threshold=confidence_threshold)

    comparison = compare_clustering(
        stays,
        eps_m=eps_m,
        min_samples=min_samples,
        n_clusters=n_clusters,
    )

    modes = mode_summary(enriched)
    daily = daily_stats(enriched)
    weekly = weekly_stats(enriched)
    monthly = monthly_stats(enriched)
    hourly = hourly_stats(enriched)
    weekday = weekday_stats(enriched)
    weekend = weekday_weekend_stats(enriched)
    routes = repeated_routes(enriched)
    overview = overview_stats(
        TravelData(
            points=data.points,
            segments=enriched,
            paths=data.paths,
            places=places,
            meta=data.meta,
        ),
        places,
        daily=daily,
        modes=modes,
    )

    analysed_data = TravelData(
        points=data.points,
        segments=enriched,
        paths=data.paths,
        places=places,
        meta=data.meta,
    )

    return AnalysisResult(
        data=analysed_data,
        places=places,
        cluster_comparison=comparison,
        mode_summary=modes,
        daily=daily,
        weekly=weekly,
        monthly=monthly,
        hourly=hourly,
        weekday=weekday,
        weekday_weekend=weekend,
        repeated_routes=routes,
        overview=overview,
        classifier=classifier,
        settings={
            "eps_m": eps_m,
            "min_samples": min_samples,
            "algorithm": algorithm,
            "n_clusters": n_clusters,
            "predict_modes": predict_modes,
            "confidence_threshold": confidence_threshold,
        },
    )
