from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split

from gps_travel.config import DEFAULT_CONFIDENCE_THRESHOLD
from gps_travel.geo import haversine_arrays
from gps_travel.modes import normalize_mode

FEATURE_COLUMNS = [
    "distance_km",
    "duration_min",
    "avg_speed_kmh",
    "straight_km",
    "straightness",
    "point_count",
    "hour_sin",
    "hour_cos",
    "weekday",
    "is_weekend",
    "log_distance",
]


def _move_rows(segments: pd.DataFrame) -> pd.DataFrame:
    if segments is None or segments.empty:
        return pd.DataFrame()
    moves = segments[segments["segment_type"] == "move"].copy()
    if moves.empty:
        return pd.DataFrame()
    moves["start_ts"] = pd.to_datetime(moves["start_ts"], utc=True, errors="coerce")
    moves["distance_m"] = pd.to_numeric(moves["distance_m"], errors="coerce").fillna(0.0)
    moves["duration_s"] = pd.to_numeric(moves["duration_s"], errors="coerce").fillna(0.0)
    return moves.dropna(subset=["start_ts"])


def extract_features(segments: pd.DataFrame) -> pd.DataFrame:
    moves = _move_rows(segments)
    if moves.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS)
    stamps = moves["start_ts"]
    hours = stamps.dt.hour.astype(float)
    weekday = stamps.dt.weekday.astype(float)
    distance_km = moves["distance_m"] / 1000.0
    duration_min = moves["duration_s"] / 60.0
    safe_duration = moves["duration_s"].where(moves["duration_s"] > 0, np.nan)
    avg_speed = (moves["distance_m"] / safe_duration) * 3.6
    straight = haversine_arrays(
        moves["start_lat"].to_numpy(dtype=float),
        moves["start_lon"].to_numpy(dtype=float),
        moves["end_lat"].to_numpy(dtype=float),
        moves["end_lon"].to_numpy(dtype=float),
    ) / 1000.0
    distance_for_ratio = distance_km.where(distance_km > 0.001, np.nan)
    straightness = (straight / distance_for_ratio).clip(upper=5.0).fillna(1.0)
    features = pd.DataFrame(
        {
            "distance_km": distance_km,
            "duration_min": duration_min,
            "avg_speed_kmh": avg_speed.fillna(0.0),
            "straight_km": straight,
            "straightness": straightness,
            "point_count": 1.0,
            "hour_sin": np.sin(2 * np.pi * hours / 24.0),
            "hour_cos": np.cos(2 * np.pi * hours / 24.0),
            "weekday": weekday,
            "is_weekend": (weekday >= 5).astype(float),
            "log_distance": np.log1p(distance_km),
        },
        index=moves.index,
    )
    if "path_point_count" in moves:
        features["point_count"] = pd.to_numeric(moves["path_point_count"], errors="coerce").fillna(1.0)
    return features[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0.0)


@dataclass
class TransportModeClassifier:
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
    model: Any = None
    metrics: dict[str, Any] = field(default_factory=dict)
    feature_importance: pd.DataFrame = field(default_factory=pd.DataFrame)
    trained: bool = False

    def fit(self, segments: pd.DataFrame, test_size: float = 0.25) -> TransportModeClassifier:
        moves = _move_rows(segments)
        if moves.empty:
            self.metrics = {"trained": False, "reason": "No movement segments available."}
            return self
        features = extract_features(moves)
        labels = moves["mode"].map(normalize_mode)
        usable = (~labels.isin(["unknown", "other"])) & features.notna().all(axis=1)
        features = features[usable]
        labels = labels[usable]
        if len(features) < 20:
            self.metrics = {
                "trained": False,
                "reason": f"Only {len(features)} labelled movement segments; at least 20 are required.",
                "labelled_segments": len(features),
            }
            return self
        if labels.nunique() < 2:
            self.metrics = {
                "trained": False,
                "reason": "At least two distinct transport modes are required for training.",
                "labelled_segments": len(features),
            }
            return self

        class_counts = labels.value_counts()
        can_stratify = bool(len(labels) >= 2 * len(class_counts) and class_counts.min() >= 2)
        stratify = labels if can_stratify else None
        X_train, X_test, y_train, y_test = train_test_split(
            features,
            labels,
            test_size=min(max(test_size, 0.2), 0.4),
            random_state=42,
            stratify=stratify,
        )

        evaluation_model = RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
            n_jobs=-1,
        )
        evaluation_model.fit(X_train, y_train)
        predictions = evaluation_model.predict(X_test)
        labels_order = sorted(labels.unique().tolist())
        report = classification_report(
            y_test,
            predictions,
            output_dict=True,
            zero_division=0,
            labels=labels_order,
        )
        matrix = confusion_matrix(y_test, predictions, labels=labels_order)
        self.metrics = {
            "trained": True,
            "labelled_segments": len(features),
            "training_segments": len(X_train),
            "test_segments": len(X_test),
            "classes": labels_order,
            "accuracy": float(accuracy_score(y_test, predictions)),
            "f1_macro": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
            "report": report,
            "confusion_matrix": matrix.tolist(),
            "confusion_labels": labels_order,
            "test_size": float(len(X_test) / len(features)),
        }

        final_model = RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
            n_jobs=-1,
        )
        final_model.fit(features, labels)
        self.model = final_model
        self.trained = True
        self.feature_importance = (
            pd.DataFrame(
                {
                    "feature": FEATURE_COLUMNS,
                    "importance": final_model.feature_importances_,
                }
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
        return self

    def predict(self, segments: pd.DataFrame) -> pd.DataFrame:
        moves = _move_rows(segments)
        if moves.empty:
            return pd.DataFrame(columns=["predicted_mode", "mode_confidence", "second_mode"])
        if not self.trained or self.model is None:
            return pd.DataFrame(
                {
                    "predicted_mode": "unknown",
                    "mode_confidence": 0.0,
                    "second_mode": "unknown",
                },
                index=moves.index,
            )
        features = extract_features(moves)
        probabilities = self.model.predict_proba(features)
        classes = np.asarray(self.model.classes_)
        best = np.argmax(probabilities, axis=1)
        predicted = classes[best]
        confidence = probabilities[np.arange(len(best)), best]
        second = np.full(len(best), "unknown", dtype=object)
        if probabilities.shape[1] > 1:
            second_indices = np.argsort(probabilities, axis=1)[:, -2]
            second = classes[second_indices]
        return pd.DataFrame(
            {
                "predicted_mode": predicted,
                "mode_confidence": confidence,
                "second_mode": second,
            },
            index=moves.index,
        )

    def per_class_report(self) -> pd.DataFrame:
        if not self.metrics.get("trained"):
            return pd.DataFrame()
        report = self.metrics.get("report", {})
        rows = []
        for label in self.metrics.get("confusion_labels", []):
            values = report.get(label)
            if isinstance(values, dict):
                rows.append(
                    {
                        "mode": label,
                        "precision": values.get("precision", 0.0),
                        "recall": values.get("recall", 0.0),
                        "f1": values.get("f1-score", 0.0),
                        "support": values.get("support", 0),
                    }
                )
        return pd.DataFrame(rows)


def enrich_segments(
    segments: pd.DataFrame,
    classifier: TransportModeClassifier | None,
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> pd.DataFrame:
    if segments is None or segments.empty:
        return segments
    enriched = segments.copy()
    enriched["mode_original"] = enriched["mode"].map(normalize_mode)
    enriched["mode_final"] = enriched["mode_original"]
    enriched["mode_confidence"] = np.nan
    enriched["predicted_mode"] = ""
    enriched["second_mode"] = ""
    enriched["mode_source"] = np.where(enriched["segment_type"] == "stay", "stationary", "timeline")

    move_mask = enriched["segment_type"] == "move"
    if classifier is None or not classifier.trained:
        unknown_mask = move_mask & enriched["mode_original"].isin(["unknown", "other"])
        enriched.loc[unknown_mask, "mode_source"] = "unknown"
        return enriched

    predictions = classifier.predict(enriched)
    if predictions.empty:
        return enriched
    aligned = predictions.reindex(enriched.index)
    enriched.loc[move_mask, "predicted_mode"] = aligned.loc[move_mask, "predicted_mode"].fillna("unknown")
    enriched.loc[move_mask, "mode_confidence"] = aligned.loc[move_mask, "mode_confidence"]
    enriched.loc[move_mask, "second_mode"] = aligned.loc[move_mask, "second_mode"].fillna("unknown")

    unknown_mask = move_mask & enriched["mode_original"].isin(["unknown", "other"])
    confident = unknown_mask & (enriched["mode_confidence"].fillna(0.0) >= threshold)
    enriched.loc[confident, "mode_final"] = enriched.loc[confident, "predicted_mode"]
    enriched.loc[confident, "mode_source"] = "model"
    enriched.loc[unknown_mask & ~confident, "mode_final"] = "unknown"
    enriched.loc[unknown_mask & ~confident, "mode_source"] = "unknown"
    return enriched


def train_and_enrich(
    segments: pd.DataFrame,
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> tuple[TransportModeClassifier, pd.DataFrame]:
    classifier = TransportModeClassifier(threshold=threshold)
    classifier.fit(segments)
    enriched = enrich_segments(segments, classifier, threshold=threshold)
    return classifier, enriched
