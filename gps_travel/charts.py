from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from gps_travel.transport import TransportModeClassifier

_TEMPLATE = "plotly_white"


def _empty(message: str) -> go.Figure:
    figure = go.Figure()
    figure.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 14, "color": "#78909c"},
    )
    figure.update_xaxes(visible=False)
    figure.update_yaxes(visible=False)
    figure.update_layout(height=300, template=_TEMPLATE, margin={"l": 10, "r": 10, "t": 10, "b": 10})
    return figure


def _finish(figure: go.Figure, height: int = 320) -> go.Figure:
    figure.update_layout(
        template=_TEMPLATE,
        height=height,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    return figure


def daily_chart(daily: pd.DataFrame) -> go.Figure:
    if daily is None or daily.empty or daily["distance_km"].sum() <= 0:
        return _empty("No daily movement records available")
    figure = px.bar(
        daily,
        x="date",
        y="distance_km",
        labels={"date": "Date", "distance_km": "Distance (km)"},
        color_discrete_sequence=["#1565c0"],
    )
    figure.update_traces(customdata=daily[["journeys", "duration_min"]], hovertemplate="%{x|%d %b %Y}<br>%{y:.2f} km<br>%{customdata[0]} journeys<br>%{customdata[1]:.0f} min<extra></extra>")
    return _finish(figure)


def monthly_chart(monthly: pd.DataFrame) -> go.Figure:
    if monthly is None or monthly.empty:
        return _empty("No monthly records available")
    figure = px.bar(
        monthly,
        x="month",
        y="distance_km",
        labels={"month": "Month", "distance_km": "Distance (km)"},
        color_discrete_sequence=["#6a1b9a"],
    )
    return _finish(figure)


def hourly_chart(hourly: pd.DataFrame) -> go.Figure:
    if hourly is None or hourly.empty or hourly["journeys"].sum() <= 0:
        return _empty("No hourly movement records available")
    figure = px.bar(
        hourly,
        x="hour",
        y="journeys",
        labels={"hour": "Hour of day", "journeys": "Journeys"},
        color_discrete_sequence=["#00838f"],
    )
    figure.update_xaxes(dtick=2)
    return _finish(figure)


def weekday_chart(weekday: pd.DataFrame, weekend: pd.DataFrame | None = None) -> go.Figure:
    if weekday is None or weekday.empty or weekday["journeys"].sum() <= 0:
        return _empty("No weekday records available")
    colors = ["#90a4ae"] * 5 + ["#ef6c00"] * 2
    figure = px.bar(
        weekday,
        x="weekday_name",
        y="distance_km",
        labels={"weekday_name": "", "distance_km": "Distance (km)"},
    )
    figure.update_traces(marker_color=colors)
    return _finish(figure)


def mode_pie(modes: pd.DataFrame, metric: str = "share_distance") -> go.Figure:
    if modes is None or modes.empty:
        return _empty("No transport modes available")
    figure = px.pie(
        modes,
        names="label",
        values="distance_km",
        color="label",
        color_discrete_map={row["label"]: row["color"] for row in modes.to_dict("records")},
        hole=0.42,
    )
    figure.update_traces(textposition="inside", textinfo="percent+label", hovertemplate="%{label}<br>%{value:.2f} km<br>%{percent}<extra></extra>")
    return _finish(figure)


def mode_bar(modes: pd.DataFrame) -> go.Figure:
    if modes is None or modes.empty:
        return _empty("No transport modes available")
    figure = px.bar(
        modes.sort_values("distance_km"),
        x="distance_km",
        y="label",
        orientation="h",
        color="label",
        color_discrete_map={row["label"]: row["color"] for row in modes.to_dict("records")},
        labels={"distance_km": "Distance (km)", "label": ""},
    )
    figure.update_layout(showlegend=False)
    return _finish(figure)


def places_chart(places: pd.DataFrame, metric: str = "visit_count", top_n: int = 10) -> go.Figure:
    if places is None or places.empty:
        return _empty("No clustered places available")
    frame = places.sort_values(metric, ascending=False).head(top_n).sort_values(metric)
    if metric == "total_stay_s":
        frame = frame.assign(value=frame["total_stay_s"] / 3600.0, axis_label="Total stay (hours)")
    else:
        frame = frame.assign(value=frame[metric], axis_label="Visits")
    figure = px.bar(
        frame,
        x="value",
        y="name",
        orientation="h",
        color="name",
        color_discrete_map={row["name"]: row["color"] for row in frame.to_dict("records")},
        labels={"value": frame["axis_label"].iloc[0], "name": ""},
    )
    figure.update_layout(showlegend=False)
    return _finish(figure)


def cluster_comparison_chart(comparison: pd.DataFrame) -> go.Figure:
    if comparison is None or comparison.empty:
        return _empty("Not enough stay records to compare clustering methods")
    figure = go.Figure()
    figure.add_bar(name="Clusters", x=comparison["algorithm"].str.upper(), y=comparison["clusters"], marker_color="#1565c0")
    figure.add_bar(name="Noise points", x=comparison["algorithm"].str.upper(), y=comparison["noise_points"], marker_color="#ef6c00")
    figure.update_layout(barmode="group", yaxis_title="Count")
    return _finish(figure)


def confusion_matrix_figure(classifier: TransportModeClassifier) -> go.Figure:
    metrics = classifier.metrics if classifier is not None else {}
    if not metrics.get("trained"):
        return _empty("Transport classifier has not been trained")
    labels = metrics.get("confusion_labels", [])
    matrix = metrics.get("confusion_matrix", [])
    figure = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=labels,
            y=labels,
            colorscale="Blues",
            text=matrix,
            texttemplate="%{text}",
            hovertemplate="Actual %{y}<br>Predicted %{x}<br>%{z}<extra></extra>",
        )
    )
    figure.update_layout(xaxis_title="Predicted", yaxis_title="Actual")
    return _finish(figure, height=360)


def feature_importance_figure(classifier: TransportModeClassifier) -> go.Figure:
    if classifier is None or classifier.feature_importance.empty:
        return _empty("No feature importance available")
    frame = classifier.feature_importance.sort_values("importance")
    figure = px.bar(
        frame,
        x="importance",
        y="feature",
        orientation="h",
        color_discrete_sequence=["#2e7d32"],
        labels={"importance": "Importance", "feature": ""},
    )
    figure.update_layout(showlegend=False)
    return _finish(figure, height=340)


def transport_dataset_chart(classifier: TransportModeClassifier) -> go.Figure:
    metrics = classifier.metrics if classifier is not None else {}
    if not metrics.get("trained"):
        return _empty("Transport classifier has not been trained")
    report = classifier.per_class_report()
    if report.empty:
        return _empty("No per-class evaluation available")
    figure = px.bar(
        report,
        x="mode",
        y=["precision", "recall", "f1"],
        barmode="group",
        labels={"value": "Score", "variable": "Metric", "mode": "Mode"},
        color_discrete_sequence=["#1565c0", "#2e7d32", "#ef6c00"],
    )
    figure.update_yaxes(range=[0, 1.05])
    return _finish(figure)
