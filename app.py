from __future__ import annotations

import hashlib
import html
import json

import pandas as pd
import streamlit as st

from gps_travel.charts import (
    cluster_comparison_chart,
    confusion_matrix_figure,
    daily_chart,
    feature_importance_figure,
    hourly_chart,
    mode_bar,
    mode_pie,
    monthly_chart,
    places_chart,
    transport_dataset_chart,
    weekday_chart,
)
from gps_travel.config import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_EPS_M,
    DEFAULT_KMEANS_CLUSTERS,
    DEFAULT_MIN_SAMPLES,
    HEAT_LAYERS,
    PAGE_ICON,
)
from gps_travel.geocoding import reverse_geocode_places
from gps_travel.mapping import build_heatmap, build_route_map
from gps_travel.modes import mode_label
from gps_travel.parser import parse_timeline
from gps_travel.pipeline import AnalysisResult, analyze
from gps_travel.report import generate_report_pdf
from gps_travel.sample_data import generate_sample_timeline

st.set_page_config(
    page_title="GPS Travel History Analysis",
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1500px;}
    div[data-testid="stMetric"] {background: #f7f9fb; border: 1px solid #e1e7ec; border-radius: 10px; padding: 12px 14px;}
    div[data-testid="stMetricLabel"] {color: #546e7a;}
    .app-title {font-size: 2.1rem; font-weight: 700; color: #0f4c81; margin-bottom: 0.1rem;}
    .app-subtitle {font-size: 1rem; color: #546e7a; margin-bottom: 1.2rem;}
    .small-note {font-size: 0.82rem; color: #607d8b;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_timeline(raw: bytes):
    payload = json.loads(raw.decode("utf-8-sig"))
    return parse_timeline(payload)


@st.cache_data(show_spinner=False, max_entries=4)
def run_analysis(
    raw: bytes,
    start_iso: str,
    end_iso: str,
    eps_m: float,
    min_samples: int,
    algorithm: str,
    n_clusters: int,
    predict_modes: bool,
    threshold: float,
):
    data = parse_timeline(json.loads(raw.decode("utf-8-sig")))
    filtered = data.filtered(start_iso, end_iso)
    return analyze(
        filtered,
        eps_m=eps_m,
        min_samples=min_samples,
        algorithm=algorithm,
        n_clusters=n_clusters,
        predict_modes=predict_modes,
        confidence_threshold=threshold,
    )


@st.cache_data(show_spinner=False)
def demo_payload(days: int = 45, seed: int = 7) -> bytes:
    return json.dumps(generate_sample_timeline(days=days, seed=seed)).encode("utf-8")


def render_map(folium_map, height: int = 540) -> None:
    st.iframe(folium_map._repr_html_(), height=height)


def fmt_hours(hours: float) -> str:
    total_minutes = round(hours * 60)
    return f"{total_minutes // 60} h {total_minutes % 60:02d} min"


def fmt_coord(lat: float, lon: float) -> str:
    return f"{lat:.5f}, {lon:.5f}"


def csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8")


st.sidebar.markdown("## Configuration")
st.sidebar.markdown("### Data source")
uploaded = st.sidebar.file_uploader("Google Maps Timeline JSON", type=["json"])
if st.sidebar.button("Load demo dataset", width="stretch"):
    st.session_state["raw_bytes"] = demo_payload()
    st.session_state["raw_name"] = "demo_timeline.json"
    st.session_state.pop("pdf_bytes", None)
    st.session_state.pop("pdf_signature", None)
    st.session_state.pop("geocoded_places", None)
    st.session_state.pop("geocoded_signature", None)

demo_data = demo_payload()
st.sidebar.download_button(
    "Download demo JSON",
    data=demo_data,
    file_name="demo_timeline.json",
    mime="application/json",
    width="stretch",
)

raw_bytes: bytes | None = None
source_name = ""
if uploaded is not None:
    raw_bytes = uploaded.getvalue()
    source_name = uploaded.name
elif st.session_state.get("raw_bytes"):
    raw_bytes = st.session_state["raw_bytes"]
    source_name = st.session_state.get("raw_name", "demo_timeline.json")

if raw_bytes is not None and st.sidebar.button("Clear loaded data", width="stretch"):
    st.session_state.pop("raw_bytes", None)
    st.session_state.pop("raw_name", None)
    st.session_state.pop("pdf_bytes", None)
    st.session_state.pop("pdf_signature", None)
    st.session_state.pop("geocoded_places", None)
    st.session_state.pop("geocoded_signature", None)
    st.rerun()

with st.sidebar.expander("Privacy and data handling"):
    st.markdown(
        "Timeline files are processed inside this application session. No coordinates are sent to external "
        "services unless you explicitly enable reverse geocoding, which sends place coordinates to OpenStreetMap "
        "Nominatim. Remove uploaded files when they are no longer needed."
    )

if raw_bytes is None:
    st.markdown('<div class="app-title">Intelligent GPS-Based Travel History Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-subtitle">Transform a Google Maps Timeline export into travel insights, maps, transport '
        "statistics and a downloadable report.</div>",
        unsafe_allow_html=True,
    )
    left, right = st.columns(2)
    left.markdown("#### Geospatial analytics\nStay detection, DBSCAN / K-Means place clustering, heatmaps and route maps.")
    right.markdown("#### Automated reporting\nDaily, weekly and monthly statistics rendered into a presentable PDF report.")
    st.divider()
    st.subheader("Try the demo dataset")
    st.write("A synthetic 45-day Timeline export is included so every dashboard section can be explored without uploading personal data.")
    if st.button("Load demo dataset", type="primary", key="landing_demo"):
        st.session_state["raw_bytes"] = demo_data
        st.session_state["raw_name"] = "demo_timeline.json"
        st.rerun()
    st.download_button("Download sample Timeline JSON", data=demo_data, file_name="demo_timeline.json", mime="application/json")
    with st.expander("Supported Timeline formats"):
        st.markdown(
            "- Modern exports with `semanticSegments` (visit, activity and timelinePath records)\n"
            "- Google Takeout semantic monthly files with `timelineObjects` (`placeVisit` and `activitySegment`)\n"
            "- Classic Location History files with a `locations` array\n"
            "- Exports containing only `rawSignals` with embedded position records"
        )
    st.stop()

try:
    with st.spinner("Parsing Timeline JSON..."):
        base_data = load_timeline(raw_bytes)
except (ValueError, TypeError, KeyError, OSError, UnicodeError) as error:
    st.error(f"Could not parse the uploaded file: {error}")
    st.stop()

if base_data.points.empty and base_data.segments.empty:
    st.error("The uploaded file was parsed but contained no usable location, visit or activity records.")
    st.stop()

start_limit = base_data.start_time
end_limit = base_data.end_time
if start_limit is None or end_limit is None:
    stamps = pd.to_datetime(base_data.points["timestamp"], utc=True, errors="coerce").dropna()
    if stamps.empty:
        st.error("No timestamped records were available.")
        st.stop()
    start_limit, end_limit = stamps.min(), stamps.max()

start_date = start_limit.date()
end_date = end_limit.date()
signature = hashlib.md5(raw_bytes).hexdigest()[:10]

st.sidebar.markdown("### Analysis period")
selected_dates = st.sidebar.date_input(
    "Date range",
    value=(start_date, end_date),
    min_value=start_date,
    max_value=end_date,
    key=f"date_range_{signature}",
)
if isinstance(selected_dates, (tuple, list)):
    if len(selected_dates) == 0:
        range_start, range_end = start_date, end_date
    elif len(selected_dates) == 1:
        range_start = range_end = selected_dates[0]
    else:
        range_start, range_end = selected_dates[0], selected_dates[1]
else:
    range_start = range_end = selected_dates
if range_end < range_start:
    range_start, range_end = range_end, range_start

st.sidebar.markdown("### Geospatial analysis")
algorithm_label = st.sidebar.radio("Place clustering", ["DBSCAN", "K-Means"], horizontal=True)
if algorithm_label == "DBSCAN":
    eps_m = st.sidebar.slider("Cluster radius (metres)", min_value=50, max_value=500, value=int(DEFAULT_EPS_M), step=10)
    min_samples = st.sidebar.slider("Minimum stays per place", min_value=2, max_value=10, value=DEFAULT_MIN_SAMPLES)
    n_clusters = DEFAULT_KMEANS_CLUSTERS
    algorithm = "dbscan"
else:
    n_clusters = st.sidebar.slider("Number of clusters", min_value=2, max_value=15, value=DEFAULT_KMEANS_CLUSTERS)
    eps_m = float(DEFAULT_EPS_M)
    min_samples = DEFAULT_MIN_SAMPLES
    algorithm = "kmeans"

st.sidebar.markdown("### Transport-mode estimation")
predict_modes = st.sidebar.toggle("Estimate unlabelled modes", value=True)
threshold = st.sidebar.slider(
    "Minimum prediction confidence",
    min_value=0.40,
    max_value=0.90,
    value=DEFAULT_CONFIDENCE_THRESHOLD,
    step=0.05,
    disabled=not predict_modes,
)

start_iso = range_start.isoformat()
end_iso = range_end.isoformat()

with st.spinner("Running geospatial and statistical analysis..."):
    result: AnalysisResult = run_analysis(
        raw_bytes,
        start_iso,
        end_iso,
        eps_m,
        min_samples,
        algorithm,
        n_clusters,
        predict_modes,
        threshold,
    )

if result.data.segments.empty:
    st.warning("No stay or movement records fall within the selected date range. Widen the date range to continue.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.markdown("### Loaded data")
st.sidebar.write(f"**File:** {source_name}")
st.sidebar.write(f"**Format:** {base_data.meta.get('source_format', 'unknown')}")
st.sidebar.write(f"**Records:** {len(result.data.segments):,} segments, {len(result.data.points):,} points")
st.sidebar.caption("Analysis settings and date range are applied immediately through cached recomputation.")

overview = result.overview
analysis_signature = hashlib.md5(
    f"{signature}|{start_iso}|{end_iso}|{eps_m}|{min_samples}|{algorithm}|{n_clusters}|{predict_modes}|{threshold}".encode()
).hexdigest()

st.markdown('<div class="app-title">Intelligent GPS-Based Travel History Analysis</div>', unsafe_allow_html=True)
period_start = pd.Timestamp(range_start).strftime("%d %b %Y")
period_end = pd.Timestamp(range_end).strftime("%d %b %Y")
st.markdown(
    f'<div class="app-subtitle">Analysis period: {period_start} to {period_end} &nbsp;|&nbsp; '
    f"Source: {html.escape(source_name)}</div>",
    unsafe_allow_html=True,
)

for warning in base_data.meta.get("warnings", []):
    st.warning(warning)

tabs = st.tabs(["Overview", "Frequently visited places", "Heatmap", "Transport analysis", "Routes and report"])

with tabs[0]:
    metric_columns = st.columns(4)
    metric_columns[0].metric("Total distance", f"{overview['total_distance_km']:,.1f} km")
    metric_columns[1].metric("Journeys", f"{overview['journeys']:,}")
    metric_columns[2].metric("Clustered places", f"{overview['visited_places']:,}")
    metric_columns[3].metric("Active travel days", f"{overview['active_days']:,}")

    secondary = st.columns(4)
    secondary[0].metric("Moving time", fmt_hours(overview["total_duration_h"]))
    secondary[1].metric("Stay time", fmt_hours(overview["total_stay_hours"]))
    secondary[2].metric("Dominant mode", overview["top_mode"] or "n/a", f"{overview['top_mode_share'] * 100:.1f}% of distance")
    secondary[3].metric("Average daily distance", f"{overview['average_daily_distance_km']:.1f} km")

    st.plotly_chart(daily_chart(result.daily), width="stretch")
    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.plotly_chart(monthly_chart(result.monthly), width="stretch")
    with chart_right:
        st.plotly_chart(weekday_chart(result.weekday, result.weekday_weekend), width="stretch")
    hour_columns = st.columns([2, 1])
    with hour_columns[0]:
        st.plotly_chart(hourly_chart(result.hourly), width="stretch")
    with hour_columns[1]:
        st.markdown("#### Weekday vs weekend")
        if result.weekday_weekend.empty:
            st.caption("No movement records available.")
        else:
            st.dataframe(
                result.weekday_weekend.assign(
                    distance_km=lambda frame: frame["distance_km"].round(2),
                    duration_min=lambda frame: frame["duration_min"].round(0),
                ),
                hide_index=True,
                width="stretch",
            )

    with st.expander("Data quality and method details"):
        st.write(f"**Source format:** `{base_data.meta.get('source_format', 'unknown')}`")
        st.write(f"**Raw counts:** {base_data.meta.get('raw_counts', {})}")
        st.write(f"**Derived records:** {base_data.meta.get('moves', 0)} movement segments, {base_data.meta.get('stays', 0)} stay segments.")
        unlabeled = result.data.segments[
            (result.data.segments["segment_type"] == "move")
            & result.data.segments["mode_original"].isin(["unknown", "other"])
        ] if "mode_original" in result.data.segments else pd.DataFrame()
        st.write(
            f"**Unlabelled movement segments:** {len(unlabeled):,} of {int((result.data.segments['segment_type'] == 'move').sum()):,}"
        )
        st.write(
            "Distances are computed from recorded route geometry where available; otherwise consecutive location points "
            "are used after removing implausible speed jumps. Clustered places are candidates, not confirmed addresses."
        )
        st.download_button(
            "Download analysed segments CSV",
            data=csv_bytes(result.data.segments),
            file_name="analysed_segments.csv",
            mime="text/csv",
        )
        st.download_button(
            "Download processed points CSV",
            data=csv_bytes(result.data.points),
            file_name="processed_points.csv",
            mime="text/csv",
        )

with tabs[1]:
    st.subheader("Frequently visited locations")
    if result.places.empty:
        st.info("No repeated stay locations were detected with the current clustering settings.")
    else:
        places_view = result.places.copy()
        places_view["Total stay"] = places_view["total_stay_s"].map(fmt_hours)
        places_view["Average stay"] = (places_view["avg_stay_s"] / 60.0).round(0).astype(int).astype(str) + " min"
        places_view["Coordinates"] = [fmt_coord(lat, lon) for lat, lon in zip(places_view["lat"], places_view["lon"])]
        places_view["Radius (m)"] = places_view["radius_m"].round(0).astype(int)
        places_view["First visit"] = pd.to_datetime(places_view["first_visit"], utc=True).dt.strftime("%d %b %Y")
        places_view["Last visit"] = pd.to_datetime(places_view["last_visit"], utc=True).dt.strftime("%d %b %Y")
        display_columns = ["place_id", "name", "visit_count", "Total stay", "Average stay", "Coordinates", "Radius (m)", "First visit", "Last visit"]
        geocoded = st.session_state.get("geocoded_places")
        if (
            st.session_state.get("geocoded_signature") == analysis_signature
            and geocoded is not None
            and "address" in geocoded.columns
            and len(geocoded) == len(places_view)
        ):
            places_view["Address"] = geocoded["address"].to_numpy()
            display_columns.insert(2, "Address")
        st.dataframe(
            places_view[display_columns].rename(
                columns={"place_id": "ID", "name": "Place", "visit_count": "Visits"}
            ),
            hide_index=True,
            width="stretch",
        )
        st.plotly_chart(places_chart(result.places, metric="visit_count"), width="stretch")
        render_map(
            build_route_map(result.data, result.places, show_places=True, show_stays=False, show_routes=False),
            height=520,
        )
        with st.expander("Clustering experiment: DBSCAN vs K-Means"):
            st.caption(
                "Both algorithms are run on the same stay locations. Silhouette is computed with haversine distance "
                "on clustered points and is indicative rather than a definitive measure of place quality."
            )
            if result.cluster_comparison.empty:
                st.info("At least three stay records are required for the comparison.")
            else:
                st.dataframe(result.cluster_comparison, hide_index=True, width="stretch")
                st.plotly_chart(cluster_comparison_chart(result.cluster_comparison), width="stretch")
        with st.expander("Optional reverse geocoding"):
            st.warning(
                "Reverse geocoding sends the coordinates of detected places to the OpenStreetMap Nominatim service. "
                "Enable it only if you accept that disclosure."
            )
            if st.button("Resolve place addresses", key="geocode_button"):
                try:
                    with st.spinner("Querying Nominatim..."):
                        geocoded = reverse_geocode_places(
                            result.places,
                            cache_path="data/geocode_cache.json",
                            max_places=12,
                        )
                    st.session_state["geocoded_places"] = geocoded
                    st.session_state["geocoded_signature"] = analysis_signature
                    st.success("Address lookup completed. Addresses are shown in the table above.")
                except Exception as error:
                    st.error(f"Reverse geocoding failed: {error}")

with tabs[2]:
    st.subheader("Geographical heatmap")
    weight_key = st.radio(
        "Heatmap weighting",
        options=list(HEAT_LAYERS.keys()),
        format_func=lambda key: HEAT_LAYERS[key],
        horizontal=True,
    )
    explanations = {
        "travel_density": "Each recorded or route-sampled location point contributes equally, highlighting corridors and frequently travelled areas.",
        "visit_frequency": "Detected places are weighted by the number of visits, highlighting repeated destinations.",
        "stay_duration": "Detected places are weighted by estimated total stay duration, highlighting places where the most time was spent.",
    }
    st.caption(explanations[weight_key])
    render_map(build_heatmap(result.data, result.places, weight_by=weight_key), height=580)
    st.markdown(
        '<span class="small-note">Heatmap intensity is a relative analytical weight, not a measure of absolute GPS accuracy. '
        "Sparse exports can produce broader and less precise hot spots.</span>",
        unsafe_allow_html=True,
    )

with tabs[3]:
    st.subheader("Transport mode analysis")
    if result.mode_summary.empty:
        st.info("No movement segments were available for transport analysis.")
    else:
        unlabeled_moves = result.data.segments[
            (result.data.segments["segment_type"] == "move")
            & result.data.segments["mode_original"].isin(["unknown", "other"])
        ]
        transport_metrics = st.columns(4)
        transport_metrics[0].metric("Dominant mode", result.mode_summary.iloc[0]["label"])
        transport_metrics[1].metric("Mode count", len(result.mode_summary))
        transport_metrics[2].metric("Unlabelled moves", f"{len(unlabeled_moves):,}")
        transport_metrics[3].metric(
            "Estimated by model",
            f"{int((result.data.segments['mode_source'] == 'model').sum()):,}",
        )
        chart_left, chart_right = st.columns(2)
        with chart_left:
            st.plotly_chart(mode_pie(result.mode_summary), width="stretch")
        with chart_right:
            st.plotly_chart(mode_bar(result.mode_summary), width="stretch")
        mode_table = result.mode_summary.copy()
        mode_table["Distance share"] = (mode_table["share_distance"] * 100).round(1)
        mode_table["Journey share"] = (mode_table["share_journeys"] * 100).round(1)
        st.dataframe(
            mode_table[["label", "journeys", "distance_km", "duration_h", "Distance share", "Journey share"]].rename(
                columns={
                    "label": "Mode",
                    "journeys": "Journeys",
                    "distance_km": "Distance (km)",
                    "duration_h": "Duration (h)",
                }
            ),
            hide_index=True,
            width="stretch",
        )
        st.markdown("#### Machine-learning experiment")
        metrics = result.classifier.metrics
        if metrics.get("trained"):
            ml_columns = st.columns(4)
            ml_columns[0].metric("Accuracy", f"{metrics.get('accuracy', 0):.3f}")
            ml_columns[1].metric("Macro F1", f"{metrics.get('f1_macro', 0):.3f}")
            ml_columns[2].metric("Training segments", f"{metrics.get('training_segments', 0):,}")
            ml_columns[3].metric("Held-out segments", f"{metrics.get('test_segments', 0):,}")
            report_tab, confusion_tab, importance_tab = st.tabs(["Per-class scores", "Confusion matrix", "Feature importance"])
            with report_tab:
                st.dataframe(
                    result.classifier.per_class_report().rename(
                        columns={"mode": "Mode", "precision": "Precision", "recall": "Recall", "f1": "F1", "support": "Support"}
                    ),
                    hide_index=True,
                    width="stretch",
                )
                st.plotly_chart(transport_dataset_chart(result.classifier), width="stretch")
            with confusion_tab:
                st.plotly_chart(confusion_matrix_figure(result.classifier), width="stretch")
            with importance_tab:
                st.plotly_chart(feature_importance_figure(result.classifier), width="stretch")
            st.caption(
                "The model is trained only on movement segments that already carry a transport label. "
                "Low-confidence predictions remain Unknown rather than being forced into a mode."
            )
        else:
            st.info(metrics.get("reason", "A transport classifier could not be trained on this dataset."))
        with st.expander("Show unlabelled journeys and model estimates"):
            if unlabeled_moves.empty:
                st.write("No unlabelled movement segments are present in the selected period.")
            else:
                estimate_view = unlabeled_moves.copy()
                estimate_view["start_ts"] = pd.to_datetime(estimate_view["start_ts"], utc=True).dt.strftime("%d %b %Y %H:%M")
                estimate_view["mode_confidence"] = pd.to_numeric(estimate_view["mode_confidence"], errors="coerce").round(3)
                columns = ["start_ts", "distance_m", "duration_s", "predicted_mode", "mode_confidence", "mode_final"]
                st.dataframe(
                    estimate_view[columns].head(300).rename(
                        columns={
                            "start_ts": "Start",
                            "distance_m": "Distance (m)",
                            "duration_s": "Duration (s)",
                            "predicted_mode": "Predicted mode",
                            "mode_confidence": "Confidence",
                            "mode_final": "Final mode",
                        }
                    ),
                    hide_index=True,
                    width="stretch",
                )
        st.download_button(
            "Download transport analysis CSV",
            data=csv_bytes(result.data.segments),
            file_name="transport_segments.csv",
            mime="text/csv",
        )

with tabs[4]:
    st.subheader("Routes and repeated journeys")
    available_modes = result.mode_summary["mode"].tolist() if not result.mode_summary.empty else []
    selected_modes = st.multiselect(
        "Map transport mode filter",
        options=available_modes,
        default=available_modes,
        format_func=mode_label,
    )
    map_options = st.columns(4)
    show_places = map_options[0].checkbox("Show clustered places", value=True)
    show_stays = map_options[1].checkbox("Show stay markers", value=False)
    show_endpoints = map_options[2].checkbox("Show journey endpoints", value=False)
    render_map(
        build_route_map(
            result.data,
            result.places,
            modes=selected_modes,
            show_places=show_places,
            show_stays=show_stays,
            show_endpoints=show_endpoints,
        ),
        height=620,
    )
    if result.repeated_routes.empty:
        st.info("No repeated origin-destination journeys were found.")
    else:
        st.markdown("#### Frequently repeated journeys")
        repeated = result.repeated_routes.copy()
        repeated["distance_km"] = repeated["distance_km"].round(2)
        repeated["average_duration_min"] = repeated["average_duration_min"].round(0)
        st.dataframe(
            repeated.rename(
                columns={
                    "route": "Route",
                    "mode": "Mode",
                    "trips": "Trips",
                    "distance_km": "Average distance (km)",
                    "total_distance_km": "Total distance (km)",
                    "average_duration_min": "Average duration (min)",
                    "first_trip": "First trip",
                    "last_trip": "Last trip",
                }
            ),
            hide_index=True,
            width="stretch",
        )
    with st.expander("Weekly travel statistics"):
        if result.weekly.empty:
            st.write("No weekly statistics available.")
        else:
            weekly_view = result.weekly.copy()
            weekly_view["distance_km"] = weekly_view["distance_km"].round(2)
            weekly_view["duration_min"] = weekly_view["duration_min"].round(0)
            st.dataframe(weekly_view, hide_index=True, width="stretch")

    st.divider()
    st.subheader("Automated PDF report")
    st.write(
        "The report contains the analysis period, overall summary, top places, transport statistics, clustering and "
        "machine-learning evaluations, map figures, temporal charts and methodology limitations."
    )
    if st.button("Generate PDF report", type="primary"):
        try:
            with st.spinner("Building PDF report..."):
                st.session_state["pdf_bytes"] = generate_report_pdf(result)
                st.session_state["pdf_signature"] = analysis_signature
            st.success("Report generated.")
        except Exception as error:
            st.error(f"Report generation failed: {error}")
    if st.session_state.get("pdf_signature") == analysis_signature and st.session_state.get("pdf_bytes"):
        st.download_button(
            "Download PDF report",
            data=st.session_state["pdf_bytes"],
            file_name="travel_history_report.pdf",
            mime="application/pdf",
            type="primary",
        )
    else:
        st.caption("Generate the report after confirming the date range and analysis settings.")
