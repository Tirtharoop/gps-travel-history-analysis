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
from gps_travel.mapping import build_heatmap, build_route_map
from gps_travel.parser import parse_timeline
from gps_travel.pipeline import analyze
from gps_travel.report import generate_report_pdf
from gps_travel.sample_data import generate_sample_timeline


def _result():
    payload = generate_sample_timeline(days=14, seed=8)
    data = parse_timeline(payload)
    return analyze(data, eps_m=200, min_samples=2, predict_modes=True)


def test_figures_and_maps_render():
    result = _result()
    assert daily_chart(result.daily).data
    assert monthly_chart(result.monthly).data
    assert hourly_chart(result.hourly).data
    assert weekday_chart(result.weekday, result.weekday_weekend).data
    assert mode_pie(result.mode_summary).data
    assert mode_bar(result.mode_summary).data
    assert places_chart(result.places).data
    assert cluster_comparison_chart(result.cluster_comparison).data
    assert confusion_matrix_figure(result.classifier).data
    assert feature_importance_figure(result.classifier).data
    assert transport_dataset_chart(result.classifier).data

    route_map = build_route_map(result.data, result.places)
    heat_map = build_heatmap(result.data, result.places, weight_by="stay_duration")
    assert route_map._repr_html_()
    assert heat_map._repr_html_()


def test_pdf_report_generation():
    result = _result()
    pdf = generate_report_pdf(result)
    assert pdf.startswith(b"%PDF")
    assert len(pdf) > 10_000


def test_pdf_report_without_classifier():
    payload = generate_sample_timeline(days=6, seed=2)
    data = parse_timeline(payload)
    result = analyze(data, predict_modes=False)
    pdf = generate_report_pdf(result)
    assert pdf.startswith(b"%PDF")
