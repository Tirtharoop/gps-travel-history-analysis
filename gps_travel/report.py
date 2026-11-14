from __future__ import annotations

import io
from datetime import UTC, datetime
from typing import Any
from xml.sax.saxutils import escape

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from gps_travel.modes import mode_color, mode_label
from gps_travel.pipeline import AnalysisResult

ACCENT = colors.HexColor("#0f4c81")
MUTED = colors.HexColor("#546e7a")
HEADER_BG = colors.HexColor("#263238")
ROW_BG = colors.HexColor("#f4f6f8")
GRID = colors.HexColor("#cfd8dc")


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "ReportTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=26,
            textColor=ACCENT,
            alignment=TA_LEFT,
            spaceAfter=4,
        ),
        "subtitle": ParagraphStyle(
            "ReportSubtitle",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=10.5,
            leading=14,
            textColor=MUTED,
            spaceAfter=12,
        ),
        "heading": ParagraphStyle(
            "ReportHeading",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=16,
            textColor=ACCENT,
            spaceBefore=12,
            spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "ReportBody",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=13,
            textColor=colors.HexColor("#263238"),
            spaceAfter=5,
        ),
        "small": ParagraphStyle(
            "ReportSmall",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=7.6,
            leading=9.5,
            textColor=colors.HexColor("#37474f"),
        ),
        "cell": ParagraphStyle(
            "ReportCell",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=7.8,
            leading=9.8,
            textColor=colors.HexColor("#263238"),
        ),
        "header": ParagraphStyle(
            "ReportHeader",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=7.8,
            leading=9.8,
            textColor=colors.white,
        ),
        "caption": ParagraphStyle(
            "ReportCaption",
            parent=base["Normal"],
            fontName="Helvetica-Oblique",
            fontSize=7.6,
            leading=9.5,
            textColor=MUTED,
            spaceBefore=2,
            spaceAfter=8,
        ),
    }


def _p(value: Any, style: ParagraphStyle) -> Paragraph:
    return Paragraph(escape(str(value)), style)


def _format_timestamp(value: Any) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    try:
        stamp = pd.Timestamp(value)
    except (ValueError, TypeError):
        return str(value)
    if pd.isna(stamp):
        return "n/a"
    return stamp.strftime("%d %b %Y, %H:%M")


def _format_date(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        stamp = pd.Timestamp(value)
    except (ValueError, TypeError):
        return str(value)
    if pd.isna(stamp):
        return "n/a"
    return stamp.strftime("%d %b %Y")


def _table(rows: list[list[Any]], styles: dict[str, ParagraphStyle], widths: list[float], header: bool = True) -> Table:
    converted = []
    for row_index, row in enumerate(rows):
        converted_row = []
        for value in row:
            style = styles["header"] if header and row_index == 0 else styles["cell"]
            converted_row.append(_p(value, style))
        converted.append(converted_row)
    table = Table(converted, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("GRID", (0, 0), (-1, -1), 0.4, GRID),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
    ]
    if header:
        commands.append(("BACKGROUND", (0, 0), (-1, 0), HEADER_BG))
    for row_index in range(1 if header else 0, len(rows)):
        if row_index % 2 == 0:
            commands.append(("BACKGROUND", (0, row_index), (-1, row_index), ROW_BG))
    table.setStyle(TableStyle(commands))
    return table


def _figure_buffer(fig: plt.Figure, dpi: int = 150) -> io.BytesIO:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buffer.seek(0)
    return buffer


def _daily_figure(daily: pd.DataFrame) -> io.BytesIO | None:
    if daily is None or daily.empty or daily["distance_km"].sum() <= 0:
        return None
    fig, ax = plt.subplots(figsize=(9.2, 3.2))
    ax.bar(daily["date"], daily["distance_km"], color="#1565c0", width=0.8)
    ax.set_ylabel("Distance (km)")
    ax.set_title("Daily travel distance", fontsize=10, loc="left", color="#263238")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=7.5, colors="#546e7a")
    fig.tight_layout()
    return _figure_buffer(fig)


def _monthly_figure(monthly: pd.DataFrame) -> io.BytesIO | None:
    if monthly is None or monthly.empty:
        return None
    fig, ax = plt.subplots(figsize=(9.2, 2.8))
    ax.bar(monthly["month"], monthly["distance_km"], color="#6a1b9a", width=0.55)
    ax.set_ylabel("Distance (km)")
    ax.set_title("Monthly travel distance", fontsize=10, loc="left", color="#263238")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=7.5, colors="#546e7a")
    fig.tight_layout()
    return _figure_buffer(fig)


def _mode_figure(modes: pd.DataFrame) -> io.BytesIO | None:
    if modes is None or modes.empty or modes["distance_km"].sum() <= 0:
        return None
    frame = modes[modes["distance_km"] > 0]
    if frame.empty:
        return None
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.pie(
        frame["distance_km"],
        labels=frame["label"],
        autopct=lambda value: f"{value:.0f}%" if value >= 4 else "",
        colors=[mode_color(mode) for mode in frame["mode"]],
        startangle=90,
        textprops={"fontsize": 8, "color": "#263238"},
        wedgeprops={"linewidth": 0.6, "edgecolor": "white"},
    )
    ax.set_title("Transport mode share by distance", fontsize=10, color="#263238")
    fig.tight_layout()
    return _figure_buffer(fig)


def _places_figure(places: pd.DataFrame) -> io.BytesIO | None:
    if places is None or places.empty:
        return None
    frame = places.sort_values("visit_count", ascending=False).head(10).sort_values("visit_count")
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    ax.barh(frame["name"], frame["visit_count"], color=frame["color"])
    for index, value in enumerate(frame["visit_count"]):
        ax.text(value + 0.2, index, str(int(value)), va="center", fontsize=7.5, color="#37474f")
    ax.set_xlabel("Visits")
    ax.set_title("Most frequently visited places", fontsize=10, loc="left", color="#263238")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=7.5, colors="#37474f")
    fig.tight_layout()
    return _figure_buffer(fig)


def _route_figure(result: AnalysisResult) -> io.BytesIO | None:
    data = result.data
    moves = data.segments[data.segments["segment_type"] == "move"] if not data.segments.empty else data.segments
    if moves is None or moves.empty:
        return None
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    used_modes: list[str] = []
    for row in moves.itertuples(index=False):
        path = data.paths.get(int(row.segment_id), [])
        if len(path) < 2:
            continue
        lats = [point[0] for point in path]
        lons = [point[1] for point in path]
        mode = str(getattr(row, "mode_final", getattr(row, "mode", "unknown")))
        ax.plot(lons, lats, color=mode_color(mode), linewidth=0.9, alpha=0.55)
        if mode not in used_modes:
            used_modes.append(mode)
    if result.places is not None and not result.places.empty:
        ax.scatter(
            result.places["lon"],
            result.places["lat"],
            s=55,
            c=result.places["color"],
            edgecolors="white",
            linewidths=0.8,
            zorder=5,
        )
        for row in result.places.head(8).itertuples(index=False):
            ax.annotate(
                str(row.name),
                (float(row.lon), float(row.lat)),
                fontsize=6.5,
                color="#263238",
                xytext=(4, 3),
                textcoords="offset points",
            )
    handles = [Line2D([0], [0], color=mode_color(mode), lw=2, label=mode_label(mode)) for mode in used_modes[:8]]
    if handles:
        ax.legend(handles=handles, fontsize=6.5, loc="best", frameon=True)
    ax.set_xlabel("Longitude", fontsize=8, color="#546e7a")
    ax.set_ylabel("Latitude", fontsize=8, color="#546e7a")
    ax.set_title("Recorded journey routes and clustered places", fontsize=10, loc="left", color="#263238")
    ax.tick_params(labelsize=7, colors="#546e7a")
    ax.grid(color="#eceff1", linewidth=0.5)
    fig.tight_layout()
    return _figure_buffer(fig)


def _heat_figure(result: AnalysisResult) -> io.BytesIO | None:
    points = result.data.points
    if points is None or points.empty or len(points) < 5:
        return None
    sample = points.iloc[:: max(1, len(points) // 15000)]
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    hexbin = ax.hexbin(
        sample["lon"].astype(float),
        sample["lat"].astype(float),
        gridsize=48,
        cmap="YlOrRd",
        mincnt=1,
        linewidths=0,
        bins="log",
    )
    fig.colorbar(hexbin, ax=ax, shrink=0.8, label="log(point count)")
    ax.set_xlabel("Longitude", fontsize=8, color="#546e7a")
    ax.set_ylabel("Latitude", fontsize=8, color="#546e7a")
    ax.set_title("Geographical travel density heatmap", fontsize=10, loc="left", color="#263238")
    ax.tick_params(labelsize=7, colors="#546e7a")
    fig.tight_layout()
    return _figure_buffer(fig)


def _footer(canvas: Any, doc: Any) -> None:
    canvas.saveState()
    canvas.setStrokeColor(GRID)
    canvas.setLineWidth(0.5)
    canvas.line(2 * cm, 1.55 * cm, A4[0] - 2 * cm, 1.55 * cm)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(2 * cm, 1.1 * cm, "Intelligent GPS-Based Travel History Analysis")
    canvas.drawRightString(A4[0] - 2 * cm, 1.1 * cm, f"Page {doc.page}")
    canvas.restoreState()


def generate_report_pdf(result: AnalysisResult, title: str | None = None) -> bytes:
    styles = _styles()
    analysis_title = title or "Intelligent GPS-Based Travel History Analysis"
    overview = result.overview
    buffer = io.BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=1.8 * cm,
        bottomMargin=2 * cm,
        title=analysis_title,
        author="GPS Travel Analytics",
        subject="Automated travel history report",
    )
    story: list[Any] = []
    story.append(Paragraph(analysis_title, styles["title"]))
    story.append(
        Paragraph(
            "Automated travel report generated from a Google Maps Timeline export using geospatial analytics, "
            "clustering, transport-mode analysis and interactive visualisation components.",
            styles["subtitle"],
        )
    )
    start = overview.get("analysis_start")
    end = overview.get("analysis_end")
    story.append(
        Paragraph(
            f"Analysis period: <b>{_format_date(start)}</b> to <b>{_format_date(end)}</b> &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"Report generated: {datetime.now(UTC).strftime('%d %b %Y, %H:%M UTC')}",
            styles["body"],
        )
    )

    story.append(Paragraph("1. Overall travel summary", styles["heading"]))
    summary_rows = [
        ["Metric", "Value", "Metric", "Value"],
        ["Total distance", f"{overview.get('total_distance_km', 0):.2f} km", "Journeys", str(overview.get("journeys", 0))],
        ["Moving duration", f"{overview.get('total_duration_h', 0):.2f} h", "Stay segments", str(overview.get("stay_segments", 0))],
        ["Total stay duration", f"{overview.get('total_stay_hours', 0):.2f} h", "Clustered places", str(overview.get("visited_places", 0))],
        ["Active travel days", str(overview.get("active_days", 0)), "Recorded points", str(overview.get("total_points", 0))],
        ["Average daily distance", f"{overview.get('average_daily_distance_km', 0):.2f} km", "Maximum daily distance", f"{overview.get('max_daily_distance_km', 0):.2f} km"],
        ["Dominant mode", str(overview.get("top_mode", "n/a")), "Dominant mode share", f"{overview.get('top_mode_share', 0) * 100:.1f}%"],
    ]
    story.append(_table(summary_rows, styles, [4.0 * cm, 4.0 * cm, 4.0 * cm, 4.6 * cm]))
    story.append(Spacer(1, 6))

    story.append(Paragraph("2. Frequently visited locations", styles["heading"]))
    if result.places is not None and not result.places.empty:
        place_rows = [["Place", "Visits", "Total stay (h)", "Coordinates", "First visit", "Last visit"]]
        for row in result.places.head(15).itertuples(index=False):
            place_rows.append(
                [
                    row.name,
                    int(row.visit_count),
                    f"{float(row.total_stay_s) / 3600.0:.1f}",
                    f"{float(row.lat):.5f}, {float(row.lon):.5f}",
                    _format_date(row.first_visit),
                    _format_date(row.last_visit),
                ]
            )
        story.append(_table(place_rows, styles, [4.2 * cm, 1.5 * cm, 2.4 * cm, 4.2 * cm, 2.2 * cm, 2.2 * cm]))
        story.append(Paragraph("Places are candidate locations derived from geographical clustering of stay events, not confirmed addresses.", styles["caption"]))
    else:
        story.append(Paragraph("No repeated stay locations were detected in the selected period.", styles["body"]))

    if result.cluster_comparison is not None and not result.cluster_comparison.empty:
        comparison_rows = [["Algorithm", "Clusters", "Noise points", "Noise ratio", "Silhouette", "Mean radius (m)"]]
        for row in result.cluster_comparison.itertuples(index=False):
            silhouette = "n/a" if row.silhouette is None or (isinstance(row.silhouette, float) and not np.isfinite(row.silhouette)) else f"{float(row.silhouette):.3f}"
            comparison_rows.append(
                [
                    str(row.algorithm).upper(),
                    int(row.clusters),
                    int(row.noise_points),
                    f"{float(row.noise_ratio) * 100:.1f}%",
                    silhouette,
                    f"{float(row.avg_radius_m):.0f}",
                ]
            )
        story.append(_table(comparison_rows, styles, [3.0 * cm, 2.5 * cm, 2.8 * cm, 2.8 * cm, 2.8 * cm, 3.0 * cm]))
        story.append(Paragraph("Clustering comparison is indicative: silhouette values are computed on the haversine distance between stay points.", styles["caption"]))

    story.append(Paragraph("3. Transport mode analysis", styles["heading"]))
    if result.mode_summary is not None and not result.mode_summary.empty:
        mode_rows = [["Mode", "Journeys", "Distance (km)", "Duration (h)", "Distance share"]]
        for row in result.mode_summary.itertuples(index=False):
            mode_rows.append(
                [
                    row.label,
                    int(row.journeys),
                    f"{float(row.distance_km):.2f}",
                    f"{float(row.duration_h):.2f}",
                    f"{float(row.share_distance) * 100:.1f}%",
                ]
            )
        story.append(_table(mode_rows, styles, [4.2 * cm, 2.6 * cm, 3.2 * cm, 3.2 * cm, 3.4 * cm]))
        story.append(Paragraph("Mode labels are taken from the Timeline export where available. Estimates for unlabelled journeys are reported separately below.", styles["caption"]))
    else:
        story.append(Paragraph("No movement segments were available for transport analysis.", styles["body"]))

    mode_image = _mode_figure(result.mode_summary)
    if mode_image is not None:
        story.append(Image(mode_image, width=10.5 * cm, height=6.6 * cm))
        story.append(Paragraph("Transport mode share by distance.", styles["caption"]))

    story.append(Paragraph("4. Transport-mode estimation experiment", styles["heading"]))
    classifier = result.classifier
    metrics = classifier.metrics if classifier is not None else {}
    if metrics.get("trained"):
        story.append(
            Paragraph(
                f"A Random Forest classifier was trained on {metrics.get('training_segments', 0)} labelled movement segments "
                f"and evaluated on {metrics.get('test_segments', 0)} held-out segments. "
                f"Accuracy: <b>{metrics.get('accuracy', 0):.3f}</b>; macro F1: <b>{metrics.get('f1_macro', 0):.3f}</b>.",
                styles["body"],
            )
        )
        per_class = classifier.per_class_report()
        if not per_class.empty:
            rows = [["Mode", "Precision", "Recall", "F1", "Support"]]
            for row in per_class.itertuples(index=False):
                rows.append(
                    [
                        mode_label(row.mode),
                        f"{float(row.precision):.3f}",
                        f"{float(row.recall):.3f}",
                        f"{float(row.f1):.3f}",
                        int(row.support),
                    ]
                )
            story.append(_table(rows, styles, [4.0 * cm, 3.0 * cm, 3.0 * cm, 3.0 * cm, 3.0 * cm]))
            labels = metrics.get("confusion_labels", [])
            matrix = metrics.get("confusion_matrix", [])
            if labels and matrix:
                header = ["Actual \\ Predicted"] + [mode_label(str(label)) for label in labels]
                rows = [header]
                for label, values in zip(labels, matrix):
                    rows.append([mode_label(str(label))] + [int(value) for value in values])
                first_width = 3.2 * cm
                remaining = max(2.0 * cm, (17.0 * cm - first_width) / max(1, len(labels)))
                story.append(Spacer(1, 4))
                story.append(_table(rows, styles, [first_width] + [remaining] * len(labels)))
        story.append(
            Paragraph(
                "Mode estimation uses speed, distance, duration, straightness and time-of-day features. "
                "Overlapping speed profiles can make bus, car and rail trips ambiguous; low-confidence predictions are retained as Unknown.",
                styles["caption"],
            )
        )
    else:
        story.append(
            Paragraph(
                f"Transport-mode classification was not evaluated: {metrics.get('reason', 'insufficient labelled segments were available.')}",
                styles["body"],
            )
        )

    story.append(PageBreak())
    story.append(Paragraph("5. Geographical visualisations", styles["heading"]))
    route_image = _route_figure(result)
    if route_image is not None:
        story.append(Image(route_image, width=16.5 * cm, height=10.2 * cm))
        story.append(Paragraph("Recorded routes coloured by transport mode, with clustered places shown as markers.", styles["caption"]))
    heat_image = _heat_figure(result)
    if heat_image is not None:
        story.append(Image(heat_image, width=16.5 * cm, height=9.0 * cm))
        story.append(Paragraph("Travel-density heatmap computed from recorded location points.", styles["caption"]))
    if route_image is None and heat_image is None:
        story.append(Paragraph("Insufficient geospatial points were available to render map figures.", styles["body"]))

    story.append(Paragraph("6. Temporal travel patterns", styles["heading"]))
    daily_image = _daily_figure(result.daily)
    if daily_image is not None:
        story.append(Image(daily_image, width=16.5 * cm, height=5.8 * cm))
        story.append(Paragraph("Daily travel distance over the analysis period.", styles["caption"]))
    monthly_image = _monthly_figure(result.monthly)
    if monthly_image is not None:
        story.append(Image(monthly_image, width=16.5 * cm, height=5.0 * cm))
        story.append(Paragraph("Monthly travel distance.", styles["caption"]))
    if result.monthly is not None and not result.monthly.empty:
        rows = [["Month", "Distance (km)", "Duration (h)", "Journeys"]]
        for row in result.monthly.itertuples(index=False):
            rows.append([row.month, f"{float(row.distance_km):.2f}", f"{float(row.duration_min) / 60.0:.2f}", int(row.journeys)])
        story.append(_table(rows, styles, [4.2 * cm, 4.2 * cm, 4.2 * cm, 4.2 * cm]))

    if result.weekday_weekend is not None and not result.weekday_weekend.empty:
        rows = [["Period", "Distance (km)", "Duration (h)", "Journeys"]]
        for row in result.weekday_weekend.itertuples(index=False):
            rows.append([row.period, f"{float(row.distance_km):.2f}", f"{float(row.duration_min) / 60.0:.2f}", int(row.journeys)])
        story.append(Spacer(1, 6))
        story.append(_table(rows, styles, [4.2 * cm, 4.2 * cm, 4.2 * cm, 4.2 * cm]))
        story.append(Paragraph("Weekday and weekend travel comparison.", styles["caption"]))

    if result.repeated_routes is not None and not result.repeated_routes.empty:
        story.append(Paragraph("7. Frequently repeated journeys", styles["heading"]))
        rows = [["Route", "Mode", "Trips", "Avg distance (km)", "Avg duration (min)"]]
        for row in result.repeated_routes.head(10).itertuples(index=False):
            rows.append(
                [
                    row.route,
                    mode_label(row.mode),
                    int(row.trips),
                    f"{float(row.distance_km):.2f}",
                    f"{float(row.average_duration_min):.1f}",
                ]
            )
        story.append(_table(rows, styles, [7.0 * cm, 2.6 * cm, 1.6 * cm, 2.8 * cm, 2.8 * cm]))

    story.append(Paragraph("8. Methodology and limitations", styles["heading"]))
    story.append(
        Paragraph(
            "The pipeline parses the uploaded Timeline JSON, normalises timestamps to UTC, validates coordinates, orders records "
            "chronologically and derives stay and movement segments. Stay locations are grouped with DBSCAN or K-Means using "
            "haversine distance. Travel distances are computed from recorded route geometry where available, otherwise from "
            "consecutive location points after filtering implausible speed jumps. Transport modes use source labels when present "
            "and a Random Forest model for unlabelled segments when sufficient labelled data exists.",
            styles["body"],
        )
    )
    story.append(
        Paragraph(
            "Results depend on the completeness and accuracy of the source export. Timeline formats vary and may omit transport "
            "labels, route geometry or complete point sequences. Detected places are approximate clusters and should not be treated "
            "as confirmed home or workplace locations without additional evidence. GPS histories are sensitive personal data; "
            "this report was generated from a file processed within the application session.",
            styles["body"],
        )
    )

    document.build(story, onFirstPage=_footer, onLaterPages=_footer)
    payload = buffer.getvalue()
    buffer.close()
    return payload
