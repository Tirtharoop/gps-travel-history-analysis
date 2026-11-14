# Intelligent GPS-Based Travel History Analysis

M.Tech mini project implementation: an end-to-end framework that converts a Google Maps Timeline JSON export into travel analytics, interactive maps, machine-learning experiments and a downloadable PDF report.

The application combines geospatial data mining, clustering, transport-mode estimation and interactive visualisation in a single Streamlit dashboard.

## Features

- Parses modern `semanticSegments` exports, Google Takeout `timelineObjects`, classic `locations` files and `rawSignals` position records.
- Normalises timestamps, validates coordinates, removes duplicate points and derives stay/movement segments.
- Detects frequently visited places with DBSCAN or K-Means using haversine distance.
- Estimates visit counts, total and average stay duration, first/last visit and cluster radius.
- Generates travel-density, visit-frequency and stay-duration heatmaps.
- Visualises routes on an interactive Folium map with transport-mode colours and place markers.
- Summarises transport modes by journeys, distance, duration and share.
- Trains and evaluates a Random Forest model for unlabelled movement segments, including accuracy, macro F1, per-class scores, confusion matrix and feature importance.
- Produces daily, weekly, monthly, hourly and weekday/weekend statistics plus repeated-journey detection.
- Generates a presentation-ready PDF report with tables, charts and map figures.
- Includes a synthetic demo dataset, so the dashboard can be explored without personal data.

## Project structure

```text
app.py                         Streamlit dashboard
gps_travel/
  clustering.py                DBSCAN / K-Means place detection and comparison
  charts.py                    Plotly charts
  config.py                    Thresholds, heatmap layers, palette
  geocoding.py                 Optional cached Nominatim reverse geocoding
  geo.py                       Haversine distance and coordinate parsing
  mapping.py                   Folium route maps and heatmaps
  models.py                    Canonical TravelData model and filters
  modes.py                     Transport-mode normalisation and colours
  parser.py                    Multi-format Timeline parser
  pipeline.py                  End-to-end analysis orchestration
  report.py                    ReportLab PDF generation
  sample_data.py               Synthetic semantic/classic Timeline exports
  segmentation.py              Stay and movement segmentation
  statistics.py                Temporal and route statistics
  transport.py                 Feature extraction and Random Forest classifier
tests/                         Pytest suite
.streamlit/config.toml         Deployment theme and runtime settings
DEPLOYMENT.md                  Streamlit Community Cloud deployment guide
requirements.txt               Runtime dependencies
requirements-dev.txt           Testing and lint dependencies
```

## Installation

Python 3.11 or later is required. Install runtime dependencies with:

```powershell
python -m pip install -r requirements.txt
```

For tests and linting, install the development dependencies instead:

```powershell
python -m pip install -r requirements-dev.txt
```

## Running the dashboard

```powershell
python -m streamlit run app.py
```

Open the URL printed by Streamlit. Use **Load demo dataset** to explore the full workflow, or upload a Google Maps Timeline JSON file.

## Deployment

The project includes a runtime configuration in `.streamlit/config.toml` and a step-by-step [Streamlit Community Cloud deployment guide](DEPLOYMENT.md). Push the repository to GitHub, create a new app at <https://share.streamlit.io>, and set the main file path to `app.py`. No secrets are required.

## Supported input formats

### Modern semantic Timeline

```json
{
  "semanticSegments": [
    {
      "startTime": "2026-01-01T08:00:00+00:00",
      "endTime": "2026-01-01T08:30:00+00:00",
      "timelinePath": [{"point": "geo:12.9,77.5", "durationMinutesOffsetFromStartTime": "0"}],
      "activity": {
        "start": {"latLng": "geo:12.9,77.5"},
        "end": {"latLng": "geo:12.91,77.51"},
        "distanceMeters": 1500,
        "topCandidate": {"type": "WALKING", "probability": 0.9}
      }
    }
  ]
}
```

### Classic Location History

```json
{
  "locations": [
    {
      "latitudeE7": 129716000,
      "longitudeE7": 775946000,
      "timestamp": "2026-01-01T08:00:00Z",
      "accuracy": 10,
      "activity": [{"type": "IN_PASSENGER_VEHICLE", "confidence": 90}]
    }
  ]
}
```

The parser also accepts `timelineObjects` from semantic monthly Takeout files and `rawSignals` position records. Missing fields are handled gracefully; the dashboard reports the detected schema and any warnings.

## Analysis workflow

1. **Parsing** - detect the schema, extract locations, visits, activities and route geometry.
2. **Preprocessing** - convert timestamps to UTC, validate coordinates, sort records and remove duplicates.
3. **Segmentation** - use explicit visit/activity segments when available; otherwise derive stays and movements from consecutive points with configurable radius and minimum stay time.
4. **Clustering** - group stay locations with DBSCAN or K-Means. Isolated DBSCAN points are reported as noise.
5. **Transport analysis** - use source labels where present. Train a Random Forest on labelled movements and estimate unlabelled movements when at least 20 labelled segments and two mode classes are available. Low-confidence predictions remain `unknown`.
6. **Visualisation** - interactive route maps, place markers and weighted heatmaps.
7. **Statistics** - daily, weekly, monthly, hourly, weekday/weekend and repeated-route summaries.
8. **Reporting** - generate a multi-section PDF with summary tables, charts and map figures.

## Dashboard sections

- **Overview** - headline KPIs, daily/monthly distance, weekday and hourly patterns, data-quality details.
- **Frequently visited places** - ranked table, place map, clustering comparison and optional reverse geocoding.
- **Heatmap** - switch between travel density, visit frequency and stay duration.
- **Transport analysis** - mode shares, model metrics, confusion matrix, feature importance and unlabelled journey estimates.
- **Routes and report** - date/mode-filtered route map, repeated journeys, weekly statistics and PDF generation.

## Tests

```powershell
python -m pytest
python -m ruff check app.py gps_travel tests
```

The test suite covers coordinate parsing, all supported Timeline formats, segmentation, clustering, statistics, transport-mode training, map/chart rendering and PDF generation. Ruff may report two intentional broad exception handlers at the Streamlit UI boundary.

## Demo data

```python
from gps_travel.sample_data import generate_classic_timeline, generate_sample_timeline

semantic = generate_sample_timeline(days=45, seed=7)
classic = generate_classic_timeline(days=30, seed=11)
```

The generated data includes home/work/lunch/leisure trips, multiple transport modes, route geometry and a small proportion of unlabelled movements for the classification experiment.

## Research-level evaluation

The dashboard exposes two quantitative comparisons:

- **Place clustering** - DBSCAN versus K-Means using cluster count, noise ratio, silhouette score and mean cluster radius.
- **Transport-mode classification** - Random Forest accuracy, macro F1, per-class precision/recall/F1 and a confusion matrix on a held-out split.

These experiments are only meaningful when the source export contains reliable transport labels or when clusters are manually validated.

## Privacy and limitations

- GPS histories are sensitive personal data. Files are processed in the application session; no external upload occurs unless reverse geocoding is explicitly enabled.
- Reverse geocoding sends place coordinates to OpenStreetMap Nominatim and respects a local cache in `data/geocode_cache.json`.
- Timeline formats vary. Not every export contains transport labels, route geometry or complete point sequences.
- Distances from consecutive points are estimates and can be affected by GPS noise and sampling gaps. Implausible speed jumps are filtered.
- Clustered places are candidate locations, not confirmed home or workplace addresses.
- GPS speed alone cannot reliably distinguish bus, car and rail. Predictions therefore include confidence and an `unknown` class.
