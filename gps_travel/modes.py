from __future__ import annotations

MODE_COLORS = {
    "walking": "#2e7d32",
    "running": "#66bb6a",
    "cycling": "#1565c0",
    "motorcycle": "#8e24aa",
    "car": "#c62828",
    "bus": "#ef6c00",
    "train": "#6a1b9a",
    "metro": "#00838f",
    "tram": "#00acc1",
    "flight": "#ad1457",
    "boat": "#0277bd",
    "skiing": "#5d4037",
    "unknown": "#757575",
    "other": "#9e9e9e",
}

MODE_LABELS = {
    "walking": "Walking",
    "running": "Running",
    "cycling": "Cycling",
    "motorcycle": "Motorcycle",
    "car": "Car",
    "bus": "Bus",
    "train": "Train",
    "metro": "Metro",
    "tram": "Tram",
    "flight": "Flight",
    "boat": "Boat",
    "skiing": "Skiing",
    "unknown": "Unknown",
    "other": "Other",
}

MODE_ORDER = [
    "walking",
    "running",
    "cycling",
    "motorcycle",
    "car",
    "bus",
    "train",
    "metro",
    "tram",
    "flight",
    "boat",
    "skiing",
    "unknown",
    "other",
]

_MODE_ALIASES = {
    "walking": "walking",
    "walk": "walking",
    "on_foot": "walking",
    "on foot": "walking",
    "foot": "walking",
    "pedestrian": "walking",
    "hiking": "walking",
    "running": "running",
    "run": "running",
    "jogging": "running",
    "cycling": "cycling",
    "bicycling": "cycling",
    "bicycle": "cycling",
    "bike": "cycling",
    "on_bicycle": "cycling",
    "in_vehicle": "car",
    "in vehicle": "car",
    "in_passenger_vehicle": "car",
    "in a passenger vehicle": "car",
    "passenger vehicle": "car",
    "driving": "car",
    "drive": "car",
    "car": "car",
    "automobile": "car",
    "taxi": "car",
    "in_taxi": "car",
    "in_4_wheeler": "car",
    "4 wheeler": "car",
    "motorcycle": "motorcycle",
    "motorcycling": "motorcycle",
    "in_2_wheeler": "motorcycle",
    "2 wheeler": "motorcycle",
    "scooter": "motorcycle",
    "bus": "bus",
    "in_bus": "bus",
    "public_transport": "bus",
    "train": "train",
    "in_train": "train",
    "in_rail": "train",
    "rail": "train",
    "railway": "train",
    "commuter_train": "train",
    "metro": "metro",
    "subway": "metro",
    "in_subway": "metro",
    "underground": "metro",
    "tram": "tram",
    "in_tram": "tram",
    "light_rail": "tram",
    "flight": "flight",
    "flying": "flight",
    "airplane": "flight",
    "plane": "flight",
    "in_airplane": "flight",
    "boat": "boat",
    "sailing": "boat",
    "in_ferry": "boat",
    "ferry": "boat",
    "skiing": "skiing",
    "ski": "skiing",
    "unknown": "unknown",
    "unknow": "unknown",
    "unconfirmed": "unknown",
    "other": "other",
    "still": "unknown",
    "stationary": "unknown",
    "none": "unknown",
    "": "unknown",
}


def normalize_mode(value: object) -> str:
    if value is None:
        return "unknown"
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if text in _MODE_ALIASES:
        return _MODE_ALIASES[text]
    compact = text.replace("_", " ")
    if compact in _MODE_ALIASES:
        return _MODE_ALIASES[compact]
    if any(token in compact for token in ("walk", "foot", "hike")):
        return "walking"
    if any(token in compact for token in ("bicy", "bike", "cycle")):
        return "cycling"
    if any(token in compact for token in ("subway", "metro", "underground")):
        return "metro"
    if "tram" in compact:
        return "tram"
    if any(token in compact for token in ("train", "rail")):
        return "train"
    if "bus" in compact:
        return "bus"
    if any(token in compact for token in ("flight", "fly", "plane")):
        return "flight"
    if any(token in compact for token in ("ferry", "boat", "sail")):
        return "boat"
    if any(token in compact for token in ("motor", "scooter")):
        return "motorcycle"
    if any(token in compact for token in ("car", "vehicle", "taxi", "drive")):
        return "car"
    return "unknown"


def mode_color(mode: str) -> str:
    return MODE_COLORS.get(normalize_mode(mode), MODE_COLORS["unknown"])


def mode_label(mode: str) -> str:
    normalized = normalize_mode(mode)
    return MODE_LABELS.get(normalized, normalized.title())
