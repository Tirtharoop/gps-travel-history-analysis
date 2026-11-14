from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
from geopy.exc import GeopyError

DEFAULT_USER_AGENT = "gps-travel-analytics-mtech-mini-project"


def reverse_geocode_places(
    places: pd.DataFrame,
    user_agent: str = DEFAULT_USER_AGENT,
    cache_path: str | Path | None = None,
    max_places: int = 15,
    timeout: int = 10,
    delay_seconds: float = 1.0,
) -> pd.DataFrame:
    from geopy.geocoders import Nominatim

    if places is None or places.empty:
        return places
    result = places.copy()
    cache: dict[str, str] = {}
    path = Path(cache_path) if cache_path is not None else None
    if path is not None and path.exists():
        try:
            cache = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            cache = {}

    geolocator = Nominatim(user_agent=user_agent, timeout=timeout)
    addresses: list[str] = []
    for index, row in result.iterrows():
        if index >= max_places:
            addresses.append("")
            continue
        lat = float(row["lat"])
        lon = float(row["lon"])
        key = f"{lat:.5f},{lon:.5f}"
        if key in cache:
            addresses.append(cache[key])
            continue
        address = ""
        try:
            location = geolocator.reverse((lat, lon), exactly_one=True, language="en")
            if location is not None:
                address = str(location.address)
        except (GeopyError, ValueError, TypeError):
            address = ""
        cache[key] = address
        addresses.append(address)
        time.sleep(max(0.0, delay_seconds))

    result["address"] = addresses
    if path is not None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass
    return result
