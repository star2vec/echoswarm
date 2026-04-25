"""
src/satellite/local.py — Copernicus EMS flood polygon loader.

Loads pre-processed flood extent vectors from the Copernicus Emergency
Management Service activation EMSR773 (Valencia DANA, October 2024).
No API calls, no rasterio, no authentication required.

Data note: the EMS file uses 'event_type' / 'notation' properties, not
'obj_type'. Within this file every feature is a flood feature, so the
meaningful filter is 'notation' == 'Flooded area' (standing water that
blocks routing) vs. 'Flood trace' (high-water mark — where water was,
not where it is now; excluded so reset roads stay passable).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Union

from shapely.geometry import MultiPolygon, Polygon, shape

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_FLOOD_FILE = _PROJECT_ROOT / "data" / "valencia_flood_peak.json"


def get_flooded_sectors(
    source: str = "local",
    path: str | Path | None = None,
) -> list[Union[Polygon, MultiPolygon]]:
    """
    Return Shapely geometries for currently flooded areas.

    Each returned polygon is ready to pass directly to inject_flood().
    Coordinates are WGS-84 (lon, lat) — Shapely's (x, y) convention.

    Args:
        source: 'local'  — load Copernicus EMS EMSR773 vectors from disk.
                'live'   — Phase-2 placeholder; raises until cdse.py is implemented.

    Returns:
        List of Shapely Polygon / MultiPolygon objects (1 117 features at flood peak).

    Raises:
        NotImplementedError: if source != 'local'.
        FileNotFoundError:   if the EMS data file is missing from /data.
    """
    if source != "local":
        raise NotImplementedError(
            "source='live' is the Phase-2 CDSE integration. "
            "Implement src/satellite/cdse.py once credentials are confirmed."
        )

    if path is not None:
        flood_file = Path(path)
        if not flood_file.is_absolute():
            flood_file = _PROJECT_ROOT / flood_file
    else:
        flood_file = _FLOOD_FILE

    if not flood_file.exists():
        raise FileNotFoundError(
            f"EMS flood data not found: {flood_file}\n"
            "Download EMSR773 vectors from emergency.copernicus.eu and place "
            "the GeoJSON in the /data directory."
        )

    with flood_file.open(encoding="utf-8") as f:
        collection = json.load(f)

    polygons: list[Union[Polygon, MultiPolygon]] = []
    n_traces = 0

    for feature in collection["features"]:
        props = feature.get("properties", {})

        # 'Flood trace' = historical high-water mark, not standing water.
        # Injecting these into the routing graph would block roads that have
        # already been cleared — only 'Flooded area' features are returned.
        if props.get("notation") != "Flooded area":
            n_traces += 1
            continue

        geometry = feature.get("geometry")
        if geometry is None:
            continue

        geom = shape(geometry)
        # Self-intersecting rings (10 features in EMSR773) break contains() checks.
        # buffer(0) is the standard Shapely repair for this — no geometry is lost.
        polygons.append(geom if geom.is_valid else geom.buffer(0))

    logger.info(
        "EMS EMSR773: %d flood-area polygons loaded (%d flood-trace features excluded)",
        len(polygons),
        n_traces,
    )
    return polygons
