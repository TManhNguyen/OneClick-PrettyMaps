"""
Download OpenStreetMap layers using prettymaps (github.com/marceloprates/prettymaps,
(C) Marcelo Prates, AGPL v3).

Fetching is kept separate from drawing so a map can be restyled many times
without downloading it again.

Copyright (C) 2026 TManhNguyen. Licensed under the GNU AGPL v3 (see LICENSE).
"""

from __future__ import annotations

import importlib
import sys
import types
from copy import deepcopy
from typing import Dict, Tuple, Union

import geopandas as gp
from shapely.geometry import Polygon, shape

# Street widths (metres, buffer radius) - same as prettymaps' default preset
STREET_WIDTHS = {
    "motorway": 5,
    "trunk": 5,
    "primary": 4.5,
    "secondary": 4,
    "tertiary": 3.5,
    "cycleway": 3.5,
    "residential": 3,
    "service": 2,
    "unclassified": 2,
    "pedestrian": 2,
    "footway": 1,
}
WATERWAY_WIDTHS = {"river": 20, "stream": 10}

# Layers mirror prettymaps' default preset
LAYERS = {
    "perimeter": {},
    "streets": {},
    "waterway": {"tags": {"waterway": ["river", "stream"]}},
    "building": {"tags": {"building": True, "landuse": "construction"}},
    "water": {"tags": {"natural": ["water", "bay"]}},
    "sea": {},
    "forest": {"tags": {"landuse": "forest"}},
    "green": {
        "tags": {
            "landuse": ["grass", "orchard"],
            "natural": ["island", "wood", "wetland"],
            "leisure": [
                "dog_park",
                "disc_golf_course",
                "garden",
                "golf_course",
                "park",
                "pitch",
                "sports_centre",
                "track",
            ],
        }
    },
    "rock": {"tags": {"natural": "bare_rock"}},
    "beach": {"tags": {"natural": "beach"}},
    "parking": {"tags": {"amenity": "parking", "highway": "pedestrian", "man_made": "pier"}},
}

Query = Union[str, Tuple[float, float], gp.GeoDataFrame]


def _prettymaps_get_gdfs():
    """
    Import prettymaps' downloader.

    prettymaps is installed without its dependency list (see README), which
    would otherwise upgrade ipykernel and break Colab, and pull ~400 MB of
    pen-plotter packages (vsketch -> PySide6). vsketch (plotter mode) and
    cv2 (hillshade) are imported by prettymaps but never used by this
    package, so an empty stand-in is enough when they are missing.
    """
    for name in ("vsketch", "cv2"):
        try:
            importlib.import_module(name)
        except ImportError:
            sys.modules[name] = types.ModuleType(name)
    from prettymaps.fetch import get_gdfs

    return get_gdfs


def polygon_query(geojson: dict) -> gp.GeoDataFrame:
    """Turn a GeoJSON polygon (e.g. drawn on the map picker) into a prettymaps query."""
    geom = shape(geojson.get("geometry", geojson))
    if not isinstance(geom, Polygon):
        raise ValueError("Draw a polygon or rectangle")
    return gp.GeoDataFrame(geometry=[geom], crs="EPSG:4326")


def fetch(
    query: Query,
    radius: float | None = 1000,
    circle: bool = True,
    sea: bool = True,
    landuse: bool = True,
    logging: bool = False,
) -> Dict[str, gp.GeoDataFrame]:
    """
    Download map layers.

    Args:
        query: address / place name, (lat, lon) tuple, or a GeoDataFrame polygon.
        radius: metres around the point. None plots the place's own boundary
            (or the polygon given as query).
        circle: round map (True) or square (False). Ignored without a radius.
        sea: compute the sea polygon (slower, only useful near coasts).
        landuse: also fetch land-use areas (residential, commercial, ...),
            drawn as a soft colour patchwork where few buildings are mapped.

    Returns:
        dict of layer name -> GeoDataFrame (EPSG:4326).
    """
    get_gdfs = _prettymaps_get_gdfs()

    layers = deepcopy(LAYERS)
    # prettymaps fetches "waterway" as a road-style network without a filter,
    # which downloads the whole street network a second time. Rivers are
    # fetched below with a small feature query instead.
    layers.pop("waterway")
    if not sea:
        layers.pop("sea")
    for kwargs in layers.values():
        kwargs.setdefault("circle", circle)
        kwargs.setdefault("dilate", None)

    if isinstance(query, gp.GeoDataFrame):
        query = query.reset_index(drop=True).to_crs(4326)
        radius = None
    if not radius:
        radius = None

    gdfs = get_gdfs(query, layers, radius, None, 0, logging=logging)
    gdfs.update(fetch_extras(gdfs["perimeter"], landuse=landuse))
    return gdfs


def fetch_extras(perimeter: gp.GeoDataFrame, landuse: bool = True) -> Dict[str, gp.GeoDataFrame]:
    """
    Rivers/streams (and land-use areas) in one small request.

    Returns {"waterway": ..., "landuse": ...}; layers are empty (never an
    error) when nothing is mapped or the download fails.
    """
    import osmnx as ox

    area = perimeter.to_crs(4326).geometry.union_all()
    tags = {"waterway": list(WATERWAY_WIDTHS)}
    if landuse:
        tags["landuse"] = True
    try:
        features = ox.features.features_from_polygon(area, tags=tags)
    except Exception:
        features = None

    def pick(column, kinds):
        if features is None or features.empty or column not in features:
            return gp.GeoDataFrame({column: []}, geometry=[], crs=4326)
        gdf = features[features[column].notna() & features.geom_type.isin(kinds)][[column, "geometry"]].copy()
        gdf.geometry = gdf.geometry.intersection(area)
        return gdf[~gdf.geometry.is_empty].reset_index(drop=True)

    out = {"waterway": pick("waterway", ["LineString", "MultiLineString"])}
    if landuse:
        out["landuse"] = pick("landuse", ["Polygon", "MultiPolygon"])
    return out
