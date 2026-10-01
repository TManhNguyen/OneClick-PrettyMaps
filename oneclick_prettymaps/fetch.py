"""
Download OpenStreetMap layers using prettymaps (github.com/marceloprates/prettymaps,
(C) Marcelo Prates, AGPL v3).

Fetching is kept separate from drawing so a map can be restyled many times
without downloading it again.

Copyright (C) 2026 TManhNguyen. Licensed under the GNU AGPL v3 (see LICENSE).
"""

from __future__ import annotations

import importlib
import re
import sys
import time
import types
from copy import deepcopy
from typing import Callable, Dict, Optional, Tuple, Union

import geopandas as gp
from shapely.geometry import Point, Polygon, box, shape
from shapely.ops import unary_union

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


def _prettymaps():
    """
    Import prettymaps' downloader module (prettymaps.fetch).

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
    import prettymaps.fetch

    return prettymaps.fetch


def find_place(name: str) -> Optional[gp.GeoDataFrame]:
    """
    Official boundary of a named place (ward, district, park...), or None
    when the name is not found or only matches a point. Also accepts an
    OpenStreetMap id such as "R1234567".
    """
    import osmnx as ox

    name = name.strip()
    by_osmid = re.fullmatch(r"[NWR]\d+", name, re.I) is not None
    try:
        gdf = ox.geocoder.geocode_to_gdf(name.upper() if by_osmid else name, by_osmid=by_osmid)
    except Exception:
        return None
    if gdf.empty or not gdf.geom_type.isin(["Polygon", "MultiPolygon"]).all():
        return None
    return gdf.reset_index(drop=True)


def fit_area(boundary: gp.GeoDataFrame, margin: float = 0.02) -> dict:
    """
    Centre and sizes that frame a boundary: the centre of its extent, the
    circle radius and the square half-side (metres) that contain all of it,
    plus `margin` of its width on each side.
    """
    utm = boundary.to_crs(4326).estimate_utm_crs()
    shape = boundary.to_crs(utm).geometry.union_all()
    xmin, ymin, xmax, ymax = shape.bounds
    centre = Point((xmin + xmax) / 2, (ymin + ymax) / 2)
    grow = 1 + 2 * margin
    lonlat = gp.GeoSeries([centre], crs=utm).to_crs(4326).iloc[0]
    return {
        "center": (lonlat.y, lonlat.x),
        "circle_radius": centre.hausdorff_distance(shape) * grow,
        "square_half": max(xmax - xmin, ymax - ymin) / 2 * grow,
    }


def polygon_query(geojson: dict) -> gp.GeoDataFrame:
    """Turn a GeoJSON polygon (e.g. drawn on the map picker) into a prettymaps query."""
    geom = shape(geojson.get("geometry", geojson))
    if not isinstance(geom, Polygon):
        raise ValueError("Draw a polygon or rectangle")
    return gp.GeoDataFrame(geometry=[geom], crs="EPSG:4326")


Progress = Callable[[str, Dict[str, gp.GeoDataFrame], float], None]


def fetch(
    query: Query,
    radius: float | None = 1000,
    circle: bool = True,
    sea: bool = True,
    landuse: bool = True,
    progress: Optional[Progress] = None,
    logging: bool = False,
) -> Dict[str, gp.GeoDataFrame]:
    """
    Download map layers, in stages.

    Args:
        query: address / place name, (lat, lon) tuple, or a GeoDataFrame polygon.
        radius: metres around the point. None ("this object only") cuts the
            map out along the place's own boundary, e.g. "Phường Tân Hưng,
            Hồ Chí Minh" or an OSM id like "R1234567" (or the polygon given
            as query); everything outside it is left out.
        circle: round map (True) or square (False). Ignored without a radius.
        sea: compute the sea polygon (slower, only useful near coasts).
        landuse: also fetch land-use areas (residential, commercial, ...),
            drawn as a soft colour patchwork where few buildings are mapped.
        progress: called as progress(stage, layers_so_far, seconds) after each
            stage ("outline", "features", "rivers", "streets", "sea"), e.g.
            `LivePreview()` to draw the map while it downloads.

    Returns:
        dict of layer name -> GeoDataFrame (EPSG:4326).
    """
    pf = _prettymaps()
    start = time.time()

    def done(stage):
        if progress is not None:
            progress(stage, gdfs, time.time() - start)

    layers = deepcopy(LAYERS)
    # prettymaps fetches "waterway" as a road-style network without a filter,
    # which downloads the whole street network a second time. Rivers are
    # fetched with a small feature query instead (fetch_extras).
    layers.pop("waterway")
    for kwargs in layers.values():
        kwargs.setdefault("circle", circle)
        kwargs.setdefault("dilate", None)

    if isinstance(query, gp.GeoDataFrame):
        query = query.reset_index(drop=True).to_crs(4326)
        radius = None
    if not radius:
        radius = None

    # Same steps as prettymaps.fetch.get_gdfs, split so each can be shown
    by_osmid = isinstance(query, str) and re.fullmatch(r"[NWR]\d+", query.strip(), re.I) is not None
    if by_osmid:
        query = query.strip().upper()
    perimeter = pf.get_perimeter(
        query, radius=radius, circle=circle, dilate=None, rotation=0, by_osmid=by_osmid
    )
    if radius is None and not perimeter.geom_type.isin(["Polygon", "MultiPolygon"]).all():
        raise ValueError(
            f"{query!r} was found as a single point, not an area with a boundary. "
            "Try a more specific name (e.g. 'Phường Tân Hưng, Hồ Chí Minh, Việt Nam') "
            "or its OpenStreetMap relation id, e.g. 'R1234567'."
        )
    gdfs: Dict[str, gp.GeoDataFrame] = {"perimeter": perimeter}
    done("outline")

    features = {k: v for k, v in layers.items() if k not in ("perimeter", "streets", "sea")}
    gdfs.update(pf.unified_osm_request(perimeter, features, logging=logging))
    done("features")

    gdfs.update(fetch_extras(perimeter, landuse=landuse))
    done("rivers")

    gdfs["streets"] = fetch_streets(perimeter)
    done("streets")

    if sea:
        gdfs["sea"] = fetch_sea(perimeter)
        done("sea")

    if hasattr(progress, "finish"):
        progress.finish(time.time() - start)
    return gdfs


def _clip(gdf: gp.GeoDataFrame, area) -> gp.GeoDataFrame:
    gdf = gdf.copy()
    gdf.geometry = gdf.geometry.intersection(area)
    return gdf[~gdf.geometry.is_empty]


def fetch_streets(perimeter: gp.GeoDataFrame) -> gp.GeoDataFrame:
    """Street network edges inside the perimeter (as prettymaps does it)."""
    import osmnx as ox

    area = perimeter.to_crs(4326).geometry.union_all()
    try:
        graph = ox.graph_from_polygon(box(*area.bounds), truncate_by_edge=True)
        return _clip(ox.graph_to_gdfs(graph, nodes=False), area)
    except Exception:
        return gp.GeoDataFrame(geometry=[], crs=4326)


def fetch_sea(perimeter: gp.GeoDataFrame) -> gp.GeoDataFrame:
    """
    Sea polygon: the parts of the map cut off by the coastline that no
    (non-bridge) road crosses. Same method as prettymaps.
    """
    import osmnx as ox

    area = perimeter.to_crs(4326).geometry.union_all()
    bbox = box(*area.bounds)
    try:
        coast = ox.features.features_from_polygon(bbox, tags={"natural": "coastline"})
        candidates = bbox.difference(unary_union(coast.geometry.tolist()).buffer(1e-9))
        drive = ox.graph_to_gdfs(ox.graph_from_polygon(bbox, network_type="drive"), nodes=False)
    except Exception:
        return gp.GeoDataFrame(geometry=[], crs=4326)

    def is_sea(candidate):
        crossing = drive[drive.geometry.intersects(candidate)]
        if "bridge" in crossing:
            crossing = crossing[crossing["bridge"] != "yes"]
        return crossing.empty

    parts = getattr(candidates, "geoms", [candidates])
    sea = unary_union([c for c in parts if is_sea(c)])
    if sea.is_empty:
        return gp.GeoDataFrame(geometry=[], crs=4326)
    return _clip(gp.GeoDataFrame(geometry=[sea.buffer(1e-8)], crs=4326), area)


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
        gdf = features[features[column].notna() & features.geom_type.isin(kinds)][[column, "geometry"]]
        return _clip(gdf, area).reset_index(drop=True)

    out = {"waterway": pick("waterway", ["LineString", "MultiLineString"])}
    if landuse:
        out["landuse"] = pick("landuse", ["Polygon", "MultiPolygon"])
    return out
