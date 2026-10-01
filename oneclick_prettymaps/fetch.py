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
import warnings
from copy import deepcopy
from typing import Callable, Dict, Optional, Tuple, Union

import geopandas as gp
import pandas as pd
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


# Tried in order when the default OpenStreetMap (Overpass) server fails
BACKUP_OVERPASS = ("https://overpass.kumi.systems/api",)

Progress = Callable[[str, Dict[str, gp.GeoDataFrame], float, Optional[str]], None]


class DownloadError(RuntimeError):
    """No map data could be downloaded."""


def _empty(*columns) -> gp.GeoDataFrame:
    return gp.GeoDataFrame({c: [] for c in columns}, geometry=[], crs=4326)


def _nothing_mapped(error: Exception) -> bool:
    """osmnx's way of saying the area simply has no such features."""
    import osmnx as ox

    return isinstance(error, ox._errors.InsufficientResponseError) or (
        isinstance(error, ValueError) and "no graph nodes" in str(error).lower()
    )


def _download(fn: Callable, empty: Callable):
    """
    Run a download. Returns (result, error message or None).

    "Nothing mapped here" gives an empty result. Any other failure (server
    busy, rate limited, timeout...) is retried once on a backup Overpass
    server, and reported instead of silently returning nothing.
    """
    import osmnx as ox

    original = ox.settings.overpass_url
    error = None
    try:
        for url in (original, *BACKUP_OVERPASS):
            ox.settings.overpass_url = url
            try:
                return fn(), None
            except Exception as e:  # noqa: BLE001 - reported to the user below
                if _nothing_mapped(e):
                    return empty(), None
                error = e
    finally:
        ox.settings.overpass_url = original
    return empty(), f"{type(error).__name__}: {error}"


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
        progress: called as progress(stage, layers_so_far, seconds, error)
            after each stage ("outline", "features", "rivers", "streets",
            "sea"); error is None or a message. E.g. `LivePreview()` to draw
            the map while it downloads.

    Returns:
        dict of layer name -> GeoDataFrame (EPSG:4326).

    Raises:
        DownloadError: when every download failed (e.g. servers busy).
    """
    pf = _prettymaps()
    start = time.time()
    errors: Dict[str, str] = {}

    def done(stage):
        if progress is not None:
            progress(stage, gdfs, time.time() - start, errors.get(stage))

    def run(stage, fn, empty):
        result, error = _download(fn, empty)
        if error:
            errors[stage] = error
            warnings.warn(f"{stage}: download failed ({error})")
        return result

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
    gdfs.update(run("features", lambda: fetch_features(perimeter, features), lambda: {k: _empty() for k in features}))
    done("features")

    extras_empty = lambda: {"waterway": _empty("waterway"), **({"landuse": _empty("landuse")} if landuse else {})}  # noqa: E731
    gdfs.update(run("rivers", lambda: fetch_extras(perimeter, landuse=landuse), extras_empty))
    done("rivers")

    gdfs["streets"] = run("streets", lambda: fetch_streets(perimeter), _empty)
    done("streets")

    if sea:
        gdfs["sea"] = run("sea", lambda: fetch_sea(perimeter), _empty)
        done("sea")

    if hasattr(progress, "finish"):
        progress.finish(time.time() - start)
    content = [k for k in gdfs if k != "perimeter"]
    if errors and all(gdfs[k].empty for k in content):
        detail = "\n".join(f"  - {stage}: {msg}" for stage, msg in errors.items())
        raise DownloadError(
            "Could not download any map data from OpenStreetMap:\n" + detail +
            "\nThe servers may be busy or rate-limiting: wait a minute and run this step again, "
            "or choose a smaller area."
        )
    return gdfs


def _clip(gdf: gp.GeoDataFrame, area) -> gp.GeoDataFrame:
    gdf = gdf.copy()
    gdf.geometry = gdf.geometry.intersection(area)
    return gdf[~gdf.geometry.is_empty]


def _area(perimeter: gp.GeoDataFrame):
    return perimeter.to_crs(4326).geometry.union_all()


def fetch_features(perimeter: gp.GeoDataFrame, layers: dict) -> Dict[str, gp.GeoDataFrame]:
    """
    Buildings, parks, water... in one request, split into layers by their
    tags (prettymaps' method). Unlike prettymaps' version, download errors
    are raised instead of returning empty layers.
    """
    import osmnx as ox

    area = _area(perimeter)
    feats = ox.features.features_from_polygon(area, tags=_prettymaps().merge_tags(layers))
    out = {}
    for layer, kwargs in layers.items():
        parts = []
        for key, value in kwargs.get("tags", {}).items():
            if key not in feats:
                continue
            column = feats[key]
            if value is True:
                parts.append(feats[column.notna()])
            elif isinstance(value, list):
                parts.append(feats[column.isin(value)])
            else:
                parts.append(feats[column == value])
        gdf = pd.concat(parts) if parts else _empty()
        out[layer] = _clip(gdf[~gdf.index.duplicated()], area)
    return out


def fetch_streets(perimeter: gp.GeoDataFrame) -> gp.GeoDataFrame:
    """Street network edges inside the perimeter (as prettymaps does it)."""
    import osmnx as ox

    area = _area(perimeter)
    graph = ox.graph_from_polygon(area, truncate_by_edge=True)
    return _clip(ox.graph_to_gdfs(graph, nodes=False), area)


def fetch_sea(perimeter: gp.GeoDataFrame) -> gp.GeoDataFrame:
    """
    Sea polygon: the parts of the map cut off by the coastline that no
    (non-bridge) road crosses. Same method as prettymaps.
    """
    import osmnx as ox

    area = _area(perimeter)
    bbox = box(*area.bounds)
    coast = ox.features.features_from_polygon(bbox, tags={"natural": "coastline"})
    candidates = bbox.difference(unary_union(coast.geometry.tolist()).buffer(1e-9))
    drive = ox.graph_to_gdfs(ox.graph_from_polygon(bbox, network_type="drive"), nodes=False)

    def is_sea(candidate):
        crossing = drive[drive.geometry.intersects(candidate)]
        if "bridge" in crossing:
            crossing = crossing[crossing["bridge"] != "yes"]
        return crossing.empty

    parts = getattr(candidates, "geoms", [candidates])
    sea = unary_union([c for c in parts if is_sea(c)])
    if sea.is_empty:
        return _empty()
    return _clip(gp.GeoDataFrame(geometry=[sea.buffer(1e-8)], crs=4326), area)


def fetch_extras(perimeter: gp.GeoDataFrame, landuse: bool = True) -> Dict[str, gp.GeoDataFrame]:
    """
    Rivers/streams (and land-use areas) in one small request.

    Returns {"waterway": ..., "landuse": ...}; layers are empty when nothing
    is mapped. Download errors are raised.
    """
    import osmnx as ox

    area = _area(perimeter)
    tags = {"waterway": list(WATERWAY_WIDTHS)}
    if landuse:
        tags["landuse"] = True
    try:
        features = ox.features.features_from_polygon(area, tags=tags)
    except ox._errors.InsufficientResponseError:  # nothing mapped here
        features = None

    def pick(column, kinds):
        if features is None or features.empty or column not in features:
            return _empty(column)
        gdf = features[features[column].notna() & features.geom_type.isin(kinds)][[column, "geometry"]]
        return _clip(gdf, area).reset_index(drop=True)

    out = {"waterway": pick("waterway", ["LineString", "MultiLineString"])}
    if landuse:
        out["landuse"] = pick("landuse", ["Polygon", "MultiPolygon"])
    return out
