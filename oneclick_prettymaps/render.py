"""
Draw map layers fetched by `oneclick_prettymaps.fetch` as a poster.

Adds colouring modes that do not depend on a city having lots of parks or
water: buildings by type / height / size / distance from the centre, streets
by road class or compass orientation, plus effects (shadows, neon glow,
paper grain, gradient paper).

Inspired by and built on prettymaps by Marcelo Prates
(github.com/marceloprates/prettymaps, AGPL v3).

Copyright (C) 2026 TManhNguyen. Licensed under the GNU AGPL v3 (see LICENSE).
"""

from __future__ import annotations

import re
import warnings
from typing import Dict, Optional, Sequence

import geopandas as gp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
from matplotlib.colors import LinearSegmentedColormap, to_hex
from matplotlib.figure import Figure

from .fetch import STREET_WIDTHS, WATERWAY_WIDTHS
from .fonts import font_family
from .palettes import Theme, mix

CREDIT = (
    "Map data © OpenStreetMap contributors  ·  "
    "made with prettymaps by Marcelo Prates (github.com/marceloprates/prettymaps)"
)

LAYOUTS = {
    "poster": (8.0, 10.0),
    "square": (8.0, 8.0),
    "a4": (8.27, 11.69),
    "wallpaper": (6.0, 13.0),
}

# OSM landuse values -> which palette accent tints that area
LANDUSE_GROUPS = {
    "residential": 0,
    "commercial": 1, "retail": 1,
    "industrial": 2, "construction": 2, "railway": 2, "port": 2, "brownfield": 2,
    "education": 3, "institutional": 3, "religious": 3, "military": 3, "cemetery": 3,
}
LANDUSE_GREEN = {"farmland", "meadow", "farmyard", "orchard", "vineyard", "grass",
                 "village_green", "recreation_ground", "allotments", "greenfield", "aquaculture"}

BUILDING_MODES = ("auto", "type", "height", "size", "distance", "random")
STREET_MODES = ("type", "orientation", "solid")

# OSM tag values -> building category (checked in this order)
CATEGORIES = {
    "home": {
        "house", "residential", "apartments", "detached", "terrace", "semidetached_house",
        "dormitory", "bungalow", "farm", "cabin", "hut",
    },
    "shop & office": {
        "commercial", "retail", "office", "supermarket", "kiosk", "hotel", "mall",
        "marketplace", "restaurant", "cafe", "bank", "fuel",
    },
    "civic": {
        "public", "government", "civic", "school", "university", "college", "hospital",
        "kindergarten", "townhall", "library", "museum", "train_station", "transportation",
        "stadium", "sports_hall", "theatre", "clinic", "police", "fire_station",
    },
    "worship": {
        "church", "cathedral", "chapel", "mosque", "temple", "shrine", "synagogue",
        "pagoda", "monastery", "religious", "place_of_worship",
    },
    "industry": {
        "industrial", "warehouse", "factory", "manufacture", "garage", "garages",
        "hangar", "storage_tank", "construction", "service", "shed", "parking",
    },
}
_LOOKUP = {v: cat for cat, values in CATEGORIES.items() for v in values}
_CATEGORY_COLUMNS = ("building", "amenity", "shop", "office", "tourism", "landuse", "leisure")


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _project(gdf: Optional[gp.GeoDataFrame], crs) -> gp.GeoDataFrame:
    if gdf is None or gdf.empty:
        return gp.GeoDataFrame(geometry=[], crs=crs)
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    return gdf.to_crs(crs)


def _parts(gdf: gp.GeoDataFrame, kind: str) -> gp.GeoDataFrame:
    """Explode multi-geometries and keep only 'Polygon' or 'LineString' parts."""
    if gdf.empty:
        return gdf
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()]
    gdf = gdf.explode(index_parts=False)
    gdf = gdf.explode(index_parts=False)  # second pass for GeometryCollections
    return gdf[gdf.geom_type == kind].reset_index(drop=True)


def _first_match(value, table: dict):
    if isinstance(value, (list, tuple, np.ndarray)):
        for v in value:
            if v in table:
                return table[v]
        return np.nan
    return table.get(value, np.nan) if isinstance(value, str) else np.nan


def _offset(geoms: np.ndarray, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """Translate every geometry by its own (dx, dy)."""
    coords, idx = shapely.get_coordinates(geoms, return_index=True)
    coords = coords + np.column_stack([dx[idx], dy[idx]])
    return shapely.set_coordinates(np.array(geoms, dtype=object).copy(), coords)


def _buffer(gdf: gp.GeoDataFrame, distances, quad_segs: int = 8) -> gp.GeoSeries:
    """Buffer each geometry by its own distance (works on every geopandas version)."""
    geoms = shapely.buffer(gdf.geometry.values, np.asarray(distances, dtype=float), quad_segs=quad_segs)
    return gp.GeoSeries(geoms, crs=gdf.crs)


def _bearing(lines: gp.GeoSeries) -> np.ndarray:
    """Orientation of each street in degrees [0, 180)."""
    start = shapely.get_point(lines.values, 0)
    end = shapely.get_point(lines.values, -1)
    dx = shapely.get_x(end) - shapely.get_x(start)
    dy = shapely.get_y(end) - shapely.get_y(start)
    return np.degrees(np.arctan2(dy, dx)) % 180


def _to_number(value) -> float:
    if isinstance(value, (int, float)) and not pd.isna(value):
        return float(value)
    if isinstance(value, str):
        m = re.search(r"\d+(\.\d+)?", value)
        if m:
            return float(m.group())
    return np.nan


# ---------------------------------------------------------------------------
# Colouring
# ---------------------------------------------------------------------------


def _cmap(colors: Sequence[str], cyclic: bool = False) -> LinearSegmentedColormap:
    colors = list(colors)
    if len(colors) == 1:
        colors = colors * 2
    if cyclic:
        colors = colors + colors[:1]
    return LinearSegmentedColormap.from_list("palette", colors)


def building_category(row) -> Optional[str]:
    for col in _CATEGORY_COLUMNS:
        value = row.get(col)
        if isinstance(value, str) and value in _LOOKUP:
            return _LOOKUP[value]
    if isinstance(row.get("shop"), str) or isinstance(row.get("office"), str):
        return "shop & office"
    if isinstance(row.get("amenity"), str):
        return "civic"
    return None


def building_levels(gdf: gp.GeoDataFrame) -> pd.Series:
    levels = pd.Series(np.nan, index=gdf.index)
    if "building:levels" in gdf:
        levels = gdf["building:levels"].map(_to_number)
    if "height" in gdf:
        levels = levels.fillna(gdf["height"].map(_to_number) / 3.0)
    return levels.clip(lower=1, upper=80)


def building_colors(
    buildings: gp.GeoDataFrame,
    theme: Theme,
    mode: str,
    center: shapely.Point,
    rng: np.random.Generator,
) -> tuple[list, str, dict]:
    """
    Returns (colours, mode actually used, legend {label: colour}).

    'auto' uses building types when at least half of the buildings are
    tagged with a usable type, otherwise footprint size. Many cities (e.g.
    most of Vietnam) only tag buildings as 'yes', so size gives them variety.
    """
    accents = theme.buildings
    n = len(buildings)
    if n == 0:
        return [], mode, {}

    categories = buildings.apply(building_category, axis=1) if mode in ("auto", "type") else None
    if mode == "auto":
        mode = "type" if categories.notna().mean() >= 0.5 else "size"

    if mode == "height":
        levels = building_levels(buildings)
        if levels.notna().sum() < max(10, 0.1 * n):
            warnings.warn("Too few buildings have a height in OpenStreetMap here; using size instead.")
            mode = "size"
        else:
            values = levels.fillna(levels.median()).rank(pct=True)
            cmap = _cmap(accents)
            return [to_hex(cmap(v)) for v in values], mode, {}

    if mode == "type":
        names = list(CATEGORIES)
        if len(accents) >= len(names):
            colors_ = accents
        else:  # not enough palette colours: sample in-between shades
            cmap = _cmap(accents)
            colors_ = [to_hex(cmap(v)) for v in np.linspace(0, 1, len(names))]
        palette = dict(zip(names, colors_))
        unknown = mix(theme.background, theme.streets, 0.22)
        colors = [palette.get(c, unknown) for c in categories]
        present = set(categories.dropna())
        legend = {k: v for k, v in palette.items() if k in present}
        return colors, mode, legend

    if mode == "size":
        values = buildings.geometry.area.rank(pct=True)
        cmap = _cmap(accents)
        return [to_hex(cmap(v)) for v in values], mode, {}

    if mode == "distance":
        d = buildings.geometry.centroid.distance(center)
        values = d / max(d.max(), 1e-9)
        cmap = _cmap(accents)
        return [to_hex(cmap(v)) for v in values], mode, {}

    if mode == "random":
        return list(rng.choice(accents, size=n)), mode, {}

    raise ValueError(f"buildings must be one of {BUILDING_MODES}, got {mode!r}")


def street_colors(streets: gp.GeoDataFrame, theme: Theme, mode: str) -> list:
    if mode == "solid":
        return [theme.streets] * len(streets)
    if mode == "orientation":
        cmap = _cmap(theme.buildings if theme.dark else [theme.streets] + theme.buildings, cyclic=True)
        return [to_hex(cmap(b / 180)) for b in streets["bearing"]]
    if mode == "type":
        major = {"motorway", "trunk", "primary"}
        mid = {"secondary", "tertiary"}
        accent1 = theme.buildings[0]
        accent2 = theme.buildings[1 % len(theme.buildings)]
        if not theme.dark:
            # keep roads readable: tint the street colour rather than replace it
            accent1, accent2 = mix(theme.streets, accent1, 0.65), mix(theme.streets, accent2, 0.45)

        def pick(h):
            h = h if isinstance(h, (list, tuple)) else [h]
            if major.intersection(h):
                return accent1
            if mid.intersection(h):
                return accent2
            return theme.streets

        return [pick(h) for h in streets["highway"]]
    raise ValueError(f"streets must be one of {STREET_MODES}, got {mode!r}")


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def _draw(ax, gdf, color, zorder, ec="none", lw=0.0, alpha=1.0, hatch=None):
    if gdf is None or len(gdf) == 0:
        return
    gdf.plot(ax=ax, color=color, edgecolor=ec, linewidth=lw, alpha=alpha, zorder=zorder, hatch=hatch)


def _layout(fig: Figure, has_text: bool):
    """Map axes rectangle (figure fractions) and y of the text block."""
    W, H = fig.get_size_inches()
    margin = 0.07 * W
    size = W - 2 * margin
    text_band = 1.5 if has_text else 0.5
    if size + 2 * margin + text_band > H:  # e.g. square layout
        size = H - 2 * margin - text_band
    left = (W - size) / 2
    bottom = H - margin - size
    if not has_text:
        bottom = (H - size) / 2
    return [left / W, bottom / H, size / W, size / H], (bottom / 2) / H


def _paper(fig: Figure, theme: Theme, gradient: bool, grain: float, rng):
    back = fig.add_axes([0, 0, 1, 1], zorder=-10)
    back.set_axis_off()
    if gradient:
        top, bottom = theme.page, mix(theme.page, theme.streets if not theme.dark else theme.buildings[0], 0.10)
        grad = _cmap([top, bottom])(np.linspace(0, 1, 256))[:, None, :3]
        back.imshow(grad, aspect="auto", extent=(0, 1, 0, 1))
    else:
        back.set_facecolor(theme.page)
    if grain > 0:
        front = fig.add_axes([0, 0, 1, 1], zorder=10)
        front.set_axis_off()
        noise = rng.normal(0.5, 0.18, (700, 700)).clip(0, 1)
        front.imshow(noise, cmap="gray", alpha=grain, aspect="auto", interpolation="bilinear", extent=(0, 1, 0, 1))


def render(
    gdfs: Dict[str, gp.GeoDataFrame],
    theme: Theme | str | Sequence[str] = "saigon-sunset",
    buildings: str = "auto",
    streets: str = "type",
    dark: Optional[bool] = None,
    glow: Optional[bool] = None,
    shadows: bool = False,
    hatch: bool = False,
    outline: bool = True,
    gradient: bool = True,
    grain: float = 0.0,
    title: Optional[str] = None,
    subtitle: Optional[str] = "auto",
    legend: bool = False,
    landuse: float = 0.35,
    layout: str = "poster",
    figsize: Optional[tuple] = None,
    street_scale: float | str = "auto",
    seed: int = 0,
) -> Figure:
    """
    Draw a poster from fetched layers.

    Args:
        gdfs: output of `fetch()`.
        theme: Theme, palette name (see PALETTES), Color Hunt link or list of hex colours.
        buildings: 'auto' | 'type' | 'height' | 'size' | 'distance' | 'random'.
        streets: 'type' | 'orientation' (rainbow by compass direction) | 'solid'.
        dark: force a dark (True) or light (False) map. None decides from the palette.
        glow: neon glow under streets (default on for dark maps).
        shadows: drop shadows under buildings (taller buildings cast longer ones).
        hatch: prettymaps-style dotted hatch on parks and water.
        outline: thin outline around buildings, parks and water.
        gradient: subtle gradient on the paper.
        grain: paper grain strength, 0 (off) to ~0.15.
        title / subtitle: poster text. subtitle='auto' prints the coordinates.
        legend: show building categories (when coloured by type).
        landuse: strength (0-1) of the land-use colour patchwork (residential,
            commercial, industrial...). 0 turns it off. Needs fetch(landuse=True).
        layout: 'poster' | 'square' | 'a4' | 'wallpaper'.
        street_scale: multiply street widths; 'auto' widens them for big areas.
        seed: makes 'random' colouring and grain repeatable.
    """
    if not isinstance(theme, Theme):
        theme = Theme.from_palette(theme, dark=dark, glow=glow)
    elif glow is not None:
        theme = theme.with_(glow=glow)
    rng = np.random.default_rng(seed)

    perimeter_ll = gdfs["perimeter"]
    if perimeter_ll.crs is None:
        perimeter_ll = perimeter_ll.set_crs(4326)
    crs = perimeter_ll.to_crs(4326).estimate_utm_crs()
    perimeter = shapely.union_all(perimeter_ll.to_crs(crs).geometry.values)
    center = perimeter.centroid
    xmin, ymin, xmax, ymax = perimeter.bounds
    extent = max(xmax - xmin, ymax - ymin) / 2
    if street_scale == "auto":
        street_scale = float(np.clip((extent / 1000) ** 0.6, 0.7, 3.0))

    def layer(name, kind="Polygon"):
        return _parts(_project(gdfs.get(name), crs), kind)

    # --- figure -----------------------------------------------------------
    figsize = figsize or LAYOUTS[layout]
    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor(theme.page)
    has_text = bool(title) or bool(subtitle)
    rect, text_y = _layout(fig, has_text)
    _paper(fig, theme, gradient, grain, rng)
    ax = fig.add_axes(rect)
    ax.set_axis_off()
    ax.set_facecolor("none")
    pad = extent * 0.02
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_aspect("equal")

    edge = theme.outline if outline else "none"
    lw = 0.35 if outline else 0.0
    clip = gp.GeoSeries([perimeter], crs=crs)

    # --- land & nature ----------------------------------------------------
    _draw(ax, gp.GeoDataFrame(geometry=clip), theme.background, 0)
    zones = layer("landuse")
    if landuse > 0 and len(zones) and "landuse" in zones:
        accents = theme.buildings

        def tint(value):
            if value in LANDUSE_GREEN:
                return mix(theme.background, theme.green, landuse)
            group = LANDUSE_GROUPS.get(value)
            if group is None:
                return None
            return mix(theme.background, accents[group % len(accents)], landuse)

        zones["c"] = zones["landuse"].map(tint)
        zones = zones.dropna(subset=["c"]).clip(clip)
        _draw(ax, zones, zones["c"].tolist(), 0.5)
    nature = [
        ("parking", theme.parking, 1),
        ("green", theme.green, 2),
        ("forest", theme.forest, 3),
        ("beach", theme.beach, 3),
        ("rock", theme.rock, 3),
        ("water", theme.water, 6),
        ("sea", theme.water, 6),
    ]
    for name, color, z in nature:
        gdf = layer(name)
        _draw(ax, gdf, color, z, ec=edge, lw=lw)
        if hatch and name in ("green", "forest", "water", "sea") and len(gdf):
            hatch_color = mix(color, theme.streets, 0.18)
            _draw(ax, gdf, "none", z + 0.5, ec=hatch_color, lw=0, hatch="ooo...")

    waterway = layer("waterway", "LineString")
    if len(waterway):
        widths = waterway.get("waterway", pd.Series(index=waterway.index, dtype=object))
        waterway["w"] = widths.map(lambda v: _first_match(v, WATERWAY_WIDTHS))
        waterway = waterway.dropna(subset=["w"])
        waterway = gp.GeoDataFrame(geometry=_buffer(waterway, waterway["w"].values), crs=crs).clip(clip)
        _draw(ax, waterway, theme.water, 6, ec=edge, lw=lw)

    # --- streets ----------------------------------------------------------
    roads = layer("streets", "LineString")
    if len(roads) and "highway" in roads:
        roads["w"] = roads["highway"].map(lambda v: _first_match(v, STREET_WIDTHS)) * street_scale
        roads = roads.dropna(subset=["w"]).reset_index(drop=True)
        roads["bearing"] = _bearing(roads.geometry)
        colors = street_colors(roads, theme, streets)
        if theme.glow:
            for k, alpha in ((3.2, 0.06), (2.0, 0.14)):
                halo = gp.GeoDataFrame(
                    {"c": colors}, geometry=_buffer(roads, roads["w"].values * k, quad_segs=3), crs=crs
                ).clip(clip)
                _draw(ax, halo, halo["c"].tolist(), 7, alpha=alpha)
        polys = gp.GeoDataFrame(
            {"c": colors}, geometry=_buffer(roads, roads["w"].values, quad_segs=3), crs=crs
        ).clip(clip)
        _draw(ax, polys, polys["c"].tolist(), 8)

    # --- buildings --------------------------------------------------------
    blds = layer("building")
    legend_items, used = {}, None
    if len(blds):
        colors, used, legend_items = building_colors(blds, theme, buildings, center, rng)
        blds["c"] = colors
        if shadows:
            levels = building_levels(blds).fillna(2).values
            length = extent / 450 * (1 + 0.35 * np.minimum(levels, 25))
            sweep = blds.geometry.values
            for t in (0.33, 0.66, 1.0):
                sweep = shapely.union(sweep, _offset(blds.geometry.values, length * t, -length * t))
            shade = gp.GeoDataFrame(geometry=sweep, crs=crs).clip(clip)
            _draw(ax, shade, mix(theme.background, "#000000", 0.55 if theme.dark else 0.3), 9, alpha=0.55)
        _draw(ax, blds, blds["c"].tolist(), 10, ec=edge, lw=lw * 0.6)

    # --- outline of the map shape ------------------------------------------
    ax.plot(*_ring_xy(perimeter), color=theme.outline, lw=1.2, zorder=20)

    # --- text -------------------------------------------------------------
    text_color = theme.text
    display_font = font_family("display")
    mono_font = font_family("mono")
    W = figsize[0]
    if title:
        fig.text(
            0.5, text_y + 0.025, " ".join(title.upper()), ha="center", va="center",
            fontsize=W * 4.2, color=text_color, family=display_font, weight="bold",
        )
    if subtitle == "auto":
        c = perimeter_ll.to_crs(4326).geometry.union_all().centroid
        subtitle = f"{abs(c.y):.4f}° {'N' if c.y >= 0 else 'S'}  ·  {abs(c.x):.4f}° {'E' if c.x >= 0 else 'W'}"
    if subtitle:
        fig.text(
            0.5, text_y - (0.02 if title else 0), subtitle, ha="center", va="center",
            fontsize=W * 1.35, color=mix(text_color, theme.page, 0.25), family=mono_font,
        )
    fig.text(
        0.5, 0.012, CREDIT, ha="center", va="bottom", fontsize=W * 0.62,
        color=mix(text_color, theme.page, 0.45), family=mono_font,
    )
    if legend and legend_items:
        handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in legend_items.values()]
        fig.legend(
            handles, legend_items.keys(), loc="upper center", ncol=len(handles), frameon=False,
            labelcolor=text_color, prop={"family": mono_font, "size": W * 1.0},
            bbox_to_anchor=(0.5, rect[1] - 0.004),
        )

    fig._oneclick = {"buildings": used, "theme": theme}
    return fig


def _ring_xy(geom):
    geom = geom if geom.geom_type == "Polygon" else max(geom.geoms, key=lambda g: g.area)
    x, y = geom.exterior.xy
    return np.asarray(x), np.asarray(y)


def save(fig: Figure, path: str, dpi: int = 300) -> str:
    """Save PNG (dpi applies), SVG or PDF depending on the extension."""
    fig.savefig(path, dpi=dpi, facecolor=fig.get_facecolor())
    return path
