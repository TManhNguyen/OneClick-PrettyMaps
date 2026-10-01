"""Synthetic map layers shaped like `fetch()` output, so rendering can be tested offline."""

import geopandas as gp
import numpy as np
from shapely.geometry import LineString, Point, box

LAT, LON = 10.7769, 106.7009  # Ho Chi Minh City


def fake_city(radius=600, circle=True, tagged_share=0.7, seed=1):
    rng = np.random.default_rng(seed)
    utm = gp.GeoSeries([Point(LON, LAT)], crs=4326).estimate_utm_crs()
    cx, cy = gp.GeoSeries([Point(LON, LAT)], crs=4326).to_crs(utm).iloc[0].coords[0]
    perim = Point(cx, cy).buffer(radius) if circle else box(cx - radius, cy - radius, cx + radius, cy + radius)

    def to_ll(gdf):
        return gdf.set_crs(utm).to_crs(4326)

    # Slightly rotated street grid
    lines, kinds = [], []
    step = 90
    for i, off in enumerate(np.arange(-radius, radius + 1, step)):
        kind = "primary" if i % 5 == 0 else "secondary" if i % 3 == 0 else "residential"
        a = LineString([(cx - radius, cy + off), (cx + radius, cy + off + 40)])
        b = LineString([(cx + off, cy - radius), (cx + off + 25, cy + radius)])
        lines += [a, b]
        kinds += [kind, ["residential", "service"] if kind == "residential" else kind]
    streets = to_ll(gp.GeoDataFrame({"highway": kinds}, geometry=lines))

    # Buildings in the blocks
    values = ["house", "apartments", "retail", "school", "temple", "warehouse", "office"]
    geoms, tags, levels = [], [], []
    for x in np.arange(cx - radius, cx + radius, step):
        for y in np.arange(cy - radius, cy + radius, step):
            for _ in range(3):
                w, h = rng.uniform(12, 35, 2)
                ox, oy = rng.uniform(12, step - 40, 2)
                g = box(x + ox, y + oy, x + ox + w, y + oy + h)
                if perim.contains(g):
                    geoms.append(g)
                    tags.append(rng.choice(values) if rng.random() < tagged_share else "yes")
                    levels.append(str(rng.integers(1, 20)) if rng.random() < 0.5 else None)
    building = to_ll(gp.GeoDataFrame({"building": tags, "building:levels": levels}, geometry=geoms))

    green = to_ll(gp.GeoDataFrame({"leisure": ["park"]}, geometry=[Point(cx - 200, cy + 150).buffer(130)]))
    water = to_ll(gp.GeoDataFrame({"natural": ["water"]}, geometry=[Point(cx + 250, cy - 220).buffer(90)]))
    river = to_ll(gp.GeoDataFrame(
        {"waterway": ["river"]},
        geometry=[LineString([(cx - radius, cy - 300), (cx, cy - 330), (cx + radius, cy - 420)])],
    ))
    landuse = to_ll(gp.GeoDataFrame(
        {"landuse": ["residential", "commercial", "industrial", "farmland", "unknown_kind"]},
        geometry=[box(cx - radius, cy, cx, cy + radius), box(cx, cy, cx + radius, cy + radius),
                  box(cx - radius, cy - radius, cx, cy), box(cx, cy - radius, cx + radius, cy),
                  box(cx - 50, cy - 50, cx + 50, cy + 50)],
    ))
    empty = gp.GeoDataFrame(geometry=[])

    return {
        "perimeter": to_ll(gp.GeoDataFrame(geometry=[perim])),
        "streets": streets,
        "building": building,
        "green": green,
        "water": water,
        "waterway": river,
        "forest": empty,
        "rock": empty,
        "beach": empty,
        "parking": empty,
        "sea": gp.GeoDataFrame(geometry=[], crs=4326),
        "landuse": landuse,
    }
