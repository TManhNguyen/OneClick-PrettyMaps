"""fetch() wiring into prettymaps, with the OpenStreetMap request mocked out."""

import importlib

import geopandas as gp
import pytest
from shapely.geometry import box

from oneclick_prettymaps import fetch, polygon_query
from oneclick_prettymaps.fetch import _prettymaps_get_gdfs

_prettymaps_get_gdfs()  # installs stand-ins for vsketch / cv2 when missing
import prettymaps.fetch as pf  # noqa: E402

# the package re-exports the fetch() function under the same name as its module
fetch_module = importlib.import_module("oneclick_prettymaps.fetch")


@pytest.fixture
def captured(monkeypatch):
    calls = {}

    def fake_request(perimeter, layers_dict, logging=False):
        calls["perimeter"] = perimeter
        calls["layers"] = layers_dict
        return {name: gp.GeoDataFrame(geometry=[]) for name in layers_dict}

    monkeypatch.setattr(pf, "unified_osm_request", fake_request)

    def fake_extras(perimeter, landuse=True):
        calls["extras_landuse"] = landuse
        out = {"waterway": gp.GeoDataFrame(geometry=[], crs=4326)}
        if landuse:
            out["landuse"] = gp.GeoDataFrame(geometry=[], crs=4326)
        return out

    monkeypatch.setattr(fetch_module, "fetch_extras", fake_extras)
    return calls


def _radius_m(perimeter):
    g = perimeter.to_crs(perimeter.estimate_utm_crs()).geometry.iloc[0]
    return (g.bounds[2] - g.bounds[0]) / 2


def test_point_circle(captured):
    gdfs = fetch((10.7769, 106.7009), radius=750, circle=True, sea=False)
    assert "sea" not in captured["layers"]
    # rivers come from fetch_extras, not from prettymaps (which would
    # download the street network a second time)
    assert "waterway" not in captured["layers"]
    assert "waterway" in gdfs
    assert abs(_radius_m(gdfs["perimeter"]) - 750) < 5
    g = gdfs["perimeter"].to_crs(gdfs["perimeter"].estimate_utm_crs()).geometry.iloc[0]
    assert len(g.exterior.coords) > 10  # round, not square


def test_point_square(captured):
    gdfs = fetch((10.7769, 106.7009), radius=500, circle=False)
    g = gdfs["perimeter"].to_crs(gdfs["perimeter"].estimate_utm_crs()).geometry.iloc[0]
    assert len(g.exterior.coords) == 5


def test_drawn_polygon(captured):
    shape = {"type": "Feature", "geometry": box(106.70, 10.77, 106.71, 10.78).__geo_interface__}
    gdfs = fetch(polygon_query(shape), radius=1000)
    drawn = box(106.70, 10.77, 106.71, 10.78)
    # prettymaps reprojects the shape, so allow floating point noise
    assert gdfs["perimeter"].geometry.iloc[0].symmetric_difference(drawn).area < 1e-9


def test_polygon_query_rejects_points():
    with pytest.raises(ValueError):
        polygon_query({"type": "Point", "coordinates": [106.7, 10.77]})


def test_landuse_layer_optional(captured):
    assert "landuse" in fetch((10.7769, 106.7009), radius=300)
    assert "landuse" not in fetch((10.7769, 106.7009), radius=300, landuse=False)


def test_fetch_extras_splits_rivers_and_landuse(monkeypatch):
    import osmnx as ox
    from shapely.geometry import LineString, box as bbox

    features = gp.GeoDataFrame(
        {"waterway": ["river", None, "stream"], "landuse": [None, "residential", None]},
        geometry=[LineString([(106.70, 10.775), (106.71, 10.776)]), bbox(106.700, 10.770, 106.705, 10.775),
                  LineString([(106.80, 10.90), (106.81, 10.91)])],  # last one is outside the map
        crs=4326,
    )
    seen = {}

    def fake_features(polygon, tags):
        seen["tags"] = tags
        return features

    monkeypatch.setattr(ox.features, "features_from_polygon", fake_features)
    perimeter = gp.GeoDataFrame(geometry=[bbox(106.69, 10.76, 106.72, 10.79)], crs=4326)
    out = fetch_module.fetch_extras(perimeter)
    assert set(seen["tags"]) == {"waterway", "landuse"}
    assert out["waterway"]["waterway"].tolist() == ["river"]
    assert out["landuse"]["landuse"].tolist() == ["residential"]


def test_fetch_extras_empty_area(monkeypatch):
    import osmnx as ox
    from shapely.geometry import box as bbox

    def no_data(polygon, tags):
        raise ox._errors.InsufficientResponseError("nothing here")

    monkeypatch.setattr(ox.features, "features_from_polygon", no_data)
    perimeter = gp.GeoDataFrame(geometry=[bbox(106.69, 10.76, 106.72, 10.79)], crs=4326)
    out = fetch_module.fetch_extras(perimeter, landuse=False)
    assert list(out) == ["waterway"] and out["waterway"].empty
