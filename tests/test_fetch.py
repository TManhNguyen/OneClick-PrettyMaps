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
    monkeypatch.setattr(
        fetch_module, "fetch_landuse", lambda perimeter: gp.GeoDataFrame(geometry=[], crs=4326)
    )
    return calls


def _radius_m(perimeter):
    g = perimeter.to_crs(perimeter.estimate_utm_crs()).geometry.iloc[0]
    return (g.bounds[2] - g.bounds[0]) / 2


def test_point_circle(captured):
    gdfs = fetch((10.7769, 106.7009), radius=750, circle=True, sea=False)
    assert "sea" not in captured["layers"]
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
