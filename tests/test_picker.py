"""MapPicker logic with the place lookup mocked (no network, no browser)."""

import importlib

import geopandas as gp
import pytest
from shapely.geometry import Point, Polygon

pytest.importorskip("ipyleaflet")
picker_module = importlib.import_module("oneclick_prettymaps.picker")

WARD = gp.GeoDataFrame(
    {"display_name": ["Phường Tân Phong, Hồ Chí Minh"]},
    geometry=[Polygon([(106.695, 10.725), (106.715, 10.722), (106.722, 10.738), (106.700, 10.745)])],
    crs=4326,
)


@pytest.fixture
def picker(monkeypatch):
    monkeypatch.setattr(picker_module, "find_place", lambda name: WARD if "Tân Phong" in name else None)
    return picker_module.MapPicker((10.7769, 106.7009), radius=1000)


def _contains(lat, lon, radius_m, shape):
    utm = WARD.estimate_utm_crs()
    centre = gp.GeoSeries([Point(lon, lat)], crs=4326).to_crs(utm).iloc[0]
    return centre.buffer(radius_m).contains(gp.GeoSeries([shape], crs=4326).to_crs(utm).iloc[0])


def test_find_centres_on_boundary_and_fits_radius(picker):
    assert picker.find("Phường Tân Phong")
    lat, lon = picker.center
    assert picker.marker.location == [lat, lon] or tuple(picker.marker.location) == (lat, lon)
    # centre of the ward's extent (computed in metres, so allow ~10 m), not the label point
    x0, y0, x1, y1 = WARD.total_bounds
    assert abs(lon - (x0 + x1) / 2) < 1e-4 and abs(lat - (y0 + y1) / 2) < 1e-4
    assert _contains(lat, lon, picker.radius.value, WARD.geometry.iloc[0])
    assert picker.outline.data["features"]


def test_boundary_only_selection(picker):
    picker.find("Phường Tân Phong")
    picker.shape.value = "Boundary only"
    sel = picker.selection()
    assert sel["radius"] is None and sel["query"].geometry.iloc[0].equals(WARD.geometry.iloc[0])
    assert picker.radius.disabled


def test_square_refits_and_moving_pin_keeps_radius(picker):
    picker.find("Phường Tân Phong")
    circle_r = picker.radius.value
    picker.shape.value = "Square"
    assert picker.radius.value <= circle_r          # half-side needs less than the circle radius
    picker._move((10.80, 106.65))                    # user moves the pin elsewhere
    r = picker.radius.value
    picker.shape.value = "Circle"
    assert picker.radius.value == r                  # no longer fitted to the ward


def test_boundary_only_without_place_explains(picker):
    picker.shape.value = "Boundary only"
    with pytest.raises(ValueError, match="Find box"):
        picker.selection()
