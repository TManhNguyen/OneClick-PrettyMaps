"""fetch() wiring into prettymaps, with the OpenStreetMap request mocked out."""

import importlib

import geopandas as gp
import pytest
from shapely.geometry import box

from oneclick_prettymaps import fetch, polygon_query
from oneclick_prettymaps.fetch import _prettymaps

_prettymaps()  # installs stand-ins for vsketch / cv2 when missing
import prettymaps.fetch as pf  # noqa: E402

# the package re-exports the fetch() function under the same name as its module
fetch_module = importlib.import_module("oneclick_prettymaps.fetch")


@pytest.fixture
def captured(monkeypatch):
    calls = {}

    def fake_features(perimeter, layers):
        calls["perimeter"] = perimeter
        calls["layers"] = layers
        return {name: gp.GeoDataFrame(geometry=[]) for name in layers}

    monkeypatch.setattr(fetch_module, "fetch_features", fake_features)

    def fake_extras(perimeter, landuse=True):
        calls["extras_landuse"] = landuse
        out = {"waterway": gp.GeoDataFrame(geometry=[], crs=4326)}
        if landuse:
            out["landuse"] = gp.GeoDataFrame(geometry=[], crs=4326)
        return out

    monkeypatch.setattr(fetch_module, "fetch_extras", fake_extras)
    monkeypatch.setattr(fetch_module, "fetch_streets", lambda p: gp.GeoDataFrame(geometry=[], crs=4326))
    monkeypatch.setattr(fetch_module, "fetch_sea", lambda p: gp.GeoDataFrame(geometry=[], crs=4326))
    return calls


def _radius_m(perimeter):
    g = perimeter.to_crs(perimeter.estimate_utm_crs()).geometry.iloc[0]
    return (g.bounds[2] - g.bounds[0]) / 2


def test_point_circle(captured):
    gdfs = fetch((10.7769, 106.7009), radius=750, circle=True, sea=False)
    assert "sea" not in gdfs
    # rivers come from fetch_extras, not from prettymaps (which would
    # download the street network a second time); streets are fetched
    # separately so they can be previewed as their own stage
    assert not {"waterway", "streets", "sea", "perimeter"} & set(captured["layers"])
    assert {"waterway", "streets", "building"} <= set(gdfs)
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


def test_progress_stages(captured):
    seen = []

    class Recorder:
        def __call__(self, stage, gdfs, seconds, error=None):
            seen.append((stage, sorted(gdfs)))

        def finish(self, seconds):
            seen.append(("finish", seconds >= 0))

    fetch((10.7769, 106.7009), radius=300, sea=True, progress=Recorder())
    stages = [s for s, _ in seen]
    assert stages == ["outline", "features", "rivers", "streets", "sea", "finish"]
    assert seen[0][1] == ["perimeter"]          # outline drawn before any download
    assert "building" in seen[1][1] and "streets" not in seen[1][1]


def test_fetch_streets_clips_to_map(monkeypatch):
    import osmnx as ox
    from shapely.geometry import LineString, box as bbox

    edges = gp.GeoDataFrame(
        {"highway": ["primary", "residential"]},
        geometry=[LineString([(106.70, 10.775), (106.73, 10.775)]),   # leaves the map
                  LineString([(106.80, 10.90), (106.81, 10.91)])],    # fully outside
        crs=4326,
    )
    monkeypatch.setattr(ox, "graph_from_polygon", lambda polygon, **kw: "graph")
    monkeypatch.setattr(ox, "graph_to_gdfs", lambda graph, nodes=False: edges)
    perimeter = gp.GeoDataFrame(geometry=[bbox(106.69, 10.76, 106.72, 10.79)], crs=4326)
    out = fetch_module.fetch_streets(perimeter)
    assert out["highway"].tolist() == ["primary"]
    assert out.geometry.iloc[0].bounds[2] <= 106.72 + 1e-9


def test_fetch_sea_keeps_side_without_roads(monkeypatch):
    import osmnx as ox
    from shapely.geometry import LineString, box as bbox

    coast = gp.GeoDataFrame(geometry=[LineString([(106.69, 10.775), (106.72, 10.775)])], crs=4326)
    roads = gp.GeoDataFrame({"bridge": [None]}, geometry=[LineString([(106.70, 10.78), (106.71, 10.785)])], crs=4326)
    monkeypatch.setattr(ox.features, "features_from_polygon", lambda polygon, tags: coast)
    monkeypatch.setattr(ox, "graph_from_polygon", lambda polygon, **kw: "graph")
    monkeypatch.setattr(ox, "graph_to_gdfs", lambda graph, nodes=False: roads)
    perimeter = gp.GeoDataFrame(geometry=[bbox(106.69, 10.76, 106.72, 10.79)], crs=4326)
    sea = fetch_module.fetch_sea(perimeter).geometry.union_all()
    assert sea.bounds[3] <= 10.775 + 1e-6      # only the southern half (no roads) is sea


def test_fetch_sea_raises_download_errors(monkeypatch):
    import osmnx as ox
    from shapely.geometry import box as bbox

    def fail(*a, **k):
        raise ConnectionError("offline")

    monkeypatch.setattr(ox.features, "features_from_polygon", fail)
    perimeter = gp.GeoDataFrame(geometry=[bbox(106.69, 10.76, 106.72, 10.79)], crs=4326)
    with pytest.raises(ConnectionError):
        fetch_module.fetch_sea(perimeter)


def test_download_retries_backup_server_then_reports(monkeypatch):
    import osmnx as ox

    tried = []

    def busy():
        tried.append(ox.settings.overpass_url)
        raise ConnectionError("429 Too Many Requests")

    original = ox.settings.overpass_url
    result, error = fetch_module._download(busy, lambda: "empty")
    assert result == "empty" and "429" in error
    assert tried == [original, *fetch_module.BACKUP_OVERPASS]
    assert ox.settings.overpass_url == original          # restored

    tried.clear()

    def flaky():
        tried.append(ox.settings.overpass_url)
        if len(tried) == 1:
            raise TimeoutError("timed out")
        return "data"

    assert fetch_module._download(flaky, lambda: None) == ("data", None)


def test_download_nothing_mapped_is_not_an_error():
    import osmnx as ox

    def none_here():
        raise ox._errors.InsufficientResponseError("No matching features")

    assert fetch_module._download(none_here, lambda: "empty") == ("empty", None)


def test_partial_failure_is_reported_to_progress(captured, monkeypatch):
    from shapely.geometry import LineString

    def busy(p):
        raise ConnectionError("504 Gateway Timeout")

    roads = gp.GeoDataFrame({"highway": ["primary"]}, geometry=[LineString([(106.700, 10.776), (106.702, 10.777)])], crs=4326)
    monkeypatch.setattr(fetch_module, "fetch_features", lambda p, layers: {k: gp.GeoDataFrame(geometry=[]) for k in layers})
    monkeypatch.setattr(fetch_module, "fetch_extras", lambda p, landuse=True: (_ for _ in ()).throw(ConnectionError("504")))
    monkeypatch.setattr(fetch_module, "fetch_streets", lambda p: roads)
    seen = {}
    with pytest.warns(UserWarning, match="rivers: download failed"):
        gdfs = fetch((10.7769, 106.7009), radius=300, sea=False,
                     progress=lambda stage, g, t, error=None: seen.__setitem__(stage, error))
    assert seen["rivers"] and "504" in seen["rivers"]
    assert seen["streets"] is None and len(gdfs["streets"]) == 1


def test_everything_failing_raises_clear_error(captured, monkeypatch):
    def busy(*a, **k):
        raise ConnectionError("429 Too Many Requests")

    monkeypatch.setattr(fetch_module, "fetch_features", busy)
    monkeypatch.setattr(fetch_module, "fetch_extras", busy)
    monkeypatch.setattr(fetch_module, "fetch_streets", busy)
    with pytest.warns(UserWarning), pytest.raises(fetch_module.DownloadError, match="wait a minute"):
        fetch((10.7769, 106.7009), radius=300, sea=False)


def test_fetch_features_splits_layers_by_tags(monkeypatch):
    import osmnx as ox
    from shapely.geometry import box as bbox

    feats = gp.GeoDataFrame(
        {"building": ["yes", None, None], "leisure": [None, "park", None], "natural": [None, None, "water"]},
        geometry=[bbox(106.700, 10.770, 106.701, 10.771), bbox(106.702, 10.772, 106.704, 10.774),
                  bbox(106.705, 10.775, 106.706, 10.776)],
        crs=4326,
    )
    monkeypatch.setattr(ox.features, "features_from_polygon", lambda polygon, tags: feats)
    perimeter = gp.GeoDataFrame(geometry=[bbox(106.69, 10.76, 106.72, 10.79)], crs=4326)
    layers = {k: v for k, v in fetch_module.LAYERS.items() if k in ("building", "green", "water", "forest")}
    out = fetch_module.fetch_features(perimeter, layers)
    assert (len(out["building"]), len(out["green"]), len(out["water"]), len(out["forest"])) == (1, 1, 1, 0)


def _fake_perimeter(monkeypatch, geometry):
    seen = {}

    def get_perimeter(query, **kwargs):
        seen.update(kwargs, query=query)
        return gp.GeoDataFrame(geometry=[geometry], crs=4326)

    monkeypatch.setattr(pf, "get_perimeter", get_perimeter)
    return seen


def test_object_only_uses_place_boundary(captured, monkeypatch):
    from shapely.geometry import Polygon

    ward = Polygon([(106.69, 10.73), (106.71, 10.73), (106.72, 10.75), (106.70, 10.76)])
    seen = _fake_perimeter(monkeypatch, ward)
    gdfs = fetch("Phường Tân Hưng, Hồ Chí Minh", radius=None)
    assert seen["radius"] is None and seen["by_osmid"] is False
    assert gdfs["perimeter"].geometry.iloc[0].equals(ward)


def test_object_only_accepts_osm_id(captured, monkeypatch):
    from shapely.geometry import box as bbox

    seen = _fake_perimeter(monkeypatch, bbox(106.69, 10.73, 106.71, 10.75))
    fetch(" r1234567 ", radius=None)
    assert seen["query"] == "R1234567" and seen["by_osmid"] is True


def test_object_only_rejects_point_results(captured, monkeypatch):
    from shapely.geometry import Point

    _fake_perimeter(monkeypatch, Point(106.7, 10.74))
    with pytest.raises(ValueError, match="single point"):
        fetch("Some café", radius=None)


def test_fit_area_covers_whole_boundary():
    from shapely.geometry import Polygon

    ward = gp.GeoDataFrame(
        geometry=[Polygon([(106.69, 10.73), (106.71, 10.73), (106.72, 10.75), (106.70, 10.76)])], crs=4326
    )
    fit = fetch_module.fit_area(ward, margin=0.02)
    lat, lon = fit["center"]
    assert 106.69 < lon < 106.72 and 10.73 < lat < 10.76
    utm = ward.estimate_utm_crs()
    shape = ward.to_crs(utm).geometry.iloc[0]
    centre = gp.GeoSeries([gp.points_from_xy([lon], [lat])[0]], crs=4326).to_crs(utm).iloc[0]
    assert centre.buffer(fit["circle_radius"]).contains(shape)          # circle holds the whole ward
    assert fit["square_half"] * 2 >= max(shape.bounds[2] - shape.bounds[0], shape.bounds[3] - shape.bounds[1])


def test_find_place_returns_boundary_or_none(monkeypatch):
    import osmnx as ox
    from shapely.geometry import Point, box as bbox

    results = {
        "Phường Tân Phong": gp.GeoDataFrame({"display_name": ["Phường Tân Phong"]}, geometry=[bbox(106.69, 10.72, 106.71, 10.74)], crs=4326),
        "Some café": gp.GeoDataFrame(geometry=[Point(106.7, 10.73)], crs=4326),
    }
    seen = {}

    def geocode_to_gdf(q, by_osmid=False):
        seen[q] = by_osmid
        if q not in results:
            raise ValueError("not found")
        return results[q]

    monkeypatch.setattr(ox.geocoder, "geocode_to_gdf", geocode_to_gdf)
    assert fetch_module.find_place(" Phường Tân Phong ") is not None
    assert fetch_module.find_place("Some café") is None      # a point, not an area
    assert fetch_module.find_place("nowhere") is None
    fetch_module.find_place("r42")
    assert seen["R42"] is True
