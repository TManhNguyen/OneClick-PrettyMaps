"""
Interactive map to choose the area to plot (ipyleaflet).

- Find a place by name (box above the map, or the 🔍 on the map). When it
  has an official boundary (ward, district, park...) the boundary is
  outlined, the map is centred on it and the radius is set to cover all of
  it plus a small border.
- Or click anywhere to move the centre (the pin is draggable).
- Pick Circle / Square and adjust the radius with the slider, or
  "Boundary only" to keep just the found place (neighbours removed).
- Or choose "Drawn shape" and draw any polygon / rectangle on the map.

In Google Colab, run `google.colab.output.enable_custom_widget_manager()`
first (the notebook does this for you).

Copyright (C) 2026 TManhNguyen. Licensed under the GNU AGPL v3 (see LICENSE).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import geopandas as gp

from .fetch import find_place, fit_area, polygon_query

NOMINATIM = "https://nominatim.openstreetmap.org/search?format=json&q={s}"
# Background map tiles. CartoDB now needs an API key, and OpenStreetMap's own
# servers block tiles requested from Colab's sandboxed widget frame, so the
# default is Esri's keyless street map. Pass `tiles=` to use another source.
TILES = {
    "Streets": (
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
        "Tiles &copy; Esri &mdash; Esri, HERE, Garmin, &copy; OpenStreetMap contributors",
    ),
    "Satellite": (
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "Tiles &copy; Esri &mdash; Source: Esri, Maxar, Earthstar Geographics",
    ),
}


def fit_zoom(lat: float, radius: float, half_view_px: int = 220) -> int:
    """Web-map zoom level at which a circle of `radius` metres fits in the view."""
    metres_per_px = radius / half_view_px
    zoom = math.log2(156_543.03 * math.cos(math.radians(lat)) / metres_per_px)
    return int(max(3, min(18, math.floor(zoom))))


def _square_bounds(lat: float, lon: float, radius: float):
    dlat = radius / 111_320
    dlon = radius / (111_320 * max(math.cos(math.radians(lat)), 1e-6))
    return ((lat - dlat, lon - dlon), (lat + dlat, lon + dlon))


class MapPicker:
    """
    picker = MapPicker()                  # starts on Ho Chi Minh City; or MapPicker("Hoi An")
    picker.show()
    ...
    gdfs = fetch(**picker.selection())
    """

    SHAPES = ("Circle", "Square", "Boundary only", "Drawn shape")

    def __init__(
        self,
        center: str | Tuple[float, float] = (10.7769, 106.7009),
        radius: int = 1000,
        shape: str = "Circle",
        height: str = "520px",
        tiles: str | None = None,
        margin: float = 0.02,
    ):
        import ipywidgets as w
        from ipyleaflet import (
            Circle, DrawControl, GeoJSON, LayersControl, Map, Marker, Rectangle, SearchControl, TileLayer,
        )

        self.margin = margin
        self.polygon: Optional[gp.GeoDataFrame] = None   # drawn shape
        self.boundary: Optional[gp.GeoDataFrame] = None  # found place
        self.place_name = ""
        self._fit: dict = {}
        start_name = center if isinstance(center, str) else ""
        if start_name:
            boundary = find_place(start_name)
            if boundary is not None:
                center = fit_area(boundary, margin)["center"]
            else:
                import osmnx as ox

                center = ox.geocoder.geocode(start_name)
        self.center: Tuple[float, float] = tuple(center)

        self.map = Map(
            center=self.center,
            zoom=fit_zoom(self.center[0], radius),
            scroll_wheel_zoom=True,
            basemap=TileLayer(
                url=tiles or TILES["Streets"][0], max_zoom=19, name="Streets", detect_retina=True,
                attribution="Custom tiles" if tiles else TILES["Streets"][1],
            ),
            layout=w.Layout(width="100%", height=height),
        )
        if not tiles:
            url, attribution = TILES["Satellite"]
            self.map.add(TileLayer(
                url=url, attribution=attribution, name="Satellite", base=True, visible=False, max_zoom=19,
                detect_retina=True,
            ))
            self.map.add(LayersControl(position="topright"))
        style = dict(color="#E76F51", fill_color="#E76F51", fill_opacity=0.12, weight=2)
        self.marker = Marker(location=self.center, draggable=True, title="Map centre")
        self.circle = Circle(location=self.center, radius=radius, **style)
        self.square = Rectangle(bounds=_square_bounds(*self.center, radius), **style)
        self.outline = GeoJSON(
            data={"type": "FeatureCollection", "features": []},
            style={"color": "#264653", "weight": 3, "fillOpacity": 0.05, "dashArray": "6 4"},
        )
        self.map.add(self.outline)
        self.map.add(self.marker)

        search = SearchControl(position="topleft", url=NOMINATIM, zoom=15, marker=Marker(visible=False))
        search.on_location_found(self._on_search)
        self.map.add(search)

        self.draw = DrawControl(
            polygon={"shapeOptions": {"color": "#264653", "fillOpacity": 0.15}},
            rectangle={"shapeOptions": {"color": "#264653", "fillOpacity": 0.15}},
            circlemarker={},
            polyline={},
            circle={},
            marker={},
        )
        self.draw.on_draw(self._on_draw)

        self.radius = w.IntSlider(
            value=radius, min=100, max=8000, step=50, description="Radius (m)",
            continuous_update=True, style={"description_width": "initial"},
            layout=w.Layout(width="420px"),
        )
        self.shape = w.ToggleButtons(options=self.SHAPES, value=shape, description="Shape")
        self.info = w.HTML()
        self.query = w.Text(
            value=start_name, placeholder="Find a place, e.g. Phường Tân Phong, Hồ Chí Minh",
            layout=w.Layout(width="420px"), continuous_update=False,  # changes on Enter
        )
        find = w.Button(description="Find", icon="search", layout=w.Layout(width="120px"))
        find.on_click(lambda _: self.find(self.query.value))
        self.query.observe(lambda ch: self.find(ch["new"]), "value")
        recentre = w.Button(description="Recentre", icon="crosshairs", layout=w.Layout(width="120px"))
        recentre.on_click(lambda _: self.recentre())

        self.map.on_interaction(self._on_click)
        self.marker.observe(lambda ch: self._move(ch["new"], from_marker=True), "location")
        self.radius.observe(lambda ch: self._refresh(recentre=True), "value")
        self.shape.observe(lambda ch: (self._fit_radius(), self._refresh(recentre=True)), "value")
        self.widget = w.VBox(
            [w.HBox([self.query, find]), self.shape, w.HBox([self.radius, recentre]), self.map, self.info]
        )
        if start_name and boundary is not None:
            self._set_boundary(boundary, start_name)
        self._refresh()

    # -- places -----------------------------------------------------------------
    def find(self, name: str) -> bool:
        """
        Look up a place. With an official boundary: outline it, centre on it
        and set the radius to cover it. Otherwise just move the pin there.
        """
        name = (name or "").strip()
        if not name:
            return False
        boundary = find_place(name)
        if boundary is not None:
            self._set_boundary(boundary, name)
            self._refresh(recentre=True)
            return True
        try:
            import osmnx as ox

            self._move(ox.geocoder.geocode(name), recentre=True)
            self.info.value += f"<br><span style='font-family:monospace'>“{name}” has no boundary in OpenStreetMap; pin placed on it.</span>"
        except Exception:
            self.info.value = f"<span style='font-family:monospace'>“{name}” not found.</span>"
        return False

    def _set_boundary(self, boundary: gp.GeoDataFrame, name: str):
        self.boundary = boundary
        self.place_name = str(boundary["display_name"].iloc[0]) if "display_name" in boundary else name
        self._fit = fit_area(boundary, self.margin)
        self.outline.data = boundary[["geometry"]].to_crs(4326).__geo_interface__
        self.center = self._fit["center"]
        self.marker.location = self.center
        self._fit_radius()

    def _fit_radius(self):
        """Radius slider = what the current shape needs to cover the place."""
        if not self._fit or self.shape.value not in ("Circle", "Square"):
            return
        need = self._fit["circle_radius" if self.shape.value == "Circle" else "square_half"]
        need = int(math.ceil(need / 50) * 50)
        if need > self.radius.max:
            self.radius.max = need
        self.radius.value = max(self.radius.min, need)

    def _on_search(self, **kw):
        # The map's 🔍 only reports a point; look the text up for its boundary
        text = kw.get("text") or ""
        if not (text and self.find(text)):
            self._move(kw["location"], recentre=True)

    # -- events ---------------------------------------------------------------
    def _on_click(self, **kw):
        if kw.get("type") == "click" and self.shape.value != "Drawn shape":
            self._move(kw["coordinates"])

    def _on_draw(self, target, action, geo_json):
        if action == "created":
            self.polygon = polygon_query(geo_json)
            # keep only the latest drawing
            self.draw.data = [geo_json]
        elif action == "deleted":
            self.polygon = None
        self._refresh()

    def _move(self, location, from_marker=False, recentre=False):
        self.center = (float(location[0]), float(location[1]))
        if self._fit and max(abs(a - b) for a, b in zip(self.center, self._fit["center"])) > 1e-7:
            self._fit = {}  # moved away from the found place: keep the radius as is
        if not from_marker:
            self.marker.location = self.center
        self._refresh(recentre=recentre)

    def recentre(self):
        """Centre and zoom the map on the selection.

        Also repairs a map that Colab drew before its cell had a size (pin
        stuck in a corner): nudging the centre forces the browser to redo
        the layout with the real size.
        """
        lat, lon = self.center
        size = self.radius.value
        if self.shape.value == "Boundary only" and self._fit:
            size = self._fit["circle_radius"]
        self.map.zoom = fit_zoom(lat, size)
        self.map.center = (lat + 1e-7, lon)
        self.map.center = (lat, lon)

    def _refresh(self, recentre=False):
        lat, lon = self.center
        r = self.radius.value
        self.circle.location = self.center
        self.circle.radius = r
        self.square.bounds = _square_bounds(lat, lon, r)
        mode = self.shape.value

        for layer, wanted in ((self.circle, mode == "Circle"), (self.square, mode == "Square"),
                              (self.draw, mode == "Drawn shape")):
            present = layer in self.map.layers or layer in self.map.controls
            if wanted and not present:
                self.map.add(layer)
            elif not wanted and present:
                self.map.remove(layer)
        self.radius.disabled = mode in ("Drawn shape", "Boundary only")
        if recentre and mode != "Drawn shape":
            self.recentre()

        if mode == "Drawn shape":
            msg = ("Shape drawn ✔" if self.polygon is not None
                   else "Use the ⬟ / ▭ tools on the left of the map to draw your area.")
        elif mode == "Boundary only":
            msg = (f"Boundary only: {self.place_name} ✔ (neighbouring areas removed)" if self.boundary is not None
                   else "Find a place with a boundary first (box above the map).")
        else:
            secs = "~20 s" if r <= 1000 else "~1 min" if r <= 2000 else "a few minutes"
            msg = f"Centre {lat:.5f}, {lon:.5f} · {mode.lower()} · radius {r} m · download {secs}"
        self.info.value = f"<span style='font-family:monospace'>{msg}</span>"

    # -- API --------------------------------------------------------------------
    def show(self):
        from IPython.display import display

        display(self.widget)
        self.recentre()
        return self

    def selection(self) -> dict:
        """Arguments for `fetch()`: query, radius, circle."""
        mode = self.shape.value
        if mode == "Drawn shape":
            if self.polygon is None:
                raise ValueError("No shape drawn yet - draw one on the map, or switch to Circle/Square.")
            return {"query": self.polygon, "radius": None, "circle": False}
        if mode == "Boundary only":
            if self.boundary is None:
                raise ValueError("No place found yet - use the Find box above the map, or switch to Circle/Square.")
            return {"query": self.boundary[["geometry"]], "radius": None, "circle": False}
        return {"query": self.center, "radius": self.radius.value, "circle": mode == "Circle"}
