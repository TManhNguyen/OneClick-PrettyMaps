"""
OneClick PrettyMaps - colourful map posters on Google Colab.

Built on prettymaps by Marcelo Prates (https://github.com/marceloprates/prettymaps).
Map data © OpenStreetMap contributors.

Copyright (C) 2026 TManhNguyen

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

__version__ = "0.3.2"

from .fetch import fetch, polygon_query
from .fonts import load_fonts
from .palettes import PALETTES, Theme, from_colorhunt, get_palette
from .render import BUILDING_MODES, LAYOUTS, STREET_MODES, render, save


def __getattr__(name):
    # ipyleaflet is only needed for the interactive picker
    if name == "MapPicker":
        from .picker import MapPicker

        return MapPicker
    raise AttributeError(name)


__all__ = [
    "fetch", "polygon_query", "render", "save", "load_fonts", "MapPicker",
    "Theme", "PALETTES", "from_colorhunt", "get_palette",
    "BUILDING_MODES", "STREET_MODES", "LAYOUTS",
]
