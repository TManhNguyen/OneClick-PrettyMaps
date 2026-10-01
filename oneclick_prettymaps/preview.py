"""
Live preview while map data downloads.

    gdfs = fetch("Hoi An", radius=800, progress=LivePreview())

After each download stage the notebook shows what has arrived so far: a
quick, low-resolution render plus a status line with timings.

Copyright (C) 2026 TManhNguyen. Licensed under the GNU AGPL v3 (see LICENSE).
"""

from __future__ import annotations

import io
from typing import Dict

import geopandas as gp
import matplotlib.pyplot as plt

STAGES = {
    "outline": "Map outline",
    "features": "Buildings, parks & water",
    "rivers": "Rivers & land use",
    "streets": "Streets",
    "sea": "Sea",
}
_LAYERS = {
    "features": ("building", "green", "forest", "water", "beach", "rock", "parking"),
    "rivers": ("waterway", "landuse"),
    "streets": ("streets",),
    "sea": ("sea",),
}


class LivePreview:
    """Progress callback for `fetch()` that redraws the map after every stage."""

    def __init__(self, theme="saigon-sunset", size: float = 5.0, dpi: int = 80):
        from IPython.display import HTML, display

        self.theme = theme
        self.size = size
        self.dpi = dpi
        self.rows = []
        self._status = display(HTML(self._html(running=True)), display_id=True)
        self._image = display(HTML(""), display_id=True)

    def __call__(self, stage: str, gdfs: Dict[str, gp.GeoDataFrame], seconds: float):
        from IPython.display import HTML, Image

        shapes = sum(len(gdfs.get(name, ())) for name in _LAYERS.get(stage, ()))
        self.rows.append((STAGES.get(stage, stage), shapes, seconds))
        self._status.update(HTML(self._html(running=True)))
        try:
            self._image.update(Image(data=self._render(gdfs)))
        except Exception as e:  # a preview must never break the download
            self._image.update(HTML(f"<i>Preview unavailable: {e}</i>"))

    def finish(self, seconds: float):
        from IPython.display import HTML

        self._status.update(HTML(self._html(running=False, total=seconds)))

    def _render(self, gdfs) -> bytes:
        from .render import render

        fig = render(
            gdfs, theme=self.theme, layout="square", figsize=(self.size, self.size), dpi=self.dpi,
            title=None, subtitle=None, outline=False, gradient=False, grain=0,
        )
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=self.dpi, facecolor=fig.get_facecolor())
        plt.close(fig)
        return buf.getvalue()

    def _html(self, running: bool, total: float | None = None) -> str:
        lines = []
        previous = 0.0
        for label, shapes, seconds in self.rows:
            count = f" · {shapes:,} shapes" if shapes else ""
            lines.append(f"✔ {label}{count} · {seconds - previous:.1f} s")
            previous = seconds
        if running:
            lines.append("⏳ downloading next layer…" if self.rows else "⏳ starting…")
        else:
            lines.append(f"<b>Done in {total:.1f} s</b>")
        return "<div style='font-family:monospace;line-height:1.5'>" + "<br>".join(lines) + "</div>"
