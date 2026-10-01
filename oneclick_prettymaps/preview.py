"""
Live preview while map data downloads.

    gdfs = fetch("Hoi An", radius=800, progress=LivePreview())

After each download stage the notebook shows what has arrived so far: a
quick, low-resolution render plus a status line with timings.

Copyright (C) 2026 TManhNguyen. Licensed under the GNU AGPL v3 (see LICENSE).
"""

from __future__ import annotations

import html
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

    def __call__(self, stage: str, gdfs: Dict[str, gp.GeoDataFrame], seconds: float, error: str | None = None):
        from IPython.display import HTML, Image

        shapes = sum(len(gdfs.get(name, ())) for name in _LAYERS.get(stage, ()))
        self.rows.append((STAGES.get(stage, stage), shapes, seconds, error))
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
        for name, shapes, seconds, error in self.rows:
            label = html.escape(name)
            took = f"{seconds - previous:.1f} s"
            previous = seconds
            if error:
                lines.append(
                    f"<span style='color:#d9480f'>⚠ {label} · download failed after {took}: "
                    f"{html.escape(error[:200])}</span>"
                )
                continue
            if name == STAGES["outline"]:
                count = ""
            elif shapes:
                count = f" · {shapes:,} shapes"
            else:
                count = " · none mapped here"
            lines.append(f"✔ {label}{count} · {took}")
        if running:
            lines.append("⏳ downloading next layer…" if self.rows else "⏳ starting…")
        elif any(row[3] for row in self.rows):
            lines.append(
                f"<b>Finished in {total:.1f} s with download problems</b> (⚠ above). "
                "The OpenStreetMap servers may be busy: wait a minute and run this step again."
            )
        else:
            lines.append(f"<b>Done in {total:.1f} s</b>")
        return "<div style='font-family:monospace;line-height:1.5'>" + "<br>".join(lines) + "</div>"
