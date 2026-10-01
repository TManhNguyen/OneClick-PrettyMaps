"""
Palettes and themes.

A palette is just a list of hex colours (e.g. a 4-colour palette from
https://colorhunt.co). A Theme assigns those colours to map roles
(background, streets, buildings, parks, water...) automatically, so any
palette produces a readable map.

Copyright (C) 2026 TManhNguyen. Licensed under the GNU AGPL v3 (see LICENSE).
"""

from __future__ import annotations

import colorsys
import re
from dataclasses import dataclass, field, replace
from typing import List, Sequence

from matplotlib.colors import to_hex, to_rgb

# Hand-picked palettes. Each one works as a light or dark map; feel free to
# add your own, or paste any Color Hunt link into `from_colorhunt`.
PALETTES = {
    "saigon-sunset": ["#FFF4E0", "#F4A259", "#E76F51", "#8E3B46", "#264653"],
    "pastel-pop": ["#FDF6F0", "#F7B2BD", "#A0D2DB", "#C3B1E1", "#FFE29A"],
    "tropical": ["#F6F7EB", "#2EC4B6", "#FF9F1C", "#E71D36", "#011627"],
    "terracotta": ["#F3E9DC", "#C08552", "#895737", "#DAB49D", "#5E3023"],
    "nordic": ["#ECEFF4", "#88C0D0", "#81A1C1", "#B48EAD", "#2E3440"],
    "mint-candy": ["#F1FAEE", "#A8DADC", "#F4ACB7", "#FFCAD4", "#1D3557"],
    "retro-80s": ["#FFF1D0", "#F72585", "#7209B7", "#4CC9F0", "#3A0CA3"],
    "autumn": ["#FEFAE0", "#DDA15E", "#BC6C25", "#606C38", "#283618"],
    "ink": ["#F5F5F0", "#BDBDBD", "#8C8C8C", "#5A5A5A", "#1E1E1E"],
    "neon-night": ["#0B0E1A", "#00F5D4", "#F15BB5", "#FEE440", "#9B5DE5"],
    "blueprint": ["#0F3057", "#E7F2F8", "#5DA9E9", "#9AD1F5", "#0B2545"],
}

_HEX_RE = re.compile(r"#?([0-9a-fA-F]{6})")


def from_colorhunt(link_or_hex: str) -> List[str]:
    """
    Turn a Color Hunt link into a palette.

    Accepts a palette URL such as
    'https://colorhunt.co/palette/f9ed69f08a5db83b5e6a2c70', a bare 24-char
    code like 'f9ed69f08a5db83b5e6a2c70', or comma/space separated hex
    colours ('#f9ed69, #f08a5d, ...').
    """
    text = link_or_hex.strip()
    code = text.rstrip("/").split("/")[-1].split("?")[0]
    if re.fullmatch(r"[0-9a-fA-F]+", code) and len(code) % 6 == 0:
        colors = [code[i : i + 6] for i in range(0, len(code), 6)]
    else:
        colors = _HEX_RE.findall(text)
    if len(colors) < 2:
        raise ValueError(f"Could not find at least 2 colours in {link_or_hex!r}")
    return ["#" + c.lower() for c in colors]


def get_palette(name_or_link: str | Sequence[str]) -> List[str]:
    """Palette by name, Color Hunt link, hex string, or list of colours."""
    if not isinstance(name_or_link, str):
        return [to_hex(c) for c in name_or_link]
    if name_or_link in PALETTES:
        return list(PALETTES[name_or_link])
    return from_colorhunt(name_or_link)


def luminance(color) -> float:
    r, g, b = to_rgb(color)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def mix(c1, c2, t: float) -> str:
    """Blend c1 towards c2 by t (0 = c1, 1 = c2)."""
    a, b = to_rgb(c1), to_rgb(c2)
    return to_hex(tuple(x + (y - x) * t for x, y in zip(a, b)))


def _hue_match(colors: Sequence[str], lo: float, hi: float) -> str | None:
    """Most saturated colour whose hue (degrees) lies in [lo, hi]."""
    best, best_sat = None, 0.15
    for c in colors:
        h, l, s = colorsys.rgb_to_hls(*to_rgb(c))
        if lo <= h * 360 <= hi and s > best_sat and 0.15 < l < 0.9:
            best, best_sat = c, s
    return best


@dataclass
class Theme:
    page: str  # paper around the map
    background: str  # land inside the map
    streets: str
    buildings: List[str]  # accent colours used for buildings / gradients
    green: str
    forest: str
    water: str
    beach: str
    rock: str
    parking: str
    text: str
    outline: str
    dark: bool = False
    glow: bool = False
    extras: dict = field(default_factory=dict)

    @classmethod
    def from_palette(cls, palette, dark: bool | None = None, glow: bool | None = None) -> "Theme":
        """
        Assign palette colours to map roles.

        Light themes: lightest colour = land, darkest = streets, the rest
        colour the buildings. Dark themes swap land and streets. Parks and
        water reuse a green / blue from the palette when there is one,
        otherwise a soft green / blue is blended into the land colour so
        they still fit the palette.
        """
        colors = get_palette(palette)
        ordered = sorted(colors, key=luminance)
        darkest, lightest = ordered[0], ordered[-1]
        if dark is None:
            # Palettes listed dark-first (e.g. neon-night) become dark maps
            dark = luminance(colors[0]) < 0.25
        land, ink = (darkest, lightest) if dark else (lightest, darkest)
        accents = [c for c in colors if c not in (land, ink)] or [ink]
        # All-pastel palettes are common: deepen the street colour (same hue)
        # until it stands out from the land.
        target = "#FFFFFF" if dark else "#111111"
        t = 0.0
        while abs(luminance(mix(ink, target, t)) - luminance(land)) < 0.45 and t < 1:
            t += 0.05
        ink = mix(ink, target, t)

        base_green, base_blue = ("#2D6A4F", "#1B3A5C") if dark else ("#95C77E", "#9CCFE0")
        green = _hue_match(accents, 70, 170) or mix(land, base_green, 0.55)
        water = _hue_match(accents, 175, 250) or mix(land, base_blue, 0.6)

        return cls(
            page=mix(land, ink, 0.06),
            background=land,
            streets=ink,
            buildings=accents,
            green=green,
            forest=mix(green, ink, 0.2),
            water=water,
            beach=mix(land, "#F2D49B", 0.5),
            rock=mix(land, "#9E9E9E", 0.4),
            parking=mix(land, ink, 0.08),
            text=ink,
            outline=mix(ink, land, 0.15),
            dark=dark,
            glow=dark if glow is None else glow,
        )

    def with_(self, **kwargs) -> "Theme":
        return replace(self, **kwargs)
