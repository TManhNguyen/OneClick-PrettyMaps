"""
Poster fonts (Google Fonts, SIL Open Font License), downloaded on first use.
Falls back to matplotlib's DejaVu fonts when offline.

Copyright (C) 2026 TManhNguyen. Licensed under the GNU AGPL v3 (see LICENSE).
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

from matplotlib import font_manager

FONTS = {
    "display": ("Bebas Neue", "https://github.com/google/fonts/raw/main/ofl/bebasneue/BebasNeue-Regular.ttf"),
    "mono": ("Space Mono", "https://github.com/google/fonts/raw/main/ofl/spacemono/SpaceMono-Regular.ttf"),
}
FALLBACK = {"display": "DejaVu Sans", "mono": "DejaVu Sans Mono"}
CACHE = Path.home() / ".cache" / "oneclick_prettymaps" / "fonts"

_loaded: dict = {}


def load_fonts(timeout: float = 10) -> dict:
    """Download and register the poster fonts. Returns {role: family}."""
    CACHE.mkdir(parents=True, exist_ok=True)
    for role, (family, url) in FONTS.items():
        if role in _loaded:
            continue
        path = CACHE / url.rsplit("/", 1)[-1]
        try:
            if not path.exists():
                with urllib.request.urlopen(url, timeout=timeout) as r:
                    path.write_bytes(r.read())
            font_manager.fontManager.addfont(str(path))
            _loaded[role] = family
        except Exception:
            path.unlink(missing_ok=True)
            _loaded[role] = FALLBACK[role]
    return dict(_loaded)


def font_family(role: str) -> str:
    return _loaded.get(role, FALLBACK[role])
