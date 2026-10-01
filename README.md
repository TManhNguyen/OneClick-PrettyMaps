# OneClick PrettyMaps

Colourful map posters of any place, straight from Google Colab. No coding needed.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TManhNguyen/OneClick-PrettyMaps/blob/main/PrettyMaps.ipynb)

Built on [**prettymaps**](https://github.com/marceloprates/prettymaps) by **Marcelo Prates**.
Map data © [OpenStreetMap](https://www.openstreetmap.org/copyright) contributors.
Both credits are printed on every poster, as the prettymaps author and the OSM licence ask.

## What it adds on top of prettymaps

**Pick the area on a map**
- Search for a place, click to move the pin, and choose a circle or square with a radius slider.
- Or draw any polygon or rectangle on the map.

**Colour that doesn't depend on parks and water**

Plain prettymaps colours buildings at random from two colours, so dense cities with little greenery come out mostly brown. Here you choose how buildings and streets are coloured:

| Buildings | |
|---|---|
| `auto` | Building `type` when most buildings are tagged, otherwise `size`. Many cities only tag buildings as "yes". |
| `type` | home / shop & office / civic / worship / industry |
| `height` | Number of floors (from OSM `building:levels` / `height`) |
| `size` | Footprint area |
| `distance` | Rings of colour out from the centre |
| `random` | Random, but repeatable with a seed |

| Streets | |
|---|---|
| `type` | Major roads stand out |
| `orientation` | Rainbow by compass direction |
| `solid` | One colour |

**Any palette, including [Color Hunt](https://colorhunt.co)**
- Paste a link such as `https://colorhunt.co/palette/f9ed69f08a5db83b5e6a2c70`, or pick a built-in palette.
- Colours are assigned to roles automatically: land, streets, buildings, parks and water.
- Streets are darkened when needed so all-pastel palettes stay readable.

**Effects**
- Light or dark (neon) mode, with glowing streets.
- Building drop shadows: taller buildings cast longer ones.
- prettymaps-style dotted hatch on parks and water.
- Paper gradient and paper grain.

**Poster layouts**
- `poster`, `square`, `a4` and `wallpaper`, with title, coordinates and legend.
- PDF or SVG export (vector, sharp at any print size) or PNG up to 600 dpi.

**Downloading and restyling**
- A live preview shows the map building up while each layer downloads (outline, buildings and parks, rivers and land use, streets).
- The data is downloaded once.
- Re-run the style cell as often as you like.

## Use from Python

```python
import oneclick_prettymaps as opm

gdfs = opm.fetch("Hoan Kiem, Hanoi", radius=1200, circle=True)   # or (lat, lon), or a drawn polygon
fig = opm.render(
    gdfs,
    theme="https://colorhunt.co/palette/f9ed69f08a5db83b5e6a2c70",
    buildings="auto", streets="orientation", shadows=True, title="Ha Noi",
)
opm.save(fig, "hanoi.png", dpi=300)
```

Install (prettymaps goes in without its own dependency list, which pulls heavy pen-plotter packages and upgrades `ipykernel`, breaking Colab):

```sh
pip install "oneclick-prettymaps[picker] @ git+https://github.com/TManhNguyen/OneClick-PrettyMaps"
pip install --no-deps "prettymaps @ git+https://github.com/marceloprates/prettymaps@02f85870ced807b7d24ce1764764ff877f9edffd"
```

Tests (offline, using synthetic map data): `pip install -e .[test] && cd tests && pytest`

## Licence

GNU Affero General Public License v3.0 (see [LICENSE](LICENSE)), the same licence as prettymaps.
You can use, modify and share this project, including commercially. If you distribute it, or run a modified version as a service, you must publish your source under the same licence and keep the credits.

The prettymaps author also asks that it **not be used to sell NFTs**. Please respect that.
