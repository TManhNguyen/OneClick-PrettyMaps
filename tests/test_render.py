import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from oneclick_prettymaps import (
    BUILDING_MODES,
    LAYOUTS,
    PALETTES,
    STREET_MODES,
    Theme,
    from_colorhunt,
    render,
    save,
)
from oneclick_prettymaps.palettes import luminance

from fake_city import fake_city


@pytest.fixture(scope="module")
def city():
    return fake_city()


def test_colorhunt_link():
    assert from_colorhunt("https://colorhunt.co/palette/f9ed69f08a5db83b5e6a2c70") == [
        "#f9ed69", "#f08a5d", "#b83b5e", "#6a2c70",
    ]
    assert from_colorhunt("F9ED69F08A5D") == ["#f9ed69", "#f08a5d"]
    assert from_colorhunt("#112233, #445566 #778899") == ["#112233", "#445566", "#778899"]
    with pytest.raises(ValueError):
        from_colorhunt("https://colorhunt.co/")


@pytest.mark.parametrize("name", PALETTES)
def test_theme_contrast(name):
    t = Theme.from_palette(name)
    assert abs(luminance(t.background) - luminance(t.streets)) > 0.4
    assert t.buildings


@pytest.mark.parametrize("mode", BUILDING_MODES)
def test_building_modes(city, mode):
    fig = render(city, buildings=mode, seed=3)
    assert fig._oneclick["buildings"] in BUILDING_MODES[1:]
    plt.close(fig)


@pytest.mark.parametrize("mode", STREET_MODES)
def test_street_modes(city, mode):
    plt.close(render(city, streets=mode))


def test_auto_falls_back_to_size_for_untagged_city():
    fig = render(fake_city(tagged_share=0.1), buildings="auto")
    assert fig._oneclick["buildings"] == "size"
    plt.close(fig)


def test_effects_layouts_and_save(city, tmp_path):
    for layout in LAYOUTS:
        fig = render(
            city, theme="neon-night", streets="orientation", shadows=True, hatch=True,
            grain=0.08, title="Sai Gon", legend=True, layout=layout,
        )
        out = save(fig, str(tmp_path / f"{layout}.png"), dpi=60)
        plt.close(fig)
        assert (tmp_path / f"{layout}.png").stat().st_size > 10_000, out


def test_square_map_and_colorhunt_theme(city):
    fig = render(fake_city(circle=False), theme="https://colorhunt.co/palette/f9ed69f08a5db83b5e6a2c70")
    plt.close(fig)


def test_landuse_patchwork_toggle(city):
    with_zones = render(city, landuse=0.5)
    without = render(city, landuse=0)
    n_with = sum(len(a.collections) for a in with_zones.axes)
    n_without = sum(len(a.collections) for a in without.axes)
    assert n_with == n_without + 1
    plt.close(with_zones)
    plt.close(without)


def test_render_without_landuse_layer(city):
    plt.close(render({k: v for k, v in city.items() if k != "landuse"}))


def test_margin_centres_shape_with_border(city):
    from shapely.geometry import Polygon
    import geopandas as gp

    ward = gp.GeoDataFrame(
        geometry=[Polygon([(106.690, 10.770), (106.700, 10.770), (106.705, 10.785), (106.692, 10.790)])],
        crs=4326,
    )
    layers = {**city, "perimeter": ward}
    for margin in (0.0, 0.05):
        fig = render(layers, margin=margin, title=None, subtitle=None)
        ax = fig.axes[1]  # [paper, map, ...]
        shape = ward.to_crs(ward.estimate_utm_crs()).geometry.iloc[0]
        x0, y0, x1, y1 = shape.bounds
        width = max(x1 - x0, y1 - y0)
        assert abs(sum(ax.get_xlim()) / 2 - (x0 + x1) / 2) < 1e-6      # centred
        assert abs((ax.get_xlim()[1] - ax.get_xlim()[0]) - width * (1 + 2 * margin)) < 1e-6
        plt.close(fig)


def test_multipart_boundary_outlines_every_part(city):
    from shapely.geometry import MultiPolygon, box as bbox
    import geopandas as gp

    two_islands = gp.GeoDataFrame(
        geometry=[MultiPolygon([bbox(106.690, 10.770, 106.695, 10.775), bbox(106.700, 10.780, 106.704, 10.784)])],
        crs=4326,
    )
    fig = render({**city, "perimeter": two_islands}, title=None, subtitle=None)
    outlines = [l for l in fig.axes[1].lines if l.get_zorder() == 20]
    assert len(outlines) == 2
    plt.close(fig)


def test_seed_names_are_repeatable(city):
    import io
    import re as _re
    from oneclick_prettymaps import random_seed_name
    from oneclick_prettymaps.render import seed_to_int

    name = random_seed_name()
    assert _re.fullmatch(r"[a-z]+-\d{3}", name)
    assert seed_to_int("lotus-482") == seed_to_int(" lotus-482 ")
    assert seed_to_int("lotus-482") != seed_to_int("lotus-483")
    assert seed_to_int("7") == 7 == seed_to_int(7)

    def png(seed):
        fig = render(city, buildings="random", grain=0.1, seed=seed, title=None, subtitle=None, dpi=40)
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        return buf.getvalue()

    assert png("lotus-482") == png("lotus-482")
    assert png("lotus-482") != png("jade-101")
