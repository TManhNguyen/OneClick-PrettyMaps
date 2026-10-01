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
