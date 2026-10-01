import matplotlib

matplotlib.use("Agg")

from unittest import mock

from fake_city import fake_city
from oneclick_prettymaps import LivePreview


def test_live_preview_updates_each_stage():
    handles = []

    def fake_display(obj, display_id=False):
        handle = mock.Mock()
        handles.append(handle)
        return handle

    with mock.patch("IPython.display.display", fake_display):
        preview = LivePreview(size=3, dpi=40)
    status, image = handles
    city = fake_city()

    preview("outline", {"perimeter": city["perimeter"]}, 0.1)
    preview("features", {k: city[k] for k in ("perimeter", "building", "green", "water")}, 2.0)
    preview.finish(3.0)

    pngs = [c.args[0].data for c in image.update.call_args_list]
    assert len(pngs) == 2 and all(p.startswith(b"\x89PNG") for p in pngs)
    final = status.update.call_args_list[-1].args[0].data
    assert "Map outline" in final and "shapes" in final and "Done in 3.0 s" in final
