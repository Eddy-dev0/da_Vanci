import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


class _CvStub(SimpleNamespace):
    INTER_NEAREST = 0
    BORDER_REFLECT101 = 0
    COLOR_BGR2RGB = 0
    COLOR_RGB2BGR = 1
    COLOR_BGR2GRAY = 2
    COLOR_RGB2GRAY = 3

    @staticmethod
    def resize(image, dsize, interpolation=None):
        arr = np.asarray(image, dtype=np.float32)
        if arr.size == 0:
            return arr.astype(np.float32)
        new_w, new_h = dsize
        sy = max(1, int(round(new_h / arr.shape[0])))
        sx = max(1, int(round(new_w / arr.shape[1])))
        arr = np.repeat(np.repeat(arr, sy, axis=0), sx, axis=1)
        return arr.astype(np.float32)

    @staticmethod
    def GaussianBlur(image, ksize, sigmaX, sigmaY=None, borderType=None):
        return np.asarray(image, dtype=np.float32)


sys.modules.setdefault("cv2", _CvStub())

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from painterslicer.image_analysis.analyzer import ImageAnalyzer, segment_image_into_layers
from painterslicer.image_analysis.pipeline import PaintingPipeline, enhance_layer
from painterslicer.image_analysis.layer_superres import compose_layers


def _make_mask(shape, active_coords):
    mask = np.zeros(shape, dtype=np.float32)
    for y, x in active_coords:
        mask[y, x] = 1.0
    return mask


def test_segment_image_into_layers_uses_masks(monkeypatch):
    img = np.zeros((4, 4, 3), dtype=np.float32)
    img[0:2, 0:2] = [1.0, 0.0, 0.0]  # background area
    img[2:4, 0:2] = [0.0, 1.0, 0.0]  # mid area
    img[0:2, 2:4] = [0.0, 0.0, 1.0]  # detail area

    background_mask = _make_mask((4, 4), [(y, x) for y in range(0, 2) for x in range(0, 2)])
    mid_mask = _make_mask((4, 4), [(y, x) for y in range(2, 4) for x in range(0, 2)])
    detail_mask = _make_mask((4, 4), [(y, x) for y in range(0, 2) for x in range(2, 4)])

    def fake_make_layer_masks(self, image_source):
        self.last_enhanced_rgb01 = np.asarray(image_source, dtype=np.float32)
        return {
            "background_mask": background_mask,
            "mid_mask": mid_mask,
            "detail_mask": detail_mask,
        }

    monkeypatch.setattr(ImageAnalyzer, "make_layer_masks", fake_make_layer_masks)

    result = segment_image_into_layers(img)

    masks = result["masks"]
    np.testing.assert_array_equal(masks["background_mask"], background_mask)
    np.testing.assert_array_equal(masks["mid_mask"], mid_mask)
    np.testing.assert_array_equal(masks["detail_mask"], detail_mask)

    layers = result["layers"]
    np.testing.assert_array_equal(layers["background"], img * background_mask[..., None])
    np.testing.assert_array_equal(layers["mid"], img * mid_mask[..., None])
    np.testing.assert_array_equal(layers["detail"], img * detail_mask[..., None])


def test_enhance_layer_respects_mask(monkeypatch):
    layer = np.array(
        [[
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]],
        dtype=np.float32,
    )
    mask = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    def fake_run(self, rgb_linear, *, scale, model_path):
        upsampled = np.repeat(np.repeat(rgb_linear, scale, axis=0), scale, axis=1)
        return upsampled, True

    monkeypatch.setattr(PaintingPipeline, "_run_super_resolution", fake_run)

    enhanced, applied = enhance_layer(layer, mask, scale=2)

    assert applied is True
    expected_mask = np.repeat(np.repeat(mask, 2, axis=0), 2, axis=1)
    np.testing.assert_array_equal(
        enhanced * (1.0 - expected_mask)[..., None], np.zeros_like(enhanced)
    )
    on_mask = enhanced[expected_mask == 1]
    np.testing.assert_allclose(on_mask, np.round(on_mask), atol=1e-5)


def test_compose_layers_without_feather_preserves_geometry():
    bg = np.zeros((2, 2, 4), dtype=np.float32)
    bg[..., :3] = [1.0, 0.0, 0.0]
    bg[..., 3] = np.array([[1.0, 1.0], [0.0, 0.0]], dtype=np.float32)

    mid = np.zeros((2, 2, 4), dtype=np.float32)
    mid[..., :3] = [0.0, 1.0, 0.0]
    mid[..., 3] = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.float32)

    fg = np.zeros((2, 2, 4), dtype=np.float32)
    fg[..., :3] = [0.0, 0.0, 1.0]
    fg[..., 3] = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32)

    composed = compose_layers(bg, mid, fg, feather_radius=0)

    expected = np.array(
        [
            [[0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 0.0, 1.0]],
            [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(composed, expected)


def test_plan_painting_layers_falls_back_without_alias(monkeypatch):
    called_impl = []

    def fake_extract_color_layers_impl(self, image_source, **kwargs):
        called_impl.append(True)
        self.last_color_analysis = {
            "labels": np.zeros((2, 2), dtype=np.int32),
        }
        return [
            {
                "label": 0,
                "color_rgb": (255, 0, 0),
                "pixel_paths": [[(0, 0), (0, 1), (1, 0), (1, 1)]],
            }
        ]

    def fake_make_layer_masks(self, image_source):
        mask = np.ones((2, 2), dtype=np.float32)
        self.last_layer_analysis = {"score_maps": {}}
        self.last_enhanced_rgb01 = np.asarray(image_source, dtype=np.float32)
        return {
            "background_mask": mask,
            "mid_mask": np.zeros_like(mask),
            "detail_mask": np.zeros_like(mask),
        }

    monkeypatch.setattr(
        ImageAnalyzer,
        "_extract_color_layers_impl",
        fake_extract_color_layers_impl,
        raising=False,
    )
    monkeypatch.setattr(ImageAnalyzer, "make_layer_masks", fake_make_layer_masks)
    monkeypatch.setattr(ImageAnalyzer, "enhance_image_quality", lambda self, src: None)
    monkeypatch.setattr(
        ImageAnalyzer,
        "_ensure_rgb01",
        lambda self, src: np.asarray(src, dtype=np.float32),
        raising=False,
    )

    analyzer = ImageAnalyzer()
    analyzer.extract_color_layers = None

    image = np.ones((2, 2, 3), dtype=np.float32)

    result = analyzer.plan_painting_layers(image)

    assert called_impl
    assert result["layers"]
    assert result["layer_masks"]["background_mask"].shape == (2, 2)
