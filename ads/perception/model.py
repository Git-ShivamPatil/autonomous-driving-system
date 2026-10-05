"""Multi-task YOLO11: one backbone and neck, a detection head and two segmentation heads.

The detection path is Ultralytics' YOLO11 DetectionModel unchanged (12 classes). Two lightweight
decoders, for drivable area and lane lines, read the neck's stride-8 P3 feature (layer 16) and the
backbone's stride-4 feature (layer 2), in the spirit of YOLOP and A-YOLOM: upsample, fuse the finer
feature, and predict one logit per pixel at stride 2, bilinearly upsampled to the input size.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F
from ultralytics.nn.modules import Conv
from ultralytics.nn.tasks import DetectionModel

P3_LAYER, P2_LAYER = 16, 2  # YOLO11 layer indices: neck P3/8 output, backbone C3k2 at stride 4
DETECTION_HYP = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)  # Ultralytics' default loss gains


class SegDecoder(nn.Module):
    def __init__(self, c_p3: int, c_p2: int, mid: int = 64) -> None:
        super().__init__()
        self.reduce = Conv(c_p3, mid, 3)
        self.fuse = Conv(mid + c_p2, mid, 3)
        self.refine = Conv(mid, mid // 2, 3)
        self.out = nn.Conv2d(mid // 2, 1, 1)

    def forward(self, p3: torch.Tensor, p2: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        x = F.interpolate(self.reduce(p3), size=p2.shape[-2:], mode="nearest")  # stride 8 -> 4
        x = self.fuse(torch.cat([x, p2], dim=1))
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)  # stride 4 -> 2
        x = self.out(self.refine(x))
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)  # logits at input size


class MultiTaskYOLO(DetectionModel):
    """forward(x) returns (detection output, drivable logits, lane logits); logits are (B, 1, H, W)."""

    def __init__(self, cfg: str = "yolo11s.yaml", nc: int = 12, verbose: bool = False) -> None:
        super().__init__(cfg, ch=3, nc=nc, verbose=verbose)
        self.save = sorted(set(self.save) | {P3_LAYER, P2_LAYER})
        with torch.no_grad():
            feats = self._features(torch.zeros(1, 3, 64, 64))
        c_p3, c_p2 = feats[P3_LAYER].shape[1], feats[P2_LAYER].shape[1]
        self.drivable_head = SegDecoder(c_p3, c_p2)
        self.lane_head = SegDecoder(c_p3, c_p2)
        self.args = DETECTION_HYP

    def _features(self, x: torch.Tensor) -> dict[int, torch.Tensor]:
        y, out = [], {}
        for m in self.model:
            if m.i == len(self.model) - 1:
                break
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)
            out[m.i] = x
        return out

    def _predict_once(self, x, profile=False, embed=None):
        size = x.shape[-2:]
        y = []
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)
        if not hasattr(self, "lane_head"):  # DetectionModel.__init__ probes strides before the heads exist
            return x
        return x, self.drivable_head(y[P3_LAYER], y[P2_LAYER], size), self.lane_head(y[P3_LAYER], y[P2_LAYER], size)


def build_model(scale: str = "s", nc: int = 12, pretrained: bool = True) -> MultiTaskYOLO:
    """YOLO11 multi-task model; with pretrained=True, COCO weights are copied wherever shapes match."""
    model = MultiTaskYOLO(f"yolo11{scale}.yaml", nc=nc)
    if pretrained:
        from ultralytics import YOLO
        from ultralytics.utils.torch_utils import intersect_dicts

        source = YOLO(f"yolo11{scale}.pt").model.float().state_dict()
        matched = intersect_dicts(source, model.state_dict())
        model.load_state_dict(matched, strict=False)
        model.transferred = f"{len(matched)}/{len(model.state_dict())} tensors from yolo11{scale}.pt"
    return model
