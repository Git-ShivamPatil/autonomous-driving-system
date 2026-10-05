import numpy as np
import pytest
import torch

from ads.perception import data
from ads.perception.evaluate import BinaryConfusion, box_iou, match_predictions


def test_flip_moves_boxes_with_the_image():
    img = np.zeros((360, 640, 3), np.uint8)
    img[100:120, 10:50] = 255
    mask = (img[..., 0] > 0).astype(np.uint8)
    boxes = np.array([[10.0, 100.0, 50.0, 120.0]], dtype=np.float32)
    fimg, (fmask,), fboxes = data.flip_lr(img, [mask], boxes)
    ys, xs = np.nonzero(fmask)
    assert fboxes[0].tolist() == [590.0, 100.0, 630.0, 120.0]
    assert (xs.min(), xs.max() + 1) == (590, 630)
    assert fimg[110, 600, 0] == 255


def test_affine_keeps_boxes_on_the_object():
    rng = np.random.default_rng(0)
    img = np.zeros((360, 640, 3), np.uint8)
    img[150:210, 300:380] = 255
    mask = (img[..., 0] > 0).astype(np.uint8)
    boxes = np.array([[300.0, 150.0, 380.0, 210.0]], dtype=np.float32)
    for _ in range(20):
        _, (m,), b, keep = data.random_affine(img, [mask], boxes, rng, data.AugmentConfig())
        ys, xs = np.nonzero(m)
        assert keep[0]
        assert b[0, 0] == pytest.approx(xs.min(), abs=2) and b[0, 2] == pytest.approx(xs.max() + 1, abs=2)
        assert b[0, 1] == pytest.approx(ys.min(), abs=2) and b[0, 3] == pytest.approx(ys.max() + 1, abs=2)


def test_collate_builds_batch_indices():
    s = [
        {
            "img": torch.zeros(3, 4, 4, dtype=torch.uint8),
            "drivable": torch.zeros(4, 4),
            "lane": torch.zeros(4, 4),
            "cls": torch.tensor([1.0, 2.0]),
            "bboxes": torch.rand(2, 4),
            "name": "a",
        },
        {
            "img": torch.zeros(3, 4, 4, dtype=torch.uint8),
            "drivable": torch.zeros(4, 4),
            "lane": torch.zeros(4, 4),
            "cls": torch.tensor([3.0]),
            "bboxes": torch.rand(1, 4),
            "name": "b",
        },
    ]
    b = data.collate(s)
    assert b["batch_idx"].tolist() == [0, 0, 1]
    assert b["cls"].shape == (3, 1) and b["img"].shape == (2, 3, 4, 4)


def test_box_iou():
    a = np.array([[0, 0, 10, 10]], float)
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]], float)
    assert box_iou(a, b)[0].round(3).tolist() == [1.0, 0.333, 0.0]


def test_match_predictions_is_one_to_one_and_class_aware():
    gt_cls = np.array([0.0])
    pred_cls = np.array([0.0, 0.0, 1.0])
    iou = np.array([[0.9, 0.8, 0.95]])  # second prediction duplicates the first; third has the wrong class
    correct = match_predictions(pred_cls, gt_cls, iou)
    assert correct[:, 0].tolist() == [True, False, False]
    assert correct[0].sum() == 9  # IoU 0.9 passes thresholds 0.50 .. 0.90


def test_match_predictions_gives_the_ground_truth_to_the_most_confident_prediction():
    # Two predictions both overlap one ground truth; prediction 0 is more confident (NMS order) but has
    # the lower IoU. As in Ultralytics and COCO, the more confident one is the true positive.
    iou = np.array([[0.6, 0.9]])
    correct = match_predictions(np.array([0.0, 0.0]), np.array([0.0]), iou)
    assert correct[:, 0].tolist() == [True, False]


def test_match_predictions_agrees_with_ultralytics():
    pytest.importorskip("ultralytics")
    from ultralytics.models.yolo.detect.val import DetectionValidator

    rng = np.random.default_rng(0)
    for _ in range(50):
        n_gt, n_pred = rng.integers(1, 6), rng.integers(1, 9)
        iou = rng.uniform(0, 1, (n_gt, n_pred)) * (rng.uniform(size=(n_gt, n_pred)) > 0.4)
        gt_cls = rng.integers(0, 3, n_gt).astype(float)
        pred_cls = rng.integers(0, 3, n_pred).astype(float)
        ours = match_predictions(pred_cls, gt_cls, iou)
        v = DetectionValidator.__new__(DetectionValidator)
        v.iouv = torch.linspace(0.5, 0.95, 10)
        theirs = v.match_predictions(torch.tensor(pred_cls), torch.tensor(gt_cls), torch.tensor(iou)).cpu().numpy()
        assert (ours == theirs).all()


def test_binary_confusion_metrics():
    c = BinaryConfusion()
    c.add(np.array([[1, 1], [0, 0]]), np.array([[1, 0], [1, 0]]))
    s = c.summary()
    assert s["iou"] == pytest.approx(1 / 3)
    assert s["recall"] == pytest.approx(0.5)
    assert s["balanced_accuracy"] == pytest.approx((0.5 + 0.5) / 2)
    assert s["iou_background"] == pytest.approx(1 / 3)


def test_model_shapes_and_loss_are_finite():
    pytest.importorskip("ultralytics")
    from ads.perception.losses import MultiTaskLoss
    from ads.perception.model import build_model

    torch.manual_seed(0)
    model = build_model("n", pretrained=False)
    model.train()
    x = torch.rand(2, 3, data.NET_H, data.NET_W)
    det, da, ll = model(x)
    assert da.shape == ll.shape == (2, 1, data.NET_H, data.NET_W)
    batch = {
        "cls": torch.tensor([[2.0], [11.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.3, 0.3, 0.05, 0.1]]),
        "batch_idx": torch.tensor([0.0, 1.0]),
        "drivable": (torch.rand(2, data.NET_H, data.NET_W) > 0.5).float(),
        "lane": (torch.rand(2, data.NET_H, data.NET_W) > 0.98).float(),
    }
    loss, items = MultiTaskLoss(model)((det, da, ll), batch)
    assert torch.isfinite(loss) and loss.item() > 0
    loss.backward()
    assert model.lane_head.out.weight.grad is not None
    assert model.drivable_head.out.weight.grad is not None
    assert {"box_loss", "cls_loss", "drivable_loss", "lane_loss"} <= set(items)
    model.eval()
    with torch.no_grad():
        det_eval, _, _ = model(x)
    assert det_eval[0].shape[1] == 4 + 12
