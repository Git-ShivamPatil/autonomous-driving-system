import numpy as np

from ads.perception import bdd


def _label(objects):
    return {"name": "x", "frames": [{"objects": objects}]}


def _box(cat, colour=None, x1=10.0, y1=20.0, x2=50.0, y2=80.0):
    attrs = {"occluded": False, "truncated": False, "trafficLightColor": colour or "none"}
    return {"category": cat, "attributes": attrs, "box2d": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}}


def test_twelve_classes():
    assert len(bdd.CLASSES) == 12
    assert len(set(bdd.CLASSES)) == 12


def test_2018_names_map_to_det20_and_lights_split_by_colour():
    lab = _label(
        [
            _box("person"),
            _box("motor"),
            _box("bike"),
            _box("traffic light", "red"),
            _box("traffic light", "yellow"),
            _box("traffic light", "green"),
            _box("traffic light", "none"),
            _box("trailer"),  # not a det_20 class
        ]
    )
    targets, dropped = bdd.detection_targets(lab)
    names = [bdd.CLASSES[int(c)] for c in targets[:, 0]]
    assert names == [
        "pedestrian",
        "motorcycle",
        "bicycle",
        "traffic light red",
        "traffic light yellow",
        "traffic light green",
    ]
    assert dropped == 1


def test_boxes_are_clipped_and_degenerate_boxes_dropped():
    lab = _label([_box("car", x1=-5, y1=700, x2=1300, y2=760), _box("car", x1=10, y1=10, x2=10.5, y2=40)])
    targets, _ = bdd.detection_targets(lab)
    assert targets.shape == (1, 5)
    assert targets[0, 1:].tolist() == [0.0, 700.0, 1280.0, 720.0]


def test_area_and_lane_objects_are_not_detections():
    lab = _label([{"category": "area/drivable", "poly2d": [[0, 0, "L"], [1, 1, "L"]]}])
    targets, _ = bdd.detection_targets(lab)
    assert targets.shape == (0, 5)


def test_bezier_segment_passes_through_its_end_points():
    pts = bdd.poly2d_points([[0, 0, "L"], [0, 10, "C"], [10, 10, "C"], [10, 0, "L"]])
    assert np.allclose(pts[0], [0, 0]) and np.allclose(pts[-1], [10, 0])
    assert pts[:, 1].max() > 5  # bulges toward the control points


def test_straight_polyline_is_unchanged():
    pts = bdd.poly2d_points([[0, 0, "L"], [5, 5, "L"], [9, 1, "L"]])
    assert pts.tolist() == [[0, 0], [5, 5], [9, 1]]


def test_lane_polylines_skip_crosswalks_and_vertical_lanes():
    seg = [[100, 700, "L"], [600, 400, "L"]]
    lab = _label(
        [
            {"category": "lane/single white", "attributes": {"direction": "parallel"}, "poly2d": seg},
            {"category": "lane/crosswalk", "attributes": {"direction": "parallel"}, "poly2d": seg},
            {"category": "lane/single white", "attributes": {"direction": "vertical"}, "poly2d": seg},
        ]
    )
    assert len(bdd.lane_polylines(lab)) == 1


def test_lane_mask_thickness_scales_with_resolution():
    line = [np.array([[0.0, 360.0], [1279.0, 360.0]])]
    full = bdd.lane_mask(line, 1280, 720, bdd.LANE_TRAIN_PX)
    half = bdd.lane_mask(line, 640, 360, bdd.LANE_TRAIN_PX)
    assert full[:, 640].sum() in (8, 9)
    assert half[:, 320].sum() in (4, 5)


def test_drivable_binary_merges_direct_and_alternative():
    ids = np.array([[0, 1], [2, 0]], dtype=np.uint8)
    assert bdd.drivable_binary(ids).tolist() == [[0, 1], [1, 0]]
