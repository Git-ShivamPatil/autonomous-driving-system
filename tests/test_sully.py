import numpy as np

from ads import config
from ads.eval import sully


def test_split_has_a_gap_and_does_not_overlap():
    fit, test = sully.split_indices(1000)
    assert fit[-1] == 699 and test[0] == 750 and test[-1] == 999
    assert not set(fit) & set(test)


def test_real_frame_maps_to_the_network_input():
    frame = np.zeros((256, 455, 3), dtype=np.uint8)
    out = sully.preprocess_real(frame)
    assert out.shape == (config.NET_H, config.NET_W, 3) and out.dtype == np.uint8


def test_read_index_parses_both_line_formats(tmp_path):
    (tmp_path / "data.txt").write_text("0.jpg 0.000000\n1.jpg -12.5\n2.jpg 3.1,2018-07-01 12:00:00:000\n")
    names, angles = sully.read_index(tmp_path)
    assert names == ["0.jpg", "1.jpg", "2.jpg"]
    assert angles.tolist() == [0.0, -12.5, np.float32(3.1)]


def test_scores():
    s = sully.scores(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 5.0]))
    assert s["mae_deg"] == 2 / 3
    assert s["frames"] == 3
