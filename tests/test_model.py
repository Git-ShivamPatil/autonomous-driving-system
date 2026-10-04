import torch

from ads import config
from ads.model import PilotNet, count_parameters


def test_parameter_count_is_about_250k():
    n = count_parameters(PilotNet())
    assert 240_000 <= n <= 260_000


def test_output_shape_and_uint8_input():
    model = PilotNet().eval()
    x = torch.randint(0, 256, (4, 3, config.NET_H, config.NET_W), dtype=torch.uint8)
    y = model(x, torch.full((4,), 30.0))
    assert y.shape == (4,)
    assert torch.isfinite(y).all()


def test_speed_input_reaches_the_output():
    torch.manual_seed(0)
    model = PilotNet().eval()
    x = torch.randint(0, 256, (2, 3, config.NET_H, config.NET_W), dtype=torch.uint8)
    slow = model(x, torch.tensor([5.0, 5.0]))
    fast = model(x, torch.tensor([60.0, 60.0]))
    assert not torch.allclose(slow, fast)


def test_gradients_flow():
    model = PilotNet()
    x = torch.randint(0, 256, (2, 3, config.NET_H, config.NET_W), dtype=torch.uint8)
    model(x, torch.tensor([10.0, 20.0])).sum().backward()
    assert all(p.grad is not None for p in model.parameters())
