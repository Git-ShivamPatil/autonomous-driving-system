from types import SimpleNamespace

from ads.sim.expert import reset_integral


class FakePID:
    """Same update rule as MetaDrive's PIDController: the integral accumulates without bound."""

    def __init__(self, kp, ki, kd):
        self.k_p, self.k_i, self.k_d = kp, ki, kd
        self.p_error = self.i_error = self.d_error = 0.0

    def get_result(self, err):
        self.i_error += err
        self.d_error = err - self.p_error
        self.p_error = err
        return -self.k_p * self.p_error - self.k_i * self.i_error - self.k_d * self.d_error


def test_reset_integral_removes_wind_up_but_keeps_derivative_state():
    policy = SimpleNamespace(heading_pid=FakePID(1.7, 0.01, 3.5), lateral_pid=FakePID(0.3, 0.002, 0.05))
    for _ in range(600):  # a steady 0.2 m offset the expert never gets to correct
        policy.lateral_pid.get_result(-0.2)
    wound = policy.lateral_pid.get_result(-0.2)
    reset_integral(policy)
    fresh = policy.lateral_pid.get_result(-0.2)
    assert abs(wound) > 0.2  # 0.002 * 0.2 * 601 of integral on top of 0.3 * 0.2 proportional
    assert abs(fresh - (0.3 * 0.2 + 0.002 * 0.2)) < 1e-9  # P plus a single step of integral; derivative 0
    assert policy.heading_pid.i_error == 0.0
