"""梯形速度剖面的数值属性: 收敛、不超调、限速、限加速度、存在减速段。"""
from __future__ import annotations

import numpy as np

from ...robot.trajectory import next_ramped_velocity

_DT = 0.02


def _simulate(error0: float, vmax: float, accel: float, dt: float = _DT, steps: int = 10000):
    """模拟单关节从 error0 收敛, 返回 (位置误差序列, 速度序列, 末误差)。"""
    err = error0
    v = 0.0
    poss = []
    vels = []
    for _ in range(steps):
        v = next_ramped_velocity(err, v, dt, vmax, accel)
        err -= v * dt
        poss.append(err)
        vels.append(v)
        if v == 0.0 and abs(err) < 1e-4:
            break
    return np.array(poss), np.array(vels), err


def test_converges_to_target() -> None:
    _, _, final_err = _simulate(1.5, vmax=1.0, accel=3.0)
    assert abs(final_err) < 1e-4


def test_never_overshoots() -> None:
    """位置误差序列不应变号(不越过目标)。"""
    for start in (1.5, -0.8, 0.2, -2.0):
        poss, _, _ = _simulate(start, vmax=1.0, accel=3.0)
        assert np.all(np.sign(poss[0]) * poss >= -1e-12), f"start={start} 发生超调"


def test_speed_within_vmax() -> None:
    _, vels, _ = _simulate(1.5, vmax=1.0, accel=3.0)
    assert np.max(np.abs(vels)) <= 1.0 + 1e-9


def test_accel_bounded() -> None:
    """加速度受限(严格 ≤ accel, 含最后一步精确着陆)。

    离散制动包络(见 trajectory.py)保证每步速度变化 ≤ accel·dt,
    因此该上界是 accel 本身, 而不是连续包络的 2·accel。
    """
    for start in (1.5, -0.8, 0.2, -2.0, 3.0, 0.02):
        _, vels, _ = _simulate(start, vmax=1.0, accel=3.0)
        dv = np.diff(vels) / _DT
        assert np.max(np.abs(dv)) <= 3.0 + 1e-9, f"start={start}"


def test_has_cruise_and_decel_phases() -> None:
    """长距离移动应出现匀速段(达到 vmax)和末段减速到很小的速度。"""
    _, vels, _ = _simulate(3.0, vmax=1.0, accel=3.0)
    assert abs(np.max(np.abs(vels)) - 1.0) < 1e-9           # 出现匀速段
    assert np.abs(vels[-3:]).max() < 0.2                     # 末段已明显减速


def test_deadband_holds() -> None:
    """死区内速度为 0。"""
    v = next_ramped_velocity(1e-5, 0.5, _DT, 1.0, 3.0)
    assert v == 0.0


def test_short_move_no_cruise() -> None:
    """短距离移动不应强行达到 vmax(只有加/减速, 三角剖面)。"""
    _, vels, _ = _simulate(0.02, vmax=1.0, accel=3.0)
    assert np.max(np.abs(vels)) < 1.0
