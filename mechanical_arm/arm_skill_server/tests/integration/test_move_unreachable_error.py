"""FINAL ACCEPTANCE AUDIT 回归: move_to_coordinates 不可达点的误差必须真实。

背景(修复前 bug):
    (0.20,-0.05,0.15) 在可达半径内(max_reach)但物理上 IK 不可行
    (x<0.22 的 descend 需要肘部落到底座/桌面以下, collision_guard=True 拒绝)。
    修复前 _motion.py 用 `max(pose_distance(target, proj), 1e-3)` 作下界:
    此时 proj==target, pose_distance==0 -> 报出 0.0010 的**假误差**(人为下界),
    而真实末端距目标 ~0.18m, 差距一个量级。
    修复后: 一律从 backend 实时读真实末端位置, 误差 = ‖ee - target‖₂。
    本文件锁死该行为, 防止回退成"写死 0.0010"。
"""
from __future__ import annotations

import re

import numpy as np
import pytest

from ...skills.move_to_coordinates import move_to_coordinates

pytestmark = pytest.mark.simulator

# 从协议字符串解析误差: "未到达0.200,-0.050,0.150，误差是0.1795"
_ERR_RE = re.compile(r"未到达[^，]*，误差是([0-9.]+)")


def test_unreachable_in_reach_error_is_real_distance(mujoco_env) -> None:
    """目标在可达半径内但 IK 不可行: 误差必须 = 真实末端到目标点的欧氏距离。

    回归断言:
        1) 不返回假成功(必须以 未到达 开头)。
        2) 解析出的误差 ≈ backend 实时读取的 ‖ee - target‖₂(容差 2mm)。
        3) 误差明显不是 0.0010(修复前的写死下界)。
    """
    b = mujoco_env["backend"]
    b.reset("all")
    target = np.array([0.20, -0.05, 0.15])

    r = move_to_coordinates("0.20", "-0.05", "0.15")

    # 1) 诚实报告未到达
    assert r.startswith("未到达"), f"不可达点不应报成功: {r!r}"
    m = _ERR_RE.search(r)
    assert m is not None, f"未解析出误差: {r!r}"
    reported = float(m.group(1))

    # 2) 误差必须等于当前真实末端到目标点的欧氏距离(不是估计/下界)
    actual = b.get_end_effector_pose().position
    real = float(np.linalg.norm(actual - target))
    assert abs(reported - real) < 0.002, (
        f"误差不真实: reported={reported:.4f}, real=‖ee-target‖₂={real:.4f}, "
        f"ee={actual}, target={target}, result={r!r}")

    # 3) 明确不是修复前的写死 0.0010
    assert not (0.0008 <= reported <= 0.0012), (
        f"回归到人为下界 0.0010: {r!r}")


def test_unreachable_out_of_workspace_reports_large_real_error(mujoco_env) -> None:
    """工作空间外目标: 误差也是真实距离, 且大(不会退回 0.0010)。"""
    b = mujoco_env["backend"]
    b.reset("all")
    target = np.array([0.90, 0.90, 0.30])

    r = move_to_coordinates("0.90", "0.90", "0.30")
    assert r.startswith("未到达"), f"工作空间外不应报成功: {r!r}"
    m = _ERR_RE.search(r)
    assert m is not None
    reported = float(m.group(1))
    assert reported > 0.05, f"工作空间外误差应很大, 却报出 {reported:.4f}: {r!r}"

    actual = b.get_end_effector_pose().position
    real = float(np.linalg.norm(actual - target))
    assert abs(reported - real) < 0.002, (
        f"误差不真实: reported={reported:.4f}, real={real:.4f}, result={r!r}")
