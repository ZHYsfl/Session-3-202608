"""skills 公共工具: 坐标解析/格式化/可达规划/IK 求解。

格式化约定(实现选择, 已在 docs/api.md 记录, 待负责人确认):
    坐标分量  ":.3f"   -> "0.180"
    误差      ":.4f"   -> "0.0123"
协议只约束字符串形状(如 "我的坐标是{x},{y},{z}"), 不约束小数位数。
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from ..robot.kinematics import RobotModel, solve_ik


class CoordinateError(ValueError):
    """坐标参数错误(非数值/NaN/Inf)。API 层转 HTTP 400。"""


class UnreachableError(Exception):
    """目标不可达或 IK 无解。携带当前真实末端到目标点的欧氏距离(m)。"""

    def __init__(self, err_m: float) -> None:
        super().__init__(f"目标不可达(当前末端-目标点距离 {err_m:.4f} m)")
        self.error_m = err_m


# ---------------------------------------------------------------------------
# 格式化
# ---------------------------------------------------------------------------

def fmt_coord(v: float) -> str:
    return f"{v:.3f}"


def fmt_error(v: float) -> str:
    return f"{v:.4f}"


def fmt_xyz(xyz: np.ndarray | tuple[float, float, float]) -> str:
    return f"{fmt_coord(xyz[0])},{fmt_coord(xyz[1])},{fmt_coord(xyz[2])}"


# ---------------------------------------------------------------------------
# 坐标解析(协议要求 x/y/z 为字符串)
# ---------------------------------------------------------------------------

def parse_coordinate(s: str, field: str) -> float:
    """把字符串坐标解析为 float。非法 -> CoordinateError(API 层转 400)。"""
    try:
        v = float(s)
    except (TypeError, ValueError):
        raise CoordinateError(f"字段 {field} 必须是数值字符串, 实际: {s!r}")
    if not math.isfinite(v):
        raise CoordinateError(f"字段 {field} 必须是有限数值, 实际: {s!r}")
    return v


# ---------------------------------------------------------------------------
# 可达规划 + IK
# ---------------------------------------------------------------------------

def reachable_projection(model: RobotModel, target: np.ndarray) -> np.ndarray:
    """目标超出工作空间时, 投影到可达球面上(方向不变, 半径为 max_reach)。

    这样即便目标不可达, 也能让机械臂走到“最近可达点”, 之后用真实测量误差
    报告“未到达”(诚实, 不编造误差值)。
    """
    d = float(np.linalg.norm(target))
    if d <= model.max_reach_m:
        return target
    return target * (model.max_reach_m / d)


def solve_ik_robust(model: RobotModel, target: np.ndarray,
                    q0: np.ndarray) -> Optional[np.ndarray]:
    """IK 求解: 依次尝试 [当前关节角, home, 3 个随机抖动], 返回第一个收敛解。

    随机重启用于远离局部极小/奇异初值。全部失败返回 None。
    """
    rng = np.random.default_rng(0)
    candidates: list[np.ndarray] = [np.asarray(q0, dtype=float), model.q_home()]
    for _ in range(3):
        jitter = (model.q_max() - model.q_min()) * (rng.random(model.dof) - 0.5) * 0.2
        candidates.append(np.clip(model.q_home() + jitter, model.q_min(), model.q_max()))

    for q in candidates:
        result = solve_ik(model, target, q0=q, tol=1e-3, max_iter=150)
        if result is not None:
            return result
    return None
