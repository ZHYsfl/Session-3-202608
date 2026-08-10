"""move_to_coordinates(x, y, z) -> str：末端移动到指定笛卡尔坐标。

目标 xyz -> IK -> 目标关节角 -> 50Hz 闭环 -> 实际位置校验。
误差 = |实际 - 目标| 欧氏距离, 阈值 position_threshold_m 在 control.yaml。
不训练神经网络。
"""
from __future__ import annotations

import numpy as np

from ..robot.controller import make_pacing
from ..robot.kinematics import pose_distance
from ..runtime.state import state
from ._common import UnreachableError, fmt_error, fmt_xyz, parse_coordinate
from ._motion import GraspFailed, move_to_point


def move_to_coordinates(x: str, y: str, z: str) -> str:
    # ---- 参数解析(非法 -> CoordinateError, API 层转 HTTP 400) ---------
    target = np.array([parse_coordinate(x, "x"),
                       parse_coordinate(y, "y"),
                       parse_coordinate(z, "z")])

    backend = state.require_backend()
    model = state.require_model()
    control = state.require_control()

    token = state.begin_action()
    err: float = 0.0
    reached = False
    try:
        pacing = make_pacing(control.pacing)
        try:
            move_to_point(target, backend, model, control, token, pacing)
        except (UnreachableError, GraspFailed):
            # IK 无解 / 目标不可达 / 取消 / 超时 / 安全停止:
            # 一律诚实报告未到达, 误差 = ‖当前真实末端 - 目标点‖₂
            # (从 backend 实时读取, 禁止用人为下界/估计值冒充真实误差)。
            actual = backend.get_end_effector_pose().position
            err = pose_distance(actual, target)
        else:
            reached = True
            actual = backend.get_end_effector_pose().position
            err = pose_distance(actual, target)       # 到达后: 真实测量误差
    finally:
        state.end_action(token)

    # ---- 协议返回字符串(严格匹配文档) ---------------------------------
    if reached and err < control.position_threshold_m:
        return f"成功到达{fmt_xyz(target)}"
    return f"未到达{fmt_xyz(target)}，误差是{fmt_error(err)}"
