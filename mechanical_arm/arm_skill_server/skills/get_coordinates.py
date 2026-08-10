"""get_current_coordinates() -> str：获取机械臂末端当前坐标。

实现: 直接读末端 FK 位姿(仿真) / 关节编码器->FK(真机)。
禁止使用神经网络。
"""
from __future__ import annotations

from ..runtime.state import state
from ._common import fmt_coord


def get_current_coordinates() -> str:
    backend = state.require_backend()
    p = backend.get_end_effector_pose().position

    # TODO(协议待确认): 协议文档 §2.1 对“速度低于阈值时返回的坐标位置描述”
    #   没有定义具体字符串格式; “速度高于阈值返回 我的坐标是{x},{y},{z}” 已明确。
    #   当前统一返回 "我的坐标是{x},{y},{z}"。
    #   get_end_effector_velocity() 已可用, 阈值语义确认后再按
    #   control.velocity_threshold_mps 分支出不同格式。见 docs/architecture_notes.md。
    return f"我的坐标是{fmt_coord(p[0])},{fmt_coord(p[1])},{fmt_coord(p[2])}"
