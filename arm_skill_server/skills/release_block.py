"""release_the_block() -> str：释放当前抓取的物块。

内部先判定是否真的夹着物块(仿真: gripper 闭合 + 物块在末端附近 + 物块在桌面上方),
再 open_gripper() 并验证。真机阶段将用夹爪位置/电流 + 相机交叉验证。
"""
from __future__ import annotations

from ..perception.camera2_verify import camera2_check_enabled, verify_release
from ..runtime.state import state


def release_the_block() -> str:
    backend = state.require_backend()

    # 1) 是否真的夹着物块(协议 §2.4 第 1 步)
    if not backend.is_holding_object():
        return "本身就没加起来物块"

    # 2) 释放
    backend.open_gripper()

    # 3) 验证释放结果(协议 §2.4 第 2 步; Phase 6: 物理 + camera2 第三视角交叉验证)
    if backend.is_holding_object():
        return "释放物块失败，请用手直接拿出来物块"
    if _release_verified_by_camera2(state.last_grabbed_color):
        state.last_grabbed_color = None
        return "成功释放物块"
    return "释放物块失败，请用手直接拿出来物块"


def _release_verified_by_camera2(color: str | None) -> bool:
    """物理验证通过后, 再用 camera2 确认物块已放回(协议 §2.4「交叉验证」)。

    未启用 / 未注入视觉 / 无 camera2 / 取帧失败 / 未知颜色 -> 返回 True(物理
    已验证 open_gripper 后不再 holding, 足以判定); camera2 正常且知道颜色时,
    视觉判定参与最终结果。
    """
    control = state.require_control()
    if not camera2_check_enabled(control) or color is None:
        return True
    r = verify_release(color, state.require_backend(), state.require_camera(),
                       state.require_vision(), state.require_vision_cfg(), control)
    return True if r is None else r
