"""Phase 3 物理集成测试(MuJoCo 真 3D 仿真)。

标记 @pytest.mark.simulator -> 快速回归 `pytest -m "not simulator"` 不执行,
需要 `python -m pip install mujoco`。

关键断言对应 Phase 3 抓取物理 bug(已修复, 见 grab_block.py / _motion.py /
mujoco_so101_backend.py):
    A. 下降不推开物块: 修复前前倾工具的前指在下降时扫进物块(0.26 -> 0.23),
       修复(腕部自旋 q4=π/2 侧向夹取)后物块位移 < 2mm、全程无手指-物块接触。
    B. 闭合 + 抬升真实夹住并抬起物块(物理接触/摩擦, 不是"命令已发")。
    C. 释放后接触丢失 + 重力回落(物块回到桌面, 不跟随末端)。
"""
from __future__ import annotations

import numpy as np
import pytest

from ...runtime.cancellation import CancellationToken
from ...runtime.state import state
from ...skills._motion import GraspFailed, move_to_point
from ...skills.grab_block import _GRASP_OK, _NO_COLOR, grab_the_block
from ...skills.release_block import release_the_block

pytestmark = pytest.mark.simulator

_BLOCK_XY_TOL = 0.002          # 下降后物块水平位移阈值(m), 修复前是 0.030
_LIFTED_Z_M = 0.05             # 抓取后物块必须明显离开桌面(m)


def _pacing(control):
    from ...robot.controller import make_pacing
    return make_pacing(control.pacing)


def test_descend_does_not_push_cube(mujoco_env):
    """下降阶段不推开物块(修复前 bug: 前指扫进物块, 位移 30mm)。"""
    b = mujoco_env["backend"]
    model, control = mujoco_env["model"], mujoco_env["control"]
    b.reset("red_only")
    x0, y0, _ = b.cube_position("red")
    token = CancellationToken()
    wrist = b.grasp_wrist_roll
    assert wrist is not None, "物理后端必须侧向夹取(q4=π/2)"
    # 与 grab_block._grasp_sequence 相同的两次移动: 接近 + 下降
    move_to_point((x0, y0, 0.015 + control.approach_height_m),
                  b, model, control, token, _pacing(control), force_wrist_roll=wrist)
    move_to_point((x0, y0, 0.015 + control.descent_offset_m),
                  b, model, control, token, _pacing(control), force_wrist_roll=wrist)
    x1, y1, _ = b.cube_position("red")
    assert abs(x1 - x0) < _BLOCK_XY_TOL, f"下降推开了物块: x {x0:.3f}->{x1:.3f}"
    assert abs(y1 - y0) < _BLOCK_XY_TOL, f"下降推开了物块: y {y0:.3f}->{y1:.3f}"


def test_grasp_lift_release_physics(mujoco_env):
    """闭合 + 抬升真实夹住并抬起物块; 释放后重力落回桌面。"""
    b = mujoco_env["backend"]
    b.reset("red_only")
    assert grab_the_block("red") == _GRASP_OK
    assert b.is_holding_object(), "夹取后必须真的夹着物块"
    _, _, z = b.cube_position("red")
    assert z > _LIFTED_Z_M, f"物块必须被抬离桌面: z={z:.3f}"
    # 释放: 接触丢失 + 重力回落
    assert release_the_block() == "成功释放物块"
    assert not b.is_holding_object()
    _, _, z2 = b.cube_position("red")
    assert z2 <= 0.020, f"释放后物块应落回桌面(重力): z={z2:.3f}"


def test_grasp_missing_color_does_not_use_physics(mujoco_env):
    """没有该颜色 -> 协议失败串, 且不扰动场景。"""
    b = mujoco_env["backend"]
    b.reset("red_only")
    x0, y0, _ = b.cube_position("red")
    assert grab_the_block("yellow") == _NO_COLOR
    x1, y1, _ = b.cube_position("red")
    assert (x1, y1) == pytest.approx((x0, y0))
    assert not b.is_holding_object()


def test_grasp_works_across_presets(mujoco_env):
    """不同物块位置(含 y≠0, 靠底座最近 x=0.24)都能物理夹住。"""
    b = mujoco_env["backend"]
    for preset, color in (("red_yellow", "red"), ("red_yellow", "yellow"),
                          ("all", "red"), ("all", "yellow"), ("all", "white")):
        b.reset(preset)
        assert grab_the_block(color) == _GRASP_OK, f"{preset}/{color} 应夹住"
        _, _, z = b.cube_position(color)
        assert z > _LIFTED_Z_M, f"{preset}/{color} 未抬起: z={z:.3f}"
        assert release_the_block() == "成功释放物块"
