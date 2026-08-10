"""Phase 6 camera2 第三视角交叉验证(协议 §2.3/§2.4「第二机位摄像头判断」)。

用 MuJoCo 真渲染帧 + HSV 检测验证 verify_grasp / verify_release 的三值语义:
    True   视觉确认(物块在末端附近 且 末端已离桌; 释放后物块不再在末端附近)
    False  视觉反证(物块可见但不在末端附近, 物理却说已夹住)
    None   无法判定(未检出被遮挡 / 无 camera2 / 取帧失败) -> 调用方回退物理

标记 simulator -> 快速回归 `pytest -m "not simulator"` 不执行。
"""
from __future__ import annotations

import numpy as np
import pytest

from ...config import ControlConfig, load_vision_config
from ...perception.camera2_verify import verify_grasp, verify_release
from ...perception.hsv_color_detector import HSVColorDetector
from ...perception.simulator_camera import SimulatorCamera
from ...robot.controller import make_pacing
from ...runtime.cancellation import CancellationToken
from ...runtime.state import state
from ...skills._motion import move_to_point

pytestmark = pytest.mark.simulator

_GT = {"red": (0.26, -0.06), "yellow": (0.30, 0.05), "white": (0.24, 0.06)}


def _setup_vision(mujoco_env):
    """配置全局状态注入 MuJoCo 后端 + SimulatorCamera + HSV 检测 + camera2 启用。"""
    backend = mujoco_env["backend"]
    vision_cfg = load_vision_config()
    camera = SimulatorCamera(backend, vision_cfg)
    detector = HSVColorDetector(vision_cfg, camera_model=camera.camera_model)
    control = ControlConfig(**{**mujoco_env["control"].__dict__,
                               "grasp_verify_camera2": True})
    state.configure(backend=backend, queue=mujoco_env["queue"], control=control,
                    model=mujoco_env["model"],
                    vision=detector, camera=camera, vision_cfg=vision_cfg)
    return backend, camera, detector, vision_cfg, control


def _grab_physically(backend, model, control, color: str) -> None:
    """纯物理抓取(不经 camera2 验证): approach -> descend -> close -> lift。"""
    bz = 0.015                                   # 物块中心 z = 桌面 + 半高
    gx, gy = _GT[color]
    token = CancellationToken()
    wrist = backend.grasp_wrist_roll
    backend.open_gripper()
    move_to_point((gx, gy, bz + control.approach_height_m),
                  backend, model, control, token, make_pacing(control.pacing),
                  force_wrist_roll=wrist)
    move_to_point((gx, gy, bz + control.descent_offset_m),
                  backend, model, control, token, make_pacing(control.pacing),
                  force_wrist_roll=wrist)
    backend.close_gripper()
    assert backend.is_holding_object(), f"物理上应夹住 {color}"
    move_to_point((gx, gy, bz + control.lift_height_m),
                  backend, model, control, token, make_pacing(control.pacing),
                  force_wrist_roll=wrist)


@pytest.mark.parametrize("color", ["red", "yellow", "white"])
def test_verify_grasp_true_when_physically_held(mujoco_env, color: str) -> None:
    """物理上已夹起(末端离桌)时, camera2 应视觉确认(True)或无法判定(None)。

    诚实性关键: 物理已成功时 camera2 **绝不返回 False**(被遮挡 -> None, 回退物理)。
    """
    backend, camera, detector, vc, control = _setup_vision(mujoco_env)
    backend.reset("all")
    _grab_physically(backend, mujoco_env["model"], control, color)

    r = verify_grasp(color, backend, camera, detector, vc, control)
    assert r is True or r is None, f"{color} 夹起后 verify_grasp={r} 不应为 False"


def test_verify_grasp_false_when_block_on_table(mujoco_env) -> None:
    """物块在桌上(未夹起)时, camera2 不应视觉确认(False 或 None, 绝不 True)。"""
    backend, camera, detector, vc, control = _setup_vision(mujoco_env)
    backend.reset("all")                         # 物块在预设位置, 机械臂 home

    r = verify_grasp("red", backend, camera, detector, vc, control)
    assert r is not True, f"物块在桌上时 verify_grasp 不应为 True(实际 {r})"


def test_verify_release_true_after_release(mujoco_env) -> None:
    """释放后(open_gripper + 抬离) camera2 应确认已放回(True)或无法判定(None)。"""
    backend, camera, detector, vc, control = _setup_vision(mujoco_env)
    backend.reset("all")
    _grab_physically(backend, mujoco_env["model"], control, "red")

    backend.open_gripper()
    assert not backend.is_holding_object()
    # 抬离, 让物块留在桌面
    model = mujoco_env["model"]
    token = CancellationToken()
    move_to_point((0.26, -0.06, 0.1 + control.lift_height_m),
                  backend, model, control, token, make_pacing(control.pacing))

    r = verify_release("red", backend, camera, detector, vc, control)
    assert r is True or r is None, f"释放后 verify_release={r} 不应为 False"


def test_verify_functions_return_none_on_kinematic_backend(env) -> None:
    """无 camera2 的运动学后端: 三值语义返回 None(调用方回退物理验证)。"""
    from ...perception.camera2_verify import _camera2_model

    backend = env["backend"]
    assert _camera2_model(backend, load_vision_config()) is None
    assert verify_grasp("red", backend, None, None, None, env["control"]) is None
    assert verify_release("red", backend, None, None, None, env["control"]) is None
