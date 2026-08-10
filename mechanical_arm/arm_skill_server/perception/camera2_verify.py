"""Phase 6: camera2 第三视角交叉验证(协议 §2.3/§2.4「第二机位摄像头判断」)。

抓取/释放结果的视觉交叉验证: 用 camera2(第三视角)图像 + 视觉检测确认物块是否
真的被夹起(出现在末端附近且末端离桌)/ 已放回(不再出现在末端附近)。

反作弊边界: 验证**只用图像 + 相机标定 + 末端本体位姿**; 不读
backend.get_block_states() 的物块位置(防作弊: 验证也走感知, 不是仿真真值)。

可用性降级(稳定优先):
    - 无 camera2(运动学后端/无 scene_info)  -> 返回 None, 调用方回退物理验证。
    - camera2 取帧失败(渲染线程故障)       -> 返回 None, 回退物理(不因相机故障
      误报一次物理上已成功的抓取/释放失败)。
    - camera2 正常 -> 返回 True/False, 视觉判定参与最终结果。

阈值配置在 control.yaml `grasp_verify_*`(禁止 hardcode)。
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ..config import ControlConfig, VisionConfig
from ..perception.base import PerceptionBackend
from ..perception.camera import CameraModel, CameraProvider
from ..robot.backend import RobotBackend

_log = logging.getLogger(__name__)


def _camera2_model(backend: RobotBackend, vc: VisionConfig) -> Optional[CameraModel]:
    """构造 camera2 的相机标定(camera_id/内参/位姿)。无 camera2 时返回 None。"""
    info = getattr(backend, "scene_info", None)
    T = getattr(info, "camera_world_T", None)
    if T is None or "camera2" not in T:
        return None
    return CameraModel(
        camera_id="camera2",
        width=vc.image_width,
        height=vc.image_height,
        fx=vc.fx, fy=vc.fy,
        cx=vc.image_width / 2.0, cy=vc.image_height / 2.0,
        world_T_cam=T["camera2"],
    )


def _grab_frame(camera: CameraProvider) -> Optional[object]:
    """取 camera2 帧; 失败(相机无 camera2 / 渲染故障)返回 None(调用方回退)。"""
    try:
        return camera.capture(camera_id="camera2")
    except Exception as e:  # noqa: BLE001 - 相机故障不应杀死抓取, 回退物理验证
        _log.warning("camera2 取帧失败, 回退物理验证: %s", e)
        return None


def _color_candidates(frame, color: str, vision: PerceptionBackend) -> list[tuple[float, float, float]]:
    """该色物块的检测候选 [(u, v, confidence)], 按置信度降序。"""
    out = []
    for d in vision.detect_blocks(frame.bgr):
        if d.color == color and d.center_pixel is not None:
            u, v = d.center_pixel
            out.append((float(u), float(v), float(d.confidence)))
    out.sort(key=lambda t: -t[2])
    return out


def verify_grasp(color: str, backend: RobotBackend, camera: CameraProvider,
                 vision: PerceptionBackend, vc: VisionConfig,
                 control: ControlConfig) -> Optional[bool]:
    """camera2 验证抓取: 该色物块在末端附近 且 末端已离桌。

    三值语义(诚实交叉验证, 不误报物理上已成功的抓取):
        True  视觉确认: 物块质心在末端投影像素附近。
        False 视觉反证: 物块可见但不在末端附近(物理说已夹住, 视觉说没夹住)。
        None  无法判定: 无 camera2 / 取帧失败 / 该色物块未被检出(被夹爪/臂遮挡,
              如 white 在近底座位置) -> 调用方回退物理验证。
    """
    m2 = _camera2_model(backend, vc)
    if m2 is None:
        return None
    ee = backend.get_end_effector_pose().position
    if ee[2] < vc.table_top_z_m + control.grasp_verify_lift_z_m:
        return False                      # 末端未离桌, 不可能已抬起
    ee_uv = m2.project(ee)
    if ee_uv is None:
        return None
    frame = _grab_frame(camera)
    if frame is None:
        return None
    cands = _color_candidates(frame, color, vision)
    if not cands:
        return None                       # 未检出 -> 无法确认, 不误报失败
    u, v, _ = cands[0]
    return bool(np.hypot(u - ee_uv[0], v - ee_uv[1]) <= control.grasp_verify_camera2_px_tol)


def verify_release(color: str, backend: RobotBackend, camera: CameraProvider,
                   vision: PerceptionBackend, vc: VisionConfig,
                   control: ControlConfig) -> Optional[bool]:
    """camera2 验证释放: 该色物块已不在末端附近(已放回/移出)。

    True  物块可见且不在末端附近(确认已放回), 或未检出(已移出视野, 物理已验证)。
    False 物块可见且仍在末端附近(未释放成功)。
    None  无法判定(无 camera2 / 取帧失败) -> 回退物理。
    """
    m2 = _camera2_model(backend, vc)
    if m2 is None:
        return None
    ee = backend.get_end_effector_pose().position
    ee_uv = m2.project(ee)
    if ee_uv is None:
        return None
    frame = _grab_frame(camera)
    if frame is None:
        return None
    cands = _color_candidates(frame, color, vision)
    if not cands:
        return True                       # 未检出该色 -> 已不在夹爪(接受)
    u, v, _ = cands[0]
    return not bool(np.hypot(u - ee_uv[0], v - ee_uv[1])
                    <= control.grasp_verify_camera2_px_tol)


def camera2_check_enabled(control: ControlConfig) -> bool:
    """总开关: 配置启用 且 已注入视觉(camera/vision/vc)。"""
    if not control.grasp_verify_camera2:
        return False
    from ..runtime.state import state
    return (state.camera is not None and state.vision is not None
            and state.vision_cfg is not None)
