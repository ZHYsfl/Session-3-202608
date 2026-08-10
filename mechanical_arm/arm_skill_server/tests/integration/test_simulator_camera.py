"""SimulatorCamera 验证(Phase 3): 渲染器相机 + HSV 检测在真渲染帧上的表现。

标记 @pytest.mark.simulator -> 快速回归 `pytest -m "not simulator"` 不执行。

覆盖(对应 Phase 3 视觉标定的关键 bug, 见 docs/progress/phase3_report.md):
    A. 相机约定: 渲染帧里, 物块 GT 世界坐标投影处能看到对应颜色像素。
       验证 cam_xpos/cam_xmat 的「perception 约定 world_T_cam 列=[x,y,z]」写法。
    B. 渲染帧 HSV 检测: 三色物块都被 HSVColorDetector 以正确世界位置检出
       (xy 误差 < 2cm), 无多余分量。
    C. 白物块 vs 桌面/机身: 白 HSV 掩码只命中白物块, 不误检深蓝灰桌面/机身
       (修复前: 浅灰桌面/机身 V≈247 落入 [0,0,180]-[180,50,255], 白掩码 28 万像素)。
    D. 机械臂贴近物块(下降姿态)时检测仍然可靠(机身不产生白色误检)。
    E. camera2 第三视角也能取帧渲染(抓取/释放交叉验证用)。
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from ...perception.hsv_color_detector import HSVColorDetector
from ...perception.simulator_camera import SimulatorCamera
from ...robot.controller import make_pacing
from ...runtime.cancellation import CancellationToken
from ...skills._motion import move_to_point

pytestmark = pytest.mark.simulator

_GT = {"red": (0.26, -0.06), "yellow": (0.30, 0.05), "white": (0.24, 0.06)}
_XY_TOL_M = 0.02          # 检测世界位置误差上限(HSV 标定: 实际 ≤ 6mm)


def _make_camera_detector(mujoco_env):
    """从 mujoco_env 构造 SimulatorCamera + HSVColorDetector(不依赖全局 state)。

    mujoco_env fixture 的后端 __init__ 默认 reset("no_blocks"), 这里显式铺满
    三色物块, 供相机/检测测试使用。
    """
    from ...config import load_vision_config

    backend = mujoco_env["backend"]
    backend.reset("all")
    vision_cfg = load_vision_config()
    camera = SimulatorCamera(backend, vision_cfg)
    detector = HSVColorDetector(vision_cfg, camera_model=camera.camera_model)
    return camera, detector, vision_cfg


def _xy_err(det, color: str) -> float | None:
    """返回某颜色最佳候选的世界 xy 误差; 未检出返回 None。"""
    cands = [d for d in det if d.color == color and d.world_position is not None]
    if not cands:
        return None
    d = max(cands, key=lambda d: d.confidence)
    gx, gy = _GT[color]
    return float(np.hypot(d.world_position[0] - gx, d.world_position[1] - gy))


def _pacing(control):
    return make_pacing(control.pacing)


def test_camera_convention_renders_blocks_at_gt_pixels(mujoco_env):
    """渲染帧里, 物块 GT 世界坐标投影处出现该颜色像素(相机位姿约定正确)。

    若 world_T_cam 写错(列序/符号), 投影点会落在其它位置 -> 采样到桌面/机身色。
    """
    camera, _, vision_cfg = _make_camera_detector(mujoco_env)
    frame = camera.capture()
    hsv = cv2.cvtColor(frame.bgr, cv2.COLOR_BGR2HSV)
    cam_model = camera.camera_model
    for color, (gx, gy) in _GT.items():
        uv = cam_model.project((gx, gy, vision_cfg.block_half_height_m))
        assert uv is not None, f"{color} GT 应在视野内"
        u, v = int(round(uv[0])), int(round(uv[1]))
        H, S, V = hsv[v, u]
        # 红色/黄色 S 高; 白色 V 高且 S 低
        if color == "white":
            assert V >= 180, f"white 物块顶部 V={V} 应≥180"
        else:
            assert S >= 80, f"{color} 物块顶部 S={S} 应≥80"


def test_rendered_hsv_detects_all_colors(mujoco_env):
    """HSVColorDetector 在真渲染帧上检出三色物块, 世界位置误差 < 2cm, 无多余分量。"""
    camera, detector, _ = _make_camera_detector(mujoco_env)
    dets = detector.detect_blocks(camera.capture().bgr)
    for color in _GT:
        err = _xy_err(dets, color)
        assert err is not None, f"{color} 应被检出"
        assert err < _XY_TOL_M, f"{color} xy_err={err:.3f}m 超 2cm"
        n = len([d for d in dets if d.color == color])
        assert n == 1, f"{color} 出现 {n} 个分量(应为 1)"


def test_white_mask_not_fired_by_table_or_arm(mujoco_env):
    """白物块 HSV 掩码只命中白物块, 不误检桌面/机身(修复 Phase 3 白色误检 bug)。"""
    camera, detector, vision_cfg = _make_camera_detector(mujoco_env)
    frame = camera.capture()
    hsv = cv2.cvtColor(frame.bgr, cv2.COLOR_BGR2HSV)
    mask = detector.build_mask(hsv, "white")
    # 桌面+机身(深蓝灰 S=59..79)不得进入白范围: 掩码只该有白物块那一片
    assert (mask > 0).sum() < 10000, \
        f"白掩码 {int((mask > 0).sum())}px 应≈白物块一片(标定 ~2200px), 桌面/机身误检"
    # 最大分量中心应落在白物块 GT 投影附近
    n, _, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cam_model = camera.camera_model
    g_uv = cam_model.project((*_GT["white"], vision_cfg.block_half_height_m))
    assert g_uv is not None
    best = max(range(1, n), key=lambda i: stats[i, cv2.CC_STAT_AREA])
    cx, cy = cents[best]
    dist = float(np.hypot(cx - g_uv[0], cy - g_uv[1]))
    assert dist < 20, f"白掩码主分量中心 ({cx:.0f},{cy:.0f}) 距白物块投影 {dist:.0f}px"


def test_detection_with_hand_at_descent_over_cube(mujoco_env):
    """机械臂降到物块正上方(机身贴近相机)时, 三色检测仍然可靠、无白色误检。"""
    b = mujoco_env["backend"]
    model, control = mujoco_env["model"], mujoco_env["control"]
    b.reset("all")
    camera, detector, vision_cfg = _make_camera_detector(mujoco_env)
    token = CancellationToken()
    wrist = b.grasp_wrist_roll
    b.open_gripper()
    gx, gy = _GT["white"]
    move_to_point((gx, gy, 0.015 + control.approach_height_m),
                  b, model, control, token, _pacing(control), force_wrist_roll=wrist)
    move_to_point((gx, gy, 0.015 + control.descent_offset_m),
                  b, model, control, token, _pacing(control), force_wrist_roll=wrist)

    dets = detector.detect_blocks(camera.capture().bgr)
    for color in _GT:
        err = _xy_err(dets, color)
        assert err is not None, f"手在物块上方时 {color} 应仍被检出"
        assert err < _XY_TOL_M, f"手在物块上方时 {color} xy_err={err:.3f}m"


def test_capture_from_worker_thread(mujoco_env):
    """跨线程取帧必须有效(渲染线程持有 GL 上下文)。

    REST(uvicorn/TestClient)在线程池里跑 handler, 若 Renderer 在主线程创建、
    在 worker 线程 render, GLFW 在 Windows 无法切上下文 -> 黑帧(实测 mean≈0.1,
    检测全丢)。修复: SimulatorCamera 用专用渲染线程, 任意线程 capture 都有效。
    """
    import threading

    camera, detector, _ = _make_camera_detector(mujoco_env)
    results = {}

    def worker():
        f = camera.capture()
        dets = detector.detect_blocks(f.bgr)
        results["mean"] = float(f.bgr.mean())
        results["reds"] = len([d for d in dets if d.color == "red"])

    threads = [threading.Thread(target=worker) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results["mean"] > 100, f"worker 线程取帧应为正常画面, mean={results['mean']:.1f}"
    assert results["reds"] == 1, "worker 线程取帧应能检出 red"


def test_camera2_renders_third_person_view(mujoco_env):
    """camera2 第三视角取帧正常, 渲染帧非纯色、含机械臂(抓取/释放验证用)。"""
    camera, _, _ = _make_camera_detector(mujoco_env)
    frame = camera.capture(camera_id="camera2")
    assert frame.camera_id == "camera2"
    img = frame.bgr
    # 场景应包含桌面(亮)与机身(暗): 灰度方差显著, 不是纯色/黑帧
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    assert float(gray.std()) > 20, f"camera2 帧应含明显结构, std={gray.std():.1f}"
    assert float(gray.mean()) > 40, f"camera2 帧不应近黑, mean={gray.mean():.1f}"
