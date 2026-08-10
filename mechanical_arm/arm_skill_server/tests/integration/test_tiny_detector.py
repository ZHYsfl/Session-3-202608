"""Phase 4 TinyDetector(自训练微型 CNN)验证。

标记 @pytest.mark.simulator -> 快速回归 `pytest -m "not simulator"` 不执行。

覆盖:
    A. 纯函数: probs_to_detections 在无 torch 的情况下把 per-cell 概率图转成
       BlockDetection(供单测; 不需要 checkpoint)。
    B. 训练模型在演示布局(与 Phase 3 GT 一致)上检出三色物块。
    C. 端到端: perception_mode="tiny_detector" 下 grab_the_block 真实夹住物块
       (自训练模型 -> 图像感知 -> 像素投影 -> 物理夹取)。

依赖: mujoco(Phase 3) + torch(Phase 4) + 训练产物 models/tiny_detector.pt
(checkpoint 缺失时跳过, 提示先运行 generate/train 脚本)。
"""
from __future__ import annotations

import numpy as np
import pytest

from ...config import ControlConfig, VisionConfig, load_all, load_vision_config
from ...perception.tiny_detector import probs_to_detections
from ...queue.in_memory_queue import InMemoryQueue
from ...runtime.state import state
from ...skills.grab_block import _GRASP_OK, grab_the_block

pytestmark = pytest.mark.simulator

_XY_TOL_M = 0.02


def _vision_cfg():
    vc = load_vision_config()
    return vc


def test_probs_to_detections_pure():
    """per-cell 概率图 -> BlockDetection(纯函数, 无 torch / checkpoint)。"""
    vc = _vision_cfg()
    GW, GH = vc.tiny_grid_width, vc.tiny_grid_height
    probs = np.zeros((GH, GW, 4), dtype=np.float32)
    probs[..., 0] = 0.05                          # background 打底
    # red 物块占 cells (v=15..17, u=20..22)
    probs[15:18, 20:23, 1] = 0.9
    probs[15:18, 20:23, 0] = 0.1
    dets = probs_to_detections(probs, (0, 0, 640, 480), (480, 640), None, vc)
    reds = [d for d in dets if d.color == "red"]
    assert len(reds) == 1, f"应有 1 个 red 分量, 实际 {len(reds)}"
    u, v = reds[0].center_pixel
    assert abs(u - (21.0 + 0.5) / GW * 640) < 1.0, f"u={u:.1f}"
    assert abs(v - (16.0 + 0.5) / GH * 480) < 1.0, f"v={v:.1f}"
    # 无 camera_model -> world_position 为 None(纯像素定位)
    assert reds[0].world_position is None
    assert all(d.color != "yellow" and d.color != "white" for d in dets)


def test_tiny_detector_on_demo_layout(mujoco_env):
    """训练模型在演示布局上检出三色物块(世界位置误差 < 2cm)。"""
    from ...perception.simulator_camera import SimulatorCamera
    from ...perception.tiny_detector import TinyDetector

    checkpoint = _project_root() / "models" / "tiny_detector.pt"
    if not checkpoint.exists():
        pytest.skip("TinyDetector checkpoint 缺失: 先运行 "
                    "scripts/generate_tiny_dataset.py && scripts/train_tiny_detector.py")

    vc = _vision_cfg()
    b = mujoco_env["backend"]
    b.reset("all")
    camera = SimulatorCamera(b, vc)
    det = TinyDetector(vc, camera_model=camera.camera_model)

    gt = {"red": (0.26, -0.06), "yellow": (0.30, 0.05), "white": (0.24, 0.06)}
    dets = det.detect_blocks(camera.capture().bgr)
    for color, (gx, gy) in gt.items():
        cands = [d for d in dets if d.color == color and d.world_position is not None]
        assert cands, f"{color} 应被 TinyDetector 检出"
        best = max(cands, key=lambda d: d.confidence)
        err = float(np.hypot(best.world_position[0] - gx, best.world_position[1] - gy))
        assert err < _XY_TOL_M, f"{color} xy_err={err:.3f}m 超 2cm"


def test_grab_with_tiny_detector_mode(mujoco_env):
    """端到端: perception_mode="tiny_detector" 下 grab_the_block 真实夹住物块。

    目标坐标完全来自 自训练模型 图像感知 -> 像素投影, 不是 ground truth。
    """
    from ...perception.simulator_camera import SimulatorCamera
    from ...perception.tiny_detector import TinyDetector

    checkpoint = _project_root() / "models" / "tiny_detector.pt"
    if not checkpoint.exists():
        pytest.skip("TinyDetector checkpoint 缺失: 先运行训练脚本")

    vc = _vision_cfg()
    b = mujoco_env["backend"]
    model, sim_cfg = mujoco_env["model"], mujoco_env["sim_cfg"]
    control = ControlConfig(**{**mujoco_env["control"].__dict__,
                               "perception_mode": "tiny_detector"})
    camera = SimulatorCamera(b, vc)
    det = TinyDetector(vc, camera_model=camera.camera_model)
    queue = InMemoryQueue()
    state.configure(backend=b, queue=queue, control=control, model=model,
                    vision=det, camera=camera, vision_cfg=vc)

    for color, preset in (("red", "all"), ("yellow", "all"), ("white", "all")):
        b.reset(preset)
        assert grab_the_block(color) == _GRASP_OK, f"tiny_detector 模式应夹住 {color}"
        assert b.is_holding_object(), f"夹取 {color} 后必须真的夹着物块"
        _, _, z = b.cube_position(color)
        assert z > 0.05, f"{color} 应被抬离桌面: z={z:.3f}"


def _project_root():
    from pathlib import Path
    return Path(__file__).resolve().parents[3]
