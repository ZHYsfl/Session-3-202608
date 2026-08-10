"""HSVColorDetector 单元测试: 三色检出 / 缺色 / 随机位姿 / 光照 / 边缘 / 防作弊。

所有图像来自 SyntheticCamera(合成 fixture)或手工绘制的 ndarray。检测器只吃图像,
从不接触场景 ground truth —— 防作弊测试专门验证这一点。
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from ...config import load_simulation_config, load_vision_config
from ...perception.camera import CameraModel
from ...perception.hsv_color_detector import HSVColorDetector
from ...perception.synthetic_camera import SyntheticCamera
from ...simulation.blocks import Block
from ...simulation.environment import Scene

BLOCK_Z = 0.015           # 桌面上的物块中心 z = table_top + half_height


def make_scene(sim_cfg: dict, blocks: list[Block]) -> Scene:
    """用任意物块构造场景(直接把 blocks 赋给 scene.blocks)。"""
    scene = Scene(sim_cfg)
    scene.blocks = list(blocks)
    return scene


def draw_square(img: np.ndarray, color_bgr, cx: int, cy: int, half: int = 25) -> None:
    """在图像上画一个实心方块(手工构造输入用, 与任何场景无关)。"""
    cv2.rectangle(img, (int(cx - half), int(cy - half)),
                  (int(cx + half), int(cy + half)), color_bgr, -1)


@pytest.fixture(scope="module")
def vc():
    return load_vision_config()


@pytest.fixture(scope="module")
def sim_cfg():
    return load_simulation_config()


@pytest.fixture(scope="module")
def camera(vc, sim_cfg):
    return SyntheticCamera(make_scene(sim_cfg, []), vc)


@pytest.fixture(scope="module")
def pure_detector(vc):
    """纯 2D 检测器: 不携带 camera_model, world_position 恒为 None。"""
    return HSVColorDetector(vc)


@pytest.fixture(scope="module")
def world_detector(vc, camera):
    """带标定的检测器: 用 像素+相机模型+桌面平面 填充 world_position。"""
    return HSVColorDetector(vc, camera_model=camera.camera_model)


# ---------------------------------------------------------------------------
# 三色检出
# ---------------------------------------------------------------------------

def test_hsv_detect_red_yellow_white(vc, sim_cfg):
    """all 场景: 三色各检出一次, 中心像素与 world 均接近 ground truth。"""
    from ...simulation.blocks import load_preset
    scene = make_scene(sim_cfg, load_preset(sim_cfg, "all"))
    cam = SyntheticCamera(scene, vc)
    det = HSVColorDetector(vc, camera_model=cam.camera_model)
    dets = det.detect_blocks(cam.capture().bgr)

    found = {d.color: d for d in dets}
    assert set(found) == {"red", "yellow", "white"}, f"检出颜色 {set(found)}"

    for b in scene.get_block_states():
        d = found[b.color]
        uv = cam.camera_model.project((b.x, b.y, 0.0))
        assert abs(d.center_pixel[0] - uv[0]) < 3.0, f"{b.color} u 偏差大"
        assert abs(d.center_pixel[1] - uv[1]) < 3.0, f"{b.color} v 偏差大"
        assert abs(d.world_position[0] - b.x) < 0.01, f"{b.color} x 误差大"
        assert abs(d.world_position[1] - b.y) < 0.01, f"{b.color} y 误差大"
        assert d.confidence >= 0.4


# ---------------------------------------------------------------------------
# 缺色 / 多色
# ---------------------------------------------------------------------------

def test_hsv_absent_color(world_detector, vc, sim_cfg):
    """red_only 场景: 缺 yellow/white, 必须一个都不检出。"""
    from ...simulation.blocks import load_preset
    scene = make_scene(sim_cfg, load_preset(sim_cfg, "red_only"))
    cam = SyntheticCamera(scene, vc)
    dets = world_detector.detect_blocks(cam.capture().bgr)
    colors = {d.color for d in dets}
    assert "red" in colors
    assert "yellow" not in colors
    assert "white" not in colors


def test_hsv_multiple_colors_close_together(world_detector, vc, sim_cfg):
    """两种颜色紧挨着(约 4cm 间距, 与物块边长同级): 应各自分开检出。"""
    scene = make_scene(sim_cfg, [
        Block("red", 0.18, -0.02, BLOCK_Z, 0.0),
        Block("yellow", 0.19, 0.02, BLOCK_Z, 0.5),
    ])
    cam = SyntheticCamera(scene, vc)
    dets = world_detector.detect_blocks(cam.capture().bgr)
    assert {d.color for d in dets} == {"red", "yellow"}


def test_hsv_multiple_blocks_same_color_best_confidence(world_detector, vc, sim_cfg):
    """同颜色出现多个(非协议场景, 但检测器应返回多个), 保底不抛。"""
    scene = make_scene(sim_cfg, [
        Block("red", 0.15, -0.05, BLOCK_Z, 0.0),
        Block("red", 0.25, 0.05, BLOCK_Z, 0.3),
    ])
    cam = SyntheticCamera(scene, vc)
    dets = [d for d in world_detector.detect_blocks(cam.capture().bgr) if d.color == "red"]
    assert len(dets) == 2


# ---------------------------------------------------------------------------
# 随机位姿 / 光照 / 边缘
# ---------------------------------------------------------------------------

def test_hsv_random_positions(world_detector, vc, sim_cfg):
    """随机位姿(含随机偏航): 每次都应检出且 world 误差 < 1.5cm。"""
    rng = np.random.default_rng(2026)
    for _ in range(30):
        x = float(rng.uniform(0.12, 0.30))
        y = float(rng.uniform(-0.10, 0.10))
        yaw = float(rng.uniform(-np.pi, np.pi))
        scene = make_scene(sim_cfg, [Block("red", x, y, BLOCK_Z, yaw)])
        cam = SyntheticCamera(scene, vc)
        dets = [d for d in world_detector.detect_blocks(cam.capture().bgr)
                if d.color == "red"]
        assert len(dets) == 1, f"({x:.3f},{y:.3f},yaw={yaw:.2f}) 检出 {len(dets)} 个"
        d = dets[0]
        assert abs(d.world_position[0] - x) < 0.015, f"x 误差 {(d.world_position[0]-x)*1000:.1f}mm"
        assert abs(d.world_position[1] - y) < 0.015, f"y 误差 {(d.world_position[1]-y)*1000:.1f}mm"


def test_hsv_lighting_variation(world_detector, vc, sim_cfg):
    """光照变化(0.8x / 1.2x 亮度): 三色仍全部检出。"""
    from ...simulation.blocks import load_preset
    scene = make_scene(sim_cfg, load_preset(sim_cfg, "all"))
    cam = SyntheticCamera(scene, vc)
    for brightness in (0.8, 1.2):
        dets = world_detector.detect_blocks(cam.capture(brightness=brightness).bgr)
        assert {d.color for d in dets} == {"red", "yellow", "white"}, \
            f"brightness={brightness} 检出 { {d.color for d in dets} }"


def test_hsv_edge_positions(world_detector, vc, sim_cfg):
    """工作空间边缘位姿: 应检出; 视野外物块应正确地不检出。"""
    for x, y in ((0.12, -0.10), (0.30, 0.10), (0.12, 0.10), (0.30, -0.10)):
        scene = make_scene(sim_cfg, [Block("yellow", x, y, BLOCK_Z, 0.7)])
        cam = SyntheticCamera(scene, vc)
        dets = [d for d in world_detector.detect_blocks(cam.capture().bgr)
                if d.color == "yellow"]
        assert len(dets) == 1, f"边缘位姿 ({x},{y}) 应检出"
    # 视野外: 物块 (0.80, 0.80) 远在画面外 -> 不应检出
    scene = make_scene(sim_cfg, [Block("red", 0.80, 0.80, BLOCK_Z, 0.0)])
    cam = SyntheticCamera(scene, vc)
    dets = world_detector.detect_blocks(cam.capture().bgr)
    assert not dets, f"视野外物块不应检出, 实际 {dets}"


# ---------------------------------------------------------------------------
# 白色特判: 灰色桌面不误检
# ---------------------------------------------------------------------------

def test_white_does_not_false_positive_on_gray_table(world_detector, camera, vc):
    """空桌面(灰)上没有任何物块: 不得检出 white(也不得检出其它颜色)。"""
    dets = world_detector.detect_blocks(camera.capture().bgr)
    assert dets == []


# ---------------------------------------------------------------------------
# 防作弊: 检测器只用图像, 不读场景 ground truth
# ---------------------------------------------------------------------------

def test_detector_does_not_use_ground_truth(vc, sim_cfg):
    """手工绘制图像(red 在图像 (100,100)): 检测器必须按图像内容检出,
    而不得去查场景里 red 的真实位置(0.18, -0.05)。"""
    scene = make_scene(sim_cfg, [Block("red", 0.18, -0.05, BLOCK_Z, 0.0)])
    cam = SyntheticCamera(scene, vc)

    img = np.full((vc.image_height, vc.image_width, 3),
                  vc.table_color_bgr, dtype=np.uint8)
    draw_square(img, vc.block_colors_bgr["red"], cx=100, cy=100, half=25)

    pure = HSVColorDetector(vc)                  # 无 camera_model: 纯 2D
    dets = pure.detect_blocks(img)
    assert len(dets) == 1 and dets[0].color == "red"
    assert abs(dets[0].center_pixel[0] - 100) < 2
    assert abs(dets[0].center_pixel[1] - 100) < 2
    assert dets[0].world_position is None        # 没有标定就不编造 3D

    # 场景 ground truth 的投影像素与检测中心必须不同(证明按图像, 不按场景)
    gt_uv = cam.camera_model.project((0.18, -0.05, 0.0))
    assert abs(dets[0].center_pixel[0] - gt_uv[0]) > 20


def test_world_position_comes_from_pixel_and_calibration(vc, sim_cfg):
    """world_position 由「像素 + 相机标定 + 桌面平面」得出, 不是场景 ground truth。"""
    scene = make_scene(sim_cfg, [Block("red", 0.18, -0.05, BLOCK_Z, 0.0)])
    cam = SyntheticCamera(scene, vc)
    det = HSVColorDetector(vc, camera_model=cam.camera_model)

    img = np.full((vc.image_height, vc.image_width, 3),
                  vc.table_color_bgr, dtype=np.uint8)
    draw_square(img, vc.block_colors_bgr["red"], cx=100, cy=100, half=25)

    d = det.detect_blocks(img)[0]
    expected = cam.camera_model.pixel_to_world_on_plane(100.0, 100.0, vc.table_top_z_m)
    assert d.world_position is not None
    assert abs(d.world_position[0] - expected[0]) < 0.01
    assert abs(d.world_position[1] - expected[1]) < 0.01
    # 场景里 red 的真实位置 (0.18, -0.05) 不应是检测结果
    assert abs(d.world_position[0] - 0.18) > 0.03 or abs(d.world_position[1] + 0.05) > 0.03


# ---------------------------------------------------------------------------
# ROI / 配置
# ---------------------------------------------------------------------------

def test_hsv_roi_limits_detection(vc, sim_cfg):
    """ROI 配置生效: 只检测 ROI 内的物块, bbox 坐标仍回到图像完整坐标系。

    相机 x 轴指向世界 -y 方向 -> 图像 u 编码世界 y: y<0 投到 u>320, y>0 投到 u<320。
    """
    # ROI 设为图像右半部分
    vc_roi = load_vision_config()
    vc_roi.roi = (320, 0, 640, 480)
    det = HSVColorDetector(vc_roi)

    scene = make_scene(sim_cfg, [
        Block("red", 0.18, -0.08, BLOCK_Z, 0.0),      # y<0 -> u>320(在 ROI 内)
        Block("yellow", 0.18, 0.08, BLOCK_Z, 0.0),    # y>0 -> u<320(在 ROI 外)
    ])
    cam = SyntheticCamera(scene, vc)

    # 先确认两物块确实分居 ROI 边界两侧(防测试场景失效)
    uv_red = cam.camera_model.project((0.18, -0.08, 0.0))
    uv_yellow = cam.camera_model.project((0.18, 0.08, 0.0))
    assert uv_red[0] > 320.0, f"red 应投影到 u>320, 实际 {uv_red}"
    assert uv_yellow[0] + 25 < 320.0, f"yellow 应投影到 u<320, 实际 {uv_yellow}"

    dets = det.detect_blocks(cam.capture().bgr)
    assert {d.color for d in dets} == {"red"}, f"ROI 外 yellow 不应检出, 实际 {dets}"
    assert all(d.bbox_xyxy[0] >= 320 for d in dets)   # bbox 在图像完整坐标系下


def test_vision_config_red_dual_range(vc):
    """红色跨 H=0 边界 -> 配置里必须有双区间; 其它颜色单区间。"""
    assert len(vc.hsv_ranges["red"]) == 2
    assert len(vc.hsv_ranges["yellow"]) == 1
    assert len(vc.hsv_ranges["white"]) == 1
    assert all(0 <= r.lower[0] <= r.upper[0] <= 180 for rngs in vc.hsv_ranges.values()
               for r in rngs)
