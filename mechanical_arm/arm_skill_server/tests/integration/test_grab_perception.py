"""grab_the_block 感知接入: hsv 模式走 相机->检测->投影->抓取; ground_truth 不变。

关键断言:
    1. 非法颜色不调用相机/检测器(协议 §2.3 第 1 步前置)。
    2. hsv 模式不读 backend.get_block_states()(防作弊)。
    3. 协议返回字符串逐字不变。
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from ...perception.camera import ImageFrame
from ...runtime.state import state
from ...skills.grab_block import _GRASP_FAIL, _GRASP_OK, _NO_COLOR, grab_the_block

BLOCK_Z = 0.015


def draw_square(img: np.ndarray, color_bgr, cx: int, cy: int, half: int = 25) -> None:
    """手工绘制实心方块(与场景无关的图像输入)。"""
    cv2.rectangle(img, (int(cx - half), int(cy - half)),
                  (int(cx + half), int(cy + half)), color_bgr, -1)


# ---------------------------------------------------------------------------
# hsv 模式端到端
# ---------------------------------------------------------------------------

def test_grab_hsv_uses_perception_end_to_end(env_hsv):
    """相机->HSV->投影->抓取 全链路: red 物块被视觉定位并真正夹住。"""
    backend = env_hsv["backend"]
    backend.reset("all")
    result = grab_the_block("red")
    assert result == _GRASP_OK
    assert backend.is_holding_object()


def test_grab_hsv_missing_visible_block(env_hsv, monkeypatch):
    """画面里没有该颜色 -> _NO_COLOR; 且证明确实走了视觉(相机被调用一次)。"""
    backend = env_hsv["backend"]
    backend.reset("red_only")                    # 场景只有 red
    calls = {"capture": 0}
    orig_capture = env_hsv["camera"].capture

    def spy_capture(*a, **k):
        calls["capture"] += 1
        return orig_capture(*a, **k)

    monkeypatch.setattr(env_hsv["camera"], "capture", spy_capture)
    result = grab_the_block("yellow")
    assert result == _NO_COLOR
    assert calls["capture"] == 1                 # 走的是视觉, 不是直接读场景


def test_grab_hsv_block_outside_view_returns_no_color(env_hsv):
    """物块在相机视野外 -> 诚实报「没有这种颜色的物块」(与 ground_truth 的
    「夹取失败」语义不同: 视野外不可感知, 就是没有)。"""
    backend = env_hsv["backend"]
    backend.reset("unreachable_block")           # red 在 (0.80, 0.80), 画面外
    assert grab_the_block("red") == _NO_COLOR


# ---------------------------------------------------------------------------
# 非法颜色: 不调用相机/检测器
# ---------------------------------------------------------------------------

def test_invalid_color_skips_camera(env_hsv, monkeypatch):
    """grab('blue'): 颜色校验在第一步返回, camera.capture 与 vision.detect 调用数 == 0。"""
    calls = {"capture": 0, "detect": 0}

    class SpyCamera:
        camera_model = env_hsv["camera"].camera_model

        def capture(self):
            calls["capture"] += 1
            return None                          # 不应被调用

    class SpyVision:
        def detect_blocks(self, image):
            calls["detect"] += 1
            return []

    monkeypatch.setattr(state, "camera", SpyCamera())
    monkeypatch.setattr(state, "vision", SpyVision())

    result = grab_the_block("blue")
    assert result == _NO_COLOR
    assert calls["capture"] == 0
    assert calls["detect"] == 0


# ---------------------------------------------------------------------------
# 防作弊: hsv 模式绝不读 ground truth, 末端去图像指示的位置
# ---------------------------------------------------------------------------

def test_grab_hsv_does_not_read_ground_truth(env_hsv, monkeypatch):
    """场景里 GT red 在 (0.26, -0.06); 相机返回一张 red 画在别处的图像。

    机械臂必须去「图像指示的位置」, 且不调用 backend.get_block_states()。
    证明抓取用的是感知, 不是仿真状态。
    """
    backend = env_hsv["backend"]
    backend.reset("all")
    vc = env_hsv["vision_cfg"]
    cam_model = env_hsv["camera"].camera_model

    # 图像指示的位置必须是当前场景(all 预设)里**真正空**的位置: 与每个物块的
    # 距离都大于 holding_proximity(0.04), 否则 kin 后端 close_gripper 会按距离
    # 抓到旁边的白块(Phase 3 物块移到 x>=0.22 后, 原 (0.25,0.05) 距 white
    # (0.24,0.06) 仅 0.021 -> 误抓, 见 _mj_probe_target.py)。(0.26,0.12) 距最近
    # 物块 0.065, 且与 GT red(0.26,-0.06) 明显不同。
    target_xy = (0.26, 0.12)
    uv = cam_model.project((target_xy[0], target_xy[1], 0.0))
    img = np.full((vc.image_height, vc.image_width, 3),
                  vc.table_color_bgr, dtype=np.uint8)
    draw_square(img, vc.block_colors_bgr["red"], int(uv[0]), int(uv[1]), half=25)

    calls = {"capture": 0, "block_states": 0}

    class FakeCamera:
        camera_model = cam_model

        def capture(self):
            calls["capture"] += 1
            return ImageFrame(camera_id=vc.camera_id, bgr=img, timestamp=0.0)

    def spy_block_states():
        calls["block_states"] += 1
        return backend.get_block_states()

    monkeypatch.setattr(state, "camera", FakeCamera())
    monkeypatch.setattr(backend, "get_block_states", spy_block_states)

    result = grab_the_block("red")

    # 绝不读 ground truth, 相机只取一帧
    assert calls["block_states"] == 0
    assert calls["capture"] == 1
    # 图像位置没有真物块 -> 诚实报失败
    assert result == _GRASP_FAIL
    # 末端到了图像指示的位置, 而不是场景 GT(0.26, -0.06)
    ee = backend.get_end_effector_pose().position
    assert abs(ee[0] - target_xy[0]) < 0.03, f"EE x={ee[0]:.3f} 应≈{target_xy[0]}"
    assert abs(ee[1] - target_xy[1]) < 0.03, f"EE y={ee[1]:.3f} 应≈{target_xy[1]}"
    assert not (abs(ee[0] - 0.26) < 0.03 and abs(ee[1] + 0.06) < 0.03), \
        "末端不应停在场景 ground truth 位置"


# ---------------------------------------------------------------------------
# ground_truth 模式保持 Phase 1 行为
# ---------------------------------------------------------------------------

def test_grab_ground_truth_skips_camera(env, monkeypatch):
    """ground_truth 模式: 直接读物块状态, 不碰相机。"""
    backend = env["backend"]
    backend.reset("all")
    calls = {"capture": 0}

    class SpyCamera:
        camera_model = None

        def capture(self):
            calls["capture"] += 1
            return None

    monkeypatch.setattr(state, "camera", SpyCamera())
    result = grab_the_block("red")
    assert result == _GRASP_OK
    assert backend.is_holding_object()
    assert calls["capture"] == 0


# ---------------------------------------------------------------------------
# API 层: 感知抓取经完整网关返回
# ---------------------------------------------------------------------------

def test_api_grab_hsv_end_to_end(env_hsv):
    """hsv 模式经 REST API: 返回协议字符串, 且真的夹住物块。"""
    from fastapi.testclient import TestClient

    from ...api.server import create_app

    backend = env_hsv["backend"]
    backend.reset("all")
    app = create_app(backend, env_hsv["queue"], env_hsv["control"], env_hsv["model"],
                     vision=env_hsv["vision"], camera=env_hsv["camera"],
                     vision_cfg=env_hsv["vision_cfg"])
    client = TestClient(app)
    r = client.post("/api/v1/grab_the_block", json={"color": "red"})
    assert r.status_code == 200
    assert r.json()["result"] == "有这种颜色的物块，且夹取物块成功"
    assert backend.is_holding_object()
