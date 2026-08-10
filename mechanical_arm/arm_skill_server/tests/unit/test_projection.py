"""像素 <-> 世界投影数学: 针孔模型 + 射线与桌面平面求交。

坐标框架(与 docs/perception.md 一致):
    世界系 = base frame(z 向上, 桌面 z=0); 相机系 +z 朝前; 图像系 (u,v) 左上原点。
"""
from __future__ import annotations

import numpy as np
import pytest

from ...config import load_vision_config
from ...perception import projection


@pytest.fixture(scope="module")
def vc():
    return load_vision_config()


@pytest.fixture(scope="module")
def pose(vc):
    return projection.build_camera_pose(vc.camera_position, vc.camera_look_at,
                                        vc.camera_up)


def test_build_camera_pose_orthonormal(pose, vc):
    """相机位姿的旋转部分必须正交、右手系, 平移部分 = 光心位置。"""
    R = pose[:3, :3]
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-12)
    assert np.allclose(pose[:3, 3], vc.camera_position, atol=1e-12)
    # 相机 +z 指向注视点
    axis = pose[:3, 2]
    toward = np.asarray(vc.camera_look_at) - np.asarray(vc.camera_position)
    assert np.dot(axis, toward) > 0.0


def test_look_at_projects_to_principal_point(pose, vc):
    """注视点位于光轴上 -> 应投到主点 (cx, cy)。"""
    cam = _model(vc, pose)
    uv = cam.project(vc.camera_look_at)
    assert uv is not None
    assert np.isclose(uv[0], vc.image_width / 2.0, atol=0.5)
    assert np.isclose(uv[1], vc.image_height / 2.0, atol=0.5)


def test_plane_roundtrip_exact(vc, pose):
    """桌面平面 z=0 上任意点: 投影 -> 反投影回 z=0 应精确恢复(误差 ~1e-9 m)。"""
    cam = _model(vc, pose)
    rng = np.random.default_rng(7)
    for _ in range(50):
        x = float(rng.uniform(0.10, 0.30))
        y = float(rng.uniform(-0.10, 0.10))
        uv = cam.project((x, y, 0.0))
        assert uv is not None, f"({x}, {y}) 应可见"
        back = cam.pixel_to_world_on_plane(uv[0], uv[1], plane_z=0.0)
        assert back is not None
        assert abs(back[0] - x) < 1e-6
        assert abs(back[1] - y) < 1e-6


def test_pixel_grid_reprojects(vc, pose):
    """FOV 内像素网格 -> 世界 -> 再投影, 应回到同一像素(往返误差 < 0.01px)。"""
    cam = _model(vc, pose)
    for u in (100.0, 320.0, 540.0):
        for v in (100.0, 240.0, 380.0):
            w = cam.pixel_to_world_on_plane(u, v, plane_z=0.0)
            assert w is not None, f"pixel ({u}, {v}) 无交点"
            uv2 = cam.project((w[0], w[1], 0.0))
            assert uv2 is not None
            assert abs(uv2[0] - u) < 0.01
            assert abs(uv2[1] - v) < 0.01


def test_point_behind_camera_is_none(pose, vc):
    """相机后方的点不可见 -> project 返回 None。"""
    cam = _model(vc, pose)
    behind = (vc.camera_position[0], vc.camera_position[1],
              vc.camera_position[2] + 0.5)     # 远在相机上方(视线反方向)
    assert cam.project(behind) is None


def test_ray_not_pointing_down_returns_none():
    """视线不朝下的相机无法与桌面平面求交 -> 返回 None(诚实, 不编造坐标)。"""
    # 相机从桌面下方向上照(up 取 y 轴避免与视线平行)
    up_pose = projection.build_camera_pose((0.3, 0.0, 0.1), (0.3, 0.0, 0.6),
                                           up=(0.0, 1.0, 0.0))
    origin, direction = projection.unproject_pixel(320.0, 240.0, up_pose,
                                                   600.0, 600.0, 320.0, 240.0)
    assert direction[2] > 0.0                   # 视线朝上
    assert projection.ray_intersect_plane(origin, direction, plane_z=0.0) is None


def test_up_parallel_to_view_raises(vc):
    """up 与视线平行(纯俯视)时无法构造相机系 -> 抛 ValueError。"""
    with pytest.raises(ValueError):
        projection.build_camera_pose((0.2, 0.0, 0.5), (0.2, 0.0, 0.0),
                                     up=(0.0, 0.0, 1.0))


def _model(vc, pose):
    from ...perception.camera import CameraModel
    return CameraModel(
        camera_id=vc.camera_id, width=vc.image_width, height=vc.image_height,
        fx=vc.fx, fy=vc.fy, cx=vc.image_width / 2.0, cy=vc.image_height / 2.0,
        world_T_cam=pose,
    )
