"""SyntheticCamera —— 明确标记为 test fixture 的合成图像源。

背景(Phase 1): 仿真没有真实渲染管线, 只有物块 ground truth 状态
(`cameras.py::make_mock_frame` 直接返回可见物块摘要, 无像素图像)。

本类用针孔模型把场景物块"画"成一张与 ground truth 位置一致的 BGR 图像,
供 HSV baseline / CI / benchmark 使用。它是**确定性的图像发生器**, 不是真实
视觉仿真。迁移到真实 3D 仿真(Phase 2.5, 见 docs/perception.md)或真机后,
本类换成真实相机实现, PerceptionBackend 与 grab 逻辑不变。

⚠️ 请不要把本类渲染出的图像宣称成真实视觉仿真。真实相机应在 Phase 2.5 或
   Phase 7 提供; 本类只负责把"已知物块位置"变成"看起来像相机看到的图像"。
"""
from __future__ import annotations

import time

import cv2
import numpy as np

from ..config import VisionConfig
from ..simulation.environment import Scene
from . import projection
from .camera import CameraModel, CameraProvider, ImageFrame


class SyntheticCamera(CameraProvider):
    """合成相机: 针孔投影渲染场景物块到图像平面(测试 fixture)。"""

    def __init__(self, scene: Scene, cfg: VisionConfig) -> None:
        self._scene = scene
        self._cfg = cfg
        self._world_T_cam = projection.build_camera_pose(
            cfg.camera_position, cfg.camera_look_at, cfg.camera_up)

    @property
    def camera_model(self) -> CameraModel:
        return CameraModel(
            camera_id=self._cfg.camera_id,
            width=self._cfg.image_width,
            height=self._cfg.image_height,
            fx=self._cfg.fx,
            fy=self._cfg.fy,
            cx=self._cfg.image_width / 2.0,     # 主点取图像中心
            cy=self._cfg.image_height / 2.0,
            world_T_cam=self._world_T_cam,
        )

    def capture(self, brightness: float = 1.0, noise_sigma: float = 0.0) -> ImageFrame:
        """取一帧。brightness 缩放整体亮度(光照变化测试), noise_sigma 加高斯噪声。

        合成相机**不做任何 sleep/仿真延时**——延迟 benchmark 里不混入渲染等待。
        """
        W = self._cfg.image_width
        H = self._cfg.image_height
        table = np.asarray(self._cfg.table_color_bgr, dtype=np.float32)
        img = np.full((H, W, 3), table, dtype=np.float32)

        for b in self._scene.get_block_states():
            self._render_block(img, b)

        # 光照: 全局亮度缩放, 夹到 [0, 255]
        img = np.clip(img * float(brightness), 0.0, 255.0)
        if noise_sigma > 0.0:
            noise = np.random.default_rng().normal(0.0, float(noise_sigma), img.shape)
            img = np.clip(img + noise, 0.0, 255.0)

        return ImageFrame(
            camera_id=self._cfg.camera_id,
            bgr=img.astype(np.uint8),
            timestamp=time.time(),
        )

    def _render_block(self, img: np.ndarray, b) -> None:
        """把单个物块**底面**(z = block.z - half_height)投影成多边形并填充。

        底面中心 = 物块垂直中心在桌面平面上的投影: 物块在桌面时底面 z=0,
        于是"像素 -> 平面 z=0 求交"能精确回到物块的 (x, y)。
        """
        cfg = self._cfg
        half = cfg.block_side_m / 2.0
        cos_y, sin_y = float(np.cos(b.yaw)), float(np.sin(b.yaw))
        bottom_z = float(b.z) - cfg.block_half_height_m

        pts = []
        for dx, dy in ((-half, -half), (half, -half), (half, half), (-half, half)):
            wx = b.x + dx * cos_y - dy * sin_y    # 绕 (b.x, b.y) 旋转偏航角
            wy = b.y + dx * sin_y + dy * cos_y
            uv = self.camera_model.project((wx, wy, bottom_z))
            if uv is None:
                return                            # 角点在相机后方: 整块不可见, 跳过
            pts.append([int(round(uv[0])), int(round(uv[1]))])

        color = cfg.block_colors_bgr.get(b.color, (200, 200, 200))
        cv2.fillPoly(img, np.array([pts], dtype=np.int32), color=color)
