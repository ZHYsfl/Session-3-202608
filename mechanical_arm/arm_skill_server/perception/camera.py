"""相机抽象: CameraProvider(ABC) + ImageFrame + CameraModel。

注意与 robot/backend.py 的 CameraFrame 区分:
    - backend.CameraFrame   = Phase 1 占位(ground truth 可见物块摘要, 无图像)。
    - perception.ImageFrame = Phase 2 起的**图像帧**(带像素), 供 PerceptionBackend 消费。
两个相机: camera1 主操作/识别, camera2 第三视角验证(Phase 2 聚焦 camera1)。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from . import projection


@dataclass(frozen=True)
class ImageFrame:
    """一张相机图像。图像像素为 OpenCV 原生 **BGR**(与 cv2 直接兼容)。

    - bgr      : (H, W, 3) uint8。
    - width    : 图像宽度(像素)。
    - height   : 图像高度(像素)。
    - timestamp: 采集时刻(秒, time.time())。benchmark/回放用。
    预留: 深度图 / 内参 / 外参不塞进本类(超工程), 由 CameraModel 单独承载。
    """

    camera_id: str
    bgr: np.ndarray
    timestamp: float

    @property
    def width(self) -> int:
        return self.bgr.shape[1]

    @property
    def height(self) -> int:
        return self.bgr.shape[0]


class CameraProvider(ABC):
    """相机取帧抽象。Sim2Real: 合成相机 / 真实相机(如 RealSense)都实现 capture()。"""

    @property
    @abstractmethod
    def camera_model(self) -> "CameraModel":
        """该相机的标定模型(内参 + 外参), 用于像素 <-> 世界换算。"""
        raise NotImplementedError

    @abstractmethod
    def capture(self) -> ImageFrame:
        """取一帧图像。合成相机无额外等待(不混入渲染延时), 真实相机按硬件帧率阻塞。"""
        raise NotImplementedError


@dataclass(frozen=True)
class CameraModel:
    """针孔相机标定模型。

    - width/height : 图像尺寸(像素)。
    - fx/fy        : 焦距(像素)。fx = f / d_pixel_x。
    - cx/cy        : 主点(像素)。
    - world_T_cam  : 相机位姿(4x4), p_world = world_T_cam @ [p_cam, 1]。
    """

    camera_id: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    world_T_cam: np.ndarray

    def project(self, p_world) -> tuple[float, float] | None:
        """世界点 -> 像素 (u, v); 不可见返回 None。"""
        return projection.project_point(
            p_world, self.world_T_cam, self.fx, self.fy, self.cx, self.cy)

    def pixel_to_world_on_plane(self, u: float, v: float, plane_z: float):
        """像素 -> 水平平面 z=plane_z 上的世界坐标 (x, y, plane_z); 无解返回 None。"""
        return projection.pixel_to_world_on_plane(
            u, v, self.world_T_cam, self.fx, self.fy, self.cx, self.cy, plane_z)
