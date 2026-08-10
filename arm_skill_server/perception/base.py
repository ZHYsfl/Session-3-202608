"""感知抽象: PerceptionBackend(ABC) + BlockDetection。

grab_the_block 只依赖 PerceptionBackend.detect_blocks(image) -> list[BlockDetection]。
实现可以是 HSVColorDetector(Phase 2)、TinyDetector(Phase 3)、未来任何模型。

GroundTruth vs Perceived:
    BlockDetection 来自对**图像**的感知(Perceived)。检测器不允许访问场景 ground truth。
    2D->3D(world_position)只能通过"像素 + 相机标定 + 桌面平面"得到, 不能直接用
    仿真物块坐标填充。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BlockDetection:
    """一次图像上对单个物块的检测结果。单位: 长度 meter, 像素 pixel。

    - bbox_xyxy    : (x0, y0, x1, y1), 图像完整坐标系(非 ROI 局部), 半开区间 [x0, x1)。
    - center_pixel : 物块包围盒中心 (u, v), 图像完整坐标系。
    - confidence   : [0, 1], 形状/填充比质量分。
    - world_position : base frame (x, y, z)。只有拿到相机标定且能可靠投影时才非 None;
                     否则为 None(仅像素定位, 文档标注 "world projection 尚未完成")。
    """

    color: str
    confidence: float
    bbox_xyxy: tuple[int, int, int, int]
    center_pixel: tuple[float, float]
    world_position: tuple[float, float, float] | None = None


class PerceptionBackend(ABC):
    """视觉感知后端抽象。Sim2Real: 同一接口对接合成相机/真实相机图像。"""

    @abstractmethod
    def detect_blocks(self, image_bgr: np.ndarray) -> list[BlockDetection]:
        """在 BGR 图像上检测物块。

        image_bgr: 形状 (H, W, 3) 的 uint8 图像(OpenCV 原生 BGR)。
        返回: 按颜色、按质量排序不重要——调用方(grab)会挑目标颜色的最佳候选。
        禁止: 访问任何仿真/scene/后端 ground truth。
        """
        raise NotImplementedError
