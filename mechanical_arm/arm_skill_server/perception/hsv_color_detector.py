"""HSVColorDetector(PerceptionBackend): BGR -> HSV -> 颜色掩码 -> 连通域。

流程:
    1. 按 ROI 裁剪(图像完整坐标 -> ROI 局部)
    2. BGR -> HSV
    3. 每个颜色: inRange 取多个区间并集 -> 掩码
    4. 形态学开/闭(可选): 去噪点 / 填洞
    5. connectedComponentsWithStats -> 每个连通域 -> bbox / 中心 / 填充比置信度
    6. bbox/中心坐标加回 ROI 偏移, 回到图像完整坐标系

检测器是**纯 2D 感知**: 只吃一张图像, 不访问任何 scene/后端 ground truth。
world_position 仅在显式传入 camera_model 时, 用「像素 + 相机标定 + 桌面平面」
投影填充(见 docs/perception.md §像素到世界), 不是 ground truth。
"""
from __future__ import annotations

import cv2
import numpy as np

from ..config import VisionConfig
from .base import BlockDetection, PerceptionBackend
from .camera import CameraModel


class HSVColorDetector(PerceptionBackend):
    def __init__(self, cfg: VisionConfig,
                 camera_model: CameraModel | None = None) -> None:
        self._cfg = cfg
        self._camera_model = camera_model     # 可选: 提供则填充 world_position

    # ------------------------------------------------------------------
    # 分阶段原语(benchmark 分阶段计时用; 不是公共 API, 名字前不加下划线
    # 以便 scripts/benchmark_vision.py 直接复用同一段代码路径)
    # ------------------------------------------------------------------
    def to_hsv(self, image_bgr: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    def build_mask(self, hsv: np.ndarray, color: str) -> np.ndarray:
        """单个颜色的 HSV 掩码: 多区间并集 + 可选形态学。"""
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for r in self._cfg.hsv_ranges.get(color, ()):
            lower = np.asarray(r.lower, dtype=np.uint8)
            upper = np.asarray(r.upper, dtype=np.uint8)
            mask |= cv2.inRange(hsv, lower, upper)
        k = self._cfg.morphology_kernel
        if k > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    def find_components(self, mask: np.ndarray):
        return cv2.connectedComponentsWithStats(mask, connectivity=8)

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------
    def detect_blocks(self, image_bgr: np.ndarray) -> list[BlockDetection]:
        x0, y0, x1, y1 = self._cfg.roi
        roi = image_bgr[y0:y1, x0:x1]
        hsv = self.to_hsv(roi)

        H_img, W_img = image_bgr.shape[:2]
        dets: list[BlockDetection] = []
        for color in self._cfg.hsv_ranges:           # 只检测配置了阈值的颜色
            mask = self.build_mask(hsv, color)
            n, _labels, stats, _cents = self.find_components(mask)
            for i in range(1, n):
                area = int(stats[i, cv2.CC_STAT_AREA])
                if area < self._cfg.min_area_px:
                    continue
                bw = int(stats[i, cv2.CC_STAT_WIDTH])
                bh = int(stats[i, cv2.CC_STAT_HEIGHT])
                fill = area / float(max(1, bw * bh))
                if fill < self._cfg.confidence_threshold:
                    continue

                # ROI 局部坐标 -> 图像完整坐标(并夹到图像范围, 处理靠边物块)
                u0 = x0 + int(stats[i, cv2.CC_STAT_LEFT])
                v0 = y0 + int(stats[i, cv2.CC_STAT_TOP])
                u1 = min(u0 + bw, W_img)
                v1 = min(v0 + bh, H_img)
                cu = (u0 + u1) / 2.0
                cv_ = (v0 + v1) / 2.0

                world_position = self._project_world(cu, cv_)
                dets.append(BlockDetection(
                    color=color,
                    confidence=fill,
                    bbox_xyxy=(u0, v0, u1, v1),
                    center_pixel=(cu, cv_),
                    world_position=world_position,
                ))
        return dets

    def _project_world(self, u: float, v: float):
        """像素 -> 桌面平面上世界坐标(仅当配置了 camera_model)。不可靠返回 None。"""
        if self._camera_model is None:
            return None
        xy = self._camera_model.pixel_to_world_on_plane(u, v, self._cfg.table_top_z_m)
        if xy is None:
            return None
        z_block = self._cfg.table_top_z_m + self._cfg.block_half_height_m
        return (xy[0], xy[1], z_block)
