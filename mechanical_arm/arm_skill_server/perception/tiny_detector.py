"""TinyDetector(PerceptionBackend): 自训练微型 CNN 物块分割(Phase 4)。

定位思路(确定性优先, 不做目标检测里的 NMS 那一套):
    1. 按 ROI 裁剪 -> 双线性缩放到 GRID_W x GRID_H(默认 40x30)小图
    2. TinyDetectorNet: 4 层 3x3 卷积(padding=1 保持分辨率)+ 1x1 head,
       输出每 cell 的 {background, red, yellow, white} 4 类 logits(~17k 参数)
    3. softmax -> per-cell 类别掩码 -> connectedComponentsWithStats(8 连通)
    4. 用 softmax 概率做**加权质心**(子 cell 精度), 反算回图像完整坐标
    5. 像素 -> 相机标定 -> 桌面平面 -> 世界坐标(与 HSVColorDetector 完全一致)

与 HSVColorDetector 的差异: 判定依据从「手工 HSV 阈值」变成「在自训练渲染帧上
学到的像素分类器」。训练(scripts/generate_tiny_dataset.py + train_tiny_detector.py)
用 ground truth 造标签, 但 **inference 路径不接触任何 scene/后端 ground truth**,
detect_blocks 只吃一张图像 + 已加载的网络, 行为与 HSV 完全对齐(grab 零改动)。

自训练数据与真实推理都基于 MuJoCo 渲染帧(camera1), 与 HSV 在同一批帧上对比
(scripts/benchmark_tiny_vs_hsv.py)。
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "Phase 4 需要 torch(CPU 版即可): python -m pip install torch "
        "--index-url https://download.pytorch.org/whl/cpu") from e

from ..config import VisionConfig
from .base import BlockDetection, PerceptionBackend
from .camera import CameraModel

# 类别表: 训练与推理共用同一映射(0 = background)。
CLASS_IDS = {"background": 0, "red": 1, "yellow": 2, "white": 3}
CLASS_NAMES = ("background", "red", "yellow", "white")
N_CLASS = len(CLASS_IDS)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class TinyDetectorNet(nn.Module):
    """微型全卷积分类网络: (B,3,GH,GW) -> (B,4,GH,GW) 逐 cell logits。

    4 层 3x3 卷积 + BN + ReLU(padding=1 保持分辨率), 1x1 head 出类别 logits。
    参数量 ~17k。感受野 9 cell, 物块投影 ~3 cell, 足够; 全分辨率逐像素
    CNN 对小模型是浪费, 40x30 网格是精度/速度/训练样本量的平衡点
    (一个 0.04m 物块投影 ~48px = 3 cell)。
    """

    def __init__(self, grid_w: int = 40, grid_h: int = 30,
                 n_class: int = N_CLASS) -> None:
        super().__init__()
        self.grid_w = int(grid_w)
        self.grid_h = int(grid_h)
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(32, n_class, 1)

    def forward(self, x):
        return self.head(self.features(x))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def probs_to_detections(
    probs: np.ndarray,
    roi: tuple[int, int, int, int],
    img_shape: tuple[int, int],
    camera_model: CameraModel | None,
    cfg: VisionConfig,
) -> list[BlockDetection]:
    """per-cell 概率图 -> BlockDetection 列表(纯函数, 无 torch, 可单测)。

    probs   : (GH, GW, N_CLASS) float32, softmax 后的逐 cell 概率。
    roi     : 图像完整坐标 (x0,y0,x1,y1) —— cell -> 完整像素的偏移基准。
    img_shape: (H, W) 完整图像, 用于把 bbox 夹到图像范围。
    流程: 每色 argmax 掩码 & 概率>=confidence -> 8 连通域 -> 概率加权质心
          -> 反算回完整像素 -> (可选)像素->世界投影。与 HSV 的后处理等价。
    """
    GW, GH = cfg.tiny_grid_width, cfg.tiny_grid_height
    roi_w, roi_h = float(roi[2] - roi[0]), float(roi[3] - roi[1])
    H_img, W_img = img_shape[:2]
    dets: list[BlockDetection] = []

    amax = probs.argmax(axis=-1)
    for color, cid in CLASS_IDS.items():
        if color == "background":
            continue
        color_prob = probs[..., cid]
        mask = ((amax == cid) & (color_prob >= cfg.tiny_confidence_threshold)).astype(np.uint8)
        n, labels, stats, _cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < cfg.tiny_min_area_cells:
                continue
            comp = labels == i
            # 质心只用高置信 cell(cutoff, 默认 0.8): 低置信边缘(如手臂遮挡造成的
            # 50-80% 不确定 cell)会把概率加权质心往遮挡侧拉偏(Phase 4 标定:
            # 演示布局白物块 x=0.24 处质心偏 7mm -> 抓取滑落)。高置信核心才反映
            # 物块真实中心。bbox 仍用整块组件(低阈值), 与置信度语义一致。
            core = comp & (color_prob >= cfg.tiny_centroid_prob)
            ys, xs = np.nonzero(core)
            if ys.size == 0:
                ys, xs = np.nonzero(comp)          # 退化保护: 全低置信时退回全组件
            weights = color_prob[ys, xs]
            cy = float((weights * ys).sum() / weights.sum())     # 子 cell 质心(v 向)
            cx = float((weights * xs).sum() / weights.sum())     # 子 cell 质心(u 向)

            j0 = int(stats[i, cv2.CC_STAT_LEFT])
            i0 = int(stats[i, cv2.CC_STAT_TOP])
            bw = int(stats[i, cv2.CC_STAT_WIDTH])
            bh = int(stats[i, cv2.CC_STAT_HEIGHT])

            # cell -> 完整像素(ROI 局部 -> 图像完整坐标)
            u = roi[0] + (cx + 0.5) / GW * roi_w
            v = roi[1] + (cy + 0.5) / GH * roi_h
            u0 = roi[0] + j0 / GW * roi_w
            v0 = roi[1] + i0 / GH * roi_h
            u1 = roi[0] + (j0 + bw) / GW * roi_w
            v1 = roi[1] + (i0 + bh) / GH * roi_h
            u0 = int(max(0, min(W_img, u0)))
            u1 = int(max(0, min(W_img, u1)))
            v0 = int(max(0, min(H_img, v0)))
            v1 = int(max(0, min(H_img, v1)))

            world_position = _project_world(camera_model, cfg, u, v)
            dets.append(BlockDetection(
                color=color,
                confidence=float(weights.mean()),
                bbox_xyxy=(u0, v0, u1, v1),
                center_pixel=(u, v),
                world_position=world_position,
            ))
    return dets


def _project_world(camera_model: CameraModel | None, cfg: VisionConfig,
                   u: float, v: float):
    """像素 -> 桌面平面上世界坐标(与 HSVColorDetector._project_world 一致)。"""
    if camera_model is None:
        return None
    xy = camera_model.pixel_to_world_on_plane(u, v, cfg.table_top_z_m)
    if xy is None:
        return None
    z_block = cfg.table_top_z_m + cfg.block_half_height_m
    return (xy[0], xy[1], z_block)


class TinyDetector(PerceptionBackend):
    """已训练的 TinyDetectorNet 包装: 与 HSVColorDetector 同一 PerceptionBackend 接口。

    构造需要 checkpoint(默认 configs/vision.yaml tiny.checkpoint, 项目根相对路径)。
    未训练时给出明确指引: 先跑 scripts/generate_tiny_dataset.py + train_tiny_detector.py。
    """

    def __init__(self, cfg: VisionConfig,
                 camera_model: CameraModel | None = None,
                 checkpoint: str | Path | None = None,
                 device: str = "cpu") -> None:
        self._cfg = cfg
        self._camera_model = camera_model
        self._device = torch.device(device)

        net = TinyDetectorNet(cfg.tiny_grid_width, cfg.tiny_grid_height)
        cp = self._resolve_checkpoint(checkpoint)
        if not cp.exists():
            raise FileNotFoundError(
                f"TinyDetector checkpoint 不存在: {cp}\n"
                f"先运行:\n"
                f"  python scripts/generate_tiny_dataset.py\n"
                f"  python scripts/train_tiny_detector.py")
        state = torch.load(cp, map_location=self._device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        net.load_state_dict(state)
        net.eval()
        self._net = net

    def _resolve_checkpoint(self, checkpoint) -> Path:
        if checkpoint is None:
            cp = Path(self._cfg.tiny_checkpoint)
        else:
            cp = Path(checkpoint)
        if not cp.is_absolute():
            cp = _PROJECT_ROOT / cp
        return cp

    # ------------------------------------------------------------------
    def predict_probs(self, image_bgr: np.ndarray) -> np.ndarray:
        """ROI -> 网格小图 -> 网络前向 -> (GH, GW, N_CLASS) softmax 概率。"""
        x0, y0, x1, y1 = self._cfg.roi
        roi = image_bgr[y0:y1, x0:x1]
        if roi.size == 0:
            raise ValueError(f"ROI 为空: {self._cfg.roi} vs 图像 {image_bgr.shape}")
        small = cv2.resize(roi, (self._cfg.tiny_grid_width, self._cfg.tiny_grid_height),
                           interpolation=cv2.INTER_LINEAR)
        x = small.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))[None]          # (1, 3, GH, GW)
        with torch.no_grad():
            logits = self._net(torch.from_numpy(x).to(self._device))
            probs = torch.softmax(logits, dim=1)[0].permute(1, 2, 0).cpu().numpy()
        return probs

    def detect_blocks(self, image_bgr: np.ndarray) -> list[BlockDetection]:
        """纯 2D 感知: 只吃一张图像 + 已加载网络, 不访问任何 ground truth。"""
        probs = self.predict_probs(image_bgr)
        return probs_to_detections(
            probs, self._cfg.roi, image_bgr.shape,
            self._camera_model, self._cfg)
