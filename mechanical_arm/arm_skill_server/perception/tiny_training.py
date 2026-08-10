"""Phase 4 自训练 TinyDetector 的数据生成 + 训练辅助函数。

职责(确定性优先, 固定 seed):
    - sample_layout: 在机械臂可达 + camera1 可视的桌面上, 随机选取物块
      子集与位置(互不重叠)。位置范围来自配置, 禁止 hardcode。
    - render_label: 用物块 **自设坐标**(数据生成器自己的 ground truth)投影
      桌面顶面矩形 -> 全分辨率类别图 -> 双线性缩放到 40x30 网格。
    - train_tiny_net: 在渲染帧小图上训练 TinyDetectorNet, 返回网络 + 指标。

反作弊边界(必须保持): 本模块**只用自设坐标造训练标签**, 是训练管线;
TinyDetector.detect_blocks 的推理路径不接触任何 scene/后端 ground truth,
只吃图像 + 已加载网络(见 tiny_detector.py)。HSV 对比基准的"真值"也来自
本模块自设坐标, 不是后端 get_block_states()。
"""
from __future__ import annotations

import dataclasses
import math
from typing import Callable

import cv2
import numpy as np

from ..config import VisionConfig
from ..perception.camera import CameraModel
from .tiny_detector import CLASS_IDS, TinyDetectorNet

COLORS = ("red", "yellow", "white")


@dataclasses.dataclass(frozen=True)
class TinyDataset:
    """渲染帧小图数据集。images/labels 形状 (N, GH, GW, ...)。"""

    images: np.ndarray      # (N, GH, GW, 3) uint8 BGR
    labels: np.ndarray      # (N, GH, GW) uint8 类别 {0..3}
    block_gt: list[dict]    # 每样本: {color: (x, y, yaw)}
    grid: tuple[int, int]   # (GW, GH)


def sample_layout(rng: np.random.Generator, vision_cfg: VisionConfig,
                  bounds: tuple[float, float, float, float],
                  min_sep_m: float = 0.08,
                  include_prob: float = 0.9) -> dict:
    """随机选物块子集与位置(互不重叠), 返回 {color: (x, y, yaw)}。

    bounds: (x_min, x_max, y_min, y_max) 世界坐标(m), 与机械臂可达区域一致。
    每颜色以 include_prob 概率出现(可空场景, 训练背景类); 用拒绝采样保证
    物块间距 >= min_sep_m。
    """
    rng = np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng
    x0, x1, y0, y1 = bounds
    layout: dict = {}
    for c in COLORS:
        if rng.random() > include_prob:
            continue
        for _ in range(200):                       # 拒绝采样, 防重叠
            x = rng.uniform(x0, x1)
            y = rng.uniform(y0, y1)
            if all(math.hypot(x - bx, y - by) >= min_sep_m for bx, by, _ in layout.values()):
                layout[c] = (float(x), float(y), float(rng.uniform(0.0, np.pi)))
                break
        # 200 次都没找到空位 -> 该场景少一个物块, 不插入(保持其它块不变)
    return layout


def _pixel_is_color(bgr_patch: np.ndarray, color: str, vision_cfg: VisionConfig,
                    min_frac: float = 0.5) -> bool:
    """渲染帧小补丁是否属于该颜色(用 vision.yaml 的 HSV 区间判定)。

    用于训练标签的遮挡剔除: 若物块顶面中心被机械臂/其它物块挡住, 渲染像素
    不是该物块颜色 -> 跳过该物块, 不把它标成可见(否则标签与像素不符, 教坏
    网络把暗色机身像素学成红色)。
    """
    hsv = cv2.cvtColor(bgr_patch, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for r in vision_cfg.hsv_ranges.get(color, ()):
        mask |= cv2.inRange(hsv, np.asarray(r.lower, np.uint8),
                            np.asarray(r.upper, np.uint8))
    return float(mask.mean()) >= min_frac


def render_label(layout: dict, camera_model: CameraModel, vision_cfg: VisionConfig,
                 img_shape: tuple[int, int], grid: tuple[int, int],
                 frame_bgr: np.ndarray | None = None) -> np.ndarray:
    """自设坐标物块 -> 网格类别图 (GH, GW) uint8。

    每个物块: 投影顶面(边长 block_side_m, 高度 桌面+2*half_height)的 4 角
    到像素 -> 填充矩形 -> 全分辨率类别图 -> INTER_NEAREST 缩到网格。
    任何角投影失败(在相机后/视野外)则跳过该角(靠边物块仍能标出可见部分)。
    若提供 frame_bgr, 用 _pixel_is_color 做遮挡剔除(顶面中心被挡住则整块不标)。
    """
    GW, GH = grid
    H_img, W_img = img_shape[:2]
    z_top = vision_cfg.table_top_z_m + 2.0 * vision_cfg.block_half_height_m
    half = vision_cfg.block_side_m / 2.0

    label_full = np.zeros((H_img, W_img), dtype=np.uint8)
    for color, (x, y, _yaw) in layout.items():
        if frame_bgr is not None:
            c = camera_model.project((x, y, z_top))
            if c is None:
                continue
            u, v = int(round(c[0])), int(round(c[1]))
            if not (0 <= v < H_img and 0 <= u < W_img):
                continue
            patch = frame_bgr[max(0, v - 3):v + 4, max(0, u - 3):u + 4]
            if not _pixel_is_color(patch, color, vision_cfg):
                continue                                  # 被机身/它块遮挡, 不标
        corners = []
        for sx in (-half, half):
            for sy in (-half, half):
                p = camera_model.project((x + sx, y + sy, z_top))
                if p is not None:
                    corners.append((float(p[0]), float(p[1])))
        if len(corners) < 2:
            continue
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        u0, u1 = int(max(0.0, min(xs))), int(min(W_img, max(xs)))
        v0, v1 = int(max(0.0, min(ys))), int(min(H_img, max(ys)))
        if u1 <= u0 or v1 <= v0:
            continue
        cv2.rectangle(label_full, (u0, v0), (u1 - 1, v1 - 1),
                      int(CLASS_IDS[color]), thickness=-1)

    # 全分辨率 -> 网格: 最近邻(保留类别语义, 不做插值)
    label_grid = cv2.resize(label_full, (GW, GH), interpolation=cv2.INTER_NEAREST)
    return label_grid


def render_dataset_scene(backend, camera, layout: dict, vision_cfg: VisionConfig,
                         grid: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, dict]:
    """放置一个随机场景并取帧, 返回 (网格小图, 网格标签, 布局 dict)。

    backend/camera 必须是 MuJoCo 后端 + SimulatorCamera(训练数据与推理同源渲染)。
    """
    backend.reset("no_blocks")
    table_top = vision_cfg.table_top_z_m
    z_rest = table_top + vision_cfg.block_half_height_m      # 底面贴桌
    for color, (x, y, yaw) in layout.items():
        qaddr = backend.scene_info.cube_free_addr[color]
        backend._spawn_cube(qaddr, (x, y, z_rest), yaw)
    backend._settle(20)                                       # 让物块在接触下稳定

    frame = camera.capture()
    label = render_label(layout, camera.camera_model, vision_cfg,
                         frame.bgr.shape, grid, frame_bgr=frame.bgr)
    small = cv2.resize(frame.bgr, grid, interpolation=cv2.INTER_LINEAR)
    return small, label, layout


def make_dataset(render_scene: Callable, n_samples: int, grid: tuple[int, int],
                 seed: int = 0) -> TinyDataset:
    """调用 render_scene(layout, rng) 生成 n_samples 个样本的数据集。

    render_scene(layout, rng) -> (small_image, label_grid, layout_dict)。
    需要随机数时就取 rng 参数(保证整体可复现), 布局由调用方用 rng 生成。
    """
    GW, GH = grid
    images = np.zeros((n_samples, GH, GW, 3), dtype=np.uint8)
    labels = np.zeros((n_samples, GH, GW), dtype=np.uint8)
    block_gt: list[dict] = []
    for i in range(n_samples):
        img, lab, layout = render_scene(i, seed)
        images[i] = img
        labels[i] = lab
        block_gt.append(layout)
    return TinyDataset(images=images, labels=labels, block_gt=block_gt, grid=grid)


def train_tiny_net(dataset: TinyDataset, cfg: VisionConfig,
                   epochs: int = 30, lr: float = 1e-3, bg_weight: float = 0.25,
                   seed: int = 0, device: str = "cpu",
                   val_frac: float = 0.1,
                   progress: Callable[[str], None] | None = None) -> tuple[TinyDetectorNet, dict]:
    """在渲染帧小图上训练 TinyDetectorNet(CPU, 秒级), 返回 (net, metrics)。

    metrics: {train_loss, val_acc, per-class IoU, params}。
    用固定 seed 划分 train/val; 背景类占比高(≈95%), 用 class weight 压制,
    避免网络全预测背景也能得到高准确率。
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    if progress is None:
        progress = lambda msg: None   # noqa: E731

    rng = np.random.default_rng(seed)
    n = len(dataset.images)
    val_n = max(1, int(n * val_frac))
    idx = rng.permutation(n)
    val_idx, train_idx = idx[:val_n], idx[val_n:]

    x_train = torch.from_numpy(dataset.images[train_idx].astype(np.float32) / 255.0)
    y_train = torch.from_numpy(dataset.labels[train_idx].astype(np.int64))
    x_val = torch.from_numpy(dataset.images[val_idx].astype(np.float32) / 255.0)
    y_val = torch.from_numpy(dataset.labels[val_idx].astype(np.int64))
    # (N, GH, GW, 3) -> (N, 3, GH, GW)
    x_train = x_train.permute(0, 3, 1, 2).contiguous()
    x_val = x_val.permute(0, 3, 1, 2).contiguous()

    dev = torch.device(device)
    x_train, y_train, x_val, y_val = (t.to(dev) for t in (x_train, y_train, x_val, y_val))

    torch.manual_seed(seed)
    net = TinyDetectorNet(cfg.tiny_grid_width, cfg.tiny_grid_height)
    net.to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    # 背景类权重(压制背景主导)
    weights = torch.full((len(CLASS_IDS),), 1.0, dtype=torch.float32)
    weights[CLASS_IDS["background"]] = bg_weight
    weights = weights.to(dev)

    for ep in range(epochs):
        net.train()
        opt.zero_grad()
        logits = net(x_train)
        loss = F.cross_entropy(logits, y_train, weight=weights)
        loss.backward()
        opt.step()
        progress(f"  epoch {ep + 1}/{epochs} loss={float(loss.item()):.4f}")

    # 验证: 逐 cell 准确率 + 每类 IoU
    net.eval()
    with torch.no_grad():
        pred = net(x_val).argmax(dim=1)
        val_acc = float((pred == y_val).float().mean())
        ious = {}
        for c in CLASS_IDS:
            cid = CLASS_IDS[c]
            inter = int(((pred == cid) & (y_val == cid)).sum())
            union = int(((pred == cid) | (y_val == cid)).sum())
            ious[c] = inter / union if union else 0.0

    metrics = {
        "val_acc": val_acc,
        "iou": ious,
        "params": net.n_params(),
        "train_n": len(train_idx),
        "val_n": len(val_idx),
    }
    return net, metrics
