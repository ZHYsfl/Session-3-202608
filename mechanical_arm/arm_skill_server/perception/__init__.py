"""感知层: 相机 -> 图像 -> PerceptionBackend -> BlockDetection -> 抓取。

Phase 2 用 HSV/经典视觉 baseline。设计目标: 低延迟、可测试、可 benchmark、可替换。

替换约定(Phase 3 兼容): 任何模块只依赖 PerceptionBackend 抽象,
    perception = HSVColorDetector(...)   ->   perception = TinyDetector(checkpoint)
业务代码(grab_the_block)不感知具体检测器。
"""
from .base import BlockDetection, PerceptionBackend
from .camera import CameraModel, CameraProvider, ImageFrame

__all__ = [
    "BlockDetection",
    "PerceptionBackend",
    "CameraModel",
    "CameraProvider",
    "ImageFrame",
]
