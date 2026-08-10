"""相机：Phase 1 占位(ground-truth 摘要), Phase 2 起提供真实图像帧。

两个相机:
    camera1  主操作/识别视角
    camera2  第三视角验证抓取/释放(协议 §2.3/§2.4 用它交叉验证)
"""
from __future__ import annotations

from dataclasses import dataclass

from ..robot.backend import BlockState, CameraFrame


@dataclass(frozen=True)
class Camera:
    """相机定义 (base frame 下位姿, 单位 meter/radian)。"""

    camera_id: str
    x: float
    y: float
    z: float
    yaw: float
    pitch: float


def parse_cameras(sim_cfg: dict) -> dict[str, Camera]:
    """从 simulation.yaml 解析相机。camera_id -> Camera。"""
    out: dict[str, Camera] = {}
    for cid in ("camera1", "camera2"):
        if cid in sim_cfg:
            p = sim_cfg[cid]["pose"]
            out[cid] = Camera(camera_id=cid, x=p["x"], y=p["y"], z=p["z"],
                              yaw=p["yaw"], pitch=p["pitch"])
    return out


def make_mock_frame(camera: Camera, visible_blocks: list[BlockState]) -> CameraFrame:
    """Phase 1 占位帧: 直接给出该视角“可见”的物块 ground truth。

    Phase 2 起这里替换为真实投影/渲染, 返回带图像的帧。
    """
    return CameraFrame(camera_id=camera.camera_id, visible_blocks=tuple(visible_blocks))
