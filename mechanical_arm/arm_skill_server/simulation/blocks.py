"""物块：场景内物体的 ground-truth 描述(Phase 1)。

Phase 1 完全不用视觉: 仿真直接知道每个物块的颜色、位置、朝向。
Phase 2 起由相机+视觉模块提供同样的 BlockState。
"""
from __future__ import annotations

from dataclasses import dataclass

from ..robot.backend import BlockState

VALID_COLORS = ("yellow", "red", "white")


@dataclass(frozen=True)
class Block:
    """场景内一个物块 (base frame, meter)。"""

    color: str
    x: float
    y: float
    z: float
    yaw: float

    def to_state(self) -> BlockState:
        return BlockState(color=self.color, x=self.x, y=self.y, z=self.z, yaw=self.yaw)


def load_preset(sim_cfg: dict, preset_name: str) -> list[Block]:
    """按 simulation.yaml 的预设生成物块列表。

    preset_name 不在配置中时抛 KeyError(上层应捕获, 返回失败)。
    """
    presets = sim_cfg["scene_presets"]
    if preset_name not in presets:
        raise KeyError(f"未知场景预设: {preset_name!r}, 可选: {sorted(presets)}")
    return [Block(color=b["color"], x=b["x"], y=b["y"], z=b["z"], yaw=b["yaw"])
            for b in presets[preset_name]]
