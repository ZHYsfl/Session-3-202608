"""场景环境：桌面上的物块集合 + 相机定义。

Phase 1 为固定场景(无 domain randomization)。Scene 只持有数据,
“谁在动”由 KinematicSO101Backend 决定(夹取时物块跟随末端)。
"""
from __future__ import annotations

from ..robot.backend import BlockState
from .blocks import Block, load_preset
from .cameras import Camera, parse_cameras


class Scene:
    def __init__(self, sim_cfg: dict) -> None:
        self.sim_cfg = sim_cfg
        self.block_half_height_m = float(sim_cfg["block"]["half_height_m"])
        self.colors: list[str] = list(sim_cfg["block"]["colors"])
        self.cameras: dict[str, Camera] = parse_cameras(sim_cfg)
        self.blocks: list[Block] = []          # 当前场景物块(可变列表, Block 不可变)

    def reset(self, preset_name: str | None = None) -> None:
        """按预设重建物块。None -> 空场景。"""
        if preset_name is None:
            self.blocks = []
        else:
            self.blocks = load_preset(self.sim_cfg, preset_name)

    # ---- 查询 ------------------------------------------------------------
    def get_block_states(self) -> list[BlockState]:
        return [b.to_state() for b in self.blocks]

    def blocks_of_color(self, color: str) -> list[Block]:
        return [b for b in self.blocks if b.color == color]

    # ---- 夹取跟随(仅由后端调用) ----------------------------------------
    def find_held_index(self, held: Block) -> int:
        """按对象身份找到 held 在 blocks 中的下标; 找不到返回 -1。"""
        for i, b in enumerate(self.blocks):
            if b is held:
                return i
        return -1

    def replace_block(self, index: int, new: Block) -> None:
        self.blocks[index] = new
