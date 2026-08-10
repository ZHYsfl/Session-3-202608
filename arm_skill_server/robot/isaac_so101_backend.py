"""IsaacSO101Backend：IsaacLab / LeIsaac 仿真后端(占位 stub)。

IsaacLab/IsaacSim 官方仅支持 Ubuntu + NVIDIA。当前开发机是 Windows 原生,
无法运行 Isaac。本文件是接口占位, 接入条件与最小迁移方案见 docs/simulation_setup.md。

接入流程(Phase 6):
    1. WSL2 / Ubuntu 24.04 安装 Isaac Sim + Isaac Lab(版本要求见 simulation_setup.md)。
    2. 下载 so101_follower.usd 放入 assets。
    3. 实现以下每个方法, 把 IsaacLab 的 state/command 映射到 RobotBackend 语义。
    4. skills/api 无需任何改动(RobotBackend 是唯一接口)。

注意: 本模块不会在 Windows 上被导入。所有方法 raise NotImplementedError。
"""
from __future__ import annotations

import numpy as np

from .backend import BlockState, CameraFrame, EEPose, RobotBackend


class IsaacSO101Backend(RobotBackend):
    def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        raise NotImplementedError(
            "IsaacSO101Backend 需在 Ubuntu + IsaacLab 环境中使用。"
            "见 docs/simulation_setup.md 的最小迁移方案。"
        )

    # 以下方法签名与 RobotBackend 完全一致, 供 Phase 6 实现参考。
    def get_joint_positions(self) -> np.ndarray:
        raise NotImplementedError

    def get_joint_velocities(self) -> np.ndarray:
        raise NotImplementedError

    def get_end_effector_pose(self) -> EEPose:
        raise NotImplementedError

    def get_end_effector_velocity(self) -> np.ndarray:
        raise NotImplementedError

    def set_joint_targets(self, q: np.ndarray) -> None:
        raise NotImplementedError

    def step(self, dt: float) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError

    def open_gripper(self) -> None:
        raise NotImplementedError

    def close_gripper(self) -> None:
        raise NotImplementedError

    def is_holding_object(self) -> bool:
        raise NotImplementedError

    def get_block_states(self) -> list[BlockState]:
        raise NotImplementedError

    def get_camera_frame(self, camera_id: str) -> CameraFrame:
        raise NotImplementedError

    def reset(self, scene_preset: str | None = None) -> None:
        raise NotImplementedError
