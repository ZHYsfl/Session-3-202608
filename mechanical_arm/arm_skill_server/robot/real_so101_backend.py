"""RealSO101Backend：真机后端(占位 stub, Sim2Real 目标实现)。

真机需要:
    - 串口驱动(Feetech STS3215 总线舵机, 12 位磁编码器, 115200 baud)
    - 标定结果替换 configs/robot.yaml 的占位运动学参数
      (lerobot-calibrate 或官方 URDF)
    - 关节编码器 -> 关节角 -> FK -> 末端位姿
    - 夹爪: 位置/电流 + 相机验证 交叉判定 holding(协议 §2.4)
    - 物块感知: 视觉模块(Phase 2/3)填充 get_block_states

本文件先把接口钉死, 实现留到 Phase 7(Sim2Real)。不会在 Phase 1 被导入。
"""
from __future__ import annotations

import numpy as np

from .backend import BlockState, CameraFrame, EEPose, RobotBackend


class RealSO101Backend(RobotBackend):
    def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        raise NotImplementedError(
            "RealSO101Backend 尚未实现(Phase 7 Sim2Real)。"
            "需要串口驱动 + 真机标定, 见 docs/architecture_notes.md。"
        )

    # 以下方法签名与 RobotBackend 完全一致, 供 Phase 7 实现参考。
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
