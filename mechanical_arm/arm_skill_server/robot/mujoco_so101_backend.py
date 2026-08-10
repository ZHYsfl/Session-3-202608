"""MuJoCoSO101Backend: 真正的 3D 物理仿真后端(Phase 3)。

与 KinematicSO101Backend 实现同一个 RobotBackend 抽象, 但物理由 MuJoCo 引擎驱动:
    - 关节: 力矩电机执行器; 控制器内层沿用 Phase 1 的离散梯形剖面生成参考轨迹,
      step() 显式计算力矩 PD + 重力补偿(qfrc_bias)下发到 ctrl, 物理决定真实运动。
      (Phase 3 修正: <position> 伺服稳态下垂=重力矩/kp, 0.005 rad 容差下无法收敛,
      且高 kp 触发饱和伪平衡; 改力矩电机 + 重力补偿 -> 稳态误差 ~0。)
    - 末端: 读 MuJoCo ee_tip site 的真实位姿(FK 验证见 test_mujoco_backend.py)。
    - 物块: 三色 cube 为 free-joint 刚体(重力/碰撞/接触/摩擦), reset() 按场景预设
      放置, 缺失颜色停放远处。get_block_states() 返回**物理引擎当前真实位置**。
    - 夹取: close_gripper 用接触力判定是否真的咬住物块(不是"命令已发")。
    - 安全: 末端低于桌面由真实碰撞兜底; 控制器安全代理仍保留。

Sim2Real 角度: 本类对应 Phase 6/7 的 Isaac/真机后端, 上层 skills/api 零改动。
"""
from __future__ import annotations

import threading

import numpy as np

try:
    import mujoco
except ImportError as e:  # pragma: no cover - 依赖未装时给出明确错误
    raise ImportError("Phase 3 需要 mujoco: python -m pip install mujoco") from e

from ..config import ControlConfig, VisionConfig
from ..simulation.mujoco_scene import MujocoSceneInfo, build_mujoco_scene
from ..robot.kinematics import RobotModel
from .backend import BlockState, CameraFrame, EEPose, RobotBackend
from .kinematics import numeric_jacobian
from .trajectory import next_ramped_velocity

# 夹取判定阈值(与 holding_proximity 同级)
_GRIP_R_XY_M = 0.03            # 物块中心距末端水平距离上限(m)
_LIFT_EPS_M = 0.004            # 「明显离开桌面」判定(m)
_CONTACT_FORCE_N = 1e-4        # 接触法向力阈值(N), 证明真的压紧
_TABLE_MARGIN_M = 0.005


class MuJoCoSO101Backend(RobotBackend):
    # 抓取时固定腕部自旋到 π/2: 工具 x 轴(夹爪张开方向)由「前倾方向」
    # (0.601,0,-0.799) 旋到世界 Y (0,1,0), 两个手指同高、侧向卡住物块。
    # 若不固定, 前倾工具的前指(fR)会低垂到物块/桌面高度, 任何下降深度
    # 都必然碰撞(Phase 3 集成测试证明; 见 _mj_grab_q4.py)。
    # q4 绕工具自身轴旋转、不移动 tool tip(FK 位置保持不变), 故可安全强制。
    grasp_wrist_roll: float | None = np.pi / 2.0

    def __init__(self, model: RobotModel, sim_cfg: dict, control: ControlConfig,
                 vision_cfg: VisionConfig | None = None) -> None:
        self.model = model                    # 运动学模型(与 Phase 1 相同)
        # Phase 3: 物理后端必须用「物理可行解」IK(连杆不入地/不撞底座)。
        # 碰撞守卫默认关闭以保持 Phase 1/2 行为; 这里在共享 model 实例上显式打开。
        # (skill 通过 state.require_model() 拿到的是同一个实例, 见 _motion.py)
        model.collision_guard = True
        self.control = control
        self.sim_cfg = sim_cfg
        self.scene_info: MujocoSceneInfo = build_mujoco_scene(
            model, sim_cfg, vision_cfg if vision_cfg is not None else VisionConfig())
        self.m = self.scene_info.model
        self.d = self.scene_info.data

        self._lock = threading.RLock()
        self.q_target = model.q_home().copy()
        self.q_dot_ref = np.zeros(model.dof)
        self.safety_stop_reason: str | None = None
        self._active_colors: set[str] = set()
        self._timestep = float(self.scene_info.physics.get("timestep", 0.002))
        self._settle_steps = int(self.scene_info.physics.get("gripper_settle_steps", 150))
        # 臂关节力矩 PD(Phase 3: motor 执行器, 控制器显式做 PD + 重力补偿)。
        self._arm_kp = float(self.scene_info.physics.get("arm_kp", 200.0))
        self._arm_kv = float(self.scene_info.physics.get("arm_kv", 10.0))
        self._arm_force = float(self.scene_info.physics.get("arm_force_limit", 8.0))
        lims = self.scene_info.physics.get("arm_force_limits", None)
        if lims is None:
            self._arm_forces = np.full(model.dof, self._arm_force)
        else:
            self._arm_forces = np.asarray([float(x) for x in lims], dtype=float)
            if self._arm_forces.shape != (model.dof,):
                raise ValueError("arm_force_limits 长度必须等于关节数")

        self.reset("no_blocks")

    # ------------------------------------------------------------------
    # 状态读取
    # ------------------------------------------------------------------
    def get_joint_positions(self) -> np.ndarray:
        q = np.empty(self.model.dof)
        for i, nm in enumerate(self.scene_info.arm_joint_names):
            q[i] = self.d.qpos[self.scene_info.qpos_addr[nm]]
        return q

    def get_joint_velocities(self) -> np.ndarray:
        v = np.empty(self.model.dof)
        for i, nm in enumerate(self.scene_info.arm_joint_names):
            v[i] = self.d.qvel[self.scene_info.qvel_addr[nm]]
        return v

    def get_end_effector_pose(self) -> EEPose:
        pos = self.d.site_xpos[self.scene_info.ee_site_id].copy()
        mat = self.d.site_xmat[self.scene_info.ee_site_id].reshape(3, 3).copy()
        return EEPose(position=pos, rotation=mat)

    def get_end_effector_velocity(self) -> np.ndarray:
        q = self.get_joint_positions()
        qdot = self.get_joint_velocities()
        return numeric_jacobian(self.model, q) @ qdot

    # ------------------------------------------------------------------
    # 运动指令
    # ------------------------------------------------------------------
    def set_joint_targets(self, q: np.ndarray) -> None:
        self.q_target = np.clip(np.asarray(q, dtype=float), self.model.q_min(), self.model.q_max())
        self.safety_stop_reason = None

    def step(self, dt: float) -> None:
        with self._lock:
            if self.safety_stop_reason is not None:
                self._hold()
                return
            q = self.get_joint_positions()
            vmax = self.model.vmax()
            accel = self.model.accel()
            err = self.q_target - q
            new_dot = np.zeros(self.model.dof)
            for i in range(self.model.dof):
                new_dot[i] = next_ramped_velocity(
                    error=err[i], v_prev=self.q_dot_ref[i], dt=dt,
                    vmax=float(vmax[i]), accel=float(accel[i]))
            self.q_dot_ref = new_dot
            q_ref = q + new_dot * dt

            # 内层高速力矩环(子步率, 500Hz): 跟踪 (q_ref, v_ref)。
            #   tau = kp*(q_ref - q) + kv*(v_ref - qvel) + qfrc_bias
            # 速度前馈: kv 只作用在速度误差上 -> 匀速跟踪时不饱和;
            # 重力补偿: 稳态误差 ~0(无下垂)。外层 50Hz 只出参考轨迹。
            self._servo_track(q_ref, self.q_dot_ref, dt)

            # 安全代理(与 Phase 1 一致): 末端低于桌面 -> 冻结
            ee = self.get_end_effector_pose().position
            if ee[2] < self.model.table_top_z_m - _TABLE_MARGIN_M:
                self.safety_stop_reason = "末端低于桌面(碰撞代理)"
                self._hold()

    def stop(self) -> None:
        with self._lock:
            self.q_target = self.get_joint_positions().copy()
            self.q_dot_ref[:] = 0.0
            self._hold()

    # ------------------------------------------------------------------
    # 夹爪(物理咬合判定)
    # ------------------------------------------------------------------
    def open_gripper(self) -> None:
        with self._lock:
            self._set_gripper_ctrl(self.scene_info.gripper_ctrl_open)
            self._settle(self._settle_steps)

    def close_gripper(self) -> None:
        with self._lock:
            self._set_gripper_ctrl(self.scene_info.gripper_ctrl_closed)
            self._settle(self._settle_steps)

    def is_holding_object(self) -> bool:
        """物理诚实的夹取判定: 物块「被抬起」或「被手指压紧」。

        - 抬起: 物块中心明显高于桌面 + 仍在末端正下方(水平距离 < 0.03)。
        - 压紧: 手指-物块存在带法向力的接触 + 物块仍在桌面高度(还没抬)。
        两条都是物块被夹持的物理证据, 不是"命令已发"。
        """
        with self._lock:
            tool = self.d.site_xpos[self.scene_info.ee_site_id]
            table_top = self.model.table_top_z_m
            for color in self._active_colors:
                p = self._cube_center(color)
                d_xy = float(np.hypot(p[0] - tool[0], p[1] - tool[1]))
                if d_xy > _GRIP_R_XY_M:
                    continue
                lifted = p[2] > table_top + self.scene_info.block_half_height_m + _LIFT_EPS_M
                if lifted:
                    return True
                if self._finger_cube_contact(color):
                    return True
            return False

    # ------------------------------------------------------------------
    # 感知
    # ------------------------------------------------------------------
    def get_block_states(self) -> list[BlockState]:
        out = []
        for color in sorted(self._active_colors):
            p = self._cube_center(color)
            yaw = self._cube_yaw(color)
            out.append(BlockState(color=color, x=float(p[0]), y=float(p[1]),
                                  z=float(p[2]), yaw=float(yaw)))
        return out

    def get_camera_frame(self, camera_id: str) -> CameraFrame:
        if camera_id not in self.scene_info.camera_id:
            raise KeyError(f"未知相机: {camera_id!r}, 可选 {sorted(self.scene_info.camera_id)}")
        return CameraFrame(camera_id=camera_id, visible_blocks=tuple(self.get_block_states()))

    # ------------------------------------------------------------------
    # 场景
    # ------------------------------------------------------------------
    def reset(self, scene_preset: str | None = None) -> None:
        with self._lock:
            presets = self.sim_cfg.get("scene_presets", {})
            if scene_preset is not None and scene_preset not in presets:
                raise KeyError(f"未知场景预设: {scene_preset!r}, 可选: {sorted(presets)}")

            # 1) 机械臂回 home + 夹爪张开
            home = self.model.q_home()
            for i, nm in enumerate(self.scene_info.arm_joint_names):
                a = self.scene_info.qpos_addr[nm]
                self.d.qpos[a] = home[i]
                self.d.qvel[self.scene_info.qvel_addr[nm]] = 0.0
            self._set_gripper_ctrl(self.scene_info.gripper_ctrl_open)
            for nm, ctrl_v in zip(self.scene_info.finger_joint_names,
                                  self.scene_info.gripper_ctrl_open):
                self.d.qpos[self.scene_info.finger_qpos_addr[nm]] = ctrl_v
                self.d.qvel[self.scene_info.finger_dof_addr[nm]] = 0.0
            self.q_target = home.copy()
            self.q_dot_ref[:] = 0.0
            self.safety_stop_reason = None
            # 刷新重力补偿: 回位后让电机输出当前位形的重力补偿力矩(不是位置)。
            mujoco.mj_forward(self.m, self.d)
            grav = self._arm_gravity()
            for i, nm in enumerate(self.scene_info.arm_joint_names):
                self.d.ctrl[self.scene_info.actuator_id[f"act_{nm}"]] = grav[i]

            # 2) 物块: 预设放置 / 缺失停放远处
            self._active_colors = set()
            preset_blocks = [] if scene_preset is None else presets[scene_preset]
            for b in preset_blocks:
                self._active_colors.add(b["color"])
            for color in self.scene_info.cube_free_addr:
                qaddr = self.scene_info.cube_free_addr[color]
                if color in self._active_colors:
                    b = next(b for b in preset_blocks if b["color"] == color)
                    self._spawn_cube(qaddr, (float(b["x"]), float(b["y"]), float(b["z"])),
                                     float(b.get("yaw", 0.0)))
                else:
                    self._spawn_cube(qaddr, (0.5, -1.5, 0.05), 0.0)

            mujoco.mj_forward(self.m, self.d)
            # 让方块/机械臂在重力下短暂稳定(0.1s)
            self._settle(50)

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------
    def _advance(self, dt: float) -> None:
        substeps = max(1, int(round(dt / self._timestep)))
        for _ in range(substeps):
            mujoco.mj_step(self.m, self.d)

    def _servo_track(self, q_ref: np.ndarray, v_ref: np.ndarray, dt: float) -> None:
        """子步率力矩伺服: 在每个物理子步里刷新扭矩命令, 跟踪 (q_ref, v_ref)。

        外层 50Hz 由 step() 出参考轨迹; 这里以 timestep(=0.002s, 500Hz)为周期
        计算 tau = kp*err_q + kv*err_v + qfrc_bias 并写 ctrl, 再推进一个子步。
        速度前馈(kv 作用在速度误差)避免匀速跟踪时 kv 项饱和; 重力补偿项
        消除稳态下垂。Phase 3 集成测试证明这是让 0.005 rad 收敛容差可达的
        唯一方案(位置伺服或 50Hz 单环都会因手腕低惯量+大力矩 bang-bang 振荡)。
        """
        mujoco.mj_forward(self.m, self.d)   # 刷新 qfrc_bias 到当前位形
        substeps = max(1, int(round(dt / self._timestep)))
        names = self.scene_info.arm_joint_names
        for _ in range(substeps):
            qq = self.get_joint_positions()
            qv = self.get_joint_velocities()
            grav = self._arm_gravity()
            tau = (self._arm_kp * (q_ref - qq)
                   + self._arm_kv * (v_ref - qv)
                   + grav)
            tau = np.clip(tau, -self._arm_forces, self._arm_forces)
            for i, nm in enumerate(names):
                self.d.ctrl[self.scene_info.actuator_id[f"act_{nm}"]] = tau[i]
            mujoco.mj_step(self.m, self.d)

    def _settle(self, steps: int) -> None:
        """在受控伺服保持下让夹爪/系统稳定(500Hz 刷新臂力矩)。

        修复 Phase 3 释放 bug: 原实现只裸 mj_step 不刷新臂 ctrl, 若动作结束时
        的 ctrl 是伺服收敛残留值(收敛容差 0.005rad 内 err≈0.002rad, 见
        _mj_open_trace2.py: ctrl=-1.29 而该位形重力=-1.69), 夹爪张开、物块
        脱离后手指-物块接触约束消失, 臂会欠补偿下垂并把刚释放的物块压住。
        这里在稳定期间持续向 q_target 做 PD+重力补偿伺服保持: 重力补偿逐子步
        刷新(位形微变也跟踪), PD 提供阻尼, 臂在夹爪动作期间纹丝不动。
        """
        self._servo_track(self.q_target, np.zeros(self.model.dof),
                          steps * self._timestep)

    def _hold(self) -> None:
        """冻结机械臂: 只输出当前位形的重力补偿力矩(PD 误差项为零)。"""
        mujoco.mj_forward(self.m, self.d)
        grav = self._arm_gravity()
        for i, nm in enumerate(self.scene_info.arm_joint_names):
            self.d.ctrl[self.scene_info.actuator_id[f"act_{nm}"]] = grav[i]

    def _arm_gravity(self) -> np.ndarray:
        """当前位形的关节重力/科氏/离心补偿力矩(长度 dof)。"""
        return np.array([self.d.qfrc_bias[self.scene_info.qvel_addr[nm]]
                         for nm in self.scene_info.arm_joint_names])

    def _set_gripper_ctrl(self, targets: tuple[float, float]) -> None:
        self.d.ctrl[self.scene_info.actuator_id["act_finger_L"]] = targets[0]
        self.d.ctrl[self.scene_info.actuator_id["act_finger_R"]] = targets[1]

    def _spawn_cube(self, qaddr: int, xyz: tuple[float, float, float], yaw: float) -> None:
        """放置 free-joint 物块。qpos: 3 位置 + 4 四元数; qvel: 6(free 关节 dof)。"""
        self.d.qpos[qaddr:qaddr + 3] = xyz
        self.d.qpos[qaddr + 3:qaddr + 7] = [
            float(np.cos(yaw / 2.0)), 0.0, 0.0, float(np.sin(yaw / 2.0))]
        color = next(c for c, a in self.scene_info.cube_free_addr.items() if a == qaddr)
        dof = self.scene_info.cube_free_dof_addr[color]
        self.d.qvel[dof:dof + 6] = 0.0

    def _cube_center(self, color: str) -> np.ndarray:
        qaddr = self.scene_info.cube_free_addr[color]
        return self.d.qpos[qaddr:qaddr + 3].copy()

    def _cube_yaw(self, color: str) -> float:
        qaddr = self.scene_info.cube_free_addr[color]
        w, x, y, z = self.d.qpos[qaddr + 3:qaddr + 7]
        return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))

    def _finger_cube_contact(self, color: str) -> bool:
        """手指与物块是否存在带法向力的接触(MuJoCo 接触数组)。"""
        cube_geom = self.scene_info.cube_geom_names[color]
        _name = lambda gi: mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, gi)
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            g1 = _name(c.geom1)
            g2 = _name(c.geom2)
            is_finger = (g1 is not None and "finger" in g1) or \
                        (g2 is not None and "finger" in g2)
            is_cube = (g1 == cube_geom) or (g2 == cube_geom)
            if not (is_finger and is_cube):
                continue
            n = np.zeros(6)
            mujoco.mj_contactForce(self.m, self.d, i, n)
            if abs(n[0]) > _CONTACT_FORCE_N:
                return True
        return False

    # ---- 测试钩子(仅测试用) -------------------------------------------
    def cube_position(self, color: str) -> np.ndarray:
        """读取物块物理位置(测试/调试用)。"""
        return self._cube_center(color)
