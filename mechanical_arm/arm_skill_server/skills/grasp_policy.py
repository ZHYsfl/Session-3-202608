"""Phase 5 抓取策略: 脚本 IK 专家(数据生成) + 自训练 BC 策略(闭环推理)。

两条可切换路径(grasp_mode: "scripted" | "bc"):
    scripted  脚本 IK 专家: approach -> descend -> close -> lift(_grasp_sequence,
              确定性, 默认, 见 grab_block.py)。
    bc        自训练 BC 策略: 用专家演示在渲染帧/物理后端上自监督训练出的
              微型 MLP 闭环抓取。策略每 tick 以 当前末端位置 + 物块目标位置
              (obs = block - ee, holding) 重新决策目标点 (dx, dy, dz, grip)。

执行模型(Phase 5 集成定标, 见 run_bc_grasp docstring): 策略每 tick 输出
    wp = ee + dxyz 与 grip; 移动用与 scripted 同款的关节空间收敛(move_to_point)
    把目标点执行到位, 仅当目标点改变(相位切换/到位/漂移超容差)才重新规划。
    原因: 每 tick 对任意中间点重解 IK, 在物块附近(近底座)的奇异姿态下会解出
    扭矩伺服无法执行的关节角, 末端爬行停住(白块抓取超时, Phase 5 标定 bug)。
    策略仍是唯一决策者(目标点与闭夹爪都由它决定), move_to_point 只是下层伺服。

反作弊边界:
    训练侧: 专家用「生成器自己放下的物块坐标」(ground truth)出演示, 是标准
    监督学习管线(与 Phase 4 render_label 同源)。_dense_phase_samples 沿相位路径
    稠密重采样专家策略, 是标准 BC 数据增强(等价于专家更细时间分辨率记录)。
    推理侧: run_bc_grasp 只消费 调用方传入的物块目标(由 perception 视觉定位),
    + backend 的当前末端位姿; **不读取 backend.get_block_states()**。物块目标
    一经感知固定, 抓取全程不再更新(与 scripted 路径一致)。

obs/action 约定:
    obs    = (block - ee, holding)(相对向量 + 夹爪持有标志)。
              holding 是关键去歧义特征: 「下降经过抬升高度」(未持有) 与
              「抬升到位」(已持有) 的 block-ee 完全相同, 不加 holding 网络
              会把两个时刻的矛盾标签平均 -> 在抬升高度停住(Phase 5 标定 bug)。
              夹爪是否咬住物块是真机可得的本体感受(舵机电流/力传感), 不是
              ground truth 位置感知。
    action = (waypoint - ee, grip): 专家当前阶段目标点相对末端的位移 + 夹爪
             闭合意图(approach=0, descend=1, lift=1)。训练即拟合这个映射。
             descend 起 grip=1 让策略在接近物块时就有闭合意图, 由推理端
             几何守卫在「已到下降点」时真正执行闭夹爪。
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np

from ..config import ControlConfig
from ..robot.backend import RobotBackend
from ..robot.controller import Pacing
from ..robot.kinematics import RobotModel, fk
from ..runtime.cancellation import CancellationToken
from ._common import UnreachableError, reachable_projection, solve_ik_robust
from ._motion import GraspFailed, move_to_point

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

try:
    import torch
    from torch import nn
except ImportError as e:  # pragma: no cover - 依赖未装时给出明确错误
    raise ImportError(
        "Phase 5 需要 torch: python -m pip install torch\n"
        "(CPU 版即可; 训练/推理都是小型 MLP。)") from e


# ---------------------------------------------------------------------------
# 网络
# ---------------------------------------------------------------------------

class GraspPolicyNet(nn.Module):
    """BC 抓取策略: obs(4) -> (dx, dy, dz, grip_logit)(4)。

    4 -> 64 -> 64 -> 4 的微型 MLP(~4.9k 参数)。obs = (block - ee, holding)。
    因为 obs 前 3 维是相对向量且工作空间已知, 小网络即可拟合
    「approach/descend/close/lift」的分段映射(见模块 docstring)。
    """

    def __init__(self) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 4),
        )

    def forward(self, x):
        return self.mlp(x)


def clamp_magnitude(vec: np.ndarray, max_m: float) -> np.ndarray:
    """把向量幅值限制到 max_m(方向不变)。纯函数, 推理与训练共用。"""
    v = np.asarray(vec, dtype=float)
    n = float(np.linalg.norm(v))
    if n <= max_m or n == 0.0:
        return v
    return v * (max_m / n)


# ---------------------------------------------------------------------------
# 策略加载(缓存)
# ---------------------------------------------------------------------------

class BCGraspPolicy:
    """已加载的 BC 策略。act(obs) -> (位移增量(3), grip 概率)。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"BC 抓取策略 checkpoint 缺失: {self.path}\n"
                "请先运行 scripts/generate_grasp_dataset.py && "
                "scripts/train_grasp_policy.py 产出模型。")
        ckpt = torch.load(self.path, map_location="cpu", weights_only=True)
        self.net = GraspPolicyNet()
        self.net.load_state_dict(ckpt["state_dict"])
        self.net.eval()
        # 归一化标定(训练时由数据集统计; 存为 tensor 便于 weights_only=True 加载)
        self.obs_mean = ckpt["obs_mean"].numpy()
        self.obs_std = np.where(ckpt["obs_std"].numpy() < 1e-6, 1.0,
                                ckpt["obs_std"].numpy())
        self.act_std = ckpt["act_std"].numpy()

    def act(self, obs: np.ndarray) -> tuple[np.ndarray, float]:
        """obs(4,) = (block - ee, holding) -> (位移增量(3,), grip 概率 [0,1])。"""
        x = (np.asarray(obs, dtype=float) - self.obs_mean) / self.obs_std
        t = torch.as_tensor(x, dtype=torch.float32).reshape(1, -1)
        with torch.no_grad():
            out = self.net(t)[0]                      # (4,)
        dxyz = out[:3].numpy() * self.act_std
        grip = float(torch.sigmoid(out[3]))
        return dxyz, grip


_cached_policy: BCGraspPolicy | None = None
_cached_path: str | None = None
_cached_lock = threading.Lock()


def load_grasp_policy(control: ControlConfig) -> BCGraspPolicy:
    """按 control.grasp_policy_checkpoint 加载策略(进程内缓存, 同路径只读一次)。"""
    global _cached_policy, _cached_path
    p = Path(control.grasp_policy_checkpoint)
    path = str(p if p.is_absolute() else _PROJECT_ROOT / p)
    with _cached_lock:
        if _cached_policy is not None and _cached_path == path:
            return _cached_policy
        _cached_policy = BCGraspPolicy(path)
        _cached_path = path
        return _cached_policy


# ---------------------------------------------------------------------------
# 脚本 IK 专家(仅训练数据生成)
# ---------------------------------------------------------------------------

class ScriptedExpert:
    """脚本 IK 专家: 逐 tick 复刻 approach/descend/close/lift 并记录 (obs, action)。

    与生产 _grasp_sequence 同一套参数(approach/descent/lift 高度、关节收敛容差、
    腕部固定), 只是把「移动到位」拆成 tick 级并逐 tick 记录。仅在训练数据生成
    时使用; 物块位置为 ground truth(生成器自己放下, 见 generate_grasp_dataset.py)。

    返回 trajectory: list[(obs(3, float32), action(4, float32))]。
    失败(不可达/未夹住/取消/安全)抛 GraspFailed。
    """

    def __init__(self, backend: RobotBackend, model: RobotModel,
                 control: ControlConfig, token: CancellationToken,
                 pacing: Pacing) -> None:
        self.backend = backend
        self.model = model
        self.control = control
        self.token = token
        self.pacing = pacing
        self.wrist = backend.grasp_wrist_roll

    # -- 公开 ------------------------------------------------------------
    def run_episode(self, color: str, block_xyz) -> list[tuple[np.ndarray, np.ndarray]]:
        """单块布局跑一次完整抓取, 返回逐 tick 的 (obs, action)。

        color/block_xyz: 训练侧 ground truth(物块定位, 非视觉)。
        """
        b = self.backend
        b.reset("no_blocks")
        self._place_block(color, block_xyz)
        b.open_gripper()

        block = np.asarray(block_xyz, dtype=float)
        ctl = self.control
        traj: list[tuple[np.ndarray, np.ndarray]] = []

        # 1) approach: 移到物块上方 approach_height(无闭合意图)
        self._drive_to(block + [0.0, 0.0, ctl.approach_height_m], grip=0.0,
                       block=block, traj=traj)
        # 2) descend: 下降到物块中心附近(grip=1: 接近物块即有闭合意图)
        self._drive_to(block + [0.0, 0.0, ctl.descent_offset_m], grip=1.0,
                       block=block, traj=traj)
        # 3) close: 闭合夹爪, 必须物理咬住
        b.close_gripper()
        if not b.is_holding_object():
            raise GraspFailed("专家闭合夹爪后未检测到物块")
        # 4) lift: 抬升(夹爪已闭, grip=1)
        self._drive_to(block + [0.0, 0.0, ctl.lift_height_m], grip=1.0,
                       block=block, traj=traj)
        # 5) 稠密重采样: 沿 descend/lift 相位路径补足专家策略样本(标准 BC 数据增强)
        self._dense_phase_samples(block, traj)
        return traj

    # -- 内部 ------------------------------------------------------------
    def _place_block(self, color: str, xyz) -> None:
        """放一个物块(训练侧 ground truth; 复用 Phase 4 同款后端内部放置钩子)。"""
        b = self.backend
        qaddr = b.scene_info.cube_free_addr[color]
        b._active_colors.add(color)                        # noqa: SLF001 - 训练侧
        b._spawn_cube(qaddr, (float(xyz[0]), float(xyz[1]), float(xyz[2])), 0.0)
        import mujoco                                        # 专家仅 MuJoCo 后端使用
        mujoco.mj_forward(b.m, b.d)                          # noqa: SLF001 - 训练侧
        b._settle(50)                                        # noqa: SLF001 - 训练侧

    def _drive_to(self, waypoint: np.ndarray, grip: float, block: np.ndarray,
                  traj: list) -> None:
        """驱动末端到 waypoint(关节空间收敛, 与 execute_joint_move 同语义), 逐 tick 记录。

        waypoint: 当前阶段目标点(base frame)。grip: 该阶段的夹爪标签。
        obs = (block - ee, holding); action = (waypoint - ee, grip)。
        """
        b, m, ctl = self.backend, self.model, self.control
        target = reachable_projection(m, np.asarray(waypoint, dtype=float))
        q_target = solve_ik_robust(m, target, b.get_joint_positions())
        if q_target is None:
            raise GraspFailed("专家 IK 无解")
        if self.wrist is not None:
            q_target = q_target.copy()
            q_target[4] = float(np.clip(self.wrist, m.q_min()[4], m.q_max()[4]))
            p_check, _ = fk(m, q_target)
            if np.linalg.norm(p_check - target) > 1e-2:
                raise GraspFailed("腕部强制后 IK 位置偏移")

        b.set_joint_targets(q_target)
        wp = np.asarray(waypoint, dtype=float)
        while True:
            if self.token.cancelled:
                raise GraspFailed("取消")
            b.step(ctl.dt)
            ee = b.get_end_effector_pose().position
            if not np.all(np.isfinite(ee)) or ee[2] < m.table_top_z_m - 0.005:
                raise GraspFailed("安全停止(末端低于桌面)")
            holding = float(b.is_holding_object())
            obs = np.concatenate([(block - ee), [holding]]).astype(np.float32)
            act = np.concatenate([(wp - ee), [float(grip)]]).astype(np.float32)
            traj.append((obs, act))
            q = b.get_joint_positions()
            if float(np.max(np.abs(q - q_target))) < ctl.joint_convergence_tol_rad:
                break
            self.pacing.sleep(ctl.dt)

    def _dense_phase_samples(self, block: np.ndarray, traj: list) -> None:
        """沿 descend/lift 相位路径稠密采样专家策略, 平滑相位过渡并去偏。

        真实演示中 descend(从 approach 高度到下降点只有 ~5-10 tick)与 lift
        (~10 tick)样本太少, 网络在「接近下降点」区域学不干净 -> 推理端策略
        位移输出抖动, 目标点 wp=ee+d 漂移、闭环爬行。这里用专家的分段策略
        函数在这些路径上稠密采样 (obs, action), 与真实轨迹同格式。这是标准
        BC 的专家策略重采样(等价于专家以更细的时间分辨率记录, 非伪造数据)。

        obs = (block - ee, holding); action = (waypoint - ee, grip)。
        descend: 未持有, grip=1, 位移 = D - ee;  lift: 已持有, grip=1, 位移 = L - ee。
        另外在 approach 末端(A 附近)补一小段过渡, 让 grip 从 0 到 1 的相位
        切换有明确样本(approach 阶段 grip≈0, descend 阶段 grip≈1)。
        """
        ctl = self.control
        A = block + [0.0, 0.0, ctl.approach_height_m]
        D = block + [0.0, 0.0, ctl.descent_offset_m]
        L = block + [0.0, 0.0, ctl.lift_height_m]
        xy = block[:2]
        n = 48
        # descend 线: ee 沿 A -> D 竖直向下(未持有)
        for z in np.linspace(A[2], D[2], n):
            ee = np.array([xy[0], xy[1], z], dtype=float)
            obs = np.concatenate([block - ee, [0.0]]).astype(np.float32)
            act = np.concatenate([D - ee, [1.0]]).astype(np.float32)
            traj.append((obs, act))
        # 下降端点密集带: 末端停在 D 附近(未持有, 即将闭夹)。数据偏斜点:
        # 下降端点(obs≈(0,0,-0.015,0)) 与抬升起点(obs≈(0,0,-0.015,1)) 在 obs
        # 空间仅差 holding, 网络学不干净会把两者平均 -> false-lift(把末端带走,
        # 打断闭夹 settle 计数, 需二次下降)。在 D 上方再稠密放一批「已到下降点、
        # 未持有、位移≈0」样本, 让 holding=0 -> d≈0 有足够权重; 推理端策略在
        # 下降端点输出停留而非抬升(几何闭夹守卫仍保留作最终防御)。
        for dz in np.linspace(0.006, 0.0, 30):
            ee = np.array([xy[0], xy[1], D[2] + dz], dtype=float)
            obs = np.concatenate([block - ee, [0.0]]).astype(np.float32)
            act = np.concatenate([D - ee, [1.0]]).astype(np.float32)
            traj.append((obs, act))
        # lift 线: ee 沿 D -> L 竖直向上(已持有)
        for z in np.linspace(D[2], L[2], n):
            ee = np.array([xy[0], xy[1], z], dtype=float)
            obs = np.concatenate([block - ee, [1.0]]).astype(np.float32)
            act = np.concatenate([L - ee, [1.0]]).astype(np.float32)
            traj.append((obs, act))
        # approach 末端过渡: ee 在 A 正上方小范围(弥合 grip=0 与 grip=1 的相位边界)
        for dz in np.linspace(0.012, -0.012, 25):
            ee = np.array([xy[0], xy[1], A[2] + dz], dtype=float)
            disp, g = (A - ee, 0.0) if dz > 0 else (D - ee, 1.0)
            obs = np.concatenate([block - ee, [0.0]]).astype(np.float32)
            act = np.concatenate([disp, [g]]).astype(np.float32)
            traj.append((obs, act))


# ---------------------------------------------------------------------------
# BC 闭环抓取(推理)
# ---------------------------------------------------------------------------

def _near_descend(ee: np.ndarray, block: np.ndarray, ctl: ControlConfig) -> bool:
    """闭夹爪防御性几何守卫: 末端位置在下降点附近(水平 + 高度双容差)。

    只判几何(不再看策略位移增量——执行模型已改为「相位目标点收敛」, 末端到达
    下降点时是静止的, settle 判定由 run_bc_grasp 的连续 tick 计数完成)。
    """
    descend = block + [0.0, 0.0, ctl.descent_offset_m]
    return (abs(ee[0] - descend[0]) < ctl.grasp_policy_close_xy_m
            and abs(ee[1] - descend[1]) < ctl.grasp_policy_close_xy_m
            and abs(ee[2] - descend[2]) < ctl.grasp_policy_close_z_tol_m)


def run_bc_grasp(target_xyz, backend: RobotBackend, model: RobotModel,
                 control: ControlConfig, token: CancellationToken,
                 pacing: Pacing) -> None:
    """BC 策略闭环抓取: 每 tick 以 (末端, 物块目标) 重新决策目标点。

    target_xyz: 物块目标(base frame, meter)——由 perception 视觉定位传入,
    本函数**不读 ground truth**。失败(取消/超时/目标不可达/安全/抬升未夹住)
    抛 GraspFailed。

    执行模型(Phase 5 定标): 策略每 tick 输出 (dxyz, grip), 目标点 wp = ee + dxyz
    (approach 阶段约等于 A, descend 阶段约等于 D, lift 阶段约等于 L)。移动用与
    scripted 路径同款的关节空间收敛(move_to_point)执行到位, 仅当目标点改变
    (相位切换/到位/漂移超容差)时才重新规划。原因(集成标定发现): 每 tick 对
    任意中间点重解 IK, 在物块附近(近底座)的奇异姿态下会解出扭矩伺服无法执行的
    关节角, 末端爬行停住(白块超时)。固定相位目标点 A/D/L 都被验证可执行。
    策略仍是唯一决策者(目标点 + 闭夹爪都由它决定), move_to_point 只是下层伺服。
    """
    policy = load_grasp_policy(control)
    block = np.asarray(target_xyz, dtype=float)
    ctl = control
    wrist = backend.grasp_wrist_roll

    grip_issued = False
    resolved_wp: np.ndarray | None = None
    prev_grip_hi = False
    prev_holding = False
    close_streak = 0
    t0 = time.monotonic()
    while True:
        if token.cancelled:
            raise GraspFailed("取消")
        if time.monotonic() - t0 > ctl.grasp_timeout_s:
            raise GraspFailed("超时")

        ee = backend.get_end_effector_pose().position
        if not np.all(np.isfinite(ee)) or ee[2] < model.table_top_z_m - 0.005:
            backend.stop()
            raise GraspFailed("安全停止(末端低于桌面)")

        # 1) 策略决策(obs 含夹爪持有标志, 去歧义 approach 经过/抬升到位)
        holding = float(backend.is_holding_object())
        dxyz, grip = policy.act(np.concatenate([block - ee, [holding]]))
        wp = ee + dxyz                            # 策略期望的下一目标点

        # 2) 闭夹爪事件: grip 过阈值 且 末端停在下降点附近 且 连续 settle N tick。
        #    settle 计数在「执行到位后」自然累积(移动用 move_to_point 收敛, 到位即静止)。
        #    [Phase 5 稳定性修复] 一旦进入「闭夹爪候选」(已在下降点附近、grip 意图=1、
        #    尚未闭夹), 就地稳住不再让策略位移打断计数: 训练数据在「下降端点(未持有)」
        #    区域样本稀疏, 策略在该点可能输出 lift 位移(false-lift, d_z≈+0.08), 放行会
        #    使 |wp-ee| 超容差触发 move_to_point 把末端带走、重置 settle -> 抓取需二次
        #    下降才闭夹(时序相关 flaky)。此分支只在移动收敛到下降点后才进入(循环内
        #    只见过 move_to_point 的收敛位姿), 几何守卫保证已在下降点, 无需再移动。
        close_candidate = (not grip_issued
                           and grip > ctl.grasp_policy_grip_threshold
                           and _near_descend(ee, block, ctl))
        if close_candidate:
            close_streak += 1
            if close_streak >= ctl.grasp_policy_close_settle_ticks:
                backend.close_gripper()
                grip_issued = True
                close_streak = 0
                continue                      # 闭夹爪期间后端已 settle, 下一 tick 再决策
            backend.step(ctl.dt)              # 尚未达标: 就地稳住, 不执行策略位移
            pacing.sleep(ctl.dt)
            continue
        close_streak = 0

        # 3) 抬升验收: 已闭夹爪 且 末端到抬升高度
        if grip_issued and ee[2] >= block[2] + ctl.lift_height_m - 0.01:
            if backend.is_holding_object():
                backend.stop()                # 刷掉 PD 残余, 臂停在真平衡点(同 scripted)
                return
            raise GraspFailed("抬升后未夹住物块")

        # 4) 移动: 目标点改变(相位切换/到位/漂移超容差)时才用 move_to_point 重新规划。
        #    否则保持当前关节目标推进(servo 维持), 不再每 tick 对中间点重解 IK。
        #    漂移用 |wp - ee|(策略想走的相对位移): 收敛后臂停在目标点, 若策略
        #    仍想走(approach 末端相位过渡下移 ~4-18mm)则推进; 收敛位姿的策略
        #    噪声(~2mm)低于该容差不会误触发。
        grip_hi = grip >= ctl.grasp_policy_phase_threshold
        wp_changed = (resolved_wp is None
                      or grip_hi != prev_grip_hi
                      or bool(holding) != prev_holding
                      or np.linalg.norm(wp - ee) > ctl.grasp_policy_waypoint_re_tol_m)
        if wp_changed:
            try:
                move_to_point(wp, backend, model, control, token, pacing,
                              force_wrist_roll=wrist)
            except UnreachableError as e:
                raise GraspFailed(f"BC 目标点不可达: {e}")
            resolved_wp = wp
        else:
            backend.step(ctl.dt)
            pacing.sleep(ctl.dt)

        prev_grip_hi = grip_hi
        prev_holding = bool(holding)
