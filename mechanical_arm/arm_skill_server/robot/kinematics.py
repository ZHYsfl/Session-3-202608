"""SO-101 运动学：正向运动学(FK) + 逆向运动学(IK)。

坐标系与单位约定（全项目统一）:
    - 坐标系: base frame。机器人底座原点, z 向上, 桌面在 z = 0。
    - 长度单位: meter; 角度单位: radian。
    - 关节链模型: 每个关节先绕本地 axis("z" 或 "y") 旋转 q,
      再沿本地 link 向量平移, 依次链接。见 configs/robot.yaml。

IK 使用阻尼最小二乘(DLS / damped least squares)数值求解:
    Δq = Jᵀ (J Jᵀ + λ² I)⁻¹ e,   e = 目标位置 - 当前末端位置
    λ 是阻尼系数(接近奇异时防止爆炸), 迭代时把 q 夹在关节限位内。
    对 SO-101 这类 5 自由度臂求解 3 维位置, 腕部自由度自然保留。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

# 轴名 -> 单位向量
_AXIS_VEC = {
    "z": np.array([0.0, 0.0, 1.0]),
    "y": np.array([0.0, 1.0, 0.0]),
}


def rot_matrix(axis: str, angle: float) -> np.ndarray:
    """绕单位轴(axis ∈ {"z","y"})旋转 angle 的 3x3 旋转矩阵(Rodrigues 公式)。

    R = cosθ·I + (1-cosθ)·uuᵀ + sinθ·[u]×
    """
    u = _AXIS_VEC[axis]
    c, s = np.cos(angle), np.sin(angle)
    ux = np.array([[0.0, -u[2], u[1]],
                   [u[2], 0.0, -u[0]],
                   [-u[1], u[0], 0.0]])          # [u]× 反对称阵
    return c * np.eye(3) + (1.0 - c) * np.outer(u, u) + s * ux


@dataclass
class JointModel:
    """单个关节的运动学 + 动力学限位。"""

    name: str
    axis: str                       # "z" 或 "y"：旋转轴
    link: np.ndarray                # 旋转后平移向量 [dx, dy, dz] (meter)
    q_home: float                   # 初始关节角 (rad)
    limit_min: float                # 关节限位下界 (rad)
    limit_max: float                # 关节限位上界 (rad)
    vmax: float                     # 最大关节速度 (rad/s)
    accel: float                    # 最大关节加速度 (rad/s^2)


@dataclass
class RobotModel:
    """从 configs/robot.yaml 解析出的运动学模型。"""

    name: str
    joints: list[JointModel]        # 顺序即 FK 链顺序
    max_reach_m: float
    table_top_z_m: float
    # Phase 3 碰撞守卫(物理后端用): 默认关闭, 保持 Phase 1/2 行为完全不变。
    collision_guard: bool = False   # 开启后 IK 只返回物理可行解(连杆不入地/不撞底座)
    base_radius_m: float = 0.014    # 底座立柱半径(已含裕量), 用于碰撞检查
    base_z_hi_m: float = 0.090      # 底座立柱顶面高度
    link_radius_m: float = 0.022    # 连杆半厚

    # ---- 便捷数组接口 -------------------------------------------------
    @property
    def dof(self) -> int:
        return len(self.joints)

    def q_home(self) -> np.ndarray:
        return np.array([j.q_home for j in self.joints])

    def q_min(self) -> np.ndarray:
        return np.array([j.limit_min for j in self.joints])

    def q_max(self) -> np.ndarray:
        return np.array([j.limit_max for j in self.joints])

    def vmax(self) -> np.ndarray:
        return np.array([j.vmax for j in self.joints])

    def accel(self) -> np.ndarray:
        return np.array([j.accel for j in self.joints])

    def joint_names(self) -> list[str]:
        return [j.name for j in self.joints]


# ---------------------------------------------------------------------------
# 正向运动学 FK
# ---------------------------------------------------------------------------

def fk(model: RobotModel, q: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """正向运动学：给定关节角 q, 返回末端(tool tip)位置 p(3,) 和姿态 R(3x3)。

    T₀,i = T₀,i-1 · Rot(axis_i, q_i) · Trans(link_i)
    """
    q = np.asarray(q, dtype=float)
    T = np.eye(4)
    for joint, qi in zip(model.joints, q):
        R = rot_matrix(joint.axis, float(qi))
        step = np.eye(4)
        step[:3, :3] = R
        # link 沿「旋转后的本地轴」平移: 先转后移。
        # 这使本关节的连杆随自身旋转一起运动(肩抬升会抬起上臂),
        # 与 T·Rot(q)·Trans(link) 的教科书顺序一致。若写 T·Trans(link)·Rot(q)
        # (link 在旋转前的坐标系里), 连杆不会随关节转动, 物理上错误。
        step[:3, 3] = R @ joint.link
        T = T @ step
    return T[:3, 3].copy(), T[:3, :3].copy()


def numeric_jacobian(model: RobotModel, q: Sequence[float], delta: float = 1e-6) -> np.ndarray:
    """末端位置对关节角的数值雅可比(3 x dof)，中心差分。

    J = ∂p/∂q，用于 IK。对 5 自由度小规模系统足够快且直观。
    """
    q = np.asarray(q, dtype=float)
    J = np.zeros((3, model.dof))
    for i in range(model.dof):
        q_plus = q.copy(); q_plus[i] += delta
        q_minus = q.copy(); q_minus[i] -= delta
        p_plus, _ = fk(model, q_plus)
        p_minus, _ = fk(model, q_minus)
        J[:, i] = (p_plus - p_minus) / (2.0 * delta)
    return J


# ---------------------------------------------------------------------------
# 逆向运动学 IK（阻尼最小二乘）
# ---------------------------------------------------------------------------

def solve_ik(
    model: RobotModel,
    target_pos: Sequence[float],
    q0: Sequence[float] | None = None,
    tol: float = 1e-3,
    max_iter: int = 100,
    damping: float = 1e-3,
    restarts: int = 12,
    collision_guard: bool | None = None,
) -> Optional[np.ndarray]:
    """逆向运动学：求关节角 q, 使 FK(q) 的末端位置 ≈ target_pos。

    参数:
        target_pos: 目标位置 (x, y, z), base frame, meter。
        q0: 迭代初值。默认用 home。传入当前关节角可保证解连续(贴近当前姿态)。
        tol: 位置收敛阈值 (meter)。
        max_iter: 每次 DLS 求解的最大迭代次数。
        damping: DLS 阻尼系数。
        restarts: q0 种子失败后, 额外确定性重启次数。
    返回:
        收敛时的 q (已夹在关节限位内); 不收敛或不可达返回 None。

    重启策略(Phase 3 物理集成测试证明: 单种子 DLS 对桌面低目标是局部极小,
    会漏掉真实可达点): q0 失败后, 从一组**确定性**种子(全零、home、
    pan 对准目标的中位构型 + 固定随机种子采样)重试, 返回「离 q0 关节距离最近」
    的收敛解, 尽量贴近当前姿态, 避免大幅甩动。全程无随机性。

    Phase 3 碰撞守卫(collision_guard=True): 物理后端用。候选解须通过
    link_chain_clear(连杆不入地/不撞底座), 并额外加入「工具朝下」种子族
    (正 lift 会把臂折进底座, 默认种子常解出物理不可行解)。默认参数取自
    RobotModel, 不传则保持 Phase 1/2 行为。
    """
    guard = model.collision_guard if collision_guard is None else collision_guard
    target = np.asarray(target_pos, dtype=float)
    if target.shape != (3,):
        raise ValueError(f"target_pos 必须是 (3,) 向量, 实际 {target.shape}")

    q_init = np.asarray(q0, dtype=float).copy() if q0 is not None else model.q_home()
    if q_init.shape != (model.dof,):
        raise ValueError(f"q0 长度必须是 {model.dof}, 实际 {q_init.shape}")
    q_init = np.clip(q_init, model.q_min(), model.q_max())

    best: Optional[np.ndarray] = None
    best_dist = float("inf")

    def consider(seed: np.ndarray) -> None:
        nonlocal best, best_dist
        q = _dls_solve(model, target, seed, tol, max_iter, damping)
        if q is None:
            return
        # 已夹限位; 位置再确认一次(数值误差在 tol*10 内视为成功)
        p, _ = fk(model, q)
        if np.linalg.norm(p - target) <= tol * 10.0:
            if guard and not link_chain_clear(model, q):
                return
            d = float(np.linalg.norm(q - q_init))
            if d < best_dist:
                best, best_dist = q, d

    consider(q_init)
    if best is not None:
        return best

    # 确定性重启种子: 覆盖不同工作空间分区
    seeds = [np.zeros(model.dof), model.q_home()]
    # pan 对准目标方向、其余关节取常见折叠位置的中位构型
    az = float(np.arctan2(target[1], target[0]))
    for lift, elbow, wf in ((-0.5, -1.5, 0.5), (1.2, -1.2, 1.0),
                            (1.5, -2.4, 0.6), (0.3, -1.8, -0.6)):
        seeds.append(np.clip(np.array([az, lift, elbow, wf, 0.0]),
                             model.q_min(), model.q_max()))
    if guard:
        seeds.extend(_tool_down_seeds(target, model))
    rng = np.random.default_rng(0)
    for _ in range(max(0, restarts - len(seeds))):
        seeds.append(rng.uniform(model.q_min(), model.q_max()))
    for s in seeds:
        consider(s)
    return best


def _dls_solve(
    model: RobotModel,
    target: np.ndarray,
    q0: np.ndarray,
    tol: float,
    max_iter: int,
    damping: float,
) -> Optional[np.ndarray]:
    """单种子 DLS 求解: 迭代 max_iter 次, 收敛(误差 < tol)或宽松(tol*10)返回 q。"""
    q = np.clip(q0, model.q_min(), model.q_max())
    identity3 = np.eye(3)
    for _ in range(max_iter):
        p, _ = fk(model, q)
        err = target - p
        if np.linalg.norm(err) < tol:
            return q

        J = numeric_jacobian(model, q)
        # DLS: Δq = Jᵀ (J Jᵀ + λ²I)⁻¹ e
        dq = J.T @ np.linalg.solve(J @ J.T + damping**2 * identity3, err)

        # 限制单步关节增量(防止大旋转导致发散/穿限位)
        step = min(1.0, 0.5 / (np.linalg.norm(dq) + 1e-9))
        q = q + step * dq
        q = np.clip(q, model.q_min(), model.q_max())

        if not np.all(np.isfinite(q)):
            return None

    p, _ = fk(model, q)
    if np.linalg.norm(target - p) < tol * 10.0:
        return q
    return None


def link_chain_clear(
    model: RobotModel,
    q: Sequence[float],
    base_radius: float | None = None,
    base_z_hi: float | None = None,
    link_radius: float | None = None,
    table_margin: float = 0.005,
) -> bool:
    """物理可行性检查(Phase 3 集成测试证明: 折叠式 IK 解会撞底座, 伺服卡死)。

    沿 FK 链检查每段可动连杆(跳过 shoulder_pan 固定立柱段):
      - 连杆下边缘不低于桌面(table_top_z_m - margin);
      - 连杆中心线不进入底座圆柱(base_radius 为柱半径, 阈值 = base_radius
        + link_radius, 已含连杆半厚裕量)。
    纯几何、确定性, 供 solve_ik 的碰撞守卫过滤候选解。默认参数取自 RobotModel。

    实现注意(Phase 3 修正): 旧实现把「水平最近点参数 t」错除以 3D 长度平方
    (d@d), 且只检查单个点、不看线段在 z 带内的那一段, 导致:
      - 折叠式解(上臂贴着立柱下摆)被判为可行 -> 物理撞底座;
      - 而一些贴着立柱顶部的可行姿势又可能被误判。
    正确做法: 对每条线段求它与 z 带 [-lr, z_hi+lr] 的交集 t∈[t_lo,t_hi],
    再在该区间内取到 z 轴的水平最近点(t 用水平分量长度 dx²+dy² 归一)。
    """
    base_r = model.base_radius_m if base_radius is None else base_radius
    z_hi = model.base_z_hi_m if base_z_hi is None else base_z_hi
    lr = model.link_radius_m if link_radius is None else link_radius
    q = np.asarray(q, dtype=float)

    T = np.eye(4)
    segs: list[tuple[np.ndarray, np.ndarray]] = []
    zmin = float("inf")
    for j, qi in zip(model.joints, q):
        origin = T[:3, 3].copy()
        R = rot_matrix(j.axis, float(qi))
        step = np.eye(4)
        step[:3, :3] = R
        step[:3, 3] = R @ j.link
        T = T @ step
        segs.append((origin, T[:3, 3].copy()))
        zmin = min(zmin, float(T[2, 3]))

    # 连杆下边缘不低于桌面
    if zmin - lr < model.table_top_z_m - table_margin:
        return False

    z_low, z_high = -lr, z_hi + lr
    # 跳过 index 0(shoulder_pan 固定立柱段本身)
    for a, b in segs[1:]:
        d = b - a
        dz = float(d[2])
        # 该段中心线经过 z 带内的 t 区间 [t_lo, t_hi]
        if abs(dz) < 1e-12:
            if not (z_low <= a[2] <= z_high):
                continue
            t_lo, t_hi = 0.0, 1.0
        else:
            t_at_zlo = (z_low - a[2]) / dz
            t_at_zhi = (z_high - a[2]) / dz
            t_lo = max(0.0, min(t_at_zlo, t_at_zhi))
            t_hi = min(1.0, max(t_at_zlo, t_at_zhi))
            if t_hi < t_lo:
                continue
        # 底座圆柱轴在原点(0,0): 在 [t_lo,t_hi] 内求水平最近点(凸函数, 最值在钳制后 t*)
        dxy2 = float(d[0] * d[0] + d[1] * d[1])
        if dxy2 < 1e-12:
            t_star = t_lo
        else:
            t_star = -float(a[0] * d[0] + a[1] * d[1]) / dxy2
            t_star = min(max(t_star, t_lo), t_hi)
        p = a + t_star * d
        if np.hypot(p[0], p[1]) < base_r + lr:
            return False
    return True


def _tool_down_seeds(target: np.ndarray, model: RobotModel) -> list[np.ndarray]:
    """「工具朝下」种子族: lift+elbow+wflex ≈ 0 时工具竖直朝下。

    物理后端需要这类姿势(正 lift 会把臂折进底座, 见 Phase 3 集成测试);
    默认解不了低桌面目标时靠这组种子覆盖「从上方垂下」的工作空间分区。
    """
    az = float(np.arctan2(target[1], target[0]))
    out: list[np.ndarray] = []
    for lift in np.linspace(-1.0, 1.4, 13):
        for elbow in (-0.6, -1.2, -1.8, -2.2):
            wf = -(lift + elbow)
            if not (model.joints[3].limit_min <= wf <= model.joints[3].limit_max):
                continue
            out.append(np.clip(np.array([az, lift, elbow, wf, 0.0]),
                               model.q_min(), model.q_max()))
    return out


def is_reachable(model: RobotModel, target_pos: Sequence[float], margin: float = 0.005) -> bool:
    """快速可达性预检：目标到 base 原点的距离是否超过可达上界。不代替 IK。"""
    target = np.asarray(target_pos, dtype=float)
    return float(np.linalg.norm(target)) <= model.max_reach_m + margin


def pose_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """两点欧氏距离 (meter)，用于位置误差计算。"""
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))
