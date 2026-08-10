"""轨迹插值：标量梯形速度剖面（加速 → 匀速 → 减速 → 精确着陆）。

对单个关节, 每一步返回受「速度上限 / 加速度上限」约束的速度指令。

核心数学 —— 离散制动包络:
    连续梯形剖面的减速段用 v_stop = sqrt(2·accel·|error|)。但在离散
    积分里 error 按矩形法(每步减去 v·dt)收缩, 连续包络会被“追不上”,
    导致最后一步需要一次过大的减速跳变。因此这里改用离散制动距离:

        从速度 v 出发, 每步减速 dv_max = accel·dt, 直到可以一步精确着陆:
        D(v) = m·v·dt − dv_max·dt·m(m−1)/2,   m = ceil(v/dv_max)

    (可一步着陆时 m=1, D(v)=v·dt; 每多一段刹车 m+1。)
    剩余距离 e 下允许的最大速度 v_safe(e) = D⁻¹(e):
        v_safe = (e + dv_max·dt·m(m−1)/2) / (m·dt)

    保证:
        1. |v| ≤ vmax;
        2. 每一步的速度变化 |Δv| ≤ accel·dt, 含最后一步精确着陆
           (比连续包络的 2·accel 上界更紧);
        3. 位置误差不越过目标(不超调);
        4. 收敛: 进入死区后一步落位到 error = 0。

单位: 角度 rad, 速度 rad/s, 加速度 rad/s², 时间 s。
"""
from __future__ import annotations

import math

# 收敛死区：|error| 小于该值视为到达(此时精确落位, 一步到零)
_DEADBAND_RAD = 1e-4


def _discrete_stop_speed(e: float, dt: float, dv_max: float) -> float:
    """离散制动包络的反函数 D⁻¹(e)：剩余距离 e 下可安全刹停的最大速度。

    参数:
        e:      剩余距离 (rad), 非负。
        dt:     积分步长 (s)。
        dv_max: 单步最大速度变化 = accel·dt (rad/s)。
    返回:
        速度 v_safe (rad/s), 从该速度出发每步减速 dv_max 可无超调地停在目标。
    """
    # D(m·dv_max) = dv_max·dt·m(m+1)/2 → 由 e 反解 m = ceil((-1+√(1+8e/(dv_max·dt)))/2)
    m = max(1, int(math.ceil((-1.0 + math.sqrt(1.0 + 8.0 * e / (dv_max * dt))) / 2.0)))
    if e > dv_max * dt * m * (m + 1) / 2.0:
        m += 1  # 数值取整兜底
    return (e + dv_max * dt * m * (m - 1) / 2.0) / (m * dt)


def next_ramped_velocity(
    error: float,
    v_prev: float,
    dt: float,
    vmax: float,
    accel: float,
    deadband: float = _DEADBAND_RAD,
) -> float:
    """根据当前位置误差, 计算本步的关节速度指令。

    输入:
        error:   目标角 - 当前角 (rad), 可为负。
        v_prev:  上一步速度 (rad/s)。
        dt:      积分步长 (s)。
        vmax:    关节速度上限 (rad/s)。
        accel:   关节加速度上限 (rad/s²)。
        deadband: 死区 (rad), 误差小于该值视为到达并精确落位。
    返回:
        本步速度指令 (rad/s)。
    """
    if error == 0.0:
        return 0.0
    e = abs(error)
    sign = math.copysign(1.0, error)
    dv_max = accel * dt

    # 死区: 误差已收敛, 一步精确落位(覆盖剩余误差)。
    # 若残余速度过大(病态状态, 正常包络不会产生), 直接刹停兜底。
    if e < deadband:
        v_land = sign * e / dt
        return v_land if abs(v_land - v_prev) <= dv_max else 0.0

    # 1) 目标速度: 离散制动包络(减速段随剩余距离收紧)
    v_target = sign * min(vmax, _discrete_stop_speed(e, dt, dv_max))

    # 2) 加速度受限: 单步最多变化 accel·dt
    v = v_prev + min(max(v_target - v_prev, -dv_max), dv_max)

    # 3) 精确着陆安全兜底: 本步会越过目标时只走剩余距离(包络保证跳变 ≤ dv_max)
    if abs(v) * dt > e:
        return sign * e / dt
    return v
