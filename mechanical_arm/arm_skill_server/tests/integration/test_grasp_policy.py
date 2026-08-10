"""Phase 5 BC 抓取策略验证。

标记 @pytest.mark.simulator -> 快速回归 `pytest -m "not simulator"` 不执行。
(与 Phase 4 test_tiny_detector.py 同约定; torch 是 Phase 4/5 硬依赖, 缺失时
grasp_policy.py 模块加载即抛明确 ImportError。)

覆盖:
    A. 纯函数: clamp_magnitude 限幅(方向不变, 零向量不变)。
    B. 网络: GraspPolicyNet 前向 obs(4) -> (dx,dy,dz,grip)(4)。
    C. checkpoint 加载: 缺失路径抛 FileNotFoundError(带「先训练」提示)。
    D. 下降端点去偏回归(Phase 5 稳定性修复核心):
       推理端在下降端点 obs=(0,0,-0.015,holding=0) 必须输出「停留」(|dz| 小,
       而非 false-lift 把末端带走); holding=1 时必须输出「抬升」(dz>0)。
    E. 端到端: grasp_mode="bc" 下 grab_the_block 真实夹住三色物块(需 checkpoint,
       缺失时跳过, 提示先运行 generate/train 脚本)。

依赖: mujoco(Phase 3) + torch(Phase 4/5) + 训练产物 models/grasp_policy.pt。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ...config import ControlConfig
from ...queue.in_memory_queue import InMemoryQueue
from ...runtime.state import state
from ...skills.grab_block import _GRASP_OK, grab_the_block
from ...skills.grasp_policy import BCGraspPolicy, GraspPolicyNet, clamp_magnitude

pytestmark = pytest.mark.simulator

_DESCEND_ENDPOINT_OBS_Z = -0.015   # block_z(0.015) - D_z(0.030), 见模块 docstring


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _policy_path() -> Path:
    from ...config import load_control_config
    return _project_root() / load_control_config().grasp_policy_checkpoint


# ---------------------------------------------------------------------------
# A. 纯函数
# ---------------------------------------------------------------------------

def test_clamp_magnitude_pure():
    """限幅: 超幅值截到 max_m 且方向不变; 未超/零向量不变。"""
    v = np.array([0.6, 0.8, 0.0])
    c = clamp_magnitude(v, 0.5)
    assert abs(float(np.linalg.norm(c)) - 0.5) < 1e-6
    assert c[1] / c[0] == pytest.approx(0.8 / 0.6)   # 方向不变
    # 未超幅值: 原样返回(同一对象)
    assert clamp_magnitude(np.array([0.1, 0.0, 0.0]), 0.5)[0] == pytest.approx(0.1)
    # 零向量: 不产生 NaN
    z = clamp_magnitude(np.zeros(3), 0.5)
    assert np.all(np.isfinite(z))


# ---------------------------------------------------------------------------
# B. 网络
# ---------------------------------------------------------------------------

def test_grasp_policy_net_forward():
    """GraspPolicyNet: obs(4,) -> 输出(4,)(位移 3 + grip logit 1)。"""
    net = GraspPolicyNet()
    import torch
    out = net(torch.zeros(1, 4))
    assert out.shape == (1, 4)
    assert np.all(np.isfinite(out.detach().numpy()))


# ---------------------------------------------------------------------------
# C. checkpoint 加载
# ---------------------------------------------------------------------------

def test_bc_policy_checkpoint_missing_raises():
    """缺失 checkpoint -> FileNotFoundError, 且错误信息带「先训练」提示。"""
    with pytest.raises(FileNotFoundError, match="先运行"):
        BCGraspPolicy(_project_root() / "models" / "no_such_policy.pt")


# ---------------------------------------------------------------------------
# D. 下降端点去偏回归(Phase 5 稳定性修复)
# ---------------------------------------------------------------------------

def test_bc_policy_descend_endpoint_no_false_lift():
    """下降端点 holding 标志必须干净区分停留/抬升(无 false-lift)。

    Phase 5 标定 bug: 数据在「下降端点(未持有)」稀疏, 网络把下降端点样本与
    抬升起点样本平均 -> 在下降端点输出 lift 位移(d_z≈+0.08), 把已到下降点的
    末端带走, 打断闭夹 settle 计数(需二次下降才闭夹, 时序相关 flaky)。修复 =
    稠密采样下降端点(holding=0, d≈0) + 几何闭夹候选就地稳住。本测试直接探
    策略输出, 断言该 bug 不复现:
        holding=0 -> |dz| 小(停留, 等闭夹); holding=1 -> dz>0(抬升)。
    """
    cp = _policy_path()
    if not cp.exists():
        pytest.skip("BC 策略 checkpoint 缺失: 先运行 "
                    "scripts/generate_grasp_dataset.py && scripts/train_grasp_policy.py")
    policy = BCGraspPolicy(cp)

    d_stay, _ = policy.act(np.array([0.0, 0.0, _DESCEND_ENDPOINT_OBS_Z, 0.0]))
    assert abs(float(d_stay[2])) < 0.01, \
        f"下降端点(未持有)应停留, 实际 d_z={d_stay[2]:+.4f}(false-lift)"

    d_lift, _ = policy.act(np.array([0.0, 0.0, _DESCEND_ENDPOINT_OBS_Z, 1.0]))
    assert d_lift[2] > 0.05, \
        f"下降端点(已持有)应抬升, 实际 d_z={d_lift[2]:+.4f}"


# ---------------------------------------------------------------------------
# E. BC 模式端到端
# ---------------------------------------------------------------------------

def test_grab_bc_mode_end_to_end(mujoco_env):
    """grasp_mode="bc" 下 grab_the_block 真实夹住三色物块。

    目标坐标来自 ground_truth(Phase 1 行为, 测试默认); 执行路径完全走
    run_bc_grasp 的逐 tick 策略闭环。断言真的夹住 + 抬离桌面(反作弊)。
    """
    cp = _policy_path()
    if not cp.exists():
        pytest.skip("BC 策略 checkpoint 缺失: 先运行训练脚本")

    b = mujoco_env["backend"]
    model, sim_cfg = mujoco_env["model"], mujoco_env["sim_cfg"]
    control = ControlConfig(**{**mujoco_env["control"].__dict__, "grasp_mode": "bc"})
    queue = InMemoryQueue()
    state.configure(backend=b, queue=queue, control=control, model=model)

    for color, preset in (("red", "all"), ("yellow", "all"), ("white", "all")):
        b.reset(preset)
        assert grab_the_block(color) == _GRASP_OK, f"bc 模式应夹住 {color}"
        assert b.is_holding_object(), f"夹取 {color} 后必须真的夹着物块"
        _, _, z = b.cube_position(color)
        assert z > 0.05, f"{color} 应被抬离桌面: z={z:.3f}"
