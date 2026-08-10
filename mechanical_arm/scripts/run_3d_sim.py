"""SO-101 真 3D 物理仿真 GUI(Phase 3 硬验收): `python scripts/run_3d_sim.py`

打开 MuJoCo 交互式 3D 窗口, 实时显示机械臂 + 物块的物理仿真(MuJoCo 真物理,
重力/接触/摩擦), 同时启动 REST API(http://127.0.0.1:8000)供 Voice Agent /
命令行驱动。释放的物块会真实地受重力落下。

架构:
    - 主线程: MuJoCo passive viewer 渲染循环。空闲时按 50Hz 推进 backend.step()
      (让落下的物块继续运动); 动作执行期间(技能线程持有执行权)跳过 step, 由
      技能线程的控制器按 realtime 推进, 主线程只 viewer.sync() 同步画面。
    - 技能线程: 消费键盘命令队列, 调用 skills(*grab/release/reset*)。
    - REST 线程: uvicorn 后台线程, 与键盘共用同一套 state 动作串行化。

键盘快捷键:
    1 / 2 / 3  抓取 red / yellow / white
    R          释放当前物块
    H          重置场景(机械臂回 home, 物块回到预设)
    ESC        退出(或关闭窗口)

无显示环境(无 GL)时会自动降级: 不起 3D 窗口, 只跑 REST + 无窗口物理步进。
"""
from __future__ import annotations

import queue
import sys
import threading
import time
from pathlib import Path

# 允许从项目根目录直接运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

# -- MuJoCo(Phase 3 必需) ---------------------------------------------------
try:
    import mujoco
    import mujoco.viewer
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "Phase 3 需要 mujoco: python -m pip install mujoco\n"
        "GUI 3D 演示必须用 MuJoCo 物理后端(backend_mode=mujoco)。") from e

from arm_skill_server.config import ControlConfig, load_all
from arm_skill_server.queue.in_memory_queue import InMemoryQueue
from arm_skill_server.runtime.factory import build_runtime
from arm_skill_server.runtime.state import state
from arm_skill_server.skills.grab_block import grab_the_block
from arm_skill_server.skills.release_block import release_the_block

# GLFW 键码(与 ASCII 一致的可打印字符直接用 ASCII)
_KEY = {49: ("grab", "red"), 50: ("grab", "yellow"), 51: ("grab", "white"),
        114: ("release", None), 104: ("reset", None)}

_DT = 0.02                 # 控制步长(s), 与 control.yaml dt 一致(50Hz)


def _override_backend_mode(control: ControlConfig) -> ControlConfig:
    """GUI 必须用 mujoco 后端; 配置不是时打印提示并就地覆盖(不写回文件)。"""
    if control.backend_mode != "mujoco":
        print(f"  [提示] backend_mode={control.backend_mode!r}, GUI 需要 mujoco, 已临时切换")
        return ControlConfig(**{**control.__dict__, "backend_mode": "mujoco"})
    return control


def _command_worker(cmd_queue: "queue.Queue[str]", backend) -> None:
    """消费键盘命令并执行对应 skill(串行; 与 REST 共用 state 动作锁)。"""
    while True:
        cmd = cmd_queue.get()
        kind, arg = cmd
        try:
            if kind == "grab":
                print(f"\n>> 键盘: 抓取 {arg} ...")
                print(f"   结果: {grab_the_block(arg)}")
            elif kind == "release":
                print(f"\n>> 键盘: 释放物块 ...")
                print(f"   结果: {release_the_block()}")
            elif kind == "reset":
                print(f"\n>> 键盘: 重置场景(home + 物块预设)...")
                backend.reset("all")
                print(f"   完成")
        except Exception as e:  # 技能内部异常不应杀死 GUI
            print(f"   !! 命令失败: {type(e).__name__}: {e}")


def _start_rest_server(app) -> None:
    import uvicorn
    t = threading.Thread(target=uvicorn.run, args=(app,),
                         kwargs={"host": "127.0.0.1", "port": 8000,
                                 "log_level": "warning"},
                         daemon=True)
    t.start()
    print("  REST API   : http://127.0.0.1:8000 (后台线程)")


def main() -> None:
    model, control, sim_cfg = load_all()
    control = _override_backend_mode(control)
    # GUI 必须 realtime: 物理按真实时间推进(否则技能瞬间完成, 看不到动作)
    if control.pacing != "realtime":
        control = ControlConfig(**{**control.__dict__, "pacing": "realtime"})

    backend, camera, vision, vision_cfg = build_runtime(model, control, sim_cfg)
    backend.reset("all")
    queue_ = InMemoryQueue()

    from arm_skill_server.api.server import create_app
    app = create_app(backend, queue_, control, model,
                     vision=vision, camera=camera, vision_cfg=vision_cfg)
    state.configure(backend=backend, queue=queue_, control=control, model=model,
                    vision=vision, camera=camera, vision_cfg=vision_cfg)
    _start_rest_server(app)

    print("\n" + "=" * 64)
    print("SO-101 真 3D 物理仿真 (MuJoCo)")
    print("=" * 64)
    print(f"视觉: perception_mode={control.perception_mode}  "
          f"(hsv=HSV 经典 | tiny_detector=自训练 TinyDetector, 改 configs/control.yaml)")
    print(f"抓取: grasp_mode={control.grasp_mode}  "
          f"(scripted=脚本 IK 专家 | bc=自训练 BC 策略, 改 configs/control.yaml)")
    print("键盘: 1=抓红 2=抓黄 3=抓白  R=释放  H=重置场景  关闭窗口=退出")
    print("REST: POST /api/v1/grab_the_block {'color':'red'} 等(见 docs/api_of_embodied_tool.md)")

    # 命令队列 + 技能线程
    cmd_q: "queue.Queue[tuple[str, str | None]]" = queue.Queue()
    threading.Thread(target=_command_worker, args=(cmd_q, backend),
                     daemon=True).start()

    def on_key(keycode: int) -> None:
        cmd = _KEY.get(keycode)
        if cmd is not None:
            cmd_q.put(cmd)

    viewer = None
    try:
        viewer = mujoco.viewer.launch_passive(
            backend.m, backend.d, key_callback=on_key,
            show_left_ui=False, show_right_ui=False)
    except Exception as e:
        print(f"\n[警告] 无法打开 3D 窗口({e}), 降级为无窗口物理仿真 "
              f"(REST 仍可用)。")
        viewer = None

    # 主循环: 空闲步进物理 + 同步画面, 50Hz realtime
    while True:
        if viewer is not None:
            if not viewer.is_running():
                break
        try:
            if not state.action_in_progress():
                backend.step(_DT)
        except Exception as e:
            print(f"[警告] 物理步进异常: {e}")
            break
        t0 = time.monotonic()
        if viewer is not None:
            # MuJoCo mjData 非线程安全: viewer.sync() 内部 mj_copyDataVisual 在技能线程
            # 处于 mj_step 中(arena stack 占用)时报 "mjData while stack is in use"。
            # 与 backend 物理步进/渲染线程共用 backend._lock 串行化(同 run_arm_demo)。
            with backend._lock:
                viewer.sync()
        # 50Hz 节奏: 减去本次消耗, 剩余 sleep
        elapsed = time.monotonic() - t0
        time.sleep(max(0.0, _DT - elapsed))

    if viewer is not None:
        viewer.close()
    print("\nGUI 已关闭。")


if __name__ == "__main__":
    main()
