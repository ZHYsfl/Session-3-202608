"""机械臂模块独立 Demo（arm-only，不依赖 Voice Agent）: `python scripts/run_arm_demo.py`

打开 MuJoCo 真 3D 物理窗口（SO-101 + 桌子 + red/yellow/white 物块），然后**自动顺序演示**
机械臂模块自己的四个工具，每一步的返回字符串打印在终端：

    1. get_current_coordinates
    2. move_to_coordinates
    3. grab_the_block(red)
    4. get_current_coordinates
    5. move_to_coordinates（夹着物块移动到另一位置）
    6. release_the_block

演示结束后 GUI 保持打开，可用键盘继续操作：
    1 / 2 / 3  抓取 red / yellow / white
    R          释放当前物块
    H          重置场景（机械臂回 home，物块回到预设）
    ESC / 关闭窗口  退出

同时启动 arm REST 网关 http://127.0.0.1:8000（与 run_3d_sim.py 一致），供外部联调观察。

参数:
    --headless   无显示环境（无 GL）时自动降级：不启 3D 窗口，只跑 REST + 无窗口物理
    --fast       快速模式（不 sleep、控制即时推进），用于冒烟测试/无头 CI
"""
from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from pathlib import Path

# 允许从项目根目录直接运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from arm_skill_server.config import ControlConfig, load_all
from arm_skill_server.queue.in_memory_queue import InMemoryQueue
from arm_skill_server.runtime.factory import build_runtime
from arm_skill_server.runtime.state import state
from arm_skill_server.skills.get_coordinates import get_current_coordinates
from arm_skill_server.skills.grab_block import grab_the_block
from arm_skill_server.skills.move_to_coordinates import move_to_coordinates
from arm_skill_server.skills.release_block import release_the_block

# GLFW 键码（与 ASCII 一致）
_KEY = {49: ("grab", "red"), 50: ("grab", "yellow"), 51: ("grab", "white"),
        114: ("release", None), 104: ("reset", None)}

_DT = 0.02                 # 控制步长(s), 与 control.yaml dt 一致(50Hz)


def _override(control: ControlConfig, fast: bool) -> ControlConfig:
    """GUI Demo 必须用 mujoco 后端 + realtime；配置不是时打印提示并就地覆盖（不写回）。"""
    if control.backend_mode != "mujoco":
        print(f"  [提示] backend_mode={control.backend_mode!r}, 3D 演示需要 mujoco, 已临时切换")
        control = ControlConfig(**{**control.__dict__, "backend_mode": "mujoco"})
    if not fast and control.pacing != "realtime":
        print(f"  [提示] pacing={control.pacing!r}, 3D 演示需要 realtime, 已临时切换")
        control = ControlConfig(**{**control.__dict__, "pacing": "realtime"})
    return control


def run_demo_sequence(backend, pause: float, done: "threading.Event | None" = None) -> None:
    """按 6 步顺序演示四个工具，每步打印返回字符串。pause 为步骤间暂停（秒）。

    done: 可选。演示完成时置位（headless 冒烟测试据此退出）。
    """
    def show(i: int, title: str, fn) -> None:
        print(f"\n[演示 {i}/6] {title}")
        time.sleep(pause)
        print(f"  返回: {fn()}")

    show(1, "get_current_coordinates（获取坐标）", get_current_coordinates)
    show(2, "move_to_coordinates(0.25, -0.05, 0.15)", lambda: move_to_coordinates("0.25", "-0.05", "0.15"))
    show(3, "grab_the_block('red')（抓红色）", lambda: grab_the_block("red"))
    show(4, "get_current_coordinates（抓取后坐标）", get_current_coordinates)
    show(5, "move_to_coordinates(0.32, -0.05, 0.20)（夹着物块移动）",
         lambda: move_to_coordinates("0.32", "-0.05", "0.20"))
    show(6, "release_the_block（释放）", release_the_block)
    print("\n[演示完成] 机械臂模块四个工具顺序演示结束。")
    if done is not None:
        done.set()
    print("可用键盘继续操作: 1=抓红 2=抓黄 3=抓白  R=释放  H=重置  关闭窗口=退出")


def _command_worker(cmd_queue: "queue.Queue[tuple[str, str | None]]", backend) -> None:
    """消费键盘命令并执行对应 skill（串行；与 REST 共用 state 动作锁）。"""
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
        except Exception as e:
            print(f"   !! 命令失败: {type(e).__name__}: {e}")


def _start_rest_server(app) -> None:
    import uvicorn
    threading.Thread(target=uvicorn.run, args=(app,),
                     kwargs={"host": "127.0.0.1", "port": 8000,
                             "log_level": "warning"},
                     daemon=True).start()
    print("  REST API   : http://127.0.0.1:8000 (后台线程)")


def main() -> None:
    ap = argparse.ArgumentParser(description="机械臂模块独立 3D 演示（arm-only, 无 Voice Agent）")
    ap.add_argument("--headless", action="store_true", help="无 3D 窗口（无 GL 环境自动降级）")
    ap.add_argument("--fast", action="store_true", help="快速模式（冒烟测试/无头 CI）")
    args = ap.parse_args()

    model, control, sim_cfg = load_all()
    control = _override(control, fast=args.fast)
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
    print("SO-101 机械臂模块独立演示 (MuJoCo 真 3D 物理, arm-only)")
    print("=" * 64)
    print(f"视觉: perception_mode={control.perception_mode}  抓取: grasp_mode={control.grasp_mode}")
    print("本演示只使用机械臂模块的四个工具，不涉及 Voice Agent。")

    cmd_q: "queue.Queue[tuple[str, str | None]]" = queue.Queue()
    threading.Thread(target=_command_worker, args=(cmd_q, backend), daemon=True).start()

    # 自动演示线程（一次性）
    pause = 0.0 if args.fast else 1.2
    demo_done: "threading.Event | None" = None
    if args.headless:
        demo_done = threading.Event()     # headless 冒烟测试: 演示完成即退出
    threading.Thread(target=run_demo_sequence,
                     args=(backend, pause, demo_done), daemon=True).start()

    viewer = None
    if not args.headless:
        try:
            import mujoco.viewer
            viewer = mujoco.viewer.launch_passive(
                backend.m, backend.d, key_callback=lambda k: cmd_q.put(_KEY[k]) if k in _KEY else None,
                show_left_ui=False, show_right_ui=False)
        except Exception as e:
            print(f"\n[警告] 无法打开 3D 窗口({e}), 降级为无窗口物理仿真(REST 仍可用)。")

    # 主循环：空闲步进物理 + 同步画面（动作执行期间由技能线程的控制器推进）
    headless_exit_at = None
    while True:
        if viewer is not None:
            if not viewer.is_running():
                break
        if headless_exit_at is not None and time.monotonic() > headless_exit_at:
            break
        try:
            if not state.action_in_progress():
                backend.step(_DT)
        except Exception as e:
            print(f"[警告] 物理步进异常: {e}")
            break
        # headless 冒烟测试: 演示完成后留 1s 让释放的物块落定, 再退出
        if demo_done is not None and demo_done.is_set() and headless_exit_at is None:
            headless_exit_at = time.monotonic() + 1.0
        t0 = time.monotonic()
        if viewer is not None:
            # MuJoCo mjData 非线程安全: viewer.sync() 内部 mj_copyDataVisual 在技能线程
            # 处于 mj_step 中(arena stack 占用)时会报 "mjData while stack is in use"
            # 崩溃(GUI 实测: grab 的夹爪 settle 连续步进 0.5s, 与 50Hz sync 必然相撞)。
            # 与 backend 物理步进/渲染线程共用同一把 _lock 串行化(见 simulator_camera)。
            with backend._lock:
                viewer.sync()
        time.sleep(max(0.0, _DT - (time.monotonic() - t0)))

    if viewer is not None:
        viewer.close()
    print("\nGUI 已关闭。")


if __name__ == "__main__":
    main()
