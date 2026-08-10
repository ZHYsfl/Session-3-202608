"""Phase 6 最终 Demo: 双 Agent 全双工 + 3D 物理仿真(`python scripts/run_final_demo.py`)

完整管道(全部纯仿真, 无外部 API):
    语音(键盘输入=模拟STT) -> Voice Agent -> voice→arm 队列
        -> Arm Agent -> REST 工具(:8000 协议 6 工具)
        -> MuJoCo 3D 物理 -> SO-101 -> 视觉(camera2 交叉验证)
        -> 抓取/移动/释放 -> arm→voice 队列 -> Voice Agent 播报(模拟TTS)

全双工 + 实时中断(协议 §3, 人优先):
    - Voice Agent 说话(send_voice)不阻塞; 结果由独立 speaker 线程异步播报。
    - Arm Agent 执行一个工具期间, 若语音队列出现新指令, monitor 线程
      `state.request_cancel()` 立即中断当前动作(≈20ms), 如实上报, 再执行新指令。
    - "取消/停止" 只取消; "改抓黄色" 取消后执行新命令。

进程内组件:
    - 主线程: MuJoCo passive viewer 渲染循环(空闲 50Hz 步进物理)
    - Arm REST :8000 + Voice 网关 :8001(uvicorn 后台线程, 共享同一队列)
    - Voice Agent 线程(语音输入 -> 队列) + speaker 线程(队列 -> 播报)
    - Arm Agent 主循环线程(队列 -> REST 工具 -> 结果) + monitor 线程(中断)

用法:
    python scripts/run_final_demo.py            # 默认: GUI + realtime
    python scripts/run_final_demo.py --headless # 无 3D 窗口(无 GL 环境降级)
    python scripts/run_final_demo.py --fast     # fast 节奏(自动化/联调, 技能瞬间完成)

退出: 关闭 3D 窗口 / ESC / Ctrl+C。
"""
from __future__ import annotations

import queue
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arm_skill_server.config import ControlConfig, load_all
from arm_skill_server.queue.in_memory_queue import InMemoryQueue
from arm_skill_server.runtime.factory import build_runtime
from arm_skill_server.runtime.state import state
from arm_skill_server.skills.grab_block import grab_the_block
from arm_skill_server.skills.release_block import release_the_block

try:
    import mujoco
    import mujoco.viewer
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "Phase 6 最终 Demo 需要 mujoco: python -m pip install mujoco\n"
        "3D 演示必须用 MuJoCo 物理后端(backend_mode=mujoco)。") from e

# GLFW 键码
_KEY = {49: ("grab", "red"), 50: ("grab", "yellow"), 51: ("grab", "white"),
        114: ("release", None), 104: ("reset", None)}

_DT = 0.02                 # 控制步长(s), 与 control.yaml dt 一致(50Hz)

_BANNER = """\
============================================================
 SO-101 双 Agent 全双工最终 Demo (纯仿真)
============================================================
 管道: 语音 -> Voice Agent -> voice→arm 队列 -> Arm Agent
        -> REST 工具 -> MuJoCo 3D 物理 -> SO-101 -> 视觉(camera2)
        -> 抓取/移动/释放 -> arm→voice 队列 -> Voice Agent 播报
 语音输入: 在下方提示符输入(模拟 STT, 中英混合, 实时中断)
 Arm REST : http://127.0.0.1:8000  (6 个工具)
 Voice 网关: http://127.0.0.1:8001 (队列转发)
------------------------------------------------------------
 语音命令示例:
   抓红色的物块 / grab red / 拿黄色 / 捡起白块
   移动到 0.26 0.06 0.1 / move to 0.26,0.06,0.1
   释放 / 放下物块 / put down
   你的坐标在哪 / where are you
   取消 / 停止 / 改抓黄色(执行中改主意=实时中断)
   重置 / 复位
 键盘备选: 1=抓红 2=抓黄 3=抓白  R=释放  H=重置  关闭窗口=退出
============================================================
"""


def _override_backend_mode(control: ControlConfig) -> ControlConfig:
    if control.backend_mode != "mujoco":
        print(f"  [提示] backend_mode={control.backend_mode!r}, GUI 需要 mujoco, 已临时切换")
        return ControlConfig(**{**control.__dict__, "backend_mode": "mujoco"})
    return control


def _start_rest_server(app, port: int, name: str) -> None:
    import uvicorn
    threading.Thread(target=uvicorn.run, args=(app,),
                     kwargs={"host": "127.0.0.1", "port": port,
                             "log_level": "warning"},
                     daemon=True).start()
    print(f"  {name}: http://127.0.0.1:{port} (后台线程)")


def _command_worker(cmd_queue: "queue.Queue[tuple[str, str | None]]", backend) -> None:
    """键盘命令(备选路径, 与 REST/语音共用 state 动作锁)。"""
    while True:
        kind, arg = cmd_queue.get()
        try:
            if kind == "grab":
                print(f"\n>> 键盘: 抓取 {arg} ...")
                print(f"   结果: {grab_the_block(arg)}")
            elif kind == "release":
                print(f"\n>> 键盘: 释放物块 ...")
                print(f"   结果: {release_the_block()}")
            elif kind == "reset":
                print(f"\n>> 键盘: 重置场景...")
                backend.reset("all")
                state.last_grabbed_color = None
                print(f"   完成")
        except Exception as e:
            print(f"   !! 命令失败: {type(e).__name__}: {e}")


def _voice_input_loop(voice_agent) -> None:
    """语音输入线程: 键盘输入 = 模拟语音(模拟 STT)。"""
    while True:
        try:
            text = input("\n语音> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        if text.lower() in ("quit", "exit", "退出"):
            print("退出语音输入。")
            break
        action = voice_agent.submit_voice(text)
        if action == "cancel":
            print("  (已发取消信号: 执行中会立即中断, 空闲则忽略)")


def main() -> None:
    headless = "--headless" in sys.argv
    fast = "--fast" in sys.argv

    model, control, sim_cfg = load_all()
    control = _override_backend_mode(control)
    if fast:
        control = ControlConfig(**{**control.__dict__, "pacing": "fast"})
    elif control.pacing != "realtime":
        control = ControlConfig(**{**control.__dict__, "pacing": "realtime"})

    backend, camera, vision, vision_cfg = build_runtime(model, control, sim_cfg)
    backend.reset("all")
    queue_ = InMemoryQueue()

    from arm_skill_server.api.server import create_app
    from arm_skill_server.api.voice_server import create_voice_app
    from arm_skill_server.voice import ArmAgent, HttpTransport, VoiceAgent

    app = create_app(backend, queue_, control, model,
                     vision=vision, camera=camera, vision_cfg=vision_cfg)
    voice_app = create_voice_app(queue_)
    state.configure(backend=backend, queue=queue_, control=control, model=model,
                    vision=vision, camera=camera, vision_cfg=vision_cfg)

    _start_rest_server(app, 8000, "Arm REST")
    _start_rest_server(voice_app, 8001, "Voice 网关")

    # 双 Agent(经真实 REST 路径通信; trust_env=False 直连本机网关)
    transport = HttpTransport(arm_url="http://127.0.0.1:8000",
                              voice_url="http://127.0.0.1:8001")
    voice_agent = VoiceAgent(transport)
    arm_agent = ArmAgent(
        transport,
        reset_fn=lambda: (backend.reset("all"), setattr(state, "last_grabbed_color", None)))
    voice_agent.start()
    arm_agent.start()
    threading.Thread(target=arm_agent.run, daemon=True).start()

    print("\n" + "=" * 60)
    print("SO-101 双 Agent 全双工最终 Demo (MuJoCo 真 3D 物理)")
    print("=" * 60)
    print(f"视觉: perception_mode={control.perception_mode}  "
          f"(hsv | tiny_detector, 改 configs/control.yaml)")
    print(f"抓取: grasp_mode={control.grasp_mode}  "
          f"(scripted | bc, 改 configs/control.yaml)")
    print(f"节奏: pacing={control.pacing}  3D 窗口={'off(headless)' if headless else 'on'}")
    print(_BANNER)

    # 键盘命令队列(备选)
    cmd_q: "queue.Queue[tuple[str, str | None]]" = queue.Queue()
    threading.Thread(target=_command_worker, args=(cmd_q, backend), daemon=True).start()

    def on_key(keycode: int) -> None:
        cmd = _KEY.get(keycode)
        if cmd is not None:
            cmd_q.put(cmd)

    viewer = None
    if not headless:
        try:
            viewer = mujoco.viewer.launch_passive(
                backend.m, backend.d, key_callback=on_key,
                show_left_ui=False, show_right_ui=False)
        except Exception as e:
            print(f"\n[警告] 无法打开 3D 窗口({e}), 降级为无窗口物理仿真(REST 仍可用)。")
            viewer = None

    # 语音输入线程
    threading.Thread(target=_voice_input_loop, args=(voice_agent,), daemon=True).start()

    # 主循环: 空闲步进物理 + 同步画面, 50Hz realtime
    try:
        while True:
            if viewer is not None and not viewer.is_running():
                break
            try:
                if not state.action_in_progress():
                    backend.step(_DT)
            except Exception as e:
                print(f"[警告] 物理步进异常: {e}")
                break
            t0 = time.monotonic()
            if viewer is not None:
                viewer.sync()
            elapsed = time.monotonic() - t0
            time.sleep(max(0.0, _DT - elapsed))
    except KeyboardInterrupt:
        pass

    if viewer is not None:
        viewer.close()
    print("\nDemo 已关闭。")


if __name__ == "__main__":
    main()
