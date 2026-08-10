"""Arm Agent: 机械臂执行侧 Agent(与 Voice Agent 是**两个独立 Agent**)。

职责:
    - 消费 voice→arm 队列里的语音命令(经 arm 网关 :8000 的
      get_message_from_voice_agent / send_to_voice_agent)
    - 用协议 §2 的 6 个工具执行; 把结果发回 arm→voice 队列
      (Voice Agent 经 voice 网关 :8001 排空并播报)
    - 实时中断(协议 §3 队列与状态栏约定 + 人优先):
          Arm Agent 执行一个工具期间(动作进行中), 若 voice→arm 队列出现
          新指令, monitor 线程调用 state.request_cancel() 取消当前动作
          (控制循环约 20ms 内停止), 如实上报「动作被新指令中断(返回: 原结果)」,
          随后继续执行新指令。
          "取消/停止" -> 只取消当前动作(无新任务)。
          "改主意"(如"改抓黄色") -> 取消当前动作后执行新命令。

诚实原则: 中断时如实上报原工具返回的结果字符串; 绝不说「已下发命令」等于
「动作成功」。工具结果原样来自协议字符串(抓取成功/失败/释放成功等)。
"""
from __future__ import annotations

import threading
import time

from ..runtime.state import state
from .intent import UnknownIntentError, parse_intent


class ArmAgent:
    def __init__(self, transport, reset_fn=None, poll_s: float = 0.02,
                 monitor_s: float = 0.01) -> None:
        self._transport = transport
        self._reset_fn = reset_fn            # 可选: 场景重置回调(演示端注入 backend.reset)
        self._poll_s = poll_s
        self._monitor_s = monitor_s
        self._stop = threading.Event()
        self._interrupt_pending = False
        self._monitor = threading.Thread(target=self._monitor_loop,
                                         name="arm-monitor", daemon=True)

    def start(self) -> None:
        """启动实时中断监控线程(必须在 run() 之前调用)。"""
        self._monitor.start()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        """执行主循环(阻塞)。在独立线程中运行。"""
        while not self._stop.is_set():
            try:
                self._tick()
            except Exception as e:  # noqa: BLE001 - 单次故障不杀 Agent 主循环
                try:
                    self._report(f"[arm agent 异常] {e.__class__.__name__}")
                except Exception:  # noqa: BLE001 - 上报失败也不该逃出线程
                    pass
            time.sleep(self._poll_s)

    # ---- 内部: 消费队列 -> 逐条执行 ------------------------------------
    def _tick(self) -> None:
        for text in self._transport.pop_voice_to_arm_all():
            self._handle_command(text)

    def _handle_command(self, text: str) -> None:
        try:
            cmd = parse_intent(text)
        except UnknownIntentError:
            self._report(f"没听懂: {text}")
            return

        if cmd.action == "cancel":
            state.request_cancel()
            self._report("已取消当前动作")
            return
        if cmd.action == "reset":
            if self._reset_fn is not None:
                self._reset_fn()
            self._report("场景已重置")
            return

        # 工具执行(阻塞; 期间 monitor 可能取消它)。每个新命令重置中断标记。
        self._interrupt_pending = False
        result = self._call_tool(cmd)
        if self._interrupt_pending:
            # 执行期间被 monitor 中断: 如实上报原工具结果 + 中断事实
            self._interrupt_pending = False
            self._report(f"动作被新指令中断(返回: {result})")
            return
        self._report(result)

    def _call_tool(self, cmd) -> str:
        if cmd.action == "grab":
            return self._transport.call_tool("grab_the_block", color=cmd.args["color"])
        if cmd.action == "release":
            return self._transport.call_tool("release_the_block")
        if cmd.action == "coordinates":
            return self._transport.call_tool("get_current_coordinates")
        if cmd.action == "move":
            a = cmd.args
            return self._transport.call_tool(
                "move_to_coordinates", x=_fmt(a["x"]), y=_fmt(a["y"]), z=_fmt(a["z"]))
        return f"未知动作: {cmd.action}"

    def _report(self, text: str) -> None:
        self._transport.push_arm_to_voice(text)

    # ---- 内部: 实时中断监控 -------------------------------------------
    def _monitor_loop(self) -> None:
        """动作进行中 且 队列出现新指令 -> 取消当前动作。

        不产生业务返回, 只发取消信号; 由正在执行的工具感知并尽快退出。
        """
        while not self._stop.is_set():
            try:
                if (state.action_in_progress()
                        and self._transport.voice_to_arm_size() > 0
                        and not self._interrupt_pending):
                    self._interrupt_pending = True
                    state.request_cancel()
            except Exception:  # noqa: BLE001 - 监控故障不杀线程
                pass
            time.sleep(self._monitor_s)


def _fmt(v: float) -> str:
    """坐标格式化: 保留 4 位小数(协议字段为字符串类型)。"""
    return f"{float(v):.4f}"
