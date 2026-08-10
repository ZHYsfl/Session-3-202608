"""voice: Phase 6 双 Agent 中「语音侧」纯仿真组件(零外部 API/密钥)。

Voice Agent 是**独立的第二个 Agent**, 与 Arm Agent 只通过两条 FIFO 队列 +
REST 工具通信(见 docs/api_of_voice_tools.md 与 async_dual_agent_system_design.md)。

本包提供:
    intent      确定性意图解析(仿真 LLM: 中英混合 -> 结构化命令)
    io          仿真语音 IO(键盘文本=STT, 终端播报=TTS)
    transport   双 Agent 通信传输层(进程内直连 / HTTP 网关)
    voice_agent Voice Agent(识别/理解/发命令/异步播报结果)
    arm_agent   Arm Agent 执行侧(消费命令/调工具/上报/实时中断)
"""
from .intent import Command, UnknownIntentError, parse_intent
from .io import SimulatedSTT, SimulatedTTS
from .transport import HttpTransport, InProcessTransport
from .voice_agent import VoiceAgent
from .arm_agent import ArmAgent

__all__ = [
    "Command",
    "UnknownIntentError",
    "parse_intent",
    "SimulatedSTT",
    "SimulatedTTS",
    "HttpTransport",
    "InProcessTransport",
    "VoiceAgent",
    "ArmAgent",
]
