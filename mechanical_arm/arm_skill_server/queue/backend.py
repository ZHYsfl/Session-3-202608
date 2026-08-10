"""QueueBackend 抽象：voice agent 与 arm agent 之间的两条 FIFO 队列。

对应协议文档 §3 队列与状态栏约定:
    message_from_voice_agent_queue (voice→arm):
        语音网关 send_to_arm_agent 生产; 本网关 get_message_from_voice_agent 消费。
    message_from_arm_agent_queue (arm→voice):
        本网关 send_to_voice_agent 生产; 语音网关 get_message_from_arm_agent 消费。

本仓库只包含 arm 侧(8000)。voice 侧方法(生产/消费的“另一半”)也抽象出来,
以便日后对接同一个队列后端(如 Redis/文件)。当前进程内实现见 in_memory_queue.py。
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class QueueBackend(ABC):
    # ---- voice → arm 队列 ------------------------------------------------
    @abstractmethod
    def push_voice_to_arm(self, content: str) -> None:
        """语音网关生产(voice → arm)。arm 侧 get_message_from_voice_agent 消费。"""

    @abstractmethod
    def pop_voice_to_arm_all(self) -> list[str]:
        """arm 侧消费: 一次性排空队列, 按入队顺序返回全部消息。"""

    @abstractmethod
    def voice_to_arm_size(self) -> int:
        """arm 侧查询队列长度(供 <queue_status> 状态栏注入)。"""

    # ---- arm → voice 队列 ------------------------------------------------
    @abstractmethod
    def push_arm_to_voice(self, content: str) -> None:
        """arm 侧生产(arm → voice)。voice 网关 get_message_from_arm_agent 消费。"""

    @abstractmethod
    def pop_arm_to_voice_all(self) -> list[str]:
        """voice 网关消费: 排空 arm→voice 队列。"""

    @abstractmethod
    def arm_to_voice_size(self) -> int:
        """arm→voice 队列长度。"""
