"""两个双 Agent 通信工具(协议 §2.5 / §2.6)。

send_to_voice_agent:        生产到 arm→voice 队列。
get_message_from_voice_agent:排空 voice→arm 队列。

队列实现是抽象的(QueueBackend), 进程内实现为 InMemoryQueue。
返回字符串严格匹配协议。
"""
from __future__ import annotations

from ..runtime.state import state


def send_to_voice_agent(content: str) -> str:
    """把一条消息追加到 message_from_arm_agent_queue 队尾。"""
    state.require_queue().push_arm_to_voice(content)
    return "发送成功"


def get_message_from_voice_agent() -> str:
    """排空 message_from_voice_agent_queue, 一次性取出全部消息。"""
    msgs = state.require_queue().pop_voice_to_arm_all()
    if not msgs:
        return "当前没有新消息"
    return "all_messages_from_voice_agent:" + ";".join(msgs)
