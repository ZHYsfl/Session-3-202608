"""队列单元测试: FIFO 顺序/排空 + 两个通信工具的协议字符串。"""
from __future__ import annotations

from ...queue.in_memory_queue import InMemoryQueue
from ...skills import voice_queue


def test_fifo_order_and_drain() -> None:
    q = InMemoryQueue()
    assert q.voice_to_arm_size() == 0

    q.push_voice_to_arm("a")
    q.push_voice_to_arm("b")
    q.push_voice_to_arm("c")
    assert q.voice_to_arm_size() == 3

    assert q.pop_voice_to_arm_all() == ["a", "b", "c"]   # 保持入队顺序
    assert q.voice_to_arm_size() == 0                    # 排空


def test_arm_to_voice_separate() -> None:
    q = InMemoryQueue()
    q.push_arm_to_voice("A")
    assert q.arm_to_voice_size() == 1
    assert q.voice_to_arm_size() == 0                    # 两条队列互不干扰
    assert q.pop_arm_to_voice_all() == ["A"]


def test_send_and_receive_strings(env) -> None:
    """协议 §2.5/§2.6 返回字符串严格匹配。"""
    assert voice_queue.send_to_voice_agent("任务完成") == "发送成功"

    # voice→arm 队列为空
    assert voice_queue.get_message_from_voice_agent() == "当前没有新消息"

    env["queue"].push_voice_to_arm("用户改主意了，请改抓 yellow 物块")
    env["queue"].push_voice_to_arm("目标位置不变")
    assert voice_queue.get_message_from_voice_agent() == (
        "all_messages_from_voice_agent:用户改主意了，请改抓 yellow 物块;目标位置不变"
    )
