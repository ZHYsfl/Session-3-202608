"""Voice Agent 意图解析(确定性 NLU)纯函数测试。

仿真 LLM 是纯函数: 中英混合语音文本 -> 结构化命令。全部离线, 无外部 API。
"""
from __future__ import annotations

import pytest

from ...voice.intent import Command, UnknownIntentError, parse_intent


def _cmd(action: str, **args) -> Command:
    return Command(action, args)


# ---- 抓取 -----------------------------------------------------------------

@pytest.mark.parametrize("text,color", [
    ("抓红色的物块", "red"),
    ("红色", "red"),                        # 兜底: 只有颜色词
    ("grab red", "red"),
    ("red block", "red"),
    ("拿黄色", "yellow"),
    ("yellow", "yellow"),
    ("捡起白块", "white"),
    ("pick up the white block", "white"),
    ("改抓黄色", "yellow"),                 # 改主意 -> 新抓取命令
    ("夹取红色物块", "red"),
])
def test_grab_intent(text: str, color: str) -> None:
    assert parse_intent(text) == _cmd("grab", color=color)


# ---- 移动 -----------------------------------------------------------------

@pytest.mark.parametrize("text,x,y,z", [
    ("移动到 0.26 0.06 0.1", 0.26, 0.06, 0.1),
    ("move to 0.26,0.06,0.1", 0.26, 0.06, 0.1),
    ("走到 0.24 -0.08 0.05", 0.24, -0.08, 0.05),
    ("把红色移到0.3 0 0.05", 0.3, 0.0, 0.05),   # 有颜色词但带坐标 -> 移动优先
    ("去 0.5 0.0 0.12", 0.5, 0.0, 0.12),
])
def test_move_intent(text: str, x: float, y: float, z: float) -> None:
    assert parse_intent(text) == _cmd("move", x=x, y=y, z=z)


# ---- 释放 / 坐标 / 取消 / 重置 ---------------------------------------------

@pytest.mark.parametrize("text", [
    "放下物块", "释放", "松开", "put down", "drop the block", "放下来",
])
def test_release_intent(text: str) -> None:
    assert parse_intent(text) == _cmd("release")


@pytest.mark.parametrize("text", [
    "你的坐标在哪", "where are you", "现在在哪", "当前位置", "位置",
])
def test_coordinates_intent(text: str) -> None:
    assert parse_intent(text) == _cmd("coordinates")


@pytest.mark.parametrize("text", [
    "取消", "停止", "别抓了", "中断", "stop", "cancel", "不用了",
])
def test_cancel_intent(text: str) -> None:
    assert parse_intent(text) == _cmd("cancel")


@pytest.mark.parametrize("text", ["重置", "复位", "reset", "恢复初始"])
def test_reset_intent(text: str) -> None:
    assert parse_intent(text) == _cmd("reset")


# ---- 优先级(歧义消解) -----------------------------------------------------

def test_cancel_wins_over_grab() -> None:
    # "取消" 含抓取动词 "取", 但取消语义必须优先
    assert parse_intent("取消抓红色").action == "cancel"


def test_cancel_wins_over_release() -> None:
    assert parse_intent("停止放下").action == "cancel"


def test_reset_wins_over_coordinates() -> None:
    # "重置" 含 "置"(位置), 但重置必须优先
    assert parse_intent("重置位置").action == "reset"


def test_coordinates_wins_over_move() -> None:
    # "在哪" 含移动动词 "去" 的变体? 这里验证 坐标 优先于 移动
    assert parse_intent("你去哪").action == "coordinates"


def test_move_needs_three_numbers() -> None:
    # 只有移动动词没有 3 个数字 -> 不是移动命令(落到抓取或 unknown)
    with pytest.raises(UnknownIntentError):
        parse_intent("移动到那里")


# ---- 未知 -----------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "", "   ", "今天天气怎么样", "你好", "asdf qwer",
])
def test_unknown_intent_raises(text: str) -> None:
    with pytest.raises(UnknownIntentError):
        parse_intent(text)
