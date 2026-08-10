"""skill 层单元测试: 协议字符串、颜色校验、场景缺块、抓取/释放、不可达。"""
from __future__ import annotations

import re

import pytest

from ...skills import get_coordinates, grab_block, move_to_coordinates, release_block
from ...skills._common import CoordinateError, parse_coordinate

XYZ_RE = re.compile(r"^-?\d+\.\d{3},-?\d+\.\d{3},-?\d+\.\d{3}$")


def _reset(env, preset: str) -> None:
    env["backend"].reset(preset)


# ---------------------------------------------------------------------------
# get_current_coordinates
# ---------------------------------------------------------------------------

def test_get_coordinates_format(env) -> None:
    _reset(env, "all")
    r = get_coordinates.get_current_coordinates()
    assert r.startswith("我的坐标是")
    assert XYZ_RE.match(r[len("我的坐标是"):])


# ---------------------------------------------------------------------------
# 坐标解析
# ---------------------------------------------------------------------------

def test_parse_coordinate_ok() -> None:
    assert parse_coordinate("0.25", "x") == 0.25


@pytest.mark.parametrize("bad", ["abc", "1,2", "", "0x10", None])
def test_parse_coordinate_rejects_non_numeric(bad) -> None:
    with pytest.raises(CoordinateError):
        parse_coordinate(bad, "x")


def test_parse_coordinate_rejects_nan_inf() -> None:
    with pytest.raises(CoordinateError):
        parse_coordinate("nan", "x")
    with pytest.raises(CoordinateError):
        parse_coordinate("inf", "y")


# ---------------------------------------------------------------------------
# move_to_coordinates
# ---------------------------------------------------------------------------

def test_move_success(env) -> None:
    _reset(env, "all")
    r = move_to_coordinates.move_to_coordinates("0.20", "-0.05", "0.15")
    assert r.startswith("成功到达0.200,-0.050,0.150")


def test_move_unreachable_reports_error(env) -> None:
    """工作空间外目标: 返回 未到达 + 误差(不返回假成功)。"""
    _reset(env, "all")
    r = move_to_coordinates.move_to_coordinates("0.90", "0.90", "0.30")
    assert r.startswith("未到达")
    assert "，误差是" in r


def test_move_bad_parse_raises(env) -> None:
    _reset(env, "all")
    with pytest.raises(CoordinateError):
        move_to_coordinates.move_to_coordinates("abc", "0", "0")


# ---------------------------------------------------------------------------
# grab_the_block
# ---------------------------------------------------------------------------

def test_grab_invalid_color(env) -> None:
    _reset(env, "all")
    assert grab_block.grab_the_block("blue") == "没有这种颜色的物块，无法夹取"
    assert grab_block.grab_the_block("RED") == "没有这种颜色的物块，无法夹取"


def test_grab_color_not_present(env) -> None:
    """预设 red_only 时抓 yellow -> 无法夹取。"""
    _reset(env, "red_only")
    assert grab_block.grab_the_block("yellow") == "没有这种颜色的物块，无法夹取"


def test_grab_success(env) -> None:
    _reset(env, "all")
    r = grab_block.grab_the_block("red")
    assert r == "有这种颜色的物块，且夹取物块成功"
    assert env["backend"].is_holding_object()


def test_grab_unreachable_block_fails(env) -> None:
    """物块在机械臂工作空间外 -> 尝试夹取失败(不是假成功)。"""
    _reset(env, "unreachable_block")
    r = grab_block.grab_the_block("red")
    assert r == "有这种颜色的物块，但夹取物块失败"
    assert not env["backend"].is_holding_object()


# ---------------------------------------------------------------------------
# release_the_block
# ---------------------------------------------------------------------------

def test_release_nothing_held(env) -> None:
    _reset(env, "all")
    assert release_block.release_the_block() == "本身就没加起来物块"


def test_release_success(env) -> None:
    _reset(env, "all")
    grab_block.grab_the_block("red")
    assert env["backend"].is_holding_object()
    assert release_block.release_the_block() == "成功释放物块"
    assert not env["backend"].is_holding_object()


def test_release_failure_when_gripper_stuck(env) -> None:
    """夹爪故障(卡死): 释放失败, 返回协议要求的求助字符串。"""
    _reset(env, "all")
    grab_block.grab_the_block("red")
    assert env["backend"].is_holding_object()
    env["backend"].set_gripper_fault(True)
    assert release_block.release_the_block() == "释放物块失败，请用手直接拿出来物块"
    assert env["backend"].is_holding_object()
