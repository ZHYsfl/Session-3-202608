"""HTTP Final Regression: 从 coworker 调用方视角, 对四个 Arm Tool 做真实 HTTP 测试。

必须启动**真实 uvicorn + MuJoCo backend**, 通过 HTTP 请求测试(不是 unit test)。
本脚本同时持有 backend 句柄, 用来验证「API 返回值 vs 物理真实状态」的一致性。

覆盖 docs/api_of_embodied_tool.md 冻结协议的 §4 全部正例/负例:
    4.1 get_current_coordinates    -> HTTP 200, 返回坐标 ≈ ee_tip site_xpos
    4.2 move_to_coordinates 正常点  -> 真实运动(禁 teleport), 到达后在 tolerance 内
    4.3 move_to_coordinates 不可达点 -> code=0, 误差 = 真实 ‖ee - target‖₂
    4.4 参数错误                    -> 缺 x/y/z / 非数值 / extra field / 坏 JSON -> HTTP 400
    4.5 grab_the_block              -> red/yellow/white 成功; blue / 场景缺色 -> 没有这种颜色
    4.6 physical grasp              -> 真实接触咬合抬升, 不 teleport/焊接
    4.7 release_the_block           -> 没夹着 -> 本身就没加起来物块; 夹着 -> 重力释放成功

用法: python scripts/final_api_regression.py
"""
from __future__ import annotations

import json
import re
import socket
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Windows 控制台默认 GBK 无法编码 ‖ 与 ₂ 等字符, 强制 UTF-8 输出, 避免 UnicodeEncodeError。
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import httpx
import numpy as np

from arm_skill_server.api.server import create_app
from arm_skill_server.config import ControlConfig, load_all
from arm_skill_server.queue.in_memory_queue import InMemoryQueue
from arm_skill_server.runtime.factory import build_runtime
from arm_skill_server.robot.kinematics import pose_distance

_RESULTS: list[tuple[str, bool, str]] = []


def _check(name: str, ok: bool, detail: str = "") -> None:
    _RESULTS.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _start_server(backend, queue, control, model, vision, camera, vision_cfg, port: int):
    import uvicorn
    app = create_app(backend, queue, control, model,
                     vision=vision, camera=camera, vision_cfg=vision_cfg)
    t = threading.Thread(target=uvicorn.run, args=(app,),
                         kwargs={"host": "127.0.0.1", "port": port, "log_level": "warning"},
                         daemon=True)
    t.start()
    # 就绪轮询: httpx trust_env=False 绕过本机 Clash 代理
    base = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            r = httpx.get(f"{base}/api/v1/get_current_coordinates", timeout=0.5,
                          trust_env=False)
            if r.status_code == 200:
                return base
        except Exception:
            pass
        time.sleep(0.05)
    raise RuntimeError("服务器启动超时")


def _j(body) -> dict:
    return body


def _err_val(result: str) -> float:
    """从 "未到达{x,y,z}，误差是{error}" 解析误差数值。"""
    m = re.search(r"，误差是([0-9.]+)", result)
    assert m, f"无法从 result 解析误差: {result!r}"
    return float(m.group(1))


def main() -> None:
    model, control, sim_cfg = load_all()
    control = ControlConfig(**{**control.__dict__,
                               "backend_mode": "mujoco",   # 真 3D 物理
                               "pacing": "fast"})          # 回归用 fast(物理仍真实, 只是不 sleep)
    backend, camera, vision, vision_cfg = build_runtime(model, control, sim_cfg)
    backend.reset("all")
    queue = InMemoryQueue()
    base = _start_server(backend, queue, control, model, vision, camera, vision_cfg,
                         _free_port())
    print(f"real server on {base}  (backend={control.backend_mode} pacing={control.pacing})")
    C = httpx.Client(trust_env=False, timeout=60.0)

    # =====================================================================
    # 4.1 get_current_coordinates
    # =====================================================================
    print("\n===== 4.1 get_current_coordinates =====")
    r = C.get(f"{base}/api/v1/get_current_coordinates")
    body = r.json()
    _check("4.1 HTTP 200 + code=0", r.status_code == 200 and body["code"] == 0
           and body["error"] is None, f"HTTP {r.status_code}")
    result = body["result"]
    _check("4.1 result 形如 我的坐标是{x},{y},{z}",
           bool(re.match(r"^我的坐标是-?\d+\.\d+,-?\d+\.\d+,-?\d+\.\d+$", result)),
           result)
    api_xyz = np.array([float(v) for v in re.findall(r"-?\d+\.\d+", result)])
    ee = backend.get_end_effector_pose().position
    _check("4.1 API 坐标 ≈ MuJoCo ee_tip site_xpos",
           float(np.linalg.norm(api_xyz - ee)) < 0.005,
           f"api={api_xyz.round(4)} ee={ee.round(4)}")

    # =====================================================================
    # 4.2 move_to_coordinates 正常点
    # =====================================================================
    print("\n===== 4.2 move_to_coordinates 正常点 (0.25,-0.05,0.15) =====")
    target = np.array([0.25, -0.05, 0.15])
    ee_before = backend.get_end_effector_pose().position.copy()
    r = C.post(f"{base}/api/v1/move_to_coordinates",
               json={"x": "0.25", "y": "-0.05", "z": "0.15"})
    body = r.json()
    _check("4.2 HTTP 200 + code=0", r.status_code == 200 and body["code"] == 0
           and body["error"] is None, f"HTTP {r.status_code}")
    result = body["result"]
    _check("4.2 返回 成功到达0.250,-0.050,0.150",
           result == "成功到达0.250,-0.050,0.150", result)
    ee_after = backend.get_end_effector_pose().position
    _check("4.2 到达后真实 EE 在 tolerance 内 (2cm)",
           pose_distance(ee_after, target) < control.position_threshold_m,
           f"err={pose_distance(ee_after, target):.4f}")
    _check("4.2 真实运动(末端位置发生改变, 非 teleport 瞬移)",
           float(np.linalg.norm(ee_after - ee_before)) > 0.01,
           f"move={np.linalg.norm(ee_after - ee_before):.4f}")

    # =====================================================================
    # 4.3 move_to_coordinates 不可达点
    # =====================================================================
    print("\n===== 4.3 move_to_coordinates 不可达点 (0.20,-0.05,0.15) =====")
    t_unc = np.array([0.20, -0.05, 0.15])
    r = C.post(f"{base}/api/v1/move_to_coordinates",
               json={"x": "0.20", "y": "-0.05", "z": "0.15"})
    body = r.json()
    _check("4.3 HTTP 200 + code=0 (网关调用本身成功)",
           r.status_code == 200 and body["code"] == 0 and body["error"] is None,
           f"HTTP {r.status_code}")
    result = body["result"]
    _check("4.3 返回 未到达", result.startswith("未到达"), result)
    reported = _err_val(result)
    real = float(np.linalg.norm(backend.get_end_effector_pose().position - t_unc))
    _check("4.3 误差 = 真实 ‖ee - target‖₂",
           abs(reported - real) < 0.002,
           f"reported={reported:.4f} real={real:.4f}")
    _check("4.3 不是修复前的 0.0010",
           not (0.0008 <= reported <= 0.0012), f"reported={reported:.4f}")

    # =====================================================================
    # 4.4 参数错误 -> HTTP 400, code=400
    # =====================================================================
    print("\n===== 4.4 参数错误 =====")
    bad_cases = [
        ("缺 x", {"y": "-0.05", "z": "0.15"}),
        ("缺 y", {"x": "0.25", "z": "0.15"}),
        ("缺 z", {"x": "0.25", "y": "-0.05"}),
        ("x=abc", {"x": "abc", "y": "-0.05", "z": "0.15"}),
        ("y=abc", {"x": "0.25", "y": "abc", "z": "0.15"}),
        ("z=abc", {"x": "0.25", "y": "-0.05", "z": "abc"}),
        ("extra field", {"x": "0.25", "y": "-0.05", "z": "0.15", "extra": 1}),
        ("坏 JSON", "not-json"),
    ]
    for name, payload in bad_cases:
        data = payload if isinstance(payload, str) else json.dumps(payload)
        r = C.post(f"{base}/api/v1/move_to_coordinates", content=data,
                   headers={"Content-Type": "application/json"})
        body = r.json()
        ok = (r.status_code == 400 and body.get("code") == 400
              and body.get("result") is None and isinstance(body.get("error"), str))
        _check(f"4.4 {name} -> HTTP 400 code=400", ok,
               f"HTTP {r.status_code} body={json.dumps(body, ensure_ascii=False)}")

    # =====================================================================
    # 4.5 grab_the_block
    # =====================================================================
    print("\n===== 4.5 grab_the_block =====")
    # 4.5a 合法颜色 + 场景中有 -> 成功
    for color in ("red", "yellow", "white"):
        backend.reset("all")
        r = C.post(f"{base}/api/v1/grab_the_block", json={"color": color})
        body = r.json()
        _check(f"4.5 grab {color} 成功",
               r.status_code == 200 and body["code"] == 0
               and body["result"] == "有这种颜色的物块，且夹取物块成功",
               body["result"])
        backend.reset("all")  # 释放后复位, 避免上一轮夹持影响下一轮

    # 4.5b 非法颜色 blue -> 没有这种颜色的物块，无法夹取, 且不触发运动
    backend.reset("all")
    ee_before = backend.get_end_effector_pose().position.copy()
    r = C.post(f"{base}/api/v1/grab_the_block", json={"color": "blue"})
    body = r.json()
    _check("4.5 grab blue -> 没有这种颜色的物块，无法夹取",
           r.status_code == 200 and body["result"] == "没有这种颜色的物块，无法夹取",
           body["result"])
    _check("4.5 非法颜色不启动机械臂运动(EE 未动)",
           float(np.linalg.norm(backend.get_end_effector_pose().position - ee_before)) < 0.005,
           "EE 保持原位")

    # 4.5c 场景缺色(red_only 场景抓 yellow) -> 没有这种颜色的物块
    backend.reset("red_only")
    r = C.post(f"{base}/api/v1/grab_the_block", json={"color": "yellow"})
    body = r.json()
    _check("4.5 red_only 场景抓 yellow -> 没有这种颜色的物块，无法夹取",
           r.status_code == 200 and body["result"] == "没有这种颜色的物块，无法夹取",
           body["result"])

    # =====================================================================
    # 4.6 physical grasp: 真实接触咬合抬升, 不 teleport
    # =====================================================================
    print("\n===== 4.6 physical grasp (真实接触/抬升) =====")
    backend.reset("all")
    cube_z0 = backend.cube_position("red")[2]
    r = C.post(f"{base}/api/v1/grab_the_block", json={"color": "red"})
    body = r.json()
    _check("4.6 grab red 成功", body["result"] == "有这种颜色的物块，且夹取物块成功",
           body["result"])
    cube_z1 = backend.cube_position("red")[2]
    _check("4.6 物块真实被抬起(接触摩擦咬合, 非改 xyz)",
           cube_z1 > cube_z0 + 0.04,
           f"cube z {cube_z0:.3f} -> {cube_z1:.3f}")
    _check("4.6 is_holding_object() 物理判定为真",
           backend.is_holding_object(), "物理接触判定")

    # =====================================================================
    # 4.7 release_the_block
    # =====================================================================
    print("\n===== 4.7 release_the_block =====")
    # A. 没抓任何东西
    backend.reset("all")
    r = C.post(f"{base}/api/v1/release_the_block")
    body = r.json()
    _check("4.7A 没抓物块 -> 本身就没加起来物块",
           r.status_code == 200 and body["result"] == "本身就没加起来物块",
           body["result"])
    # B. 真实抓起后释放 -> 重力下落
    r = C.post(f"{base}/api/v1/grab_the_block", json={"color": "red"})
    assert r.json()["result"] == "有这种颜色的物块，且夹取物块成功"
    cube_z_held = backend.cube_position("red")[2]
    r = C.post(f"{base}/api/v1/release_the_block")
    body = r.json()
    _check("4.7B 夹着时释放 -> 成功释放物块",
           r.status_code == 200 and body["result"] == "成功释放物块",
           body["result"])
    time.sleep(0.6)  # 让物块在重力下落到桌面
    cube_z_after = backend.cube_position("red")[2]
    table_top = 0.0
    _check("4.7B 物块重力落回桌面(接触丢失 -> 自由落体)",
           abs(cube_z_after - (table_top + 0.015)) < 0.01,
           f"cube z {cube_z_held:.3f} -> {cube_z_after:.3f}")

    C.close()
    print("\n" + "=" * 64)
    passed = sum(1 for _, ok, _ in _RESULTS if ok)
    total = len(_RESULTS)
    print(f"HTTP FINAL REGRESSION: {passed}/{total} PASSED")
    fails = [n for n, ok, _ in _RESULTS if not ok]
    if fails:
        print(f"FAILED: {fails}")
        sys.exit(1)
    print("ALL HTTP FINAL REGRESSION PASSED")


if __name__ == "__main__":
    main()
