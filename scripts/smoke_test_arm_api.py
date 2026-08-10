"""机械臂 Arm API 冒烟测试: 对**已运行的指定服务器**做 HTTP 冒烟。

用法:
    python scripts/smoke_test_arm_api.py                          # 默认 http://127.0.0.1:8000
    python scripts/smoke_test_arm_api.py --base-url http://127.0.0.1:8000
    python scripts/smoke_test_arm_api.py --base-url http://<SERVER_IP>:8000

检查项(至少):
    1. get_current_coordinates        -> 200, 我的坐标是...
    2. move_to_coordinates(0.25,-0.05,0.15) -> 成功到达
    3. grab_the_block(red)            -> 有这种颜色的物块，且夹取物块成功
    4. release_the_block              -> 成功释放物块
    5. grab blue                      -> 没有这种颜色的物块，无法夹取
    6. invalid request                -> HTTP 400

本脚本**不启动、不 mock backend** —— 测试的就是 --base-url 指向的当前服务器。
"""
from __future__ import annotations

import argparse
import json
import sys

import httpx

SMOKE_OK = "ALL ARM API SMOKE TESTS PASSED"
SMOKE_FAIL = "ARM API SMOKE TESTS FAILED"


def main() -> None:
    ap = argparse.ArgumentParser(description="机械臂 Arm API 冒烟测试(测已运行的服务器)")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000",
                    help="目标服务器地址, 默认 http://127.0.0.1:8000")
    ap.add_argument("--timeout", type=float, default=60.0,
                    help="单请求超时(秒)。同步阻塞 API, 移动/抓取需足够大, 默认 60s")
    args = ap.parse_args()

    base = args.base_url.rstrip("/")
    print(f"smoke target: {base}")
    results: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        results.append((name, ok, detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))

    try:
        c = httpx.Client(trust_env=False, timeout=args.timeout)
    except Exception as e:
        print(f"无法创建 HTTP 客户端: {e}")
        sys.exit(1)

    def show(name: str, resp) -> None:
        try:
            body = resp.json()
        except Exception:
            body = resp.text
        ok = resp.status_code == 200 and isinstance(body, dict) \
            and body.get("code") == 0 and body.get("error") is None
        check(name, ok, f"HTTP {resp.status_code} result={body.get('result')!r}")

    # ---- 1. get_current_coordinates ------------------------------------
    print("[1/6] get_current_coordinates")
    r = c.get(f"{base}/api/v1/get_current_coordinates")
    body = r.json()
    ok = (r.status_code == 200 and body.get("code") == 0
          and isinstance(body.get("result"), str)
          and body["result"].startswith("我的坐标是"))
    check("get_current_coordinates", ok, f"HTTP {r.status_code} {body.get('result')!r}")

    # ---- 2. move_to_coordinates 正常点 ----------------------------------
    print("[2/6] move_to_coordinates(0.25,-0.05,0.15)")
    r = c.post(f"{base}/api/v1/move_to_coordinates",
               json={"x": "0.25", "y": "-0.05", "z": "0.15"})
    body = r.json()
    ok = (r.status_code == 200 and body.get("code") == 0
          and body.get("result") == "成功到达0.250,-0.050,0.150")
    check("move_to_coordinates 正常点", ok,
          f"HTTP {r.status_code} {body.get('result')!r}")

    # ---- 3. grab_the_block(red) -----------------------------------------
    print("[3/6] grab_the_block(red)")
    r = c.post(f"{base}/api/v1/grab_the_block", json={"color": "red"})
    body = r.json()
    ok = (r.status_code == 200 and body.get("code") == 0
          and body.get("result") == "有这种颜色的物块，且夹取物块成功")
    check("grab_the_block(red)", ok,
          f"HTTP {r.status_code} {body.get('result')!r}")

    # ---- 4. release_the_block ------------------------------------------
    print("[4/6] release_the_block")
    r = c.post(f"{base}/api/v1/release_the_block")
    body = r.json()
    ok = (r.status_code == 200 and body.get("code") == 0
          and body.get("result") == "成功释放物块")
    check("release_the_block", ok,
          f"HTTP {r.status_code} {body.get('result')!r}")

    # ---- 5. grab blue (非法颜色) ----------------------------------------
    print("[5/6] grab_the_block(blue)")
    r = c.post(f"{base}/api/v1/grab_the_block", json={"color": "blue"})
    body = r.json()
    ok = (r.status_code == 200 and body.get("code") == 0
          and body.get("result") == "没有这种颜色的物块，无法夹取")
    check("grab_the_block(blue)", ok,
          f"HTTP {r.status_code} {body.get('result')!r}")

    # ---- 6. invalid request (缺 z) ---------------------------------------
    print("[6/6] invalid request (缺 z)")
    r = c.post(f"{base}/api/v1/move_to_coordinates",
               json={"x": "0.25", "y": "-0.05"})
    body = r.json()
    ok = (r.status_code == 400 and body.get("code") == 400
          and body.get("result") is None and isinstance(body.get("error"), str))
    check("invalid request -> HTTP 400", ok,
          f"HTTP {r.status_code} error={body.get('error')!r}")

    c.close()

    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    print(f"\n{passed}/{total} checks passed")
    if passed == total:
        print(SMOKE_OK)
        sys.exit(0)
    print(SMOKE_FAIL)
    sys.exit(1)


if __name__ == "__main__":
    main()
