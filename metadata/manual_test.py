"""秘塔搜索的交互式人工测试程序。"""

from __future__ import annotations

import os
from pprint import pprint

from matadata import Metadata
from metaso_search import DeepSeekAPIError, MetasoAPIError, QwenImageAPIError, search_metaso


def run(question: str) -> list[Metadata]:
    """调用搜索并验证返回值确实为 list[Metadata]。"""

    results = search_metaso(question)
    if not isinstance(results, list):
        raise TypeError(f"返回值应为 list，实际为 {type(results).__name__}")
    if not all(isinstance(item, Metadata) for item in results):
        raise TypeError("列表中存在不是 Metadata 类型的结果")
    return results


def main() -> int:
    print("=== 秘塔 API → Metadata 测试 ===")

    if not os.getenv("METASO_API_KEY"):
        print("\n未检测到 METASO_API_KEY。")
        print('$env:METASO_API_KEY = "你的秘塔 API Key"')
        print("设置后，请在同一个 PowerShell 窗口重新运行本程序。")
        return 1

    if not os.getenv("DEEPSEEK_API_KEY"):
        print("\n未检测到 DEEPSEEK_API_KEY。")
        print('$env:DEEPSEEK_API_KEY = "你的 DeepSeek API Key"')
        print("设置后，请在同一个 PowerShell 窗口重新运行本程序。")
        return 1

    if not os.getenv("DASHSCOPE_API_KEY"):
        print("\n未检测到 DASHSCOPE_API_KEY。")
        print('$env:DASHSCOPE_API_KEY = "你的阿里云百炼 API Key"')
        print("该 Key 用于让 qwen3.7-plus 筛选图片并生成 image_summary。")
        return 1

    question = input("\n请输入问题：").strip()
    if not question:
        print("错误：问题不能为空。")
        return 1

    print("\n正在调用秘塔 API，请稍候……")
    try:
        results = run(question)
    except (ValueError, TypeError, MetasoAPIError, DeepSeekAPIError, QwenImageAPIError) as exc:
        print(f"\n测试失败：{exc}")
        return 1

    print("\n测试通过")
    print("返回类型：list[Metadata]")
    print(f"结果数量：{len(results)}")
    print("\n返回内容：")
    pprint(results, sort_dicts=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
