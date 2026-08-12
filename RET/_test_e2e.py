"""
端到端测试脚本
测试所有检索路径，验证入库后的系统可用性
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from agent import run_agent, AGENT_TOOLS


def test_all():
    print("=" * 60)
    print("🔬 端到端测试 — 生物医学论文检索系统")
    print("=" * 60)

    tests = [
        # (tool, query, kwargs, label)
        ("search_papers", "他汀类药物对低密度脂蛋白胆固醇的影响", {"top_k": 3, "verbose": True}, "中文混合检索"),
        ("search_papers", "coronary heart disease LDL-C management", {"top_k": 3}, "英文混合检索-跨语言"),
        ("semantic_search", "心血管疾病预防与血脂管理", {"top_k": 3}, "纯语义检索"),
        ("lexical_search", "LDL-C 130 mg/dL", {"top_k": 3}, "纯词汇检索-精确术语"),
        ("search_papers", "肺癌患者的血清胆固醇水平", {"top_k": 3, "keywords": "肺癌 胆固醇"}, "带关键词的混合检索"),
    ]

    passed = 0
    failed = 0

    for tool, query, kwargs, label in tests:
        print(f"\n{'─'*60}")
        print(f"🧪 [{label}] tool={tool} query='{query}'")
        print(f"{'─'*60}")
        try:
            t0 = time.time()
            result = run_agent(query, tool=tool, **kwargs)
            elapsed = time.time() - t0

            # 简单验证
            if "❌" in result:
                print(f"⚠️  返回错误信息")
                failed += 1
            elif result.strip():
                lines = result.split("\n")
                print(f"✅ 成功 ({len(lines)} 行, {elapsed:.1f}s)")
                print(result[:1500])
                passed += 1
            else:
                print(f"⚠️  空结果")
                failed += 1
        except Exception as e:
            print(f"❌ 异常: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"📊 测试结果: {passed} 通过 / {failed} 失败 / {passed+failed} 总计")
    print(f"{'='*60}")


if __name__ == "__main__":
    test_all()
