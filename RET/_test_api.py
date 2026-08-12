"""测试 SiliconFlow API 限流阈值"""
from openai import OpenAI
import time

c = OpenAI(
    api_key='sk-esyilubtqzobjaylaraitkyaclmbjgzkijhdkwyzsfzwmtpp',
    base_url='https://api.siliconflow.cn/v1',
)

MODEL = 'BAAI/bge-m3'

# 单条测试
print("1. 单条 embedding...", end=" ")
t = time.time()
r = c.embeddings.create(model=MODEL, input=["测试"])
print(f"OK dim={len(r.data[0].embedding)}, {time.time()-t:.1f}s")

# 批次 8 条
print("2. 批次 8 条...", end=" ")
t = time.time()
r = c.embeddings.create(model=MODEL, input=[f"文本{i}" for i in range(8)])
print(f"OK, {time.time()-t:.1f}s")

# 批次 16 条
print("3. 批次 16 条...", end=" ")
t = time.time()
r = c.embeddings.create(model=MODEL, input=[f"这是一个稍微长一点的测试文本内容编号{i}" for i in range(16)])
print(f"OK, {time.time()-t:.1f}s")

# 5 次连续请求 (每次间隔 0.3s)
print("4. 连续 5 次请求 (interval=0.3s)...")
for i in range(5):
    t = time.time()
    r = c.embeddings.create(model=MODEL, input=[f"连续请求测试{i}", f"内容{i}"])
    print(f"   #{i}: OK, {time.time()-t:.1f}s")
    time.sleep(0.3)

# 2 次无间隔请求
print("5. 2 次无间隔请求...")
t = time.time()
r = c.embeddings.create(model=MODEL, input=["无间隔A", "无间隔B"])
print(f"   #1: OK, {time.time()-t:.1f}s")
t = time.time()
r = c.embeddings.create(model=MODEL, input=["无间隔C", "无间隔D"])
print(f"   #2: OK, {time.time()-t:.1f}s")

print("\n✅ 所有测试通过！")
