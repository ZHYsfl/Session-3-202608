---
theme: seriph
background: ./cover.jpg
title: "全双工语音 × 具身执行双 Agent 系统"
class: text-center
transition: slide-left
mdc: true
---

<style scoped>
.slidev-layout {
  background-image: url('./cover.jpg') !important;
  background-size: cover !important;
  background-position: center !important;
  background-repeat: no-repeat !important;
  width: 100% !important;
  height: 100% !important;
  display: flex !important;
  flex-direction: column !important;
  justify-content: center !important;
  align-items: center !important;
  color: white !important;
  text-shadow: 0 2px 10px rgba(0,0,0,0.5) !important;
}
.slidev-layout h1 {
  font-size: 2.9rem !important;
  font-weight: 500 !important;
  margin-bottom: 0.5rem !important;
  color: white !important;
  white-space: nowrap !important;
}
.slidev-layout h2 {
  font-size: 1.4rem !important;
  font-weight: 400 !important;
  margin-bottom: 2rem !important;
  color: rgba(255,255,255,0.9) !important;
}
</style>

# 全双工语音 × 具身执行双 Agent 系统

## Voice Agent + Arm Agent：从串行 Pipeline 到异步双 Agent 的设计演进

---
transition: slide-up
title: "传统串行语音 Pipeline"
---

<div class="text-center" style="margin-top: 0.75rem;">
<div style="font-size: 2rem; font-weight: 400; color: #5a7a8a;">传统方案：串行语音 Pipeline</div>
<div style="font-size: 0.9rem; color: #6b7280; margin-top: 0.3rem;">VAD → STT → LLM → TTS，严格串行执行</div>
</div>

<div class="w-full flex justify-center" style="margin-top: 3rem;">
  <img src="./pics/0.png" alt="传统串行 VAD-STT-LLM-TTS 流程" style="max-width: 88%; width: 88%; border: 1px solid #e5e7eb; border-radius: 10px;" />
</div>

<div class="grid grid-cols-3 gap-5 px-10" style="margin-top: 3rem;">

<div class="text-center">
<div class="font-medium" style="color: #374151; font-size: 0.95rem;">延迟叠加</div>
<div class="text-sm text-gray-500 mt-1">一轮交互延迟 ≈ 四级延迟之和</div>
</div>

<div class="text-center">
<div class="font-medium" style="color: #374151; font-size: 0.95rem;">无法插话</div>
<div class="text-sm text-gray-500 mt-1">系统说话时，用户只能等待</div>
</div>

<div class="text-center">
<div class="font-medium" style="color: #374151; font-size: 0.95rem;">资源闲置</div>
<div class="text-sm text-gray-500 mt-1">任意时刻只有一个组件在工作</div>
</div>

</div>

---
transition: slide-up
title: "改进一：Buffer"
---

<div class="text-center" style="margin-top: 0.75rem;">
<div style="font-size: 2rem; font-weight: 400; color: #5a7a8a;">改进 ① —— Buffer</div>
<div style="font-size: 0.9rem; color: #6b7280; margin-top: 0.3rem;">环形 buffer：动态记录截至当前、往前 1s 的音频片段</div>
</div>

<div class="w-full flex justify-center" style="margin-top: 2.5rem;">
  <img src="./pics/1.png" alt="Buffer 改进示意" style="max-width: 88%; width: 88%; border: 1px solid #e5e7eb; border-radius: 10px;" />
</div>

<div class="grid grid-cols-2 gap-6 px-16" style="margin-top: 2.5rem;">

<div class="text-center">
<div class="font-medium" style="color: #374151; font-size: 0.95rem;">防止吞字</div>
<div class="text-sm text-gray-500 mt-1">VAD 检测到说话时句首已被吃掉；buffer 让 STT 从开口前 1s 开始识别，不丢字</div>
</div>

<div class="text-center">
<div class="font-medium" style="color: #374151; font-size: 0.95rem;">环形结构</div>
<div class="text-sm text-gray-500 mt-1">只保留最近 1s 的滚动窗口，持续覆写，内存恒定</div>
</div>

</div>

---
transition: slide-up
title: "改进二：打断机制（说话阶段）"
---

<div class="text-center" style="margin-top: 0.75rem;">
<div style="font-size: 2rem; font-weight: 400; color: #5a7a8a;">改进 ② —— 打断机制（一）：说话阶段被打断</div>
<div style="font-size: 0.9rem; color: #6b7280; margin-top: 0.3rem;">说到一半可以随时打断，上下文自动重组</div>
</div>

<div class="w-full flex justify-center" style="margin-top: 1.5rem;">
  <img src="./pics/2-1.png" alt="说话阶段被打断" style="max-width: 90%; max-height: 320px; object-fit: contain; border: 1px solid #e5e7eb; border-radius: 10px;" />
</div>

<div class="max-w-5xl mx-auto px-10 text-left" style="margin-top: 1.5rem; font-size: 0.85rem; line-height: 1.6; color: #4b5563;">
<div><span class="font-semibold text-slate-700">打断回写：</span>被打断的 assistant 消息以 <code>&lt;/interrupted&gt;</code> 标记截断点、原样保留在上下文中，用户新输入自然接续其后，上下文保持连贯。</div>
</div>

---
transition: slide-up
title: "改进二：打断机制（tool_call 阶段）"
---

<div class="text-center" style="margin-top: 0.75rem;">
<div style="font-size: 2rem; font-weight: 400; color: #5a7a8a;">改进 ② —— 打断机制（二）：tool_call 阶段被打断</div>
<div style="font-size: 0.9rem; color: #6b7280; margin-top: 0.3rem;">工具调用生成期间用户插话，两条线索并行推进</div>
</div>

<div class="w-full flex justify-center" style="margin-top: 1.2rem;">
  <img src="./pics/2-2.png" alt="tool_call 阶段被打断" style="max-width: 92%; max-height: 330px; object-fit: contain; border: 1px solid #e5e7eb; border-radius: 10px;" />
</div>

<div class="max-w-5xl mx-auto px-10 text-left" style="margin-top: 1.2rem; font-size: 0.85rem; line-height: 1.6; color: #4b5563;">
<div><span class="font-semibold text-slate-700">核心观察：</span>tool_call 生成阶段对用户天然静默，用户打断可与当前工具调用<span class="text-blue-700 font-medium">并行处理</span>，互不阻塞。</div>
</div>

---
transition: slide-up
title: "改进二：打断机制（边听边想）"
---

<div class="text-center" style="margin-top: 0.75rem;">
<div style="font-size: 2rem; font-weight: 400; color: #5a7a8a;">改进 ② —— 打断机制（三）：边听边想</div>
<div style="font-size: 0.9rem; color: #6b7280; margin-top: 0.3rem;">goroutine 并行处理，双双就绪后再做下一步推理</div>
</div>

<div class="w-full flex justify-center" style="margin-top: 1.2rem;">
  <img src="./pics/2-3.png" alt="边听边想：并行等待双双就绪" style="max-width: 94%; max-height: 320px; object-fit: contain; border: 1px solid #e5e7eb; border-radius: 10px;" />
</div>

<div class="max-w-5xl mx-auto px-10 text-left" style="margin-top: 1.2rem; font-size: 0.85rem; line-height: 1.6; color: #4b5563;">
<div><span class="font-semibold text-slate-700">双双就绪再推理：</span>等待 tool_call 生成完毕并得到 tool response、且用户说完话 STT 转录完毕后，再让 LLM 做下一步推理——一次推理同时看到工具结果与新输入。</div>
</div>

---
transition: slide-up
title: "改进三：异步 Agent 系统"
---

<div class="text-center" style="margin-top: 0.75rem;">
<div style="font-size: 2rem; font-weight: 400; color: #5a7a8a;">改进 ③ —— 异步双 Agent 系统</div>
<div style="font-size: 0.9rem; color: #6b7280; margin-top: 0.3rem;">状态栏的引入：本质是一个生产者–消费者问题</div>
</div>

<div class="w-full flex justify-center" style="margin-top: 0.8rem;">
  <img src="./pics/3.png" alt="异步 Agent 系统工作原理" style="max-width: 92%; max-height: 330px; object-fit: contain; border: 1px solid #e5e7eb; border-radius: 10px;" />
</div>

<div class="max-w-5xl mx-auto px-10 text-left" style="margin-top: 1rem; font-size: 0.85rem; line-height: 1.6; color: #4b5563;">
<div><span class="font-semibold text-slate-700">双向消息队列解耦：</span><code>send_to_xxx_agent</code> 生产、<code>get_message_from_xxx_agent</code> 消费，两个 Agent 异步协作、互不阻塞。</div>
<div class="mt-1"><span class="font-semibold text-slate-700">人优先：</span>后台 Agent 空闲时自动消费队列；忙碌时通过 <code>&lt;queue_status&gt;</code> 状态栏感知未读消息，主动消费，绝不插队打断当前推理。</div>
</div>

---
transition: slide-up
title: "上下文工程（一）：状态栏统一"
---

<div class="text-center" style="margin-top: 0.75rem;">
<div style="font-size: 2rem; font-weight: 400; color: #5a7a8a;">上下文工程 ① —— 状态栏统一为独立 user 消息</div>
<div style="font-size: 0.9rem; color: #6b7280; margin-top: 0.3rem;">异步 Agent 加入后，上下文如何保持连贯、状态可感知</div>
</div>

<div class="w-full flex justify-center" style="margin-top: 1.5rem;">
  <img src="./pics/4.png" alt="状态栏统一与打断重组" style="max-width: 92%; max-height: 300px; object-fit: contain; border: 1px solid #e5e7eb; border-radius: 10px;" />
</div>

<div class="grid grid-cols-2 gap-6 px-14" style="margin-top: 1.8rem;">

<div class="text-center">
<div class="font-medium" style="color: #374151; font-size: 0.95rem;"><code>&lt;queue_status&gt;</code> 统一改动</div>
<div class="text-sm text-gray-500 mt-1">状态栏统一以独立的 user 消息注入，不再拼接进用户文本</div>
</div>

<div class="text-center">
<div class="font-medium" style="color: #374151; font-size: 0.95rem;">延迟调用能力的考验</div>
<div class="text-sm text-gray-500 mt-1">打断引入后，工具调用可延迟到双双就绪后再触发推理</div>
</div>

</div>

---
transition: slide-up
title: "上下文工程（二）：固定排序"
---

<div class="text-center" style="margin-top: 0.75rem;">
<div style="font-size: 2rem; font-weight: 400; color: #5a7a8a;">上下文工程 ② —— 多轮全链路与固定排序</div>
<div style="font-size: 0.9rem; color: #6b7280; margin-top: 0.3rem;">tool response → user input → 状态栏，上下文始终自洽</div>
</div>

<div class="w-full flex justify-center" style="margin-top: 1.2rem;">
  <img src="./pics/5.png" alt="多轮交互与固定排序" style="max-width: 94%; max-height: 310px; object-fit: contain; border: 1px solid #e5e7eb; border-radius: 10px;" />
</div>

<div class="max-w-5xl mx-auto px-10 text-left" style="margin-top: 1.5rem; font-size: 0.85rem; line-height: 1.6; color: #4b5563;">
<div><span class="font-semibold text-slate-700">排序约定：</span>tool response、user input、状态栏三者同时存在时，固定顺序为 <span class="text-blue-700 font-medium">tool response → user input → 状态栏</span>；工具结果以单条 role=tool 消息写回（chat template 渲染进 user 的 &lt;tool_response&gt; 块），多轮之后队列状态与上下文始终自洽。</div>
</div>

---
transition: slide-up
title: "训练数据集全景"
---

<div class="text-center" style="margin-top: 0.75rem;">
<div style="font-size: 2rem; font-weight: 400; color: #5a7a8a;">训练数据集全景结构</div>
<div style="font-size: 0.9rem; color: #6b7280; margin-top: 0.3rem;">异步双 Agent 训练数据集 · 2,900 条</div>
</div>

<div class="w-full flex justify-center" style="margin-top: 0.5rem;">
  <img src="./pics/数据架构概览图.png" alt="数据集全景结构" style="max-width: 92%; max-height: 375px; object-fit: contain; border: 1px solid #e5e7eb; border-radius: 10px;" />
</div>

---
transition: slide-up
title: "Voice Agent 训练曲线"
---

<div class="text-center" style="margin-top: 0.75rem;">
<div style="font-size: 2rem; font-weight: 400; color: #5a7a8a;">Voice Agent · SFT 训练曲线</div>
<div style="font-size: 0.9rem; color: #6b7280; margin-top: 0.3rem;">Qwen3-4B + QLoRA · 训练 loss 收敛至 ≈ 0.057，held-out 验证 loss 0.243 → 0.156</div>
</div>

<div class="w-full flex justify-center" style="margin-top: 0.8rem;">
  <img src="./pics/voice_agent训练曲线图.png" alt="Voice Agent SFT 训练曲线" style="max-width: 90%; max-height: 385px; object-fit: contain; border: 1px solid #e5e7eb; border-radius: 10px;" />
</div>

---
layout: center
class: text-center
transition: fade
title: "训练中"
---

<div style="font-size: 2.6rem; font-weight: 400; color: #5a7a8a;">Still training now…</div>

<div class="mt-4 text-slate-500" style="font-size: 1rem;">Voice Agent ready · Arm Agent SFT still in progress</div>

---
transition: slide-up
title: "具身工具 API 网关"
---

<div class="text-center" style="margin-top: 0.75rem;">
<div style="font-size: 2rem; font-weight: 400; color: #5a7a8a;">具身工具 REST 网关（:8000）</div>
<div style="font-size: 0.9rem; color: #6b7280; margin-top: 0.3rem;">6 个协议工具对外暴露为 RESTful API：同步阻塞调用，联调无需轮询</div>
</div>

<style scoped>
table { font-size: 0.78rem; }
table td, table th { padding: 0.3rem 0.6rem; }
</style>

<div class="max-w-5xl mx-auto px-10" style="margin-top: 1.2rem;">

| 接口 | 方法 | 说明 |
| --- | --- | --- |
| `get_current_coordinates` | GET | 查当前坐标（低速 / 高速两种返回话术） |
| `move_to_coordinates` | POST | 移动到 x,y,z，内部误差校验，未达标如实报误差 |
| `grab_the_block` | POST | 视觉观察 → 夹取，camera2 第三视角判定成败 |
| `release_the_block` | POST | 交叉验证是否夹持，释放失败则请人介入 |
| `send_to_voice_agent` | POST | 生产消息进 arm→voice 队列（进度 / 完成 / 求助） |
| `get_message_from_voice_agent` | POST | 排空 voice→arm 队列，一次性取回全部新指令 |

</div>

<div class="max-w-5xl mx-auto px-10 text-left" style="margin-top: 1rem; font-size: 0.82rem; line-height: 1.6; color: #4b5563;">
<div><span class="font-semibold text-slate-700">统一约定：</span>所有接口返回 <code>{code, result, error}</code> 三段式，工具原始字符串经 <code>result</code> 逐字透传；HTTP 200 / 400 / 500 对应网关成功 / 参数错误 / 内部异常。本网关与语音网关（:8001）对接<span class="text-blue-700 font-medium">同一个队列后端</span>，两条 FIFO 队列即双 Agent 的生产–消费通道。</div>
</div>

---
transition: slide-up
title: "演示：具身工具 API 联调"
---

<div class="text-center" style="margin-top: 0.75rem;">
<div style="font-size: 2rem; font-weight: 400; color: #5a7a8a;">演示 —— 具身工具 API 网关联调</div>
<div style="font-size: 0.9rem; color: #6b7280; margin-top: 0.3rem;">REST 网关搭建完成后的端到端真机演示</div>
</div>

<div class="w-full flex justify-center" style="margin-top: 2rem;">
  <video src="./embodied_tools.mp4" controls muted style="max-width: 76%; border: 1px solid #e5e7eb; border-radius: 10px; box-shadow: 0 4px 16px rgba(0,0,0,0.08);" />
</div>

---
transition: slide-up
title: "团队协作"
---

<div class="text-center" style="margin-top: 0.75rem;">
<div style="font-size: 2rem; font-weight: 400; color: #5a7a8a;">团队协作 —— GitHub 协同开发</div>
<div style="font-size: 0.9rem; color: #6b7280; margin-top: 0.3rem;">14 位贡献者 · 分支并行开发 · PR 合入主干</div>
</div>

<div class="w-full flex justify-center" style="margin-top: 1.2rem;">
  <img src="./pics/github_cowork.png" alt="GitHub 团队协作截图" style="max-width: 88%; max-height: 380px; object-fit: contain; border: 1px solid #e5e7eb; border-radius: 10px; box-shadow: 0 4px 16px rgba(0,0,0,0.08);" />
</div>

---
layout: center
class: text-center
transition: fade
title: "系统即将就绪"
---

<div style="font-size: 2.6rem; font-weight: 400; color: #5a7a8a;">The full system goes live tonight…</div>

---
layout: center
transition: fade
class: text-center
title: "谢谢"
---

# 谢谢聆听

<div class="mt-6 text-slate-500" style="font-size: 1rem;">
全双工语音 × 具身执行双 Agent 系统 · 设计演进
</div>
