---
theme: seriph
background: ./cover.jpg
title: "全双工语音 Agent 系统设计"
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
  font-size: 3.6rem !important;
  font-weight: 500 !important;
  margin-bottom: 0.5rem !important;
  color: white !important;
}
.slidev-layout h2 {
  font-size: 1.4rem !important;
  font-weight: 400 !important;
  margin-bottom: 2rem !important;
  color: rgba(255,255,255,0.9) !important;
}
</style>

# 全双工语音 Agent 系统

## 从串行 Pipeline 到异步双 Agent 的设计演进

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
  <img src="./pics/3.png" alt="异步 Agent 系统工作原理" style="max-width: 90%; max-height: 300px; object-fit: contain; border: 1px solid #e5e7eb; border-radius: 10px;" />
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
<div><span class="font-semibold text-slate-700">排序约定：</span>tool response、user input、状态栏三者同时存在时，固定顺序为 <span class="text-blue-700 font-medium">tool response → user input → 状态栏</span>；tool/user 双角色回写工具结果，多轮之后队列状态与上下文始终自洽。</div>
</div>

---
transition: fade
class: text-center
title: "谢谢"
---

# 谢谢聆听

<div class="mt-6 text-slate-500" style="font-size: 1rem;">
全双工语音 Agent 系统 · 设计演进
</div>
