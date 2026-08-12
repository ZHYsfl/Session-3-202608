# 秘塔搜索结果转换为 Metadata

设置秘塔、DeepSeek 和阿里云百炼 API Key：

```powershell
$env:METASO_API_KEY = "你的秘塔 API Key"
$env:DEEPSEEK_API_KEY = "你的 DeepSeek API Key"
$env:DASHSCOPE_API_KEY = "你的阿里云百炼 API Key"
```

默认并发配置为 DeepSeek 15、Qwen 8。可以按账号实际限流调整：

```powershell
$env:DEEPSEEK_MAX_CONCURRENCY = "15"
$env:QWEN_MAX_CONCURRENCY = "8"
```

最终结果确定后，20 条 DeepSeek 文本摘要不再逐条串行，而是由 15 个异步队列 worker 消费：任意一条完成后立即出队下一条。DeepSeek 摘要、网页/PDF 图片抓取和 Qwen 图片处理构成流水线；某条来源的图片抓取完成后会立即进入 8 个 Qwen worker 的队列，不等待其余来源。遇到 429、超时或临时服务错误仍会自动退避重试。

这两个并发值不是模型本身的统一上限：实际容量取决于所用后端、账号等级和 RPM/TPM 配额。若出现 HTTP 429，先降低对应并发；若连续多次正式测试均无 429 且服务延迟稳定，可以每次增加 2，逐步寻找当前账号的最佳值。

在 Python 中调用：

```python
from metaso_search import search_metaso

results = search_metaso("人工智能在医学影像中的最新应用")
# results 的类型为 list[Metadata]
```

也可以直接运行：

```powershell
python .\metaso_search.py "人工智能在医学影像中的最新应用"
```

正式命令将 `list[Metadata]` 以 JSON 格式直接输出到终端。默认不请求来源全文，
`text` 通常来自秘塔返回的搜索片段。每条结果的 `last_retrieved_at` 为 Python `None`，
序列化到 JSON 后显示为 `null`。

程序会在最终结果确定后，将每一条结果分别发送给 DeepSeek 后端，使用代码中配置的
`deepseek-v4-pro` 独立生成一句中文 `text_summary`。因此最终纳入 20 条时会发生 20 次
Pro 总结调用。总结严格按结果顺序逐条执行，每次只发送该条 Metadata 的 `text`；代码会
显式关闭 V4 Pro 思考模式，避免思考 token 占满输出预算后 `content` 为空。空响应、临时
连接错误和无效响应会自动重试 3 次；只有
所有结果都获得非空摘要时才会返回，否则会指出失败结果的序号和标题，不会输出含
`null` 摘要的列表。

对最终入选的每条 `scholar`、`document`、`webpage` 结果，程序还会：

- 每条最多收集 10 张候选图片，并保留 URL、`alt`、`title`、`figcaption` 和邻近文本；
- 访问来源网页，解析 `img`、`source/srcset` 和 Open Graph 图片；
- 对 PDF 来源使用 PyMuPDF 提取内嵌图片；
- 下载图片、转换为不带 `data:image/...` 前缀的 Base64 字符串；
- 本地排除尺寸异常、损坏、追踪像素、Logo、图标、广告及平台品牌图片；
- 将 20 条结果的图片文字上下文合并为一次 DeepSeek 请求，每条最多预选 3 张；
- 每条结果只调用一次 `qwen3.7-plus`，同时完成选图、相关性评分和图片摘要；
- 相关性不足 60 分时不强制选图，返回两个空列表；
- 将图片保存为 `image_base_64: list[str]`，将对应说明保存为 `image_summary: list[str]`；
- 两个列表严格按索引一一对应且长度相等；无图时二者都返回 `[]`。

图片抓取单张超时为 2 秒，本地图片过滤预算为每条 0.5 秒。图片阶段 50 秒后不再启动新任务，55 秒停止等待未完成任务；未完成、无关或处理失败的图片返回 `image_base_64: []` 和 `image_summary: []`，不能阻塞或取消 20 条文本结果。`qwen3.7-plus` 是代码传给阿里云百炼或兼容后端的精确模型名。

安装图片处理依赖：

```powershell
python -m pip install -r .\requirements.txt
```

受登录、付费墙、反爬虫、动态 JavaScript 页面或网络错误限制的图片可能无法抓取；这类
来源会返回当前实际获得的图片列表，不会伪造图片。
如使用 OpenAI Chat Completions 兼容的自建后端，可配置：

```powershell
$env:DEEPSEEK_API_URL = "https://你的后端地址/chat/completions"
```

Qwen 图片处理使用百炼 OpenAI 兼容的 Chat Completions API。若需要通过兼容后端调用，可配置：

```powershell
$env:QWEN_API_URL = "https://你的后端地址/v1/chat/completions"
```

秘塔 API Key 可在秘塔 AI 搜索的“API Keys”页面创建。DeepSeek API Key 可在
DeepSeek 开放平台创建。程序使用 `deepseek-v4-flash` 判断问题类型，并使用
`deepseek-v4-pro` 总结检索结果：

- 专业医学问题：12 条 `scholar`、4 条 `document`、4 条 `webpage`
- 普通患者问题：14 条 `webpage`、6 条 `document`
- 非医学问题：20 条 `webpage`

默认最终返回不超过 Top-20。多范围结果会合并，只要 URL、DOI、标题中的任意一项
相同便会去重，再根据问题相关性、秘塔评分和原始排名重新排序。某一范围不足或请求
失败时，会从其他成功范围自动补足。可以用 `query_type` 手动指定
`professional`、`patient` 或 `non_medical`；也可以传入 `scope` 强制只检索单一范围。

如需临时改用 DeepSeek V4 Pro：

```powershell
$env:DEEPSEEK_MODEL = "deepseek-v4-pro"
```

命令行示例：

```powershell
python .\metaso_search.py --query-type professional --size 20 "慢性肾病循证治疗进展"
python .\metaso_search.py --scope scholar --size 5 "SGLT2抑制剂随机对照试验"
```

## 交互测试

先在 PowerShell 中设置 API Key，然后运行测试程序：

```powershell
$env:METASO_API_KEY = "你的秘塔 API Key"
$env:DEEPSEEK_API_KEY = "你的 DeepSeek API Key"
$env:DASHSCOPE_API_KEY = "你的阿里云百炼 API Key"
$env:DEEPSEEK_MAX_CONCURRENCY = "15"
$env:QWEN_MAX_CONCURRENCY = "8"
python .\manual_test.py
```

看到 `请输入问题：` 后输入问题并按回车。程序会显示 `返回类型：list[Metadata]`、
结果数量以及每一条 `Metadata(...)` 的完整字段。
