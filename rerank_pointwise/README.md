# RAG metadata_list 精排序

这是第 5 步 `rerank listwise` 的可运行脚本。输入粗排 `metadata_list`，输出同字段、同结构、仅顺序变化的 JSON。

## 文件

- `rerank_listwise.py`：主脚本，使用 Python 标准库，无需安装第三方包。
- `rerank_config.example.json`：配置模板，拿到粗排文件后改这里的字段名即可。
- `rerank_prompt.md`：Prompt 模板和评分规则说明。

## 快速开始

1. 把粗排文件复制为 `metadata_list.json`（或改配置里的 `input`）。
2. 编辑 `rerank_config.example.json`：
   - `input` / `output`：输入输出路径；
   - `model` / `base_url` / `api_key_env`：DeepSeek 或 Qwen 的模型、接口地址、环境变量名；
   - `fields`：把字段名改成你们 `metadata_list` 里的真实字段；
   - `default_query`：如果条目里没有 query 字段且只做一个全局排序，在这里写问题或留空。
3. 设置 API key：

```powershell
$env:DEEPSEEK_API_KEY = "sk-..."
```

4. 运行：

```powershell
python rerank_listwise.py --config rerank_config.example.json
```

先不花钱验证字段映射（不会调用大模型）：

```powershell
python rerank_listwise.py --config rerank_config.example.json --input work/sample_metadata.json --dry-run
```

## 本地 Ollama + Qwen2.5（完全免费，不联网）

1. 安装 [Ollama](https://ollama.com/download)，然后拉取模型：

```powershell
ollama pull qwen2.5:7b
```

没有 GPU 或内存小于 8GB 时建议用更小的模型：

```powershell
ollama pull qwen2.5:3b
```

2. 确认 Ollama 服务在运行（Windows 安装后一般自动后台运行）：

```powershell
ollama serve
```

或在浏览器打开 `http://localhost:11434` 检查。

3. 使用 `rerank_config.ollama.json` 运行，本地接口不需要 API key：

```powershell
python rerank_listwise.py --config rerank_config.ollama.json
```

如果换模型，把配置里的 `model` 改成 `qwen2.5:3b`、`qwen2.5:14b` 等即可。本地 CPU 推理较慢，建议把 `max_workers` 保持为 1，并把 `max_candidates_per_call` 调小到 15-20。

## 拿到粗排文件后需要确认的字段

脚本会按配置自动取以下字段：

- `id`：条目唯一标识（没有也可以，脚本内部会用 `paper_序号` 代替）；
- `title`：标题；
- `summary`：摘要或概述，例如 `text_summary` / `abstract`；
- `evidence_level`：证据等级；
- `source_type`：来源类型（指南、RCT、期刊、临床试验等）；
- `year`：发表年份（没有年份字段时，脚本会尝试 `pub_year` / `year` / `date`）；
- `query`：条目对应的用户问题；如果没有，用 `default_query`。

`list_path` 用来告诉脚本列表在 JSON 里的位置：

- 整个文件就是列表：留 `"auto"` 即可；
- 文件是 `{"query": "...", "metadata_list": [...]}`：也会自动找到 `metadata_list`；
- 如果键名特殊，改成真实键名，例如 `"results"`。

## 输出和中间产物

- `output`：最终文件，和输入同字段、同结构，只改顺序；
- `work/rerank_scores.json`：逐篇打分缓存，重跑复用；
- `work/rerank_details.json`：每个 query 的精排方式、分数和理由，便于第 6 步人工复核。

强制重新计算（不使用缓存）：

```powershell
python rerank_listwise.py --config rerank_config.example.json --fresh
```
