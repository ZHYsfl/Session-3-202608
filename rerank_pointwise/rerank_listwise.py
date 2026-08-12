#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Listwise rerank for a RAG metadata_list.

Input : JSON file that is either a list of metadata dicts, or a dict that
        contains such a list under a key such as "metadata_list".
Output: the same JSON structure with the same fields, only the order changed.

The script talks to any OpenAI-compatible chat completions API
(DeepSeek / Qwen DashScope / local vLLM, ...). No third-party packages needed.
"""

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib import error, request


DEFAULT_RUBRIC = (
    "排序标准（按重要性从高到低）：\n"
    "1. 问题相关性：论文是否直接回答用户问题，是否提供了问题所需要的具体证据。\n"
    "2. 证据等级：指南/系统综述/Meta分析 高于 RCT，RCT 高于队列研究，"
    "队列研究 高于 病例对照，病例对照 高于 病例报告/专家意见。\n"
    "3. 时效性：近 5 年内的证据优先；除非是经典定义性文献，否则更新的证据更可靠。\n"
    "4. 来源可靠性：权威期刊、官方指南、注册临床试验比一般来源更可信。\n"
    "只能依据候选信息中给出的内容判断，不得编造论文没有的信息。"
)

LIST_KEYS = ["metadata_list", "results", "data", "items", "list"]

DEFAULT_FIELDS = {
    "id": "id",
    "title": "title",
    "summary": "text_summary",
    "evidence_level": "evidence_level",
    "source_type": "source_type",
    "year": "published_at",
    "query": "query",
}

DEFAULTS = {
    "input": "metadata_list.json",
    "output": "reranked_metadata_list.json",
    "cache_dir": "work",
    "list_path": "auto",
    "model": "deepseek-chat",
    "base_url": "https://api.deepseek.com/v1",
    "api_key_env": "DEEPSEEK_API_KEY",
    "api_key": "",
    "json_mode": False,
    "temperature": 0,
    "timeout": 180,
    "max_retries": 3,
    "max_workers": 4,
    "max_candidates_per_call": 30,
    "score_top_n": 40,
    "default_query": "",
    "fields": DEFAULT_FIELDS,
    "summary_fallbacks": ["abstract", "summary", "text"],
    "year_fallbacks": ["pub_year", "year", "publication_date", "date"],
    "query_fallbacks": ["question", "user_query", "q"],
    "rubric": DEFAULT_RUBRIC,
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="rerank_config.example.json",
                   help="JSON config file")
    p.add_argument("--input", help="input metadata_list JSON path")
    p.add_argument("--output", help="output JSON path")
    p.add_argument("--model")
    p.add_argument("--base-url")
    p.add_argument("--api-key-env")
    p.add_argument("--max-candidates-per-call", type=int)
    p.add_argument("--score-top-n", type=int)
    p.add_argument("--max-workers", type=int)
    p.add_argument("--dry-run", action="store_true",
                   help="parse input and show query groups without calling the LLM")
    p.add_argument("--fresh", action="store_true",
                   help="ignore cached scores and rerun everything")
    return p.parse_args()


def load_config(args):
    with open(args.config, encoding="utf-8") as f:
        cfg = json.load(f)
    for key, value in DEFAULTS.items():
        cfg.setdefault(key, value)
    if args.input:
        cfg["input"] = args.input
    if args.output:
        cfg["output"] = args.output
    if args.model:
        cfg["model"] = args.model
    if args.base_url:
        cfg["base_url"] = args.base_url
    if args.api_key_env:
        cfg["api_key_env"] = args.api_key_env
    if args.max_candidates_per_call:
        cfg["max_candidates_per_call"] = args.max_candidates_per_call
    if args.score_top_n:
        cfg["score_top_n"] = args.score_top_n
    if args.max_workers:
        cfg["max_workers"] = args.max_workers
    if not isinstance(cfg.get("fields"), dict):
        cfg["fields"] = dict(DEFAULT_FIELDS)
    return cfg


def load_input(path, list_path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data, None
    if not isinstance(data, dict):
        raise SystemExit("input JSON must be a list or an object containing a list")
    keys = [list_path] if list_path and list_path != "auto" else LIST_KEYS
    for key in keys:
        value = data.get(key)
        if isinstance(value, list):
            return value, key
    raise SystemExit(
        "cannot find a metadata list in input; set 'list_path' in config "
        "(e.g. \"metadata_list\", \"results\", or \"\" if the file itself is a list)"
    )


def internal_id(idx):
    return "paper_%d" % idx


def get_query(item, idx, cfg):
    fields = cfg["fields"]
    qf = fields.get("query")
    if qf and item.get(qf):
        return str(item[qf]).strip()
    for fallback in cfg.get("query_fallbacks", []):
        if item.get(fallback):
            return str(item[fallback]).strip()
    if cfg.get("default_query"):
        return str(cfg["default_query"]).strip()
    raise SystemExit(
        "cannot determine query for item %d; set 'fields.query' in config "
        "or 'default_query'" % idx
    )


def truncate(text, limit):
    text = str(text or "").strip().replace("\n", " ")
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def pick_field(item, field, fallbacks):
    if field and item.get(field) not in (None, ""):
        return item[field]
    for fallback in fallbacks:
        if item.get(fallback) not in (None, ""):
            return item[fallback]
    return ""


def candidate_text(item, idx, cfg):
    fields = cfg["fields"]
    title = truncate(item.get(fields.get("title") or "title", "") or "", 200)
    summary = truncate(
        pick_field(item, fields.get("summary"), cfg.get("summary_fallbacks", [])),
        800,
    )
    evidence = truncate(
        item.get(fields.get("evidence_level") or "evidence_level", "") or "",
        100,
    )
    source = truncate(
        item.get(fields.get("source_type") or "source_type", "") or "",
        100,
    )
    year = truncate(
        pick_field(item, fields.get("year"), cfg.get("year_fallbacks", [])),
        20,
    )
    lines = ["%d. id=%s" % (idx, internal_id(idx))]
    if title:
        lines.append("标题: %s" % title)
    if summary:
        lines.append("摘要/概述: %s" % summary)
    if evidence:
        lines.append("证据等级: %s" % evidence)
    if source:
        lines.append("来源类型: %s" % source)
    if year:
        lines.append("年份: %s" % year)
    return "\n".join(lines)


def resolve_api_key(cfg):
    env_name = cfg.get("api_key_env")
    if env_name and os.environ.get(env_name):
        return os.environ[env_name]
    if cfg.get("api_key"):
        return cfg["api_key"]
    base = str(cfg.get("base_url", ""))
    if base.startswith("http://localhost") or base.startswith("http://127.0.0.1"):
        return None
    raise SystemExit(
        "no API key found; set environment variable %s, or 'api_key' in config"
        % (env_name or "API_KEY")
    )


def call_llm(messages, cfg):
    url = cfg["base_url"].rstrip("/") + "/chat/completions"
    payload = {
        "model": cfg["model"],
        "messages": messages,
        "temperature": cfg.get("temperature", 0),
    }
    if cfg.get("json_mode"):
        payload["response_format"] = {"type": "json_object"}
    headers = {"Content-Type": "application/json"}
    key = resolve_api_key(cfg)
    if key:
        headers["Authorization"] = "Bearer " + key
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    last_error = None
    for attempt in range(1, cfg.get("max_retries", 3) + 1):
        req = request.Request(url, data=body, headers=headers, method="POST")
        try:
            with request.urlopen(req, timeout=cfg.get("timeout", 180)) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            return result["choices"][0]["message"]["content"]
        except (error.URLError, KeyError, ValueError) as exc:
            last_error = exc
            time.sleep(2 ** attempt)
    raise SystemExit("LLM call failed after retries: %s" % last_error)


def extract_json(text):
    if not text:
        raise ValueError("model returned empty output")
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except ValueError:
        pass
    for open_char, close_char in (("{", "}"), ("[", "]")):
        start = text.find(open_char)
        end = text.rfind(close_char)
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except ValueError:
                continue
    raise ValueError("cannot parse JSON from model output: %s" % text[:300])


def parse_reranked_ids(content):
    data = extract_json(content)
    if isinstance(data, list):
        return data
    if not isinstance(data, dict):
        raise ValueError("listwise output should be a JSON object or array")
    for key in ("reranked_ids", "ids", "order", "rerank", "ordered_ids"):
        if isinstance(data.get(key), list):
            return data[key]
    for value in data.values():
        if isinstance(value, list):
            return value
    raise ValueError("no reranked id list found in: %s" % str(data)[:300])


def parse_score(content):
    data = extract_json(content)
    if isinstance(data, list) and data:
        data = data[0]
    if not isinstance(data, dict):
        raise ValueError("score output should be a JSON object")
    sid = data.get("id") or data.get("paper_id") or data.get("doc_id")
    score = data.get("score", data.get("final_score", data.get("总分")))
    reason = data.get("reason", data.get("理由", ""))
    return str(sid), float(score), str(reason)


def listwise_messages(query, items, cfg):
    candidates = [candidate_text(item, idx, cfg) for item, idx in items]
    user = (
        "问题：%s\n\n候选论文列表：\n%s\n\n"
        "请按上述标准对候选论文重新排序，输出 JSON：\n"
        "{\"reranked_ids\": [\"paper_0\", \"paper_3\", ...], "
        "\"reasons\": {\"paper_0\": \"一句话理由\"}}\n"
        "reranked_ids 必须包含全部候选 id，且不重复。只输出 JSON。"
        % (query, "\n\n".join(candidates))
    )
    return [
        {"role": "system", "content": cfg["rubric"]},
        {"role": "user", "content": user},
    ]


def score_messages(query, item, idx, cfg):
    candidate = candidate_text(item, idx, cfg)
    user = (
        "问题：%s\n\n候选论文：\n%s\n\n"
        "请按上述标准对这篇论文打 0-10 分（10 为最相关、证据最强），"
        "输出 JSON：\n{\"id\": \"paper_%d\", \"score\": 8, "
        "\"reason\": \"一句话理由\"}\n只输出 JSON。"
        % (query, candidate, idx)
    )
    return [
        {"role": "system", "content": cfg["rubric"]},
        {"role": "user", "content": user},
    ]


def reorder_by_ids(items, ids):
    by_id = {internal_id(idx): (item, idx) for item, idx in items}
    ordered = []
    seen = set()
    for sid in ids:
        sid = str(sid)
        if sid in by_id and sid not in seen:
            ordered.append(by_id[sid])
            seen.add(sid)
    for item, idx in items:
        if internal_id(idx) not in seen:
            ordered.append((item, idx))
            seen.add(internal_id(idx))
    return ordered


def listwise_sort(items, query, cfg):
    """Sort one group listwise; splits into blocks when too large."""
    batch = cfg["max_candidates_per_call"]
    if len(items) <= batch:
        content = call_llm(listwise_messages(query, items, cfg), cfg)
        ids = parse_reranked_ids(content)
        reasons = {}
        try:
            data = extract_json(content)
            if isinstance(data, dict) and isinstance(data.get("reasons"), dict):
                reasons = data["reasons"]
        except ValueError:
            pass
        return reorder_by_ids(items, ids), reasons

    blocks = [items[i:i + batch] for i in range(0, len(items), batch)]
    ordered_blocks = []
    for block in blocks:
        content = call_llm(listwise_messages(query, block, cfg), cfg)
        ids = parse_reranked_ids(content)
        ordered_blocks.append(reorder_by_ids(block, ids))

    top_pool = []
    for block in ordered_blocks:
        top_pool.extend(block[:5])
    content = call_llm(listwise_messages(query, top_pool, cfg), cfg)
    ids = parse_reranked_ids(content)
    merged_top = reorder_by_ids(top_pool, ids)
    top_ids = {internal_id(idx) for _, idx in merged_top}
    final = list(merged_top)
    for block in ordered_blocks:
        for item, idx in block:
            if internal_id(idx) not in top_ids:
                final.append((item, idx))
    return final, {}


def load_score_cache(cfg, fresh):
    path = os.path.join(cfg["cache_dir"], "rerank_scores.json")
    if fresh or not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_score_cache(cfg, cache):
    os.makedirs(cfg["cache_dir"], exist_ok=True)
    path = os.path.join(cfg["cache_dir"], "rerank_scores.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def score_all(items, query, cfg, cache):
    results = dict(cache.get(query, {}))
    todo = []
    for item, idx in items:
        sid = internal_id(idx)
        if sid not in results:
            todo.append((item, idx, sid))
    if not todo:
        return results

    def work(task):
        item, idx, sid = task
        content = call_llm(score_messages(query, item, idx, cfg), cfg)
        parsed_id, score, reason = parse_score(content)
        return sid, {"score": score, "reason": reason, "model_id": parsed_id}

    with ThreadPoolExecutor(max_workers=cfg["max_workers"]) as executor:
        futures = [executor.submit(work, task) for task in todo]
        for future in as_completed(futures):
            sid, entry = future.result()
            results[sid] = entry
    cache[query] = results
    save_score_cache(cfg, cache)
    return results


def validate(original, reranked):
    if len(original) != len(reranked):
        raise SystemExit("validation failed: item count changed")
    problems = 0
    for index, (old, new) in enumerate(zip(original, reranked)):
        if set(old.keys()) != set(new.keys()):
            problems += 1
            print("WARNING: item %d keys differ: %s" % (index, sorted(set(old) ^ set(new))))
    if problems:
        raise SystemExit("validation failed: %d items have different fields" % problems)


def write_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def main():
    args = parse_args()
    cfg = load_config(args)
    original, container = load_input(cfg["input"], cfg["list_path"])
    if not original:
        raise SystemExit("input metadata_list is empty")

    groups = {}
    for idx, item in enumerate(original):
        query = get_query(item, idx, cfg)
        groups.setdefault(query, []).append((item, idx))

    if args.dry_run:
        for query, items in groups.items():
            print("query: %s (%d candidates)" % (query, len(items)))
            print(candidate_text(items[0][0], items[0][1], cfg))
            print("---")
        print("dry run ok; no LLM call made")
        return

    score_cache = load_score_cache(cfg, args.fresh)
    final_items = []
    details = {}
    for query, items in groups.items():
        print("processing %d candidates for query: %s" % (len(items), query[:80]))
        if len(items) <= cfg["score_top_n"]:
            ordered, reasons = listwise_sort(items, query, cfg)
            details[query] = {"method": "listwise", "reasons": reasons}
        else:
            scores = score_all(items, query, cfg, score_cache)
            top = sorted(
                items,
                key=lambda t: -scores[internal_id(t[1])]["score"],
            )[:cfg["score_top_n"]]
            ordered_top, reasons = listwise_sort(top, query, cfg)
            ordered = list(ordered_top)
            top_ids = {internal_id(idx) for _, idx in ordered_top}
            for item, idx in items:
                if internal_id(idx) not in top_ids:
                    ordered.append((item, idx))
            details[query] = {
                "method": "score+listwise",
                "scores": scores,
                "reasons": reasons,
            }
        final_items.extend(item for item, _ in ordered)

    validate(original, final_items)
    if container is None:
        output_data = final_items
    else:
        with open(cfg["input"], encoding="utf-8") as f:
            output_data = json.load(f)
        output_data[container] = final_items

    write_json(cfg["output"], output_data)
    os.makedirs(cfg["cache_dir"], exist_ok=True)
    write_json(
        os.path.join(cfg["cache_dir"], "rerank_details.json"),
        {"queries": details},
    )
    print("done: %d items -> %s" % (len(final_items), cfg["output"]))


if __name__ == "__main__":
    main()
