"""调用秘塔搜索 API，并把最终结果转换为 ``list[Metadata]``。"""

from __future__ import annotations

import base64
import asyncio
import hashlib
import io
import ipaddress
import json
import os
import re
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Iterable, TypeVar
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup

from matadata import Metadata

try:
    import pymupdf
except ImportError:  # pragma: no cover - requirements.txt 已声明该依赖
    pymupdf = None

try:
    from PIL import Image
except ImportError:  # pragma: no cover - 仅影响非常规图片格式/缩放
    Image = None


METASO_SEARCH_URL = "https://metaso.cn/api/v1/search"
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = "deepseek-v4-flash"
DEEPSEEK_SUMMARY_MODEL = "deepseek-v4-pro"
QWEN_API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
QWEN_IMAGE_MODEL = "qwen3.7-plus"

VALID_SCOPES = {"webpage", "document", "scholar", "podcast", "video", "image"}
_RESULT_KEYS = ("webpages", "documents", "scholars", "podcasts", "videos", "images")
_QUERY_TYPES = {"auto", "professional", "patient", "non_medical"}
_STRATEGIES = {
    "professional": (("scholar", 0.6), ("document", 0.2), ("webpage", 0.2)),
    "patient": (("webpage", 0.7), ("document", 0.3)),
    "non_medical": (("webpage", 1.0),),
}
MAX_SOURCE_BYTES = 30 * 1024 * 1024
MAX_IMAGE_BYTES = 15 * 1024 * 1024
MAX_RELEVANT_IMAGES = 1
IMAGE_SELECTION_BATCH_SIZE = 8
MAX_IMAGE_CANDIDATES = 10
DEEPSEEK_PRESELECT_IMAGES = 3
IMAGE_DOWNLOAD_TIMEOUT = 2.0
LOCAL_IMAGE_FILTER_BUDGET = 0.5
IMAGE_RELEVANCE_THRESHOLD = 60.0
IMAGE_START_CUTOFF_SECONDS = 50.0
IMAGE_HARD_DEADLINE_SECONDS = 55.0
MIN_IMAGE_SIDE = 11
MIN_IMAGE_AREA = 32 * 32
MAX_IMAGE_ASPECT_RATIO = 20.0
DEFAULT_DEEPSEEK_CONCURRENCY = 15
DEFAULT_QWEN_CONCURRENCY = 8
DEFAULT_IMAGE_FETCH_CONCURRENCY = 8

# 独立别名便于分别测试/替换三个后端。
deepseek_urlopen = urlopen
image_urlopen = urlopen
qwen_urlopen = urlopen


class MetasoAPIError(RuntimeError):
    """秘塔 API 请求或响应异常。"""


class DeepSeekAPIError(RuntimeError):
    """DeepSeek API 请求或响应异常。"""


class QwenImageAPIError(RuntimeError):
    """Qwen 图片筛选或图片摘要异常。"""

    def __init__(
        self, message: str, *, status_code: int | None = None, detail: str = ""
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.detail = detail

    @property
    def is_invalid_image(self) -> bool:
        lowered = self.detail.lower()
        return self.status_code == 400 and any(
            marker in lowered
            for marker in (
                "image length and width",
                "image size",
                "image width",
                "image height",
                "invalid image",
                "image format",
                "image_url",
            )
        )


def _configured_concurrency(environment_name: str, default: int) -> int:
    raw_value = os.getenv(environment_name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{environment_name} 必须是正整数") from exc
    if value < 1:
        raise ValueError(f"{environment_name} 必须是正整数")
    return value


def _run_async(coroutine: Any) -> Any:
    """从同步入口执行协程；若调用方已有事件循环，则放到独立线程执行。"""

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(asyncio.run, coroutine).result()


async def _run_queue_workers(
    jobs: list[Any],
    process: Callable[[Any], Any],
    max_concurrency: int,
) -> list[BaseException]:
    """固定 worker 消费队列：完成一个任务后立即出队下一个任务。"""

    if not jobs:
        return []
    queue: asyncio.Queue[Any] = asyncio.Queue()
    for job in jobs:
        queue.put_nowait(job)
    failures: list[BaseException] = []

    async def worker() -> None:
        while True:
            try:
                job = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                await process(job)
            except BaseException as exc:
                failures.append(exc)
            finally:
                queue.task_done()

    workers = [
        asyncio.create_task(worker())
        for _ in range(min(max_concurrency, len(jobs)))
    ]
    await queue.join()
    await asyncio.gather(*workers)
    return failures


def _first_text(item: dict[str, Any], keys: Iterable[str]) -> str | None:
    for key in keys:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _extract_year(item: dict[str, Any]) -> str | None:
    for key in ("year", "date", "publishDate", "publishedAt", "publish_time"):
        value = item.get(key)
        if value is not None:
            match = re.search(r"(?:19|20)\d{2}", str(value))
            if match:
                return match.group(0)
    return None


def _stable_id(item: dict[str, Any]) -> str:
    supplied_id = item.get("id")
    if supplied_id is not None and str(supplied_id).strip():
        return str(supplied_id)
    identity = _first_text(item, ("doi", "link", "url", "sourceUrl", "source_url", "title"))
    return str(uuid.uuid5(uuid.NAMESPACE_URL, identity)) if identity else str(uuid.uuid4())


def _normalize_base64(value: str) -> str | None:
    candidate = value.strip()
    if candidate.startswith("data:"):
        header, separator, candidate = candidate.partition(",")
        if not separator or ";base64" not in header.lower():
            return None
    try:
        decoded = base64.b64decode(candidate, validate=True)
    except (ValueError, base64.binascii.Error):
        return None
    return base64.b64encode(decoded).decode("ascii") if decoded else None


def _embedded_images(item: dict[str, Any]) -> list[str]:
    """递归收集秘塔响应中直接携带的 Base64 图片。"""

    encoded: list[str] = []

    def visit(value: Any, parent_key: str = "") -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                visit(child, str(key).lower())
        elif isinstance(value, list):
            for child in value:
                visit(child, parent_key)
        elif isinstance(value, str):
            looks_encoded = (
                "base64" in parent_key
                or parent_key in {"data", "image", "images"}
                or value.startswith("data:image/")
                or value.startswith(("/9j/", "iVBOR", "R0lGOD", "UklGR"))
            )
            if looks_encoded:
                normalized = _normalize_base64(value)
                if normalized:
                    encoded.append(normalized)

    visit(item)
    return list(dict.fromkeys(encoded))


def _infer_evidence_level(item: dict[str, Any]) -> str | None:
    content = " ".join(
        str(item.get(key, ""))
        for key in ("title", "summary", "snippet", "articleType", "article_type")
    ).lower()
    rules = (
        ("1A", ("meta-analysis of randomized", "systematic review of randomized", "随机对照试验的系统综述", "随机对照试验的meta分析")),
        ("1B", ("randomized controlled trial", "randomised controlled trial", "随机对照试验", " rct ")),
        ("2A", ("systematic review of cohort", "队列研究的系统综述")),
        ("2B", ("cohort study", "队列研究")),
        ("3A", ("systematic review of case-control", "病例对照研究的系统综述")),
        ("3B", ("case-control study", "病例对照研究")),
        ("4", ("case series", "case report", "病例系列", "病例报告")),
        ("5", ("expert opinion", "narrative review", "专家意见", "叙述性综述")),
    )
    padded = f" {content} "
    return next((level for level, terms in rules if any(term in padded for term in terms)), None)


def _result_items(payload: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    container = payload.get("data") if isinstance(payload.get("data"), dict) else payload
    found: list[tuple[str, dict[str, Any]]] = []
    for key in _RESULT_KEYS:
        values = container.get(key)
        if isinstance(values, list):
            found.extend((key.removesuffix("s"), value) for value in values if isinstance(value, dict))
    if not found:
        for key in ("results", "items"):
            values = container.get(key)
            if isinstance(values, list):
                found.extend((str(value.get("scope") or value.get("type") or "webpage"), value) for value in values if isinstance(value, dict))
                break
    return found


def _to_metadata(result_scope: str, item: dict[str, Any]) -> Metadata:
    source_type = _first_text(item, ("source_type", "articleType", "article_type", "type"))
    if source_type is None or source_type in {"webpages", "documents", "scholars"}:
        source_type = result_scope.removesuffix("s")
    text = _first_text(item, ("rawContent", "raw_content", "content", "snippet", "description"))
    summary = _first_text(item, ("summary", "textSummary", "text_summary"))
    return Metadata(
        title=_first_text(item, ("title", "name")),
        year=_extract_year(item),
        source_type=source_type,
        id=_stable_id(item),
        text=text or summary,
        text_summary=summary,
        image_base_64=_embedded_images(item),
        image_summary=[],
        evidence_level=_infer_evidence_level(item),
        last_retrieved_at=None,
    )


def _decode_json_response(raw_response: bytes, service: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw_response.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MetasoAPIError(f"{service} 返回了无效 JSON") from exc
    if not isinstance(payload, dict):
        raise MetasoAPIError(f"{service} 响应顶层必须是 JSON 对象")
    return payload


def _request_scope(
    question: str,
    *,
    token: str,
    scope: str,
    size: int,
    include_summary: bool,
    include_raw_content: bool,
    concise_snippet: bool,
    timeout: float,
    api_url: str,
    retries: int = 3,
) -> list[tuple[str, int, dict[str, Any]]]:
    body = json.dumps(
        {
            "q": question,
            "scope": scope,
            "size": size,
            "includeSummary": include_summary,
            "includeRawContent": include_raw_content,
            "conciseSnippet": concise_snippet,
        },
        ensure_ascii=False,
    ).encode("utf-8")
    request = Request(
        api_url,
        data=body,
        method="POST",
        headers={"Accept": "application/json", "Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    last_error: Exception | None = None
    payload: dict[str, Any] | None = None
    for attempt in range(retries):
        try:
            with urlopen(request, timeout=timeout) as response:
                payload = _decode_json_response(response.read(), "秘塔 API")
            break
        except HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", errors="replace")
            except (OSError, TimeoutError):
                detail = "无法读取错误响应"
            last_error = MetasoAPIError(f"秘塔 API 请求失败（HTTP {exc.code}）：{detail}")
            if exc.code not in {408, 409, 429, 500, 502, 503, 504}:
                raise last_error from exc
        except URLError as exc:
            last_error = MetasoAPIError(f"无法连接秘塔 API：{exc.reason}")
        except (TimeoutError, OSError) as exc:
            last_error = MetasoAPIError(f"秘塔 API 的 {scope} 范围读取超时或连接中断：{exc}")
        if attempt + 1 < retries:
            time.sleep(0.5 * (2**attempt))
    if payload is None:
        raise MetasoAPIError(
            f"秘塔 API 的 {scope} 范围重试 {retries} 次后仍失败：{last_error}"
        ) from last_error
    if payload.get("errCode") not in (None, 0):
        message = payload.get("errMsg") or payload.get("message") or "未知错误"
        raise MetasoAPIError(f"秘塔 API 返回错误 {payload['errCode']}：{message}")
    items = [(item_scope or scope, item) for item_scope, item in _result_items(payload)]
    return [(scope, rank, item) for rank, (_item_scope, item) in enumerate(items)]


def _scope_quotas(query_type: str, size: int) -> tuple[list[str], dict[str, int]]:
    strategy = _STRATEGIES[query_type]
    exact = [(scope, size * weight) for scope, weight in strategy]
    quotas = {scope: int(value) for scope, value in exact}
    remainder = size - sum(quotas.values())
    order = sorted(
        enumerate(exact),
        key=lambda indexed: (-(indexed[1][1] - int(indexed[1][1])), indexed[0]),
    )
    for _index, (scope, _value) in order[:remainder]:
        quotas[scope] += 1
    return [scope for scope, _ in strategy], quotas


def _canonical_url(item: dict[str, Any]) -> str | None:
    value = _first_text(item, ("link", "url", "sourceUrl", "source_url"))
    if not value:
        return None
    try:
        parts = urlsplit(value)
    except ValueError:
        return value.strip().lower()
    if not parts.netloc:
        return value.strip().lower()
    ignored = {"utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content", "tracking", "ref"}
    query = urlencode([(k, v) for k, v in parse_qsl(parts.query) if k.lower() not in ignored])
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path.rstrip("/"), query, ""))


def _doi(item: dict[str, Any]) -> str | None:
    text = " ".join(str(item.get(key, "")) for key in ("doi", "link", "url", "title", "summary"))
    match = re.search(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", text, re.I)
    return match.group(0).rstrip(".,;)").lower() if match else None


def _normalized_title(item: dict[str, Any]) -> str | None:
    title = _first_text(item, ("title", "name"))
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", title).lower() if title else None


def _dedupe_keys(item: dict[str, Any]) -> set[str]:
    keys: set[str] = set()
    if value := _canonical_url(item):
        keys.add(f"url:{value}")
    if value := _doi(item):
        keys.add(f"doi:{value}")
    if value := _normalized_title(item):
        keys.add(f"title:{value}")
    return keys


def _candidate_score(question: str, candidate: tuple[str, int, dict[str, Any]]) -> float:
    _scope, rank, item = candidate
    question_terms = set(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", question.lower()))
    content = " ".join(str(item.get(key, "")) for key in ("title", "summary", "snippet", "description")).lower()
    relevance = sum(1 for term in question_terms if term in content) / max(len(question_terms), 1)
    raw_score = item.get("score") or item.get("relevanceScore") or item.get("relevance_score")
    if isinstance(raw_score, (int, float)):
        metaso_score = float(raw_score)
        if metaso_score > 1:
            metaso_score /= 100
    else:
        metaso_score = {"high": 1.0, "medium": 0.6, "low": 0.2}.get(str(raw_score).lower(), 0.5)
    return relevance * 0.55 + metaso_score * 0.30 + (1 / (rank + 1)) * 0.15


def _select_candidates(
    question: str,
    candidates: list[tuple[str, int, dict[str, Any]]],
    scopes: list[str],
    quotas: dict[str, int],
    size: int,
) -> list[tuple[str, int, dict[str, Any]]]:
    by_scope = {
        scope: sorted((item for item in candidates if item[0] == scope), key=lambda item: _candidate_score(question, item), reverse=True)
        for scope in scopes
    }
    selected: list[tuple[str, int, dict[str, Any]]] = []
    selected_ids: set[int] = set()
    seen_keys: set[str] = set()

    def add(candidate: tuple[str, int, dict[str, Any]]) -> bool:
        keys = _dedupe_keys(candidate[2])
        if keys and keys & seen_keys:
            return False
        selected.append(candidate)
        selected_ids.add(id(candidate))
        seen_keys.update(keys)
        return True

    for scope in scopes:
        added = 0
        for candidate in by_scope[scope]:
            if add(candidate):
                added += 1
                if added >= quotas[scope]:
                    break
    leftovers = sorted((item for item in candidates if id(item) not in selected_ids), key=lambda item: _candidate_score(question, item), reverse=True)
    for candidate in leftovers:
        if len(selected) >= size:
            break
        add(candidate)
    return selected[:size]


def _deepseek_content(payload: dict[str, Any]) -> str:
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise DeepSeekAPIError("DeepSeek 返回的响应格式无效") from exc
    if isinstance(content, list):
        content = "".join(str(block.get("text", "")) for block in content if isinstance(block, dict))
    return content.strip() if isinstance(content, str) else ""


def _deepseek_request(
    messages: list[dict[str, str]],
    *,
    api_key: str,
    model: str,
    timeout: float,
    json_mode: bool = False,
    retries: int = 3,
    api_url: str | None = None,
) -> str:
    endpoint = api_url or os.getenv("DEEPSEEK_API_URL", DEEPSEEK_API_URL)
    body: dict[str, Any] = {"model": model, "messages": messages, "stream": False, "thinking": {"type": "disabled"}}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    last_error: Exception | None = None
    for attempt in range(retries):
        request = Request(endpoint, data=json.dumps(body, ensure_ascii=False).encode("utf-8"), method="POST", headers={"Accept": "application/json", "Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
        try:
            with deepseek_urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            content = _deepseek_content(payload)
            if not content:
                raise DeepSeekAPIError("DeepSeek 返回了空摘要")
            return content
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            last_error = DeepSeekAPIError(f"DeepSeek API 请求失败（HTTP {exc.code}）：{detail}")
            if exc.code not in {408, 409, 429, 500, 502, 503, 504}:
                break
        except (URLError, OSError, json.JSONDecodeError, DeepSeekAPIError) as exc:
            last_error = exc
        if attempt + 1 < retries:
            time.sleep(0.5 * (2**attempt))
    raise DeepSeekAPIError(f"重试 {retries} 次后仍失败：{last_error}") from last_error


def classify_question(
    question: str,
    *,
    api_key: str | None = None,
    model: str = DEEPSEEK_MODEL,
    timeout: float = 30.0,
    api_url: str | None = None,
) -> str:
    token = api_key or os.getenv("DEEPSEEK_API_KEY")
    if not token:
        raise ValueError("未找到 DEEPSEEK_API_KEY")
    content = _deepseek_request(
        [
            {"role": "system", "content": "把问题仅分类为 professional、patient、non_medical 之一，并只返回 JSON：{\"category\":\"...\"}。专业医学检索归 professional；患者口语健康问题归 patient；非医学归 non_medical。"},
            {"role": "user", "content": question},
        ],
        api_key=token,
        model=model,
        timeout=timeout,
        json_mode=True,
        api_url=api_url,
    )
    try:
        category = json.loads(content)["category"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise DeepSeekAPIError("DeepSeek 返回的分类响应格式无效") from exc
    if category not in _STRATEGIES:
        raise DeepSeekAPIError(f"不支持的问题类型：{category}")
    return category


def summarize_metadata(
    question: str,
    results: list[Metadata],
    *,
    api_key: str | None = None,
    model: str = DEEPSEEK_SUMMARY_MODEL,
    timeout: float = 30.0,
    api_url: str | None = None,
) -> list[Metadata]:
    token = api_key or os.getenv("DEEPSEEK_API_KEY")
    if not token:
        raise ValueError("未找到 DEEPSEEK_API_KEY")
    for index, item in enumerate(results, start=1):
        source_text = item.text or item.title or "（来源未提供正文，仅能依据标题总结）"
        try:
            item.text_summary = _deepseek_request(
                [
                    {"role": "system", "content": f"针对用户问题生成准确、简洁、非空的中文资料摘要。不得编造原文未提供的信息。用户问题：{question}"},
                    {"role": "user", "content": json.dumps({"text": source_text}, ensure_ascii=False)},
                ],
                api_key=token,
                model=model,
                timeout=timeout,
                api_url=api_url,
            )
        except DeepSeekAPIError as exc:
            raise DeepSeekAPIError(f"第 {index} 条结果总结失败（{item.title or '无标题'}）：{exc}") from exc
    return results


async def summarize_metadata_async(
    question: str,
    results: list[Metadata],
    *,
    api_key: str | None = None,
    model: str = DEEPSEEK_SUMMARY_MODEL,
    timeout: float = 30.0,
    api_url: str | None = None,
    max_concurrency: int | None = None,
) -> list[Metadata]:
    """并行生成文本摘要，同时限制发往 DeepSeek 的在途请求数。"""

    token = api_key or os.getenv("DEEPSEEK_API_KEY")
    if not token:
        raise ValueError("未找到 DEEPSEEK_API_KEY")
    concurrency = max_concurrency or _configured_concurrency(
        "DEEPSEEK_MAX_CONCURRENCY", DEFAULT_DEEPSEEK_CONCURRENCY
    )
    if concurrency < 1:
        raise ValueError("DeepSeek 最大并发量必须是正整数")

    failed_jobs: list[tuple[int, Metadata, DeepSeekAPIError]] = []

    def request_summary(item: Metadata, *, retries: int, request_timeout: float) -> str:
        source_text = item.text or item.title or "（来源未提供正文，仅能依据标题总结）"
        return _deepseek_request(
            [
                {
                    "role": "system",
                    "content": f"针对用户问题生成准确、简洁、非空的中文资料摘要。不得编造原文未提供的信息。用户问题：{question}",
                },
                {
                    "role": "user",
                    "content": json.dumps({"text": source_text}, ensure_ascii=False),
                },
            ],
            api_key=token,
            model=model,
            timeout=request_timeout,
            api_url=api_url,
            retries=retries,
        )

    async def summarize_one(job: tuple[int, Metadata]) -> None:
        index, item = job
        try:
            item.text_summary = await asyncio.to_thread(
                request_summary,
                item,
                retries=2,
                request_timeout=min(timeout, 18.0),
            )
        except DeepSeekAPIError as exc:
            # 首轮失败不终止整个 Top-N；失败项稍后以较低并发重新入队。
            failed_jobs.append((index, item, exc))

    await _run_queue_workers(
        list(enumerate(results, start=1)), summarize_one, concurrency
    )

    if failed_jobs:
        recovery_failures: list[tuple[int, Metadata, DeepSeekAPIError]] = []

        async def recover_one(job: tuple[int, Metadata, DeepSeekAPIError]) -> None:
            index, item, _first_error = job
            try:
                item.text_summary = await asyncio.to_thread(
                    request_summary,
                    item,
                    retries=1,
                    request_timeout=min(timeout, 10.0),
                )
            except DeepSeekAPIError as exc:
                recovery_failures.append((index, item, exc))

        # TLS EOF/限流常由瞬时并发造成；只把失败项以低并发补偿，不拖慢成功项。
        await _run_queue_workers(
            failed_jobs,
            recover_one,
            min(3, len(failed_jobs)),
        )

        for _index, item, _error in recovery_failures:
            # DeepSeek 连续不可用时也不能让单条结果导致全部结果失败或返回 null。
            existing = (item.text_summary or "").strip()
            source = re.sub(r"\s+", " ", item.text or item.title or "来源未提供可摘要内容").strip()
            item.text_summary = existing or f"来源内容摘录：{source[:800]}"

    return results


def _source_url(item: dict[str, Any]) -> str | None:
    value = _first_text(item, ("link", "url", "sourceUrl", "source_url"))
    if value is None:
        for key in ("file_meta", "fileMeta", "source", "document"):
            nested = item.get(key)
            if isinstance(nested, dict):
                value = _first_text(nested, ("link", "url", "sourceUrl", "source_url"))
                if value:
                    break
    return urljoin("https://metaso.cn", value) if value and value.startswith("/") else value


def _is_public_http_url(value: str) -> bool:
    parts = urlsplit(value)
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        return False
    hostname = parts.hostname.lower().rstrip(".")
    if hostname == "localhost" or hostname.endswith(".localhost"):
        return False
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        return True
    return not (address.is_private or address.is_loopback or address.is_link_local or address.is_multicast or address.is_reserved or address.is_unspecified)


def _read_url(
    url: str, *, timeout: float, max_bytes: int
) -> tuple[bytes, str, str, str | None]:
    if not _is_public_http_url(url):
        raise ValueError("只允许获取公开 HTTP/HTTPS 来源")
    request = Request(url, headers={"Accept": "text/html,application/pdf,image/*;q=0.9,*/*;q=0.5", "User-Agent": "Mozilla/5.0 (compatible; MetasoMetadataBot/1.0)"})
    with image_urlopen(request, timeout=timeout) as response:
        final_url = response.geturl() if hasattr(response, "geturl") else url
        if not _is_public_http_url(final_url):
            raise ValueError("重定向后的地址不是公开 HTTP/HTTPS 地址")
        data = response.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise ValueError(f"响应超过大小限制 {max_bytes} 字节")
        headers = getattr(response, "headers", None)
        charset: str | None = None
        if headers is not None and hasattr(headers, "get_content_type"):
            content_type = headers.get_content_type()
            if hasattr(headers, "get_content_charset"):
                charset = headers.get_content_charset()
        elif headers is not None and hasattr(headers, "get"):
            content_type_header = str(headers.get("Content-Type", ""))
            content_type = content_type_header.split(";", 1)[0].strip()
            match = re.search(r"charset\s*=\s*['\"]?([^;\s'\"]+)", content_type_header, re.I)
            if match:
                charset = match.group(1)
        else:
            content_type = ""
    return data, content_type.lower(), final_url, charset


def _image_references_from_item(item: dict[str, Any]) -> list[str]:
    references: list[str] = []

    def visit(value: Any, parent_key: str = "") -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                visit(child, str(key).lower())
        elif isinstance(value, list):
            for child in value:
                visit(child, parent_key)
        elif isinstance(value, str) and any(marker in parent_key for marker in ("image", "figure", "thumbnail", "cover")):
            if value.startswith("data:image/") or value.startswith(("http://", "https://", "/")):
                references.append(value)

    visit(item)
    return list(dict.fromkeys(references))


def _decode_html(html: bytes, declared_charset: str | None = None) -> str:
    """按 HTTP/BOM/meta 声明解码 HTML，最终以替换模式兜底且不触发 bs4 警告。"""

    candidates: list[str] = []
    if declared_charset:
        candidates.append(declared_charset)
    if html.startswith(b"\xef\xbb\xbf"):
        candidates.append("utf-8-sig")
    elif html.startswith((b"\xff\xfe", b"\xfe\xff")):
        candidates.append("utf-16")
    head = html[:8192].decode("ascii", errors="ignore")
    match = re.search(
        r"<meta[^>]+charset\s*=\s*['\"]?\s*([a-zA-Z0-9._-]+)", head, re.I
    ) or re.search(
        r"<meta[^>]+content\s*=\s*['\"][^'\"]*charset\s*=\s*([a-zA-Z0-9._-]+)",
        head,
        re.I,
    )
    if match:
        candidates.append(match.group(1))
    candidates.extend(("utf-8", "gb18030", "big5", "windows-1252"))
    seen: set[str] = set()
    for encoding in candidates:
        normalized = encoding.strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        try:
            return html.decode(normalized)
        except (LookupError, UnicodeDecodeError):
            continue
    return html.decode(declared_charset or "utf-8", errors="replace")


def _image_references_from_html(
    html: bytes, base_url: str, declared_charset: str | None = None
) -> list[str]:
    return [
        candidate["url"]
        for candidate in _image_candidates_from_html(
            html, base_url, declared_charset
        )
    ]


def _image_candidates_from_html(
    html: bytes, base_url: str, declared_charset: str | None = None
) -> list[dict[str, str]]:
    """提取图片URL及其用于 DeepSeek 文本预筛的页面上下文。"""

    soup = BeautifulSoup(_decode_html(html, declared_charset), "html.parser")
    candidates: list[dict[str, str]] = []

    def add(url: str, tag: Any = None, context: str = "") -> None:
        absolute = urljoin(base_url, url.strip())
        if not absolute:
            return
        alt = str(tag.get("alt") or "") if tag is not None else ""
        title = str(tag.get("title") or "") if tag is not None else ""
        figure = tag.find_parent("figure") if tag is not None else None
        caption_tag = figure.find("figcaption") if figure is not None else None
        caption = caption_tag.get_text(" ", strip=True) if caption_tag else ""
        parent_text = ""
        if tag is not None and tag.parent is not None:
            parent_text = tag.parent.get_text(" ", strip=True)[:500]
        candidates.append(
            {
                "url": absolute,
                "alt": alt[:300],
                "title": title[:300],
                "caption": caption[:500],
                "context": (context or parent_text)[:500],
            }
        )

    for tag in soup.find_all(["img", "source"]):
        for attribute in ("src", "data-src", "data-original", "data-lazy-src"):
            value = tag.get(attribute)
            if isinstance(value, str) and value.strip():
                add(value, tag)
        srcset = tag.get("srcset") or tag.get("data-srcset")
        if isinstance(srcset, str):
            for part in srcset.split(","):
                if part.strip():
                    add(part.strip().split(" ", 1)[0], tag)
    for tag in soup.find_all("meta"):
        name = str(tag.get("property") or tag.get("name") or "").lower()
        if name in {"og:image", "og:image:url", "twitter:image", "twitter:image:src"}:
            value = tag.get("content")
            if isinstance(value, str) and value.strip():
                add(value, context=name)
    unique: list[dict[str, str]] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate["url"] not in seen:
            seen.add(candidate["url"])
            unique.append(candidate)
    return unique


def _images_from_pdf(data: bytes) -> list[str]:
    if pymupdf is None:
        return []
    encoded: list[str] = []
    seen_xrefs: set[int] = set()
    try:
        with pymupdf.open(stream=data, filetype="pdf") as document:
            for page in document:
                for image_info in page.get_images(full=True):
                    xref = int(image_info[0])
                    if xref in seen_xrefs:
                        continue
                    seen_xrefs.add(xref)
                    image = document.extract_image(xref).get("image")
                    if isinstance(image, bytes) and image:
                        encoded.append(base64.b64encode(image).decode("ascii"))
    except (RuntimeError, ValueError):
        return []
    return encoded


def _download_image(reference: str, base_url: str, timeout: float) -> str | None:
    if reference.startswith("data:image/"):
        return _normalize_base64(reference)
    image_url = urljoin(base_url, reference)
    if not _is_public_http_url(image_url):
        return None
    try:
        data, content_type, _, _charset = _read_url(
            image_url, timeout=timeout, max_bytes=MAX_IMAGE_BYTES
        )
    except (HTTPError, URLError, OSError, ValueError):
        return None
    looks_like_image = content_type.startswith("image/") or data.startswith((b"\xff\xd8\xff", b"\x89PNG\r\n\x1a\n", b"GIF87a", b"GIF89a", b"RIFF"))
    return base64.b64encode(data).decode("ascii") if data and looks_like_image else None


def _obviously_irrelevant_image_context(candidate: dict[str, str]) -> bool:
    text = " ".join(candidate.values()).lower()
    markers = (
        "logo", "icon", "favicon", "avatar", "sprite", "banner", "advert",
        "tracking", "tracker", "pixel", "analytics", "semantic-scholar",
        "semanticscholar", "二维码", "网站标志", "公众号",
    )
    return any(marker in text for marker in markers)


def _collect_result_image_candidates(
    item: dict[str, Any],
    timeout: float = IMAGE_DOWNLOAD_TIMEOUT,
    max_candidates: int = MAX_IMAGE_CANDIDATES,
) -> list[dict[str, str]]:
    """抓取最多10张候选图片，同时保留图片周边文本供 DeepSeek 预筛。"""

    collected: list[dict[str, str]] = []
    embedded = _embedded_images(item)
    for encoded in embedded[:max_candidates]:
        collected.append(
            {"base64": encoded, "url": "embedded", "alt": "", "title": "", "caption": "", "context": "秘塔响应内嵌图片"}
        )
    source_url = _source_url(item)
    base_url = source_url or "https://metaso.cn/"
    references = [
        {"url": reference, "alt": "", "title": "", "caption": "", "context": "秘塔图片字段"}
        for reference in _image_references_from_item(item)
    ]
    if source_url and _is_public_http_url(source_url) and len(collected) < max_candidates:
        try:
            source_data, content_type, final_url, charset = _read_url(
                source_url,
                timeout=min(timeout, IMAGE_DOWNLOAD_TIMEOUT),
                max_bytes=MAX_SOURCE_BYTES,
            )
            if content_type == "application/pdf" or source_data.startswith(b"%PDF-"):
                for page_index, encoded in enumerate(_images_from_pdf(source_data)):
                    collected.append(
                        {"base64": encoded, "url": f"{final_url}#image-{page_index}", "alt": "", "title": "", "caption": "PDF内嵌图片", "context": ""}
                    )
                    if len(collected) >= max_candidates:
                        break
            elif content_type.startswith("image/"):
                collected.append(
                    {"base64": base64.b64encode(source_data).decode("ascii"), "url": final_url, "alt": "", "title": "", "caption": "", "context": "来源图片"}
                )
            else:
                references.extend(
                    _image_candidates_from_html(source_data, final_url, charset)
                )
        except (HTTPError, URLError, OSError, ValueError, TimeoutError):
            pass
    seen_urls: set[str] = set()
    download_candidates: list[dict[str, str]] = []
    remaining = max(0, max_candidates - len(collected))
    for candidate in references:
        reference = candidate.get("url", "")
        if (
            not reference
            or reference in seen_urls
            or _obviously_irrelevant_image_context(candidate)
        ):
            continue
        seen_urls.add(reference)
        download_candidates.append(candidate)
        if len(download_candidates) >= remaining:
            break
    if download_candidates:
        with ThreadPoolExecutor(max_workers=min(6, len(download_candidates))) as executor:
            futures = {
                executor.submit(
                    _download_image,
                    candidate["url"],
                    base_url,
                    min(timeout, IMAGE_DOWNLOAD_TIMEOUT),
                ): candidate
                for candidate in download_candidates
            }
            downloaded: dict[str, dict[str, str]] = {}
            for future in as_completed(futures):
                candidate = futures[future]
                try:
                    encoded = future.result()
                except Exception:
                    encoded = None
                if encoded:
                    downloaded[candidate["url"]] = {**candidate, "base64": encoded}
            # 恢复页面顺序，保证行为可预测。
            collected.extend(
                downloaded[candidate["url"]]
                for candidate in download_candidates
                if candidate["url"] in downloaded
            )

    filtered: list[dict[str, str]] = []
    started = time.monotonic()
    seen_images: set[str] = set()
    for candidate in collected:
        if time.monotonic() - started > LOCAL_IMAGE_FILTER_BUDGET:
            break
        prepared = _prepare_vision_image(candidate["base64"])
        if prepared is None:
            continue
        original, data_url = prepared
        digest = hashlib.sha256(original.encode("ascii")).hexdigest()
        if digest in seen_images:
            continue
        seen_images.add(digest)
        filtered.append({**candidate, "base64": original, "data_url": data_url})
    return filtered[:max_candidates]


def _collect_result_images(item: dict[str, Any], timeout: float) -> list[str]:
    images = _embedded_images(item)
    references = _image_references_from_item(item)
    source_url = _source_url(item)
    base_url = source_url or "https://metaso.cn/"
    if source_url and _is_public_http_url(source_url):
        try:
            source_data, content_type, final_url, charset = _read_url(
                source_url, timeout=timeout, max_bytes=MAX_SOURCE_BYTES
            )
            if content_type == "application/pdf" or source_data.startswith(b"%PDF-"):
                images.extend(_images_from_pdf(source_data))
            elif content_type.startswith("image/"):
                images.append(base64.b64encode(source_data).decode("ascii"))
            else:
                references.extend(
                    _image_references_from_html(source_data, final_url, charset)
                )
        except (HTTPError, URLError, OSError, ValueError):
            pass
    for reference in list(dict.fromkeys(references)):
        if encoded := _download_image(reference, base_url, timeout):
            images.append(encoded)
    unique: list[str] = []
    seen: set[str] = set()
    for encoded in images:
        digest = hashlib.sha256(encoded.encode("ascii", errors="ignore")).hexdigest()
        if digest not in seen:
            seen.add(digest)
            unique.append(encoded)
    return unique


def populate_image_base64(results: list[Metadata], source_items: list[dict[str, Any]], *, timeout: float = 15.0) -> list[Metadata]:
    """抓取每条结果的候选图片；Qwen 筛选在下一阶段执行。"""

    if len(results) != len(source_items):
        raise ValueError("Metadata 与秘塔源结果数量不一致")
    if not results:
        return results
    with ThreadPoolExecutor(max_workers=min(6, len(results))) as executor:
        futures = {executor.submit(_collect_result_images, item, timeout): index for index, item in enumerate(source_items)}
        for future in as_completed(futures):
            index = futures[future]
            try:
                results[index].image_base_64 = future.result()
            except Exception:
                results[index].image_base_64 = []
            results[index].image_summary = []
    return results


def _prepare_vision_image(encoded: str) -> tuple[str, str] | None:
    """验证真实像素尺寸，过滤追踪图，并返回适合视觉 API 的 data URL。"""

    normalized = _normalize_base64(encoded)
    if not normalized:
        return None
    data = base64.b64decode(normalized)
    if Image is None:
        return None
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.seek(0)
            width, height = image.size
            if width < MIN_IMAGE_SIDE or height < MIN_IMAGE_SIDE:
                return None
            if width * height < MIN_IMAGE_AREA:
                return None
            if max(width / height, height / width) > MAX_IMAGE_ASPECT_RATIO:
                return None
            # 强制读取像素，提前排除文件头正常但正文损坏的图片。
            image.load()
            signatures = (
                (b"\xff\xd8\xff", "image/jpeg"),
                (b"\x89PNG\r\n\x1a\n", "image/png"),
                (b"GIF8", "image/gif"),
                (b"RIFF", "image/webp"),
            )
            mime = next(
                (value for signature, value in signatures if data.startswith(signature)), None
            )
            if mime and len(data) <= 4 * 1024 * 1024:
                return normalized, f"data:{mime};base64,{normalized}"
            image.thumbnail((1600, 1600))
            if image.mode not in {"RGB", "L"}:
                image = image.convert("RGB")
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=85, optimize=True)
    except Exception:
        # 任何单张图片的解码/像素异常都只淘汰该图片，不能中断整条结果。
        return None
    prepared = base64.b64encode(buffer.getvalue()).decode("ascii")
    return normalized, f"data:image/jpeg;base64,{prepared}"


def _qwen_output_text(payload: dict[str, Any]) -> str:
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and isinstance(block.get("text"), str)
        ).strip()
    return ""


def _parse_json_object(text: str) -> dict[str, Any]:
    candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(candidate[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("Qwen 应返回 JSON 对象")
    return value


T = TypeVar("T")


def _qwen_json_request(
    content: list[dict[str, Any]],
    *,
    api_key: str,
    model: str,
    timeout: float,
    validator: Callable[[dict[str, Any]], T],
    retries: int = 3,
    api_url: str | None = None,
) -> T:
    endpoint = api_url or os.getenv("QWEN_API_URL", QWEN_API_URL)
    body = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 1400,
        "stream": False,
        "enable_thinking": False,
    }
    last_error: Exception | None = None
    for attempt in range(retries):
        request = Request(endpoint, data=json.dumps(body, ensure_ascii=False).encode("utf-8"), method="POST", headers={"Accept": "application/json", "Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
        try:
            with qwen_urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            text = _qwen_output_text(payload)
            if not text:
                raise QwenImageAPIError("Qwen 返回了空内容")
            return validator(_parse_json_object(text))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            last_error = QwenImageAPIError(
                f"Qwen API 请求失败（HTTP {exc.code}）：{detail}",
                status_code=exc.code,
                detail=detail,
            )
            if exc.code not in {408, 409, 429, 500, 502, 503, 504}:
                raise last_error from exc
        except (URLError, OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError, QwenImageAPIError) as exc:
            last_error = exc
        if attempt + 1 < retries:
            time.sleep(0.5 * (2**attempt))
    raise QwenImageAPIError(
        f"重试 {retries} 次后仍失败：{last_error}",
        status_code=getattr(last_error, "status_code", None),
        detail=getattr(last_error, "detail", ""),
    ) from last_error


def _select_relevant_images(
    question: str,
    item: Metadata,
    images: list[tuple[str, str]],
    *,
    api_key: str,
    model: str,
    timeout: float,
    api_url: str | None,
) -> list[tuple[int, str, str]]:
    scores: dict[int, float] = {}
    context = f"用户问题：{question}\n搜索结果标题：{item.title or ''}\n文字摘要：{(item.text_summary or item.text or '')[:2000]}"
    for start in range(0, len(images), IMAGE_SELECTION_BATCH_SIZE):
        batch = list(enumerate(images[start : start + IMAGE_SELECTION_BATCH_SIZE], start=start))
        indices = [index for index, _ in batch]
        content: list[dict[str, Any]] = [{"type": "text", "text": f"{context}\n请评估下面每张图与用户问题的相关性，0=无关，100=直接回答问题。必须覆盖编号 {indices}，只返回 JSON：{{\"scores\":[{{\"index\":编号,\"score\":0到100}}]}}。"}]
        for index, (_original, data_url) in batch:
            content.extend((
                {"type": "text", "text": f"候选图片编号 {index}"},
                {"type": "image_url", "image_url": {"url": data_url}},
            ))

        def validate(value: dict[str, Any], expected: set[int] = set(indices)) -> dict[int, float]:
            parsed: dict[int, float] = {}
            for entry in value.get("scores", []):
                if isinstance(entry, dict) and isinstance(entry.get("index"), int) and entry["index"] in expected and isinstance(entry.get("score"), (int, float)):
                    parsed[entry["index"]] = max(0.0, min(100.0, float(entry["score"])))
            if set(parsed) != expected:
                raise ValueError(f"Qwen 图片评分缺少编号：{sorted(expected - set(parsed))}")
            return parsed

        try:
            scores.update(
                _qwen_json_request(
                    content,
                    api_key=api_key,
                    model=model,
                    timeout=timeout,
                    validator=validate,
                    api_url=api_url,
                )
            )
        except QwenImageAPIError as exc:
            if not exc.is_invalid_image:
                raise
            # 批量请求被图片参数拒绝时逐张定位，仅跳过不合格图片。
            for index, (_original, data_url) in batch:
                single_content = [
                    {
                        "type": "text",
                        "text": (
                            f"{context}\n请评估图片编号 {index} 与问题的相关性，"
                            "只返回 JSON："
                            f'{{"scores":[{{"index":{index},"score":0到100}}]}}。'
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": data_url}},
                ]

                def validate_single(
                    value: dict[str, Any], expected_index: int = index
                ) -> dict[int, float]:
                    for entry in value.get("scores", []):
                        if (
                            isinstance(entry, dict)
                            and entry.get("index") == expected_index
                            and isinstance(entry.get("score"), (int, float))
                        ):
                            return {
                                expected_index: max(
                                    0.0, min(100.0, float(entry["score"]))
                                )
                            }
                    raise ValueError(f"Qwen 图片评分缺少编号：{expected_index}")

                try:
                    scores.update(
                        _qwen_json_request(
                            single_content,
                            api_key=api_key,
                            model=model,
                            timeout=timeout,
                            validator=validate_single,
                            api_url=api_url,
                        )
                    )
                except QwenImageAPIError as single_exc:
                    if single_exc.is_invalid_image:
                        continue
                    raise
    ranked = sorted(scores, key=lambda index: (-scores[index], index))[:MAX_RELEVANT_IMAGES]
    return [(index, images[index][0], images[index][1]) for index in ranked]


def _deepseek_preselect_image_candidates(
    question: str,
    results: list[Metadata],
    candidates_by_result: list[list[dict[str, str]]],
    *,
    api_key: str | None,
    model: str,
    timeout: float,
    api_url: str | None,
) -> list[list[dict[str, str]]]:
    """一次 DeepSeek 请求为全部结果各预选最多3张候选图。"""

    if not any(candidates_by_result):
        return candidates_by_result
    token = api_key or os.getenv("DEEPSEEK_API_KEY")
    if not token:
        return [candidates[:DEEPSEEK_PRESELECT_IMAGES] for candidates in candidates_by_result]
    compact: list[dict[str, Any]] = []
    for result_index, (item, candidates) in enumerate(
        zip(results, candidates_by_result)
    ):
        if not candidates:
            continue
        compact.append(
            {
                "result_index": result_index,
                "title": item.title,
                "text_summary": (item.text_summary or item.text or "")[:800],
                "images": [
                    {
                        "candidate_index": index,
                        "url": candidate.get("url", "")[:500],
                        "alt": candidate.get("alt", "")[:200],
                        "title": candidate.get("title", "")[:200],
                        "caption": candidate.get("caption", "")[:300],
                        "context": candidate.get("context", "")[:300],
                    }
                    for index, candidate in enumerate(candidates)
                ],
            }
        )
    try:
        content = _deepseek_request(
            [
                {
                    "role": "system",
                    "content": (
                        "你负责根据用户问题和图片周边文字预筛图片。排除网站Logo、图标、"
                        "平台品牌、广告、导航图片、二维码及与问题无直接内容关系的图片。"
                        "每条结果最多选择3张；没有相关候选时返回空数组。只返回JSON："
                        '{"selections":[{"result_index":0,"candidate_indices":[0,1]}]}'
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {"question": question, "results": compact}, ensure_ascii=False
                    ),
                },
            ],
            api_key=token,
            model=model,
            timeout=timeout,
            json_mode=True,
            retries=1,
            api_url=api_url,
        )
        payload = _parse_json_object(content)
        selections: dict[int, list[int]] = {}
        for entry in payload.get("selections", []):
            if not isinstance(entry, dict) or not isinstance(entry.get("result_index"), int):
                continue
            indices = entry.get("candidate_indices")
            if isinstance(indices, list):
                selections[entry["result_index"]] = [
                    value for value in indices if isinstance(value, int)
                ][:DEEPSEEK_PRESELECT_IMAGES]
        return [
            [candidates[index] for index in selections.get(result_index, []) if 0 <= index < len(candidates)]
            for result_index, candidates in enumerate(candidates_by_result)
        ]
    except (DeepSeekAPIError, ValueError, json.JSONDecodeError):
        # 预筛是优化层；失败时降级到前3张，不能阻塞最终文本结果。
        return [candidates[:DEEPSEEK_PRESELECT_IMAGES] for candidates in candidates_by_result]


def _qwen_choose_and_summarize_one(
    question: str,
    item: Metadata,
    candidates: list[dict[str, str]],
    *,
    api_key: str,
    model: str,
    timeout: float,
    api_url: str | None,
) -> tuple[list[str], list[str]]:
    """单次Qwen请求完成相关性判断、选1张和摘要；低于阈值返回空列表。"""

    if not candidates:
        return [], []
    indices = list(range(len(candidates)))
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                f"用户问题：{question}\n资料标题：{item.title or ''}\n"
                f"资料摘要：{(item.text_summary or item.text or '')[:1200]}\n"
                "请查看候选图片，选择与用户问题核心内容最相关的一张并总结。"
                "网站Logo、平台品牌、图标、广告和仅表示资料来源的图片必须视为无关。"
                f"只有相关性评分达到{int(IMAGE_RELEVANCE_THRESHOLD)}才可选择；"
                "否则selected_index必须为null且summary为空字符串。只返回JSON："
                '{"selected_index":0或null,"relevance_score":0到100,"summary":"中文说明"}'
            ),
        }
    ]
    for index, candidate in enumerate(candidates):
        context = " | ".join(
            value
            for value in (
                candidate.get("alt", ""),
                candidate.get("title", ""),
                candidate.get("caption", ""),
                candidate.get("context", ""),
            )
            if value
        )
        content.extend(
            (
                {"type": "text", "text": f"候选图片编号 {index}；上下文：{context[:500]}"},
                {"type": "image_url", "image_url": {"url": candidate["data_url"]}},
            )
        )

    def validate(value: dict[str, Any]) -> tuple[int | None, float, str]:
        selected_index = value.get("selected_index")
        score = value.get("relevance_score")
        summary = value.get("summary")
        if selected_index is None:
            return None, float(score) if isinstance(score, (int, float)) else 0.0, ""
        if selected_index not in indices or not isinstance(score, (int, float)):
            raise ValueError("Qwen 返回了无效的图片选择")
        return selected_index, float(score), summary.strip() if isinstance(summary, str) else ""

    try:
        selected_index, score, summary = _qwen_json_request(
            content,
            api_key=api_key,
            model=model,
            timeout=timeout,
            validator=validate,
            api_url=api_url,
            retries=1,
        )
    except QwenImageAPIError:
        return [], []
    if (
        selected_index is None
        or score < IMAGE_RELEVANCE_THRESHOLD
        or not summary
    ):
        return [], []
    return [candidates[selected_index]["base64"]], [summary]


def _summarize_selected_images(
    question: str,
    item: Metadata,
    selected: list[tuple[int, str, str]],
    *,
    api_key: str,
    model: str,
    timeout: float,
    api_url: str | None,
) -> list[str]:
    if not selected:
        return []
    indices = [index for index, _original, _url in selected]
    content: list[dict[str, Any]] = [{"type": "text", "text": f"用户问题：{question}\n资料标题：{item.title or ''}\n请逐张说明图片中可见的主要信息及其与问题的关系。不要臆测看不清的数据或给出超出图片的诊断。必须覆盖编号 {indices}，每张使用简洁中文，只返回 JSON：{{\"summaries\":[{{\"index\":编号,\"summary\":\"内容\"}}]}}。"}]
    for index, _original, data_url in selected:
        content.extend((
            {"type": "text", "text": f"待总结图片编号 {index}"},
            {"type": "image_url", "image_url": {"url": data_url}},
        ))

    def validate(value: dict[str, Any]) -> list[str]:
        parsed: dict[int, str] = {}
        for entry in value.get("summaries", []):
            if isinstance(entry, dict) and isinstance(entry.get("index"), int) and isinstance(entry.get("summary"), str) and entry["summary"].strip():
                parsed[entry["index"]] = entry["summary"].strip()
        missing = [index for index in indices if index not in parsed]
        if missing:
            raise ValueError(f"Qwen 图片摘要缺少编号：{missing}")
        return [parsed[index] for index in indices]

    return _qwen_json_request(content, api_key=api_key, model=model, timeout=timeout, validator=validate, api_url=api_url)


def _summarize_images_resiliently(
    question: str,
    item: Metadata,
    selected: list[tuple[int, str, str]],
    *,
    api_key: str,
    model: str,
    timeout: float,
    api_url: str | None,
) -> tuple[list[tuple[int, str, str]], list[str]]:
    """总结图片；若批量请求因某张坏图失败，则逐张总结并只跳过坏图。"""

    if not selected:
        return [], []
    try:
        return selected, _summarize_selected_images(
            question,
            item,
            selected,
            api_key=api_key,
            model=model,
            timeout=timeout,
            api_url=api_url,
        )
    except QwenImageAPIError as exc:
        if not exc.is_invalid_image:
            raise
    valid_selected: list[tuple[int, str, str]] = []
    summaries: list[str] = []
    for candidate in selected:
        try:
            summary = _summarize_selected_images(
                question,
                item,
                [candidate],
                api_key=api_key,
                model=model,
                timeout=timeout,
                api_url=api_url,
            )
        except QwenImageAPIError as exc:
            if exc.is_invalid_image:
                continue
            raise
        if summary:
            valid_selected.append(candidate)
            summaries.append(summary[0])
    return valid_selected, summaries


def curate_metadata_images(
    question: str,
    results: list[Metadata],
    *,
    api_key: str | None = None,
    model: str = QWEN_IMAGE_MODEL,
    timeout: float = 60.0,
    api_url: str | None = None,
) -> list[Metadata]:
    """用 Qwen 按问题筛选每条结果最相关的 1 张图，并生成对应摘要。"""

    token = api_key or os.getenv("DASHSCOPE_API_KEY")
    for result_index, item in enumerate(results, start=1):
        prepared = [value for encoded in item.image_base_64 if (value := _prepare_vision_image(encoded)) is not None]
        if not prepared:
            item.image_base_64 = []
            item.image_summary = []
            continue
        if not token:
            raise ValueError("发现候选图片，但未找到 DASHSCOPE_API_KEY；无法使用 qwen3.7-plus 筛图并生成 image_summary")
        try:
            selected = _select_relevant_images(question, item, prepared, api_key=token, model=model, timeout=timeout, api_url=api_url)
            selected, summaries = _summarize_images_resiliently(
                question,
                item,
                selected,
                api_key=token,
                model=model,
                timeout=timeout,
                api_url=api_url,
            )
        except QwenImageAPIError as exc:
            raise QwenImageAPIError(f"第 {result_index} 条结果图片处理失败（{item.title or '无标题'}）：{exc}") from exc
        item.image_base_64 = [original for _index, original, _data_url in selected]
        item.image_summary = summaries
    return results


async def curate_metadata_images_async(
    question: str,
    results: list[Metadata],
    *,
    api_key: str | None = None,
    model: str = QWEN_IMAGE_MODEL,
    timeout: float = 60.0,
    api_url: str | None = None,
    max_concurrency: int | None = None,
) -> list[Metadata]:
    """并行处理不同检索结果的图片，并限制 Qwen 在途请求量。"""

    token = api_key or os.getenv("DASHSCOPE_API_KEY")
    concurrency = max_concurrency or _configured_concurrency(
        "QWEN_MAX_CONCURRENCY", DEFAULT_QWEN_CONCURRENCY
    )
    if concurrency < 1:
        raise ValueError("Qwen 最大并发量必须是正整数")
    async def curate_one(job: tuple[int, Metadata]) -> None:
        result_index, item = job
        prepared = [
            value
            for encoded in item.image_base_64
            if (value := _prepare_vision_image(encoded)) is not None
        ]
        if not prepared:
            item.image_base_64 = []
            item.image_summary = []
            return
        if not token:
            raise ValueError(
                "发现候选图片，但未找到 DASHSCOPE_API_KEY；无法使用 "
                "qwen3.7-plus 筛图并生成 image_summary"
            )
        try:
            selected = await asyncio.to_thread(
                _select_relevant_images,
                question,
                item,
                prepared,
                api_key=token,
                model=model,
                timeout=timeout,
                api_url=api_url,
            )
            selected, summaries = await asyncio.to_thread(
                _summarize_images_resiliently,
                question,
                item,
                selected,
                api_key=token,
                model=model,
                timeout=timeout,
                api_url=api_url,
            )
        except QwenImageAPIError as exc:
            raise QwenImageAPIError(
                f"第 {result_index} 条结果图片处理失败（{item.title or '无标题'}）：{exc}"
            ) from exc
        item.image_base_64 = [original for _index, original, _data_url in selected]
        item.image_summary = summaries

    failures = await _run_queue_workers(
        list(enumerate(results, start=1)), curate_one, concurrency
    )
    if failures:
        raise failures[0]
    return results


async def fetch_and_curate_images_pipeline(
    question: str,
    results: list[Metadata],
    source_items: list[dict[str, Any]],
    *,
    api_key: str | None,
    model: str,
    image_timeout: float,
    vision_timeout: float,
    api_url: str | None,
    qwen_max_concurrency: int | None,
    deepseek_api_key: str | None,
    deepseek_model: str,
    deepseek_api_url: str | None,
) -> list[Metadata]:
    """候选抓取→一次DeepSeek预筛→Qwen选图总结；图片阶段最多55秒。"""

    if len(results) != len(source_items):
        raise ValueError("Metadata 与秘塔源结果数量不一致")
    for item in results:
        # 图片属于可降级字段：只有选图和摘要均成功时才同时写回。
        item.image_base_64 = []
        item.image_summary = []
    token = api_key or os.getenv("DASHSCOPE_API_KEY")
    qwen_concurrency = qwen_max_concurrency or _configured_concurrency(
        "QWEN_MAX_CONCURRENCY", DEFAULT_QWEN_CONCURRENCY
    )
    if qwen_concurrency < 1:
        raise ValueError("Qwen 最大并发量必须是正整数")

    started_at = time.monotonic()
    start_cutoff = started_at + IMAGE_START_CUTOFF_SECONDS
    hard_deadline = started_at + IMAGE_HARD_DEADLINE_SECONDS
    fetch_queue: asyncio.Queue[tuple[int, dict[str, Any]]] = asyncio.Queue()
    candidates_by_result: list[list[dict[str, str]]] = [
        [] for _ in results
    ]
    for index, (item, source) in enumerate(zip(results, source_items), start=1):
        del item
        fetch_queue.put_nowait((index - 1, source))

    async def fetch_worker() -> None:
        while True:
            if time.monotonic() >= start_cutoff:
                return
            try:
                index, source = fetch_queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                try:
                    candidates_by_result[index] = await asyncio.to_thread(
                        _collect_result_image_candidates,
                        source,
                        min(image_timeout, IMAGE_DOWNLOAD_TIMEOUT),
                        MAX_IMAGE_CANDIDATES,
                    )
                except Exception:
                    candidates_by_result[index] = []
            finally:
                fetch_queue.task_done()
    fetch_workers = [
        asyncio.create_task(fetch_worker())
        for _ in range(min(DEFAULT_IMAGE_FETCH_CONCURRENCY, len(results)))
    ]
    remaining = max(0.1, start_cutoff - time.monotonic())
    try:
        await asyncio.wait_for(fetch_queue.join(), timeout=remaining)
    except asyncio.TimeoutError:
        pass
    for worker in fetch_workers:
        worker.cancel()
    await asyncio.gather(*fetch_workers, return_exceptions=True)

    if time.monotonic() >= start_cutoff:
        return results
    preselect_timeout = max(1.0, min(8.0, start_cutoff - time.monotonic()))
    preselected = await asyncio.to_thread(
        _deepseek_preselect_image_candidates,
        question,
        results,
        candidates_by_result,
        api_key=deepseek_api_key,
        model=deepseek_model,
        timeout=preselect_timeout,
        api_url=deepseek_api_url,
    )
    if not token or time.monotonic() >= start_cutoff:
        return results

    qwen_queue: asyncio.Queue[tuple[int, Metadata, list[dict[str, str]]]] = asyncio.Queue()
    for index, (item, candidates) in enumerate(zip(results, preselected)):
        if candidates:
            qwen_queue.put_nowait((index, item, candidates))

    async def qwen_worker() -> None:
        while True:
            if time.monotonic() >= start_cutoff:
                return
            try:
                _index, item, candidates = qwen_queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                remaining_for_request = max(1.0, hard_deadline - time.monotonic())
                images, summaries = await asyncio.to_thread(
                    _qwen_choose_and_summarize_one,
                    question,
                    item,
                    candidates,
                    api_key=token,
                    model=model,
                    timeout=min(vision_timeout, remaining_for_request),
                    api_url=api_url,
                )
                item.image_base_64 = images
                item.image_summary = summaries
            except Exception:
                item.image_base_64 = []
                item.image_summary = []
            finally:
                qwen_queue.task_done()

    qwen_workers = [
        asyncio.create_task(qwen_worker())
        for _ in range(min(qwen_concurrency, max(1, qwen_queue.qsize())))
    ]
    remaining = max(0.1, hard_deadline - time.monotonic())
    try:
        await asyncio.wait_for(qwen_queue.join(), timeout=remaining)
    except asyncio.TimeoutError:
        pass
    for worker in qwen_workers:
        worker.cancel()
    await asyncio.gather(*qwen_workers, return_exceptions=True)
    # 任何未完成任务都保持空图片列表，文本结果照常返回。
    return results


async def enrich_metadata_async(
    question: str,
    results: list[Metadata],
    source_items: list[dict[str, Any]],
    *,
    generate_text_summary: bool,
    fetch_images: bool,
    generate_image_summary: bool,
    deepseek_api_key: str | None,
    deepseek_summary_model: str,
    qwen_api_key: str | None,
    qwen_image_model: str,
    timeout: float,
    image_timeout: float,
    vision_timeout: float,
    deepseek_api_url: str | None,
    qwen_api_url: str | None,
    deepseek_max_concurrency: int | None,
    qwen_max_concurrency: int | None,
) -> list[Metadata]:
    """重叠执行 DeepSeek 摘要与图片抓取，随后并行进行 Qwen 图片处理。"""

    initial_tasks: list[Any] = []
    if generate_text_summary:
        initial_tasks.append(
            summarize_metadata_async(
                question,
                results,
                api_key=deepseek_api_key,
                model=deepseek_summary_model,
                timeout=timeout,
                api_url=deepseek_api_url,
                max_concurrency=deepseek_max_concurrency,
            )
        )
    if fetch_images:
        if generate_image_summary:
            initial_tasks.append(
                fetch_and_curate_images_pipeline(
                    question,
                    results,
                    source_items,
                    api_key=qwen_api_key,
                    model=qwen_image_model,
                    image_timeout=image_timeout,
                    vision_timeout=vision_timeout,
                    api_url=qwen_api_url,
                    qwen_max_concurrency=qwen_max_concurrency,
                    deepseek_api_key=deepseek_api_key,
                    deepseek_model=deepseek_summary_model,
                    deepseek_api_url=deepseek_api_url,
                )
            )
        else:
            initial_tasks.append(
                asyncio.to_thread(
                    populate_image_base64,
                    results,
                    source_items,
                    timeout=image_timeout,
                )
            )
    if initial_tasks:
        await asyncio.gather(*initial_tasks)

    if not fetch_images:
        for item in results:
            item.image_base_64 = []
            item.image_summary = []
    return results


def search_metaso(
    question: str,
    *,
    api_key: str | None = None,
    scope: str | None = None,
    size: int = 20,
    query_type: str = "auto",
    deepseek_api_key: str | None = None,
    deepseek_model: str = DEEPSEEK_MODEL,
    deepseek_summary_model: str = DEEPSEEK_SUMMARY_MODEL,
    generate_text_summary: bool = True,
    fetch_images: bool = True,
    generate_image_summary: bool = True,
    qwen_api_key: str | None = None,
    qwen_image_model: str = QWEN_IMAGE_MODEL,
    include_summary: bool = True,
    include_raw_content: bool = False,
    concise_snippet: bool = False,
    timeout: float = 30.0,
    image_timeout: float = 15.0,
    vision_timeout: float = 60.0,
    api_url: str = METASO_SEARCH_URL,
    deepseek_api_url: str | None = None,
    qwen_api_url: str | None = None,
    deepseek_max_concurrency: int | None = None,
    qwen_max_concurrency: int | None = None,
) -> list[Metadata]:
    """执行分类检索，最终返回不超过 ``size`` 条 ``Metadata``。"""

    question = question.strip()
    if not question:
        raise ValueError("question 不能为空")
    if scope is not None and scope not in VALID_SCOPES:
        raise ValueError(f"scope 必须是以下值之一：{', '.join(sorted(VALID_SCOPES))}")
    if query_type not in _QUERY_TYPES:
        raise ValueError(f"query_type 必须是以下值之一：{', '.join(sorted(_QUERY_TYPES))}")
    if not 1 <= size <= 20:
        raise ValueError("size 必须在 1 到 20 之间")
    token = api_key or os.getenv("METASO_API_KEY")
    if not token:
        raise ValueError("未找到 METASO_API_KEY")

    if scope is not None:
        candidates = _request_scope(question, token=token, scope=scope, size=size, include_summary=include_summary, include_raw_content=include_raw_content, concise_snippet=concise_snippet, timeout=timeout, api_url=api_url)
        selected = candidates[:size]
    else:
        resolved_type = classify_question(question, api_key=deepseek_api_key, model=deepseek_model, timeout=timeout, api_url=deepseek_api_url) if query_type == "auto" else query_type
        scopes, quotas = _scope_quotas(resolved_type, size)
        candidates: list[tuple[str, int, dict[str, Any]]] = []
        scope_results: dict[str, list[tuple[str, int, dict[str, Any]]]] = {}
        errors: list[str] = []
        with ThreadPoolExecutor(max_workers=len(scopes)) as executor:
            futures = {
                executor.submit(
                    _request_scope,
                    question,
                    token=token,
                    scope=item_scope,
                    size=quotas[item_scope],
                    include_summary=include_summary,
                    include_raw_content=include_raw_content,
                    concise_snippet=concise_snippet,
                    timeout=timeout,
                    api_url=api_url,
                ): item_scope
                for item_scope in scopes
            }
            for future in as_completed(futures):
                item_scope = futures[future]
                try:
                    scope_results[item_scope] = future.result()
                except MetasoAPIError as exc:
                    errors.append(f"{item_scope}: {exc}")
        successful_scopes = [item_scope for item_scope in scopes if item_scope in scope_results]
        for item_scope in successful_scopes:
            candidates.extend(scope_results[item_scope])
        if not successful_scopes:
            raise MetasoAPIError("所有检索范围都失败：" + "; ".join(errors))
        selected = _select_candidates(question, candidates, scopes, quotas, size)
        if len(selected) < size:
            # 某范围不足或失败时，从成功范围扩大请求并重新去重、排序补齐。
            for item_scope in successful_scopes:
                try:
                    expanded = _request_scope(question, token=token, scope=item_scope, size=size, include_summary=include_summary, include_raw_content=include_raw_content, concise_snippet=concise_snippet, timeout=timeout, api_url=api_url)
                    existing = {(scope_name, rank, _canonical_url(item) or _normalized_title(item)) for scope_name, rank, item in candidates}
                    candidates.extend(candidate for candidate in expanded if (candidate[0], candidate[1], _canonical_url(candidate[2]) or _normalized_title(candidate[2])) not in existing)
                except MetasoAPIError:
                    continue
                selected = _select_candidates(question, candidates, scopes, quotas, size)
                if len(selected) >= size:
                    break

    results = [_to_metadata(item_scope, item) for item_scope, _rank, item in selected]
    source_items = [item for _scope, _rank, item in selected]
    return _run_async(
        enrich_metadata_async(
            question,
            results,
            source_items,
            generate_text_summary=generate_text_summary,
            fetch_images=fetch_images,
            generate_image_summary=generate_image_summary,
            deepseek_api_key=deepseek_api_key,
            deepseek_summary_model=deepseek_summary_model,
            qwen_api_key=qwen_api_key,
            qwen_image_model=qwen_image_model,
            timeout=timeout,
            image_timeout=image_timeout,
            vision_timeout=vision_timeout,
            deepseek_api_url=deepseek_api_url,
            qwen_api_url=qwen_api_url,
            deepseek_max_concurrency=deepseek_max_concurrency,
            qwen_max_concurrency=qwen_max_concurrency,
        )
    )


def _main() -> int:
    if len(sys.argv) < 2:
        print(f'用法：python {os.path.basename(__file__)} "要搜索的问题"', file=sys.stderr)
        return 2
    try:
        results = search_metaso(" ".join(sys.argv[1:]))
    except (ValueError, MetasoAPIError, DeepSeekAPIError, QwenImageAPIError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1
    print(json.dumps([item.to_dict() for item in results], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
