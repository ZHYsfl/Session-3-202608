"""医疗 RAG 项目的 Listwise 滑动窗口重排序模块。

本模块负责对上一级粗排产生的 ``list[Metadata]`` 执行两轮串行
Listwise 重排序，并返回排序后的前 ``top_k2`` 条 Metadata。

核心排序流程保持大模型供应商无关；DeepSeek API 适配器通过统一接口接入。
本模块只调整列表顺序，不修改 Metadata 的任何字段。
"""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Protocol, cast


WINDOW_SIZE = 10
RERANK_ROUNDS = 2


@dataclass
class Metadata:
    """表示一条供 Listwise 重排使用的完整文献元数据。"""

    title: str | None
    year: str | None
    source_type: str | None
    id: str | None
    text_summary: str | None
    text: str | None
    image_summary: list[str] | None
    image_base_64: list[str] | None
    evidence_level: str | None
    last_retrieved_at: str | None


@dataclass
class RankCandidate:
    """表示当前窗口中允许发送给排序模型的候选数据。"""

    id: str
    original_rank: int
    title: str | None
    year: str | None
    source_type: str | None
    text_summary: str | None
    text: str | None
    image_summary: list[str] | None


class ListwiseError(Exception):
    """Listwise 重排模块的基础异常。"""


class InputValidationError(ListwiseError):
    """输入参数或 Metadata 数据不符合契约。"""


class RankerOutputError(ListwiseError):
    """排序器返回的 ID 结果不是当前窗口的完整排列。"""


class WindowRankingError(ListwiseError):
    """当前轮次的窗口排序调用失败。"""


class WindowRanker(Protocol):
    """定义单个候选窗口的同步排序接口。"""

    def rank_ids(
        self,
        query: str,
        candidates: Sequence[RankCandidate],
    ) -> list[str]:
        """按与问题的相关性从高到低返回当前窗口的全部 ID。"""
        ...


def _validate_image_fields(metadata: Metadata) -> None:
    """验证图片总结与 Base64 列表的状态和对齐关系。"""
    image_summary = metadata.image_summary
    image_base_64 = metadata.image_base_64
    document_id = metadata.id

    if image_summary is None and image_base_64 is None:
        return

    if image_summary is None or image_base_64 is None:
        raise InputValidationError(
            f"文献 ID {document_id!r} 的 image_summary 和 image_base_64 "
            "必须同时为 None 或同时为列表"
        )

    if not isinstance(image_summary, list) or not isinstance(image_base_64, list):
        raise InputValidationError(
            f"文献 ID {document_id!r} 的 image_summary 和 image_base_64 "
            "必须是列表"
        )

    if not image_summary or not image_base_64:
        raise InputValidationError(
            f"文献 ID {document_id!r} 的图片字段不允许使用空列表"
        )

    if len(image_summary) != len(image_base_64):
        raise InputValidationError(
            f"文献 ID {document_id!r} 的 image_summary 和 image_base_64 "
            "列表长度必须相同"
        )

    if not all(isinstance(summary, str) for summary in image_summary):
        raise InputValidationError(
            f"文献 ID {document_id!r} 的 image_summary 必须是 list[str]"
        )

    if not all(isinstance(encoded_image, str) for encoded_image in image_base_64):
        raise InputValidationError(
            f"文献 ID {document_id!r} 的 image_base_64 必须是 list[str]"
        )


def _validate_inputs(
    query: object,
    metadata_list: object,
    top_k2: object,
) -> None:
    """在排序器调用前验证 Listwise 的全部输入契约。"""
    if not isinstance(query, str) or not query.strip():
        raise InputValidationError("query 必须是非空字符串")

    if not isinstance(metadata_list, list):
        raise InputValidationError("metadata_list 必须是 list[Metadata]")

    if len(metadata_list) < WINDOW_SIZE:
        raise InputValidationError(
            f"metadata_list 至少需要 {WINDOW_SIZE} 条 Metadata"
        )

    if isinstance(top_k2, bool) or not isinstance(top_k2, int) or top_k2 <= 0:
        raise InputValidationError("top_k2 必须是正整数")

    if top_k2 >= len(metadata_list):
        raise InputValidationError("top_k2 必须小于 metadata_list 的长度")

    seen_ids: set[str] = set()
    validated_metadata: list[Metadata] = []

    for index, metadata in enumerate(metadata_list, start=1):
        if not isinstance(metadata, Metadata):
            raise InputValidationError(
                f"metadata_list 中第 {index} 个元素必须是 Metadata"
            )

        if not isinstance(metadata.id, str) or not metadata.id.strip():
            raise InputValidationError(
                f"metadata_list 中第 {index} 条 Metadata 的 ID 必须是非空字符串"
            )

        if metadata.id in seen_ids:
            raise InputValidationError(f"文献 ID {metadata.id!r} 重复")

        seen_ids.add(metadata.id)
        validated_metadata.append(metadata)

    for metadata in validated_metadata:
        _validate_image_fields(metadata)


def _prepare_rerank_state(
    metadata_list: list[Metadata],
) -> tuple[list[Metadata], dict[str, int]]:
    """浅拷贝粗排列表，并保存从 1 开始的原始粗排位置。"""
    working_list = list(metadata_list)
    original_rank = {
        cast(str, metadata.id): index
        for index, metadata in enumerate(metadata_list, start=1)
    }
    return working_list, original_rank


def _build_rank_candidates(
    window: Sequence[Metadata],
    original_rank: dict[str, int],
) -> list[RankCandidate]:
    """为当前窗口构建允许发送给排序模型的临时载荷。"""
    candidates: list[RankCandidate] = []

    for metadata in window:
        document_id = cast(str, metadata.id)
        image_summary = (
            None
            if metadata.image_summary is None
            else list(metadata.image_summary)
        )
        candidates.append(
            RankCandidate(
                id=document_id,
                original_rank=original_rank[document_id],
                title=metadata.title,
                year=metadata.year,
                source_type=metadata.source_type,
                text_summary=metadata.text_summary,
                text=metadata.text,
                image_summary=image_summary,
            )
        )

    return candidates


def _window_context(
    round_number: int,
    window_start_index: int,
    window_length: int,
) -> str:
    """生成供异常消息使用的轮次和 1-based 窗口位置。"""
    window_end = window_start_index + window_length
    return (
        f"第 {round_number} 轮，"
        f"窗口位置 {window_start_index + 1}-{window_end}"
    )


def _validate_ranked_ids(
    ranked_ids: object,
    window_ids: Sequence[str],
    *,
    round_number: int,
    window_start_index: int,
) -> list[str]:
    """验证排序器返回的 ID 是当前窗口的完整排列。"""
    context = _window_context(
        round_number,
        window_start_index,
        len(window_ids),
    )

    if not isinstance(ranked_ids, list):
        raise RankerOutputError(
            f"{context}：排序器返回值必须是 list[str]，"
            f"实际类型为 {type(ranked_ids).__name__}"
        )

    if not all(isinstance(document_id, str) for document_id in ranked_ids):
        raise RankerOutputError(
            f"{context}：排序器返回的 ID 必须全部是字符串"
        )

    expected_ids = list(window_ids)
    if (
        len(ranked_ids) != len(expected_ids)
        or len(set(ranked_ids)) != len(expected_ids)
        or set(ranked_ids) != set(expected_ids)
    ):
        raise RankerOutputError(
            f"{context}：返回 ID 不是当前窗口的完整排列；"
            f"预期 ID={expected_ids!r}，实际 ID={ranked_ids!r}"
        )

    return ranked_ids


def _rank_window(
    query: str,
    window: Sequence[Metadata],
    original_rank: dict[str, int],
    ranker: WindowRanker,
    *,
    round_number: int,
    window_start_index: int,
) -> list[Metadata]:
    """调用排序器重排单个窗口，并返回原 Metadata 对象的新排列。"""
    candidates = _build_rank_candidates(window, original_rank)
    context = _window_context(
        round_number,
        window_start_index,
        len(window),
    )

    try:
        ranked_ids = ranker.rank_ids(query, candidates)
    except RankerOutputError as exc:
        raise RankerOutputError(f"{context}：{exc}") from exc
    except Exception as exc:
        raise WindowRankingError(
            f"{context}：排序器调用失败（{type(exc).__name__}）"
        ) from exc

    window_ids = [candidate.id for candidate in candidates]
    validated_ids = _validate_ranked_ids(
        ranked_ids,
        window_ids,
        round_number=round_number,
        window_start_index=window_start_index,
    )

    id_to_metadata = {
        cast(str, metadata.id): metadata
        for metadata in window
    }
    return [id_to_metadata[document_id] for document_id in validated_ids]


def _run_rerank_round(
    query: str,
    working_list: list[Metadata],
    original_rank: dict[str, int],
    ranker: WindowRanker,
    *,
    round_number: int,
) -> None:
    """对工作列表严格串行执行一轮滑动窗口重排。"""
    last_start_index = len(working_list) - WINDOW_SIZE

    for window_start_index in range(last_start_index + 1):
        window_end_index = window_start_index + WINDOW_SIZE
        window = working_list[window_start_index:window_end_index]
        ranked_window = _rank_window(
            query,
            window,
            original_rank,
            ranker,
            round_number=round_number,
            window_start_index=window_start_index,
        )
        working_list[window_start_index:window_end_index] = ranked_window


class ListwiseReranker:
    """使用统一窗口排序器执行两轮 Listwise 重排。"""

    def __init__(self, ranker: WindowRanker) -> None:
        self._ranker = ranker

    def rerank(
        self,
        query: str,
        metadata_list: list[Metadata],
        top_k2: int,
    ) -> list[Metadata]:
        """完整执行输入校验、两轮串行重排和 Top K 截取。"""
        _validate_inputs(query, metadata_list, top_k2)
        working_list, original_rank = _prepare_rerank_state(metadata_list)

        for round_number in range(1, RERANK_ROUNDS + 1):
            _run_rerank_round(
                query,
                working_list,
                original_rank,
                self._ranker,
                round_number=round_number,
            )

        return working_list[:top_k2]


class DeterministicMockRanker:
    """按预设的全局 ID 优先级稳定排序，用于无网络流程验证。"""

    def __init__(self, priority_ids: Sequence[str]) -> None:
        priority_list = list(priority_ids)
        if not priority_list:
            raise ValueError("priority_ids 不能为空")
        if not all(
            isinstance(document_id, str) and document_id.strip()
            for document_id in priority_list
        ):
            raise ValueError("priority_ids 必须全部是非空字符串")
        if len(set(priority_list)) != len(priority_list):
            raise ValueError("priority_ids 不允许包含重复 ID")

        self._priority = {
            document_id: index
            for index, document_id in enumerate(priority_list)
        }
        self.call_count = 0
        self.received_queries: list[str] = []
        self.received_windows: list[list[str]] = []
        self.received_candidates: list[list[RankCandidate]] = []

    def rank_ids(
        self,
        query: str,
        candidates: Sequence[RankCandidate],
    ) -> list[str]:
        """按全局优先级返回当前窗口的 ID 排列并记录调用。"""
        candidate_snapshots = [
            RankCandidate(
                id=candidate.id,
                original_rank=candidate.original_rank,
                title=candidate.title,
                year=candidate.year,
                source_type=candidate.source_type,
                text_summary=candidate.text_summary,
                text=candidate.text,
                image_summary=(
                    None
                    if candidate.image_summary is None
                    else list(candidate.image_summary)
                ),
            )
            for candidate in candidates
        ]
        window_ids = [candidate.id for candidate in candidate_snapshots]

        self.call_count += 1
        self.received_queries.append(query)
        self.received_windows.append(list(window_ids))
        self.received_candidates.append(candidate_snapshots)

        missing_ids = [
            document_id
            for document_id in window_ids
            if document_id not in self._priority
        ]
        if missing_ids:
            raise ValueError(
                f"priority_ids 缺少当前窗口 ID：{missing_ids!r}"
            )

        return sorted(window_ids, key=self._priority.__getitem__)


def _build_demo_metadata() -> list[Metadata]:
    """构造独立验证使用的 12 条完整 Metadata。"""
    metadata_list: list[Metadata] = []

    for index in range(1, 13):
        has_images = index % 2 == 0
        metadata_list.append(
            Metadata(
                title=f"测试医疗文献 {index}",
                year=str(2020 + index),
                source_type="journal",
                id=f"doc-{index}",
                text_summary=f"文献 {index} 的测试摘要",
                text=f"文献 {index} 的完整测试正文",
                image_summary=(
                    [f"文献 {index} 的图片总结"]
                    if has_images
                    else None
                ),
                image_base_64=(
                    [f"demo-base64-{index}"]
                    if has_images
                    else None
                ),
                evidence_level=f"source-evidence-{index}",
                last_retrieved_at=(
                    f"2026-08-12T10:{index:02d}:00+08:00"
                ),
            )
        )

    return metadata_list


def _run_standalone_validation() -> None:
    """无网络验证两轮滑动排序、载荷隔离和输入不变性。"""
    metadata_list = _build_demo_metadata()
    metadata_snapshot = deepcopy(metadata_list)
    input_ids = [metadata.id for metadata in metadata_list]
    input_by_id = {
        cast(str, metadata.id): metadata
        for metadata in metadata_list
    }
    priority_ids = [f"doc-{index}" for index in range(12, 0, -1)]
    mock_ranker = DeterministicMockRanker(priority_ids)

    top_k2 = 4
    result = ListwiseReranker(mock_ranker).rerank(
        query="测试医疗问题",
        metadata_list=metadata_list,
        top_k2=top_k2,
    )

    expected_call_count = RERANK_ROUNDS * (
        len(metadata_list) - WINDOW_SIZE + 1
    )
    assert mock_ranker.call_count == expected_call_count == 6
    assert len(result) == top_k2
    assert len({metadata.id for metadata in result}) == top_k2
    assert all(metadata.id in input_by_id for metadata in result)
    assert all(
        metadata is input_by_id[cast(str, metadata.id)]
        for metadata in result
    )
    assert [metadata.id for metadata in metadata_list] == input_ids
    assert metadata_list == metadata_snapshot

    for candidate_window in mock_ranker.received_candidates:
        assert len(candidate_window) == WINDOW_SIZE
        for candidate in candidate_window:
            source_metadata = input_by_id[candidate.id]
            assert candidate.text == source_metadata.text
            assert candidate.image_summary == source_metadata.image_summary
            if candidate.image_summary is not None:
                assert candidate.image_summary is not source_metadata.image_summary
            assert not hasattr(candidate, "image_base_64")
            assert not hasattr(candidate, "last_retrieved_at")
            assert not hasattr(candidate, "evidence_level")

    for current, before in zip(metadata_list, metadata_snapshot):
        assert current.evidence_level == before.evidence_level
        assert current.image_summary == before.image_summary
        assert current.image_base_64 == before.image_base_64

    invalid_metadata = list(metadata_list)
    invalid_metadata[0] = replace(
        invalid_metadata[0],
        image_summary=["非法图片总结"],
        image_base_64=None,
    )
    invalid_mock = DeterministicMockRanker(priority_ids)

    try:
        ListwiseReranker(invalid_mock).rerank(
            query="测试医疗问题",
            metadata_list=invalid_metadata,
            top_k2=top_k2,
        )
    except InputValidationError:
        pass
    else:
        raise AssertionError("非法图片字段未触发 InputValidationError")

    assert invalid_mock.call_count == 0

    print("Listwise 独立验证通过")
    print(f"合法案例：{len(metadata_list)} 条输入，{mock_ranker.call_count} 次窗口调用")
    print(f"Top {top_k2} ID：{[metadata.id for metadata in result]}")
    print("非法图片案例：已在首次排序器调用前拒绝")


if __name__ == "__main__":
    _run_standalone_validation()
