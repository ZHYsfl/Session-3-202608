"""
医学概念感知检索器 — 融合 retriever_015 的分字段评分策略
- 医学概念归一化 (medical_concepts)
- 分字段加权评分 (title/summary/text/image)
- 实体/方面/约束检测
- 证据等级 + 来源类型质量加权
- 通用词惩罚 (generic_penalty)
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from medical_concepts import (
    GENERIC_MATCH_TOKENS,
    aspect_doc_hit,
    constraint_doc_hit,
    detect_modality_terms,
    detect_query_aspects,
    detect_query_constraints,
    entity_in_doc,
    extract_query_entities,
    normalize_medical_text,
    specific_disease_concepts_in_query,
)
from retriever import Metadata
from tokenizer_v2 import content_tokens

_FIELD_WEIGHTS = {
    "title": 0.30,
    "summary": 0.30,
    "text": 0.20,
    "image": 0.20,
}

_EVIDENCE_WEIGHT = {
    "high": 1.0,
    "medium": 0.55,
    "low": 0.15,
}

_SOURCE_WEIGHT = {
    "meta_analysis": 1.0,
    "systematic_review": 1.0,
    "clinical_trial": 0.85,
    "randomized_controlled_trial": 0.85,
    "cohort_study": 0.70,
    "case_control_study": 0.70,
    "cross_sectional_study": 0.70,
    "guideline": 0.75,
    "review": 0.55,
    "paper": 0.55,
    "research_paper": 0.55,
    "webpage": 0.25,
    "news": 0.10,
    "blog": 0.10,
}

_UNKNOWN_SOURCE = 0.50

_EVIDENCE_QUERY_RE = re.compile(
    r"\b(evidence|effectiveness|efficacy|risk|effects?\b|associate|association|"
    r"meta[- ]?analysis|clinical trial|"
    r"does\s+\w+\s+(increase|cause|affect|reduce|prevent|improve))\b",
    re.IGNORECASE,
)


@dataclass
class ScoreBreakdown:
    index: int
    id: str | None
    title_score: float
    summary_score: float
    text_score: float
    image_score: float
    rel: float
    topic: float
    aspect: float
    joint: float
    constraint: float
    relevance: float
    evidence: float
    source_bonus: float
    quality: float
    final: float


def _field_texts(meta: Metadata) -> dict[str, str]:
    image_text = " ".join(s for s in (meta.image_summary or []) if s)
    return {
        "title": meta.title or "",
        "summary": meta.text_summary or "",
        "text": meta.text or "",
        "image": image_text,
    }


def _stem_variants(token: str) -> set[str]:
    variants = {token}
    if token.startswith("__med_") and token.endswith("__"):
        return variants
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        variants.add(token[:-1])
    else:
        variants.add(token + "s")
    if len(token) > 4 and token.endswith("ing"):
        variants.add(token[:-3])
        variants.add(token[:-3] + "e")
    if len(token) > 3 and token.endswith("ed"):
        variants.add(token[:-2])
        variants.add(token[:-1])
    return variants


def _token_match(query_token: str, field_set: set[str]) -> bool:
    return any(v in field_set for v in _stem_variants(query_token))


def _expand_field_set(field_tokens: list[str]) -> set[str]:
    expanded: set[str] = set()
    for t in field_tokens:
        expanded |= _stem_variants(t)
    return expanded


def _token_idf(docs: list[list[str]]) -> dict[str, float]:
    n = len(docs)
    df: dict[str, int] = {}
    for doc in docs:
        for t in set(doc):
            df[t] = df.get(t, 0) + 1
    return {t: math.log((n + 1) / (c + 1)) + 1.0 for t, c in df.items()}


def _weighted_coverage(
    query_tokens: list[str],
    field_tokens: list[str],
    idf: dict[str, float],
    *,
    generic_penalty: float = 1.0,
) -> float:
    if not query_tokens:
        return 0.0
    qset = list(dict.fromkeys(query_tokens))
    fset = _expand_field_set(field_tokens)
    num = sum(idf.get(t, 1.0) for t in qset if _token_match(t, fset))
    den = sum(idf.get(t, 1.0) for t in qset)
    raw = num / den if den else 0.0
    return raw * generic_penalty


def _bigram_cooccur(query_tokens: list[str], field_tokens: list[str]) -> float:
    if len(query_tokens) < 2:
        return 0.0
    fset = _expand_field_set(field_tokens)
    pairs = list(zip(query_tokens, query_tokens[1:]))
    if not pairs:
        return 0.0
    hits = sum(1 for a, b in pairs if _token_match(a, fset) and _token_match(b, fset))
    return hits / len(pairs)


def _normalize_bm25(raw: list[float]) -> list[float]:
    if not raw:
        return []
    mx = max(raw)
    mn = min(raw)
    if mx <= 0:
        return [0.0] * len(raw)
    if mx == mn:
        return [1.0 if v > 0 else 0.0 for v in raw]
    return [(v - mn) / (mx - mn) if v > 0 else 0.0 for v in raw]


def _field_bm25_scores(field_docs: list[list[str]], query_tokens: list[str]) -> list[float]:
    n = len(field_docs)
    if n == 0 or not query_tokens:
        return [0.0] * n
    corpus = [doc if doc else ["__empty__"] for doc in field_docs]
    bm25 = BM25Okapi(corpus)
    raw = list(bm25.get_scores(query_tokens))
    for i, doc in enumerate(field_docs):
        if not doc:
            raw[i] = 0.0
    return _normalize_bm25(raw)


def is_evidence_oriented_query(query: str) -> bool:
    return bool(_EVIDENCE_QUERY_RE.search(query or ""))


def evidence_score(level: str | None) -> float:
    if level is None:
        return 0.35
    return _EVIDENCE_WEIGHT.get(level.strip().lower(), 0.35)


def source_score(source_type: str | None) -> float:
    if source_type is None:
        return _UNKNOWN_SOURCE
    key = source_type.strip().lower()
    return _SOURCE_WEIGHT.get(key, _UNKNOWN_SOURCE)


def _generic_only_penalty(
    query_tokens: list[str],
    field_tokens: list[str],
    specific_diseases: list[str],
    normalized_field: str,
) -> float:
    if not specific_diseases:
        return 1.0
    for disease in specific_diseases:
        if entity_in_doc(disease, normalized_field, field_tokens):
            return 1.0
    fset = _expand_field_set(field_tokens)
    matched_q = [t for t in query_tokens if _token_match(t, fset)]
    if not matched_q:
        return 1.0
    non_generic = [
        t for t in matched_q
        if t not in GENERIC_MATCH_TOKENS
        and not t.startswith("__med_")
        and len(t) > 2
    ]
    med_hits = [t for t in matched_q if t.startswith("__med_")]
    if med_hits:
        return 1.0
    if matched_q and not non_generic:
        return 0.35
    if matched_q and all(t in GENERIC_MATCH_TOKENS or len(t) <= 3 for t in matched_q):
        return 0.35
    return 1.0


def _image_field_score(
    query_norm: str,
    image_norm: str,
    query_tokens: list[str],
    image_tokens: list[str],
    idf: dict[str, float],
    specific_diseases: list[str],
) -> float:
    if not image_norm.strip():
        return 0.0
    q_mod = detect_modality_terms(query_norm)
    i_mod = detect_modality_terms(image_norm)
    if q_mod:
        modality_match = len(q_mod & i_mod) / len(q_mod)
    else:
        modality_match = _weighted_coverage(query_tokens, image_tokens, idf)
    if specific_diseases:
        hits = sum(1 for d in specific_diseases if entity_in_doc(d, image_norm, image_tokens))
        disease_match = hits / len(specific_diseases)
    else:
        disease_match = _weighted_coverage(query_tokens, image_tokens, idf)
    return 0.4 * modality_match + 0.6 * (modality_match * disease_match)


def medical_score_all(metadatas: list[Metadata], query: str) -> list[ScoreBreakdown]:
    """
    分字段医学概念感知评分。
    返回完整的 ScoreBreakdown 列表。
    """
    query_norm = normalize_medical_text(query)
    query_tokens = content_tokens(query_norm)
    n = len(metadatas)
    if n == 0:
        return []

    aspects = detect_query_aspects(query)
    constraints = detect_query_constraints(query_norm)
    entities = extract_query_entities(query_norm, query_tokens)
    specific_diseases = specific_disease_concepts_in_query(query_norm)

    field_token_lists: dict[str, list[list[str]]] = {
        "title": [], "summary": [], "text": [], "image": [],
    }
    field_norms: dict[str, list[str]] = {
        "title": [], "summary": [], "text": [], "image": [],
    }
    doc_norms: list[str] = []

    for meta in metadatas:
        texts = _field_texts(meta)
        norms = {k: normalize_medical_text(v) for k, v in texts.items()}
        doc_norms.append(" ".join(norms.values()))
        for name in _FIELD_WEIGHTS:
            field_norms[name].append(norms[name])
            field_token_lists[name].append(content_tokens(norms[name]))

    all_tokens = [
        list(dict.fromkeys(
            field_token_lists["title"][i]
            + field_token_lists["summary"][i]
            + field_token_lists["text"][i]
            + field_token_lists["image"][i]
        ))
        for i in range(n)
    ]
    idf = _token_idf(all_tokens)

    bm25_by_field = {
        name: _field_bm25_scores(docs, query_tokens)
        for name, docs in field_token_lists.items()
    }

    evidence_q = is_evidence_oriented_query(query)
    prelim: list[tuple[Metadata, int, dict]] = []

    for i, meta in enumerate(metadatas):
        field_scores: dict[str, float] = {}
        for name in _FIELD_WEIGHTS:
            if name == "image":
                field_scores[name] = _image_field_score(
                    query_norm, field_norms["image"][i],
                    query_tokens, field_token_lists["image"][i],
                    idf, specific_diseases,
                )
                continue
            penalty = _generic_only_penalty(
                query_tokens, field_token_lists[name][i],
                specific_diseases, field_norms[name][i],
            )
            cov = _weighted_coverage(
                query_tokens, field_token_lists[name][i], idf,
                generic_penalty=penalty,
            )
            bigram = _bigram_cooccur(query_tokens, field_token_lists[name][i])
            field_scores[name] = 0.5 * cov + 0.3 * bm25_by_field[name][i] + 0.2 * bigram

        rel = sum(_FIELD_WEIGHTS[name] * field_scores[name] for name in _FIELD_WEIGHTS)
        union_pen = _generic_only_penalty(
            query_tokens, all_tokens[i], specific_diseases, doc_norms[i],
        )
        union_cov = _weighted_coverage(
            query_tokens, all_tokens[i], idf, generic_penalty=union_pen,
        )
        topic = _weighted_coverage(query_tokens, field_token_lists["title"][i], idf) if query_tokens else 0.0

        # Constraint coverage
        if constraints:
            c_hits = sum(1 for c in constraints if constraint_doc_hit(c, doc_norms[i]))
            constraint_cov = c_hits / len(constraints)
            if specific_diseases:
                disease_ok = any(
                    entity_in_doc(d, doc_norms[i], all_tokens[i])
                    for d in specific_diseases
                )
                if not disease_ok:
                    constraint_cov *= 0.15
        else:
            constraint_cov = 0.0

        # Aspect coverage
        if aspects:
            doc_l = doc_norms[i].lower()
            aspect_hits = sum(1 for a in aspects if aspect_doc_hit(a, doc_l))
            aspect_cov = aspect_hits / len(aspects)
            if specific_diseases:
                disease_ok = any(
                    entity_in_doc(d, doc_norms[i], all_tokens[i])
                    for d in specific_diseases
                )
                if not disease_ok:
                    aspect_cov *= 0.25
        else:
            aspect_cov = 0.0

        # Joint entity coverage
        if entities:
            ent_hits = sum(1 for e in entities if entity_in_doc(e, doc_norms[i], all_tokens[i]))
            joint = ent_hits / len(entities)
        else:
            joint = 0.0

        relevance = (
            rel + 0.25 * topic + 0.20 * union_cov
            + 0.20 * aspect_cov + 0.30 * joint + 0.25 * constraint_cov
        )

        # Opposite population cue
        if "elderly" in constraints:
            dl = doc_norms[i].lower()
            has_elderly = constraint_doc_hit("elderly", doc_norms[i])
            young_cue = any(x in dl for x in ["young adult", "younger adult", "young adults", "younger adults"])
            if young_cue and not has_elderly:
                relevance *= 0.55

        # Missing disease penalty
        if specific_diseases:
            disease_ok = any(
                entity_in_doc(d, doc_norms[i], all_tokens[i])
                for d in specific_diseases
            )
            if not disease_ok:
                relevance *= 0.55

        ev = evidence_score(meta.evidence_level)
        src = source_score(meta.source_type) if evidence_q else 0.0
        quality = 0.65 * ev + 0.35 * src if evidence_q else ev

        prelim.append((meta, i, {
            "field_scores": field_scores,
            "rel": rel, "topic": topic,
            "aspect": aspect_cov, "joint": joint, "constraint": constraint_cov,
            "relevance": relevance, "ev": ev, "src": src, "quality": quality,
        }))

    max_rel = max((p[2]["relevance"] for p in prelim), default=0.0)
    breakdowns: list[ScoreBreakdown] = []

    for meta, i, s in prelim:
        relevance = s["relevance"]
        quality = s["quality"]
        closeness = relevance / max_rel if max_rel > 1e-12 else 0.0
        quality_bonus = 0.15 * quality * (closeness ** 2)
        final = relevance + quality_bonus
        fs = s["field_scores"]

        breakdowns.append(ScoreBreakdown(
            index=i, id=meta.id,
            title_score=fs["title"], summary_score=fs["summary"],
            text_score=fs["text"], image_score=fs["image"],
            rel=s["rel"], topic=s["topic"],
            aspect=s["aspect"], joint=s["joint"], constraint=s["constraint"],
            relevance=relevance, evidence=s["ev"], source_bonus=s["src"],
            quality=quality, final=final,
        ))

    return breakdowns


def medical_rerank(
    metadatas: list[Metadata],
    query: str,
    top_k: int = 10,
) -> list[tuple[Metadata, ScoreBreakdown]]:
    """
    对候选集做医学概念感知重排序，返回 top-k。
    """
    if not metadatas:
        return []
    breakdowns = medical_score_all(metadatas, query)
    ranked = sorted(breakdowns, key=lambda b: (b.final, b.relevance, b.topic, b.joint, -b.index), reverse=True)
    top = ranked[:top_k]
    return [(metadatas[b.index], b) for b in top]
