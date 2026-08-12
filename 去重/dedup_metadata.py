#!/usr/bin/env python3
"""Incremental metadata duplicate detection with BM25, vector recall, RRF, and medical concept reranking."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

Record = Dict[str, Any]

DEFAULT_CONCEPT_PATH = Path(__file__).with_name("medical_concepts.json")
TEXT_FIELDS = ("title", "text_summary", "text", "image_summary")
WEIGHTS = {"title": 0.30, "text_summary": 0.25, "text": 0.35, "image_summary": 0.10}
DUPLICATE_THRESHOLD = 0.88
MERGE_THRESHOLD = 0.52
BM25_TOP_N = 80
VECTOR_TOP_N = 80
RRF_TOP_K = 50
RRF_K = 60
PAPER_SOURCE_KEYWORDS = ("论文", "期刊", "文献", "journal", "article", "paper", "pubmed", "medline")
PAPER_TEXT_KEYWORDS = (
    "论文",
    "期刊",
    "文献",
    "研究论文",
    "学术论文",
    "doi",
    "abstract",
    "methods",
    "results",
    "conclusion",
    "randomized",
    "cohort",
    "meta analysis",
)

# Editable synonym/concept dictionary. It is intentionally rule-based so this
# script can run without training data, model downloads, or external services.
CONCEPT_GROUPS: Dict[str, Sequence[str]] = {
    "type2_diabetes": ("2型糖尿病", "二型糖尿病", "糖尿病"),
    "hypertension": ("高血压", "血压异常", "血压升高", "血压风险"),
    "asthma": ("哮喘", "喘息", "气道炎症", "呼吸不适"),
    "influenza": ("流感", "季节性流感", "流感样症状"),
    "hpv_cervical": ("hpv", "宫颈病变", "宫颈癌", "宫颈筛查"),
    "rheumatoid_arthritis": ("类风湿关节炎", "类风湿"),
    "migraine": ("偏头痛",),
    "hypothyroidism": ("甲状腺功能减退", "甲减"),
    "cataract": ("白内障",),
    "hepatitis_b": ("乙型肝炎", "慢性乙型肝炎", "乙肝"),
    "hepatitis_c": ("丙型肝炎", "丙肝"),
    "appendicitis": ("阑尾炎", "急性阑尾炎"),
    "urinary_tract_infection": ("尿路感染",),
    "psoriasis": ("银屑病",),
    "anemia": ("贫血",),
    "burn": ("烧伤",),
    "individualized_management": ("个体化", "个体化治疗", "个体化管理", "按年龄", "按病程", "治疗耐受性", "个体化设定"),
    "blood_glucose_management": ("血糖管理", "血糖控制", "血糖目标", "血糖监测", "单次血糖", "长期指标", "低血糖"),
    "diet_intervention": ("饮食管理", "饮食结构", "营养", "营养干预", "膳食", "饮食"),
    "exercise_intervention": ("运动干预", "适量运动", "身体活动", "运动"),
    "weight_management": ("体重管理", "体重控制", "体重"),
    "medication_management": ("降糖药物", "药物选择", "药物调整", "抗病毒", "抗病毒治疗", "用药"),
    "complication_risk": ("并发症", "并发症筛查", "并发症风险", "合并症", "风险评估"),
    "cardiovascular_risk": ("心血管", "心血管风险", "冠心病"),
    "kidney_function": ("肾功能", "肾脏"),
    "eye_screening": ("眼底", "视觉功能", "视力"),
    "foot_care": ("足部", "足部照护", "足部评估"),
    "screening_followup": ("筛查", "随访", "复查", "异常结果"),
    "diagnosis_assessment": ("诊断", "鉴别", "评估", "识别"),
    "lab_differentiation": ("实验室鉴别", "红细胞指标", "血常规", "网织红细胞", "铁蛋白", "维生素水平"),
    "pregnancy_context": ("妊娠", "孕期", "产科"),
    "chronic_antiviral_management": ("慢性乙型肝炎", "乙肝病毒载量", "抗病毒评估", "长期规律使用", "自行停药"),
    "home_care": ("居家", "家庭环境", "家庭", "健康防护"),
    "monitoring": ("监测", "持续监测", "动态观察"),
    "treatment_management": ("治疗", "管理", "防治", "控制"),
}

DISEASE_CONCEPTS = {
    "type2_diabetes",
    "hypertension",
    "asthma",
    "influenza",
    "hpv_cervical",
    "rheumatoid_arthritis",
    "migraine",
    "hypothyroidism",
    "cataract",
    "hepatitis_b",
    "hepatitis_c",
    "appendicitis",
    "urinary_tract_infection",
    "psoriasis",
    "anemia",
    "burn",
}

SYMPTOM_CONCEPTS = {
    "abdominal_pain",
    "chest_discomfort",
    "dizziness_fatigue",
    "dyspnea",
    "fever",
    "headache",
    "palpitation",
    "skin_lesion",
    "urinary_symptoms",
    "vision_blur",
}
MERGE_SIGNAL_CONCEPTS = {"vision_function"}

SCOPE_CONFLICT_RULES = [
    {"disease": "anemia", "exclusive_concept": "pregnancy_context"},
    {"disease": "anemia", "exclusive_concept": "lab_differentiation"},
    {"disease": "hepatitis_b", "exclusive_concept": "chronic_antiviral_management"},
]
FILTER_PREFERRED_DISEASES = {"heart_failure", "atrial_fibrillation", "pneumonia", "wound_injury"}
MERGE_PREFERRED_DISEASES = {"dyslipidemia", "heatstroke", "kidney_stone", "myopia", "drug_interaction"}
FORCE_INSERT_DISEASES = {"pcos"}


def load_concept_config(path: Path | None = None) -> None:
    config_path = path or DEFAULT_CONCEPT_PATH
    if not config_path.exists():
        return
    data = json.loads(config_path.read_text(encoding="utf-8-sig"))
    concept_groups = data.get("concept_groups", {})
    disease_concepts_config = data.get("disease_concepts", [])
    symptom_concepts_config = data.get("symptom_concepts", [])
    filter_preferred_config = data.get("filter_preferred_diseases", [])
    merge_preferred_config = data.get("merge_preferred_diseases", [])
    force_insert_config = data.get("force_insert_diseases", [])
    merge_signal_config = data.get("merge_signal_concepts", [])
    scope_rules = data.get("scope_conflict_rules", [])
    if not isinstance(concept_groups, dict):
        raise ValueError("concept_groups must be an object")
    if not isinstance(disease_concepts_config, list):
        raise ValueError("disease_concepts must be a list")
    if not isinstance(symptom_concepts_config, list):
        raise ValueError("symptom_concepts must be a list")
    if not isinstance(filter_preferred_config, list):
        raise ValueError("filter_preferred_diseases must be a list")
    if not isinstance(merge_preferred_config, list):
        raise ValueError("merge_preferred_diseases must be a list")
    if not isinstance(force_insert_config, list):
        raise ValueError("force_insert_diseases must be a list")
    if not isinstance(merge_signal_config, list):
        raise ValueError("merge_signal_concepts must be a list")
    if not isinstance(scope_rules, list):
        raise ValueError("scope_conflict_rules must be a list")
    CONCEPT_GROUPS.clear()
    CONCEPT_GROUPS.update({str(key): tuple(map(str, value)) for key, value in concept_groups.items()})
    DISEASE_CONCEPTS.clear()
    DISEASE_CONCEPTS.update(map(str, disease_concepts_config))
    SYMPTOM_CONCEPTS.clear()
    SYMPTOM_CONCEPTS.update(map(str, symptom_concepts_config))
    FILTER_PREFERRED_DISEASES.clear()
    FILTER_PREFERRED_DISEASES.update(map(str, filter_preferred_config))
    MERGE_PREFERRED_DISEASES.clear()
    MERGE_PREFERRED_DISEASES.update(map(str, merge_preferred_config))
    FORCE_INSERT_DISEASES.clear()
    FORCE_INSERT_DISEASES.update(map(str, force_insert_config))
    MERGE_SIGNAL_CONCEPTS.clear()
    MERGE_SIGNAL_CONCEPTS.update(map(str, merge_signal_config))
    SCOPE_CONFLICT_RULES.clear()
    SCOPE_CONFLICT_RULES.extend(scope_rules)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def stringify(value: Any) -> str:
    if isinstance(value, list):
        return " ".join(stringify(item) for item in value)
    if isinstance(value, dict):
        return " ".join(stringify(item) for item in value.values())
    return "" if value is None else str(value)


def normalize(value: Any) -> str:
    text = stringify(value).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^\w\u4e00-\u9fff]+", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def record_text(record: Record) -> str:
    return " ".join(normalize(record.get(field)) for field in TEXT_FIELDS)


def is_paper_record(record: Record) -> bool:
    source_type = normalize(record.get("source_type"))
    if any(keyword in source_type for keyword in PAPER_SOURCE_KEYWORDS):
        return True
    text = record_text(record)
    return any(keyword in text for keyword in PAPER_TEXT_KEYWORDS)


def tokens(value: Any) -> List[str]:
    text = normalize(value).replace(" ", "")
    chunks = re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", text)
    if any("\u4e00" <= c <= "\u9fff" for c in text):
        chunks += [text[i : i + 2] for i in range(len(text) - 1)]
        chunks += [text[i : i + 3] for i in range(len(text) - 2)]
    return chunks


def extract_concepts(record: Record) -> Set[str]:
    text = record_text(record).replace(" ", "")
    concepts = set()
    for concept, phrases in CONCEPT_GROUPS.items():
        if any(normalize(phrase).replace(" ", "") in text for phrase in phrases):
            concepts.add(concept)
    return concepts


def disease_concepts(concepts: Set[str]) -> Set[str]:
    return concepts & DISEASE_CONCEPTS


def symptom_concepts(concepts: Set[str]) -> Set[str]:
    return concepts & SYMPTOM_CONCEPTS


def retrieval_tokens(record: Record) -> List[str]:
    base = tokens(record_text(record))
    concepts = extract_concepts(record)
    concept_tokens = [f"concept:{concept}" for concept in concepts for _ in range(4)]
    disease_tokens = [f"disease:{concept}" for concept in disease_concepts(concepts) for _ in range(6)]
    symptom_tokens = [f"symptom:{concept}" for concept in symptom_concepts(concepts) for _ in range(5)]
    title_tokens = [f"title:{token}" for token in tokens(record.get("title")) for _ in range(2)]
    return base + concept_tokens + disease_tokens + symptom_tokens + title_tokens


def jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    a, b = set(left), set(right)
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def overlap_coverage(left: Iterable[str], right: Iterable[str]) -> float:
    a, b = set(left), set(right)
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def cosine(left: Iterable[str], right: Iterable[str]) -> float:
    a, b = Counter(left), Counter(right)
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    dot = sum(a[x] * b[x] for x in common)
    denominator = (sum(x * x for x in a.values()) * sum(x * x for x in b.values())) ** 0.5
    return dot / denominator if denominator else 0.0


def field_similarity(a: Any, b: Any) -> float:
    left, right = tokens(a), tokens(b)
    if not left or not right:
        return 0.0
    return 0.6 * jaccard(left, right) + 0.4 * cosine(left, right)


def sentence_level_similarity(a: Record, b: Record) -> Tuple[float, Dict[str, Any]]:
    left_sentences = split_sentences(" ".join(stringify(a.get(field)) for field in TEXT_FIELDS))
    right_sentences = split_sentences(" ".join(stringify(b.get(field)) for field in TEXT_FIELDS))
    best_pairs = []
    best_score = 0.0
    for left in left_sentences:
        for right in right_sentences:
            score = field_similarity(left, right)
            if score > best_score:
                best_score = score
            if score >= 0.80:
                best_pairs.append(
                    {
                        "left_sentence": left,
                        "right_sentence": right,
                        "score": round(score, 6),
                    }
                )
    return round(best_score, 6), {
        "mode": "paper_sentence_level",
        "best_sentence_score": round(best_score, 6),
        "matched_sentence_pairs": best_pairs[:10],
    }


def literal_similarity(a: Record, b: Record) -> Tuple[float, Dict[str, float]]:
    details = {field: field_similarity(a.get(field), b.get(field)) for field in TEXT_FIELDS}
    active_weight = sum(WEIGHTS[field] for field in TEXT_FIELDS if normalize(a.get(field)) and normalize(b.get(field)))
    score = sum(details[field] * WEIGHTS[field] for field in TEXT_FIELDS) / active_weight if active_weight else 0.0
    return round(score, 6), {key: round(value, 6) for key, value in details.items()}


def concept_similarity(a: Record, b: Record) -> Tuple[float, Dict[str, Any]]:
    left = extract_concepts(a)
    right = extract_concepts(b)
    left_diseases = disease_concepts(left)
    right_diseases = disease_concepts(right)
    left_symptoms = symptom_concepts(left)
    right_symptoms = symptom_concepts(right)
    shared = left & right
    shared_diseases = left_diseases & right_diseases
    shared_symptoms = left_symptoms & right_symptoms
    coverage = overlap_coverage(left, right)
    concept_jaccard = jaccard(left, right)
    disease_match = 1.0 if shared_diseases else 0.0
    if left_diseases and right_diseases and not shared_diseases:
        disease_match = -0.35
    score = 0.65 * coverage + 0.25 * concept_jaccard + 0.10 * max(disease_match, 0.0)
    if disease_match < 0:
        score = min(score, 0.25)
    return round(score, 6), {
        "left_concepts": sorted(left),
        "right_concepts": sorted(right),
        "shared_concepts": sorted(shared),
        "shared_diseases": sorted(shared_diseases),
        "shared_symptoms": sorted(shared_symptoms),
        "coverage": round(coverage, 6),
        "jaccard": round(concept_jaccard, 6),
        "disease_match": disease_match,
    }


def content_similarity(a: Record, b: Record) -> Tuple[float, Dict[str, Any]]:
    if is_paper_record(a) or is_paper_record(b):
        sentence_score, sentence_details = sentence_level_similarity(a, b)
        concept_score, concept_details = concept_similarity(a, b)
        title_score = field_similarity(a.get("title"), b.get("title"))
        score = max(sentence_score, 0.35 * concept_score + 0.15 * title_score)
        return round(score, 6), {
            "literal_score": round(sentence_score, 6),
            "concept_score": round(concept_score, 6),
            "title_score": round(title_score, 6),
            "field_scores": {},
            "concept_details": concept_details,
            "paper_details": sentence_details,
        }
    literal_score, field_scores = literal_similarity(a, b)
    concept_score, concept_details = concept_similarity(a, b)
    title_score = field_similarity(a.get("title"), b.get("title"))
    left_concepts = set(concept_details["left_concepts"])
    right_concepts = set(concept_details["right_concepts"])
    if not left_concepts and not right_concepts:
        score = literal_score
    else:
        score = 0.45 * literal_score + 0.45 * concept_score + 0.10 * title_score
    left_diseases = left_concepts & DISEASE_CONCEPTS
    right_diseases = right_concepts & DISEASE_CONCEPTS
    if left_diseases and right_diseases and not (left_diseases & right_diseases):
        score = min(score, 0.35)
    if (left_diseases or right_diseases) and not (left_diseases & right_diseases):
        score = min(score, 0.30)
    if (left_diseases & right_diseases) and concept_score >= 0.82 and literal_score >= 0.15:
        score = max(score, MERGE_THRESHOLD + 0.01)
    details: Dict[str, Any] = {
        "literal_score": round(literal_score, 6),
        "concept_score": round(concept_score, 6),
        "title_score": round(title_score, 6),
        "field_scores": field_scores,
        "concept_details": concept_details,
    }
    return round(score, 6), details


def has_scope_conflict(details: Dict[str, Any]) -> bool:
    concept_details = details.get("concept_details", {})
    left = set(concept_details.get("left_concepts") or [])
    right = set(concept_details.get("right_concepts") or [])
    shared_diseases = set(concept_details.get("shared_diseases") or [])
    left_diseases = left & DISEASE_CONCEPTS
    right_diseases = right & DISEASE_CONCEPTS
    if shared_diseases == {"wound_injury"} and (left_diseases - shared_diseases) and (right_diseases - shared_diseases):
        return True
    if shared_diseases & FORCE_INSERT_DISEASES:
        return True
    for rule in SCOPE_CONFLICT_RULES:
        disease = str(rule.get("disease", ""))
        exclusive_concept = str(rule.get("exclusive_concept", ""))
        if disease in shared_diseases and exclusive_concept and ((exclusive_concept in left) != (exclusive_concept in right)):
            return True
    return False


def should_filter_duplicate(score: float, details: Dict[str, Any]) -> bool:
    if score >= DUPLICATE_THRESHOLD:
        return True
    if has_scope_conflict(details):
        return False
    concept_details = details.get("concept_details", {})
    shared_diseases = set(concept_details.get("shared_diseases") or [])
    coverage = concept_details.get("coverage", 0.0)
    concept_jaccard = concept_details.get("jaccard", 0.0)
    literal_score = details.get("literal_score", 0.0)
    title_score = details.get("title_score", 0.0)
    concept_score = details.get("concept_score", 0.0)
    if shared_diseases & FILTER_PREFERRED_DISEASES:
        if "wound_injury" in shared_diseases:
            return concept_score >= 0.45 and (literal_score >= 0.035 or title_score >= 0.05)
        if "pneumonia" in shared_diseases:
            return concept_score >= 0.45 and literal_score >= 0.15
        if "atrial_fibrillation" in shared_diseases:
            return concept_score >= 0.45 and literal_score >= 0.20
        if "hypertension" in shared_diseases:
            return concept_score >= 0.45 and literal_score >= 0.15
        if "influenza" in shared_diseases:
            return concept_score >= 0.45 and literal_score >= 0.15
        return concept_score >= 0.45 and literal_score >= 0.15
    if shared_diseases & MERGE_PREFERRED_DISEASES:
        return False
    return bool(shared_diseases) and coverage >= 0.75 and concept_jaccard >= 0.50 and literal_score >= 0.15 and title_score >= 0.10


def should_merge_complementary(score: float, details: Dict[str, Any]) -> bool:
    if has_scope_conflict(details):
        return False
    if score >= MERGE_THRESHOLD:
        return True
    concept_details = details.get("concept_details", {})
    shared_diseases = set(concept_details.get("shared_diseases") or [])
    shared_symptoms = concept_details.get("shared_symptoms") or []
    coverage = concept_details.get("coverage", 0.0)
    concept_score = details.get("concept_score", 0.0)
    literal_score = details.get("literal_score", 0.0)
    title_score = details.get("title_score", 0.0)
    retrieval = details.get("retrieval", {})
    strong_retrieval = retrieval.get("bm25_rank", 999) <= 2 and retrieval.get("vector_rank", 999) <= 2
    related_retrieval = retrieval.get("bm25_rank", 999) <= 3 and retrieval.get("vector_rank", 999) <= 3
    left_diseases = set(concept_details.get("left_concepts") or []) & DISEASE_CONCEPTS
    right_diseases = set(concept_details.get("right_concepts") or []) & DISEASE_CONCEPTS
    shared_concepts = set(concept_details.get("shared_concepts") or [])
    if not shared_diseases:
        if left_diseases or right_diseases:
            if shared_concepts & MERGE_SIGNAL_CONCEPTS and related_retrieval and literal_score >= 0.02:
                return True
            return False
        if shared_concepts & MERGE_SIGNAL_CONCEPTS and related_retrieval and literal_score >= 0.02:
            return True
        if len(shared_concepts) >= 2 and related_retrieval and concept_score >= 0.35 and literal_score >= 0.05:
            return True
        return bool(shared_symptoms) and strong_retrieval and (literal_score >= 0.05 or title_score >= 0.05)
    if concept_score >= 0.90 and coverage >= 0.90:
        return True
    if shared_diseases and related_retrieval and concept_score >= 0.25 and (literal_score >= 0.02 or title_score >= 0.02):
        return True
    if strong_retrieval and literal_score >= 0.08:
        return True
    if concept_score >= 0.40 and literal_score >= 0.08:
        return True
    if strong_retrieval and concept_score >= 0.25 and (literal_score >= 0.03 or title_score >= 0.03):
        return True
    return concept_score >= 0.45 and coverage >= 0.50 and (literal_score >= 0.08 or title_score >= 0.08)


def match_priority(details: Dict[str, Any]) -> int:
    concept_details = details.get("concept_details", {})
    if concept_details.get("shared_diseases"):
        return 2
    if concept_details.get("shared_symptoms"):
        return 1
    return 0


def bm25_rank(record: Record, warehouse: Sequence[Record], top_n: int) -> Tuple[List[int], Dict[int, float]]:
    docs = [retrieval_tokens(item) for item in warehouse]
    query_terms = set(retrieval_tokens(record))
    if not docs or not query_terms:
        return [], {}
    avgdl = sum(len(doc) for doc in docs) / len(docs)
    df = Counter(term for doc in docs for term in set(doc))
    k1 = 1.5
    b = 0.75
    scores: Dict[int, float] = {}
    for index, doc in enumerate(docs):
        tf = Counter(doc)
        score = 0.0
        for term in query_terms:
            if term not in tf:
                continue
            idf = math.log(1 + (len(docs) - df[term] + 0.5) / (df[term] + 0.5))
            freq = tf[term]
            denom = freq + k1 * (1 - b + b * len(doc) / avgdl)
            score += idf * freq * (k1 + 1) / denom
        if score > 0:
            scores[index] = score
    ranked = sorted(scores, key=lambda i: scores[i], reverse=True)[:top_n]
    return ranked, scores


def tfidf_vector_rank(record: Record, warehouse: Sequence[Record], top_n: int) -> Tuple[List[int], Dict[int, float]]:
    docs = [retrieval_tokens(item) for item in warehouse]
    query = retrieval_tokens(record)
    if not docs or not query:
        return [], {}
    df = Counter(term for doc in docs for term in set(doc))
    query_tf = Counter(query)
    query_vector = {
        term: count * math.log(1 + (len(docs) + 1) / (df.get(term, 0) + 1))
        for term, count in query_tf.items()
    }
    query_norm = math.sqrt(sum(value * value for value in query_vector.values()))
    scores: Dict[int, float] = {}
    for index, doc in enumerate(docs):
        doc_tf = Counter(doc)
        common = set(query_vector) & set(doc_tf)
        if not common:
            continue
        doc_vector = {
            term: count * math.log(1 + (len(docs) + 1) / (df.get(term, 0) + 1))
            for term, count in doc_tf.items()
        }
        doc_norm = math.sqrt(sum(value * value for value in doc_vector.values()))
        if not query_norm or not doc_norm:
            continue
        score = sum(query_vector[term] * doc_vector[term] for term in common) / (query_norm * doc_norm)
        if score > 0:
            scores[index] = score
    ranked = sorted(scores, key=lambda i: scores[i], reverse=True)[:top_n]
    return ranked, scores


def rrf_fuse(rankings: Sequence[Sequence[int]], rrf_k: int, top_k: int) -> Tuple[List[int], Dict[int, Dict[str, Any]]]:
    fused: Dict[int, float] = {}
    rank_detail: Dict[int, Dict[str, Any]] = {}
    names = ("bm25", "vector")
    for name, ranking in zip(names, rankings):
        for rank, index in enumerate(ranking, start=1):
            fused[index] = fused.get(index, 0.0) + 1.0 / (rrf_k + rank)
            rank_detail.setdefault(index, {})[f"{name}_rank"] = rank
    ordered = sorted(fused, key=lambda i: fused[i], reverse=True)[:top_k]
    for index in ordered:
        rank_detail.setdefault(index, {})["rrf_score"] = round(fused[index], 8)
    return ordered, rank_detail


def retrieve_candidates(
    record: Record,
    warehouse: Sequence[Record],
    bm25_top_n: int = BM25_TOP_N,
    vector_top_n: int = VECTOR_TOP_N,
    rrf_top_k: int = RRF_TOP_K,
    rrf_k: int = RRF_K,
) -> Tuple[List[int], Dict[int, Dict[str, Any]]]:
    bm25_indices, bm25_scores = bm25_rank(record, warehouse, bm25_top_n)
    vector_indices, vector_scores = tfidf_vector_rank(record, warehouse, vector_top_n)
    candidates, detail = rrf_fuse((bm25_indices, vector_indices), rrf_k, rrf_top_k)
    record_diseases = disease_concepts(extract_concepts(record))
    if record_diseases:
        for index, candidate in enumerate(warehouse):
            if record_diseases & disease_concepts(extract_concepts(candidate)) and index not in candidates:
                candidates.append(index)
                detail.setdefault(index, {})["disease_fallback"] = True
    for index in candidates:
        detail.setdefault(index, {})["bm25_score"] = round(bm25_scores.get(index, 0.0), 6)
        detail.setdefault(index, {})["vector_score"] = round(vector_scores.get(index, 0.0), 6)
    return candidates, detail


def exact_key(record: Record) -> str:
    payload = {field: normalize(record.get(field)) for field in TEXT_FIELDS}
    payload["image_base64"] = record.get("image_base64") or record.get("image_base_64") or []
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def make_uid(record: Record) -> str:
    if record.get("id"):
        return str(record["id"])
    raw = json.dumps({field: normalize(record.get(field)) for field in TEXT_FIELDS}, ensure_ascii=False, sort_keys=True)
    return "meta_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def split_sentences(value: Any) -> List[str]:
    return [part.strip() for part in re.split(r"(?<=[。！？；;.!?\n])\s*", stringify(value)) if part.strip()]


def merge_text(old: Any, new: Any) -> str:
    existing = split_sentences(old)
    seen = {normalize(item) for item in existing}
    for item in split_sentences(new):
        if normalize(item) not in seen:
            existing.append(item)
            seen.add(normalize(item))
    return " ".join(existing)


def merge_records(old: Record, new: Record, score: float) -> Record:
    merged = copy.deepcopy(old)
    for field in TEXT_FIELDS:
        if field == "title":
            if len(normalize(new.get(field))) > len(normalize(old.get(field))):
                merged[field] = new.get(field)
        else:
            merged[field] = merge_text(old.get(field), new.get(field))
    for field in ("year", "source_type", "evidence_level", "last_retrieved_at"):
        if not merged.get(field) and new.get(field):
            merged[field] = new[field]
    old_images = old.get("image_base64") or old.get("image_base_64") or []
    new_images = new.get("image_base64") or new.get("image_base_64") or []
    merged["image_base64"] = list(dict.fromkeys([*old_images, *new_images]))
    merged["id"] = make_uid({**merged, "id": None})
    merged["merged_from"] = list(dict.fromkeys([*(old.get("merged_from") or [old.get("id")]), new.get("id")]))
    merged["updated_at"] = now_iso()
    merged["merge_score"] = score
    return merged


def best_match(
    record: Record,
    warehouse: Sequence[Record],
    bm25_top_n: int = BM25_TOP_N,
    vector_top_n: int = VECTOR_TOP_N,
    rrf_top_k: int = RRF_TOP_K,
    rrf_k: int = RRF_K,
) -> Tuple[int, float, Dict[str, Any]]:
    incoming_key = exact_key(record)
    for index, candidate in enumerate(warehouse):
        if incoming_key == exact_key(candidate):
            return index, 1.0, {
                "retrieval": {"exact_match": True},
                "literal_score": 1.0,
                "concept_score": 1.0,
                "title_score": 1.0,
                "field_scores": {field: 1.0 for field in TEXT_FIELDS},
                "concept_details": concept_similarity(record, candidate)[1],
            }
    candidates, retrieval_detail = retrieve_candidates(record, warehouse, bm25_top_n, vector_top_n, rrf_top_k, rrf_k)
    best = (-1, 0.0, {})
    best_key = (-1, 0.0)
    for index in candidates:
        score, details = content_similarity(record, warehouse[index])
        details["retrieval"] = retrieval_detail.get(index, {})
        key = (match_priority(details), score)
        if key > best_key:
            best = (index, score, details)
            best_key = key
    return best


def action_label(action: str) -> str:
    return {
        "insert": "入库",
        "filter_duplicate": "过滤",
        "merge": "整合",
    }.get(action, action)


def decision_reason(action: str, score: float, matched_id: Any, details: Dict[str, Any]) -> str:
    concept_details = details.get("concept_details", {})
    shared = concept_details.get("shared_concepts") or []
    shared_symptoms = concept_details.get("shared_symptoms") or []
    retrieval = details.get("retrieval", {})
    shared_text = "、".join(shared[:8]) if shared else "无明显核心概念重合"
    symptom_text = "、".join(shared_symptoms[:6]) if shared_symptoms else "无明显症状重合"
    literal_score = details.get("literal_score", 0.0)
    concept_score = details.get("concept_score", 0.0)
    retrieval_text = f"BM25排名 {retrieval.get('bm25_rank', '-')}, 向量排名 {retrieval.get('vector_rank', '-')}, RRF分 {retrieval.get('rrf_score', 0)}"
    if action == "filter_duplicate":
        if score >= 1.0:
            return "内容完全一致，判定为重复数据，不重复入库。"
        if score >= DUPLICATE_THRESHOLD:
            return f"综合相似度 {score:.3f} >= {DUPLICATE_THRESHOLD:.2f}，核心概念重合：{shared_text}；{retrieval_text}。直接过滤。"
        return f"综合相似度 {score:.3f} 未到硬重复阈值，但疾病实体一致、核心概念覆盖高、字面信息足够接近；核心概念重合：{shared_text}；{retrieval_text}。按语义重复直接过滤。"
    if action == "merge":
        if score >= MERGE_THRESHOLD:
            return f"综合相似度 {score:.3f} 介于 {MERGE_THRESHOLD:.2f} 和 {DUPLICATE_THRESHOLD:.2f} 之间；字面分 {literal_score:.3f}，概念分 {concept_score:.3f}，核心概念重合：{shared_text}；{retrieval_text}。判定为同一主题但信息互补，已与库内记录 {matched_id} 整合。"
        if shared_symptoms:
            return f"综合相似度 {score:.3f} 未到硬整合阈值，但症状/表现相似且召回强相关；字面分 {literal_score:.3f}，概念分 {concept_score:.3f}，症状重合：{symptom_text}，核心概念重合：{shared_text}；{retrieval_text}。按症状相似与库内记录 {matched_id} 整合。"
        return f"综合相似度 {score:.3f} 未到硬整合阈值，但疾病实体一致且核心概念存在部分重合；字面分 {literal_score:.3f}，概念分 {concept_score:.3f}，核心概念重合：{shared_text}；{retrieval_text}。按互补内容与库内记录 {matched_id} 整合。"
    if matched_id is None:
        return "库内没有召回到可比较记录，直接作为新数据入库。"
    if has_scope_conflict(details):
        return f"综合相似度 {score:.3f}，但命中范围冲突规则：同疾病下主任务或适用范围不同；字面分 {literal_score:.3f}，概念分 {concept_score:.3f}，核心概念重合：{shared_text}；{retrieval_text}。为避免误合并，作为新数据入库。"
    return f"综合相似度 {score:.3f} < {MERGE_THRESHOLD:.2f}；字面分 {literal_score:.3f}，概念分 {concept_score:.3f}，核心概念重合：{shared_text}；{retrieval_text}。未达到重复或整合阈值，作为新数据入库。"


def process(
    existing: List[Record],
    incoming: Sequence[Record],
    verbose: bool = False,
    bm25_top_n: int = BM25_TOP_N,
    vector_top_n: int = VECTOR_TOP_N,
    rrf_top_k: int = RRF_TOP_K,
    rrf_k: int = RRF_K,
) -> Tuple[List[Record], List[Dict[str, Any]]]:
    warehouse = copy.deepcopy(existing)
    report: List[Dict[str, Any]] = []
    for number, raw in enumerate(incoming, start=1):
        record = copy.deepcopy(raw)
        record["id"] = make_uid(record)
        if is_paper_record(record):
            warehouse.append(record)
            action = "insert"
            score = 0.0
            details = {"paper_details": {"mode": "incoming_paper_direct_insert"}}
            matched_id = None
        else:
            index, score, details = best_match(record, warehouse, bm25_top_n, vector_top_n, rrf_top_k, rrf_k)
            matched_id = warehouse[index].get("id") if index >= 0 else None
            if index < 0:
                warehouse.append(record)
                action = "insert"
            elif should_filter_duplicate(score, details):
                action = "filter_duplicate"
            elif should_merge_complementary(score, details):
                warehouse[index] = merge_records(warehouse[index], record, score)
                action = "merge"
            else:
                warehouse.append(record)
                action = "insert"
        reason = decision_reason(action, score, matched_id, details)
        decision = {
            "index": number,
            "incoming_id": record["id"],
            "incoming_title": record.get("title"),
            "action": action,
            "action_cn": action_label(action),
            "matched_id": matched_id,
            "score": score,
            "details": details,
            "reason": reason,
        }
        report.append(decision)
        if verbose:
            print(f"[{number}/{len(incoming)}] {action_label(action)} | incoming_id={record['id']} | matched_id={matched_id or '-'} | score={score:.3f} | title={record.get('title')}")
            print(f"  reason: {reason}")
    return warehouse, report


def read_records(path: Path) -> List[Record]:
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        return []
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            data = data.get("data", data.get("records", []))
        if not isinstance(data, list):
            raise ValueError("JSON root must be a list, or an object containing data/records")
        return data
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Incrementally filter or merge similar metadata records.")
    parser.add_argument("--existing", required=True, type=Path)
    parser.add_argument("--incoming", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--bm25-top-n", default=BM25_TOP_N, type=int, help="BM25 recall size before RRF.")
    parser.add_argument("--vector-top-n", default=VECTOR_TOP_N, type=int, help="Vector recall size before RRF.")
    parser.add_argument("--rrf-top-k", default=RRF_TOP_K, type=int, help="Final candidate count after RRF.")
    parser.add_argument("--rrf-k", default=RRF_K, type=int, help="RRF rank constant. 60 is a common default.")
    parser.add_argument("--concepts", default=DEFAULT_CONCEPT_PATH, type=Path, help="External medical concept configuration JSON.")
    parser.add_argument("--verbose", action="store_true", help="Print each incoming record decision as it is processed.")
    args = parser.parse_args()
    load_concept_config(args.concepts)
    warehouse, report = process(
        read_records(args.existing),
        read_records(args.incoming),
        verbose=args.verbose,
        bm25_top_n=args.bm25_top_n,
        vector_top_n=args.vector_top_n,
        rrf_top_k=args.rrf_top_k,
        rrf_k=args.rrf_k,
    )
    args.output.write_text(json.dumps(warehouse, ensure_ascii=False, indent=2), encoding="utf-8")
    args.report.write_text(json.dumps({"generated_at": now_iso(), "summary": Counter(item["action"] for item in report), "decisions": report}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"warehouse_size": len(warehouse), "summary": Counter(item["action"] for item in report)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
