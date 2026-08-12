from __future__ import annotations

import re

# Canonical concept id -> surface forms (lowercase). Expandable, not a full ontology.
MEDICAL_CONCEPT_ALIASES: dict[str, list[str]] = {
    "elderly": [
        "elderly patients",
        "elderly patient",
        "elderly",
        "older adults",
        "older adult",
        "older people",
    ],
    "pediatric": [
        "pediatric patients",
        "pediatric patient",
        "pediatric",
        "children",
        "child",
    ],
    "pregnant": [
        "pregnant women",
        "pregnant woman",
        "pregnancy",
        "pregnant",
    ],
    "hypertension": [
        "antihypertensive therapy",
        "antihypertensive treatment",
        "antihypertensive medications",
        "antihypertensive medication",
        "antihypertensive",
        "blood pressure control",
        "blood pressure",
        "hypertensive",
        "hypertension",
    ],
    "myocardial_infarction": [
        "acute myocardial infarction",
        "myocardial infarction",
        "heart attack",
        "ami",
    ],
    "pulmonary_embolism": [
        "pulmonary embolism",
        "pulmonary emboli",
        "pe",
    ],
    "breast_cancer": [
        "breast cancer",
    ],
    "type2_diabetes": [
        "type 2 diabetes",
        "type2 diabetes",
        "t2dm",
        "t2d",
    ],
    "metastatic": [
        "metastatic",
        "metastasis",
        "metastases",
    ],
    "early_stage": [
        "early-stage",
        "early stage",
    ],
    "stroke": [
        "ischemic stroke",
        "stroke",
    ],
    "diabetes": [
        "diabetes mellitus",
        "diabetes",
    ],
    "obesity": [
        "body mass index",
        "obesity",
        "obese",
    ],
}

# Population / disease-subtype style constraints (canonical ids).
CONSTRAINT_CONCEPTS = frozenset(
    {
        "elderly",
        "pediatric",
        "pregnant",
        "metastatic",
        "early_stage",
        "type2_diabetes",
    }
)

# Imaging modality surface forms.
MODALITY_TERMS = [
    "chest ct",
    "computed tomography",
    "ct angiography",
    "ct scan",
    "mri",
    "x-ray",
    "x ray",
    "ultrasound",
    "chest",
    "ct",
]

# Query intent / aspect: (regex, aspect_label, doc phrases that evidence the aspect)
ASPECT_PATTERNS: list[tuple[re.Pattern[str], str, list[str]]] = [
    (
        re.compile(r"\b(adverse effects?|side effects?|toxicity|safety)\b", re.I),
        "adverse",
        ["adverse effect", "adverse effects", "side effect", "side effects", "toxicity", "safety"],
    ),
    (
        re.compile(r"\brisk factors?\b", re.I),
        "risk_factor",
        ["risk factor", "risk factors"],
    ),
    (
        re.compile(r"\b(diagnos\w*|diagnostic)\b", re.I),
        "diagnosis",
        ["diagnosis", "diagnose", "diagnosing", "diagnostic"],
    ),
    (
        re.compile(r"\b(prevent\w*|prevention)\b", re.I),
        "prevention",
        ["prevention", "prevent", "preventive", "primary prevention"],
    ),
    (
        re.compile(r"\b(mechanism|inhibit\w*|pharmacolog\w*)\b", re.I),
        "mechanism",
        ["mechanism", "mechanisms", "inhibit", "inhibition", "pharmacology", "pharmacological"],
    ),
    (
        re.compile(r"\b(treatment|therapy|therapies|effective|effectiveness|efficacy)\b", re.I),
        "treatment",
        ["treatment", "therapy", "therapies", "effective", "effectiveness", "efficacy"],
    ),
    (
        re.compile(r"\b(associat\w*|outcomes?|improve outcomes?)\b", re.I),
        "association",
        ["association", "associated", "outcome", "outcomes"],
    ),
    (
        re.compile(r"\b(mortality|survival)\b", re.I),
        "mortality",
        ["mortality", "survival"],
    ),
    (
        re.compile(r"\b(look like|imaging findings?|image showing|ct (image|showing))\b", re.I),
        "imaging_appearance",
        ["imaging", "image", "ct image", "findings", "showing", "demonstrating"],
    ),
]

# Tokens that are too generic to count as disease/entity anchors alone.
GENERIC_MATCH_TOKENS = frozenset(
    {
        "risk",
        "factor",
        "factors",
        "chest",
        "ct",
        "treatment",
        "therapy",
        "diagnosis",
        "disease",
        "patient",
        "patients",
        "adult",
        "adults",
        "study",
        "review",
        "cancer",  # alone without organ; breast_cancer is specific concept
        "effect",
        "effects",
        "health",
        "clinical",
    }
)

_ASPECT_STOP = frozenset(
    {
        "adverse",
        "risk_factor",
        "diagnosis",
        "prevention",
        "mechanism",
        "treatment",
        "association",
        "mortality",
        "imaging_appearance",
    }
)


def _canonical_token(concept_id: str) -> str:
    return f"__med_{concept_id}__"


def normalize_medical_text(text: str) -> str:
    """
    Phrase-level medical normalization for both query and metadata fields.
    Injects canonical tokens alongside original wording.
    """
    if not text:
        return ""
    lowered = text.lower()
    # Longest alias first to prefer multi-word matches.
    replacements: list[tuple[str, str]] = []
    for concept_id, aliases in MEDICAL_CONCEPT_ALIASES.items():
        canon = _canonical_token(concept_id)
        for alias in sorted(aliases, key=len, reverse=True):
            replacements.append((alias.lower(), canon))

    # Apply non-overlapping left-to-right longest matches via iterative scan.
    out: list[str] = []
    i = 0
    n = len(lowered)
    original = text
    while i < n:
        matched = False
        for alias, canon in replacements:
            if lowered.startswith(alias, i):
                # require word-ish boundary
                end = i + len(alias)
                before_ok = i == 0 or not lowered[i - 1].isalnum()
                after_ok = end >= n or not lowered[end].isalnum()
                if before_ok and after_ok:
                    out.append(original[i:end])
                    out.append(" ")
                    out.append(canon)
                    out.append(" ")
                    i = end
                    matched = True
                    break
        if not matched:
            out.append(original[i])
            i += 1
    return "".join(out)


def detect_query_aspects(query: str) -> list[str]:
    q = query or ""
    found: list[str] = []
    for pattern, label, _ in ASPECT_PATTERNS:
        if pattern.search(q) and label not in found:
            found.append(label)
    return found


def aspect_doc_hit(aspect: str, doc_text_lower: str) -> bool:
    for pattern, label, phrases in ASPECT_PATTERNS:
        if label != aspect:
            continue
        return any(p in doc_text_lower for p in phrases)
    return False


def detect_query_constraints(normalized_query: str) -> list[str]:
    """Return canonical constraint concept ids present in normalized query."""
    q = (normalized_query or "").lower()
    hits: list[str] = []
    for cid in CONSTRAINT_CONCEPTS:
        token = _canonical_token(cid)
        if token in q or any(
            a in q for a in MEDICAL_CONCEPT_ALIASES.get(cid, []) if len(a) > 3
        ):
            if cid not in hits:
                hits.append(cid)
    return hits


def constraint_doc_hit(constraint_id: str, normalized_doc: str) -> bool:
    d = (normalized_doc or "").lower()
    token = _canonical_token(constraint_id)
    if token in d:
        return True
    return any(a in d for a in MEDICAL_CONCEPT_ALIASES.get(constraint_id, []))


def extract_query_entities(normalized_query: str, query_tokens: list[str]) -> list[str]:
    """
    Important medical entities / canonical concepts for joint coverage.
    Prefer injected canonical tokens; also keep specific multi-word disease cues.
    """
    entities: list[str] = []
    q = normalized_query.lower()
    for cid in MEDICAL_CONCEPT_ALIASES:
        tok = _canonical_token(cid)
        if tok in q or tok in query_tokens:
            if cid not in entities:
                entities.append(cid)

    # Specific free-text disease phrases not always in alias map as single concept
    extras = [
        "pulmonary embolism",
        "statin",
        "metformin",
        "troponin",
        "aspirin",
        "vitamin d",
        "lung cancer",
    ]
    for phrase in extras:
        if phrase in q and phrase not in entities:
            entities.append(phrase)

    # Content tokens that look specific (long, non-generic, non-aspect)
    for t in query_tokens:
        if t.startswith("__med_") and t.endswith("__"):
            cid = t[len("__med_") : -2]
            if cid not in entities:
                entities.append(cid)
            continue
        if len(t) <= 3:
            continue
        if t in GENERIC_MATCH_TOKENS or t in _ASPECT_STOP:
            continue
        if t not in entities:
            entities.append(t)
    return entities


def entity_in_doc(entity: str, normalized_doc: str, doc_tokens: list[str]) -> bool:
    d = normalized_doc.lower()
    if entity.startswith("__med_"):
        return entity in d or entity in doc_tokens
    if entity in MEDICAL_CONCEPT_ALIASES:
        return constraint_doc_hit(entity, normalized_doc)
    if " " in entity:
        return entity in d
    # token match with simple plural
    toks = set(doc_tokens)
    return entity in toks or entity + "s" in toks or (
        entity.endswith("s") and entity[:-1] in toks
    )


def detect_modality_terms(text: str) -> set[str]:
    t = (text or "").lower()
    hits: set[str] = set()
    for m in sorted(MODALITY_TERMS, key=len, reverse=True):
        if m in t:
            hits.add(m)
    return hits


def specific_disease_concepts_in_query(normalized_query: str) -> list[str]:
    """Disease-like concepts used for specific-vs-generic and image disease match."""
    disease_ids = [
        "myocardial_infarction",
        "pulmonary_embolism",
        "breast_cancer",
        "type2_diabetes",
        "stroke",
        "diabetes",
        "obesity",
        "hypertension",
    ]
    q = normalized_query.lower()
    out: list[str] = []
    for cid in disease_ids:
        if _canonical_token(cid) in q or any(
            a in q for a in MEDICAL_CONCEPT_ALIASES.get(cid, []) if len(a) >= 4
        ):
            out.append(cid)
    # free phrases
    for phrase in ("pulmonary embolism", "statin", "pneumonia"):
        if phrase in q and phrase not in out:
            # pneumonia only if in query
            if phrase == "pneumonia" or phrase in q:
                out.append(phrase)
    return [x for x in out if x != "pneumonia" or "pneumonia" in q]
