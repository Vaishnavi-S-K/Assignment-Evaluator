from __future__ import annotations

import os
import re
from dataclasses import dataclass, asdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import joblib
import numpy as np


def _set_backend() -> None:
    os.environ.setdefault("KERAS_BACKEND", "jax")


_set_backend()

try:
    import keras
    import keras_nlp
except Exception as exc:  # pragma: no cover - import guard for UI startup
    keras = None  # type: ignore[assignment]
    keras_nlp = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

try:
    from transformers import AutoTokenizer
except Exception:  # pragma: no cover - optional fallback
    AutoTokenizer = None  # type: ignore[assignment]


PRESET = "deberta_v3_extra_small_en"
DEFAULT_WEIGHTS_PATH = Path("kaggle/working/best_model.weights.h5")
DEFAULT_SEQUENCE_LENGTH = 512
HF_TOKENIZER_NAME = "microsoft/deberta-v3-xsmall"
DEFAULT_SHORT_ANSWER_VECTORIZER_PATH = Path("short_answer/vectorizer.pkl")
DEFAULT_SHORT_ANSWER_MODEL_PATH = Path("short_answer/assignment_evaluator.pkl")
DEFAULT_CODE_INDEX_PATH = Path("code/checkpoint_bundle/code_index.joblib")

RESOURCE_LIBRARY: Dict[str, Dict[str, str]] = {
    "grammar": {
        "title": "Purdue OWL: Grammar",
        "url": "https://owl.purdue.edu/owl/general_writing/mechanics/grammar/index.html",
    },
    "coherence": {
        "title": "UNC Writing Center: Coherence",
        "url": "https://writingcenter.unc.edu/tips-and-tools/coherence/",
    },
    "evidence": {
        "title": "Harvard Writing Center: Using Evidence",
        "url": "https://writingcenter.fas.harvard.edu/pages/using-evidence",
    },
    "structure": {
        "title": "Essay Structure Guide",
        "url": "https://writingcenter.unc.edu/tips-and-tools/essay-structure/",
    },
    "accuracy": {
        "title": "Answering Questions Accurately",
        "url": "https://www.skillsyouneed.com/learn/answer-questions.html",
    },
    "completeness": {
        "title": "Complete Short Answers Checklist",
        "url": "https://writingcenter.unc.edu/tips-and-tools/brainstorming/",
    },
    "clarity": {
        "title": "Writing Clearly and Concisely",
        "url": "https://writingcenter.unc.edu/tips-and-tools/conciseness/",
    },
    "correctness": {
        "title": "Code Correctness & Testing",
        "url": "https://realpython.com/python-testing/",
    },
    "readability": {
        "title": "Clean Code and Readability",
        "url": "https://www.freecodecamp.org/news/how-to-write-readable-code/",
    },
    "efficiency": {
        "title": "Algorithm Efficiency Basics",
        "url": "https://www.khanacademy.org/computing/computer-science/algorithms/",
    },
    "edge_cases": {
        "title": "Handling Edge Cases in Code",
        "url": "https://www.geeksforgeeks.org/edge-cases-in-programming/",
    },
}


@dataclass
class EssayEvaluation:
    final_score: int
    confidence: float
    rubric_breakdown: Dict[str, float]
    similarity_risk: str
    strengths: List[str]
    weaknesses: List[str]
    feedback: str
    recommendations: List[str]
    resources: List[Dict[str, str]]
    model_status: str
    raw_score: float

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


EvaluationResult = EssayEvaluation


def _resources_for_weaknesses(weaknesses: Sequence[str], assignment_type: str = "Essay") -> List[Dict[str, str]]:
    """Return curated resources filtered by the assignment type.

    - For 'Essay' and 'Short Answer' only include writing-related resources
      (grammar, coherence, evidence, structure, accuracy, completeness, clarity).
    - For 'Code' include only coding resources (correctness, readability,
      efficiency, edge_cases).
    """
    items: List[Dict[str, str]] = []
    seen: set[str] = set()
    joined = " ".join(weaknesses or []).lower()

    essay_keys = {"grammar", "coherence", "evidence", "structure"}
    short_keys = {"accuracy", "completeness", "clarity"}
    code_keys = {"correctness", "readability", "efficiency", "edge_cases"}

    allowed: set[str]
    atype = (assignment_type or "Essay").lower()
    if atype == "code":
        allowed = code_keys
    elif atype == "short answer":
        allowed = short_keys
    else:
        # default to essay/writing resources (also covers generic 'Essay')
        allowed = essay_keys | short_keys

    for key in allowed:
        if key in joined:
            resource = RESOURCE_LIBRARY.get(key)
            if resource and resource["title"] not in seen:
                seen.add(resource["title"])
                items.append(resource)

    if not items:
        # fallback: suggest a short, non-code general resource for writing students
        if atype == "code":
            items.append({"title": "Intro to Testing and Debugging", "url": "https://realpython.com/python-testing/"})
        else:
            items.append({"title": "General Writing Resources", "url": "https://writingcenter.unc.edu/"})

    return items


def to_ordinal(y, num_classes=None, dtype="float32"):
    y = np.array(y, dtype="int")
    input_shape = y.shape

    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])

    y = y.reshape(-1)
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    range_values = np.arange(num_classes - 1)
    range_values = np.tile(np.expand_dims(range_values, 0), [n, 1])
    ordinal = np.zeros((n, num_classes - 1), dtype=dtype)
    ordinal[range_values < np.expand_dims(y, -1)] = 1
    output_shape = input_shape + (num_classes - 1,)
    ordinal = np.reshape(ordinal, output_shape)
    return ordinal


def build_model(preset: str = PRESET):
    if keras is None or keras_nlp is None:
        raise RuntimeError(f"Keras/KerasNLP import failed: {_IMPORT_ERROR}")

    classifier = keras_nlp.models.DebertaV3Classifier.from_preset(
        preset, preprocessor=None, num_classes=6, name="deberta_v3_classifier"
    )
    inputs = classifier.input
    logits = classifier(inputs)
    outputs = keras.layers.Activation("sigmoid")(logits)
    model = keras.Model(inputs, outputs)
    return model


def _tokenize_texts(texts: Sequence[str], preset: str = PRESET, sequence_length: int = DEFAULT_SEQUENCE_LENGTH):
    if keras_nlp is not None:
        try:
            preprocessor = keras_nlp.models.DebertaV3Preprocessor.from_preset(
                preset, sequence_length=sequence_length
            )
            return preprocessor(texts)
        except Exception:
            pass

    if AutoTokenizer is None:
        raise RuntimeError("No tokenizer backend available")

    tokenizer = AutoTokenizer.from_pretrained(HF_TOKENIZER_NAME, use_fast=True)
    encoded = tokenizer(
        list(texts),
        padding="max_length",
        truncation=True,
        max_length=sequence_length,
        return_tensors="np",
    )
    return {
        "token_ids": encoded["input_ids"],
        "padding_mask": encoded["attention_mask"],
    }


@keras.utils.register_keras_serializable(package="AES") if keras is not None else lambda x: x
class WeightedKappa(keras.metrics.Metric if keras is not None else object):  # type: ignore[misc]
    def __init__(self, num_classes=6, epsilon=1e-6):
        if keras is None:
            raise RuntimeError(f"Keras import failed: {_IMPORT_ERROR}")
        super().__init__(name="weighted_kappa")
        self.num_classes = num_classes
        self.epsilon = epsilon

        label_vec = keras.ops.arange(num_classes, dtype=keras.backend.floatx())
        self.row_label_vec = keras.ops.reshape(label_vec, [1, num_classes])
        self.col_label_vec = keras.ops.reshape(label_vec, [num_classes, 1])
        col_mat = keras.ops.tile(self.col_label_vec, [1, num_classes])
        row_mat = keras.ops.tile(self.row_label_vec, [num_classes, 1])
        self.weight_mat = (col_mat - row_mat) ** 2

        self.numerator = self.add_weight(name="numerator", initializer="zeros")
        self.denominator = self.add_weight(name="denominator", initializer="zeros")
        self.o_sum = self.add_weight(name="o_sum", initializer="zeros")
        self.e_sum = self.add_weight(name="e_sum", initializer="zeros")

    def update_state(self, y_true, y_pred, **kwargs):
        y_true = keras.ops.one_hot(
            keras.ops.sum(keras.ops.cast(y_true, dtype="int8"), axis=-1) - 1, 6
        )
        y_pred = keras.ops.one_hot(
            keras.ops.sum(keras.ops.cast(y_pred > 0.5, dtype="int8"), axis=-1) - 1, 6
        )
        y_true = keras.ops.cast(y_true, dtype=self.col_label_vec.dtype)
        y_pred = keras.ops.cast(y_pred, dtype=self.weight_mat.dtype)
        batch_size = keras.ops.shape(y_true)[0]

        cat_labels = keras.ops.matmul(y_true, self.col_label_vec)
        cat_label_mat = keras.ops.tile(cat_labels, [1, self.num_classes])
        row_label_mat = keras.ops.tile(self.row_label_vec, [batch_size, 1])

        weight = (cat_label_mat - row_label_mat) ** 2

        self.numerator.assign_add(keras.ops.sum(weight * y_pred))
        label_dist = keras.ops.sum(y_true, axis=0, keepdims=True)
        pred_dist = keras.ops.sum(y_pred, axis=0, keepdims=True)
        w_pred_dist = keras.ops.matmul(
            self.weight_mat, keras.ops.transpose(pred_dist, [1, 0])
        )
        self.denominator.assign_add(
            keras.ops.sum(keras.ops.matmul(label_dist, w_pred_dist))
        )

        self.o_sum.assign_add(keras.ops.sum(y_pred))
        self.e_sum.assign_add(
            keras.ops.sum(
                keras.ops.matmul(keras.ops.transpose(label_dist, [1, 0]), pred_dist)
            )
        )

    def result(self):
        return 1.0 - (
            keras.ops.divide_no_nan(self.numerator, self.denominator)
            * keras.ops.divide_no_nan(self.e_sum, self.o_sum)
        )

    def reset_state(self):
        self.numerator.assign(0)
        self.denominator.assign(0)
        self.o_sum.assign(0)
        self.e_sum.assign(0)


def _load_weights(model, weights_path: Path) -> str:
    if weights_path.exists():
        try:
            _load_weights_by_structure(model, weights_path)
            return f"Loaded weights from {weights_path}"
        except Exception:
            model.load_weights(str(weights_path), skip_mismatch=True)
            return f"Loaded weights from {weights_path} with skip_mismatch fallback"
    return f"Weights not found at {weights_path}; using pretrained backbone only"


def _is_input_layer(layer) -> bool:
    return keras is not None and isinstance(layer, keras.layers.InputLayer)


def _ordered_child_layers(layer):
    return [child for child in getattr(layer, "layers", []) if not _is_input_layer(child)]


def _ordered_weight_values(group) -> List[np.ndarray]:
    vars_group = group.get("vars")
    if vars_group is None:
        return []
    keys = list(vars_group.keys())
    keys.sort(key=lambda value: int(value) if str(value).isdigit() else str(value))
    return [np.array(vars_group[key]) for key in keys]


def _load_group_into_layer(layer, group) -> None:
    weight_values = _ordered_weight_values(group)
    if weight_values:
        layer.set_weights(weight_values)

    child_group_container = group.get("layers")
    if child_group_container is None:
        return

    child_layers = _ordered_child_layers(layer)
    child_groups = [child_group_container[key] for key in child_group_container.keys()]

    for child_layer, child_group in zip(child_layers, child_groups):
        _load_group_into_layer(child_layer, child_group)


def _load_weights_by_structure(model, weights_path: Path) -> None:
    with h5py.File(weights_path, "r") as handle:
        classifier_group = handle["layers"]["deberta_v3_classifier"]
        top_layers_group = classifier_group["layers"]
        top_layers = _ordered_child_layers(model)
        top_groups = [top_layers_group[key] for key in top_layers_group.keys()]

        for layer, group in zip(top_layers, top_groups):
            _load_group_into_layer(layer, group)


@lru_cache(maxsize=4)
def load_model(weights_path: str | Path = DEFAULT_WEIGHTS_PATH, preset: str = PRESET):
    weights_path = Path(weights_path)
    model = build_model(preset=preset)
    status = _load_weights(model, weights_path)
    return model, status


def _sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part for part in parts if part]


def _word_tokens(text: str) -> List[str]:
    return re.findall(r"[A-Za-z']+", text.lower())


def _transition_score(text: str) -> float:
    markers = [
        "however",
        "therefore",
        "because",
        "moreover",
        "furthermore",
        "in addition",
        "for example",
        "first",
        "second",
        "finally",
        "in conclusion",
    ]
    lower = text.lower()
    hits = sum(marker in lower for marker in markers)
    return min(1.0, hits / 4.0)


def _structure_score(text: str) -> float:
    sentences = _sentences(text)
    paragraph_count = max(1, text.count("\n\n") + 1)
    return float(np.clip((len(sentences) / 8.0 + paragraph_count / 4.0) / 2.0, 0.0, 1.0))


def _grammar_proxy(text: str) -> float:
    words = _word_tokens(text)
    if not words:
        return 0.0
    avg_word_len = sum(len(w) for w in words) / len(words)
    error_markers = text.count("!!") + text.count("??") + text.count("  ")
    base = 1.0 - min(1.0, error_markers / 8.0)
    return float(np.clip(base * (0.7 + min(avg_word_len, 8.0) / 20.0), 0.0, 1.0))


def _relevance_score(text: str, prompt: Optional[str]) -> float:
    if not prompt:
        return 0.6
    essay_words = set(_word_tokens(text))
    prompt_words = set(_word_tokens(prompt))
    if not essay_words or not prompt_words:
        return 0.0
    overlap = len(essay_words & prompt_words)
    return float(np.clip(overlap / max(10, len(prompt_words)), 0.0, 1.0))


def _content_score(text: str, prompt: Optional[str]) -> float:
    words = _word_tokens(text)
    sentences = _sentences(text)
    richness = min(1.0, len(set(words)) / max(1, len(words)))
    length_factor = min(1.0, len(words) / 250.0)
    relevance = _relevance_score(text, prompt)
    return float(np.clip(0.35 * richness + 0.35 * length_factor + 0.3 * relevance, 0.0, 1.0))


def _rubric_breakdown(text: str, prompt: Optional[str]) -> Dict[str, float]:
    return {
        "content": _content_score(text, prompt),
        "coherence": _transition_score(text),
        "grammar": _grammar_proxy(text),
        "relevance": _relevance_score(text, prompt),
        "structure": _structure_score(text),
    }


def _similarity_risk(text: str, prompt: Optional[str], reference_texts: Optional[Sequence[str]]) -> Tuple[str, float]:
    if not reference_texts:
        relevance = _relevance_score(text, prompt)
        if relevance > 0.45:
            return "low", relevance
        if relevance > 0.2:
            return "medium", relevance
        return "low", relevance

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    corpus = [text] + [item for item in reference_texts if item.strip()]
    if len(corpus) == 1:
        return "low", 0.0
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform(corpus)
    sims = cosine_similarity(tfidf[0:1], tfidf[1:]).ravel()
    best = float(np.max(sims)) if len(sims) else 0.0
    if best >= 0.65:
        return "high", best
    if best >= 0.35:
        return "medium", best
    return "low", best


def _score_to_band(score: int) -> str:
    bands = {
        6: "Excellent",
        5: "Strong",
        4: "Competent",
        3: "Developing",
        2: "Weak",
        1: "Very weak",
    }
    return bands.get(score, "Unknown")


def _confidence_from_preds(preds: np.ndarray) -> float:
    centered = np.abs(preds - 0.5)
    return float(np.clip(centered.mean() * 2.0, 0.0, 1.0))


def _decode_raw_score(preds: np.ndarray) -> int:
    return int(np.sum((preds > 0.5).astype(int), axis=-1).clip(1, 6)[0])


def _strengths_and_weaknesses(rubric: Dict[str, float]) -> Tuple[List[str], List[str]]:
    strengths = []
    weaknesses = []
    for name, value in rubric.items():
        if value >= 0.7:
            strengths.append(f"Strong {name}.")
        elif value <= 0.4:
            weaknesses.append(f"Improve {name}.")
    if not strengths:
        strengths.append("Clear essay submission with measurable signals for feedback.")
    if not weaknesses:
        weaknesses.append("Add a stronger thesis and more explicit evidence.")
    return strengths[:3], weaknesses[:3]


def _recommendations(weaknesses: Sequence[str]) -> List[str]:
    recommendations = []
    joined = " ".join(weaknesses).lower()
    if "grammar" in joined:
        recommendations.append("Use a grammar checklist before submission.")
    if "structure" in joined:
        recommendations.append("Add an introduction, body, and conclusion outline.")
    if "content" in joined or "relevance" in joined:
        recommendations.append("Add more concrete examples and topic-specific evidence.")
    if "coherence" in joined:
        recommendations.append("Use transitions like however, therefore, and for example.")
    if not recommendations:
        recommendations.append("Revise the draft once for clarity and flow.")
    return recommendations


def _feedback_text(score: int, confidence: float, rubric: Dict[str, float], similarity_risk: str) -> str:
    band = _score_to_band(score)
    low_parts = [name for name, value in rubric.items() if value < 0.45]
    high_parts = [name for name, value in rubric.items() if value >= 0.7]
    parts = [f"Overall band: {band} (score {score}/6).", f"Confidence: {confidence:.2f}."]
    if high_parts:
        parts.append("Strong areas: " + ", ".join(high_parts) + ".")
    if low_parts:
        parts.append("Needs work: " + ", ".join(low_parts) + ".")
    parts.append(f"Similarity risk: {similarity_risk}.")
    return " ".join(parts)


def _short_answer_breakdown(text: str) -> Dict[str, float]:
    words = _word_tokens(text)
    text_lower = text.lower()
    length_score = min(1.0, len(words) / 30.0)
    clarity_score = 0.9 if len(words) >= 10 and "?" not in text_lower else 0.65
    accuracy_score = 0.8 if any(tok in text_lower for tok in ["because", "therefore", "due to", "as a result", "for example"]) else 0.6
    completeness_score = min(1.0, len(words) / 40.0)
    return {
        "accuracy": float(np.clip(accuracy_score, 0.0, 1.0)),
        "completeness": float(np.clip(completeness_score, 0.0, 1.0)),
        "clarity": float(np.clip(clarity_score, 0.0, 1.0)),
    }


def _short_answer_recommendations(weaknesses: Sequence[str]) -> List[str]:
    recommendations = []
    joined = " ".join(weaknesses).lower()
    if "accuracy" in joined:
        recommendations.append("Check facts and use precise terminology.")
    if "completeness" in joined:
        recommendations.append("Expand the answer with one or two extra points.")
    if "clarity" in joined:
        recommendations.append("Use shorter sentences and clearer phrasing.")
    if not recommendations:
        recommendations.append("Review the answer for clarity and completeness before submission.")
    return recommendations


def evaluate_short_answer(
    text: str,
    reference_texts: Optional[Sequence[str]] = None,
    vectorizer_path: str | Path = DEFAULT_SHORT_ANSWER_VECTORIZER_PATH,
    model_path: str | Path = DEFAULT_SHORT_ANSWER_MODEL_PATH,
) -> EvaluationResult:
    text = (text or "").strip()
    if not text:
        raise ValueError("Short answer text cannot be empty.")

    model_status = "heuristic mode"
    raw_score = 0.0
    final_score = 2
    confidence = 0.0

    try:
        vectorizer_path = Path(vectorizer_path)
        model_path = Path(model_path)
        if not vectorizer_path.exists() or not model_path.exists():
            raise FileNotFoundError("Short answer model artifacts missing.")

        vectorizer = joblib.load(vectorizer_path)
        model = joblib.load(model_path)
        features = vectorizer.transform([text])
        preds = model.predict(features)
        probs = model.predict_proba(features) if hasattr(model, "predict_proba") else None
        raw_score = float(preds[0])
        final_score = int(np.clip(int(preds[0]) + 1, 1, 4))
        confidence = float(np.max(probs[0])) if probs is not None else 0.0
        model_status = f"short answer model loaded ({model_path})"
    except Exception as exc:
        model_status = f"short answer model unavailable, using heuristic fallback: {exc}"
        raw_score = 0.0
        final_score = 2
        confidence = 0.5

    rubric_breakdown = _short_answer_breakdown(text)
    similarity_risk, similarity_value = _similarity_risk(text, None, reference_texts)
    strengths, weaknesses = _strengths_and_weaknesses(rubric_breakdown)
    recommendations = _short_answer_recommendations(weaknesses)
    feedback = _feedback_text(final_score, confidence, rubric_breakdown, similarity_risk)

    return EvaluationResult(
        final_score=final_score,
        confidence=confidence,
        rubric_breakdown=rubric_breakdown,
        similarity_risk=f"{similarity_risk} ({similarity_value:.2f})",
        strengths=strengths,
        weaknesses=weaknesses,
        feedback=feedback,
        recommendations=recommendations,
        resources=_resources_for_weaknesses(weaknesses, assignment_type="Short Answer"),
        model_status=model_status,
        raw_score=raw_score,
    )


def _load_code_index(path: str | Path = DEFAULT_CODE_INDEX_PATH) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Code index not found at {path}")
    return joblib.load(path)


def _code_rubric_breakdown(text: str) -> Dict[str, float]:
    lines = [line for line in text.splitlines() if line.strip()]
    words = _word_tokens(text)
    text_lower = text.lower()
    comment_count = sum(1 for line in lines if line.strip().startswith(("#", "//", "/*", "*")))
    comment_ratio = min(1.0, comment_count / max(1, len(lines)))
    correctness = 0.8 if any(tok in text_lower for tok in ["return", "if ", "for ", "while ", "def ", "class "]) else 0.55
    readability = 0.7 + 0.2 * comment_ratio
    efficiency = 0.75 if len(lines) <= 80 else 0.55
    edge_cases = 0.8 if any(tok in text_lower for tok in ["try", "except", "if ", "else"]) else 0.45
    return {
        "correctness": float(np.clip(correctness, 0.0, 1.0)),
        "readability": float(np.clip(readability, 0.0, 1.0)),
        "efficiency": float(np.clip(efficiency, 0.0, 1.0)),
        "edge_cases": float(np.clip(edge_cases, 0.0, 1.0)),
    }


def _code_recommendations(weaknesses: Sequence[str]) -> List[str]:
    recommendations = []
    joined = " ".join(weaknesses).lower()
    if "correctness" in joined:
        recommendations.append("Add test cases and verify the logic end to end.")
    if "readability" in joined:
        recommendations.append("Format your code and add comments for key sections.")
    if "efficiency" in joined:
        recommendations.append("Consider algorithmic complexity and simplify loops.")
    if "edge_cases" in joined:
        recommendations.append("Handle invalid input and boundary conditions explicitly.")
    if not recommendations:
        recommendations.append("Review the code for clarity, structure, and edge-case handling.")
    return recommendations


def evaluate_code(
    text: str,
    reference_texts: Optional[Sequence[str]] = None,
    code_index_path: str | Path = DEFAULT_CODE_INDEX_PATH,
) -> EvaluationResult:
    text = (text or "").strip()
    if not text:
        raise ValueError("Code text cannot be empty.")

    model_status = "heuristic mode"
    raw_score = 0.0
    final_score = 3
    confidence = 0.0
    top_matches = []

    try:
        index_data = _load_code_index(code_index_path)
        vectorizer = index_data.get("vectorizer")
        matrix = index_data.get("matrix")
        corpus = index_data.get("corpus", [])
        if vectorizer is None or matrix is None:
            raise ValueError("Code index is missing required components.")

        features = vectorizer.transform([text])
        from sklearn.metrics.pairwise import cosine_similarity

        sims = cosine_similarity(features, matrix).ravel()
        max_sim = float(np.max(sims)) if sims.size else 0.0
        best_idx = int(np.argmax(sims)) if sims.size else -1
        top_matches = []
        if best_idx >= 0 and best_idx < len(corpus):
            top_matches = [corpus[best_idx][:320]]

        raw_score = max_sim * 6.0
        final_score = int(np.clip(round(max_sim * 6.0), 1, 6))
        confidence = float(np.clip(max_sim, 0.0, 1.0))
        similarity = "low"
        if max_sim >= 0.75:
            similarity = "high"
        elif max_sim >= 0.45:
            similarity = "medium"
        model_status = f"code index loaded ({code_index_path})"
    except Exception as exc:
        model_status = f"code index unavailable, using heuristic fallback: {exc}"
        raw_score = 0.0
        final_score = 3
        confidence = 0.55
        max_sim = 0.0
        similarity = "low"

    rubric_breakdown = _code_rubric_breakdown(text)
    strengths, weaknesses = _strengths_and_weaknesses(rubric_breakdown)
    recommendations = _code_recommendations(weaknesses)
    similarity_risk = similarity
    feedback = _feedback_text(final_score, confidence, rubric_breakdown, similarity_risk)

    result = EvaluationResult(
        final_score=final_score,
        confidence=confidence,
        rubric_breakdown=rubric_breakdown,
        similarity_risk=f"{similarity_risk} ({max_sim:.2f})",
        strengths=strengths,
        weaknesses=weaknesses,
        feedback=feedback,
        recommendations=recommendations,
        resources=_resources_for_weaknesses(weaknesses, assignment_type="Code"),
        model_status=model_status,
        raw_score=raw_score,
    )
    setattr(result, "top_matches", top_matches)
    return result


def evaluate_essay(
    text: str,
    prompt: Optional[str] = None,
    rubric: Optional[str] = None,
    reference_texts: Optional[Sequence[str]] = None,
    weights_path: str | Path = DEFAULT_WEIGHTS_PATH,
    preset: str = PRESET,
) -> EssayEvaluation:
    text = (text or "").strip()
    if not text:
        raise ValueError("Essay text cannot be empty.")

    model_status = "heuristic mode"
    raw_score = 0.0
    final_score = 3
    confidence = 0.0

    try:
        model, model_status = load_model(weights_path=weights_path, preset=preset)
        if keras is None:
            raise RuntimeError("Keras not available")

        tokenized = _tokenize_texts([text], preset=preset, sequence_length=DEFAULT_SEQUENCE_LENGTH)
        preds = model.predict(tokenized, verbose=0)
        raw_score = float(np.mean(preds))
        final_score = _decode_raw_score(preds)
        confidence = _confidence_from_preds(np.array(preds))
        model_status = f"model loaded successfully ({model_status})"
    except Exception as exc:
        model_status = f"model unavailable, using heuristic scoring: {exc}"
        rubric_map = _rubric_breakdown(text, prompt)
        raw_score = float(np.mean(list(rubric_map.values()))) * 6.0
        final_score = int(np.clip(round(raw_score), 1, 6))
        confidence = float(np.clip(np.mean(list(rubric_map.values())), 0.0, 1.0))
        rubric_breakdown = rubric_map
    else:
        rubric_breakdown = _rubric_breakdown(text, prompt)

    similarity_risk, similarity_value = _similarity_risk(text, prompt, reference_texts)
    strengths, weaknesses = _strengths_and_weaknesses(rubric_breakdown)
    recommendations = _recommendations(weaknesses)
    feedback = _feedback_text(final_score, confidence, rubric_breakdown, similarity_risk)

    if rubric:
        rubric_breakdown["rubric_match"] = min(1.0, len(rubric.strip()) / 2000.0)

    return EssayEvaluation(
        final_score=final_score,
        confidence=confidence,
        rubric_breakdown=rubric_breakdown,
        similarity_risk=f"{similarity_risk} ({similarity_value:.2f})",
        strengths=strengths,
        weaknesses=weaknesses,
        feedback=feedback,
        recommendations=recommendations,
        resources=_resources_for_weaknesses(weaknesses, assignment_type="Essay"),
        model_status=model_status,
        raw_score=raw_score,
    )
