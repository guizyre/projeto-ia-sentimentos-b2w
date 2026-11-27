"""
Módulo de avaliação experimental, análise de erros, ruído de rótulos,
desempenho por categoria e explicabilidade para inferência em tempo real.

Responsável: Hillary Maximino (Pessoa 3 – Avaliação Crítica, Ética e Interface Streamlit)
Projeto 2 – Classificação Automática de Sentimentos em Avaliações de Produtos
Disciplina: Inteligência Artificial (Unidade 3) – IFPB

Alinhado às Seções 2.5.5, 2.6 e 2.8 do edital:
- Calcula Acurácia, Precisão, Recall, F1 por classe e F1-Macro (métrica principal).
- Responde empiricamente às 5 perguntas obrigatórias da Seção 2.6 com dados reais do X_test:
  1. Melhor configuração (representação × classificador + ajuste de hiperparâmetros).
  2. Capacidade de detecção de avaliações negativas (Recall e F1 Negativo; padrão vs balanced).
  3. Erros mais frequentes (matriz de confusão, classe Neutra e avaliações mistas).
  4. Ruído introduzido pela conversão de notas em sentimentos (casos extremos e recomendações conflitantes).
  5. Estabilidade do desempenho entre diferentes categorias de produtos (site_category_lv1).
- Fornece explicação por token/bigrama para novas avaliações no simulador Streamlit (Aba 3).
- Nunca lê, armazena ou exibe colunas pessoais/demográficas (Seção 2.8).
"""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.data_loader import ID_TO_SENTIMENT, PROJECT_ROOT, SENTIMENT_TO_ID
from src.models import (
    CLASSIFIERS,
    LABEL_NAMES,
    LABELS,
    MODELS_DIR,
    REPORTS_DIR,
    REPRESENTATIONS,
    SEED,
    get_split,
    load_artifacts,
    make_vectorizer,
)
from src.preprocessing import preprocess_text

# ---------------------------------------------------------------------------
# Constantes visuais e rótulos amigáveis (padrão do projeto)
# ---------------------------------------------------------------------------

FIGURES_DIR = REPORTS_DIR / "figures"
MODEL_RESULTS_PATH = REPORTS_DIR / "model_results.json"
EVALUATION_SUMMARY_PATH = REPORTS_DIR / "evaluation_summary.json"

REP_LABELS: Dict[str, str] = {
    "bow_uni": "BoW (Unigramas)",
    "bow_bi": "BoW (Uni+Bigramas)",
    "tfidf_uni": "TF-IDF (Unigramas)",
    "tfidf_bi": "TF-IDF (Uni+Bigramas)",
}

CLF_LABELS: Dict[str, str] = {
    "nb": "Naive Bayes",
    "logreg": "Regressão Logística",
    "logreg_bal": "Regressão Logística (balanced)",
    "svm_bal": "Linear SVM (balanced)",
}

PALETTE_SENTIMENT: Dict[str, str] = {
    "Negativo": "#DC2626",  # Vermelho
    "Neutro": "#D97706",    # Âmbar
    "Positivo": "#059669",  # Verde
}

# Padrões léxicos para categorização interpretável dos erros reais do X_test
# (grupos não capturantes `(?:...)` evitam UserWarning em pandas str.contains)
_PRAISE_RE = re.compile(
    r"\b(?:ótimo|ótima|otimo|otima|excelente|bom|boa|gostei|amei|adorei|lindo|linda|perfeito|perfeita|recomendo)\b",
    flags=re.IGNORECASE,
)
_DELIVERY_OR_ISSUE_RE = re.compile(
    r"\b(?:entrega|entregue|entregar|prazo|atraso|atrasou|demora|demorou|transportadora|correios|"
    r"frete|defeito|quebrado|quebrada|riscado|arranhado|faltando|errado|errada|embalagem|amassado|avariad[oa])\b",
    flags=re.IGNORECASE,
)
_CONTRAST_RE = re.compile(
    r"\b(?:mas|porém|porem|contudo|entretanto|todavia|só que|so que|pena que|infelizmente)\b",
    flags=re.IGNORECASE,
)
_NOT_TESTED_RE = re.compile(
    r"\b(?:ainda não|ainda nao|não testei|nao testei|não abri|nao abri|não usei|nao usei|"
    r"não recebi|nao recebi|não chegou|nao chegou|não instalei|nao instalei)\b",
    flags=re.IGNORECASE,
)
# Filtro apenas para seleção didática de exemplos exibidos (sem alterar contagens quantitativas)
_DISPLAY_NOISE_FILTER_RE = re.compile(
    r"(?:\.{5,}|(.)\1{5,}|\b(?:poha|porra|merda|caralho|bosta|fdp|puta)\b)",
    flags=re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Carregamento de artefatos e métricas
# ---------------------------------------------------------------------------


def load_model_results(path: Path | str = MODEL_RESULTS_PATH) -> Dict[str, Any]:
    """Carrega o JSON com os resultados consolidados da etapa de modelagem (Pessoa 2)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Arquivo de resultados não encontrado em {p}. "
            "Execute `python -m src.models` para gerar reports/model_results.json."
        )
    return json.loads(p.read_text(encoding="utf-8"))


def load_best_pipeline() -> Pipeline:
    """Carrega o pipeline final vencedor (models_saved/best_pipeline.joblib)."""
    path = MODELS_DIR / "best_pipeline.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Pipeline final não encontrado em: {path}")
    return joblib.load(path)


def load_tuned_pipeline(clf: str) -> Pipeline:
    """Carrega um pipeline ajustado por GridSearchCV (models_saved/tuned__<clf>.joblib)."""
    path = MODELS_DIR / f"tuned__{clf}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Pipeline ajustado não encontrado em: {path}")
    return joblib.load(path)


def extract_vec_and_clf(model_obj: Any) -> Tuple[Any, Any]:
    """Extrai `(vectorizer, classifier)` de um Pipeline ou tupla `(vec, clf)`."""
    if isinstance(model_obj, Pipeline):
        return model_obj.named_steps["vec"], model_obj.named_steps["clf"]
    if isinstance(model_obj, tuple) and len(model_obj) == 2:
        return model_obj[0], model_obj[1]
    raise TypeError("Esperado um sklearn.pipeline.Pipeline ou tupla (vectorizer, model).")


def compute_detailed_metrics(
    y_true: np.ndarray | pd.Series | List[int],
    y_pred: np.ndarray | pd.Series | List[int],
    labels: List[int] = LABELS,
    label_names: List[str] = LABEL_NAMES,
) -> Dict[str, Any]:
    """
    Calcula o conjunto completo de métricas exigidas na Seção 2.5.5:
    Acurácia, F1-Macro (destaque), F1-Weighted, Precisão/Recall/F1/Suporte por classe
    e Matriz de Confusão absoluta e normalizada por classe real (recall).
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, labels=labels, zero_division=0
    )
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=labels)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(
        cm.astype(float),
        row_sums,
        out=np.zeros_like(cm, dtype=float),
        where=row_sums != 0,
    )

    out: Dict[str, Any] = {
        "n_samples": int(len(y_true_arr)),
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "f1_macro": float(f1_score(y_true_arr, y_pred_arr, labels=labels, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true_arr, y_pred_arr, labels=labels, average="weighted", zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_normalized": np.round(cm_norm, 4).tolist(),
        "per_class": {},
    }
    for i, name in enumerate(label_names):
        out[f"precision_{name}"] = float(prec[i])
        out[f"recall_{name}"] = float(rec[i])
        out[f"f1_{name}"] = float(f1[i])
        out[f"support_{name}"] = int(sup[i])
        out["per_class"][name] = {
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1[i]),
            "support": int(sup[i]),
        }
    return out


def build_comparison_dataframe(model_results: Dict[str, Any]) -> pd.DataFrame:
    """
    Constrói tabela comparativa unificada com as 16 combinações padrão e as 4
    configurações com ajuste de hiperparâmetros (GridSearchCV em X_train),
    ordenada por F1-Macro decrescente.
    """
    rows: List[Dict[str, Any]] = []
    final_cfg = model_results.get("final") or {}
    final_key = (
        final_cfg.get("representation"),
        final_cfg.get("classifier"),
        json.dumps(final_cfg.get("best_params", {}), sort_keys=True),
    )
    feat_by_rep: Dict[str, int] = {
        r["representation"]: int(r.get("n_features", 0))
        for r in model_results.get("results", [])
    }

    for r in model_results.get("results", []):
        rows.append(
            {
                "etapa": "Grade Padrão",
                "representation": r["representation"],
                "representacao_label": REP_LABELS.get(r["representation"], r["representation"]),
                "classifier": r["classifier"],
                "classificador_label": CLF_LABELS.get(r["classifier"], r["classifier"]),
                "hiperparametros": "padrão",
                "n_features": int(r.get("n_features", 0)),
                "cv_f1_macro": np.nan,
                "accuracy": float(r["accuracy"]),
                "f1_macro": float(r["f1_macro"]),
                "precision_Negativo": float(r["precision_Negativo"]),
                "recall_Negativo": float(r["recall_Negativo"]),
                "f1_Negativo": float(r["f1_Negativo"]),
                "precision_Neutro": float(r["precision_Neutro"]),
                "recall_Neutro": float(r["recall_Neutro"]),
                "f1_Neutro": float(r["f1_Neutro"]),
                "precision_Positivo": float(r["precision_Positivo"]),
                "recall_Positivo": float(r["recall_Positivo"]),
                "f1_Positivo": float(r["f1_Positivo"]),
                "is_final": False,
            }
        )

    for r in model_results.get("tuned", []):
        params = r.get("best_params", {})
        params_str = ", ".join(f"{k.replace('clf__', '')}={v}" for k, v in params.items())
        is_final = (
            r["representation"],
            r["classifier"],
            json.dumps(params, sort_keys=True),
        ) == final_key
        rows.append(
            {
                "etapa": "Ajustado (CV)",
                "representation": r["representation"],
                "representacao_label": REP_LABELS.get(r["representation"], r["representation"]),
                "classifier": r["classifier"],
                "classificador_label": CLF_LABELS.get(r["classifier"], r["classifier"]),
                "hiperparametros": params_str,
                "n_features": int(r.get("n_features") or feat_by_rep.get(r["representation"], 0)),
                "cv_f1_macro": float(r["cv_f1_macro"]),
                "accuracy": float(r["accuracy"]),
                "f1_macro": float(r["f1_macro"]),
                "precision_Negativo": float(r["precision_Negativo"]),
                "recall_Negativo": float(r["recall_Negativo"]),
                "f1_Negativo": float(r["f1_Negativo"]),
                "precision_Neutro": float(r["precision_Neutro"]),
                "recall_Neutro": float(r["recall_Neutro"]),
                "f1_Neutro": float(r["f1_Neutro"]),
                "precision_Positivo": float(r["precision_Positivo"]),
                "recall_Positivo": float(r["recall_Positivo"]),
                "f1_Positivo": float(r["f1_Positivo"]),
                "is_final": is_final,
            }
        )

    df_comp = pd.DataFrame(rows)
    df_comp = df_comp.sort_values(
        by=["f1_macro", "accuracy"], ascending=[False, False]
    ).reset_index(drop=True)
    df_comp.insert(0, "rank", np.arange(1, len(df_comp) + 1))
    return df_comp


# ---------------------------------------------------------------------------
# Predição vetorizada e explicabilidade por token (Aba 3 do Streamlit)
# ---------------------------------------------------------------------------


def predict_scores_batch(
    model_obj: Any,
    texts_clean: pd.Series | List[str],
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Executa predição vetorizada em lote retornando `(y_pred, scores, is_calibrated_proba)`.

    Para `LinearSVC` (que não implementa `predict_proba`), aplica softmax sobre
    `decision_function` como pontuação relativa de confiança (`is_calibrated_proba=False`).
    """
    vec, clf = extract_vec_and_clf(model_obj)
    X_mat = vec.transform(texts_clean)
    if hasattr(clf, "predict_proba"):
        scores = clf.predict_proba(X_mat)
        is_proba = True
    else:
        raw = clf.decision_function(X_mat)
        if raw.ndim == 1:
            raw = np.column_stack([-raw, raw])
        shifted = raw - raw.max(axis=1, keepdims=True)
        exp_s = np.exp(shifted)
        scores = exp_s / exp_s.sum(axis=1, keepdims=True)
        is_proba = False
    y_pred = np.argmax(scores, axis=1).astype(int)
    return y_pred, scores, is_proba


def explain_prediction(
    vectorizer: Any,
    model: Any,
    texto_tratado: str,
    top_k: int = 10,
) -> Dict[str, Any]:
    """
    Calcula quais termos/bigramas presentes no texto mais influenciaram a
    predição do classificador (compatível com LogisticRegression, LinearSVC e MultinomialNB).
    """
    x = vectorizer.transform([texto_tratado])
    nz_cols = x.nonzero()[1]
    feature_names = vectorizer.get_feature_names_out()

    # Obter predição e escores
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(x)[0]
        score_kind = "probabilidade"
    else:
        raw = model.decision_function(x)[0]
        e = np.exp(raw - np.max(raw))
        probs = e / e.sum()
        score_kind = "confianca_softmax_margem"

    pred_id = int(np.argmax(probs))
    pred_label = ID_TO_SENTIMENT[pred_id]

    if len(nz_cols) == 0:
        return {
            "pred_id": pred_id,
            "sentimento": pred_label,
            "score_kind": score_kind,
            "probabilidades": {n: float(p) for n, p in zip(LABEL_NAMES, probs)},
            "tokens_matched": 0,
            "top_tokens_predicted": [],
            "all_token_contributions": [],
        }

    # Matriz de pesos por classe (3 x n_features)
    if hasattr(model, "coef_"):
        weights = model.coef_
        method_note = "contribuição linear (valor_feature × peso_coef)"
    elif hasattr(model, "feature_log_prob_"):
        # Centraliza log-probabilidades em torno da média entre classes para mostrar sinal relativo
        lp = model.feature_log_prob_
        weights = lp - lp.mean(axis=0, keepdims=True)
        method_note = "log-verossimilhança relativa à média entre classes"
    else:
        weights = None
        method_note = "indisponível para este estimador"

    contributions: List[Dict[str, Any]] = []
    if weights is not None:
        for col in nz_cols:
            val = float(x[0, col])
            term = str(feature_names[col])
            c_neg = float(val * weights[0, col])
            c_neu = float(val * weights[1, col])
            c_pos = float(val * weights[2, col])
            c_pred = float(val * weights[pred_id, col])
            contributions.append(
                {
                    "token": term,
                    "valor_feature": round(val, 4),
                    "contrib_Negativo": round(c_neg, 4),
                    "contrib_Neutro": round(c_neu, 4),
                    "contrib_Positivo": round(c_pos, 4),
                    "contrib_classe_prevista": round(c_pred, 4),
                }
            )

    contributions_sorted = sorted(
        contributions,
        key=lambda d: abs(d["contrib_classe_prevista"]),
        reverse=True,
    )

    return {
        "pred_id": pred_id,
        "sentimento": pred_label,
        "score_kind": score_kind,
        "method_note": method_note,
        "probabilidades": {n: float(p) for n, p in zip(LABEL_NAMES, probs)},
        "tokens_matched": int(len(nz_cols)),
        "top_tokens_predicted": contributions_sorted[:top_k],
        "all_token_contributions": contributions_sorted,
    }


def predict_single_review(
    text_raw: str,
    model_obj: Any,
    top_k: int = 10,
) -> Dict[str, Any]:
    """
    Pipeline completo de inferência para uma nova avaliação digitada pelo usuário:
    pré-processamento (comparando com NLTK sem proteção) + predição + explicação dos tokens.
    """
    vec, clf = extract_vec_and_clf(model_obj)
    texto_tratado = preprocess_text(text_raw)
    texto_sem_protecao = preprocess_text(
        text_raw,
        remove_stopwords=True,
        keep_negations=False,
        keep_modifiers=False,
    )
    explanation = explain_prediction(vec, clf, texto_tratado, top_k=top_k)
    return {
        "texto_original": text_raw,
        "texto_tratado": texto_tratado,
        "texto_nltk_sem_protecao": texto_sem_protecao,
        **explanation,
    }


# ---------------------------------------------------------------------------
# Avaliação detalhada no conjunto de teste (Perguntas 1 a 5 — Seção 2.6)
# ---------------------------------------------------------------------------


def build_test_predictions_df(
    df: pd.DataFrame,
    X_test: pd.Series,
    y_test: pd.Series,
    model_obj: Any = None,
) -> pd.DataFrame:
    """
    Gera DataFrame alinhado a `X_test` com predições, confiança e metadados
    estritamente não pessoais (`site_category_lv1`, `overall_rating`, `recommend_to_a_friend`).
    """
    if model_obj is None:
        model_obj = load_best_pipeline()

    y_pred, scores, is_proba = predict_scores_batch(model_obj, X_test)

    df_test = df.loc[X_test.index].copy()
    safe_cols = [
        "review_text_original",
        "review_text_clean",
        "overall_rating",
        "sentiment",
        "sentiment_id",
        "site_category_lv1",
        "recommend_to_a_friend",
        "word_count",
        "char_count",
    ]
    available_safe = [c for c in safe_cols if c in df_test.columns]
    df_out = df_test[available_safe].copy()
    df_out = df_out.rename(
        columns={
            "sentiment": "sentiment_real",
            "sentiment_id": "sentiment_id_real",
        }
    )
    df_out["sentiment_id_pred"] = y_pred
    df_out["sentiment_pred"] = [ID_TO_SENTIMENT[int(i)] for i in y_pred]
    df_out["correct"] = df_out["sentiment_id_real"] == df_out["sentiment_id_pred"]
    df_out["prob_Negativo"] = scores[:, 0]
    df_out["prob_Neutro"] = scores[:, 1]
    df_out["prob_Positivo"] = scores[:, 2]
    df_out["confidence"] = scores.max(axis=1)
    df_out["is_calibrated_proba"] = is_proba
    return df_out


def analyze_error_breakdown(df_pred: pd.DataFrame) -> Dict[str, Any]:
    """
    Pergunta 3 (Seção 2.6) — Quantifica todos os erros do conjunto de teste
    por par `(Real -> Previsto)`, destacando o peso da classe Neutra e dos
    erros extremos (Negativo <-> Positivo).
    """
    total = int(len(df_pred))
    df_err = df_pred[~df_pred["correct"]].copy()
    n_errors = int(len(df_err))
    error_rate_pct = round(100.0 * n_errors / total, 2) if total else 0.0

    pairs: List[Dict[str, Any]] = []
    for real_lbl in LABEL_NAMES:
        for pred_lbl in LABEL_NAMES:
            if real_lbl == pred_lbl:
                continue
            sub = df_err[
                (df_err["sentiment_real"] == real_lbl)
                & (df_err["sentiment_pred"] == pred_lbl)
            ]
            cnt = int(len(sub))
            pairs.append(
                {
                    "real": real_lbl,
                    "previsto": pred_lbl,
                    "par": f"{real_lbl} → {pred_lbl}",
                    "quantidade": cnt,
                    "pct_dos_erros": round(100.0 * cnt / n_errors, 2) if n_errors else 0.0,
                    "pct_do_teste": round(100.0 * cnt / total, 2) if total else 0.0,
                    "confianca_media": round(float(sub["confidence"].mean()), 4) if cnt else 0.0,
                }
            )
    pairs.sort(key=lambda d: d["quantidade"], reverse=True)

    involving_neutral = int(
        (
            (df_err["sentiment_real"] == "Neutro")
            | (df_err["sentiment_pred"] == "Neutro")
        ).sum()
    )
    extreme_flips = int(
        (
            ((df_err["sentiment_real"] == "Negativo") & (df_err["sentiment_pred"] == "Positivo"))
            | ((df_err["sentiment_real"] == "Positivo") & (df_err["sentiment_pred"] == "Negativo"))
        ).sum()
    )

    # Heurística de avaliações mistas entre os erros (elogio + queixa de entrega/defeito ou conectivo adversativo)
    texts = df_err["review_text_original"].fillna("").astype(str)
    mixed_mask = (
        texts.str.contains(_PRAISE_RE)
        & (texts.str.contains(_DELIVERY_OR_ISSUE_RE) | texts.str.contains(_CONTRAST_RE))
    )
    mixed_count = int(mixed_mask.sum())

    # Extensão média de erros vs acertos
    mean_words_correct = round(float(df_pred.loc[df_pred["correct"], "word_count"].mean()), 2)
    mean_words_error = round(float(df_err["word_count"].mean()), 2)

    return {
        "n_test": total,
        "n_correct": total - n_errors,
        "n_errors": n_errors,
        "error_rate_pct": error_rate_pct,
        "errors_involving_neutral": involving_neutral,
        "errors_involving_neutral_pct": round(100.0 * involving_neutral / n_errors, 2) if n_errors else 0.0,
        "extreme_polarity_errors": extreme_flips,
        "extreme_polarity_errors_pct": round(100.0 * extreme_flips / n_errors, 2) if n_errors else 0.0,
        "mixed_sentiment_errors_estimated": mixed_count,
        "mixed_sentiment_errors_pct": round(100.0 * mixed_count / n_errors, 2) if n_errors else 0.0,
        "mean_words_correct": mean_words_correct,
        "mean_words_error": mean_words_error,
        "pairs": pairs,
    }


def _row_to_example_dict(row: pd.Series, tag: str) -> Dict[str, Any]:
    """Converte uma linha de `df_pred` em dicionário seguro para JSON/relatório/Streamlit."""
    return {
        "categoria_erro": tag,
        "overall_rating": int(row["overall_rating"]),
        "sentiment_real": str(row["sentiment_real"]),
        "sentiment_pred": str(row["sentiment_pred"]),
        "confidence": round(float(row["confidence"]), 4),
        "prob_Negativo": round(float(row["prob_Negativo"]), 4),
        "prob_Neutro": round(float(row["prob_Neutro"]), 4),
        "prob_Positivo": round(float(row["prob_Positivo"]), 4),
        "site_category_lv1": str(row["site_category_lv1"]),
        "recommend_to_a_friend": str(row.get("recommend_to_a_friend", "N/A")),
        "review_text_original": str(row["review_text_original"]),
        "review_text_clean": str(row["review_text_clean"]),
    }


def extract_error_examples(
    df_pred: pd.DataFrame,
    n_per_group: int = 4,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Pergunta 3 (Seção 2.6) — Extrai exemplos reais e representativos de erros do `X_test`
    (sem dados pessoais) organizados em categorias interpretáveis:
    - avaliações mistas (produto bom vs entrega/defeito ruim)
    - confusões envolvendo a classe Neutra (Neutro -> Positivo, Neutro -> Negativo, Positivo -> Neutro)
    - falsos positivos e falsos negativos extremos.
    """
    df_err = df_pred[~df_pred["correct"]].copy()
    # Evitar textos excessivamente curtos/gigantes ou com spam de pontos na exibição didática
    txt_col = df_err["review_text_original"].fillna("").astype(str)
    len_ok = (
        txt_col.str.len().between(45, 280)
        & (df_err["word_count"] >= 5)
        & (~txt_col.map(lambda s: bool(_DISPLAY_NOISE_FILTER_RE.search(s))))
    )
    pool = df_err[len_ok].copy()

    used_idx: set = set()

    def _pick(mask: pd.Series, tag: str, limit: int, sort_col: str = "confidence") -> List[Dict[str, Any]]:
        sub = pool[mask & (~pool.index.isin(used_idx))].sort_values(sort_col, ascending=False)
        selected = sub.head(limit)
        used_idx.update(selected.index.tolist())
        return [_row_to_example_dict(row, tag) for _, row in selected.iterrows()]

    texts = pool["review_text_original"].fillna("").astype(str)
    mixed_mask = (
        texts.str.contains(_PRAISE_RE)
        & texts.str.contains(_DELIVERY_OR_ISSUE_RE)
        & texts.str.contains(_CONTRAST_RE)
    )

    examples = {
        "avaliacoes_mistas_produto_vs_entrega": _pick(
            mixed_mask,
            "Avaliação mista (produto vs. entrega/ressalva)",
            n_per_group,
        ),
        "neutro_previsto_como_positivo": _pick(
            (pool["sentiment_real"] == "Neutro") & (pool["sentiment_pred"] == "Positivo"),
            "Neutro confundido com Positivo",
            n_per_group,
        ),
        "neutro_previsto_como_negativo": _pick(
            (pool["sentiment_real"] == "Neutro") & (pool["sentiment_pred"] == "Negativo"),
            "Neutro confundido com Negativo",
            n_per_group,
        ),
        "positivo_previsto_como_neutro": _pick(
            (pool["sentiment_real"] == "Positivo") & (pool["sentiment_pred"] == "Neutro"),
            "Positivo confundido com Neutro",
            n_per_group,
        ),
        "falso_positivo_extremo_neg_para_pos": _pick(
            (pool["sentiment_real"] == "Negativo") & (pool["sentiment_pred"] == "Positivo"),
            "Falso Positivo extremo (Negativo → Positivo)",
            n_per_group,
        ),
        "falso_negativo_extremo_pos_para_neg": _pick(
            (pool["sentiment_real"] == "Positivo") & (pool["sentiment_pred"] == "Negativo"),
            "Falso Negativo extremo (Positivo → Negativo)",
            n_per_group,
        ),
    }
    return examples


def analyze_label_noise(
    df_pred: pd.DataFrame,
    top_n: int = 6,
) -> Dict[str, Any]:
    """
    Pergunta 4 (Seção 2.6) — Investiga quantitativa e qualitativamente como a
    regra de conversão de notas (1–5) em sentimentos introduz ruído nos rótulos:
    - Nota 1 (ou 1–2) em que o modelo previu Positivo com alta confiança;
    - Nota 5 (ou 4–5) em que o modelo previu Negativo com alta confiança;
    - Taxa de contradição com `recommend_to_a_friend` no teste inteiro vs. nos erros extremos;
    - Incidência de avaliações "ainda não testei / não abri / não recebi" com nota alta.
    """
    rec = df_pred["recommend_to_a_friend"].fillna("").astype(str)

    # Taxa geral de contradição nota vs recomendação no conjunto de teste
    neg_mask = df_pred["sentiment_real"] == "Negativo"
    pos_mask = df_pred["sentiment_real"] == "Positivo"

    neg_rec_yes_all = int((neg_mask & (rec == "Yes")).sum())
    pos_rec_no_all = int((pos_mask & (rec == "No")).sum())

    # Subconjuntos de erros extremos (Negativo -> Positivo e Positivo -> Negativo)
    neg_pred_pos = df_pred[neg_mask & (df_pred["sentiment_pred"] == "Positivo")].copy()
    pos_pred_neg = df_pred[pos_mask & (df_pred["sentiment_pred"] == "Negativo")].copy()

    neg_pred_pos_rec_yes = int((neg_pred_pos["recommend_to_a_friend"] == "Yes").sum())
    pos_pred_neg_rec_no = int((pos_pred_neg["recommend_to_a_friend"] == "No").sum())

    # Casos específicos com nota 1 previstos como Positivo e nota 5 previstos como Negativo
    nota1_pred_pos = df_pred[
        (df_pred["overall_rating"] == 1) & (df_pred["sentiment_pred"] == "Positivo")
    ].sort_values("prob_Positivo", ascending=False)

    nota5_pred_neg = df_pred[
        (df_pred["overall_rating"] == 5) & (df_pred["sentiment_pred"] == "Negativo")
    ].sort_values("prob_Negativo", ascending=False)

    # Avaliações com nota 4-5 contendo "ainda não / não testei / não recebi / não abri"
    pos_not_tested = df_pred[
        pos_mask & df_pred["review_text_original"].str.contains(_NOT_TESTED_RE, na=False)
    ]
    pos_not_tested_pred_neg = pos_not_tested[pos_not_tested["sentiment_pred"] == "Negativo"]

    # Exemplos de nota 1 previstos como Positivo e nota 5 previstos como Negativo (legíveis para exibição)
    ex_nota1_pool = nota1_pred_pos[
        nota1_pred_pos["review_text_original"].str.len().between(25, 280)
        & (~nota1_pred_pos["review_text_original"].fillna("").map(lambda s: bool(_DISPLAY_NOISE_FILTER_RE.search(s))))
    ]
    ex_nota5_pool = nota5_pred_neg[
        nota5_pred_neg["review_text_original"].str.len().between(25, 280)
        & (~nota5_pred_neg["review_text_original"].fillna("").map(lambda s: bool(_DISPLAY_NOISE_FILTER_RE.search(s))))
    ]

    return {
        "n_test_negativo": int(neg_mask.sum()),
        "n_test_positivo": int(pos_mask.sum()),
        "neg_recommend_yes_all_test": neg_rec_yes_all,
        "neg_recommend_yes_all_test_pct": round(100.0 * neg_rec_yes_all / int(neg_mask.sum()), 2),
        "pos_recommend_no_all_test": pos_rec_no_all,
        "pos_recommend_no_all_test_pct": round(100.0 * pos_rec_no_all / int(pos_mask.sum()), 2),
        "extreme_neg_pred_pos_total": int(len(neg_pred_pos)),
        "extreme_neg_pred_pos_recommend_yes": neg_pred_pos_rec_yes,
        "extreme_neg_pred_pos_recommend_yes_pct": round(
            100.0 * neg_pred_pos_rec_yes / len(neg_pred_pos), 2
        ) if len(neg_pred_pos) else 0.0,
        "extreme_pos_pred_neg_total": int(len(pos_pred_neg)),
        "extreme_pos_pred_neg_recommend_no": pos_pred_neg_rec_no,
        "extreme_pos_pred_neg_recommend_no_pct": round(
            100.0 * pos_pred_neg_rec_no / len(pos_pred_neg), 2
        ) if len(pos_pred_neg) else 0.0,
        "nota1_pred_pos_total": int(len(nota1_pred_pos)),
        "nota5_pred_neg_total": int(len(nota5_pred_neg)),
        "pos_not_tested_or_received_count": int(len(pos_not_tested)),
        "pos_not_tested_pred_neg_count": int(len(pos_not_tested_pred_neg)),
        "examples_nota1_pred_positivo": [
            _row_to_example_dict(row, "Ruído de rótulo (Nota 1 → Previsto Positivo)")
            for _, row in ex_nota1_pool.head(top_n).iterrows()
        ],
        "examples_nota5_pred_negativo": [
            _row_to_example_dict(row, "Ruído de rótulo / Ironia (Nota 5 → Previsto Negativo)")
            for _, row in ex_nota5_pool.head(top_n).iterrows()
        ],
    }


def evaluate_by_category(
    df_pred: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Pergunta 5 (Seção 2.6) — Avalia o modelo separadamente por categoria de produto
    (`site_category_lv1`) nas Top `top_n` categorias por volume no conjunto de teste
    (inclui obrigatoriamente as Top 5 + `Móveis` + `Informática e Acessórios`).
    """
    counts = df_pred["site_category_lv1"].value_counts()
    top_cats = counts.head(top_n).index.tolist()
    if "Móveis" in counts.index and "Móveis" not in top_cats:
        top_cats.append("Móveis")

    rows: List[Dict[str, Any]] = []
    for cat in top_cats:
        sub = df_pred[df_pred["site_category_lv1"] == cat]
        m = compute_detailed_metrics(sub["sentiment_id_real"], sub["sentiment_id_pred"])
        n = len(sub)
        rows.append(
            {
                "categoria": cat,
                "n_test": int(n),
                "pct_negativo": round(100.0 * (sub["sentiment_real"] == "Negativo").mean(), 1),
                "pct_neutro": round(100.0 * (sub["sentiment_real"] == "Neutro").mean(), 1),
                "pct_positivo": round(100.0 * (sub["sentiment_real"] == "Positivo").mean(), 1),
                "accuracy": round(m["accuracy"], 4),
                "f1_macro": round(m["f1_macro"], 4),
                "f1_Negativo": round(m["f1_Negativo"], 4),
                "f1_Neutro": round(m["f1_Neutro"], 4),
                "f1_Positivo": round(m["f1_Positivo"], 4),
                "recall_Negativo": round(m["recall_Negativo"], 4),
                "recall_Neutro": round(m["recall_Neutro"], 4),
                "recall_Positivo": round(m["recall_Positivo"], 4),
            }
        )

    return pd.DataFrame(rows)


def evaluate_binary_scenario(
    X_train: pd.Series,
    y_train: pd.Series,
    X_test: pd.Series,
    y_test: pd.Series,
    c_val: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Experimento complementar (Funcionalidade 2 — Seção 2.4.2):
    Compara a regra padrão (3 classes: 1–2 Negativo, 3 Neutro, 4–5 Positivo)
    com o cenário binário (1–2 Negativo vs 4–5 Positivo, descartando a nota 3),
    treinando `TF-IDF uni+bi + LinearSVC(balanced)` estritamente sobre
    `X_train` (sem data leakage) e avaliando em `X_test`.
    """
    if c_val is None:
        try:
            mr = load_model_results()
            c_val = float((mr.get("final") or {}).get("best_params", {}).get("clf__C", 0.1))
        except Exception:
            c_val = 0.1

    tr_mask = y_train != 1
    te_mask = y_test != 1

    X_tr_bin = X_train[tr_mask]
    y_tr_bin = (y_train[tr_mask] == 2).astype(int)  # 0=Negativo, 1=Positivo
    X_te_bin = X_test[te_mask]
    y_te_bin = (y_test[te_mask] == 2).astype(int)

    pipe_bin = Pipeline(
        [
            ("vec", make_vectorizer("tfidf_bi")),
            ("clf", LinearSVC(C=c_val, class_weight="balanced", random_state=SEED)),
        ]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe_bin.fit(X_tr_bin, y_tr_bin)

    y_pred_bin = pipe_bin.predict(X_te_bin)
    metrics_bin = compute_detailed_metrics(
        y_te_bin,
        y_pred_bin,
        labels=[0, 1],
        label_names=["Negativo", "Positivo"],
    )
    return {
        "n_train_binary": int(len(X_tr_bin)),
        "n_test_binary": int(len(X_te_bin)),
        "discarded_neutral_train": int((~tr_mask).sum()),
        "discarded_neutral_test": int((~te_mask).sum()),
        "accuracy": round(metrics_bin["accuracy"], 4),
        "f1_macro": round(metrics_bin["f1_macro"], 4),
        "precision_Negativo": round(metrics_bin["precision_Negativo"], 4),
        "recall_Negativo": round(metrics_bin["recall_Negativo"], 4),
        "f1_Negativo": round(metrics_bin["f1_Negativo"], 4),
        "precision_Positivo": round(metrics_bin["precision_Positivo"], 4),
        "recall_Positivo": round(metrics_bin["recall_Positivo"], 4),
        "f1_Positivo": round(metrics_bin["f1_Positivo"], 4),
        "confusion_matrix": metrics_bin["confusion_matrix"],
    }


# ---------------------------------------------------------------------------
# Geração de gráficos em alta resolução (300 DPI)
# ---------------------------------------------------------------------------


def _setup_plot_style() -> None:
    matplotlib.use("Agg")
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"


def plot_confusion_matrices_comparison(
    model_results: Dict[str, Any],
    output_path: Path = FIGURES_DIR / "12_matrizes_confusao_comparacao.png",
) -> Path:
    """
    Figura 12: Compara lado a lado as matrizes de confusão normalizadas (recall por classe)
    dos 4 modelos ajustados em `tfidf_bi`, destacando o modelo final (`Linear SVM C=0.1`)
    e a diferença drástica no recall da classe Neutra.
    """
    _setup_plot_style()
    tuned = model_results.get("tuned", [])
    order = ["svm_bal", "logreg_bal", "logreg", "nb"]
    tuned_map = {r["classifier"]: r for r in tuned}

    fig, axes = plt.subplots(2, 2, figsize=(13, 10.5))
    axes_flat = axes.flatten()

    for ax, clf_key in zip(axes_flat, order):
        row = tuned_map[clf_key]
        cm = np.array(row["confusion_matrix"], dtype=int)
        cm_norm = cm / cm.sum(axis=1, keepdims=True)
        annot = np.array(
            [
                [f"{cm[i, j]:,}\n({cm_norm[i, j] * 100:.1f}%)".replace(",", ".") for j in range(3)]
                for i in range(3)
            ]
        )
        sns.heatmap(
            cm_norm,
            annot=annot,
            fmt="",
            cmap="Blues",
            vmin=0,
            vmax=1,
            cbar=False,
            xticklabels=LABEL_NAMES,
            yticklabels=LABEL_NAMES,
            linewidths=0.6,
            linecolor="white",
            ax=ax,
        )
        params_txt = ", ".join(
            f"{k.replace('clf__', '')}={v}" for k, v in row.get("best_params", {}).items()
        )
        badge = " ★ FINAL" if clf_key == model_results["final"]["classifier"] else ""
        ax.set_title(
            f"{CLF_LABELS[clf_key]} ({params_txt}){badge}\n"
            f"F1-Macro: {row['f1_macro']:.4f} | Acurácia: {row['accuracy']:.4f} | "
            f"Rec. Neg: {row['recall_Negativo']*100:.1f}%",
            fontsize=10.5,
            fontweight="bold" if badge else "normal",
        )
        ax.set_xlabel("Classe Prevista")
        ax.set_ylabel("Classe Real")

    fig.suptitle(
        "Matrizes de Confusão — Modelos Ajustados em TF-IDF (Uni+Bigramas) | Conjunto de Teste (n = 25.338)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_category_performance(
    df_cat: pd.DataFrame,
    overall_f1_macro: float,
    output_path: Path = FIGURES_DIR / "13_desempenho_por_categoria.png",
) -> Path:
    """
    Figura 13: Desempenho (F1-Macro e F1 por classe) nas Top 10 categorias de produtos
    no conjunto de teste, destacando a linha de referência global e o comportamento de Móveis.
    """
    _setup_plot_style()
    df_plot = df_cat.sort_values("f1_macro", ascending=True).reset_index(drop=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.2), gridspec_kw={"width_ratios": [1.15, 1]})

    # Painel esquerdo: F1-Macro por categoria com linha média global
    colors = [
        "#DC2626" if c == "Móveis" else "#2563EB"
        for c in df_plot["categoria"]
    ]
    bars = ax1.barh(df_plot["categoria"], df_plot["f1_macro"], color=colors, alpha=0.88)
    ax1.axvline(
        overall_f1_macro,
        color="#111827",
        linestyle="--",
        linewidth=1.6,
        label=f"F1-Macro Global ({overall_f1_macro:.4f})",
    )
    for bar, (_, r) in zip(bars, df_plot.iterrows()):
        ax1.text(
            bar.get_width() + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{r['f1_macro']:.4f} (n={r['n_test']:,})".replace(",", "."),
            va="center",
            fontsize=8.8,
        )
    ax1.set_xlim(0.55, 0.90)
    ax1.set_xlabel("F1-Score Macro")
    ax1.set_title("F1-Macro por Categoria (Top 10 no Teste)", fontweight="bold", pad=22)
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, 1.07), frameon=True, fontsize=9)

    # Painel direito: F1 por classe (Negativo, Neutro, Positivo) por categoria
    y_pos = np.arange(len(df_plot))
    h = 0.25
    ax2.barh(y_pos + h, df_plot["f1_Positivo"], height=h, color=PALETTE_SENTIMENT["Positivo"], label="Positivo")
    ax2.barh(y_pos, df_plot["f1_Negativo"], height=h, color=PALETTE_SENTIMENT["Negativo"], label="Negativo")
    ax2.barh(y_pos - h, df_plot["f1_Neutro"], height=h, color=PALETTE_SENTIMENT["Neutro"], label="Neutro")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([""] * len(df_plot))
    ax2.set_xlim(0, 1.03)
    ax2.set_xlabel("F1-Score por Classe")
    ax2.set_title("F1 Individual por Classe em cada Categoria", fontweight="bold", pad=22)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, 1.07), ncol=3, frameon=True, fontsize=9)

    fig.suptitle(
        "Estabilidade do Modelo Final entre Categorias de Produtos (site_category_lv1 — Pergunta 5)",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_error_and_noise_analysis(
    error_breakdown: Dict[str, Any],
    noise_analysis: Dict[str, Any],
    output_path: Path = FIGURES_DIR / "14_analise_erros_e_ruido.png",
) -> Path:
    """
    Figura 14: Dois painéis respondendo visualmente às Perguntas 3 e 4:
    (A) Distribuição dos 6 pares de confusão entre os erros do modelo final;
    (B) Evidência empírica de ruído de rótulo: taxa de contradição com
        `recommend_to_a_friend` no teste geral vs. nos erros extremos do modelo.
    """
    _setup_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.8))

    # Painel A: Pares de erro
    pairs_df = pd.DataFrame(error_breakdown["pairs"]).sort_values("quantidade", ascending=True)
    bar_colors = [
        "#D97706" if ("Neutro" in p) else "#DC2626"
        for p in pairs_df["par"]
    ]
    bars = ax1.barh(pairs_df["par"], pairs_df["quantidade"], color=bar_colors, alpha=0.9)
    for bar, (_, r) in zip(bars, pairs_df.iterrows()):
        ax1.text(
            bar.get_width() + 20,
            bar.get_y() + bar.get_height() / 2,
            f"{int(r['quantidade']):,} ({r['pct_dos_erros']:.1f}%)".replace(",", "."),
            va="center",
            fontsize=9.2,
        )
    ax1.set_xlim(0, pairs_df["quantidade"].max() * 1.28)
    ax1.set_xlabel("Quantidade de Erros no Conjunto de Teste")
    ax1.set_title(
        f"A) Origem dos {error_breakdown['n_errors']:,} Erros do Modelo Final\n"
        f"({error_breakdown['errors_involving_neutral_pct']:.1f}% envolvem a classe Neutra)".replace(",", "."),
        fontweight="bold",
    )

    # Painel B: Contradição nota vs recomendação (Ruído de Rótulo)
    labels_b = [
        "Negativo (nota 1–2)\nmas recomenda='Yes'",
        "Positivo (nota 4–5)\nmas recomenda='No'",
    ]
    geral_vals = [
        noise_analysis["neg_recommend_yes_all_test_pct"],
        noise_analysis["pos_recommend_no_all_test_pct"],
    ]
    extremos_vals = [
        noise_analysis["extreme_neg_pred_pos_recommend_yes_pct"],
        noise_analysis["extreme_pos_pred_neg_recommend_no_pct"],
    ]
    x = np.arange(len(labels_b))
    w = 0.34
    b1 = ax2.bar(x - w / 2, geral_vals, width=w, color="#64748B", label="Todo o conjunto de teste")
    b2 = ax2.bar(x + w / 2, extremos_vals, width=w, color="#DC2626", label="Apenas nos erros extremos do modelo")

    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            h + 1.0,
            f"{h:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
        )
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels_b)
    ax2.set_ylim(0, max(extremos_vals) * 1.25 + 5)
    ax2.set_ylabel("% com recomendação contraditória à nota")
    ax2.set_title(
        "B) Prova de Ruído de Rótulo nos Erros Extremos\n"
        "(Inconsistência entre nota e recommend_to_a_friend)",
        fontweight="bold",
    )
    ax2.legend(loc="upper right")

    fig.suptitle(
        "Análise Crítica de Erros e Ruído na Rotulagem por Notas (Perguntas 3 e 4 — Seção 2.6)",
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_negative_recall_comparison(
    model_results: Dict[str, Any],
    output_path: Path = FIGURES_DIR / "15_recall_negativo_comparacao.png",
) -> Path:
    """
    Figura 15: Compara Recall Negativo, F1 Negativo e F1-Macro nos modelos em
    `tfidf_bi` (padrão vs ajustado), respondendo visualmente à Pergunta 2.
    """
    _setup_plot_style()
    items = []
    for r in model_results.get("results", []):
        if r["representation"] == "tfidf_bi":
            # Evitar duplicata visual com LogReg balanced (C=1 é igual ao padrão)
            if r["classifier"] == "logreg_bal":
                continue
            items.append(
                {
                    "modelo": f"{CLF_LABELS[r['classifier']]}\n(padrão)",
                    "Recall Negativo": r["recall_Negativo"],
                    "F1 Negativo": r["f1_Negativo"],
                    "F1-Macro": r["f1_macro"],
                }
            )
    for r in model_results.get("tuned", []):
        p_txt = ", ".join(f"{k.replace('clf__', '')}={v}" for k, v in r["best_params"].items())
        star = " ★" if r["classifier"] == model_results["final"]["classifier"] else ""
        suffix = f"padrão/ajustado: {p_txt}" if r["classifier"] == "logreg_bal" else f"ajustado: {p_txt}"
        items.append(
            {
                "modelo": f"{CLF_LABELS[r['classifier']]}\n({suffix}){star}",
                "Recall Negativo": r["recall_Negativo"],
                "F1 Negativo": r["f1_Negativo"],
                "F1-Macro": r["f1_macro"],
            }
        )

    df_neg = pd.DataFrame(items).sort_values("F1-Macro", ascending=True).reset_index(drop=True)
    y = np.arange(len(df_neg))
    h = 0.25

    fig, ax = plt.subplots(figsize=(12.5, 6.4))
    b1 = ax.barh(y + h, df_neg["Recall Negativo"], height=h, color="#DC2626", label="Recall Negativo")
    b2 = ax.barh(y, df_neg["F1 Negativo"], height=h, color="#F97316", label="F1 Negativo")
    b3 = ax.barh(y - h, df_neg["F1-Macro"], height=h, color="#2563EB", label="F1-Macro (Global)")

    for bars in (b1, b2, b3):
        for bar in bars:
            w = bar.get_width()
            ax.text(w + 0.006, bar.get_y() + bar.get_height() / 2, f"{w:.4f}", va="center", fontsize=8.5)

    ax.set_yticks(y)
    ax.set_yticklabels(df_neg["modelo"], fontsize=9.2)
    ax.set_xlim(0.50, 1.00)
    ax.set_xlabel("Pontuação (0 a 1)")
    ax.set_title(
        "Detecção de Avaliações Negativas vs. F1-Macro — TF-IDF Uni+Bigramas (Pergunta 2)",
        fontsize=12.5,
        fontweight="bold",
        pad=28,
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=3, frameon=True, fontsize=9.5)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# Pipeline completo de avaliação da Pessoa 3
# ---------------------------------------------------------------------------


def run_full_evaluation(
    sample_size: Optional[int] = None,
    save_outputs: bool = True,
) -> Dict[str, Any]:
    """
    Executa todas as análises experimentais da Pessoa 3 sobre o conjunto de teste
    determinístico (80/20, random_state=42), gera os gráficos `12..15_*.png` e
    exporta `reports/evaluation_summary.json` para o Relatório e o Streamlit.
    """
    model_results = load_model_results()
    best_pipe = load_best_pipeline()

    df, X_train, X_test, y_train, y_test = get_split(sample_size=sample_size)
    df_pred = build_test_predictions_df(df, X_test, y_test, model_obj=best_pipe)

    final_metrics = compute_detailed_metrics(df_pred["sentiment_id_real"], df_pred["sentiment_id_pred"])
    df_comp = build_comparison_dataframe(model_results)
    error_breakdown = analyze_error_breakdown(df_pred)
    error_examples = extract_error_examples(df_pred, n_per_group=4)
    label_noise = analyze_label_noise(df_pred, top_n=6)
    df_cat = evaluate_by_category(df_pred, top_n=10)
    binary_comp = evaluate_binary_scenario(X_train, y_train, X_test, y_test)

    # Frases inéditas para validação qualitativa (nunca vistas no treino)
    unseen_samples = [
        "Produto excelente, superou minhas expectativas e chegou antes do prazo!",
        "O produto não funciona direito e não recomendo para ninguém.",
        "O celular é ótimo, mas a entrega atrasou duas semanas e veio com a caixa rasgada.",
        "Produto mediano, funciona bem para o básico, nada demais pelo preço.",
        "Não trava, bateria dura bastante e não me arrependo da compra!",
        "Parabéns pela transportadora incrível, só levou dois meses para entregar um produto quebrado.",
    ]
    unseen_predictions = [
        predict_single_review(txt, best_pipe, top_k=6) for txt in unseen_samples
    ]

    figures_saved: Dict[str, str] = {}
    if save_outputs:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        f12 = plot_confusion_matrices_comparison(model_results)
        f13 = plot_category_performance(df_cat, overall_f1_macro=final_metrics["f1_macro"])
        f14 = plot_error_and_noise_analysis(error_breakdown, label_noise)
        f15 = plot_negative_recall_comparison(model_results)
        figures_saved = {
            "confusion_matrices_comparison": str(f12.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "category_performance": str(f13.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "error_and_noise_analysis": str(f14.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "negative_recall_comparison": str(f15.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        }

    summary: Dict[str, Any] = {
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "final_model_info": {
            "representation": model_results["final"]["representation"],
            "representation_label": REP_LABELS[model_results["final"]["representation"]],
            "classifier": model_results["final"]["classifier"],
            "classifier_label": CLF_LABELS[model_results["final"]["classifier"]],
            "best_params": model_results["final"]["best_params"],
            "cv_f1_macro": round(float(model_results["final"]["cv_f1_macro"]), 4),
            "test_f1_macro": round(final_metrics["f1_macro"], 4),
            "test_accuracy": round(final_metrics["accuracy"], 4),
            "test_f1_weighted": round(final_metrics["f1_weighted"], 4),
            "per_class": final_metrics["per_class"],
            "confusion_matrix": final_metrics["confusion_matrix"],
            "confusion_matrix_normalized": final_metrics["confusion_matrix_normalized"],
        },
        "best_accuracy_model": {
            "representation": "tfidf_bi",
            "classifier": "logreg",
            "accuracy": round(float(df_comp.loc[df_comp["accuracy"].idxmax(), "accuracy"]), 4),
            "f1_macro": round(float(df_comp.loc[df_comp["accuracy"].idxmax(), "f1_macro"]), 4),
            "recall_Neutro": round(float(df_comp.loc[df_comp["accuracy"].idxmax(), "recall_Neutro"]), 4),
        },
        "error_breakdown": error_breakdown,
        "error_examples": error_examples,
        "label_noise_analysis": label_noise,
        "category_performance": df_cat.to_dict(orient="records"),
        "binary_scenario_comparison": binary_comp,
        "unseen_examples_validation": unseen_predictions,
        "figures": figures_saved,
    }

    if save_outputs:
        EVALUATION_SUMMARY_PATH.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return summary


if __name__ == "__main__":
    print("=== Avaliação Experimental e Análise de Erros (Pessoa 3 — src/evaluation.py) ===")
    res = run_full_evaluation(save_outputs=True)
    fm = res["final_model_info"]
    eb = res["error_breakdown"]
    ln = res["label_noise_analysis"]
    bc = res["binary_scenario_comparison"]

    print(f"Modelo final: {fm['representation_label']} + {fm['classifier_label']} {fm['best_params']}")
    print(f"  CV F1-Macro:   {fm['cv_f1_macro']:.4f}")
    print(f"  Test F1-Macro: {fm['test_f1_macro']:.4f} | Acurácia: {fm['test_accuracy']:.4f}")
    print(f"  Negativo -> P={fm['per_class']['Negativo']['precision']:.4f} "
          f"R={fm['per_class']['Negativo']['recall']:.4f} F1={fm['per_class']['Negativo']['f1']:.4f}")
    print(f"  Neutro   -> P={fm['per_class']['Neutro']['precision']:.4f} "
          f"R={fm['per_class']['Neutro']['recall']:.4f} F1={fm['per_class']['Neutro']['f1']:.4f}")
    print(f"  Positivo -> P={fm['per_class']['Positivo']['precision']:.4f} "
          f"R={fm['per_class']['Positivo']['recall']:.4f} F1={fm['per_class']['Positivo']['f1']:.4f}")
    print(f"\nErros no teste: {eb['n_errors']:,} ({eb['error_rate_pct']}%) | "
          f"Envolvem Neutro: {eb['errors_involving_neutral_pct']}% | "
          f"Extremos (Neg<->Pos): {eb['extreme_polarity_errors_pct']}%")
    print(f"Ruído de rótulo — Neg->Pos com recomenda='Yes': "
          f"{ln['extreme_neg_pred_pos_recommend_yes_pct']}% (vs {ln['neg_recommend_yes_all_test_pct']}% no teste geral)")
    print(f"Ruído de rótulo — Pos->Neg com recomenda='No':  "
          f"{ln['extreme_pos_pred_neg_recommend_no_pct']}% (vs {ln['pos_recommend_no_all_test_pct']}% no teste geral)")
    print(f"Cenário binário (sem nota 3): Acurácia={bc['accuracy']:.4f} | F1-Macro={bc['f1_macro']:.4f}")
    print("\nDesempenho por categoria (Top 10):")
    for row in res["category_performance"]:
        print(f"  {row['categoria']:<26} n={row['n_test']:>5} | F1-Macro={row['f1_macro']:.4f} "
              f"| Acc={row['accuracy']:.4f} | F1-Neu={row['f1_Neutro']:.4f}")
    print(f"\n[OK] Resumo salvo em: {EVALUATION_SUMMARY_PATH}")
    print("[OK] Gráficos salvos em reports/figures/12..15_*.png")
