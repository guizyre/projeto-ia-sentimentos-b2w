"""
Aplicação Demonstrável em Streamlit — Projeto 2: Análise de Sentimentos em Avaliações de Produtos (PLN)

Equipe Responsável:
- Guilherme Gonçalves (Pessoa 1): Engenharia de Dados, EDA e Pré-processamento PLN
- Anaíza Silva (Pessoa 2): Representação Vetorial e Modelagem Supervisionada
- Hillary Maximino (Pessoa 3): Avaliação Crítica, Ética e Interface Demonstrável

Disciplina: Inteligência Artificial – Unidade 3 (IFPB – Bacharelado em Engenharia da Computação)
Professor: Marcelo José Siqueira Coutinho de Almeida

Organizada em 3 abas integrando todos os módulos do pipeline:
- Aba 1: Corpus, Regra de Rotulagem (3 classes vs. Binário / Customizado) e Pré-processamento PLN
- Aba 2: Comparação Experimental (20 modelos), Matrizes de Confusão, Categorias e Radiografia de Erros
- Aba 3: Simulador em Tempo Real (Exemplos Inéditos) com IA Explicável (pesos por token/bigrama)

Privacidade (Seção 2.8): Nenhum identificador pessoal ou atributo demográfico
(reviewer_id, idade, gênero, estado) é carregado, armazenado ou exibido nesta interface.
"""

from __future__ import annotations

import html
import inspect
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import altair as alt
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from src.data_loader import DATASET_METADATA, ID_TO_SENTIMENT
from src.evaluation import (
    CLF_LABELS,
    EVALUATION_SUMMARY_PATH,
    FIGURES_DIR,
    MODEL_RESULTS_PATH,
    PALETTE_SENTIMENT,
    REP_LABELS,
    build_comparison_dataframe,
    load_best_pipeline,
    load_tuned_pipeline,
    predict_single_review,
)
from src.models import CLASSIFIERS, LABEL_NAMES, MODELS_DIR, REPORTS_DIR, REPRESENTATIONS, load_artifacts
from src.preprocessing import (
    NEGATION_WORDS,
    POLARITY_MODIFIERS,
    get_protected_terms_removed_from_nltk,
    preprocess_text,
)

# ---------------------------------------------------------------------------
# Configuração da página e sistema de design (CSS moderno)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Sentimentos B2W · PLN | IA – IFPB",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

.block-container {
    padding-top: 1.6rem !important;
    padding-bottom: 3rem !important;
    max-width: 1380px !important;
}

/* Hero Banner */
.hero-card {
    background: linear-gradient(135deg, #0F172A 0%, #1E293B 55%, #1D4ED8 100%);
    border-radius: 16px;
    padding: 1.65rem 2rem;
    color: #F8FAFC;
    margin-bottom: 1.35rem;
    box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.18);
    border: 1px solid rgba(255, 255, 255, 0.08);
}
.hero-eyebrow {
    display: inline-block;
    background: rgba(59, 130, 246, 0.22);
    border: 1px solid rgba(147, 197, 253, 0.35);
    color: #BFDBFE;
    font-size: 0.74rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    padding: 0.26rem 0.7rem;
    border-radius: 999px;
    margin-bottom: 0.65rem;
}
.hero-title {
    font-size: 1.72rem;
    font-weight: 800;
    line-height: 1.22;
    margin: 0 0 0.45rem 0;
    color: #FFFFFF;
    letter-spacing: -0.02em;
}
.hero-sub {
    font-size: 0.94rem;
    color: #CBD5E1;
    margin: 0 0 1rem 0;
    max-width: 920px;
    line-height: 1.5;
}
.hero-pills {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
}
.hero-pill {
    background: rgba(255, 255, 255, 0.10);
    border: 1px solid rgba(255, 255, 255, 0.15);
    color: #F1F5F9;
    padding: 0.28rem 0.72rem;
    border-radius: 8px;
    font-size: 0.78rem;
    font-weight: 600;
}

/* Custom KPI Cards */
.kpi-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-top: 4px solid #2563EB;
    border-radius: 12px;
    padding: 1rem 1.15rem;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05);
    height: 100%;
}
.kpi-card.accent-emerald { border-top-color: #059669; }
.kpi-card.accent-amber   { border-top-color: #D97706; }
.kpi-card.accent-rose    { border-top-color: #DC2626; }
.kpi-card.accent-indigo  { border-top-color: #4F46E5; }
.kpi-label {
    font-size: 0.74rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #64748B;
    margin-bottom: 0.35rem;
}
.kpi-value {
    font-size: 1.58rem;
    font-weight: 800;
    color: #0F172A;
    line-height: 1.15;
    margin-bottom: 0.38rem;
    letter-spacing: -0.02em;
}
.kpi-badge {
    display: inline-block;
    font-size: 0.75rem;
    font-weight: 600;
    padding: 0.16rem 0.52rem;
    border-radius: 999px;
    background: #EFF6FF;
    color: #1D4ED8;
}
.kpi-badge.emerald { background: #ECFDF5; color: #047857; }
.kpi-badge.amber   { background: #FFFBEB; color: #B45309; }
.kpi-badge.rose    { background: #FEF2F2; color: #B91C1C; }
.kpi-badge.slate   { background: #F1F5F9; color: #475569; }

/* Section Headers */
.sec-header {
    margin: 0.3rem 0 0.85rem 0;
}
.sec-title {
    font-size: 1.18rem;
    font-weight: 700;
    color: #0F172A;
    margin: 0 0 0.18rem 0;
    letter-spacing: -0.01em;
}
.sec-desc {
    font-size: 0.88rem;
    color: #64748B;
    margin: 0;
}

/* Info Callouts */
.callout {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-left: 4px solid #2563EB;
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    font-size: 0.89rem;
    color: #1E293B;
    line-height: 1.55;
    margin: 0.65rem 0;
}
.callout.emerald { background: #F0FDF4; border-color: #BBF7D0; border-left-color: #059669; }
.callout.amber   { background: #FFFBEB; border-color: #FDE68A; border-left-color: #D97706; }
.callout.rose    { background: #FEF2F2; border-color: #FECACA; border-left-color: #DC2626; }

/* PLN Comparison Cards */
.pln-box {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 0.95rem 1.05rem;
    height: 100%;
}
.pln-box.bad  { border-top: 3px solid #DC2626; background: #FFFDFD; }
.pln-box.good { border-top: 3px solid #059669; background: #FAFFFD; }
.pln-box.alt  { border-top: 3px solid #2563EB; background: #F8FAFF; }
.pln-box-title {
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 0.5rem;
}
.pln-box.bad .pln-box-title  { color: #B91C1C; }
.pln-box.good .pln-box-title { color: #047857; }
.pln-box.alt .pln-box-title  { color: #1D4ED8; }
.token-wrap {
    display: flex;
    flex-wrap: wrap;
    gap: 0.32rem;
}
.tok-chip {
    background: #F1F5F9;
    color: #1E293B;
    border: 1px solid #CBD5E1;
    border-radius: 6px;
    padding: 0.16rem 0.48rem;
    font-size: 0.81rem;
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}
.tok-chip.protected {
    background: #DCFCE7;
    color: #065F46;
    border-color: #6EE7B7;
    font-weight: 700;
}

/* Prediction Result Card (Tab 3) */
.pred-card {
    border-radius: 14px;
    padding: 1.3rem 1.4rem;
    color: #FFFFFF;
    margin-bottom: 0.9rem;
    box-shadow: 0 8px 20px -4px rgba(15, 23, 42, 0.15);
}
.pred-positivo { background: linear-gradient(135deg, #047857 0%, #059669 60%, #10B981 100%); }
.pred-neutro   { background: linear-gradient(135deg, #B45309 0%, #D97706 60%, #F59E0B 100%); }
.pred-negativo { background: linear-gradient(135deg, #B91C1C 0%, #DC2626 60%, #EF4444 100%); }
.pred-kicker {
    font-size: 0.73rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    opacity: 0.9;
    font-weight: 700;
}
.pred-main {
    font-size: 1.85rem;
    font-weight: 800;
    margin: 0.2rem 0;
    letter-spacing: -0.02em;
}
.pred-conf {
    font-size: 0.9rem;
    font-weight: 600;
    opacity: 0.95;
}

/* Custom Probability Bars */
.prob-row {
    margin-bottom: 0.65rem;
}
.prob-head {
    display: flex;
    justify-content: space-between;
    font-size: 0.83rem;
    font-weight: 600;
    color: #334155;
    margin-bottom: 0.22rem;
}
.prob-track {
    width: 100%;
    height: 10px;
    background: #E2E8F0;
    border-radius: 999px;
    overflow: hidden;
}
.prob-fill {
    height: 100%;
    border-radius: 999px;
}

/* Mini Model Cards (Tab 3 comparison) */
.mini-clf-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 0.9rem 1rem;
    height: 100%;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
}
.mini-clf-card.is-final {
    border: 1.5px solid #2563EB;
    background: #F8FAFF;
}
.mini-clf-name {
    font-size: 0.8rem;
    font-weight: 700;
    color: #475569;
    margin-bottom: 0.35rem;
}
.mini-pill {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 700;
    color: #FFFFFF;
    margin-bottom: 0.45rem;
}
.mini-pill.Positivo { background: #059669; }
.mini-pill.Neutro   { background: #D97706; }
.mini-pill.Negativo { background: #DC2626; }

/* Sidebar Styling */
section[data-testid="stSidebar"] {
    background-color: #FFFFFF !important;
    border-right: 1px solid #E2E8F0 !important;
}
.sb-card {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 11px;
    padding: 0.85rem 0.95rem;
    margin-bottom: 0.8rem;
    font-size: 0.83rem;
    color: #334155;
}
.sb-card-title {
    font-size: 0.73rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #0F172A;
    margin-bottom: 0.45rem;
}
.team-item {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.38rem 0;
    border-bottom: 1px solid #E2E8F0;
}
.team-item:last-child {
    border-bottom: none;
    padding-bottom: 0;
}
.team-avatar {
    width: 30px;
    height: 30px;
    border-radius: 8px;
    background: #EFF6FF;
    color: #1D4ED8;
    font-weight: 800;
    font-size: 0.74rem;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    border: 1px solid #BFDBFE;
}
.team-name {
    font-weight: 700;
    font-size: 0.82rem;
    color: #0F172A;
    line-height: 1.2;
}
.team-role {
    font-size: 0.72rem;
    color: #64748B;
}

/* Tabs Polish */
button[data-baseweb="tab"] {
    font-weight: 700 !important;
    font-size: 0.92rem !important;
    padding: 0.65rem 1.1rem !important;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Utilitários de compatibilidade, formatação e cache
# ---------------------------------------------------------------------------


def _stretch_kw(fn: Any) -> Dict[str, Any]:
    """Compatível com Streamlit >=1.49 (width='stretch') e versões anteriores (use_container_width=True)."""
    params = inspect.signature(fn).parameters
    if "width" in params and params["width"].default in ("content", "stretch"):
        return {"width": "stretch"}
    return {"use_container_width": True}


_IMG_KW = _stretch_kw(st.image)
_DF_KW = _stretch_kw(st.dataframe)
_BTN_KW = _stretch_kw(st.button)
_CHART_KW = _stretch_kw(st.altair_chart)


def fmt_int(val: int | float) -> str:
    return f"{int(val):,}".replace(",", ".")


def fmt_dec(val: float, digits: int = 4) -> str:
    return f"{float(val):.{digits}f}".replace(".", ",")


def fmt_pct(val_0_to_1: float, digits: int = 2) -> str:
    return f"{float(val_0_to_1) * 100:.{digits}f}%".replace(".", ",")


def kpi_card(
    label: str,
    value: str,
    badge: str = "",
    accent: str = "blue",
    badge_tone: str = "blue",
) -> str:
    accent_cls = f"accent-{accent}" if accent in ("emerald", "amber", "rose", "indigo") else ""
    tone_cls = badge_tone if badge_tone in ("emerald", "amber", "rose", "slate") else ""
    badge_html = f'<span class="kpi-badge {tone_cls}">{html.escape(badge)}</span>' if badge else ""
    return (
        f'<div class="kpi-card {accent_cls}">'
        f'<div class="kpi-label">{html.escape(label)}</div>'
        f'<div class="kpi-value">{html.escape(value)}</div>'
        f"{badge_html}"
        f"</div>"
    )


def section_header(title: str, desc: str = "") -> None:
    desc_html = f'<p class="sec-desc">{html.escape(desc)}</p>' if desc else ""
    st.markdown(
        f'<div class="sec-header"><h3 class="sec-title">{html.escape(title)}</h3>{desc_html}</div>',
        unsafe_allow_html=True,
    )


def render_tokens_html(text_clean: str, highlight_protected: bool = True) -> str:
    tokens = [t for t in text_clean.split() if t]
    if not tokens:
        return '<span class="tok-chip">(nenhum token após filtragem)</span>'
    protected_set = NEGATION_WORDS | POLARITY_MODIFIERS
    chips = []
    for tok in tokens:
        base_parts = tok.split("_")
        is_prot = highlight_protected and (any(p in protected_set for p in base_parts) or "_" in tok)
        cls = "tok-chip protected" if is_prot else "tok-chip"
        chips.append(f'<span class="{cls}">{html.escape(tok)}</span>')
    return f'<div class="token-wrap">{"".join(chips)}</div>'


@st.cache_data(show_spinner=False)
def load_json_report(path_str: str) -> Dict[str, Any]:
    path = Path(path_str)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_resource(show_spinner=False)
def get_best_model() -> Any:
    return load_best_pipeline()


@st.cache_resource(show_spinner=False)
def get_tuned_model(clf_key: str) -> Any:
    return load_tuned_pipeline(clf_key)


@st.cache_resource(show_spinner=False)
def get_default_model(rep_key: str, clf_key: str) -> Tuple[Any, Any]:
    return load_artifacts(rep_key, clf_key)


def show_figure_if_exists(filename: str, caption: str) -> None:
    fig_path = FIGURES_DIR / filename
    if fig_path.exists():
        with st.container(border=True):
            st.image(str(fig_path), caption=caption, **_IMG_KW)
    else:
        st.warning(f"Figura não encontrada: `{fig_path.name}`.")


def render_confusion_matrix_fig(cm: List[List[int]], title: str) -> plt.Figure:
    cm_arr = np.array(cm, dtype=int)
    row_sums = cm_arr.sum(axis=1, keepdims=True)
    cm_norm = np.divide(
        cm_arr.astype(float),
        row_sums,
        out=np.zeros_like(cm_arr, dtype=float),
        where=row_sums != 0,
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.1))
    fig.patch.set_facecolor("#FFFFFF")

    sns.heatmap(
        cm_arr,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=LABEL_NAMES,
        yticklabels=LABEL_NAMES,
        cbar=False,
        linewidths=0.8,
        linecolor="#FFFFFF",
        annot_kws={"size": 10.5, "weight": "bold"},
        ax=ax1,
    )
    ax1.set_title("Contagens Absolutas (Teste)", fontweight="bold", fontsize=10.5, pad=10, color="#0F172A")
    ax1.set_xlabel("Classe Prevista", fontsize=9.5, color="#334155")
    ax1.set_ylabel("Classe Real", fontsize=9.5, color="#334155")

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".1%",
        cmap="Blues",
        vmin=0,
        vmax=1,
        xticklabels=LABEL_NAMES,
        yticklabels=LABEL_NAMES,
        cbar=False,
        linewidths=0.8,
        linecolor="#FFFFFF",
        annot_kws={"size": 10.5, "weight": "bold"},
        ax=ax2,
    )
    ax2.set_title("Normalizada por Classe Real (Recall)", fontweight="bold", fontsize=10.5, pad=10, color="#0F172A")
    ax2.set_xlabel("Classe Prevista", fontsize=9.5, color="#334155")
    ax2.set_ylabel("Classe Real", fontsize=9.5, color="#334155")

    fig.suptitle(title, fontsize=11.2, fontweight="bold", y=1.02, color="#0F172A")
    plt.tight_layout()
    return fig


def render_token_contributions_fig(tokens_list: List[Dict[str, Any]], pred_label: str) -> plt.Figure:
    df_tok = pd.DataFrame(tokens_list).iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.0, max(2.8, 0.40 * len(df_tok) + 0.9)))
    fig.patch.set_facecolor("#FFFFFF")
    colors = [
        PALETTE_SENTIMENT[pred_label] if v >= 0 else "#94A3B8"
        for v in df_tok["contrib_classe_prevista"]
    ]
    bars = ax.barh(df_tok["token"], df_tok["contrib_classe_prevista"], color=colors, alpha=0.92, height=0.62)
    ax.axvline(0, color="#0F172A", linewidth=1.0)
    span = max(1e-3, float(df_tok["contrib_classe_prevista"].abs().max()))
    ax.set_xlim(
        min(-0.05, float(df_tok["contrib_classe_prevista"].min()) - 0.22 * span),
        max(0.05, float(df_tok["contrib_classe_prevista"].max()) + 0.22 * span),
    )
    for bar, val in zip(bars, df_tok["contrib_classe_prevista"]):
        offset = 0.025 * span
        ha = "left" if val >= 0 else "right"
        x_pos = val + offset if val >= 0 else val - offset
        ax.text(
            x_pos,
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.3f}",
            va="center",
            ha=ha,
            fontsize=8.5,
            fontweight="bold",
            color="#1E293B",
        )
    ax.set_xlabel(f"Contribuição para a classe prevista ({pred_label})", fontsize=9.2)
    ax.set_title("Termos / Bigramas mais influentes nesta avaliação", fontweight="bold", fontsize=10.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Carga inicial dos resumos JSON
# ---------------------------------------------------------------------------

eda = load_json_report(str(REPORTS_DIR / "eda_summary.json"))
pln = load_json_report(str(REPORTS_DIR / "preprocessing_summary.json"))
model_results = load_json_report(str(MODEL_RESULTS_PATH))
eval_summary = load_json_report(str(EVALUATION_SUMMARY_PATH))

final_rows_count = int(
    pln.get(
        "final_rows_after_preprocessing",
        eda.get("audit_cleaning_steps", {}).get("4_final_labeled_rows", 126690),
    )
)
fin_cfg = model_results.get("final") or model_results.get("best_default") or {}
fin_f1 = float(fin_cfg.get("f1_macro", 0.0))
fin_acc = float(fin_cfg.get("accuracy", 0.0))

# ---------------------------------------------------------------------------
# Barra Lateral (Identidade, Modelo em Produção, Governança e Equipe)
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        '<div style="padding: 0.2rem 0 0.75rem 0;">'
        '<div style="font-size:0.72rem;font-weight:800;color:#2563EB;letter-spacing:0.07em;text-transform:uppercase;">IFPB · Inteligência Artificial</div>'
        '<div style="font-size:1.18rem;font-weight:800;color:#0F172A;margin-top:0.15rem;">Sentimentos B2W</div>'
        '<div style="font-size:0.79rem;color:#64748B;">Unidade 3 · Projeto Prático de PLN</div>'
        "</div>",
        unsafe_allow_html=True,
    )

    if fin_cfg:
        params_str = ", ".join(
            f"{k.replace('clf__', '')}={v}" for k, v in fin_cfg.get("best_params", {}).items()
        ) or "padrão"
        rep_lbl = REP_LABELS.get(fin_cfg.get("representation", ""), fin_cfg.get("representation", ""))
        clf_lbl = CLF_LABELS.get(fin_cfg.get("classifier", ""), fin_cfg.get("classifier", ""))
        cv_txt = (
            f"<div>• <b>F1-Macro (CV 3-fold):</b> <code>{fmt_dec(fin_cfg['cv_f1_macro'])}</code></div>"
            if "cv_f1_macro" in fin_cfg
            else ""
        )
        st.markdown(
            f'<div class="sb-card" style="border-left: 3.5px solid #2563EB;">'
            f'<div class="sb-card-title">★ Modelo Final em Produção</div>'
            f'<div style="font-weight:700;color:#1D4ED8;margin-bottom:0.4rem;">{html.escape(rep_lbl)} + {html.escape(clf_lbl)} (<code>{html.escape(params_str)}</code>)</div>'
            f'<div>• <b>F1-Macro (Teste):</b> <code>{fmt_dec(fin_f1)}</code></div>'
            f"{cv_txt}"
            f'<div>• <b>Acurácia Global:</b> <code>{fmt_pct(fin_acc)}</code></div>'
            f'<div>• <b>Recall Negativo:</b> <code>{fmt_pct(fin_cfg.get("recall_Negativo", 0.0))}</code></div>'
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        f'<div class="sb-card">'
        f'<div class="sb-card-title">Corpus & Licença</div>'
        f'<div>• <b>Dataset:</b> {html.escape(str(DATASET_METADATA.get("name", "B2W-Reviews01")))} (v1.0)</div>'
        f'<div>• <b>Coleta:</b> Jan–Mai/2018 (Americanas.com)</div>'
        f'<div>• <b>Acesso:</b> {html.escape(str(DATASET_METADATA.get("access_date", "23/11/2026")))}</div>'
        f'<div>• <b>Base limpa:</b> <code>{fmt_int(final_rows_count)}</code> avaliações</div>'
        f'<div>• <b>Licença:</b> <code>CC BY-NC-SA 4.0</code></div>'
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sb-card" style="background:#F0FDF4;border-color:#BBF7D0;">'
        '<div class="sb-card-title" style="color:#065F46;">🛡️ Privacidade Garantida (Seção 2.8)</div>'
        '<div style="color:#166534;font-size:0.8rem;line-height:1.45;">'
        "Colunas sensíveis (<code>reviewer_id</code>, nascimento, gênero e estado) são "
        "eliminadas na ingestão e auditadas por testes automatizados."
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sb-card">'
        '<div class="sb-card-title">Equipe Responsável</div>'
        '<div class="team-item">'
        '<div class="team-avatar">GG</div>'
        '<div><div class="team-name">Guilherme Gonçalves</div><div class="team-role">Dados, EDA & Pré-processamento PLN</div></div>'
        "</div>"
        '<div class="team-item">'
        '<div class="team-avatar">AS</div>'
        '<div><div class="team-name">Anaíza Silva</div><div class="team-role">Representação Vetorial & Modelos</div></div>'
        "</div>"
        '<div class="team-item">'
        '<div class="team-avatar">HM</div>'
        '<div><div class="team-name">Hillary Maximino</div><div class="team-role">Avaliação Crítica, Ética & App</div></div>'
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Hero Header Principal + Navegação em 3 Abas
# ---------------------------------------------------------------------------

st.markdown(
    f'<div class="hero-card">'
    f'<div class="hero-eyebrow">Projeto 2 · Processamento de Linguagem Natural · IFPB</div>'
    f'<h1 class="hero-title">Classificação Automática de Sentimentos em Avaliações de Produtos</h1>'
    f'<p class="hero-sub">'
    f"Pipeline experimental completo sobre o corpus aberto <b>B2W-Reviews01</b> em português: "
    f"preservação de acentuação e negações, comparação entre <i>Bag of Words</i> e <i>TF-IDF</i> (unigramas e bigramas), "
    f"ajuste supervisionado sem vazamento de dados e simulador em tempo real com explicabilidade por token."
    f"</p>"
    f'<div class="hero-pills">'
    f'<span class="hero-pill">📦 {fmt_int(final_rows_count)} Avaliações Válidas</span>'
    f'<span class="hero-pill">🏆 F1-Macro Teste: {fmt_dec(fin_f1)}</span>'
    f'<span class="hero-pill">🎯 Recall Negativo: {fmt_pct(fin_cfg.get("recall_Negativo", 0.0), 1)}</span>'
    f'<span class="hero-pill">🔒 Zero Dados Pessoais</span>'
    f'<span class="hero-pill">⚖️ Licença CC BY-NC-SA 4.0</span>'
    f"</div>"
    f"</div>",
    unsafe_allow_html=True,
)

tab1, tab2, tab3 = st.tabs(
    [
        "01 · Corpus, Rotulagem & Pré-processamento",
        "02 · Modelos, Métricas & Análise de Erros",
        "03 · Simulador em Tempo Real (IA Explicável)",
    ]
)


# ===========================================================================
# ABA 1 – CORPUS, ROTULAGEM E PRÉ-PROCESSAMENTO
# ===========================================================================

with tab1:
    if not eda or not pln:
        st.error("Resumos JSON da EDA/PLN não encontrados em `reports/`.")
    else:
        steps = eda["audit_cleaning_steps"]
        vocab = pln["vocabulary_stats"]
        sent_pct = eda["clean_sentiment_pct"]
        pos_pct_val = float(sent_pct.get("Positivo", 0.0))
        neu_pct_val = float(sent_pct.get("Neutro", 1.0))
        imb_ratio = pos_pct_val / neu_pct_val if neu_pct_val else 0.0

        neg_any = next(
            (x for x in pln.get("negation_presence_pct_by_sentiment", []) if "Qualquer" in x.get("Termo", "")),
            {"Negativo": 0.0, "Positivo": 0.0},
        )

        # Linha de KPIs customizados (Funcionalidade 1)
        k1, k2, k3, k4 = st.columns(4)
        k1.markdown(
            kpi_card(
                "Registros Brutos (CSV)",
                fmt_int(steps["0_raw_rows"]),
                "14 colunas originais",
                accent="indigo",
                badge_tone="slate",
            ),
            unsafe_allow_html=True,
        )
        k2.markdown(
            kpi_card(
                "Base Final Limpa",
                fmt_int(final_rows_count),
                f"{fmt_dec(steps['retention_rate_pct'], 2)}% de retenção",
                accent="emerald",
                badge_tone="emerald",
            ),
            unsafe_allow_html=True,
        )
        k3.markdown(
            kpi_card(
                "Vocabulário Único (PLN)",
                fmt_int(vocab["unique_vocab_after"]),
                f"-{fmt_dec(vocab['token_reduction_pct'], 2)}% tokens funcionais",
                accent="blue",
                badge_tone="blue",
            ),
            unsafe_allow_html=True,
        )
        k4.markdown(
            kpi_card(
                "Negativas com Negação",
                f"{fmt_dec(neg_any['Negativo'], 2)}%",
                f"vs {fmt_dec(neg_any['Positivo'], 2)}% nas Positivas",
                accent="rose",
                badge_tone="rose",
            ),
            unsafe_allow_html=True,
        )

        st.write("")

        # Funcionalidade 2: Regra de Mapeamento Interativa
        with st.container(border=True):
            section_header(
                "Regra de Mapeamento de Notas (1–5 ★) para Sentimentos",
                "A nota atribuída pelo consumidor atua como supervisão fraca. Compare a regra padrão de 3 classes, o cenário binário e regras customizadas:",
            )

            rating_dist = {int(k): int(v) for k, v in eda["clean_rating_distribution"].items()}
            mapping_mode = st.radio(
                "Regra de conversão ativa:",
                options=[
                    "Padrão do Projeto — 3 Classes (1–2 Negativo · 3 Neutro · 4–5 Positivo)",
                    "Cenário Binário — 2 Classes (1–2 Negativo · 3 Descartada · 4–5 Positivo)",
                    "Regra Customizada (mapear estrela por estrela)",
                ],
                horizontal=True,
            )

            if mapping_mode.startswith("Padrão"):
                active_map = {1: "Negativo", 2: "Negativo", 3: "Neutro", 4: "Positivo", 5: "Positivo"}
            elif mapping_mode.startswith("Cenário Binário"):
                active_map = {1: "Negativo", 2: "Negativo", 3: "Descartar", 4: "Positivo", 5: "Positivo"}
            else:
                cols_stars = st.columns(5)
                defaults = {1: "Negativo", 2: "Negativo", 3: "Neutro", 4: "Positivo", 5: "Positivo"}
                opts = ["Negativo", "Neutro", "Positivo", "Descartar"]
                active_map = {}
                for idx, star in enumerate([1, 2, 3, 4, 5]):
                    active_map[star] = cols_stars[idx].selectbox(
                        f"Nota {star} ★",
                        options=opts,
                        index=opts.index(defaults[star]),
                        key=f"custom_star_{star}",
                    )

            counts_by_class: Dict[str, int] = {"Negativo": 0, "Neutro": 0, "Positivo": 0}
            discarded_count = 0
            for star, cnt in rating_dist.items():
                target = active_map[star]
                if target == "Descartar":
                    discarded_count += cnt
                else:
                    counts_by_class[target] += cnt

            active_total = sum(counts_by_class.values())
            df_map = pd.DataFrame(
                [
                    {
                        "Classe": cls,
                        "Avaliações": counts_by_class[cls],
                        "Proporção": round(100.0 * counts_by_class[cls] / active_total, 2) if active_total else 0.0,
                        "Rótulo": (
                            f"{fmt_int(counts_by_class[cls])} "
                            f"({fmt_dec(100.0 * counts_by_class[cls] / active_total if active_total else 0.0, 1)}%)"
                        ),
                    }
                    for cls in ["Negativo", "Neutro", "Positivo"]
                    if counts_by_class[cls] > 0
                ]
            )

            col_map_chart, col_map_info = st.columns([1.25, 1])

            with col_map_chart:
                if not df_map.empty:
                    color_scale = alt.Scale(
                        domain=["Negativo", "Neutro", "Positivo"],
                        range=[PALETTE_SENTIMENT["Negativo"], PALETTE_SENTIMENT["Neutro"], PALETTE_SENTIMENT["Positivo"]],
                    )
                    bars_chart = (
                        alt.Chart(df_map)
                        .mark_bar(cornerRadiusTopLeft=7, cornerRadiusTopRight=7, size=58)
                        .encode(
                            x=alt.X("Classe:N", sort=["Negativo", "Neutro", "Positivo"], title=None, axis=alt.Axis(labelAngle=0, labelFontWeight="bold", labelFontSize=12)),
                            y=alt.Y("Avaliações:Q", title="Nº de Avaliações", scale=alt.Scale(domain=[0, float(df_map["Avaliações"].max()) * 1.18])),
                            color=alt.Color("Classe:N", scale=color_scale, legend=None),
                            tooltip=[
                                alt.Tooltip("Classe:N", title="Sentimento"),
                                alt.Tooltip("Avaliações:Q", title="Volume", format=","),
                                alt.Tooltip("Proporção:Q", title="Proporção (%)", format=".2f"),
                            ],
                        )
                    )
                    text_chart = (
                        alt.Chart(df_map)
                        .mark_text(dy=-10, fontWeight="bold", fontSize=12, color="#0F172A")
                        .encode(
                            x=alt.X("Classe:N", sort=["Negativo", "Neutro", "Positivo"]),
                            y=alt.Y("Avaliações:Q"),
                            text="Rótulo:N",
                        )
                    )
                    st.altair_chart(
                        (bars_chart + text_chart).properties(height=275, title="Distribuição Resultante da Regra Selecionada"),
                        **_CHART_KW,
                    )

            with col_map_info:
                st.dataframe(
                    df_map[["Classe", "Avaliações", "Proporção"]].rename(columns={"Proporção": "Proporção (%)"}),
                    hide_index=True,
                    **_DF_KW,
                )
                if discarded_count > 0:
                    pct_disc = 100.0 * discarded_count / sum(rating_dist.values())
                    st.markdown(
                        f'<div class="callout amber">'
                        f"<b>Avaliações descartadas nesta regra:</b> <code>{fmt_int(discarded_count)}</code> "
                        f"({fmt_dec(pct_disc, 2)}% do corpus limpo)."
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                if mapping_mode.startswith("Cenário Binário") and eval_summary.get("binary_scenario_comparison"):
                    bc = eval_summary["binary_scenario_comparison"]
                    st.markdown(
                        f'<div class="callout emerald">'
                        f"<b>Resultado Experimental no Cenário Binário:</b><br>"
                        f"Ao remover a nota 3 em treino e teste (sem <i>data leakage</i>), o mesmo classificador final salta de "
                        f"<b>F1-Macro <code>{fmt_dec(fin_f1)}</code> → <code>{fmt_dec(bc['f1_macro'])}</code></b> e de "
                        f"<b>Acurácia <code>{fmt_pct(fin_acc)}</code> → <code>{fmt_pct(bc['accuracy'])}</code></b>. "
                        f"Isso demonstra que o teto do cenário ternário vem da ambiguidade intrínseca da nota 3."
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div class="callout">'
                        f"<b>Por que manter 3 classes e avaliar por F1-Macro?</b><br>"
                        f"A regra padrão preserva 100% da base e captura opiniões mistas, mas cria desequilíbrio de "
                        f"<b><code>{fmt_dec(pos_pct_val, 1)}%</code> Positivo vs. <code>{fmt_dec(neu_pct_val, 1)}%</code> Neutro</b> "
                        f"(razão <code>{fmt_dec(imb_ratio, 2)}:1</code>). Além disso, a auditoria encontrou "
                        f"<b>{fmt_int(eda['conflicting_rating_texts'])}</b> textos idênticos com notas distintas, "
                        f"<b>{fmt_int(eda['inconsistent_neg_rating_yes_recommend'])}</b> notas 1–2 que recomendam o produto e "
                        f"<b>{fmt_int(eda['inconsistent_pos_rating_no_recommend'])}</b> notas 4–5 que não recomendam."
                        f"</div>",
                        unsafe_allow_html=True,
                    )

        st.write("")

        # Funcionalidade 3: Demonstração Interativa Antes vs Depois do Pré-processamento
        with st.container(border=True):
            protected = get_protected_terms_removed_from_nltk()
            section_header(
                "Laboratório Ao Vivo de Pré-processamento PLN (Acentos & Negações)",
                f"A lista padrão de {pln['nltk_raw_stopwords_count']} stopwords do NLTK apaga {', '.join(protected)} — invertendo polaridades. Compare em tempo real:",
            )

            demo_txt = st.text_input(
                "Edite ou digite qualquer frase para inspecionar os tokens gerados em cada estratégia:",
                value="O produto não funciona direito, é péssimo e nunca mais recomendo!",
                key="tab1_pln_demo",
            )
            if demo_txt.strip():
                sem_prot = preprocess_text(demo_txt, keep_negations=False, keep_modifiers=False)
                com_prot = preprocess_text(demo_txt, keep_negations=True, keep_modifiers=True, mark_negation=False)
                com_mark = preprocess_text(demo_txt, keep_negations=True, keep_modifiers=True, mark_negation=True)

                c_bad, c_good, c_alt = st.columns(3)
                c_bad.markdown(
                    f'<div class="pln-box bad">'
                    f'<div class="pln-box-title">❌ NLTK Padrão (Apaga Negações)</div>'
                    f"{render_tokens_html(sem_prot, highlight_protected=False)}"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                c_good.markdown(
                    f'<div class="pln-box good">'
                    f'<div class="pln-box-title">✅ Pipeline do Projeto (Acentos + Negações)</div>'
                    f"{render_tokens_html(com_prot, highlight_protected=True)}"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                c_alt.markdown(
                    f'<div class="pln-box alt">'
                    f'<div class="pln-box-title">🔗 Com Negation Marking (Opcional)</div>'
                    f"{render_tokens_html(com_mark, highlight_protected=True)}"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        st.write("")

        # Sub-abas organizadas para explorar os gráficos da EDA, PLN e Auditoria sem poluição visual
        section_header(
            "Painel Visual de Análise Exploratória (EDA) e Evidências Lexicais",
            "Navegue pelas sub-abas abaixo para inspecionar as distribuições, o impacto das negações, bigramas, nuvens de palavras e o funil de limpeza:",
        )
        eda_sub1, eda_sub2, eda_sub3, eda_sub4 = st.tabs(
            [
                "📊 Notas, Classes & Categorias",
                "🛡️ Negações & Stopwords",
                "☁️ Bigramas & WordClouds",
                "📋 Funil de Limpeza & Exemplos Reais",
            ]
        )

        with eda_sub1:
            g1, g2 = st.columns(2)
            with g1:
                show_figure_if_exists("01_distribuicao_notas.png", "Figura 01 — Distribuição das notas originais (1 a 5 estrelas).")
                show_figure_if_exists("03_tamanho_avaliacoes.png", "Figura 03 — Comprimento dos textos por sentimento (reclamações são ~2x mais longas).")
            with g2:
                show_figure_if_exists("02_distribuicao_sentimentos.png", "Figura 02 — Desequilíbrio entre Positivo, Negativo e Neutro.")
                show_figure_if_exists("04_top_categorias.png", "Figura 04 — Top 10 categorias e proporção de sentimentos (destaque para Móveis).")

        with eda_sub2:
            g3, g4 = st.columns(2)
            with g3:
                show_figure_if_exists("05_impacto_negacoes.png", "Figura 05 — Presença de operadores de negação por classe de sentimento.")
            with g4:
                show_figure_if_exists("06_top_palavras_antes_depois_stopwords.png", "Figura 06 — Top 20 palavras antes vs. depois da filtragem customizada.")

        with eda_sub3:
            g5, g6 = st.columns(2)
            with g5:
                show_figure_if_exists("07_top_bigramas_por_sentimento.png", "Figura 07 — Top 15 bigramas mais frequentes em cada sentimento.")
            with g6:
                show_figure_if_exists("08_wordclouds_por_sentimento.png", "Figura 08 — Nuvens de palavras (WordClouds) por classe de sentimento.")

        with eda_sub4:
            st.markdown("##### Funil Completo de Auditoria e Limpeza (`src/data_loader.py`)")
            df_funnel = pd.DataFrame(
                [
                    ("0. Corpus bruto original (B2W-Reviews01.csv)", steps["0_raw_rows"], 0, "100,00%"),
                    (
                        "1. Remoção de nulos / notas fora de 1–5",
                        steps["1_after_null_and_rating_filter"],
                        steps["removed_null_or_invalid_rating"],
                        fmt_pct(steps["1_after_null_and_rating_filter"] / steps["0_raw_rows"]),
                    ),
                    (
                        "2. Remoção de textos vazios / sem conteúdo alfabético",
                        steps["2_after_empty_or_short_text_filter"],
                        steps["removed_empty_or_short_text"],
                        fmt_pct(steps["2_after_empty_or_short_text_filter"] / steps["0_raw_rows"]),
                    ),
                    (
                        "3. Remoção de duplicatas exatas (texto + nota)",
                        steps["3_after_duplicate_removal"],
                        steps["removed_duplicates"],
                        fmt_pct(steps["3_after_duplicate_removal"] / steps["0_raw_rows"]),
                    ),
                    (
                        "4. Remoção residual pós-filtragem de stopwords",
                        final_rows_count,
                        pln.get("removed_empty_after_preprocessing", 1),
                        fmt_pct(final_rows_count / steps["0_raw_rows"]),
                    ),
                ],
                columns=["Etapa de Auditoria", "Registros Restantes", "Removidos na Etapa", "% Retido"],
            )
            st.dataframe(df_funnel, hide_index=True, **_DF_KW)

            st.markdown("##### Exemplos Reais do Corpus (Antes vs. Depois do Pré-processamento)")
            df_ex_pln = pd.DataFrame(pln.get("comparison_examples", []))
            if not df_ex_pln.empty:
                st.dataframe(
                    df_ex_pln.rename(
                        columns={
                            "sentimento": "Sentimento",
                            "nota": "Nota",
                            "texto_original": "Texto Original",
                            "nltk_padrao_sem_protecao": "NLTK Padrão (sem proteção)",
                            "pipeline_recomendado": "Pipeline Recomendado (acentos + negações)",
                            "com_negation_marking": "Com Negation Marking",
                        }
                    ),
                    hide_index=True,
                    **_DF_KW,
                )


# ===========================================================================
# ABA 2 – COMPARAÇÃO DE MODELOS, MÉTRICAS E ANÁLISE DE ERROS
# ===========================================================================

with tab2:
    if not model_results:
        st.error("Arquivo `reports/model_results.json` não encontrado.")
    else:
        df_comp = build_comparison_dataframe(model_results)
        best_acc_row = df_comp.loc[df_comp["accuracy"].idxmax()]
        best_acc_info = eval_summary.get("best_accuracy_model", {})
        best_acc_val = float(best_acc_info.get("accuracy", best_acc_row["accuracy"]))
        best_acc_f1m = float(best_acc_info.get("f1_macro", best_acc_row["f1_macro"]))
        best_acc_r_neu = float(best_acc_info.get("recall_Neutro", best_acc_row["recall_Neutro"]))

        # KPIs do Modelo Campeão
        m1, m2, m3, m4, m5 = st.columns(5)
        cv_badge = f"CV 3-fold: {fmt_dec(fin_cfg['cv_f1_macro'])}" if "cv_f1_macro" in fin_cfg else "Melhor modelo"
        m1.markdown(
            kpi_card("F1-Macro ★ (Teste)", fmt_dec(fin_f1), cv_badge, accent="blue", badge_tone="blue"),
            unsafe_allow_html=True,
        )
        m2.markdown(
            kpi_card("Acurácia Global", fmt_pct(fin_acc), f"n = {fmt_int(model_results.get('n_test', 25338))}", accent="indigo", badge_tone="slate"),
            unsafe_allow_html=True,
        )
        m3.markdown(
            kpi_card(
                "Recall Negativo",
                fmt_pct(fin_cfg.get("recall_Negativo", 0.0)),
                f"F1 Neg: {fmt_dec(fin_cfg.get('f1_Negativo', 0.0))}",
                accent="rose",
                badge_tone="rose",
            ),
            unsafe_allow_html=True,
        )
        m4.markdown(
            kpi_card(
                "F1 Neutro (Minoritária)",
                fmt_dec(fin_cfg.get("f1_Neutro", 0.0)),
                f"Recall: {fmt_pct(fin_cfg.get('recall_Neutro', 0.0))}",
                accent="amber",
                badge_tone="amber",
            ),
            unsafe_allow_html=True,
        )
        m5.markdown(
            kpi_card(
                "F1 Positivo",
                fmt_dec(fin_cfg.get("f1_Positivo", 0.0)),
                f"Recall: {fmt_pct(fin_cfg.get('recall_Positivo', 0.0))}",
                accent="emerald",
                badge_tone="emerald",
            ),
            unsafe_allow_html=True,
        )

        fin_rep_name = REP_LABELS.get(fin_cfg.get("representation", ""), fin_cfg.get("representation", ""))
        fin_clf_name = CLF_LABELS.get(fin_cfg.get("classifier", ""), fin_cfg.get("classifier", ""))
        fin_params_txt = ", ".join(
            f"{k.replace('clf__', '')}={v}" for k, v in fin_cfg.get("best_params", {}).items()
        ) or "padrão"

        st.markdown(
            f'<div class="callout">'
            f"<b>Por que escolhemos pelo F1-Macro e não pela Acurácia?</b> "
            f"O modelo de maior acurácia isolada (<code>{fmt_pct(best_acc_val)}</code>) recupera apenas "
            f"<b><code>{fmt_pct(best_acc_r_neu, 1)}</code></b> das avaliações Neutras (F1-Macro = <code>{fmt_dec(best_acc_f1m)}</code>). "
            f"Já o vencedor em validação cruzada em <code>X_train</code> — <b>{html.escape(fin_rep_name)} + {html.escape(fin_clf_name)} "
            f"(<code>{html.escape(fin_params_txt)}</code>)</b> — atinge <b>F1-Macro = <code>{fmt_dec(fin_f1)}</code></b>, "
            f"<b>Recall Negativo = <code>{fmt_pct(fin_cfg.get('recall_Negativo', 0.0))}</code></b> e "
            f"<b>Acurácia = <code>{fmt_pct(fin_acc)}</code></b>, sem qualquer vazamento de dados."
            f"</div>",
            unsafe_allow_html=True,
        )

        # Sub-abas limpas na Aba 2 para navegação fluida durante a apresentação
        mod_sub1, mod_sub2, mod_sub3, mod_sub4 = st.tabs(
            [
                "🏆 Ranking & Comparação (20 Modelos)",
                "🧩 Matrizes de Confusão Interativas",
                "🏷️ Estabilidade por Categoria (Pergunta 5)",
                "🔎 Erros Reais & Ruído de Rótulos (Perguntas 3 e 4)",
            ]
        )

        with mod_sub1:
            section_header(
                "Comparação Completa: Representações × Classificadores × Hiperparâmetros",
                "Filtre as 16 combinações da grade padrão e os 4 pipelines ajustados via GridSearchCV (3 folds em X_train):",
            )
            f_col1, f_col2, f_col3 = st.columns(3)
            etapa_opt = f_col1.multiselect(
                "Etapa Experimental",
                options=["Ajustado (CV)", "Grade Padrão"],
                default=["Ajustado (CV)", "Grade Padrão"],
            )
            rep_opt = f_col2.multiselect(
                "Representação Vetorial",
                options=list(REP_LABELS.values()),
                default=list(REP_LABELS.values()),
            )
            clf_opt = f_col3.multiselect(
                "Algoritmo Classificador",
                options=list(CLF_LABELS.values()),
                default=list(CLF_LABELS.values()),
            )

            df_view = df_comp[
                df_comp["etapa"].isin(etapa_opt)
                & df_comp["representacao_label"].isin(rep_opt)
                & df_comp["classificador_label"].isin(clf_opt)
            ].copy()

            if not df_view.empty:
                df_chart_rank = df_view.copy()
                df_chart_rank["nome_curto"] = (
                    "#"
                    + df_chart_rank["rank"].astype(str)
                    + " "
                    + df_chart_rank["representacao_label"]
                    + " · "
                    + df_chart_rank["classificador_label"]
                    + " ("
                    + df_chart_rank["hiperparametros"]
                    + ")"
                )
                df_chart_rank["f1_txt"] = df_chart_rank["f1_macro"].map(lambda v: fmt_dec(v, 4))
                base_rank = alt.Chart(df_chart_rank.head(12)).encode(
                    y=alt.Y(
                        "nome_curto:N",
                        sort="-x",
                        title=None,
                        axis=alt.Axis(labelLimit=340, labelFontSize=11, labelColor="#1E293B"),
                    )
                )
                rank_bars = base_rank.mark_bar(cornerRadiusTopRight=5, cornerRadiusBottomRight=5, height=20).encode(
                    x=alt.X(
                        "f1_macro:Q",
                        title="F1-Score Macro (Conjunto de Teste)",
                        scale=alt.Scale(domain=[0, 0.84]),
                    ),
                    color=alt.condition(
                        alt.datum.is_final,
                        alt.value("#2563EB"),
                        alt.value("#94A3B8"),
                    ),
                    tooltip=[
                        alt.Tooltip("nome_curto:N", title="Configuração"),
                        alt.Tooltip("f1_macro:Q", title="F1-Macro", format=".4f"),
                        alt.Tooltip("accuracy:Q", title="Acurácia", format=".4f"),
                        alt.Tooltip("recall_Negativo:Q", title="Recall Negativo", format=".4f"),
                        alt.Tooltip("recall_Neutro:Q", title="Recall Neutro", format=".4f"),
                    ],
                )
                rank_txt = base_rank.mark_text(align="left", dx=6, fontSize=11, fontWeight="bold", color="#0F172A").encode(
                    x=alt.X("f1_macro:Q"),
                    text="f1_txt:N",
                )
                rank_chart = (rank_bars + rank_txt).properties(
                    height=max(290, 30 * min(12, len(df_chart_rank)) + 55),
                    title="Top Configurações Ordenadas por F1-Macro (azul = modelo final)",
                )
                with st.container(border=True):
                    st.altair_chart(rank_chart, **_CHART_KW)

            display_cols = [
                "rank",
                "etapa",
                "representacao_label",
                "classificador_label",
                "hiperparametros",
                "n_features",
                "f1_macro",
                "accuracy",
                "f1_Negativo",
                "recall_Negativo",
                "f1_Neutro",
                "recall_Neutro",
                "f1_Positivo",
            ]
            df_show = df_view[display_cols].rename(
                columns={
                    "rank": "#",
                    "etapa": "Etapa",
                    "representacao_label": "Representação",
                    "classificador_label": "Classificador",
                    "hiperparametros": "Params",
                    "n_features": "Features",
                    "f1_macro": "F1-Macro ★",
                    "accuracy": "Acurácia",
                    "f1_Negativo": "F1 Neg.",
                    "recall_Negativo": "Rec. Neg.",
                    "f1_Neutro": "F1 Neu.",
                    "recall_Neutro": "Rec. Neu.",
                    "f1_Positivo": "F1 Pos.",
                }
            )
            st.dataframe(
                df_show.style.format(
                    {
                        "Features": "{:,}",
                        "F1-Macro ★": "{:.4f}",
                        "Acurácia": "{:.4f}",
                        "F1 Neg.": "{:.4f}",
                        "Rec. Neg.": "{:.4f}",
                        "F1 Neu.": "{:.4f}",
                        "Rec. Neu.": "{:.4f}",
                        "F1 Pos.": "{:.4f}",
                    }
                ).highlight_max(subset=["F1-Macro ★", "Acurácia", "Rec. Neg."], color="#DCFCE7"),
                hide_index=True,
                **_DF_KW,
            )

            v1, v2 = st.columns(2)
            with v1:
                show_figure_if_exists("09_f1_macro_comparacao.png", "Figura 09 — F1-Macro nas 16 combinações da grade padrão.")
                show_figure_if_exists("15_recall_negativo_comparacao.png", "Figura 15 — Pergunta 2: Recall e F1 Negativo vs. F1-Macro.")
            with v2:
                show_figure_if_exists("11_termos_mais_influentes.png", "Figura 11 — Termos com maior peso positivo, neutro e negativo.")
                show_figure_if_exists("10_matriz_confusao_melhor_modelo.png", "Figura 10 — Matriz de confusão detalhada do modelo campeão.")

        with mod_sub2:
            section_header(
                "Explorador Dinâmico de Matriz de Confusão",
                "Selecione qualquer um dos 20 modelos treinados para inspecionar acertos, confusões adjacentes e métricas por classe:",
            )
            all_entries: List[Tuple[str, Dict[str, Any]]] = []
            for r in model_results.get("tuned", []):
                p_str = ", ".join(f"{k.replace('clf__', '')}={v}" for k, v in r.get("best_params", {}).items())
                is_fin = (
                    r.get("classifier") == fin_cfg.get("classifier")
                    and r.get("representation") == fin_cfg.get("representation")
                )
                star = " ★ FINAL" if is_fin else ""
                lbl = f"[Ajustado CV{star}] {REP_LABELS[r['representation']]} + {CLF_LABELS[r['classifier']]} ({p_str})"
                all_entries.append((lbl, r))
            for r in sorted(model_results.get("results", []), key=lambda x: x["f1_macro"], reverse=True):
                lbl = f"[Padrão] {REP_LABELS[r['representation']]} + {CLF_LABELS[r['classifier']]}"
                all_entries.append((lbl, r))

            labels_list = [x[0] for x in all_entries]
            default_idx = next((i for i, lbl in enumerate(labels_list) if "★ FINAL" in lbl), 0)
            selected_lbl = st.selectbox("Modelo inspecionado:", labels_list, index=default_idx)
            selected_rec = dict(all_entries)[selected_lbl]

            cm_col, cls_col = st.columns([1.65, 1])
            with cm_col:
                with st.container(border=True):
                    fig_cm = render_confusion_matrix_fig(selected_rec["confusion_matrix"], selected_lbl)
                    st.pyplot(fig_cm)
                    plt.close(fig_cm)
            with cls_col:
                with st.container(border=True):
                    st.markdown("##### Métricas Individuais por Classe")
                    df_cls_metrics = pd.DataFrame(
                        [
                            {
                                "Classe": cls_name,
                                "Precisão": selected_rec[f"precision_{cls_name}"],
                                "Recall": selected_rec[f"recall_{cls_name}"],
                                "F1-Score": selected_rec[f"f1_{cls_name}"],
                            }
                            for cls_name in LABEL_NAMES
                        ]
                    )
                    st.dataframe(
                        df_cls_metrics.style.format({"Precisão": "{:.4f}", "Recall": "{:.4f}", "F1-Score": "{:.4f}"}),
                        hide_index=True,
                        **_DF_KW,
                    )
                    st.markdown(
                        f'<div class="callout" style="margin-top:0.75rem;">'
                        f"<b>Resumo deste modelo:</b><br>"
                        f"• F1-Macro: <code>{fmt_dec(selected_rec['f1_macro'])}</code><br>"
                        f"• Acurácia: <code>{fmt_pct(selected_rec['accuracy'])}</code>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

            show_figure_if_exists(
                "12_matrizes_confusao_comparacao.png",
                "Figura 12 — Comparação lado a lado das matrizes de confusão dos 4 classificadores ajustados em TF-IDF Uni+Bigramas.",
            )

        with mod_sub3:
            section_header(
                "Estabilidade entre Categorias de Produtos (site_category_lv1 — Pergunta 5)",
                "Avaliação segmentada nas 10 maiores categorias do conjunto de teste (incluindo Móveis):",
            )
            if eval_summary.get("category_performance"):
                df_cat_raw = pd.DataFrame(eval_summary["category_performance"])
                cat_bars = (
                    alt.Chart(df_cat_raw)
                    .mark_bar(cornerRadiusTopRight=5, cornerRadiusBottomRight=5, height=22)
                    .encode(
                        x=alt.X("f1_macro:Q", title="F1-Score Macro", scale=alt.Scale(domain=[0, 0.86])),
                        y=alt.Y("categoria:N", sort="-x", title=None, axis=alt.Axis(labelLimit=220, labelFontWeight="bold", labelColor="#1E293B")),
                        color=alt.condition(
                            alt.datum.categoria == "Móveis",
                            alt.value("#DC2626"),
                            alt.value("#2563EB"),
                        ),
                        tooltip=[
                            alt.Tooltip("categoria:N", title="Categoria"),
                            alt.Tooltip("n_test:Q", title="Amostras (Teste)", format=","),
                            alt.Tooltip("f1_macro:Q", title="F1-Macro", format=".4f"),
                            alt.Tooltip("accuracy:Q", title="Acurácia", format=".4f"),
                            alt.Tooltip("f1_Neutro:Q", title="F1 Neutro", format=".4f"),
                            alt.Tooltip("pct_negativo:Q", title="% Negativo", format=".1f"),
                        ],
                    )
                )
                ref_rule = (
                    alt.Chart(pd.DataFrame([{"ref": fin_f1}]))
                    .mark_rule(color="#0F172A", strokeDash=[5, 4], size=2)
                    .encode(x="ref:Q")
                )
                with st.container(border=True):
                    st.altair_chart(
                        (cat_bars + ref_rule).properties(
                            height=340,
                            title=f"F1-Macro por Categoria vs. Média Global ({fmt_dec(fin_f1)} — linha tracejada; vermelho = Móveis)",
                        ),
                        **_CHART_KW,
                    )

                df_cat = df_cat_raw.rename(
                    columns={
                        "categoria": "Categoria",
                        "n_test": "Amostras (Teste)",
                        "pct_negativo": "% Neg.",
                        "pct_neutro": "% Neu.",
                        "pct_positivo": "% Pos.",
                        "accuracy": "Acurácia",
                        "f1_macro": "F1-Macro",
                        "f1_Negativo": "F1 Neg.",
                        "f1_Neutro": "F1 Neu.",
                        "f1_Positivo": "F1 Pos.",
                        "recall_Negativo": "Rec. Neg.",
                    }
                )
                st.dataframe(
                    df_cat[
                        ["Categoria", "Amostras (Teste)", "% Neg.", "% Neu.", "% Pos.", "Acurácia", "F1-Macro", "F1 Neg.", "F1 Neu.", "F1 Pos.", "Rec. Neg."]
                    ].style.format(
                        {
                            "% Neg.": "{:.1f}%",
                            "% Neu.": "{:.1f}%",
                            "% Pos.": "{:.1f}%",
                            "Acurácia": "{:.4f}",
                            "F1-Macro": "{:.4f}",
                            "F1 Neg.": "{:.4f}",
                            "F1 Neu.": "{:.4f}",
                            "F1 Pos.": "{:.4f}",
                            "Rec. Neg.": "{:.4f}",
                        }
                    ),
                    hide_index=True,
                    **_DF_KW,
                )
                show_figure_if_exists("13_desempenho_por_categoria.png", "Figura 13 — F1-Macro e F1 individual por classe em cada categoria.")

        with mod_sub4:
            section_header(
                "Radiografia dos Erros Reais e Ruído na Conversão de Notas (Perguntas 3 e 4)",
                "Decomposição quantitativa das falhas em X_test e prova estatística de rótulos contraditórios:",
            )
            if eval_summary.get("error_breakdown") and eval_summary.get("label_noise_analysis"):
                eb = eval_summary["error_breakdown"]
                ln = eval_summary["label_noise_analysis"]

                e1, e2, e3, e4 = st.columns(4)
                e1.markdown(
                    kpi_card(
                        "Total de Erros no Teste",
                        fmt_int(eb["n_errors"]),
                        f"{fmt_dec(eb['error_rate_pct'], 2)}% de {fmt_int(eb['n_test'])}",
                        accent="indigo",
                        badge_tone="slate",
                    ),
                    unsafe_allow_html=True,
                )
                e2.markdown(
                    kpi_card(
                        "Envolvendo Classe Neutra",
                        f"{fmt_dec(eb['errors_involving_neutral_pct'], 1)}%",
                        f"{fmt_int(eb['errors_involving_neutral'])} erros adjacentes",
                        accent="amber",
                        badge_tone="amber",
                    ),
                    unsafe_allow_html=True,
                )
                e3.markdown(
                    kpi_card(
                        "Ruído: Neg→Pos c/ Recomenda='Yes'",
                        f"{fmt_dec(ln['extreme_neg_pred_pos_recommend_yes_pct'], 1)}%",
                        f"vs {fmt_dec(ln['neg_recommend_yes_all_test_pct'], 1)}% no teste geral",
                        accent="rose",
                        badge_tone="rose",
                    ),
                    unsafe_allow_html=True,
                )
                e4.markdown(
                    kpi_card(
                        "Ruído: Pos→Neg c/ Recomenda='No'",
                        f"{fmt_dec(ln['extreme_pos_pred_neg_recommend_no_pct'], 1)}%",
                        f"vs {fmt_dec(ln['pos_recommend_no_all_test_pct'], 1)}% no teste geral",
                        accent="rose",
                        badge_tone="rose",
                    ),
                    unsafe_allow_html=True,
                )

                st.write("")
                show_figure_if_exists("14_analise_erros_e_ruido.png", "Figura 14 — Origem dos erros (Painel A) e prova de ruído de rótulo nos erros extremos (Painel B).")

                with st.container(border=True):
                    st.markdown("##### Explorador de Casos Reais Extraídos de `X_test` (Anonimizados)")
                    ex_groups = {
                        "Avaliações Mistas (Produto elogiado vs. entrega/ressalva negativa)": eval_summary["error_examples"]["avaliacoes_mistas_produto_vs_entrega"],
                        "Ruído de Rótulo — Nota 1 ★ prevista como Positivo (alta confiança)": ln["examples_nota1_pred_positivo"],
                        "Ruído de Rótulo / Pré-uso — Nota 5 ★ prevista como Negativo (alta confiança)": ln["examples_nota5_pred_negativo"],
                        "Fronteira Adjacente — Neutro (3 ★) confundido com Positivo": eval_summary["error_examples"]["neutro_previsto_como_positivo"],
                        "Fronteira Adjacente — Neutro (3 ★) confundido com Negativo": eval_summary["error_examples"]["neutro_previsto_como_negativo"],
                        "Fronteira Adjacente — Positivo (4–5 ★) confundido com Neutro": eval_summary["error_examples"]["positivo_previsto_como_neutro"],
                    }
                    chosen_err_group = st.selectbox("Selecione o padrão de erro / ruído para inspecionar:", list(ex_groups.keys()))
                    df_ex = pd.DataFrame(ex_groups[chosen_err_group])
                    if not df_ex.empty:
                        st.dataframe(
                            df_ex[
                                [
                                    "overall_rating",
                                    "sentiment_real",
                                    "sentiment_pred",
                                    "confidence",
                                    "recommend_to_a_friend",
                                    "site_category_lv1",
                                    "review_text_original",
                                ]
                            ].rename(
                                columns={
                                    "overall_rating": "Nota",
                                    "sentiment_real": "Rótulo (Nota)",
                                    "sentiment_pred": "Previsto",
                                    "confidence": "Confiança",
                                    "recommend_to_a_friend": "Recomenda?",
                                    "site_category_lv1": "Categoria",
                                    "review_text_original": "Avaliação Original (X_test)",
                                }
                            ).style.format({"Confiança": "{:.1%}"}),
                            hide_index=True,
                            **_DF_KW,
                        )


# ===========================================================================
# ABA 3 – SIMULADOR EM TEMPO REAL (IA EXPLICÁVEL)
# ===========================================================================

with tab3:
    with st.container(border=True):
        section_header(
            "Simulador Interativo de Avaliações Inéditas",
            "Teste frases nunca vistas no treinamento, troque de classificador e visualize exatamente quais palavras e bigramas pesaram na decisão:",
        )

        # Construção 100% dinâmica das opções de modelos ajustados a partir de model_results.json
        tuned_records = model_results.get("tuned", [])
        model_options_map: Dict[str, Tuple[str, str]] = {}
        ordered_tuned = sorted(
            tuned_records,
            key=lambda r: (
                0 if r.get("classifier") == fin_cfg.get("classifier") else 1,
                -float(r.get("f1_macro", 0.0)),
            ),
        )
        for rec in ordered_tuned:
            c_key = rec["classifier"]
            r_key = rec["representation"]
            p_str = ", ".join(f"{k.replace('clf__', '')}={v}" for k, v in rec.get("best_params", {}).items())
            is_fin = c_key == fin_cfg.get("classifier") and r_key == fin_cfg.get("representation")
            prefix = "★ Modelo Final — " if is_fin else "Ajustado (CV) — "
            opt_label = f"{prefix}{REP_LABELS[r_key]} + {CLF_LABELS[c_key]} ({p_str})"
            model_options_map[opt_label] = ("best" if is_fin else "tuned", c_key)

        custom_opt_label = "Personalizado — Escolher Representação × Classificador (Grade Padrão)"
        all_sim_options = list(model_options_map.keys()) + [custom_opt_label]

        sel_col1, sel_col2, sel_col3 = st.columns([1.5, 1, 1])
        model_mode = sel_col1.selectbox("Classificador ativo:", options=all_sim_options)

        if model_mode == custom_opt_label:
            rep_choice = sel_col2.selectbox(
                "Representação:",
                options=list(REPRESENTATIONS.keys()),
                index=3,
                format_func=lambda k: REP_LABELS[k],
            )
            clf_choice = sel_col3.selectbox(
                "Classificador:",
                options=list(CLASSIFIERS.keys()),
                index=2,
                format_func=lambda k: CLF_LABELS[k],
            )
            active_model = get_default_model(rep_choice, clf_choice)
            active_model_name = f"{REP_LABELS[rep_choice]} + {CLF_LABELS[clf_choice]} (padrão)"
        else:
            kind, chosen_clf_key = model_options_map[model_mode]
            active_model = get_best_model() if kind == "best" else get_tuned_model(chosen_clf_key)
            active_model_name = model_mode.replace("★ Modelo Final — ", "").replace("Ajustado (CV) — ", "")

        st.markdown(
            '<div style="font-size:0.82rem;font-weight:700;color:#475569;margin:0.4rem 0 0.35rem 0;">'
            "Cenários rápidos para demonstração ao vivo (1 clique):"
            "</div>",
            unsafe_allow_html=True,
        )
        preset_examples = {
            "✅ Elogio claro": "Produto excelente, superou minhas expectativas, ótimo acabamento e chegou antes do prazo!",
            "❌ Negação crítica": "O produto não funciona direito, veio com defeito e não recomendo para ninguém.",
            "⚠️ Mista (produto vs entrega)": "O celular é muito bom, mas a entrega atrasou duas semanas e a caixa chegou rasgada.",
            "🟡 Neutro moderado": "Produto ok, atende ao básico pelo preço, porém o material poderia ser um pouco mais resistente.",
            "🔄 Negação positiva": "Não trava, a bateria dura bastante e não me arrependo nem um pouco da compra!",
            "🎭 Ironia / sarcasmo": "Parabéns pela transportadora maravilhosa, só levou dois meses para entregar o produto quebrado!",
        }

        if "sim_textarea" not in st.session_state:
            st.session_state["sim_textarea"] = preset_examples["❌ Negação crítica"]

        def _set_preset(txt: str) -> None:
            st.session_state["sim_textarea"] = txt

        btn_cols = st.columns(3)
        for idx, (btn_label, sample_text) in enumerate(preset_examples.items()):
            btn_cols[idx % 3].button(
                btn_label,
                on_click=_set_preset,
                args=(sample_text,),
                **_BTN_KW,
            )

        user_review = st.text_area(
            "Avaliação do consumidor em português:",
            height=100,
            key="sim_textarea",
            placeholder="Digite qualquer avaliação inédita de produto...",
        )

        act_c1, act_c2 = st.columns([1, 2.2])
        act_c1.button("🔍 Classificar Avaliação", type="primary", **_BTN_KW)
        compare_all_tuned = act_c2.checkbox(
            "Exibir comparação simultânea entre os 4 classificadores ajustados",
            value=True,
        )

    if user_review.strip():
        pred = predict_single_review(user_review, active_model, top_k=10)
        sent = pred["sentimento"]
        probs = pred["probabilidades"]
        conf = probs[sent]

        res_left, res_right = st.columns([1.05, 1.35])

        with res_left:
            with st.container(border=True):
                emoji_map = {"Positivo": "😊", "Neutro": "😐", "Negativo": "😞"}
                score_label = (
                    "Pontuação relativa (softmax da margem)"
                    if pred["score_kind"] == "confianca_softmax_margem"
                    else "Probabilidade estimada"
                )
                st.markdown(
                    f'<div class="pred-card pred-{sent.lower()}">'
                    f'<div class="pred-kicker">Sentimento Previsto</div>'
                    f'<div class="pred-main">{emoji_map[sent]} {html.escape(sent.upper())}</div>'
                    f'<div class="pred-conf">{score_label}: {fmt_pct(conf, 1)}</div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )

                # Barras de confiança modernas em HTML/CSS
                bars_html_parts = []
                for cls_name in ["Positivo", "Neutro", "Negativo"]:
                    p_val = float(probs[cls_name]) * 100.0
                    c_hex = PALETTE_SENTIMENT[cls_name]
                    bars_html_parts.append(
                        f'<div class="prob-row">'
                        f'<div class="prob-head"><span>{cls_name}</span><span>{fmt_dec(p_val, 1)}%</span></div>'
                        f'<div class="prob-track"><div class="prob-fill" style="width:{max(1.5, p_val):.1f}%;background:{c_hex};"></div></div>'
                        f"</div>"
                    )
                st.markdown("".join(bars_html_parts), unsafe_allow_html=True)
                st.caption(f"**Modelo ativo:** {active_model_name}")
                if pred["score_kind"] == "confianca_softmax_margem":
                    st.caption(
                        "ℹ️ `LinearSVC` maximiza margem geométrica e não possui `predict_proba` nativo; "
                        "os percentuais acima são derivados via *softmax* de `decision_function`."
                    )

        with res_right:
            with st.container(border=True):
                section_header(
                    "Inspeção do Pré-processamento na Frase",
                    "Tokens verdes indicam negações, intensificadores ou conectivos adversativos preservados pelo nosso pipeline:",
                )
                st.markdown(render_tokens_html(pred["texto_tratado"], highlight_protected=True), unsafe_allow_html=True)

                toks_ok = set(pred["texto_tratado"].split())
                toks_no_prot = set(pred["texto_nltk_sem_protecao"].split())
                saved_terms = sorted((toks_ok - toks_no_prot) & (NEGATION_WORDS | POLARITY_MODIFIERS))

                if saved_terms:
                    st.markdown(
                        f'<div class="callout emerald" style="margin-top:0.8rem;">'
                        f"🛡️ <b>Proteção de Polaridade Ativada:</b> os termos <code>{html.escape(', '.join(saved_terms))}</code> "
                        f"foram preservados.<br>"
                        f"<span style='font-size:0.82rem;color:#334155;'>Com NLTK padrão (sem proteção), o texto viraria: "
                        f"<code>{html.escape(pred['texto_nltk_sem_protecao'] or '(vazio)')}</code></span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div class="callout" style="margin-top:0.8rem;">'
                        f"Nenhum termo desta frase seria removido indevidamente pelas stopwords padrão do NLTK."
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                st.caption(
                    f"Atributos ativos reconhecidos no vocabulário deste modelo (unigramas + bigramas): **{pred['tokens_matched']}**"
                )

        # Explicabilidade por Token/Bigrama (XAI)
        with st.container(border=True):
            section_header(
                f"Explicabilidade Local — Tokens e Bigramas que Mais Pesaram para \"{sent}\"",
                f"Cálculo baseado em {pred.get('method_note', 'pesos lineares do classificador')}: barras coloridas empurram a favor de {sent}; barras cinzas puxam contra.",
            )
            if pred["top_tokens_predicted"]:
                tok_c1, tok_c2 = st.columns([1.3, 1])
                with tok_c1:
                    df_tok_alt = pd.DataFrame(pred["top_tokens_predicted"])
                    df_tok_alt["direcao"] = np.where(
                        df_tok_alt["contrib_classe_prevista"] >= 0,
                        f"A favor de {sent}",
                        f"Contra {sent}",
                    )
                    tok_chart = (
                        alt.Chart(df_tok_alt)
                        .mark_bar(cornerRadius=4, height=20)
                        .encode(
                            x=alt.X("contrib_classe_prevista:Q", title=f"Contribuição na decisão ({sent})"),
                            y=alt.Y("token:N", sort=alt.EncodingSortField(field="contrib_classe_prevista", op="max", order="descending"), title=None, axis=alt.Axis(labelFontWeight="bold", labelLimit=200)),
                            color=alt.Color(
                                "direcao:N",
                                scale=alt.Scale(
                                    domain=[f"A favor de {sent}", f"Contra {sent}"],
                                    range=[PALETTE_SENTIMENT[sent], "#94A3B8"],
                                ),
                                legend=alt.Legend(orient="top", title=None),
                            ),
                            tooltip=[
                                alt.Tooltip("token:N", title="Termo / Bigrama"),
                                alt.Tooltip("valor_feature:Q", title="Valor Vetorial", format=".4f"),
                                alt.Tooltip("contrib_Negativo:Q", title="Peso Negativo", format="+.4f"),
                                alt.Tooltip("contrib_Neutro:Q", title="Peso Neutro", format="+.4f"),
                                alt.Tooltip("contrib_Positivo:Q", title="Peso Positivo", format="+.4f"),
                            ],
                        )
                        .properties(height=max(240, 28 * len(df_tok_alt) + 45))
                    )
                    zero_line = alt.Chart(pd.DataFrame([{"x": 0}])).mark_rule(color="#0F172A", size=1.2).encode(x="x:Q")
                    st.altair_chart(tok_chart + zero_line, **_CHART_KW)

                with tok_c2:
                    df_tok_tbl = pd.DataFrame(pred["top_tokens_predicted"])[
                        ["token", "contrib_Negativo", "contrib_Neutro", "contrib_Positivo"]
                    ].rename(
                        columns={
                            "token": "Termo / Bigrama",
                            "contrib_Negativo": "Peso Neg.",
                            "contrib_Neutro": "Peso Neu.",
                            "contrib_Positivo": "Peso Pos.",
                        }
                    )
                    st.dataframe(
                        df_tok_tbl.style.format(
                            {"Peso Neg.": "{:+.3f}", "Peso Neu.": "{:+.3f}", "Peso Pos.": "{:+.3f}"}
                        ),
                        hide_index=True,
                        **_DF_KW,
                    )
            else:
                st.warning(
                    "Nenhum termo desta frase consta no vocabulário treinado (`min_df=3`). "
                    "Experimente escrever uma avaliação mais descritiva em português."
                )

        # Comparação simultânea em 4 cards lado a lado
        if compare_all_tuned and tuned_records:
            st.write("")
            section_header(
                "Consenso entre os 4 Classificadores Ajustados (TF-IDF Uni+Bigramas)",
                "Veja como cada família algorítmica classifica exatamente a mesma frase:",
            )
            clf_cols = st.columns(4)
            clf_order = ["svm_bal", "logreg_bal", "logreg", "nb"]
            for col_ui, clf_key in zip(clf_cols, clf_order):
                pipe_i = get_tuned_model(clf_key)
                r_i = predict_single_review(user_review, pipe_i, top_k=3)
                s_i = r_i["sentimento"]
                c_i = r_i["probabilidades"][s_i]
                is_fin_card = clf_key == fin_cfg.get("classifier")
                card_cls = "mini-clf-card is-final" if is_fin_card else "mini-clf-card"
                title_star = " ★ FINAL" if is_fin_card else ""
                top3_str = ", ".join(t["token"] for t in r_i["top_tokens_predicted"][:3]) or "—"
                col_ui.markdown(
                    f'<div class="{card_cls}">'
                    f'<div class="mini-clf-name">{html.escape(CLF_LABELS[clf_key] + title_star)}</div>'
                    f'<div class="mini-pill {s_i}">{html.escape(s_i)} · {fmt_pct(c_i, 1)}</div>'
                    f'<div style="font-size:0.76rem;color:#475569;margin-bottom:0.3rem;">'
                    f"Neg: <b>{fmt_pct(r_i['probabilidades']['Negativo'], 0)}</b> · "
                    f"Neu: <b>{fmt_pct(r_i['probabilidades']['Neutro'], 0)}</b> · "
                    f"Pos: <b>{fmt_pct(r_i['probabilidades']['Positivo'], 0)}</b>"
                    f"</div>"
                    f'<div style="font-size:0.74rem;color:#64748B;"><b>Top termos:</b> <code>{html.escape(top3_str)}</code></div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )
