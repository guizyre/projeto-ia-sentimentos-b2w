"""
Aplicação Demonstrável em Streamlit — Projeto 2: Análise de Sentimentos em Avaliações de Produtos (PLN)

Responsável: Pessoa 3 (Avaliação Crítica, Ética e Interface Demonstrável)
Disciplina: Inteligência Artificial – Unidade 3 (IFPB – Bacharelado em Engenharia da Computação)

Organizada em 3 abas integrando os módulos das Pessoas 1, 2 e 3:
- Aba 1: Corpus, Rotulagem (3 classes vs. Binário / Customizado) e Pré-processamento PLN
- Aba 2: Comparação de Modelos, Métricas (F1-Macro), Matrizes de Confusão, Categorias e Análise de Erros
- Aba 3: Simulador em Tempo Real (Exemplos Inéditos) com explicabilidade dos tokens mais influentes

Privacidade (Seção 2.8): Nenhum identificador pessoal ou atributo demográfico
(reviewer_id, idade, gênero, estado) é carregado ou exibido nesta interface.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
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
# Configuração da página e estilos
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Análise de Sentimentos B2W | PLN – IFPB",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .sentiment-badge {
        padding: 1rem 1.25rem;
        border-radius: 10px;
        color: white;
        font-weight: 700;
        font-size: 1.45rem;
        text-align: center;
        margin-bottom: 0.75rem;
    }
    .badge-positivo { background: linear-gradient(135deg, #059669, #10B981); }
    .badge-neutro   { background: linear-gradient(135deg, #D97706, #F59E0B); }
    .badge-negativo { background: linear-gradient(135deg, #DC2626, #EF4444); }
    .small-muted { color: #6B7280; font-size: 0.88rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Cache de dados e modelos (carregamento instantâneo sem ler o CSV de 50 MB)
# ---------------------------------------------------------------------------


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


def _stretch_kw(fn: Any) -> Dict[str, Any]:
    """Compatível com Streamlit >=1.49 (width='stretch') e versões anteriores (use_container_width=True)."""
    params = inspect.signature(fn).parameters
    if "width" in params and params["width"].default in ("content", "stretch"):
        return {"width": "stretch"}
    return {"use_container_width": True}


_IMG_KW = _stretch_kw(st.image)
_DF_KW = _stretch_kw(st.dataframe)
_BTN_KW = _stretch_kw(st.button)


def show_figure_if_exists(filename: str, caption: str) -> None:
    fig_path = FIGURES_DIR / filename
    if fig_path.exists():
        st.image(str(fig_path), caption=caption, **_IMG_KW)
    else:
        st.warning(f"Figura não encontrada: `{fig_path.name}`.")


def render_confusion_matrix_fig(cm: List[List[int]], title: str) -> plt.Figure:
    cm_arr = np.array(cm, dtype=int)
    row_sums = cm_arr.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm_arr.astype(float), row_sums, out=np.zeros_like(cm_arr, dtype=float), where=row_sums != 0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.3))
    sns.heatmap(
        cm_arr,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=LABEL_NAMES,
        yticklabels=LABEL_NAMES,
        cbar=False,
        linewidths=0.5,
        ax=ax1,
    )
    ax1.set_title("Contagens Absolutas", fontweight="bold")
    ax1.set_xlabel("Previsto")
    ax1.set_ylabel("Real")

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
        linewidths=0.5,
        ax=ax2,
    )
    ax2.set_title("Normalizada por Classe Real (Recall)", fontweight="bold")
    ax2.set_xlabel("Previsto")
    ax2.set_ylabel("Real")

    fig.suptitle(title, fontsize=11.5, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig


def render_token_contributions_fig(tokens_list: List[Dict[str, Any]], pred_label: str) -> plt.Figure:
    df_tok = pd.DataFrame(tokens_list)
    df_tok = df_tok.iloc[::-1].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8.2, max(3.0, 0.42 * len(df_tok) + 1.0)))
    colors = [
        PALETTE_SENTIMENT[pred_label] if v >= 0 else "#94A3B8"
        for v in df_tok["contrib_classe_prevista"]
    ]
    bars = ax.barh(df_tok["token"], df_tok["contrib_classe_prevista"], color=colors, alpha=0.9)
    ax.axvline(0, color="#1F2937", linewidth=1.0)
    for bar, val in zip(bars, df_tok["contrib_classe_prevista"]):
        offset = 0.01 * max(1e-3, df_tok["contrib_classe_prevista"].abs().max())
        ha = "left" if val >= 0 else "right"
        x_pos = val + offset if val >= 0 else val - offset
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2, f"{val:+.3f}", va="center", ha=ha, fontsize=8.5)
    ax.set_xlabel(f"Contribuição para a classe prevista ({pred_label})")
    ax.set_title("Termos / Bigramas mais influentes nesta avaliação", fontweight="bold")
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


# ---------------------------------------------------------------------------
# Barra lateral (Contexto, Licença e Privacidade)
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("💬 Análise de Sentimentos")
    st.caption("Projeto 2 – PLN | Inteligência Artificial (Unidade 3) | IFPB")
    st.divider()

    st.subheader("Corpus e Licença")
    st.markdown(
        f"""
        - **Corpus:** {DATASET_METADATA['name']} (Americanas.com)
        - **Período:** Jan–Mai/2018
        - **Idioma:** Português (`pt-BR`)
        - **Volume final:** `126.690` avaliações válidas
        - **Licença:** `CC BY-NC-SA 4.0` *(uso não comercial, com atribuição; CSV não redistribuído)*
        """
    )

    st.subheader("Privacidade (Seção 2.8)")
    st.success(
        "Colunas pessoais (`reviewer_id`, `reviewer_birth_year`, "
        "`reviewer_gender`, `reviewer_state`) foram removidas na ingestão "
        "e **nunca** são armazenadas ou exibidas."
    )

    st.subheader("Modelo Final em Produção")
    if model_results.get("final"):
        fin = model_results["final"]
        params = ", ".join(f"{k.replace('clf__', '')}={v}" for k, v in fin["best_params"].items())
        st.info(
            f"**{REP_LABELS[fin['representation']]}** + "
            f"**{CLF_LABELS[fin['classifier']]}** (`{params}`)\n\n"
            f"- **F1-Macro (teste):** `{fin['f1_macro']:.4f}`\n"
            f"- **F1-Macro (CV 3-fold):** `{fin['cv_f1_macro']:.4f}`\n"
            f"- **Acurácia:** `{fin['accuracy']:.2%}`\n"
            f"- **Recall Negativo:** `{fin['recall_Negativo']:.2%}`"
        )

    st.divider()
    st.caption("Equipe: Pessoa 1 (Dados/PLN) · Pessoa 2 (Modelos) · Pessoa 3 (Avaliação/App)")


# ---------------------------------------------------------------------------
# Cabeçalho principal e navegação em 3 abas (Etapa 6 do Guia)
# ---------------------------------------------------------------------------

st.title("Classificação Automática de Sentimentos em Avaliações de Produtos")
st.markdown(
    "Sistema completo de **Processamento de Linguagem Natural (PLN)** sobre o corpus **B2W-Reviews01**, "
    "comparando *Bag of Words* vs. *TF-IDF*, unigramas vs. bigramas e múltiplos classificadores supervisionados."
)

tab1, tab2, tab3 = st.tabs(
    [
        "📊 1. Corpus, Rotulagem e Pré-processamento",
        "📈 2. Comparação de Modelos, Métricas e Erros",
        "🧪 3. Simulador em Tempo Real (Exemplos Inéditos)",
    ]
)


# ===========================================================================
# ABA 1 – CORPUS, ROTULAGEM E PRÉ-PROCESSAMENTO
# ===========================================================================

with tab1:
    st.header("1. Caracterização do Corpus, Rotulagem e Pré-processamento PLN")

    if not eda or not pln:
        st.error("Resumos JSON da EDA/PLN não encontrados em `reports/`.")
    else:
        steps = eda["audit_cleaning_steps"]
        final_rows = pln.get("final_rows_after_preprocessing", steps["4_final_labeled_rows"])

        # KPIs principais (Funcionalidade 1)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Registros Brutos", f"{steps['0_raw_rows']:,}".replace(",", "."))
        c2.metric(
            "Avaliações Válidas (Final)",
            f"{final_rows:,}".replace(",", "."),
            delta=f"{steps['retention_rate_pct']}% retido",
        )
        c3.metric("Colunas Pessoais Removidas", f"{len(eda['personal_columns_removed'])}", delta="Privacidade OK")
        c4.metric(
            "Vocabulário Único (PLN)",
            f"{pln['vocabulary_stats']['unique_vocab_after']:,}".replace(",", "."),
            delta=f"-{pln['vocabulary_stats']['token_reduction_pct']}% tokens",
        )

        # Funil de limpeza
        with st.expander("📋 Auditoria detalhada do funil de limpeza", expanded=False):
            df_funnel = pd.DataFrame(
                [
                    ("0. Corpus bruto original (B2W-Reviews01.csv)", steps["0_raw_rows"], 0, "100,00%"),
                    (
                        "1. Remoção de nulos / notas inválidas",
                        steps["1_after_null_and_rating_filter"],
                        steps["removed_null_or_invalid_rating"],
                        f"{100*steps['1_after_null_and_rating_filter']/steps['0_raw_rows']:.2f}%",
                    ),
                    (
                        "2. Remoção de textos vazios / sem conteúdo alfabético",
                        steps["2_after_empty_or_short_text_filter"],
                        steps["removed_empty_or_short_text"],
                        f"{100*steps['2_after_empty_or_short_text_filter']/steps['0_raw_rows']:.2f}%",
                    ),
                    (
                        "3. Remoção de duplicatas exatas (texto + nota)",
                        steps["3_after_duplicate_removal"],
                        steps["removed_duplicates"],
                        f"{100*steps['3_after_duplicate_removal']/steps['0_raw_rows']:.2f}%",
                    ),
                    (
                        "4. Remoção residual após filtragem de stopwords",
                        final_rows,
                        pln.get("removed_empty_after_preprocessing", 1),
                        f"{100*final_rows/steps['0_raw_rows']:.2f}%",
                    ),
                ],
                columns=["Etapa", "Registros Restantes", "Removidos", "% Retido"],
            )
            st.dataframe(df_funnel, hide_index=True, **_DF_KW)

        st.divider()

        # Funcionalidade 2: Regra de mapeamento interativa (3 classes vs Binário vs Customizado)
        st.subheader("⚖️ Regra de Mapeamento de Notas (1–5) para Sentimentos")
        st.markdown(
            "A nota atribuída pelo consumidor é uma **aproximação imperfeita** do sentimento textual "
            "(supervisão fraca). Alterne abaixo para comparar o impacto da regra padrão (3 classes), "
            "do cenário binário (sem nota 3) ou de uma regra customizada:"
        )

        rating_dist = {int(k): int(v) for k, v in eda["clean_rating_distribution"].items()}
        mapping_mode = st.radio(
            "Escolha a regra de mapeamento para inspecionar:",
            options=[
                "Padrão do Projeto — 3 Classes (1–2 Negativo | 3 Neutro | 4–5 Positivo)",
                "Cenário Binário — 2 Classes (1–2 Negativo | 3 Descartada | 4–5 Positivo)",
                "Regra Customizada (definir estrela por estrela)",
            ],
            horizontal=False,
        )

        if mapping_mode.startswith("Padrão"):
            active_map = {1: "Negativo", 2: "Negativo", 3: "Neutro", 4: "Positivo", 5: "Positivo"}
        elif mapping_mode.startswith("Cenário Binário"):
            active_map = {1: "Negativo", 2: "Negativo", 3: "Descartar", 4: "Positivo", 5: "Positivo"}
        else:
            st.caption("Configure livremente o destino de cada nota de 1 a 5 estrelas:")
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

        # Recalcular contagens conforme a regra selecionada
        counts_by_class: Dict[str, int] = {"Negativo": 0, "Neutro": 0, "Positivo": 0}
        discarded_count = 0
        for star, cnt in rating_dist.items():
            target = active_map[star]
            if target == "Descartar":
                discarded_count += cnt
            else:
                counts_by_class[target] += cnt

        active_total = sum(counts_by_class.values())
        col_map_chart, col_map_info = st.columns([1.2, 1])

        with col_map_chart:
            df_map = pd.DataFrame(
                [
                    {
                        "Classe": cls,
                        "Avaliações": cnt,
                        "Proporção (%)": round(100.0 * cnt / active_total, 2) if active_total else 0.0,
                    }
                    for cls, cnt in counts_by_class.items()
                    if cnt > 0
                ]
            )
            fig_m, ax_m = plt.subplots(figsize=(6.8, 3.6))
            bar_colors = [PALETTE_SENTIMENT[c] for c in df_map["Classe"]]
            bars = ax_m.bar(df_map["Classe"], df_map["Avaliações"], color=bar_colors, width=0.55)
            for bar, (_, r) in zip(bars, df_map.iterrows()):
                ax_m.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + active_total * 0.015,
                    f"{r['Avaliações']:,}\n({r['Proporção (%)']:.1f}%)".replace(",", "."),
                    ha="center",
                    va="bottom",
                    fontsize=9.5,
                    fontweight="bold",
                )
            ax_m.set_ylim(0, max(df_map["Avaliações"]) * 1.22 if len(df_map) else 1)
            ax_m.set_ylabel("Nº de Avaliações")
            ax_m.set_title("Distribuição resultante da regra selecionada", fontweight="bold")
            ax_m.spines["top"].set_visible(False)
            ax_m.spines["right"].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig_m)
            plt.close(fig_m)

        with col_map_info:
            st.dataframe(df_map, hide_index=True, **_DF_KW)
            if discarded_count > 0:
                st.warning(
                    f"**Descartadas por esta regra:** `{discarded_count:,}` avaliações "
                    f"({100*discarded_count/sum(rating_dist.values()):.2f}% do corpus limpo).".replace(",", ".")
                )

            if mapping_mode.startswith("Cenário Binário") and eval_summary.get("binary_scenario_comparison"):
                bc = eval_summary["binary_scenario_comparison"]
                st.success(
                    f"**Experimento Binário (`TF-IDF uni+bi + Linear SVM C=0.1`):**\n"
                    f"Sem a classe Neutra (nota 3), o **F1-Macro salta de `0,7318` para `{bc['f1_macro']:.4f}`** "
                    f"e a **Acurácia vai de `83,78%` para `{bc['accuracy']:.2%}`**! "
                    "Isso prova que a principal dificuldade da tarefa está na ambiguidade da nota 3."
                )
            else:
                st.info(
                    "**Justificativa da Regra Padrão (3 classes):**\n"
                    "Preserva 100% do corpus válido e reflete opiniões mistas/moderadas (nota 3), "
                    "mas gera **forte desequilíbrio** (`61,8%` Positivo vs. `12,5%` Neutro — razão `4,94:1`). "
                    f"Além disso, a auditoria detectou **{eda['conflicting_rating_texts']} textos idênticos com notas diferentes**, "
                    f"**{eda['inconsistent_neg_rating_yes_recommend']:,}** notas 1–2 que recomendam o produto e "
                    f"**{eda['inconsistent_pos_rating_no_recommend']:,}** notas 4–5 que não recomendam.".replace(",", ".")
                )

        st.divider()

        # Gráficos EDA da Pessoa 1
        st.subheader("📉 Análise Exploratória do Corpus (EDA)")
        g1, g2 = st.columns(2)
        with g1:
            show_figure_if_exists("01_distribuicao_notas.png", "Distribuição das notas originais (1 a 5 estrelas).")
            show_figure_if_exists("03_tamanho_avaliacoes.png", "Extensão das avaliações por classe (reclamações são ~2x mais longas).")
        with g2:
            show_figure_if_exists("02_distribuicao_sentimentos.png", "Distribuição das 3 classes de sentimento (desequilíbrio 4,94x).")
            show_figure_if_exists("04_top_categorias.png", "Top 10 categorias e proporção de sentimentos (Móveis tem 43,9% negativos).")

        st.divider()

        # Funcionalidade 3: Pré-processamento PLN (Acentos e Negações)
        st.subheader("🧹 Pré-processamento de PLN (Preservação de Acentos e Negações)")
        p1, p2, p3 = st.columns(3)
        p1.metric("Stopwords NLTK Padrão", f"{pln['nltk_raw_stopwords_count']}")
        p2.metric("Stopwords Customizadas", f"{pln['custom_stopwords_count']}", delta="7 termos críticos protegidos")
        p3.metric("Negativas com Negação", "79,28%", delta="vs 20,86% nas Positivas")

        protected = get_protected_terms_removed_from_nltk()
        st.warning(
            "**Decisão Crítica de PLN (Seções 2.4.3 e 2.5.2):** A lista padrão de *stopwords* do NLTK em português "
            f"remove **`{', '.join(protected)}`**. Sem proteção, *\"O produto não funciona e não recomendo\"* vira "
            "*\"produto funciona recomendo\"*, invertendo a polaridade! Nosso pipeline preserva integralmente **acentos PT-BR** "
            f"(`{pln['vocabulary_stats']['accented_tokens_pct']}%` dos tokens finais) e todas as **negações/modificadores**."
        )

        # Demonstração interativa antes vs depois
        st.markdown("#### 🔍 Demonstração Comparativa: Antes vs. Depois do Pré-processamento")
        demo_txt = st.text_input(
            "Digite ou edite uma frase para comparar os pipelines de pré-processamento:",
            value="O produto não funciona direito, é péssimo e nunca mais recomendo!",
            key="tab1_pln_demo",
        )
        if demo_txt.strip():
            col_a, col_b, col_c = st.columns(3)
            sem_prot = preprocess_text(demo_txt, keep_negations=False, keep_modifiers=False)
            com_prot = preprocess_text(demo_txt, keep_negations=True, keep_modifiers=True, mark_negation=False)
            com_mark = preprocess_text(demo_txt, keep_negations=True, keep_modifiers=True, mark_negation=True)
            col_a.error(f"**NLTK padrão (SEM proteção ❌):**\n\n`{sem_prot}`")
            col_b.success(f"**Nosso Pipeline (Recomendado ✅):**\n\n`{com_prot}`")
            col_c.info(f"**Com Negation Marking (Opcional):**\n\n`{com_mark}`")

        with st.expander("📄 Exemplos reais do B2W-Reviews01 (Antes vs. Depois)", expanded=True):
            df_ex_pln = pd.DataFrame(pln.get("comparison_examples", []))
            if not df_ex_pln.empty:
                df_ex_pln = df_ex_pln.rename(
                    columns={
                        "sentimento": "Sentimento",
                        "nota": "Nota",
                        "texto_original": "Texto Original",
                        "nltk_padrao_sem_protecao": "NLTK Padrão (sem proteção)",
                        "pipeline_recomendado": "Nosso Pipeline (acentos + negações)",
                        "com_negation_marking": "Negation Marking",
                    }
                )
                st.dataframe(df_ex_pln, hide_index=True, **_DF_KW)

        g3, g4 = st.columns(2)
        with g3:
            show_figure_if_exists("05_impacto_negacoes.png", "Presença de palavras de negação por sentimento (79,3% nas negativas).")
            show_figure_if_exists("07_top_bigramas_por_sentimento.png", "Top 15 bigramas por classe de sentimento.")
        with g4:
            show_figure_if_exists("06_top_palavras_antes_depois_stopwords.png", "Top 20 palavras antes vs. depois da filtragem customizada.")
            show_figure_if_exists("08_wordclouds_por_sentimento.png", "Nuvens de palavras (WordClouds) por sentimento.")


# ===========================================================================
# ABA 2 – COMPARAÇÃO DE MODELOS, MÉTRICAS E ANÁLISE DE ERROS
# ===========================================================================

with tab2:
    st.header("2. Comparação de Modelos, Métricas e Análise Crítica (Seção 2.6)")

    if not model_results:
        st.error("Arquivo `reports/model_results.json` não encontrado.")
    else:
        fin = model_results["final"]
        best_acc = eval_summary.get("best_accuracy_model", {})

        # Destaque do Modelo Final (Pergunta 1)
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("F1-Macro (Final — Teste)", f"{fin['f1_macro']:.4f}", delta=f"CV: {fin['cv_f1_macro']:.4f}")
        m2.metric("Acurácia Global", f"{fin['accuracy']:.2%}")
        m3.metric("Recall Negativo", f"{fin['recall_Negativo']:.2%}", delta=f"F1: {fin['f1_Negativo']:.4f}")
        m4.metric("F1 Neutro (Minoritária)", f"{fin['f1_Neutro']:.4f}", delta=f"Recall: {fin['recall_Neutro']:.2%}")
        m5.metric("F1 Positivo", f"{fin['f1_Positivo']:.4f}", delta=f"Recall: {fin['recall_Positivo']:.2%}")

        st.info(
            "**Por que F1-Macro e não Acurácia?** A configuração de maior acurácia isolada "
            f"(`TF-IDF Uni+Bi + Regressão Logística padrão`, Acurácia = **{best_acc.get('accuracy', 0.8452):.2%}**) "
            f"recupera apenas **{best_acc.get('recall_Neutro', 0.2256):.1%}** das avaliações Neutras (F1-Macro = `{best_acc.get('f1_macro', 0.6993):.4f}`). "
            "Já o modelo final selecionado por validação cruzada em `X_train` — "
            "**TF-IDF Uni+Bigramas + Linear SVM (`C=0.1`, `class_weight='balanced'`)** — alcança **F1-Macro = `0,7318`**, "
            "**Recall Negativo = `91,38%`** e **Acurácia = `83,78%`**, equilibrando todas as classes sem *data leakage*."
        )

        # Tabela comparativa completa (16 padrão + 4 ajustados)
        st.subheader("🏆 Tabela Comparativa Completa (BoW vs. TF-IDF × Uni/Bigramas × Classificadores)")
        df_comp = build_comparison_dataframe(model_results)

        f_col1, f_col2, f_col3 = st.columns(3)
        etapa_opt = f_col1.multiselect(
            "Filtrar Etapa",
            options=["Ajustado (CV)", "Grade Padrão"],
            default=["Ajustado (CV)", "Grade Padrão"],
        )
        rep_opt = f_col2.multiselect(
            "Filtrar Representação",
            options=list(REP_LABELS.values()),
            default=list(REP_LABELS.values()),
        )
        clf_opt = f_col3.multiselect(
            "Filtrar Classificador",
            options=list(CLF_LABELS.values()),
            default=list(CLF_LABELS.values()),
        )

        df_view = df_comp[
            df_comp["etapa"].isin(etapa_opt)
            & df_comp["representacao_label"].isin(rep_opt)
            & df_comp["classificador_label"].isin(clf_opt)
        ].copy()

        display_cols = [
            "rank",
            "etapa",
            "representacao_label",
            "classificador_label",
            "hiperparametros",
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
                    "F1-Macro ★": "{:.4f}",
                    "Acurácia": "{:.4f}",
                    "F1 Neg.": "{:.4f}",
                    "Rec. Neg.": "{:.4f}",
                    "F1 Neu.": "{:.4f}",
                    "Rec. Neu.": "{:.4f}",
                    "F1 Pos.": "{:.4f}",
                }
            ).highlight_max(subset=["F1-Macro ★", "Acurácia", "Rec. Neg."], color="#D1FAE5"),
            hide_index=True,
            **_DF_KW,
        )

        st.divider()

        # Matriz de confusão interativa para QUALQUER modelo
        st.subheader("🧩 Explorador Interativo de Matriz de Confusão e Métricas por Classe")
        all_entries: List[Tuple[str, Dict[str, Any]]] = []
        for r in model_results.get("tuned", []):
            p_str = ", ".join(f"{k.replace('clf__', '')}={v}" for k, v in r["best_params"].items())
            star = " ★ FINAL" if r["classifier"] == fin["classifier"] else ""
            lbl = f"[Ajustado CV{star}] {REP_LABELS[r['representation']]} + {CLF_LABELS[r['classifier']]} ({p_str})"
            all_entries.append((lbl, r))
        for r in sorted(model_results.get("results", []), key=lambda x: x["f1_macro"], reverse=True):
            lbl = f"[Padrão] {REP_LABELS[r['representation']]} + {CLF_LABELS[r['classifier']]}"
            all_entries.append((lbl, r))

        labels_list = [x[0] for x in all_entries]
        default_idx = next((i for i, lbl in enumerate(labels_list) if "★ FINAL" in lbl), 0)
        selected_lbl = st.selectbox("Selecione o modelo para inspecionar a Matriz de Confusão:", labels_list, index=default_idx)
        selected_rec = dict(all_entries)[selected_lbl]

        fig_cm = render_confusion_matrix_fig(selected_rec["confusion_matrix"], selected_lbl)
        st.pyplot(fig_cm)
        plt.close(fig_cm)

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

        st.divider()

        # Gráficos comparativos de modelagem e avaliação
        st.subheader("📊 Visualizações Experimentais")
        v1, v2 = st.columns(2)
        with v1:
            show_figure_if_exists("09_f1_macro_comparacao.png", "Comparação de F1-Macro nas 16 combinações (BoW vs. TF-IDF × Uni/Bigramas).")
            show_figure_if_exists("15_recall_negativo_comparacao.png", "Pergunta 2: Detecção de avaliações negativas (Recall e F1 Negativo vs. F1-Macro).")
        with v2:
            show_figure_if_exists("12_matrizes_confusao_comparacao.png", "Matrizes de confusão dos 4 modelos ajustados em TF-IDF Uni+Bigramas.")
            show_figure_if_exists("11_termos_mais_influentes.png", "Termos com maior peso positivo, neutro e negativo (Regressão Logística balanced).")

        st.divider()

        # Pergunta 5: Desempenho por categoria de produto
        st.subheader("🏷️ Desempenho por Categoria de Produto (`site_category_lv1` — Pergunta 5)")
        if eval_summary.get("category_performance"):
            show_figure_if_exists("13_desempenho_por_categoria.png", "F1-Macro e F1 por classe nas Top 10 categorias do conjunto de teste.")
            df_cat = pd.DataFrame(eval_summary["category_performance"]).rename(
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

        st.divider()

        # Perguntas 3 e 4: Análise de Erros Reais e Ruído de Rótulos
        st.subheader("🔎 Análise de Erros Reais e Ruído na Conversão de Notas (Perguntas 3 e 4)")
        if eval_summary.get("error_breakdown") and eval_summary.get("label_noise_analysis"):
            eb = eval_summary["error_breakdown"]
            ln = eval_summary["label_noise_analysis"]

            e1, e2, e3, e4 = st.columns(4)
            e1.metric("Total de Erros no Teste", f"{eb['n_errors']:,}".replace(",", "."), delta=f"{eb['error_rate_pct']}% de 25.338")
            e2.metric("Erros Envolvendo Neutro", f"{eb['errors_involving_neutral_pct']:.1f}%", delta=f"{eb['errors_involving_neutral']:,} casos".replace(",", "."))
            e3.metric(
                "Ruído: Neg→Pos c/ Recomenda='Yes'",
                f"{ln['extreme_neg_pred_pos_recommend_yes_pct']:.1f}%",
                delta=f"vs {ln['neg_recommend_yes_all_test_pct']:.1f}% no teste geral",
            )
            e4.metric(
                "Ruído: Pos→Neg c/ Recomenda='No'",
                f"{ln['extreme_pos_pred_neg_recommend_no_pct']:.1f}%",
                delta=f"vs {ln['pos_recommend_no_all_test_pct']:.1f}% no teste geral",
            )

            show_figure_if_exists("14_analise_erros_e_ruido.png", "Distribuição dos tipos de erro e evidência empírica de ruído nos rótulos.")

            # Explorador de exemplos reais do X_test
            st.markdown("#### 📌 Exemplos Reais Extraídos de `X_test` (sem dados pessoais)")
            ex_groups = {
                "Avaliações Mistas (Produto bom vs. entrega/defeito ruim)": eval_summary["error_examples"]["avaliacoes_mistas_produto_vs_entrega"],
                "Ruído de Rótulo — Nota 1 prevista como Positivo (alta confiança)": ln["examples_nota1_pred_positivo"],
                "Ruído de Rótulo / Ironia — Nota 5 prevista como Negativo (alta confiança)": ln["examples_nota5_pred_negativo"],
                "Neutro confundido com Positivo": eval_summary["error_examples"]["neutro_previsto_como_positivo"],
                "Neutro confundido com Negativo": eval_summary["error_examples"]["neutro_previsto_como_negativo"],
                "Positivo confundido com Neutro": eval_summary["error_examples"]["positivo_previsto_como_neutro"],
            }
            chosen_err_group = st.selectbox("Filtrar categoria de erro / ruído real:", list(ex_groups.keys()))
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
                            "sentiment_real": "Real (Nota)",
                            "sentiment_pred": "Previsto",
                            "confidence": "Confiança",
                            "recommend_to_a_friend": "Recomenda?",
                            "site_category_lv1": "Categoria",
                            "review_text_original": "Avaliação Original",
                        }
                    ).style.format({"Confiança": "{:.1%}"}),
                    hide_index=True,
                    **_DF_KW,
                )


# ===========================================================================
# ABA 3 – SIMULADOR EM TEMPO REAL (EXEMPLOS INÉDITOS)
# ===========================================================================

with tab3:
    st.header("3. Simulador em Tempo Real — Classificação de Avaliações Inéditas")
    st.markdown(
        "Digite qualquer avaliação **não vista no treinamento** (ou selecione um exemplo de teste rápido) "
        "para inspecionar o pré-processamento, a classe prevista, os escores de confiança e os **tokens que mais pesaram**."
    )

    # Seletor de modelo (Final, outros ajustados ou combinação livre)
    sel_col1, sel_col2, sel_col3 = st.columns([1.4, 1, 1])
    model_mode = sel_col1.selectbox(
        "Modo de seleção do classificador:",
        options=[
            "★ Modelo Final — TF-IDF Uni+Bi + Linear SVM (C=0.1, balanced)",
            "Ajustado (CV) — TF-IDF Uni+Bi + Regressão Logística (balanced, C=1)",
            "Ajustado (CV) — TF-IDF Uni+Bi + Regressão Logística (padrão, C=3)",
            "Ajustado (CV) — TF-IDF Uni+Bi + Naive Bayes (alpha=0.1)",
            "Personalizado (escolher Representação × Classificador da grade padrão)",
        ],
    )

    if model_mode.startswith("Personalizado"):
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
        active_model_name = f"{REP_LABELS[rep_choice]} + {CLF_LABELS[clf_choice]}"
    elif "Linear SVM" in model_mode:
        active_model = get_best_model()
        active_model_name = "TF-IDF (Uni+Bigramas) + Linear SVM (C=0.1, balanced)"
    elif "balanced, C=1" in model_mode:
        active_model = get_tuned_model("logreg_bal")
        active_model_name = "TF-IDF (Uni+Bigramas) + Regressão Logística (balanced, C=1)"
    elif "padrão, C=3" in model_mode:
        active_model = get_tuned_model("logreg")
        active_model_name = "TF-IDF (Uni+Bigramas) + Regressão Logística (padrão, C=3)"
    else:
        active_model = get_tuned_model("nb")
        active_model_name = "TF-IDF (Uni+Bigramas) + Naive Bayes (alpha=0.1)"

    # Exemplos inéditos prontos para demonstração ao vivo
    st.markdown("**Exemplos inéditos para demonstração rápida (1 clique):**")
    preset_examples = {
        "✅ Elogio claro": "Produto excelente, superou minhas expectativas, ótimo acabamento e chegou antes do prazo!",
        "❌ Negação crítica": "O produto não funciona direito, veio com defeito e não recomendo para ninguém.",
        "⚠️ Mista (produto vs entrega)": "O celular é muito bom, mas a entrega atrasou duas semanas e a caixa chegou rasgada.",
        "🟡 Neutro moderado": "Produto ok, atende ao básico pelo preço, porém o material poderia ser um pouco mais resistente.",
        "🔄 Negação positiva": "Não trava, a bateria dura bastante e não me arrependo nem um pouco da compra!",
        "😏 Ironia / sarcasmo": "Parabéns pela transportadora maravilhosa, só levou dois meses para entregar o produto quebrado!",
    }

    if "sim_textarea" not in st.session_state:
        st.session_state["sim_textarea"] = preset_examples["⚠️ Mista (produto vs entrega)"]

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
        "Digite uma avaliação de produto em português:",
        height=110,
        key="sim_textarea",
        placeholder="Ex.: O produto chegou antes do prazo, ótima qualidade, mas o manual veio confuso...",
    )

    col_btn, col_cmp = st.columns([1, 2])
    analyze_clicked = col_btn.button("🚀 Analisar Sentimento", type="primary", **_BTN_KW)
    compare_all_tuned = col_cmp.checkbox(
        "Comparar também os 4 modelos ajustados lado a lado", value=True
    )

    if user_review.strip() and (analyze_clicked or True):
        pred = predict_single_review(user_review, active_model, top_k=10)
        sent = pred["sentimento"]
        probs = pred["probabilidades"]
        conf = probs[sent]

        st.divider()
        res_left, res_right = st.columns([1, 1.35])

        with res_left:
            badge_cls = f"badge-{sent.lower()}"
            emoji_map = {"Positivo": "😊", "Neutro": "😐", "Negativo": "😞"}
            st.markdown(
                f'<div class="sentiment-badge {badge_cls}">'
                f'{emoji_map[sent]} {sent.upper()} ({conf:.1%})'
                f"</div>",
                unsafe_allow_html=True,
            )
            st.caption(f"**Modelo ativo:** {active_model_name}")
            if pred["score_kind"] == "confianca_softmax_margem":
                st.caption(
                    "ℹ️ *Nota técnica:* `LinearSVC` não estima probabilidades calibradas; "
                    "os percentuais exibidos são a pontuação relativa obtida via *softmax* do `decision_function`."
                )

            # Gráfico de barras das 3 classes
            fig_p, ax_p = plt.subplots(figsize=(5.5, 2.5))
            vals = [probs[c] * 100 for c in LABEL_NAMES]
            bar_cols = [PALETTE_SENTIMENT[c] for c in LABEL_NAMES]
            bars = ax_p.barh(LABEL_NAMES, vals, color=bar_cols, height=0.55)
            for bar, v in zip(bars, vals):
                ax_p.text(v + 1.5, bar.get_y() + bar.get_height() / 2, f"{v:.1f}%", va="center", fontweight="bold")
            ax_p.set_xlim(0, 112)
            ax_p.set_xlabel("Confiança / Probabilidade (%)")
            ax_p.spines["top"].set_visible(False)
            ax_p.spines["right"].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig_p)
            plt.close(fig_p)

        with res_right:
            st.markdown("#### 🔍 Pré-processamento Aplicado ao Texto")
            st.success(f"**Texto normalizado (enviado ao modelo):**\n\n`{pred['texto_tratado'] or '(vazio após filtragem)'}`")

            # Verificar se havia negações/modificadores que o NLTK padrão teria apagado
            toks_ok = set(pred["texto_tratado"].split())
            toks_no_prot = set(pred["texto_nltk_sem_protecao"].split())
            saved_terms = sorted((toks_ok - toks_no_prot) & (NEGATION_WORDS | POLARITY_MODIFIERS))
            if saved_terms:
                st.warning(
                    f"🛡️ **Negações/modificadores salvos pelo nosso pipeline:** `{', '.join(saved_terms)}` "
                    f"(com NLTK padrão, a frase viraria: `{pred['texto_nltk_sem_protecao']}`)."
                )
            else:
                st.caption(f"Sem negações/modificadores filtrados pelo NLTK padrão (`{pred['texto_nltk_sem_protecao']}`).")

            st.caption(f"Tokens/bigramas reconhecidos no vocabulário do modelo: **{pred['tokens_matched']}**")

        # Explicabilidade: Tokens que mais pesaram na decisão
        st.markdown("#### 🧠 Tokens e Bigramas que Mais Pesaram na Predição")
        if pred["top_tokens_predicted"]:
            tok_c1, tok_c2 = st.columns([1.25, 1])
            with tok_c1:
                fig_tok = render_token_contributions_fig(pred["top_tokens_predicted"], sent)
                st.pyplot(fig_tok)
                plt.close(fig_tok)
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
                st.caption(f"Método: *{pred.get('method_note', '')}*.")
        else:
            st.warning(
                "Nenhum termo da frase foi encontrado no vocabulário do modelo (palavras raras ou fora de domínio). "
                "Tente escrever uma frase mais descritiva em português."
            )

        # Comparação simultânea entre os 4 modelos ajustados
        if compare_all_tuned:
            st.markdown("#### ⚔️ Comparação Simultânea dos 4 Classificadores Ajustados (TF-IDF Uni+Bigramas)")
            comp_rows = []
            for clf_key in ["svm_bal", "logreg_bal", "logreg", "nb"]:
                pipe_i = get_tuned_model(clf_key)
                r_i = predict_single_review(user_review, pipe_i, top_k=3)
                top3_terms = ", ".join(t["token"] for t in r_i["top_tokens_predicted"][:3])
                comp_rows.append(
                    {
                        "Classificador": CLF_LABELS[clf_key] + (" ★" if clf_key == "svm_bal" else ""),
                        "Previsto": r_i["sentimento"],
                        "P(Negativo)": r_i["probabilidades"]["Negativo"],
                        "P(Neutro)": r_i["probabilidades"]["Neutro"],
                        "P(Positivo)": r_i["probabilidades"]["Positivo"],
                        "Top 3 termos": top3_terms or "—",
                    }
                )
            st.dataframe(
                pd.DataFrame(comp_rows).style.format(
                    {
                        "P(Negativo)": "{:.1%}",
                        "P(Neutro)": "{:.1%}",
                        "P(Positivo)": "{:.1%}",
                    }
                ),
                hide_index=True,
                **_DF_KW,
            )
