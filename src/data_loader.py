"""
Módulo de carregamento, privacidade, limpeza, auditoria e rotulagem do corpus B2W-Reviews01.

Responsável: Pessoa 1 (Engenharia de Dados, EDA e Pré-processamento PLN)
Projeto 2 – Classificação Automática de Sentimentos em Avaliações de Produtos
Disciplina: Inteligência Artificial (Unidade 3) – IFPB

Referência do Corpus:
- Nome: B2W-Reviews01
- Origem: https://github.com/americanas-tech/b2w-reviews01
- Alternativa Hugging Face: https://huggingface.co/datasets/ruanchaves/b2w-reviews01
- Período de coleta: Janeiro a Maio de 2018 (Americanas.com)
- Licença: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
  (uso não comercial, com atribuição obrigatória; não redistribuir o CSV bruto no repositório).
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import requests
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Metadados oficiais do corpus e constantes do projeto
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "B2W-Reviews01.csv"

B2W_RAW_URL = (
    "https://raw.githubusercontent.com/americanas-tech/b2w-reviews01/"
    "main/B2W-Reviews01.csv"
)

DATASET_METADATA: Dict[str, object] = {
    "name": "B2W-Reviews01",
    "source_repo": "https://github.com/americanas-tech/b2w-reviews01",
    "huggingface_mirror": "https://huggingface.co/datasets/ruanchaves/b2w-reviews01",
    "collection_period": "Janeiro a Maio de 2018 (Americanas.com)",
    "language": "Português (pt-BR)",
    "license": "CC BY-NC-SA 4.0 (Atribuição-NãoComercial-CompartilhaIgual 4.0 Internacional)",
    "personal_columns_removed": [
        "reviewer_id",
        "reviewer_birth_year",
        "reviewer_gender",
        "reviewer_state",
    ],
    "default_rating_rule": {
        "Negativo (0)": [1, 2],
        "Neutro (1)": [3],
        "Positivo (2)": [4, 5],
    },
}

# Colunas com dados pessoais/demográficos que NÃO devem ser mantidas ou exibidas (Seção 2.8)
PERSONAL_DATA_COLUMNS = [
    "reviewer_id",
    "reviewer_birth_year",
    "reviewer_gender",
    "reviewer_state",
]

# Colunas técnicas úteis mantidas para análise e modelagem
KEEP_COLUMNS = [
    "submission_date",
    "site_category_lv1",
    "site_category_lv2",
    "review_title",
    "review_text",
    "overall_rating",
    "recommend_to_a_friend",
]

# Mapeamento padrão de notas (1 a 5) para rótulos de sentimento (Seção 2.4.2)
DEFAULT_SENTIMENT_MAP: Dict[int, str] = {
    1: "Negativo",
    2: "Negativo",
    3: "Neutro",
    4: "Positivo",
    5: "Positivo",
}

SENTIMENT_TO_ID: Dict[str, int] = {
    "Negativo": 0,
    "Neutro": 1,
    "Positivo": 2,
}

ID_TO_SENTIMENT: Dict[int, str] = {v: k for k, v in SENTIMENT_TO_ID.items()}

# Mapeamento binário alternativo (opcional para comparação na interface/análises)
BINARY_SENTIMENT_MAP: Dict[int, Optional[str]] = {
    1: "Negativo",
    2: "Negativo",
    3: None,  # Descartada no cenário binário
    4: "Positivo",
    5: "Positivo",
}


# ---------------------------------------------------------------------------
# Funções de download, carregamento e auditoria
# ---------------------------------------------------------------------------


def download_b2w_dataset(
    dest_path: Path | str = DEFAULT_DATA_PATH,
    force: bool = False,
    timeout: int = 120,
) -> Path:
    """
    Baixa o arquivo B2W-Reviews01.csv do repositório oficial caso ainda não exista localmente.

    Parâmetros
    ----------
    dest_path : Path | str
        Caminho de destino para salvar o CSV.
    force : bool
        Se True, sobrescreve o arquivo mesmo se já existir.
    timeout : int
        Tempo limite (segundos) para a requisição HTTP.

    Retorna
    -------
    Path
        Caminho local do arquivo baixado/existente.
    """
    dest = Path(dest_path)
    if dest.exists() and not force:
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(B2W_RAW_URL, stream=True, timeout=timeout)
    response.raise_for_status()

    with open(dest, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
    return dest


def load_raw_dataset(
    csv_path: Path | str = DEFAULT_DATA_PATH,
    auto_download: bool = False,
) -> pd.DataFrame:
    """
    Carrega o CSV bruto do B2W-Reviews01 sem alterar o conteúdo original.

    Parâmetros
    ----------
    csv_path : Path | str
        Caminho para o arquivo B2W-Reviews01.csv.
    auto_download : bool
        Se True e o arquivo não existir, tenta baixar automaticamente do GitHub.
    """
    path = Path(csv_path)
    if not path.exists():
        if auto_download:
            download_b2w_dataset(path)
        else:
            raise FileNotFoundError(
                f"Arquivo do corpus não encontrado em: {path}\n"
                "Baixe B2W-Reviews01.csv em https://github.com/americanas-tech/b2w-reviews01 "
                "e salve na pasta data/, ou chame com auto_download=True."
            )

    df = pd.read_csv(path, low_memory=False)
    return df


def audit_raw_dataset(df_raw: pd.DataFrame) -> Dict[str, object]:
    """
    Realiza auditoria completa do conjunto bruto (Seção 2.5.1):
    colunas, total de registros, valores ausentes, textos vazios, duplicatas e notas.
    """
    total_rows = int(len(df_raw))
    null_counts = {col: int(df_raw[col].isna().sum()) for col in df_raw.columns}

    text_series = df_raw["review_text"].fillna("").astype(str)
    empty_or_whitespace = int((text_series.str.strip() == "").sum())
    no_alpha_chars = int(
        (text_series.str.strip() != "").astype(bool).mul(
            ~text_series.str.contains(r"[A-Za-zÀ-ÿ]", regex=True)
        ).sum()
    )

    dup_full_rows = int(df_raw.duplicated().sum())
    dup_text_rating = int(
        df_raw.dropna(subset=["review_text", "overall_rating"])
        .duplicated(subset=["review_text", "overall_rating"])
        .sum()
    )
    dup_text_only = int(
        df_raw.dropna(subset=["review_text"]).duplicated(subset=["review_text"]).sum()
    )

    # Casos em que o mesmo texto exato recebeu notas diferentes (evidência de ruído!)
    valid_tr = df_raw.dropna(subset=["review_text", "overall_rating"])
    ratings_per_text = valid_tr.groupby("review_text")["overall_rating"].nunique()
    conflicting_texts = int((ratings_per_text > 1).sum())

    rating_dist = (
        df_raw["overall_rating"]
        .value_counts(dropna=False)
        .sort_index()
        .to_dict()
    )
    rating_dist_clean = {
        (int(k) if pd.notna(k) else "NaN"): int(v) for k, v in rating_dist.items()
    }

    return {
        "total_raw_rows": total_rows,
        "total_columns": int(df_raw.shape[1]),
        "columns": list(df_raw.columns),
        "null_counts": null_counts,
        "missing_review_text": int(df_raw["review_text"].isna().sum()),
        "missing_overall_rating": int(df_raw["overall_rating"].isna().sum()),
        "empty_or_whitespace_text": empty_or_whitespace,
        "non_alphabetic_text": no_alpha_chars,
        "duplicate_full_rows": dup_full_rows,
        "duplicate_text_and_rating": dup_text_rating,
        "duplicate_text_only": dup_text_only,
        "conflicting_rating_texts": conflicting_texts,
        "raw_rating_distribution": rating_dist_clean,
    }


def remove_personal_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove colunas com dados pessoais/demográficos dos avaliadores conforme exigido
    na Seção 2.8 do edital (Cuidados e limitações / privacidade).
    """
    cols_to_drop = [c for c in PERSONAL_DATA_COLUMNS if c in df.columns]
    df_safe = df.drop(columns=cols_to_drop).copy()

    # Mantém apenas as colunas técnicas relevantes existentes no DataFrame
    available_keep = [c for c in KEEP_COLUMNS if c in df_safe.columns]
    return df_safe[available_keep].copy()


def map_rating_to_sentiment(
    rating: int | float,
    mapping: Optional[Dict[int, Optional[str]]] = None,
) -> Optional[str]:
    """
    Converte a nota numérica (1 a 5) em rótulo de sentimento conforme a regra definida.

    Regra padrão (Seção 2.4.2):
    - Notas 1 e 2 -> Negativo
    - Nota 3      -> Neutro
    - Notas 4 e 5 -> Positivo
    """
    if pd.isna(rating):
        return None
    rule = mapping if mapping is not None else DEFAULT_SENTIMENT_MAP
    return rule.get(int(rating), None)


def clean_and_label_dataset(
    df_raw: pd.DataFrame,
    rating_mapping: Optional[Dict[int, Optional[str]]] = None,
    min_chars: int = 3,
    min_alpha_chars: int = 2,
    include_title: bool = False,
    drop_duplicates: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Executa todo o pipeline de limpeza, privacidade e rotulagem, retornando o
    DataFrame tratado e um dicionário detalhado de auditoria para o relatório.

    Colunas garantidas no DataFrame retornado (Contrato de Interface):
    - `review_text_original`: texto intacto original (preservado para comparações)
    - `overall_rating`: nota inteira (1 a 5)
    - `sentiment`: rótulo textual ('Negativo', 'Neutro', 'Positivo')
    - `sentiment_id`: rótulo numérico (0=Negativo, 1=Neutro, 2=Positivo)
    - `site_category_lv1`: categoria principal do produto (nulos -> 'Não informada')
    - `submission_date`, `review_title`, `recommend_to_a_friend`, `char_count`, `word_count`
    """
    audit = audit_raw_dataset(df_raw)

    # 1. Privacidade: descartar dados pessoais/demográficos imediatamente
    df = remove_personal_data(df_raw)
    rows_start = len(df)

    # 2. Remover registros sem texto ou sem nota válida (1 a 5)
    df = df.dropna(subset=["review_text", "overall_rating"]).copy()
    df["overall_rating"] = pd.to_numeric(df["overall_rating"], errors="coerce")
    df = df[df["overall_rating"].isin([1, 2, 3, 4, 5])].copy()
    df["overall_rating"] = df["overall_rating"].astype(int)
    rows_after_nulls = len(df)

    # 3. Remover textos vazios ou sem conteúdo alfabético útil
    df["review_text"] = df["review_text"].astype(str).str.strip()
    valid_len = df["review_text"].str.len() >= min_chars
    valid_alpha = (
        df["review_text"]
        .str.findall(r"[A-Za-zÀ-ÿ]")
        .str.len()
        >= min_alpha_chars
    )
    df = df[valid_len & valid_alpha].copy()
    rows_after_invalid_text = len(df)

    # 4. Tratar duplicatas exatas (mesmo texto e mesma nota)
    if drop_duplicates:
        df = df.drop_duplicates(subset=["review_text", "overall_rating"]).copy()
    rows_after_duplicates = len(df)

    # 5. Preservar versão intacta do texto original (exigência do projeto)
    df["review_text_original"] = df["review_text"]
    if include_title and "review_title" in df.columns:
        titles = df["review_title"].fillna("").astype(str).str.strip()
        df["review_text_with_title"] = (
            titles.where(titles != "", "")
            + titles.where(titles == "", ". ")
            + df["review_text_original"]
        ).str.strip()

    # 6. Aplicar regra de mapeamento de notas para sentimentos
    active_rule = rating_mapping if rating_mapping is not None else DEFAULT_SENTIMENT_MAP
    df["sentiment"] = df["overall_rating"].map(lambda r: map_rating_to_sentiment(r, active_rule))
    df = df.dropna(subset=["sentiment"]).copy()
    df["sentiment_id"] = df["sentiment"].map(SENTIMENT_TO_ID).astype(int)
    rows_after_mapping = len(df)

    # 7. Tratar metadados auxiliares (categorias e data)
    if "site_category_lv1" in df.columns:
        df["site_category_lv1"] = (
            df["site_category_lv1"].fillna("Não informada").astype(str).str.strip()
        )
        df.loc[df["site_category_lv1"] == "", "site_category_lv1"] = "Não informada"
    if "site_category_lv2" in df.columns:
        df["site_category_lv2"] = (
            df["site_category_lv2"].fillna("Não informada").astype(str).str.strip()
        )
    if "review_title" in df.columns:
        df["review_title"] = df["review_title"].fillna("").astype(str).str.strip()
    if "submission_date" in df.columns:
        df["submission_date"] = pd.to_datetime(df["submission_date"], errors="coerce")

    # 8. Estatísticas de extensão do texto original para EDA
    df["char_count"] = df["review_text_original"].str.len()
    df["word_count"] = df["review_text_original"].str.split().str.len()

    # Resetar índice e organizar ordem limpa de colunas
    df = df.reset_index(drop=True)
    ordered_cols = [
        "review_text_original",
        "overall_rating",
        "sentiment",
        "sentiment_id",
        "site_category_lv1",
        "site_category_lv2",
        "review_title",
        "recommend_to_a_friend",
        "submission_date",
        "char_count",
        "word_count",
    ]
    final_cols = [c for c in ordered_cols if c in df.columns]
    if "review_text_with_title" in df.columns:
        final_cols.insert(1, "review_text_with_title")
    df = df[final_cols]

    # Atualizar relatório de auditoria
    audit["cleaning_steps"] = {
        "0_raw_rows": rows_start,
        "1_after_null_and_rating_filter": rows_after_nulls,
        "removed_null_or_invalid_rating": rows_start - rows_after_nulls,
        "2_after_empty_or_short_text_filter": rows_after_invalid_text,
        "removed_empty_or_short_text": rows_after_nulls - rows_after_invalid_text,
        "3_after_duplicate_removal": rows_after_duplicates,
        "removed_duplicates": rows_after_invalid_text - rows_after_duplicates,
        "4_final_labeled_rows": rows_after_mapping,
        "removed_by_mapping_rule": rows_after_duplicates - rows_after_mapping,
        "retention_rate_pct": round(100.0 * rows_after_mapping / rows_start, 2),
    }
    audit["clean_rating_distribution"] = {
        int(k): int(v) for k, v in df["overall_rating"].value_counts().sort_index().items()
    }
    audit["clean_sentiment_distribution"] = {
        str(k): int(v) for k, v in df["sentiment"].value_counts().items()
    }
    audit["clean_sentiment_pct"] = {
        str(k): round(100.0 * v / len(df), 2)
        for k, v in df["sentiment"].value_counts().items()
    }
    audit["personal_columns_removed"] = [
        c for c in PERSONAL_DATA_COLUMNS if c in df_raw.columns
    ]

    return df, audit


def get_stratified_sample(
    df: pd.DataFrame,
    n_samples: int = 30000,
    stratify_col: str = "overall_rating",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Extrai uma amostra estratificada do corpus preservando a proporção original
    das notas/classes (Seção 2.3 do edital).
    """
    if n_samples >= len(df):
        return df.copy().reset_index(drop=True)

    df_sample, _ = train_test_split(
        df,
        train_size=n_samples,
        stratify=df[stratify_col],
        random_state=random_state,
    )
    return df_sample.reset_index(drop=True)


def load_and_clean_data(
    csv_path: Path | str = DEFAULT_DATA_PATH,
    sample_size: Optional[int] = None,
    rating_mapping: Optional[Dict[int, Optional[str]]] = None,
    apply_preprocessing: bool = False,
    preprocessing_kwargs: Optional[Dict[str, object]] = None,
    return_audit: bool = False,
    auto_download: bool = False,
    random_state: int = 42,
) -> pd.DataFrame | Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Função principal (Contrato de Interface para Pessoa 2 e Pessoa 3).

    Carrega o B2W-Reviews01, remove dados pessoais, limpa nulos/duplicados,
    aplica a regra de sentimento, extrai amostra estratificada (opcional) e
    aplica o pré-processamento de PLN (opcional).

    Parâmetros
    ----------
    csv_path : Path | str
        Caminho do arquivo B2W-Reviews01.csv.
    sample_size : int | None
        Se definido (ex.: 30000), retorna amostra estratificada por `overall_rating`.
        Se None, retorna todo o conjunto limpo.
    rating_mapping : dict | None
        Regra customizada de conversão nota -> sentimento (padrão: 1-2 Neg, 3 Neu, 4-5 Pos).
    apply_preprocessing : bool
        Se True, adiciona a coluna `review_text_clean` usando `src.preprocessing`.
    preprocessing_kwargs : dict | None
        Parâmetros opcionais repassados para `preprocess_text` (ex.: `mark_negation=True`).
    return_audit : bool
        Se True, retorna `(df, audit_report)`; caso contrário, retorna apenas `df`.
    """
    df_raw = load_raw_dataset(csv_path=csv_path, auto_download=auto_download)
    df_clean, audit = clean_and_label_dataset(df_raw, rating_mapping=rating_mapping)

    if sample_size is not None and sample_size < len(df_clean):
        df_clean = get_stratified_sample(
            df_clean,
            n_samples=sample_size,
            stratify_col="overall_rating",
            random_state=random_state,
        )
        audit["stratified_sample"] = {
            "enabled": True,
            "sample_size": int(len(df_clean)),
            "stratify_by": "overall_rating",
            "random_state": random_state,
            "sample_sentiment_pct": {
                str(k): round(100.0 * v / len(df_clean), 2)
                for k, v in df_clean["sentiment"].value_counts().items()
            },
        }
    else:
        audit["stratified_sample"] = {"enabled": False, "sample_size": int(len(df_clean))}

    if apply_preprocessing:
        from src.preprocessing import preprocess_series

        kwargs = preprocessing_kwargs or {}
        df_clean["review_text_clean"] = preprocess_series(
            df_clean["review_text_original"], **kwargs
        )
        before_empty_clean = len(df_clean)
        df_clean = df_clean[df_clean["review_text_clean"].str.strip() != ""].reset_index(
            drop=True
        )
        audit["removed_empty_after_preprocessing"] = int(
            before_empty_clean - len(df_clean)
        )

    if return_audit:
        return df_clean, audit
    return df_clean


if __name__ == "__main__":
    print("=== Teste e Auditoria do Corpus B2W-Reviews01 ===")
    df_clean, report = load_and_clean_data(return_audit=True)
    steps = report["cleaning_steps"]
    print(f"Total bruto:                 {steps['0_raw_rows']:,}")
    print(f"Removidos (nulos/nota inv.): {steps['removed_null_or_invalid_rating']:,}")
    print(f"Removidos (vazios/curtos):   {steps['removed_empty_or_short_text']:,}")
    print(f"Removidos (duplicatas):      {steps['removed_duplicates']:,}")
    print(f"Total final limpo:           {steps['4_final_labeled_rows']:,} ({steps['retention_rate_pct']}%)")
    print(f"Colunas pessoais removidas:  {report['personal_columns_removed']}")
    print(f"Textos com notas conflitantes (ruído): {report['conflicting_rating_texts']:,}")
    print("Distribuição de notas (limpo):", report["clean_rating_distribution"])
    print("Distribuição de sentimentos:  ", report["clean_sentiment_distribution"])
    print("Proporção de sentimentos (%): ", report["clean_sentiment_pct"])
    print("Colunas finais:", list(df_clean.columns))

    # Validação de amostragem estratificada
    df_30k = get_stratified_sample(df_clean, n_samples=30000)
    sample_pct = (df_30k["sentiment"].value_counts(normalize=True) * 100).round(2).to_dict()
    print("Proporção amostra estratificada (30k):", sample_pct)
