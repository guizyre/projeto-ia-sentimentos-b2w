"""
Módulo de pré-processamento textual para Português (pt-BR) com preservação de
acentuação e tratamento explícito de palavras de negação.

Responsável: Pessoa 1 (Engenharia de Dados, EDA e Pré-processamento PLN)
Projeto 2 – Classificação Automática de Sentimentos em Avaliações de Produtos
Disciplina: Inteligência Artificial (Unidade 3) – IFPB

Decisões técnicas alinhadas ao edital (Seções 2.4.3 e 2.5.2):
1. Preservar acentuação do português (á, à, â, ã, é, ê, í, ó, ô, õ, ú, ü, ç),
   evitando ambiguidades lexicais (ex.: "é" vs "e", "está" vs "esta", "ótimo").
2. Nunca remover palavras de negação (ex.: "não", "nunca", "nem", "jamais", "sem")
   ao filtrar stopwords, pois a lista padrão do NLTK inclui "não", "nem" e "sem",
   o que inverteria a polaridade de frases como "não recomendo" -> "recomendo".
3. Preservar modificadores de intensidade e conectivos adversativos ("muito",
   "mais", "mas", "porém", "pouco") por carregarem sinal relevante em avaliações
   mistas (ex.: "produto bom, mas entrega ruim").
4. Suporte opcional a marcação de negação (negation marking: "não_gostei") para
   experimentos comparativos com unigramas.
"""

from __future__ import annotations

import html
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Optional, Set

import pandas as pd


def _register_local_nltk_data() -> None:
    """Adiciona diretórios locais de nltk_data (.venv/nltk_data) ao nltk.data.path."""
    try:
        import nltk

        candidates = [
            Path(sys.prefix) / "nltk_data",
            Path(__file__).resolve().parent.parent / ".venv" / "nltk_data",
            Path(__file__).resolve().parent.parent / "nltk_data",
        ]
        for cand in candidates:
            if cand.exists():
                cand_str = str(cand)
                if cand_str not in nltk.data.path:
                    nltk.data.path.insert(0, cand_str)
    except ImportError:
        pass


_register_local_nltk_data()

# ---------------------------------------------------------------------------
# Conjuntos linguísticos protegidos (Português)
# ---------------------------------------------------------------------------

# Palavras de negação que JAMAIS devem ser removidas (Seções 2.4.3 e 2.5.2)
NEGATION_WORDS: Set[str] = {
    "não",
    "nao",
    "nem",
    "nunca",
    "jamais",
    "nada",
    "nenhum",
    "nenhuma",
    "ninguém",
    "ninguem",
    "tampouco",
    "sem",
}

# Gatilhos utilizados quando mark_negation=True (ex.: "não" + "gostei" -> "não_gostei")
NEGATION_TRIGGERS: Set[str] = {
    "não",
    "nao",
    "nunca",
    "jamais",
    "nem",
    "sem",
}

# Modificadores de intensidade e conectivos de contraste úteis em análise de sentimentos
POLARITY_MODIFIERS: Set[str] = {
    "muito",
    "muita",
    "muitos",
    "muitas",
    "mais",
    "menos",
    "pouco",
    "pouca",
    "poucos",
    "poucas",
    "bastante",
    "demais",
    "apenas",
    "só",
    "so",
    "mas",
    "porém",
    "porem",
    "contudo",
    "entretanto",
    "todavia",
    "embora",
    "mal",
    "bem",
}

# Expressões regulares compiladas para alta performance
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"(https?://\S+|www\.\S+)", flags=re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", flags=re.IGNORECASE)
# Reduz alongamentos exagerados (ex.: "muitooooo" -> "muitoo", "ótimoooo" -> "ótimoo")
# mantendo letras duplas válidas em português ("rr", "ss", "ee", "oo")
_ELONGATION_RE = re.compile(r"([a-záàâãéêíóôõúüç])\1{2,}")
# Tokenizador alfabético que PRESERVA integralmente acentos e cedilha do português
_PT_TOKEN_RE = re.compile(r"[a-záàâãéêíóôõúüç]+")
_MULTI_SPACE_RE = re.compile(r"\s+")

# Lista oficial completa das 207 stopwords em português do NLTK (Snowball/NLTK),
# usada como fallback caso nltk_data não esteja acessível ou haja restrição de pathsec no Windows.
_NLTK_OFFICIAL_PT_STOPWORDS: Set[str] = {
    "a", "à", "ao", "aos", "aquela", "aquelas", "aquele", "aqueles", "aquilo",
    "as", "às", "até", "com", "como", "da", "das", "de", "dela", "delas",
    "dele", "deles", "depois", "do", "dos", "e", "é", "ela", "elas", "ele",
    "eles", "em", "entre", "era", "eram", "éramos", "essa", "essas", "esse",
    "esses", "esta", "está", "estamos", "estão", "estar", "estas", "estava",
    "estavam", "estávamos", "este", "esteja", "estejam", "estejamos", "estes",
    "esteve", "estive", "estivemos", "estiver", "estivera", "estiveram",
    "estivéramos", "estiverem", "estivermos", "estivesse", "estivessem",
    "estivéssemos", "estou", "eu", "foi", "fomos", "for", "fora", "foram",
    "fôramos", "forem", "formos", "fosse", "fossem", "fôssemos", "fui", "há",
    "haja", "hajam", "hajamos", "hão", "havemos", "haver", "hei", "houve",
    "houvemos", "houver", "houvera", "houverá", "houveram", "houvéramos",
    "houverão", "houverei", "houverem", "houveremos", "houveria", "houveriam",
    "houveríamos", "houvermos", "houvesse", "houvessem", "houvéssemos",
    "isso", "isto", "já", "lhe", "lhes", "mais", "mas", "me", "mesmo", "meu",
    "meus", "minha", "minhas", "muito", "na", "não", "nas", "nem", "no", "nos",
    "nós", "nossa", "nossas", "nosso", "nossos", "num", "numa", "o", "os",
    "ou", "para", "pela", "pelas", "pelo", "pelos", "por", "qual", "quando",
    "que", "quem", "são", "se", "seja", "sejam", "sejamos", "sem", "ser",
    "será", "serão", "serei", "seremos", "seria", "seriam", "seríamos", "seu",
    "seus", "só", "somos", "sou", "sua", "suas", "também", "te", "tem", "tém",
    "temos", "tenha", "tenham", "tenhamos", "tenho", "terá", "terão", "terei",
    "teremos", "teria", "teriam", "teríamos", "teu", "teus", "teve", "tinha",
    "tinham", "tínhamos", "tive", "tivemos", "tiver", "tivera", "tiveram",
    "tivéramos", "tiverem", "tivermos", "tivesse", "tivessem", "tivéssemos",
    "tu", "tua", "tuas", "um", "uma", "você", "vocês", "vos",
}


@lru_cache(maxsize=8)
def get_nltk_raw_stopwords() -> Set[str]:
    """
    Retorna o conjunto bruto oficial das 207 stopwords em português do NLTK (sem proteção).
    Trata automaticamente LookupError e restrições do pathsec do NLTK 3.10+ em Windows/NTFS.
    """
    try:
        from nltk.corpus import stopwords

        return set(stopwords.words("portuguese"))
    except (LookupError, PermissionError, OSError):
        # Tenta ler diretamente o arquivo oficial `corpora/stopwords/portuguese` em nltk.data.path
        try:
            import nltk

            for base_dir in nltk.data.path:
                sw_file = Path(base_dir) / "corpora" / "stopwords" / "portuguese"
                if sw_file.is_file():
                    with open(sw_file, "r", encoding="utf-8") as f:
                        words = {line.strip() for line in f if line.strip()}
                    if words:
                        return words
        except Exception:
            pass
        return set(_NLTK_OFFICIAL_PT_STOPWORDS)


@lru_cache(maxsize=8)
def get_portuguese_stopwords(
    keep_negations: bool = True,
    keep_modifiers: bool = True,
    extra_stopwords: Optional[tuple] = None,
) -> Set[str]:
    """
    Constrói o conjunto customizado de stopwords do português.

    Parâmetros
    ----------
    keep_negations : bool
        Se True (padrão obrigatório), remove NEGATION_WORDS das stopwords
        para que "não", "nem", "nunca", "sem" etc. sejam preservados no texto.
    keep_modifiers : bool
        Se True (padrão), preserva também intensificadores e adversativas
        ("muito", "mais", "mas", "só", "pouco").
    extra_stopwords : tuple | None
        Palavras adicionais opcionais a excluir do texto.
    """
    sw = set(get_nltk_raw_stopwords())
    if keep_negations:
        sw -= NEGATION_WORDS
    if keep_modifiers:
        sw -= POLARITY_MODIFIERS
    if extra_stopwords:
        sw |= set(extra_stopwords)
    return sw


def get_protected_terms_removed_from_nltk() -> List[str]:
    """
    Retorna a lista ordenada dos termos de negação/polaridade que originalmente
    estavam na lista padrão do NLTK e foram explicitamente protegidos.
    Útil para documentação no relatório técnico e na Aba 1 do Streamlit.
    """
    raw = get_nltk_raw_stopwords()
    protected = NEGATION_WORDS | POLARITY_MODIFIERS
    return sorted(raw.intersection(protected))


def normalize_text(text: str, normalize_elongations: bool = True) -> str:
    """
    Etapa de normalização preservando integralmente a acentuação em português:
    - Decodifica entidades HTML e remove tags HTML
    - Remove URLs e endereços de e-mail
    - Converte para minúsculas
    - Normaliza repetições exageradas de caracteres (ex.: "muitooooo" -> "muitoo")
    """
    if not isinstance(text, str):
        if pd.isna(text):
            return ""
        text = str(text)

    cleaned = html.unescape(text)
    cleaned = _HTML_TAG_RE.sub(" ", cleaned)
    cleaned = _URL_RE.sub(" ", cleaned)
    cleaned = _EMAIL_RE.sub(" ", cleaned)
    cleaned = cleaned.lower()

    if normalize_elongations:
        cleaned = _ELONGATION_RE.sub(r"\1\1", cleaned)

    cleaned = _MULTI_SPACE_RE.sub(" ", cleaned).strip()
    return cleaned


def tokenize_portuguese(text: str, method: str = "regex") -> List[str]:
    """
    Tokeniza o texto já normalizado preservando acentos do português.

    Parâmetros
    ----------
    text : str
        Texto normalizado em minúsculas.
    method : {'regex', 'nltk'}
        - 'regex' (padrão): extrai tokens alfabéticos PT-BR via regex (muito rápido em 130k textos).
        - 'nltk': usa `nltk.word_tokenize(..., language='portuguese')` e filtra tokens alfabéticos PT-BR.
    """
    if not text:
        return []

    if method == "nltk":
        try:
            from nltk.tokenize import word_tokenize

            raw_tokens = word_tokenize(text, language="portuguese")
            tokens: List[str] = []
            for tok in raw_tokens:
                tokens.extend(_PT_TOKEN_RE.findall(tok))
            return tokens
        except (LookupError, PermissionError, OSError):
            try:
                from nltk.tokenize import RegexpTokenizer

                return RegexpTokenizer(r"[a-záàâãéêíóôõúüç]+").tokenize(text)
            except ImportError:
                pass

    return _PT_TOKEN_RE.findall(text)


def apply_negation_marking(tokens: List[str]) -> List[str]:
    """
    Combina um termo de negação (ex.: 'não', 'nunca', 'nem') com o token seguinte,
    produzindo bigramas explícitos como 'não_recomendo', 'nunca_mais', 'sem_defeito'.
    Útil em experimentos comparativos com modelos baseados em unigramas.
    """
    if not tokens:
        return []

    marked: List[str] = []
    i = 0
    n = len(tokens)
    while i < n:
        tok = tokens[i]
        if tok in NEGATION_TRIGGERS and i + 1 < n:
            next_tok = tokens[i + 1]
            # Evita juntar dois gatilhos sem sentido caso haja duplicação acidental ("não não")
            if next_tok != tok:
                marked.append(f"{tok}_{next_tok}")
                i += 2
                continue
        marked.append(tok)
        i += 1
    return marked


def preprocess_text(
    text: str,
    remove_stopwords: bool = True,
    keep_negations: bool = True,
    keep_modifiers: bool = True,
    mark_negation: bool = False,
    min_token_len: int = 2,
    tokenizer: str = "regex",
    normalize_elongations: bool = True,
) -> str:
    """
    Função principal de pré-processamento (Contrato de Interface para Pessoa 2 e Pessoa 3).

    Recebe uma avaliação textual bruta (string) e retorna o texto tratado pronto para
    vetorização (Bag of Words / TF-IDF) ou inferência em tempo real no Streamlit.

    Parâmetros
    ----------
    text : str
        Texto bruto da avaliação.
    remove_stopwords : bool
        Se True (padrão), remove stopwords do português.
    keep_negations : bool
        Se True (padrão obrigatório), preserva palavras de negação ("não", "nunca", "nem", "sem" etc.).
    keep_modifiers : bool
        Se True (padrão), preserva intensificadores/adversativas ("muito", "mais", "mas", "pouco" etc.).
    mark_negation : bool
        Se True, combina negador + palavra seguinte (ex.: "não_gostei"). Padrão False
        (mantém separado para uso ideal com ngram_range=(1, 2)).
    min_token_len : int
        Comprimento mínimo do token a manter (padrão: 2 caracteres).
    tokenizer : {'regex', 'nltk'}
        Estratégia de tokenização preservando acentos.
    normalize_elongations : bool
        Se True, reduz repetições de 3+ letras iguais ("otimoooo" -> "otimoo").
    """
    normalized = normalize_text(text, normalize_elongations=normalize_elongations)
    if not normalized:
        return ""

    tokens = tokenize_portuguese(normalized, method=tokenizer)

    if remove_stopwords:
        sw = get_portuguese_stopwords(
            keep_negations=keep_negations,
            keep_modifiers=keep_modifiers,
        )
        tokens = [t for t in tokens if t not in sw]

    if min_token_len > 1:
        tokens = [t for t in tokens if len(t) >= min_token_len]

    if mark_negation:
        tokens = apply_negation_marking(tokens)

    return " ".join(tokens)


def preprocess_series(
    series: pd.Series,
    remove_stopwords: bool = True,
    keep_negations: bool = True,
    keep_modifiers: bool = True,
    mark_negation: bool = False,
    min_token_len: int = 2,
    tokenizer: str = "regex",
    normalize_elongations: bool = True,
) -> pd.Series:
    """Aplica `preprocess_text` a uma `pd.Series` inteira mantendo o índice original."""
    return series.fillna("").astype(str).map(
        lambda t: preprocess_text(
            t,
            remove_stopwords=remove_stopwords,
            keep_negations=keep_negations,
            keep_modifiers=keep_modifiers,
            mark_negation=mark_negation,
            min_token_len=min_token_len,
            tokenizer=tokenizer,
            normalize_elongations=normalize_elongations,
        )
    )


def compare_preprocessing(texts: Iterable[str]) -> pd.DataFrame:
    """
    Gera tabela comparativa lado a lado mostrando:
    1. Texto original
    2. Stopwords padrão do NLTK (SEM proteger negações — ilustra o erro evitado!)
    3. Pipeline recomendado (acentos + negações preservadas)
    4. Pipeline com Negation Marking ("não_palavra")
    """
    rows = []
    for text in texts:
        rows.append(
            {
                "texto_original": text,
                "nltk_padrao_sem_protecao": preprocess_text(
                    text,
                    remove_stopwords=True,
                    keep_negations=False,
                    keep_modifiers=False,
                ),
                "pipeline_recomendado": preprocess_text(
                    text,
                    remove_stopwords=True,
                    keep_negations=True,
                    keep_modifiers=True,
                    mark_negation=False,
                ),
                "com_negation_marking": preprocess_text(
                    text,
                    remove_stopwords=True,
                    keep_negations=True,
                    keep_modifiers=True,
                    mark_negation=True,
                ),
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("=== Teste Unitário de Pré-processamento PLN (src/preprocessing.py) ===")
    protected_from_nltk = get_protected_terms_removed_from_nltk()
    print(f"Total stopwords NLTK padrão:      {len(get_nltk_raw_stopwords())}")
    print(f"Total stopwords customizadas:     {len(get_portuguese_stopwords())}")
    print(f"Termos retirados da lista NLTK:   {protected_from_nltk}")

    exemplos = [
        "O produto não funciona direito e não recomendo para ninguém!",
        "Péssima experiência: nunca mais compro aqui, chegou sem carregador.",
        "Ótimo celular, muito rápido, mas a entrega demorou demaaaais!",
        "Não é ruim, porém esperava muito mais pela qualidade anunciada.",
        "Chegou no prazo! Veja em https://americanas.com.br/produto <b>Excelente</b>.",
    ]

    df_comp = compare_preprocessing(exemplos)
    for idx, row in df_comp.iterrows():
        print(f"\n--- Exemplo {idx + 1} ---")
        print("Original:             ", row["texto_original"])
        print("NLTK s/ proteção (!): ", row["nltk_padrao_sem_protecao"])
        print("Pipeline recomendado: ", row["pipeline_recomendado"])
        print("Com negation marking: ", row["com_negation_marking"])

    # Asserções rigorosas de integridade
    assert "não" in preprocess_text("Não gostei do produto"), "Palavra 'não' foi removida!"
    assert "nunca" in preprocess_text("Nunca mais compro"), "Palavra 'nunca' foi removida!"
    assert "ótimo" in preprocess_text("Ótimo produto"), "Acento em 'ótimo' foi perdido!"
    assert "péssima" in preprocess_text("Péssima compra"), "Acento em 'péssima' foi perdido!"
    assert "não_recomendo" in preprocess_text("Não recomendo", mark_negation=True)
    print("\n[OK] Todas as asserções de acentuação e negação passaram!")
