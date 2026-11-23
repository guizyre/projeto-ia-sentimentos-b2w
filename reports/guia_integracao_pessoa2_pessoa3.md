# Guia de Integração – Entrega da Pessoa 1 para Pessoa 2 e Pessoa 3

Este documento explica exatamente o que já está pronto, testado e padronizado na base do projeto para que a **Pessoa 2 (Modelagem)** e a **Pessoa 3 (Avaliação Crítica & Streamlit)** possam trabalhar imediatamente sem retrabalho.

---

## 1. O que já foi entregue pela Pessoa 1

| Arquivo | Função |
| :--- | :--- |
| `src/data_loader.py` | Download/carregamento do B2W-Reviews01, remoção de colunas pessoais (`reviewer_id`, idade, gênero, estado), limpeza de nulos/duplicatas, mapeamento de notas e amostragem estratificada. |
| `src/preprocessing.py` | Normalização preservando **acentos**, tokenização PT-BR, remoção de *stopwords* **protegendo negações e modificadores** (`não`, `nunca`, `nem`, `sem`, `muito`, `mas`, `mais`, `só`) e opção de *negation marking*. |
| `notebooks/01_eda_e_limpeza.ipynb` | Auditoria completa, análise de ruído nos rótulos, gráficos de notas, classes, tamanho das avaliações e categorias. |
| `notebooks/02_preprocessamento_pln.ipynb` | Comparação antes/depois, prova empírica do impacto das negações, Top 20 palavras, Top 15 bigramas por classe e WordClouds. |
| `reports/figures/01..08_*.png` | 8 gráficos prontos em 300 DPI para o Relatório Técnico e para a Aba 1 do Streamlit. |
| `reports/eda_summary.json` | Resumo JSON pronto com todas as estatísticas da EDA (carregamento instantâneo no Streamlit). |
| `reports/preprocessing_summary.json` | Resumo JSON com estatísticas de vocabulário, stopwords protegidas, negações por classe e exemplos reais antes/depois. |
| `data/b2w_processed_full.parquet` | Corpus completo limpo e pré-processado (`126.690` avaliações) — *ignorado no Git; gerado ao rodar o Notebook 02*. |
| `data/b2w_processed_30k.parquet` | Amostra estratificada de `30.000` avaliações já pré-processada (proporção idêntica ao corpus completo). |

---

## 2. Contrato de Dados (Colunas Garantidas)

Ao chamar `load_and_clean_data(apply_preprocessing=True)` ou carregar os arquivos `.parquet`, o `DataFrame` sempre contém:

- `review_text_original` (`str`): texto bruto original (preservado para auditoria e análise de erros).
- `review_text_clean` (`str`): texto pré-processado com acentos e negações preservados (pronto para `CountVectorizer` / `TfidfVectorizer`).
- `overall_rating` (`int`): nota original (`1` a `5`).
- `sentiment` (`str`): `'Negativo'`, `'Neutro'` ou `'Positivo'`.
- `sentiment_id` (`int`): `0` (Negativo), `1` (Neutro), `2` (Positivo).
- `site_category_lv1` (`str`): categoria principal do produto (usada na Pergunta 5 da Seção 2.6).
- `review_title`, `recommend_to_a_friend`, `char_count`, `word_count`: metadados auxiliares.

> **Importante (Privacidade – Seção 2.8):** Nenhuma coluna pessoal (`reviewer_id`, `reviewer_birth_year`, `reviewer_gender`, `reviewer_state`) existe no DataFrame retornado.

---

## 3. Como a Pessoa 2 deve usar (Representação e Modelos)

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from src.data_loader import load_and_clean_data

# Opção A (mais rápida, se já executou o Notebook 02):
df = pd.read_parquet("data/b2w_processed_full.parquet")
# Ou para testes rápidos com amostra estratificada de 30k:
# df = pd.read_parquet("data/b2w_processed_30k.parquet")

# Opção B (direto do CSV bruto usando a função oficial):
# df = load_and_clean_data(sample_size=None, apply_preprocessing=True)

X = df["review_text_clean"]
y = df["sentiment_id"]  # 0=Negativo, 1=Neutro, 2=Positivo

# Divisão estratificada 80/20 SEM data leakage (Seção 2.5.4)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

# DICA IMPORTANTE:
# Use token_pattern=r"(?u)\b\w+\b" (ou r"[a-záàâãéêíóôõúüç_]+") no CountVectorizer/TfidfVectorizer
# para garantir que acentos e underscores (caso teste mark_negation=True) sejam preservados!
```

### Experimento extra fácil para a Pessoa 2 (bônus no relatório)
Você pode comparar também o efeito do *Negation Marking* (`não_recomendo`) rodando:
```python
from src.preprocessing import preprocess_series
X_neg_marked = preprocess_series(df["review_text_original"], mark_negation=True)
```

---

## 4. Como a Pessoa 3 deve usar (Avaliação Crítica & Streamlit)

### A) Na Aba 1 do Streamlit (EDA e Pré-processamento)
Você não precisa ler o CSV de 50 MB toda vez que o app abrir — basta carregar os resumos JSON e as figuras prontas:

```python
import json
from pathlib import Path

eda = json.loads(Path("reports/eda_summary.json").read_text(encoding="utf-8"))
pln = json.loads(Path("reports/preprocessing_summary.json").read_text(encoding="utf-8"))

# Exibir métricas e gráficos já salvos em reports/figures/01..08_*.png
```

### B) Na Aba 3 do Streamlit (Simulador de Novas Avaliações ao Vivo)
Use exatamente a mesma função `preprocess_text()` usada no treino:

```python
from src.preprocessing import preprocess_text
from src.data_loader import ID_TO_SENTIMENT

texto_usuario = "O celular é ótimo, mas a bateria não dura nada!"
texto_tratado = preprocess_text(texto_usuario)
# -> "celular ótimo mas bateria não dura nada"

# Passar `texto_tratado` no vetorizador + modelo salvo pela Pessoa 2:
# X_novo = vectorizer.transform([texto_tratado])
# pred_id = model.predict(X_novo)[0]
# sentimento = ID_TO_SENTIMENT[pred_id]
```

### C) Na Análise de Erros por Categoria (Pergunta 5 da Seção 2.6)
O índice do `train_test_split` preserva os índices originais de `df`, então basta recuperar `df.loc[X_test.index, "site_category_lv1"]` e `df.loc[X_test.index, "review_text_original"]` para listar os erros reais e calcular o **F1-Score Macro** nas Top 5 categorias (`Celulares e Smartphones`, `Eletroportáteis`, `Beleza e Perfumaria`, `Utilidades Domésticas`, `TV e Home Theater` e também `Móveis`, que possui 43,9% de avaliações negativas!).
