# Guia de Integração e Tarefas – Pessoa 2 e Pessoa 3

Este documento explica exatamente como configurar o ambiente após clonar o repositório, o que a **Pessoa 1** já deixou pronto e o **checklist completo** do que a **Pessoa 2 (Modelagem)** e a **Pessoa 3 (Avaliação Crítica & Streamlit)** precisam desenvolver e entregar.

---

## Passo 0 – Preparação Inicial (Obrigatório ao clonar o repositório)

> **Atenção sobre a pasta `data/`:** Por exigência da licença (**CC BY-NC-SA 4.0**) e da Seção 6 do edital do professor, **o arquivo `B2W-Reviews01.csv` e os `.parquet` NÃO sobem no GitHub** (estão bloqueados no `.gitignore`). Por isso, quando você clonar o repositório, a pasta `data/` virá vazia!

Siga estes 3 passos antes de rodar qualquer código:

### 1. Clonar, criar ambiente virtual e instalar dependências (Windows PowerShell)
```powershell
git clone https://github.com/guizyre/projeto-ia-sentimentos-b2w.git
cd projeto-ia-sentimentos-b2w

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Obter o dataset `B2W-Reviews01.csv` (escolha a Opção A ou B)
- **Opção A (Download automático por comando — mais rápida):**
  ```powershell
  python -c "from src.data_loader import download_b2w_dataset; print('Salvo em:', download_b2w_dataset())"
  ```
- **Opção B (Download manual pelo navegador):**
  1. Acesse https://github.com/americanas-tech/b2w-reviews01
  2. Clique em `B2W-Reviews01.csv` -> botão **Download raw file**.
  3. Salve o arquivo exatamente em `data/B2W-Reviews01.csv` (cuidado se o navegador salvar como `B2W-Reviews01 (1).csv` — renomeie para `B2W-Reviews01.csv`).

### 3. Verificar se está tudo funcionando
```powershell
python -m src.data_loader
python -m src.preprocessing
```
*(Opcional: se quiser gerar os arquivos rápidos `data/b2w_processed_full.parquet` e `data/b2w_processed_30k.parquet` localmente, basta rodar o notebook `notebooks/02_preprocessamento_pln.ipynb` — ou carregar direto usando `load_and_clean_data(apply_preprocessing=True)` no seu código!)*

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
| `reports/relatorio_parte_pessoa1.md` | Texto pronto das Seções 1 a 4 do Relatório Técnico + roteiro oral da Pessoa 1. |

---

## 2. Contrato de Dados (Colunas Garantidas)

Ao chamar `load_and_clean_data(apply_preprocessing=True)`, o `DataFrame` sempre contém:

- `review_text_original` (`str`): texto bruto original (preservado para auditoria e análise de erros).
- `review_text_clean` (`str`): texto pré-processado com acentos e negações preservados (pronto para `CountVectorizer` / `TfidfVectorizer`).
- `overall_rating` (`int`): nota original (`1` a `5`).
- `sentiment` (`str`): `'Negativo'`, `'Neutro'` ou `'Positivo'`.
- `sentiment_id` (`int`): `0` (Negativo), `1` (Neutro), `2` (Positivo).
- `site_category_lv1` (`str`): categoria principal do produto (usada na Pergunta 5 da Seção 2.6).
- `review_title`, `recommend_to_a_friend`, `char_count`, `word_count`: metadados auxiliares.

> **Importante (Privacidade – Seção 2.8):** Nenhuma coluna pessoal (`reviewer_id`, `reviewer_birth_year`, `reviewer_gender`, `reviewer_state`) existe no DataFrame retornado.

---

## 3. Guia e Checklist da Pessoa 2 (Representação Textual e Modelagem)

### Checklist do que a Pessoa 2 precisa entregar
- [ ] **`src/models.py`**:
  - Divisão estratificada **80% treino / 20% teste** (`stratify=y`, `random_state=42`).
  - **REGRA DE OURO (Seção 2.5.4 — Sem Data Leakage):** o `.fit()` do vetorizador e qualquer `GridSearchCV` devem rodar **apenas em `X_train`**. Em `X_test`, usar somente `.transform()`.
  - Implementar e comparar **4 representações** (mínimo exigido: 2):
    - `CountVectorizer(ngram_range=(1, 1), min_df=3)` *(Bag of Words – Unigramas)*
    - `CountVectorizer(ngram_range=(1, 2), min_df=3)` *(Bag of Words – Uni + Bigramas)*
    - `TfidfVectorizer(ngram_range=(1, 1), min_df=3, sublinear_tf=True)` *(TF-IDF – Unigramas)*
    - `TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True)` *(TF-IDF – Uni + Bigramas)*
  - Treinar e comparar **3 classificadores** (mínimo exigido: 2):
    - `MultinomialNB` *(Naive Bayes)*
    - `LogisticRegression(max_iter=1000)` *(testar padrão vs `class_weight='balanced'`)*
    - `LinearSVC(class_weight='balanced', random_state=42)` *(SVM Linear)*
  - Salvar os melhores vetorizadores e modelos em `models_saved/*.joblib` (e um JSON/CSV com os resultados comparativos em `reports/model_results.json` para a Pessoa 3 usar na Aba 2 do Streamlit!).
- [ ] **`notebooks/03_treinamento_modelos.ipynb`**:
  - Executar todos os experimentos comparativos (BoW vs TF-IDF × uni/bigramas × classificadores).
  - Extrair e plotar os **termos com maior peso positivo e negativo** (via `coef_` da Regressão Logística/SVM).
  - *(Bônus fácil)* Testar também `mark_negation=True` em unigramas usando `preprocess_series(..., mark_negation=True)`.
- [ ] **Relatório Técnico + Apresentação Oral (Bloco 2 — 3 a 4 min):**
  - Escrever a metodologia de representação vetorial, divisão treino/teste, classificadores, ajuste de hiperparâmetros e tabelas quantitativas de comparação.

### Código inicial pronto para a Pessoa 2
```python
from sklearn.model_selection import train_test_split
from src.data_loader import load_and_clean_data

# Carrega direto do CSV (usa sample_size=30000 para testes rápidos ou None para o corpus completo de 126.690)
df = load_and_clean_data(sample_size=None, apply_preprocessing=True, auto_download=True)

X = df["review_text_clean"]
y = df["sentiment_id"]  # 0=Negativo, 1=Neutro, 2=Positivo

# Divisão estratificada 80/20 SEM data leakage (Seção 2.5.4)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

# DICA IMPORTANTE:
# Use token_pattern=r"[a-záàâãéêíóôõúüç_]+" no CountVectorizer/TfidfVectorizer
# para preservar acentos do português e underscores (caso teste mark_negation=True)!
```

---

## 4. Guia e Checklist da Pessoa 3 (Avaliação Crítica, Ética e App Streamlit)

### Checklist do que a Pessoa 3 precisa entregar
- [ ] **`src/evaluation.py`** e **`notebooks/04_avaliacao_e_erros.ipynb`**:
  - Calcular **Acurácia**, **Precisão**, **Recall**, **F1-Score por classe** (Negativo, Neutro, Positivo) e dar **destaque principal ao F1-Score Macro** (exigência da Seção 2.5.5 devido ao desequilíbrio de classes).
  - Gerar **Matrizes de Confusão** (heatmaps Seaborn salvos em `reports/figures/`).
  - Responder com dados e exemplos reais do `X_test` às **5 perguntas obrigatórias da Seção 2.6**:
    1. Qual configuração (representação + classificador) teve melhor resultado?
    2. O modelo consegue identificar adequadamente avaliações negativas? (Recall e F1 Negativo)
    3. Quais erros aparecem com maior frequência? *(Extrair 10 erros reais do `X_test`, destacando classe Neutra e avaliações mistas produto vs entrega)*
    4. A regra de conversão de notas em sentimentos introduz ruído nos rótulos? *(Exemplos onde nota=1 mas texto elogia, ou nota=5 com "ainda não testei")*
    5. O desempenho se mantém em categorias diferentes? *(Calcular F1-Macro separado nas Top 5 categorias + `Móveis` usando `df.loc[X_test.index, "site_category_lv1"]`)*
- [ ] **`app.py` (Interface Demonstrável em Streamlit — 3 Abas):**
  - **Aba 1 – Corpus, EDA e Pré-processamento:** carregar `reports/eda_summary.json`, `reports/preprocessing_summary.json` e as figuras `reports/figures/01..08_*.png`; mostrar antes vs depois do pré-processamento e explicação da regra de notas.
  - **Aba 2 – Comparação de Modelos:** exibir tabela de métricas (com destaque ao F1-Macro), matrizes de confusão e desempenho por categoria.
  - **Aba 3 – Simulador ao Vivo (Textos Inéditos):** caixa de texto onde o usuário/professor digita uma nova avaliação, escolhe o modelo e vê: texto tratado por `preprocess_text()`, sentimento previsto e probabilidades/confiança.
- [ ] **Relatório Técnico + Apresentação Oral (Bloco 3 — 3 a 4 min):**
  - Escrever as respostas da Seção 2.6, análise crítica de limitações/vieses/ética (Seção 2.8), prints do Streamlit e conduzir a **demonstração ao vivo** na apresentação.

### Código inicial pronto para a Pessoa 3

#### A) Carregando os resumos da Pessoa 1 na Aba 1 do Streamlit (instantâneo, sem ler 50 MB)
```python
import json
from pathlib import Path

eda = json.loads(Path("reports/eda_summary.json").read_text(encoding="utf-8"))
pln = json.loads(Path("reports/preprocessing_summary.json").read_text(encoding="utf-8"))
# Exibir gráficos salvos em reports/figures/01..08_*.png com st.image(...)
```

#### B) Simulador em tempo real na Aba 3 do Streamlit
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

---

## 5. Boas Práticas de Git para o Trio
- Sempre rode `git pull` antes de começar a editar.
- Cada pessoa edita seus arquivos dedicados (`models.py` / `03_*.ipynb` pela Pessoa 2; `evaluation.py` / `04_*.ipynb` / `app.py` pela Pessoa 3) para evitar conflitos de merge.
- **Nunca** remova `data/*.csv` ou `.venv/` do `.gitignore`.
