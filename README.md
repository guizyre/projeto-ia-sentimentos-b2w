# Projeto 2 – Análise de Sentimentos em Avaliações de Produtos (PLN)

Projeto prático da disciplina de **Inteligência Artificial (Unidade 3)** — IFPB (Bacharelado em Engenharia da Computação).

## Objetivo
Desenvolver um pipeline completo de Processamento de Linguagem Natural (PLN) em Python para classificar avaliações de produtos em português em **Negativo**, **Neutro** ou **Positivo**, comparando diferentes representações textuais (*Bag of Words* e *TF-IDF*) e algoritmos de classificação, além de disponibilizar uma aplicação demonstrável em **Streamlit**.

---

## Corpus, Licença e Privacidade
- **Dataset:** [B2W-Reviews01](https://github.com/americanas-tech/b2w-reviews01) (132.373 avaliações reais em português coletadas no site Americanas.com entre janeiro e maio de 2018; 126.690 após limpeza e pré-processamento).
- **Alternativa / Espelho:** [Hugging Face (`ruanchaves/b2w-reviews01`)](https://huggingface.co/datasets/ruanchaves/b2w-reviews01).
- **Licença:** `CC BY-NC-SA 4.0` (uso não comercial, com atribuição obrigatória e compartilhamento pela mesma licença).
- **Privacidade (Seção 2.8):** As colunas pessoais/demográficas (`reviewer_id`, `reviewer_birth_year`, `reviewer_gender`, `reviewer_state`) são descartadas imediatamente na leitura e nunca exibidas na aplicação.
- **Como obter os dados:** O arquivo `B2W-Reviews01.csv` não é versionado neste repositório (ignorado via `.gitignore`). Baixe-o diretamente em [americanas-tech/b2w-reviews01](https://github.com/americanas-tech/b2w-reviews01) e salve em `data/B2W-Reviews01.csv` (ou use `download_b2w_dataset()` em `src/data_loader.py`).

---

## Estrutura do Projeto

```text
projeto/
├── data/                                    # Corpus bruto e .parquet processados (ignorados no Git)
├── notebooks/
│   ├── 01_eda_e_limpeza.ipynb               # [OK] Pessoa 1: auditoria, limpeza e gráficos EDA
│   ├── 02_preprocessamento_pln.ipynb        # [OK] Pessoa 1: acentos, negações, bigramas e WordClouds
│   ├── 03_treinamento_modelos.ipynb         # [OK] Pessoa 2: BoW vs TF-IDF, treino e hiperparâmetros
│   └── 04_avaliacao_e_erros.ipynb           # [OK] Pessoa 3: matrizes, erros reais, ruído e análise por categoria
├── src/
│   ├── __init__.py
│   ├── data_loader.py                       # [OK] Pessoa 1: carga, privacidade, limpeza e rotulagem
│   ├── preprocessing.py                     # [OK] Pessoa 1: normalização (acentos + negações)
│   ├── models.py                            # [OK] Pessoa 2: vetorização e classificadores
│   └── evaluation.py                        # [OK] Pessoa 3: métricas, F1-Macro, erros, ruído e explicabilidade
├── models_saved/                            # [OK] Pessoa 2: modelos e vetorizadores salvos (.joblib)
├── reports/
│   ├── figures/                             # [OK] 15 gráficos em 300 DPI (01–08: EDA/PLN; 09–11: modelos; 12–15: avaliação)
│   ├── eda_summary.json                     # [OK] Resumo quantitativo da EDA para o Streamlit
│   ├── preprocessing_summary.json           # [OK] Resumo quantitativo do PLN para o Streamlit
│   ├── model_results.json                   # [OK] Pessoa 2: métricas das 16 combinações, tuning e modelo final
│   ├── evaluation_summary.json              # [OK] Pessoa 3: erros reais, ruído de rótulo, categorias e cenário binário
│   ├── guia_integracao_pessoa2_pessoa3.md   # [OK] Contrato de interfaces e exemplos para a equipe
│   ├── relatorio_parte_pessoa1.md           # [OK] Pessoa 1: texto acadêmico (Seções 1–4) + roteiro oral
│   ├── relatorio_pessoa2.md                 # [OK] Pessoa 2: texto acadêmico (Seções 5–10) + roteiro oral
│   ├── relatorio_pessoa3.md                 # [OK] Pessoa 3: Seção 2.6, análise crítica, ética, Streamlit + roteiro oral
│   └── relatorio_tecnico_completo.md        # [OK] Consolidação completa do Relatório Técnico do trio
├── tests/
│   └── test_pipeline.py                     # [OK] Pessoa 3: testes integrados (dados, privacidade, PLN, modelos e app)
├── app.py                                   # [OK] Pessoa 3: interface demonstrável em Streamlit (3 abas)
├── requirements.txt
└── README.md
```

---

## Instalação e Execução (Windows PowerShell)

1. **Crie e ative o ambiente virtual:**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. **Instale as dependências:**
   ```powershell
   pip install -r requirements.txt
   ```

3. **Obtenha o corpus `B2W-Reviews01.csv` (não versionado por exigência da licença `CC BY-NC-SA 4.0`):**
   ```powershell
   python -c "from src.data_loader import download_b2w_dataset; print('Salvo em:', download_b2w_dataset())"
   ```

4. **Execute a aplicação demonstrável em Streamlit (carrega instantaneamente dos artefatos em `models_saved/` e `reports/`):**
   ```powershell
   streamlit run app.py
   ```

5. **Reprodução completa dos módulos via CLI (opcional, requer `data/B2W-Reviews01.csv`):**
   - **Pessoa 1 (auditoria do corpus e pré-processamento):**
     ```powershell
     python -m src.data_loader
     python -m src.preprocessing
     ```
   - **Pessoa 2 (treinamento e ajuste dos classificadores):**
     - Teste rápido em amostra estratificada *(sobrescreve `reports/model_results.json`)*:
       ```powershell
       python -m src.models 30000
       ```
     - Entrega completa sobre as `126.690` avaliações:
       ```powershell
       python -m src.models
       ```
   - **Pessoa 3 (avaliação experimental, análise de erros, ruído de rótulo e categorias):**
     ```powershell
     python -m src.evaluation
     ```

6. **Execute a suíte de testes integrados:**
   ```powershell
   python -m unittest discover -s tests -v
   ```