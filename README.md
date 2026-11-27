# Projeto 2 – Análise de Sentimentos em Avaliações de Produtos (PLN)

Projeto prático da disciplina de **Inteligência Artificial (Unidade 3)** — **IFPB** (Bacharelado em Engenharia da Computação).  
**Professor:** Marcelo José Siqueira Coutinho de Almeida

## Equipe Responsável
| Integrante | Papel no Projeto | Módulos Principais |
| :--- | :--- | :--- |
| **Guilherme Gonçalves** | Engenharia de Dados, EDA e Pré-processamento PLN | `src/data_loader.py`, `src/preprocessing.py`, `notebooks/01_*.ipynb`, `02_*.ipynb` |
| **Anaíza Silva** | Representação Vetorial e Modelagem Supervisionada | `src/models.py`, `notebooks/03_treinamento_modelos.ipynb`, `models_saved/*.joblib` |
| **Hillary Maximino** | Avaliação Crítica, Ética e Interface Streamlit | `src/evaluation.py`, `notebooks/04_avaliacao_e_erros.ipynb`, `app.py`, `tests/test_pipeline.py` |

---

## Objetivo
Desenvolver um pipeline experimental completo de **Processamento de Linguagem Natural (PLN)** em Python para classificar avaliações reais de produtos em português em **Negativo**, **Neutro** ou **Positivo**, comparando representações vetoriais (*Bag of Words* vs. *TF-IDF*, unigramas vs. bigramas) e algoritmos supervisionados sem vazamento de dados, acompanhado de uma aplicação interativa em **Streamlit** com explicabilidade local por token.

---

## Corpus, Licença e Privacidade
- **Dataset:** [B2W-Reviews01](https://github.com/americanas-tech/b2w-reviews01) — versão `1.0` (repositório oficial Americanas Tech; `132.373` registros brutos coletados no site Americanas.com entre 01/01/2018 e 31/05/2018; `126.690` avaliações válidas após limpeza e pré-processamento).
- **Data de acesso/obtenção:** `23 de novembro de 2026`.
- **Espelho alternativo:** [Hugging Face (`ruanchaves/b2w-reviews01`)](https://huggingface.co/datasets/ruanchaves/b2w-reviews01).
- **Licença:** `CC BY-NC-SA 4.0` (*Creative Commons Atribuição-NãoComercial-CompartilhaIgual 4.0 Internacional*).
- **Privacidade (Seção 2.8 do Edital):** As colunas pessoais/demográficas (`reviewer_id`, `reviewer_birth_year`, `reviewer_gender`, `reviewer_state`) são descartadas imediatamente na leitura e jamais armazenadas nos artefatos ou exibidas na interface.
- **Obtenção dos dados:** O arquivo bruto `B2W-Reviews01.csv` não é redistribuído no repositório Git (`.gitignore`). Use `download_b2w_dataset()` em `src/data_loader.py` ou baixe-o do repositório oficial para `data/B2W-Reviews01.csv`.

---

## Estrutura do Repositório

```text
projeto-ia-sentimentos-b2w/
├── .streamlit/
│   └── config.toml                          # Tema visual unificado da aplicação Streamlit
├── data/                                    # Corpus bruto (ignorado no Git por licença)
├── notebooks/
│   ├── 01_eda_e_limpeza.ipynb               # Guilherme Gonçalves: auditoria, limpeza e EDA
│   ├── 02_preprocessamento_pln.ipynb        # Guilherme Gonçalves: acentos, negações e WordClouds
│   ├── 03_treinamento_modelos.ipynb         # Anaíza Silva: BoW vs TF-IDF, treino e tuning
│   └── 04_avaliacao_e_erros.ipynb           # Hillary Maximino: matrizes, erros, ruído e categorias
├── src/
│   ├── __init__.py
│   ├── data_loader.py                       # Guilherme Gonçalves: ingestão, privacidade e rotulagem
│   ├── preprocessing.py                     # Guilherme Gonçalves: normalização (acentos + negações)
│   ├── models.py                            # Anaíza Silva: vetorização e classificadores
│   └── evaluation.py                        # Hillary Maximino: métricas, erros, ruído e explicabilidade
├── models_saved/                            # Pipelines e vetorizadores treinados (.joblib)
├── reports/
│   ├── figures/                             # Gráficos em 300 DPI (01–15) + capturas do Streamlit (16–18)
│   ├── eda_summary.json                     # Resumo quantitativo da EDA
│   ├── preprocessing_summary.json           # Resumo quantitativo do PLN
│   ├── model_results.json                   # Métricas das 16 combinações, tuning e modelo final
│   ├── evaluation_summary.json              # Erros reais, ruído de rótulo, categorias e binário
│   ├── relatorio_parte_pessoa1.md           # Seção individual — Guilherme Gonçalves
│   ├── relatorio_pessoa2.md                 # Seção individual — Anaíza Silva
│   ├── relatorio_pessoa3.md                 # Seção individual — Hillary Maximino
│   ├── relatorio_tecnico_completo.md        # Relatório Técnico Final consolidado (com figuras)
│   └── roteiro_apresentacao_trio.md         # Roteiro cronometrado da apresentação oral e demo ao vivo
├── tests/
│   └── test_pipeline.py                     # Suíte de testes automatizados de ponta a ponta
├── app.py                                   # Aplicação demonstrável em Streamlit (3 abas + XAI)
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

3. **Execute a aplicação demonstrável em Streamlit (carrega instantaneamente dos artefatos salvos):**
   ```powershell
   streamlit run app.py
   ```

4. **Baixe o corpus bruto `B2W-Reviews01.csv` (apenas caso deseje reexecutar o treinamento/avaliação do zero):**
   ```powershell
   python -c "from src.data_loader import download_b2w_dataset; print('Salvo em:', download_b2w_dataset())"
   ```

5. **Reprodução completa dos módulos via linha de comando (opcional):**
   ```powershell
   python -m src.data_loader
   python -m src.preprocessing
   python -m src.models
   python -m src.evaluation
   ```

6. **Execute a suíte de testes integrados:**
   ```powershell
   python -m unittest discover -s tests -v
   ```
