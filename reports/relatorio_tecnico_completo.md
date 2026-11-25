# Relatório Técnico Completo
**Projeto 2: Classificação Automática de Sentimentos em Avaliações de Produtos (PLN)**  
**Disciplina:** Inteligência Artificial – Unidade 3  
**Instituição:** Instituto Federal de Educação, Ciência e Tecnologia da Paraíba (IFPB) — Bacharelado em Engenharia da Computação  
**Professor:** Marcelo José Siqueira Coutinho de Almeida  
**Equipe:** Pessoa 1 (Dados, EDA e Pré-processamento PLN) · Pessoa 2 (Representação Textual e Modelagem) · Pessoa 3 (Avaliação Crítica, Ética e Aplicação Streamlit)

---

## 1. Introdução e Caracterização do Problema

Consumidores de plataformas de comércio eletrônico publicam diariamente milhares de avaliações textuais expressando percepções sobre qualidade, desempenho, custo-benefício e experiência de entrega dos produtos adquiridos. A análise manual desse volume de texto é inviável em larga escala, tornando a **Análise de Sentimentos** um problema clássico e relevante de Processamento de Linguagem Natural (PLN).

O objetivo deste projeto é desenvolver, avaliar e demonstrar um pipeline completo de PLN em Python capaz de receber uma avaliação textual em português e classificá-la automaticamente em três classes de sentimento: **Negativo**, **Neutro** ou **Positivo**. Em conformidade com as diretrizes da disciplina, a investigação vai além da execução de bibliotecas prontas, documentando e justificando empiricamente cada etapa de auditoria, limpeza, preservação linguística (acentuação e negações), representação vetorial, classificação, análise crítica de erros e demonstração interativa em Streamlit.

---

## 2. Caracterização do Corpus, Licença e Cuidados Éticos

### 2.1 Origem e Licença dos Dados
Utilizou-se o corpus aberto **B2W-Reviews01**, disponibilizado pela Americanas Tech no repositório oficial [`github.com/americanas-tech/b2w-reviews01`](https://github.com/americanas-tech/b2w-reviews01) (com espelho no Hugging Face em `ruanchaves/b2w-reviews01`).

- **Domínio e período de coleta:** Avaliações reais de clientes no site *Americanas.com* coletadas entre **1º de janeiro e 31 de maio de 2018** (`2018-01-01 00:11:28` a `2018-05-31 23:50:33`).
- **Idioma:** Português brasileiro (`pt-BR`).
- **Dimensão bruta:** `132.373` registros e `14` colunas.
- **Licença:** *Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International* (**CC BY-NC-SA 4.0**), que autoriza uso acadêmico e não comercial mediante atribuição obrigatória. Para respeitar as condições de distribuição, o arquivo bruto `B2W-Reviews01.csv` não é versionado no repositório Git (listado no `.gitignore`); o módulo `src/data_loader.py` e o `README.md` documentam integralmente o procedimento reproduzível de obtenção.

### 2.2 Seleção de Campos e Privacidade (Seção 2.8)
Dos 14 campos originais do corpus:
- **Campo de entrada principal:** `review_text` (texto livre da avaliação).
- **Campo base para o rótulo:** `overall_rating` (nota inteira de 1 a 5 estrelas).
- **Metadados analíticos mantidos:** `site_category_lv1` (categoria principal do produto, usada na análise de desempenho por categoria), `submission_date`, `review_title` e `recommend_to_a_friend` (auxiliar na auditoria de consistência).
- **Colunas pessoais/demográficas descartadas na leitura:** `reviewer_id`, `reviewer_birth_year`, `reviewer_gender` e `reviewer_state`. Nenhum dado pessoal ou atributo demográfico do consumidor é armazenado nos artefatos processados nem exibido na aplicação Streamlit, respeitando rigorosamente os cuidados éticos do projeto.

---

## 3. Limpeza, Rotulagem e Análise Exploratória dos Dados (EDA)

### 3.1 Auditoria e Tratamento de Registros
A auditoria executada em `src/data_loader.py` e no notebook `notebooks/01_eda_e_limpeza.ipynb` produziu o funil de limpeza a seguir:

| Etapa do Pipeline de Limpeza | Registros Restantes | Removidos na Etapa | % Retido |
| :--- | ---: | ---: | ---: |
| 0. Corpus bruto original (`B2W-Reviews01.csv`) | 132.373 | — | 100,00% |
| 1. Remoção de nulos em `review_text`/`overall_rating` ou nota inválida | 129.098 | 3.275 | 97,53% |
| 2. Remoção de textos vazios / sem conteúdo alfabético (`< 3` chars ou `< 2` letras) | 128.986 | 112 | 97,44% |
| 3. Remoção de duplicatas exatas no par `(review_text, overall_rating)` | 126.691 | 2.295 | 95,71% |
| 4. Remoção de registros residuais vazios após filtragem de *stopwords* | **126.690** | 1 | **95,71%** |

Durante todo o processamento, preservou-se intacta a coluna `review_text_original` para permitir comparações diretas antes/depois e inspeção qualitativa de erros.

### 3.2 Regra de Mapeamento de Notas e Ruído nos Rótulos
Para converter a escala de notas (1 a 5) em classes de sentimento, adotou-se a regra padrão:
- **Notas 1 e 2 → Negativo (`0`)**: `32.599` avaliações (**25,73%** — sendo `24.607` nota 1 e `7.992` nota 2).
- **Nota 3 → Neutro (`1`)**: `15.843` avaliações (**12,51%**).
- **Notas 4 e 5 → Positivo (`2`)**: `78.249` avaliações (**61,76%** — sendo `31.782` nota 4 e `46.467` nota 5).

*(Ver `reports/figures/01_distribuicao_notas.png` e `reports/figures/02_distribuicao_sentimentos.png`.)*

**Justificativa e limitação crítica:** Embora amplamente usada como supervisão fraca em e-commerce, a nota atribuída pelo usuário é uma **aproximação imperfeita** do sentimento textual. A auditoria identificou evidências concretas desse ruído ainda na EDA:
1. **`69` textos idênticos** apareceram associados a notas distintas no corpus;
2. **`2.685` avaliações (2,12%)** receberam nota `1` ou `2`, mas o cliente marcou `recommend_to_a_friend == "Yes"`;
3. **`1.152` avaliações (0,91%)** receberam nota `4` ou `5`, mas com `recommend_to_a_friend == "No"`;
4. Casos de **ironia** e **avaliações mistas** (produto bom com entrega atrasada).

Além disso, observa-se **forte desequilíbrio de classes**: a classe Positiva (`61,76%`) é quase **5 vezes maior** (`4,94×`) que a classe Neutra (`12,51%`). Por essa razão, a acurácia isolada é insuficiente, sendo indispensável destacar o **F1-Score Macro** e o desempenho nas classes minoritárias.

### 3.3 Extensão dos Textos, Categorias e Amostragem Estratificada
- **Extensão (`reports/figures/03_tamanho_avaliacoes.png`):** avaliações **Negativas** têm média de **32,33 palavras** (mediana 24), **Neutras** têm **23,95** (mediana 17) e **Positivas** têm **18,70** (mediana 14) — consumidores insatisfeitos escrevem textos quase duas vezes mais longos.
- **Categorias (`reports/figures/04_top_categorias.png`):** *Celulares e Smartphones* lidera em volume (`20.091`; `69,7%` positivas), seguido por *Eletroportáteis* (`11.444`; `71,1%` positivas). Já **Móveis** (`6.634`) apresenta o maior índice de insatisfação (`43,9%` negativas e `16,2%` neutras contra apenas `39,9%` positivas).
- **Amostragem estratificada:** validada em `get_stratified_sample(n_samples=30000)` mantendo proporções idênticas às do corpus integral (`61,76% / 25,73% / 12,51%`). Contudo, todos os experimentos oficiais de modelagem e avaliação deste relatório foram executados sobre o **corpus completo (`126.690` avaliações)**.

---

## 4. Pipeline de Pré-processamento Textual (PLN)

O pipeline implementado em `src/preprocessing.py` e analisado em `notebooks/02_preprocessamento_pln.ipynb` executa:
1. **Limpeza estrutural:** decodificação de entidades HTML, remoção de tags HTML, URLs e e-mails.
2. **Normalização preservando acentuação:** conversão para minúsculas mantendo todos os acentos e cedilha do português (`á, à, â, ã, é, ê, í, ó, ô, õ, ú, ü, ç`). No corpus processado, **265.048 ocorrências de tokens (14,45%)** e **6.625 palavras distintas** preservam acentuação.
3. **Normalização de alongamentos expressivos:** redução de repetições de 3+ letras idênticas (`"demaaaais"` → `"demaais"`), preservando dígrafos válidos (`rr`, `ss`).
4. **Tokenização PT-BR:** extração alfabética `[a-záàâãéêíóôõúüç]+`, eliminando pontuação isolada e números soltos (`min_token_len = 2`).
5. **Remoção customizada de *stopwords* protegendo negações e modificadores:** retiraram-se **7 termos críticos** da lista padrão de 207 *stopwords* do NLTK — **`mais`, `mas`, `muito`, `nem`, `não`, `sem` e `só`** (resultando em 200 *stopwords* filtradas), além da proteção a `nunca`, `jamais`, `nada`, `nenhum`, `ninguém`, `porém` e `pouco`.
6. **Opção experimental de *Negation Marking*:** `mark_negation=True` (`"não recomendo"` → `"não_recomendo"`).

### 4.1 Evidência Empírica da Preservação de Negações e Análise Lexical
- A remoção customizada reduziu o volume de tokens de **2.874.823** para **1.833.713** (**-36,21%**), preservando **46.780** termos únicos (`reports/figures/06_top_palavras_antes_depois_stopwords.png`).
- **`79,28%`** das avaliações **Negativas** contêm ao menos uma negação (**`72,35%`** contêm `"não"/"nao"`), frente a **`40,47%`** das **Neutras** e **`20,86%`** das **Positivas** (`reports/figures/05_impacto_negacoes.png`).
- Os **Top 15 bigramas** (`reports/figures/07_top_bigramas_por_sentimento.png`) e **WordClouds** (`reports/figures/08_wordclouds_por_sentimento.png`) confirmam a separação: `"não recebi"` / `"não recomendo"` / `"não funciona"` no Negativo; `"produto bom"` / `"mas não"` / `"bom mas"` no Neutro; `"muito bom"` / `"antes prazo"` / `"super recomendo"` no Positivo.

---

## 5. Metodologia de Representação Textual e Modelagem

### 5.1 Divisão Treino/Teste e Prevenção de *Data Leakage*
Sobre as **126.690** avaliações pré-processadas aplicou-se `train_test_split(test_size=0.20, stratify=y, random_state=42)`, resultando em **101.352** avaliações de treino e **25.338** de teste, com proporções de classe idênticas nos dois conjuntos (`61,76%` Positivo, `25,73%` Negativo, `12,50–12,51%` Neutro).

Para prevenir vazamento de dados (Seção 2.5.4 do edital):
- O `fit` de `CountVectorizer`/`TfidfVectorizer` ocorreu **exclusivamente em `X_train`**; em `X_test` aplicou-se apenas `transform`;
- O `GridSearchCV` rodou **exclusivamente em `X_train`** dentro de um `Pipeline`, reajustando o vetorizador em cada fold;
- O conjunto de teste foi reservado somente à avaliação final.

### 5.2 Representações Textuais e Classificadores Avaliados
Implementaram-se **4 representações** (`min_df=3`, `token_pattern=r"[a-záàâãéêíóôõúüç_]+"`):

| Código | Representação | Nº de atributos |
| :--- | :--- | ---: |
| `bow_uni` | Bag of Words (`CountVectorizer`), unigramas | 15.978 |
| `bow_bi` | Bag of Words (`CountVectorizer`), uni+bigramas | 78.918 |
| `tfidf_uni` | TF-IDF (`sublinear_tf=True`), unigramas | 15.978 |
| `tfidf_bi` | TF-IDF (`sublinear_tf=True`), uni+bigramas | 78.918 |

Combinadas com **4 variantes de classificadores**: **Multinomial Naive Bayes**, **Regressão Logística** (`class_weight=None` e `class_weight='balanced'`) e **Linear SVM** (`LinearSVC(class_weight='balanced', random_state=42)`).

---

## 6. Resultados Experimentais — 16 Combinações Padrão (Conjunto de Teste)

| Representação | Classificador | Acurácia | F1-Macro | F1 Neg. | F1 Neu. | F1 Pos. |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| TF-IDF uni+bi | LogReg `balanced` | 0,7992 | **0,7233** | 0,8634 | 0,4398 | 0,8666 |
| TF-IDF uni+bi | SVM `balanced` | 0,8221 | 0,7100 | 0,8610 | 0,3779 | 0,8911 |
| BoW uni+bi | Naive Bayes | 0,8143 | 0,7073 | 0,8404 | 0,3914 | 0,8903 |
| BoW uni+bi | LogReg `balanced` | 0,8006 | 0,7069 | 0,8534 | 0,3948 | 0,8725 |
| TF-IDF uni+bi | LogReg padrão | **0,8452** | 0,6993 | 0,8729 | 0,3193 | 0,9058 |
| TF-IDF uni | LogReg `balanced` | 0,7681 | 0,6981 | 0,8424 | 0,4100 | 0,8419 |
| BoW uni+bi | LogReg padrão | 0,8302 | 0,6978 | 0,8602 | 0,3359 | 0,8973 |
| TF-IDF uni | SVM `balanced` | 0,8035 | 0,6937 | 0,8355 | 0,3663 | 0,8793 |
| BoW uni | LogReg `balanced` | 0,7669 | 0,6902 | 0,8260 | 0,3980 | 0,8466 |
| BoW uni | Naive Bayes | 0,7952 | 0,6770 | 0,8161 | 0,3360 | 0,8790 |
| BoW uni+bi | SVM `balanced` | 0,7958 | 0,6767 | 0,8334 | 0,3226 | 0,8742 |
| TF-IDF uni | LogReg padrão | 0,8319 | 0,6755 | 0,8518 | 0,2770 | 0,8979 |
| BoW uni | SVM `balanced` | 0,7910 | 0,6754 | 0,8124 | 0,3396 | 0,8742 |
| BoW uni | LogReg padrão | 0,8183 | 0,6684 | 0,8363 | 0,2774 | 0,8916 |
| TF-IDF uni+bi | Naive Bayes | 0,8271 | 0,5969 | 0,8504 | 0,0470 | 0,8934 |
| TF-IDF uni | Naive Bayes | 0,8103 | 0,5874 | 0,8201 | 0,0598 | 0,8823 |

*(Ver `reports/figures/09_f1_macro_comparacao.png`.)*

**Principais conclusões da grade padrão:**
1. **Acurácia vs. F1-Macro:** a maior acurácia (`0,8452`, LogReg padrão em `tfidf_bi`) recupera apenas `22,6%` dos Neutros; a melhor por F1-Macro na grade padrão (`LogReg balanced`) recupera `54,8%`.
2. **Bigramas:** elevam o F1-Macro médio em `+0,019` (`0,6898` vs `0,6707`).
3. **`class_weight='balanced'`:** eleva o F1 médio do Neutro na Regressão Logística de `0,302` para `0,411`.
4. **Naive Bayes em TF-IDF:** colapsa na classe Neutra com `alpha=1,0` (`F1 Neutro ≈ 0,05`), recuperando significativamente ao reduzir `alpha` no ajuste de hiperparâmetros.
5. **BoW vs. TF-IDF:** nos 6 pares de modelos lineares, TF-IDF supera BoW em todas as comparações (de `+0,0015` a `+0,0333`).

---

## 7. Ajuste de Hiperparâmetros, Termos Mais Influentes e *Negation Marking*

### 7.1 Ajuste por Validação Cruzada (`GridSearchCV` em `X_train`, `tfidf_bi`)
Executado com `StratifiedKFold(3, shuffle=True, random_state=42)` e `scoring='f1_macro'`:

| Classificador | Grade | Melhor valor | F1-Macro (CV) | F1-Macro (teste) | Acurácia (teste) |
| :--- | :--- | :--- | ---: | ---: | ---: |
| Naive Bayes | `alpha` ∈ {0,1; 0,3; 0,5; 1,0} | `0,1` | 0,6800 | 0,6851 | 0,8289 |
| LogReg padrão | `C` ∈ {0,5; 1; 3; 10} | `3` | 0,7010 | 0,7006 | 0,8395 |
| LogReg `balanced` | `C` ∈ {0,5; 1; 3; 10} | `1` | 0,7171 | 0,7233 | 0,7992 |
| **SVM `balanced` ★** | `C` ∈ {0,03; 0,1; 0,3; 1} | **`0,1`** | **0,7248** | **0,7318** | **0,8378** |

**Modelo final escolhido pelo CV:** **TF-IDF Uni+Bigramas + Linear SVM (`C=0,1`, `balanced`)** — Acurácia `0,8378`, F1-Macro `0,7318` (Negativo: P=`0,8249`, R=`0,9138`, F1=`0,8671`; Neutro: P=`0,4697`, R=`0,3913`, F1=`0,4269`; Positivo: P=`0,9066`, R=`0,8965`, F1=`0,9015`).

### 7.2 Termos Mais Influentes e *Negation Marking*
- **Termos mais fortes (`reports/figures/11_termos_mais_influentes.png`):**
  - *Negativo:* `não recomendo`, `não`, `não gostei`, `péssimo`, `não funciona`, `ruim`, `nao`, `dinheiro`, `horrível`.
  - *Neutro:* `bom`, `gostei`, `porém`, `produto bom`, `boa`, `mas`, `bom produto`.
  - *Positivo:* `excelente`, `ótimo`, `ótima`, `muito bom`, `perfeito`, `amei`, `adorei`, `recomendo`, `maravilhoso`.
- ***Negation Marking* em unigramas (`notebooks/03_treinamento_modelos.ipynb`):** melhora todas as 8 combinações de unigramas (`+0,0061` a `+0,0124` em F1-Macro), mas na maioria dos casos não alcança uni+bigramas, pois bigramas capturam também expressões não negativas (`muito bom`, `antes prazo`, `bom mas`).

---

## 8. Avaliação Crítica — Respostas às 5 Perguntas da Seção 2.6

Executada sobre o conjunto de teste (`n = 25.338`) via `src/evaluation.py` e `notebooks/04_avaliacao_e_erros.ipynb`:

### 8.1 Qual configuração apresentou melhor resultado?
**TF-IDF Uni+Bigramas + Linear SVM (`C=0,1`, `class_weight='balanced'`)**, atingindo **F1-Macro = `0,7318`** no teste (`0,7248` em CV) e **Acurácia = `83,78%`**. Embora `LogReg padrão` tenha acurácia marginalmente superior (`84,52%`), falha gravemente na classe minoritária (Recall Neutro = `22,56%`, F1-Macro = `0,6993`). O SVM ajustado apresenta o melhor equilíbrio global entre precisão, sensibilidade nas classes minoritárias e acurácia (`reports/figures/12_matrizes_confusao_comparacao.png`).

### 8.2 O modelo consegue identificar adequadamente avaliações negativas?
**Sim.** O modelo final alcança **Recall Negativo = `91,38%`** (`5.958` acertos em `6.520` negativas reais — o maior entre todas as 20 configurações), com **Precisão = `82,49%`** e **F1 Negativo = `0,8671`** (`reports/figures/15_recall_negativo_comparacao.png`). Apenas **`3,13%`** (`204` casos) das avaliações negativas foram confundidas com `Positivo`, e quase um terço desses casos apresenta contradição no próprio rótulo do corpus.

### 8.3 Quais erros aparecem com maior frequência?
Entre os **`4.111` erros (`16,22%` do teste)**, **`3.329` (`80,98%`) envolvem a classe Neutra** (`reports/figures/14_analise_erros_e_ruido.png`, Painel A):
- `Neutro → Positivo`: `1.242` (`30,21%` dos erros)
- `Positivo → Neutro`: `1.042` (`25,35%`)
- `Neutro → Negativo`: `687` (`16,71%`)
- `Positivo → Negativo`: `578` (`14,06%`)
- `Negativo → Neutro`: `358` (`8,71%`)
- `Negativo → Positivo`: `204` (`4,96%`)

Estimou-se que **`1.086` erros (`26,42%`)** correspondem a **avaliações mistas** (ex.: *"Produto é bom, porém a entrega não foi entregue no prazo prometido."* — nota 4 prevista como Neutro com `75,0%`; *"só atrasou alguns dias a entrega. Porém o produto é bom."* — nota 4 prevista como Neutro). Ademais, erros têm extensão média maior (`25,12` palavras) do que acertos (`22,50` palavras).

### 8.4 A regra de conversão de notas em sentimentos introduz ruído nos rótulos?
**Sim, comprovado estatisticamente (`reports/figures/14_analise_erros_e_ruido.png`, Painel B):**
- Nos `204` erros extremos `Negativo → Positivo`, **`28,92%`** dos clientes marcaram `recommend_to_a_friend == "Yes"` (vs. `8,33%` no teste geral — **3,5× mais alto**). Exemplos reais de nota 1 previstos como Positivo: *"muito boa a entrega bem antes do previsto exelente loja superr recomendo"* (`88,3%`) e *"Ótimo produto ...recomendo ..chegou antes do prazo"* (`86,4%`).
- Nos `578` erros extremos `Positivo → Negativo`, **`8,48%`** marcaram `recommend_to_a_friend == "No"` (vs. `1,48%` no teste geral — **5,7× mais alto**). Exemplos reais de nota 5 previstos como Negativo: *"como posso avaliar o produto se consta entregue e não recebi o produto... vou ter entrar no PROCON"* (`88,4%`) e *"Não recebi ainda Não recebi ainda Não recebi ainda"* (`85,4%`).
- **Experimento Binário (sem nota 3, sem *data leakage*):** descartando a nota 3 em treino (`88.678`) e teste (`22.169`), o mesmo classificador salta de **Acurácia `83,78%` → `94,77%`** e de **F1-Macro `0,7318` → `0,9388` (+0,2070)**, confirmando que a nota 3 é a principal fonte de ambiguidade.

### 8.5 O desempenho se mantém em avaliações de produtos ou categorias diferentes?
A Acurácia se mantém relativamente estável nas Top 10 categorias (`81,89%` a `85,55%`), porém o **F1-Macro oscila `0,0720` pontos** (`0,6920` a `0,7640`, `reports/figures/13_desempenho_por_categoria.png`):

| Categoria (`site_category_lv1`) | n (Teste) | % Neg. | % Neu. | % Pos. | Acurácia | F1-Macro | F1 Neg. | F1 Neu. | F1 Pos. |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Utilidades Domésticas | 1.546 | 25,1% | 13,3% | 61,6% | 0,8461 | **0,7640** | 0,8774 | **0,5099** | 0,9046 |
| Brinquedos | 941 | 27,3% | 12,1% | 60,6% | **0,8555** | 0,7633 | 0,8787 | 0,4977 | 0,9135 |
| **Móveis** | **1.303** | **42,7%** | **15,8%** | **41,5%** | 0,8289 | **0,7593** | **0,8983** | 0,5051 | 0,8744 |
| Casa e Construção | 852 | 33,2% | 14,2% | 52,6% | 0,8439 | 0,7523 | 0,8938 | 0,4660 | 0,8972 |
| Informática | 822 | 28,6% | 15,8% | 55,6% | 0,8224 | 0,7453 | 0,8687 | 0,4848 | 0,8824 |
| Beleza e Perfumaria | 1.784 | 25,4% | 12,0% | 62,6% | 0,8380 | 0,7322 | 0,8574 | 0,4388 | 0,9004 |
| Eletroportáteis | 2.317 | 16,9% | 11,4% | 71,6% | 0,8502 | 0,7232 | 0,8407 | 0,4134 | **0,9157** |
| Informática e Acessórios | 1.447 | 31,8% | 13,0% | 55,2% | 0,8189 | 0,7147 | 0,8859 | 0,3871 | 0,8711 |
| TV e Home Theater | 1.444 | 22,3% | 10,7% | 67,0% | 0,8255 | **0,6920** | 0,8309 | 0,3455 | 0,8995 |
| Celulares e Smartphones | 4.055 | 19,2% | 11,5% | 69,3% | 0,8358 | **0,6920** | 0,8314 | **0,3350** | 0,9097 |

Apesar de ser a categoria mais negativa do corpus, **Móveis** atingiu um dos melhores F1-Macro (`0,7593`) e Recall Negativo (`92,99%`) graças ao equilíbrio maior entre classes e vocabulário explícito de defeitos/montagem. Em contrapartida, **Celulares e Smartphones** e **TV e Home Theater** caem para `0,6920` devido ao F1 Neutro reduzido (`0,335–0,346`), refletindo maior desequilíbrio positivo e ressalvas técnicas sutis entre notas 3 e 4.

---

## 9. Limitações, Vieses, Ética e Privacidade (Seção 2.8)

1. **Limitações metodológicas e linguísticas:**
   - **Ironia, sarcasmo e ortografia informal:** representações *Bag of N-grams* lineares não capturam nuances pragmáticas de sarcasmo nem negações de longa distância, além de sofrerem fragmentação com abreviações e erros ortográficos.
   - **Avaliações multi-aspecto:** um único rótulo não distingue qualidade do produto de qualidade logística (extensão futura: ABSA ou *Transformers* em português como BERTimbau).
   - **Divisão única e seleção da grade:** os resultados decorrem de um único split 80/20 (diferenças `< 0,01` não devem ser lidas como superioridade estatística firme) e o `LinearSVC` em BoW teve tempo elevado indicando possível limite de `max_iter`.
   - **Calibração:** `LinearSVC` não possui `predict_proba`; na interface Streamlit, a pontuação exibida é o *softmax* do `decision_function` (pontuação relativa, não probabilidade calibrada), com aviso explícito ao usuário.
2. **Vieses do corpus:** supervisão fraca por estrelas, desequilíbrio de classes, autosseleção dos avaliadores e viés temporal/de domínio (Americanas.com, jan–mai/2018).
3. **Privacidade e licença:** todas as 4 colunas pessoais (`reviewer_id`, `reviewer_birth_year`, `reviewer_gender`, `reviewer_state`) são eliminadas na ingestão e validadas por testes unitários; a licença `CC BY-NC-SA 4.0` é respeitada integralmente sem redistribuição do CSV no GitHub.

---

## 10. Aplicação Demonstrável (`app.py`) e Verificação Integrada

A interface local em **Streamlit** (`streamlit run app.py`) integra todos os artefatos em **3 abas** (carregamento instantâneo via cache, sem leitura do CSV bruto):
- **Aba 1:** auditoria do corpus, seletor interativo de regra de mapeamento (3 classes vs. Binário vs. customizada por estrela), gráficos EDA (`01..04`) e comparador ao vivo de pré-processamento preservando acentos e negações (`05..08`).
- **Aba 2:** tabela filtrável das 20 configurações, explorador dinâmico de matriz de confusão por modelo, gráficos (`09`, `11`, `12`, `13`, `14`, `15`), desempenho por categoria e explorador de erros reais/ruído do `X_test`.
- **Aba 3:** simulador em tempo real para frases inéditas com 6 exemplos de 1 clique, seleção livre de classificador/representação, auditoria de negações preservadas, **explicabilidade visual dos tokens/bigramas que mais pesaram** e comparação simultânea entre os 4 classificadores ajustados.

A integridade de ponta a ponta foi verificada pela suíte **`tests/test_pipeline.py`** (`10` testes automatizados cobrindo privacidade, limpeza, PLN, carregamento de artefatos, explicabilidade nas 3 famílias de classificadores e validação de `app.py`).

---

## 11. Declaração Transparente sobre Uso de IA Generativa (Seção 8)

Em conformidade com a Seção 8 do edital, declaramos de forma transparente que ferramentas de Inteligência Artificial generativa foram utilizadas pela equipe como apoio à produtividade e aprendizagem nas seguintes frentes:
- **Pessoa 1:** apoio na estruturação inicial do projeto, boas práticas de modularização (`src/data_loader.py`, `src/preprocessing.py`), geração de gráficos exploratórios e revisão textual.
- **Pessoa 2:** apoio na revisão de `src/models.py`, conferência cruzada dos números do relatório contra `reports/model_results.json` e o notebook `03_treinamento_modelos.ipynb`, e revisão textual.
- **Pessoa 3:** apoio no desenvolvimento e revisão de `src/evaluation.py`, `notebooks/04_avaliacao_e_erros.ipynb`, `app.py` e `tests/test_pipeline.py`, bem como revisão redacional e consolidação do relatório.

Todos os scripts, notebooks, modelos treinados, métricas, gráficos e exemplos reais foram **integralmente executados, testados e auditados localmente** sobre o corpus real **B2W-Reviews01**, e todas as decisões técnicas são compreendidas e explicáveis pelos três integrantes.

---

## 12. Referências

1. AMERICANAS TECH / B2W DIGITAL. **B2W-Reviews01 — Open Corpus of Product Reviews in Portuguese**. Disponível em: <https://github.com/americanas-tech/b2w-reviews01>. Licença CC BY-NC-SA 4.0.
2. REAL, L.; OGAWA, M.; COELHO, F.; COELHO, L. **B2W-Reviews01: An Open Dataset for Sentiment Analysis in Portuguese**. STIL — Symposium in Information and Human Language Technology, 2019.
3. CHAVES, R. **B2W-Reviews01 (Hugging Face Dataset Mirror)**. Disponível em: <https://huggingface.co/datasets/ruanchaves/b2w-reviews01>.
4. PEDREGOSA, F. et al. **Scikit-learn: Machine Learning in Python**. Journal of Machine Learning Research, v. 12, p. 2825–2830, 2011.
5. BIRD, S.; KLEIN, E.; LOPER, E. **Natural Language Processing with Python (NLTK)**. O'Reilly Media, 2009.
6. JURAFSKY, D.; MARTIN, J. H. **Speech and Language Processing** (3rd ed. draft). Stanford University, 2024.
7. STREAMLIT INC. **Streamlit Documentation**. Disponível em: <https://docs.streamlit.io/>.
