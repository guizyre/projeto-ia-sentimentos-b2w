# Relatório Técnico — Parte da Pessoa 1 (Seções 1 a 3)
**Projeto 2: Classificação Automática de Sentimentos em Avaliações de Produtos**  
**Disciplina:** Inteligência Artificial – Unidade 3 (IFPB – Bacharelado em Engenharia da Computação)  
**Responsável por esta seção:** Pessoa 1 (Caracterização do Problema/Corpus, Análise Exploratória e Pré-processamento PLN)

---

## 1. Introdução e Caracterização do Problema

Consumidores de plataformas de comércio eletrônico publicam diariamente milhares de avaliações textuais expressando percepções sobre qualidade, desempenho, custo-benefício e experiência de entrega dos produtos adquiridos. A análise manual desse volume de texto é inviável em larga escala, tornando a **Análise de Sentimentos** um problema clássico e relevante de Processamento de Linguagem Natural (PLN).

O objetivo deste projeto é desenvolver, avaliar e demonstrar um pipeline completo de PLN em Python capaz de receber uma avaliação textual em português e classificá-la automaticamente em três classes de sentimento: **Negativo**, **Neutro** ou **Positivo**. Em conformidade com as diretrizes da disciplina, a investigação vai além da execução de bibliotecas prontas, documentando e justificando empiricamente cada etapa de auditoria, limpeza, preservação linguística (acentuação e negações), representação vetorial e classificação.

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
- **Notas 1 e 2 $\rightarrow$ Negativo (`0`)**: `32.599` avaliações (**25,73%** — sendo `24.607` nota 1 e `7.992` nota 2).
- **Nota 3 $\rightarrow$ Neutro (`1`)**: `15.843` avaliações (**12,51%**).
- **Notas 4 e 5 $\rightarrow$ Positivo (`2`)**: `78.249` avaliações (**61,76%** — sendo `31.782` nota 4 e `46.467` nota 5).

*(Ver figuras `reports/figures/01_distribuicao_notas.png` e `reports/figures/02_distribuicao_sentimentos.png`.)*

**Justificativa e limitação crítica:** Embora amplamente usada como supervisão fraca em e-commerce, a nota atribuída pelo usuário é uma **aproximação imperfeita** do sentimento textual. Nossa auditoria identificou evidências concretas desse ruído ainda na EDA:
1. **`69` textos idênticos** apareceram associados a notas distintas no corpus;
2. **`2.685` avaliações (2,12%)** receberam nota `1` ou `2`, mas o cliente marcou `recommend_to_a_friend == "Yes"`;
3. **`1.152` avaliações (0,91%)** receberam nota `4` ou `5`, mas com `recommend_to_a_friend == "No"`;
4. Casos de **ironia** (ex.: avaliações nota 5 elogiando com sarcasmo um copo acrílico de preço absurdo) e **avaliações mistas** (produto bom com entrega atrasada).

Além disso, observa-se **forte desequilíbrio de classes**: a classe Positiva (`61,76%`) é quase **5 vezes maior** (`4,94x`) que a classe Neutra (`12,51%`). Por essa razão, a acurácia isolada é insuficiente, sendo indispensável destacar o **F1-Score Macro** e o desempenho nas classes minoritárias (Neutro e Negativo) na fase experimental.

### 3.3 Extensão dos Textos e Distribuição por Categorias
A análise do comprimento das avaliações (`reports/figures/03_tamanho_avaliacoes.png`) revelou um padrão comportamental claro:
- **Negativo:** média de **32,33 palavras** (mediana **24**, P90 = 63) — consumidores insatisfeitos escrevem textos quase duas vezes mais longos para relatar defeitos, atrasos ou frustrações.
- **Neutro:** média de **23,95 palavras** (mediana **17**, P90 = 46) — posição intermediária, frequentemente ponderando pontos positivos e negativos.
- **Positivo:** média de **18,70 palavras** (mediana **14**, P90 = 34) — elogios diretos e curtos (ex.: *"Ótimo produto, chegou antes do prazo"*).

Quanto às categorias de produtos (`reports/figures/04_top_categorias.png`), *Celulares e Smartphones* lidera em volume (`20.091` avaliações; `69,7%` positivas), seguido por *Eletroportáteis* (`11.444`; `71,1%` positivas). Já a categoria **Móveis** (`6.634` avaliações) apresenta o maior índice de insatisfação (`43,9%` negativas e `16,2%` neutras contra apenas `39,9%` positivas), evidenciando que a distribuição de sentimentos varia fortemente conforme o domínio do produto.

### 3.4 Amostragem Estratificada
Para viabilizar experimentos rápidos sem distorção estatística, implementou-se `get_stratified_sample()` estratificado por `overall_rating` (`random_state=42`). Na validação com `30.000` amostras, as proporções de sentimento mantiveram-se **idênticas** às do corpus integral (`61,76%` Positivo, `25,73%` Negativo, `12,51%` Neutro — diferença de `0,00` p.p.).

---

## 4. Pipeline de Pré-processamento Textual (PLN)

O pipeline implementado em `src/preprocessing.py` e analisado em `notebooks/02_preprocessamento_pln.ipynb` executa as seguintes etapas sequenciais:

1. **Limpeza estrutural:** decodificação de entidades HTML, remoção de tags HTML, URLs (`http...`) e endereços de e-mail.
2. **Normalização preservando acentuação:** conversão para minúsculas (`.lower()`) mantendo **integralmente** todos os acentos e cedilha do português (`á, à, â, ã, é, ê, í, ó, ô, õ, ú, ü, ç`). Retirar acentos de forma cega mescla termos distintos (`é` vs `e`, `está` vs `esta`) e degrada vocábulos carregados de polaridade (`ótimo`, `péssimo`, `não`, `horrível`). No corpus processado, **265.048 ocorrências de tokens (14,45% do total)** e **6.625 palavras distintas** preservam acentuação.
3. **Normalização de alongamentos expressivos:** redução de repetições exageradas de 3 ou mais letras idênticas (ex.: `"demaaaais"` $\rightarrow$ `"demaais"`), preservando dígrafos válidos do português (`rr`, `ss`).
4. **Tokenização PT-BR:** extração de tokens alfabéticos portugueses (`[a-záàâãéêíóôõúüç]+`, compatível com `nltk.tokenize.RegexpTokenizer`), eliminando pontuação isolada e números soltos e mantendo tokens com pelo menos 2 caracteres.
5. **Remoção customizada de *stopwords* protegendo negações e modificadores:** a lista padrão do NLTK em português possui **207 *stopwords***, porém contém **7 termos críticos para polaridade** que foram explicitamente retirados da lista de exclusão: **`mais`, `mas`, `muito`, `nem`, `não`, `sem` e `só`** (resultando em 200 *stopwords* filtradas), além da proteção garantida a `nunca`, `jamais`, `nada`, `nenhum`, `ninguém`, `porém` e `pouco`.
6. **Opção experimental de *Negation Marking*:** o módulo oferece o parâmetro `mark_negation=True` (ex.: `"não recomendo"` $\rightarrow$ `"não_recomendo"`, `"nunca mais"` $\rightarrow$ `"nunca_mais"`) para que a etapa de modelagem compare abordagens de unigramas vs. bigramas.

### 4.1 Evidência Empírica da Preservação de Negações e Análise Lexical
- A remoção de *stopwords* customizada reduziu o total de tokens de **2.874.823** para **1.833.713** (**-36,21%** de ruído funcional), preservando **46.780** termos únicos (`reports/figures/06_top_palavras_antes_depois_stopwords.png`).
- A análise quantitativa (`reports/figures/05_impacto_negacoes.png`) comprovou que **79,28%** das avaliações **Negativas** contêm ao menos uma palavra de negação (**72,35%** contêm `"não"/"nao"`), frente a **40,47%** das **Neutras** e apenas **20,86%** das **Positivas**. Eliminar `"não"` destruiria o sinal mais forte de distinção da classe negativa (transformando, por exemplo, *"Não recomendo"* em *"recomendo"*).
- A inspeção dos **Top 15 bigramas por classe** (`reports/figures/07_top_bigramas_por_sentimento.png`) e das **Nuvens de Palavras** (`reports/figures/08_wordclouds_por_sentimento.png`) evidencia a separação semântica:
  - **Negativo:** `"não recebi"` (4.155), `"produto não"` (2.590), `"não recomendo"` (2.405), `"ainda não"` (2.237), `"não chegou"` (1.173), `"não funciona"` (984), `"não gostei"` (842), `"veio defeito"` (809).
  - **Neutro:** coocorrência de elogios e ressalvas — `"produto bom"` (1.318), `"muito bom"` (1.165), `"ainda não"` (469), `"mas não"` (382), `"bom mas"` (379) — antecipando por que a classe Neutra tende a apresentar maior confusão nos classificadores.
  - **Positivo:** `"muito bom"` (11.577), `"antes prazo"` (6.551), `"gostei muito"` (4.205), `"super recomendo"` (3.243), `"ótima qualidade"` (2.944), `"entrega rápida"` (2.629).

---

## 5. Nota sobre Uso Transparente de IA Generativa (Seção 8 do Edital)
*(Incluir na seção final do relatório do trio)*  
Ferramentas de Inteligência Artificial generativa foram utilizadas como apoio na estruturação inicial do projeto, revisão de boas práticas de modularização em Python, geração de visualizações exploratórias e revisão textual. Todo o código-fonte (`src/data_loader.py`, `src/preprocessing.py`), os notebooks executados e as estatísticas apresentadas foram integralmente inspecionados, testados localmente sobre o corpus real **B2W-Reviews01** e compreendidos pela equipe.

---

## 6. Roteiro Sugerido para Apresentação Oral — Bloco da Pessoa 1 (3 a 4 minutos)
1. **Problema e Corpus (~45 s):** Apresentar o B2W-Reviews01 (132.373 avaliações, Americanas.com, jan–mai/2018), a licença `CC BY-NC-SA 4.0` e o descarte imediato das 4 colunas com dados pessoais (`reviewer_id`, idade, gênero, estado).
2. **Limpeza, Rotulagem e Desequilíbrio (~1 min):** Mostrar a retenção de `95,71%` (`126.690` avaliações finais), a regra `1–2 / 3 / 4–5` e o gráfico de sentimentos (`61,8%` Positivo vs `12,5%` Neutro), explicando por que esse desequilíbrio exige foco no **F1-Score Macro** e citando os casos reais de ruído (69 textos conflitantes e 2.685 notas 1–2 com recomendação "Yes").
3. **Insights da EDA (~45 s):** Mostrar que reclamações são quase o dobro mais longas (mediana 24 vs 14 palavras) e o contraste entre categorias (*Eletroportáteis* 71% positivo vs *Móveis* 44% negativo).
4. **Pré-processamento PLN (~1 min):** Mostrar o exemplo real *"Não recomendo"* virando *"recomendo"* com stopwords padrão do NLTK vs. nosso pipeline mantendo acentos e negações, fechar com o gráfico provando que **79,3% das negativas têm negação** e os **Top Bigramas** (`não recebi`, `não recomendo`, `bom mas`, `muito bom`), passando a bola para a **Pessoa 2** falar sobre BoW/TF-IDF e bigramas nos classificadores!
