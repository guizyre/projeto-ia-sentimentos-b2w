# Roteiro Completo de Apresentação Oral e Demonstração ao Vivo
## Projeto 2: Classificação Automática de Sentimentos em Avaliações de Produtos (PLN)

- **Disciplina:** Inteligência Artificial – Unidade 3 (IFPB – Engenharia da Computação)
- **Professor:** Marcelo José Siqueira Coutinho de Almeida
- **Duração Total Planejada:** **10 min 30 s a 12 min** (divisão equilibrada de ~3 min 30 s a 4 min por integrante, conforme exigido na Seção 6 do edital)
- **Equipe:**
  1. **Bloco 1 (00:00 – 03:45) — Guilherme Gonçalves:** Problema, Corpus/Licença/Privacidade, Auditoria/EDA e Pré-processamento PLN (Acentos & Negações)
  2. **Bloco 2 (03:45 – 07:30) — Anaíza Silva:** Divisão Anti-Vazamento, BoW vs. TF-IDF, Unigramas vs. Bigramas, Classificadores, Tuning e Pesos dos Termos
  3. **Bloco 3 (07:30 – 11:30) — Hillary Maximino:** Respostas às 5 Perguntas da Seção 2.6, Ruído de Rótulo, Categorias, Ética/Limitações e **Demonstração Ao Vivo no Streamlit**

---

## Preparação Antes de Começar (2 minutos antes da banca)

1. Deixar o terminal aberto na pasta `projeto-ia-sentimentos-b2w` com o ambiente ativo e o Streamlit já rodando:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   streamlit run app.py
   ```
2. Deixar o navegador aberto em tela cheia (`F11`) com duas abas:
   - **Aba A:** Apresentação de slides (ou o próprio `relatorio_tecnico_completo.md` / gráficos).
   - **Aba B:** Aplicação **Streamlit** aberta na **Aba 01**, pronta para a demonstração ao vivo no Bloco 3.
3. Conferir que o zoom do navegador no Streamlit está em `100%` (layout limpo em tema claro).

---

## BLOCO 1 — Guilherme Gonçalves (3 min 30 s a 3 min 45 s)
**Foco de Avaliação:** *Caracterização do problema e do corpus (15 pts) + EDA e preparação dos dados (15 pts) = 30 pts*

### 1.1 Abertura, Problema, Licença e Privacidade (00:00 – 00:50)
> **Fala sugerida:**
> "Boa noite, professor Marcelo e colegas. Nós somos Guilherme Gonçalves, Anaíza Silva e Hillary Maximino, e desenvolvemos o **Projeto 2: Classificação Automática de Sentimentos em Avaliações de Produtos**.
> O problema que atacamos é transformar milhares de opiniões livres em português do e-commerce em três classes acionáveis — **Negativo**, **Neutro** e **Positivo** — investigando empiricamente cada etapa do pipeline de PLN.
> Utilizamos o corpus aberto **B2W-Reviews01** (versão 1.0, acessado em 23 de novembro de 2026 no repositório oficial da Americanas Tech), com **132.373 avaliações reais** coletadas entre janeiro e maio de 2018. Respeitamos rigorosamente a licença **CC BY-NC-SA 4.0**, não redistribuindo o CSV bruto no GitHub, e cumprimos a **Seção 2.8 de privacidade** descartando logo na leitura as 4 colunas pessoais (`reviewer_id`, ano de nascimento, gênero e estado)."

### 1.2 Auditoria, Regra de Mapeamento e Desequilíbrio (00:50 – 01:55)
- **Visual de apoio:** Figuras `01` e `02`.
> **Fala sugerida:**
> "Na auditoria do conjunto bruto, removemos 3.275 registros nulos, 112 textos sem conteúdo alfabético, 2.295 duplicatas exatas e 1 resíduo pós-filtragem, retendo **126.690 avaliações válidas (95,71% do corpus)** — e decidimos rodar todos os experimentos sobre essa base completa em vez de amostrar.
> Mapeamos notas **1 e 2 como Negativo (25,73%)**, **nota 3 como Neutro (12,51%)** e **notas 4 e 5 como Positivo (61,76%)**.
> Aqui surgem dois pontos fundamentais para o restante do projeto: primeiro, existe um **desequilíbrio de quase 5 para 1** entre Positivo e Neutro, o que torna a acurácia isolada enganosa e exige usar o **F1-Score Macro** como métrica principal. Segundo, a nota é uma *supervisão fraca*: já na EDA achamos **69 textos idênticos com notas diferentes** e **2.685 avaliações nota 1 ou 2 em que o cliente marcou que recomenda o produto**."

### 1.3 Insights da EDA: Tamanho e Categorias (01:55 – 02:35)
- **Visual de apoio:** Figuras `03` e `04`.
> **Fala sugerida:**
> "A análise exploratória revelou que quem reclama escreve quase o dobro: a média das avaliações negativas é de **32,3 palavras** contra **18,7 nas positivas**. Também vimos que o sentimento muda drasticamente por categoria: enquanto *Eletroportáteis* e *Celulares* passam de **70% positivos**, a categoria **Móveis** inverte o padrão com **43,9% de avaliações negativas**."

### 1.4 Pré-processamento PLN: Acentos e Proteção de Negações (02:35 – 03:45)
- **Visual de apoio:** Figuras `05`, `06` e `07`.
> **Fala sugerida:**
> "No pré-processamento (`src/preprocessing.py`), além de limpar HTML e URLs, tomamos duas decisões linguísticas cruciais para o português:
> 1. **Preservar toda a acentuação** (`14,45%` dos tokens finais têm acento), evitando misturar palavras como *'é'* e *'e'* ou *'está'* e *'esta'*.
> 2. **Proteger operadores de negação e contraste**: a lista padrão de 207 *stopwords* do NLTK apaga **'não', 'nem', 'sem', 'muito', 'mais', 'mas' e 'só'**. Se aplicássemos o NLTK puro, a frase real *'Não recomendo'* viraria *'recomendo'*, invertendo a polaridade!
> Provamos na Figura 05 que **79,28% das avaliações negativas contêm negação** (contra 20,86% nas positivas). Com a filtragem customizada, reduzimos 36,2% de ruído funcional e já enxergamos nos bigramas da Figura 07 a assinatura de cada classe: *'não recebi'* e *'não recomendo'* no Negativo; *'bom mas'* e *'mas não'* no Neutro; e *'muito bom'* e *'antes prazo'* no Positivo. Passo agora a palavra para a **Anaíza**, que vai mostrar como transformamos esse texto em modelos."

---

## BLOCO 2 — Anaíza Silva (3 min 30 s a 3 min 45 s)
**Foco de Avaliação:** *Implementação das técnicas de PLN (25 pts) + parte da Avaliação Experimental (20 pts)*

### 2.1 Divisão Estratificada e Prevenção de Data Leakage (03:45 – 04:30)
> **Fala sugerida:**
> "Obrigada, Guilherme. Na etapa de modelagem (`src/models.py`), dividimos as 126.690 avaliações de forma estratificada em **80% treino (101.352)** e **20% teste (25.338)** com semente fixa (`random_state=42`).
> Seguimos rigorosamente a regra anti-vazamento da Seção 2.5.4: o `.fit()` dos vetorizadores foi executado **exclusivamente em `X_train`**, aplicando apenas `.transform()` no teste; e todo o ajuste de hiperparâmetros rodou dentro de um `Pipeline` com validação cruzada estratificada de 3 folds apenas no treino, escolhendo o modelo final pelo **F1-Macro de validação cruzada**, sem nunca olhar para o teste."

### 2.2 Representações e Grade de 16 Experimentos (04:30 – 05:35)
- **Visual de apoio:** Figura `09` (comparação das 16 combinações).
> **Fala sugerida:**
> "Comparamos **4 representações vetoriais** (`min_df=3`) — *Bag of Words* e *TF-IDF sublinear*, cada um com **unigramas (15.978 atributos)** e **uni+bigramas (78.918 atributos)** — cruzadas com **4 variantes de classificadores**: *Multinomial Naive Bayes*, *Regressão Logística padrão*, *Regressão Logística `balanced`* e *Linear SVM `balanced`*, totalizando 16 combinações na grade inicial.
> Quatro resultados se destacaram:
> 1. Nos modelos lineares, **TF-IDF superou BoW nas 6 comparações pareadas**, pois reduz o peso de termos genéricos muito frequentes.
> 2. **Bigramas elevaram o F1-Macro médio em +0,019**, capturando *'não recomendo'* e *'mas não'*.
> 3. O peso **`class_weight='balanced'` subiu o F1 médio da classe Neutra de 0,30 para 0,41**.
> 4. Já o **Naive Bayes** foi muito bem em BoW (`0,7073`), mas colapsou no Neutro em TF-IDF padrão (`F1 Neutro ≈ 0,05`) porque a suavização `alpha=1,0` sufoca os valores contínuos pequenos do TF-IDF."

### 2.3 Tuning, Modelo Final e Palavras Mais Influentes (05:35 – 07:30)
- **Visual de apoio:** Figura `11` (termos mais influentes).
> **Fala sugerida:**
> "No `GridSearchCV` em `TF-IDF uni+bigramas`, baixar `alpha` para `0,1` recuperou o Naive Bayes de `0,5969` para `0,6851`. E no **Linear SVM `balanced`**, aumentar a regularização para **`C = 0,1`** controlou o sobreajuste nas quase 79 mil features, atingindo o melhor **F1-Macro de validação cruzada (`0,7248`)** e **F1-Macro de `0,7318` no teste**, com **acurácia de `83,78%`**.
> Também testamos o *Negation Marking* em unigramas (`não_gostei`): ele melhorou todas as 8 combinações de unigramas, mas ficou abaixo de `TF-IDF uni+bigramas` porque bigramas também capturam pares positivos como *'muito bom'* e *'antes prazo'*.
> Por fim, na Figura 11 extraímos os coeficientes do modelo linear e confirmamos que o aprendizado faz total sentido linguístico: *'não recomendo'*, *'não'* e *'péssimo'* dominam o Negativo; *'porém'*, *'mas'* e *'bom'* caracterizam o Neutro; e *'excelente'*, *'ótimo'* e *'perfeito'* puxam o Positivo. Passo agora para a **Hillary**, que fez a auditoria crítica dos erros e vai demonstrar nossa aplicação ao vivo."

---

## BLOCO 3 — Hillary Maximino (3 min 45 s a 4 min)
**Foco de Avaliação:** *Avaliação experimental (20 pts) + Análise crítica e ética (10 pts) + Aplicação demonstrável e apresentação (15 pts)*

### 3.1 Respostas às Perguntas 1 e 2: F1-Macro vs. Acurácia e Recall Negativo (07:30 – 08:15)
- **Visual de apoio:** Figuras `12` e `15`.
> **Fala sugerida:**
> "Obrigada, Anaíza. Na avaliação crítica sobre as **25.338 avaliações do conjunto de teste**, respondemos às 5 perguntas da Seção 2.6.
> Na **Pergunta 1**, mostramos na Figura 12 por que não escolhemos pela acurácia: a Regressão Logística padrão teve a maior acurácia (`84,52%`), mas acertou apenas **22,6% dos Neutros**. Nosso modelo final (**TF-IDF uni+bi + Linear SVM `C=0,1 balanced`**) entregou **F1-Macro de `0,7318`**, equilibrando precisão e recall nas três classes.
> Na **Pergunta 2** (Figura 15), provamos que o sistema identifica muito bem reclamações: acertamos **91,38% das 6.520 avaliações negativas reais** (o maior Recall Negativo entre os 20 modelos), e apenas **3,13%** viraram falso positivo extremo."

### 3.2 Perguntas 3, 4 e 5: Erros no Neutro, Prova de Ruído e Categorias (08:15 – 09:25)
- **Visual de apoio:** Figuras `14` e `13`.
> **Fala sugerida:**
> "Na **Pergunta 3** (Figura 14A), entre os 4.111 erros do teste, **80,98% envolvem a classe Neutra** (fronteira adjacente entre nota 3 e notas 4 ou 2) e **26,4% são avaliações mistas**, onde o cliente elogia o produto mas reclama do atraso na entrega.
> Na **Pergunta 4** (Figura 14B), provamos estatisticamente o **ruído de rótulo**: nos erros extremos em que a nota era 1 ou 2 mas nosso modelo previu Positivo, **28,9% dos próprios clientes marcaram que recomendam o produto** — 3,5 vezes mais que no teste geral! Encontramos clientes que deram nota 1 escrevendo *'excelente loja super recomendo'* e clientes que deram nota 5 ameaçando ir ao PROCON. Tanto é verdade que, quando rodamos o **experimento binário** (sem a nota 3, sem vazamento), o mesmo modelo saltou de **F1-Macro `0,7318` para `0,9388`** e **acurácia de `94,77%`**.
> Na **Pergunta 5** (Figura 13), avaliamos as Top 10 categorias: **Móveis**, mesmo sendo a mais negativa do site, alcançou **F1-Macro de `0,7593`** porque suas classes são equilibradas e os defeitos físicos são explícitos; já **Celulares** ficou em `0,6920` porque a classe Neutra mistura elogios gerais com detalhes técnicos sutis."

### 3.3 Demonstração Ao Vivo no Streamlit + Limitações e Ética (09:25 – 11:30)
- **Ação:** Alternar para o navegador com o **Streamlit (`app.py`)** aberto!
> **Passo a passo da Demo (1 min 45 s — fluido e sem digitação lenta):**
> 1. **Aba 01 (~20 s):** Mostrar rapidamente os KPIs, clicar no botão de rádio **"Cenário Binário — 2 Classes"** para mostrar ao vivo o card verde do salto para `0,9388`, e mostrar logo abaixo as 3 caixas do **Laboratório de Pré-processamento** com os chips verdes salvando `"não"`, `"nunca"` e `"mais"`.
> 2. **Aba 02 (~20 s):** Clicar na **Aba 02**, mostrar o gráfico interativo de ranking e clicar na sub-aba **"Matrizes de Confusão Interativas"** ou **"Erros Reais & Ruído de Rótulos"** para exibir a tabela de casos reais anonimizados do `X_test`.
> 3. **Aba 03 — O Ponto Alto (~1 min 05 s):** Clicar na **Aba 03 (Simulador em Tempo Real)**:
>    - Já estará selecionada **"❌ Negação crítica"**: apontar o card vermelho **`NEGATIVO`**, os chips verdes protegendo `"não"` (mostrando que o NLTK padrão deixaria `"funciona"` e `"recomendo"`!) e o gráfico de **Explicabilidade Local** mostrando os bigramas `"não recomendo"` e `"não funciona"` puxando a decisão, além do consenso dos 4 classificadores embaixo.
>    - Clicar em **"🔄 Negação positiva"** (*"Não trava, a bateria dura bastante e não me arrependo..."*): mostrar que o modelo não classifica cegamente todo `"não"` como negativo — ele acerta **`POSITIVO`** graças aos bigramas e termos positivos!
>    - Clicar em **"🎭 Ironia / sarcasmo"** (*"Parabéns pela transportadora maravilhosa, só levou dois meses para entregar o produto quebrado!"*): **usar intencionalmente para fechar com maturidade crítica!** Explicar: *"Reparem como aqui os pesos de 'parabéns' e 'maravilhosa' competem com 'quebrado' — e também avisamos na tela que o score do LinearSVC vem do softmax da margem, não sendo probabilidade calibrada. Ironia e avaliações multi-aspecto são limitações naturais de modelos Bag of N-grams, abrindo caminho para modelos Transformer como BERTimbau."*
>    - **Fechamento (10 s):** *"Professor, se quiser ditar qualquer frase inédita agora para testarmos ao vivo no simulador, estamos à disposição. Muito obrigada!"*

---

## Guia Rápido de Defesa (Possíveis Perguntas do Professor e Respostas Certas)

1. **"Vocês usaram o conjunto de teste em algum momento para escolher hiperparâmetros ou montar o vocabulário?"**
   - **Resposta (Anaíza):** Não. O `CountVectorizer`/`TfidfVectorizer` só deu `.fit()` em `X_train`. No `GridSearchCV`, usamos `sklearn.pipeline.Pipeline` com `StratifiedKFold(3)` exclusivamente dentro dos `101.352` exemplos de treino, então o vocabulário e o IDF eram recalculados dentro de cada fold. Como transparência metodológica adicional no relatório, destacamos que a representação `tfidf_bi` destacou-se na grade padrão, e a escolha final entre os 4 modelos ajustados foi feita estritamente pelo maior `cv_f1_macro` (`0,7248` do `LinearSVC C=0,1`).
2. **"Como vocês calcularam porcentagem de confiança para o `LinearSVC` se o SVM linear não tem `predict_proba`?"**
   - **Resposta (Hillary):** Excelente ponto — o `LinearSVC` otimiza margem *hinge* e expõe apenas `decision_function`. Para manter uma visualização comparável na interface sem alterar as fronteiras do modelo vencedor, aplicamos a função *softmax* sobre o vetor de margens (`decision_function`) e colocamos um aviso técnico explícito na Aba 3 e no relatório explicando que é uma pontuação relativa de margem, e não probabilidade calibrada (que exigiria `CalibratedClassifierCV`).
3. **"Por que vocês protegeram palavras como `'mas'`, `'porém'` e `'muito'` além do `'não'`?"**
   - **Resposta (Guilherme):** Porque na análise de sentimentos em três classes, intensificadores (`muito bom` vs `bom`) separam o Positivo do Neutro, e conectivos adversativos (`mas`, `porém`) são justamente os termos de maior peso da classe Neutra (como provou a Figura 11 dos coeficientes do modelo!). A lista padrão do NLTK removia `'mas'` e `'muito'`.
4. **"Por que o F1 da classe Neutra ficou em `0,4269` enquanto Negativo e Positivo passaram de `0,86` e `0,90`?"**
   - **Resposta (Hillary):** Três fatores combinados que provamos nos dados: (1) é a menor classe do corpus (`12,51%`); (2) semanticamente mistura vocabulário positivo e negativo na mesma frase (*"produto bom, mas entrega atrasou"*); e (3) a nota 3 sofre forte ruído subjetivo do consumidor — tanto que, removendo apenas a nota 3 no experimento binário, o F1-Macro do mesmo modelo sobe para `0,9388`.
