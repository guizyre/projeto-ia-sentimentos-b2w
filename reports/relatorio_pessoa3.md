# Relatório Técnico — Parte da Pessoa 3 (Avaliação Crítica, Ética e Aplicação Streamlit)
**Projeto 2: Classificação Automática de Sentimentos em Avaliações de Produtos**  
**Disciplina:** Inteligência Artificial – Unidade 3 (IFPB – Bacharelado em Engenharia da Computação)  
**Responsável por esta seção:** Pessoa 3 (Avaliação Experimental Crítica, Análise de Erros, Ética e Interface Streamlit)

---

## 11. Avaliação Experimental e Respostas às 5 Perguntas da Seção 2.6

Todas as análises quantitativas e qualitativas a seguir foram executadas sobre o conjunto de teste estratificado e determinístico (`n_test = 25.338`, `random_state = 42`, **nunca visto no treinamento ou ajuste de hiperparâmetros**), utilizando o módulo `src/evaluation.py` e o notebook `notebooks/04_avaliacao_e_erros.ipynb`, com artefatos consolidados em `reports/evaluation_summary.json`.

### 11.1 Pergunta 1 — Qual configuração apresentou melhor resultado?

Entre as **16 combinações** da grade padrão (BoW vs. TF-IDF × unigramas vs. uni+bigramas × 4 classificadores) e os **4 classificadores ajustados via `GridSearchCV`** (3 folds estratificados em `X_train`, otimizando `f1_macro`), o melhor desempenho global foi obtido por:

> **Modelo Final (escolhido via validação cruzada em `X_train`):**  
> **TF-IDF (Uni+Bigramas, `sublinear_tf=True`, `min_df=3`) + Linear SVM (`class_weight='balanced'`, `C=0,1`)**  
> **F1-Macro (CV):** `0,7248` | **F1-Macro (Teste):** **`0,7318`** | **Acurácia (Teste):** **`0,8378` (`83,78%`)** | **F1-Weighted:** `0,8333`

| Posição (F1-Macro) | Etapa | Representação | Classificador | Params | F1-Macro ★ | Acurácia | F1 Neg. | F1 Neu. | F1 Pos. | Rec. Neg. | Rec. Neu. |
| ---: | :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **1** | **Ajustado (CV)** | **TF-IDF Uni+Bi** | **Linear SVM (`balanced`)** | **`C=0,1`** | **0,7318** | **0,8378** | **0,8671** | **0,4269** | **0,9015** | **0,9138** | **0,3913** |
| 2 | Padrão / Ajustado | TF-IDF Uni+Bi | LogReg (`balanced`) | `C=1` | 0,7233 | 0,7992 | 0,8634 | 0,4398 | 0,8666 | 0,8917 | 0,5484 |
| 3 | Padrão | TF-IDF Uni+Bi | Linear SVM (`balanced`) | `C=1` | 0,7100 | 0,8221 | 0,8610 | 0,3779 | 0,8911 | 0,8913 | 0,3588 |
| 4 | Padrão | BoW Uni+Bi | Naive Bayes | `alpha=1,0` | 0,7073 | 0,8143 | 0,8404 | 0,3914 | 0,8903 | 0,8666 | 0,3932 |
| 5 | Padrão | BoW Uni+Bi | LogReg (`balanced`) | `C=1` | 0,7069 | 0,8006 | 0,8534 | 0,3948 | 0,8725 | 0,8727 | 0,4418 |
| 6 | Ajustado (CV) | TF-IDF Uni+Bi | LogReg (padrão) | `C=3` | 0,7006 | 0,8395 | 0,8706 | 0,3294 | 0,9020 | 0,9011 | 0,2490 |
| 7 | Padrão | TF-IDF Uni+Bi | LogReg (padrão) | `C=1` | 0,6993 | **0,8452** | 0,8729 | 0,3193 | 0,9058 | 0,9072 | 0,2256 |
| — | Ajustado (CV) | TF-IDF Uni+Bi | Naive Bayes | `alpha=0,1` | 0,6851 | 0,8289 | 0,8484 | 0,3081 | 0,8989 | 0,8929 | 0,2348 |

*(Ver `reports/figures/09_f1_macro_comparacao.png` e `reports/figures/12_matrizes_confusao_comparacao.png`.)*

**Síntese crítica da comparação:**
1. **Por que F1-Macro e não Acurácia:** A maior acurácia isolada pertence à Regressão Logística padrão em `tfidf_bi` (**`84,52%`**), mas ela obtém **F1-Macro = `0,6993`** porque acerta apenas **`22,56%`** das avaliações Neutras (concentrando predições na classe majoritária Positiva, com 61,76% do corpus). O Linear SVM ajustado (`C=0,1`, `balanced`) entrega quase a mesma acurácia (**`83,78%`**) elevando o **F1-Macro para `0,7318`** (+0,0325) e o **Recall Neutro para `39,13%`** (+16,57 p.p.).
2. **Trade-off entre SVM (`C=0,1`) e LogReg (`balanced`):** A Regressão Logística `balanced` atinge o maior Recall Neutro (`54,84%`), porém ao custo de reclassificar `2.408` avaliações Positivas como Neutras (Precisão Neutra de apenas `36,71%`, Acurácia `79,92%`). Já o Linear SVM com regularização mais forte (`C=0,1`) alcança melhor Precisão Neutra (`46,97%`), maior Acurácia (`83,78%`), maior F1 Negativo (`0,8671`) e maior F1-Macro (`0,7318`).

---

### 11.2 Pergunta 2 — O modelo consegue identificar adequadamente avaliações negativas?

**Sim, com elevada sensibilidade (Recall) e baixa taxa de erro polar grave.** No modelo final (`TF-IDF Uni+Bi + Linear SVM C=0,1 balanced`), entre as **`6.520` avaliações negativas reais** do conjunto de teste:

- **`5.958` (`91,38%`)** foram corretamente detectadas como **Negativo** (**maior Recall Negativo dentre todas as 20 configurações avaliadas**);
- **`358` (`5,49%`)** foram suavizadas para **Neutro** (erro adjacente de baixa gravidade);
- Apenas **`204` (`3,13%`)** foram confundidas com **Positivo** (falso positivo extremo — e, como demonstrado na Seção 11.4, quase um terço desses 204 casos corresponde a rótulos contraditórios/ruidosos).

| Configuração (em `TF-IDF Uni+Bi`) | Precisão Neg. | Recall Neg. ★ | F1 Neg. | Recall Neu. | F1-Macro |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **Linear SVM `balanced` (`C=0,1`) ★ Final** | 0,8249 | **0,9138** | 0,8671 | 0,3913 | **0,7318** |
| Regressão Logística padrão (`C=1`) | 0,8410 | 0,9072 | **0,8729** | 0,2256 | 0,6993 |
| Regressão Logística padrão (`C=3`, CV) | **0,8421** | 0,9011 | 0,8706 | 0,2490 | 0,7006 |
| Naive Bayes (`alpha=0,1`, CV) | 0,8082 | 0,8929 | 0,8484 | 0,2348 | 0,6851 |
| Regressão Logística `balanced` (`C=1`) | 0,8369 | 0,8917 | 0,8634 | **0,5484** | 0,7233 |
| Linear SVM `balanced` (`C=1`, padrão) | 0,8326 | 0,8913 | 0,8610 | 0,3588 | 0,7100 |

*(Ver `reports/figures/15_recall_negativo_comparacao.png`.)*

A preservação das palavras de negação (`não`, `nem`, `nunca`, `sem`) combinada aos bigramas (`não recebi`, `não recomendo`, `não funciona`, `não gostei`) foi decisiva para manter o F1 Negativo acima de `0,86` e o Recall acima de `91%`. Observa-se ainda que o ajuste `C=0,1` no Linear SVM reduziu sobreajuste a termos raros e elevou simultaneamente o Recall Negativo (`89,13% → 91,38%`), o Recall Neutro (`35,88% → 39,13%`) e o F1-Macro (`0,7100 → 0,7318`).

---

### 11.3 Pergunta 3 — Quais erros aparecem com maior frequência?

No conjunto de teste (`n = 25.338`), o modelo final acertou **`21.227` avaliações (`83,78%`)** e errou **`4.111` (`16,22%`)**. A decomposição completa dos erros por par `(Real → Previsto)` revela um padrão inequívoco (`reports/figures/14_analise_erros_e_ruido.png`, Painel A):

| Par de Erro (`Real → Previsto`) | Quantidade | % dos Erros | % do Teste | Confiança Média |
| :--- | ---: | ---: | ---: | ---: |
| **Neutro → Positivo** | **1.242** | **30,21%** | 4,90% | 0,5762 |
| **Positivo → Neutro** | **1.042** | **25,35%** | 4,11% | 0,4938 |
| **Neutro → Negativo** | **687** | **16,71%** | 2,71% | 0,5236 |
| Positivo → Negativo | 578 | 14,06% | 2,28% | 0,4887 |
| Negativo → Neutro | 358 | 8,71% | 1,41% | 0,4628 |
| Negativo → Positivo | 204 | 4,96% | 0,81% | 0,4580 |
| **Total de erros** | **4.111** | **100,00%** | **16,22%** | — |

**Principais padrões identificados:**
1. **A classe Neutra concentra `80,98%` de todos os erros (`3.329 / 4.111`)**: a maioria esmagadora das falhas ocorre na fronteira adjacente entre `Neutro` e `Positivo` (`55,56%` dos erros) ou entre `Neutro` e `Negativo` (`25,42%`). Inversões extremas de polaridade (`Negativo ↔ Positivo`) representam apenas **`19,02%`** dos erros (`3,09%` do conjunto de teste).
2. **Avaliações mistas (produto vs. entrega / elogio + ressalva):** estimamos que **`1.086` erros (`26,42%`)** combinam termos de elogio com queixas de entrega/defeito ou conectivos adversativos (`mas`, `porém`, `só que`). Ademais, avaliações classificadas incorretamente são em média mais longas (`25,12` palavras contra `22,50` palavras nos acertos), pois textos mais extensos tendem a misturar múltiplos aspectos.
3. **Elogios moderados e preenchimento artificial de caracteres:** na inspeção qualitativa, constatou-se que avaliações nota 4–5 contendo apenas adjetivos moderados (`"bom"`, `"produto bom"`) — inclusive com pontos repetidos (`"bom......."`) inseridos para vencer o mínimo de caracteres do site — são frequentemente previstas como `Neutro`, pois `"bom"` é o termo de maior peso positivo para a classe Neutra (Figura 11).

#### Amostra de 12 erros reais extraídos de `X_test` (sem dados pessoais)

| # | Categoria do Erro | Nota | Real → Previsto | Conf. | Categoria | Trecho real do `X_test` |
| :-: | :--- | :-: | :--- | :-: | :--- | :--- |
| 1 | Mista (produto vs. entrega) | 4 | Positivo → Neutro | 75,0% | Suplementos | *"Produto é bom, porém a entrega não foi entregue no prazo prometido."* |
| 2 | Mista (produto vs. entrega) | 4 | Positivo → Neutro | 68,7% | TV e Home Theater | *"só atrasou alguns dias a entrega. Porém o produto é bom."* |
| 3 | Mista (produto vs. ressalva) | 3 | Neutro → Positivo | 69,1% | Moda | *"Produto de qualidade , Entrega muito boa ( 7 dias) , mas o tamanho das lentes é pequeno para o meu gosto."* |
| 4 | Mista (custo-benefício) | 4 | Positivo → Neutro | 70,5% | Utilidades Dom. | *"Produto bom custo benefício, não é sofisticado. Mas entregue em perfeitas condições."* |
| 5 | Neutro → Positivo | 3 | Neutro → Positivo | 95,0% | Celulares | *"Amei recomendo e o melhor de tudo câmera perfeita_ e grande e fino maravilhoso"* |
| 6 | Neutro → Positivo | 3 | Neutro → Positivo | 92,9% | Casa e Constr. | *"Produto muito bom e de ótima qualidade! Recomendo!!!"* |
| 7 | Neutro → Negativo | 3 | Neutro → Negativo | 93,3% | Wearable | *"Horrível. Depois d um mês d espera, nao veio o produto q eu paguei, veio um outro relógio. E ainda nao devolveram meu dinheiro 1.500 reais. Q decepção, loas Americanas!!!"* |
| 8 | Neutro → Negativo | 3 | Neutro → Negativo | 92,5% | Bebês | *"Infelizmente, sou mais uma insatisfeita com o péssimo atendimento com o cliente, não recebi o produto."* |
| 9 | Positivo → Neutro | 5 | Positivo → Neutro | 81,4% | TV e Home Theater | *"Utilizei pouco a tv, mas me parece ser boa. Até o momento eu recomendaria."* |
| 10 | Positivo → Neutro | 4 | Positivo → Neutro | 77,5% | Celulares | *"E um produto bom , inteligente, porem a bateria deveria ter um pouco de durabilidade !"* |
| 11 | Falso Positivo extremo | 1 | Negativo → Positivo | 88,3% | Cama, Mesa e Banho | *"muito boa a entrega bem antes do previsto exelente loja superr recomendo"* |
| 12 | Falso Negativo extremo | 5 | Positivo → Negativo | 88,4% | Informática | *"como posso avaliar o produto se consta entregue e não recebi o produto. E ninguém me da um posição, creio que vou ter entrar no PROCON para resolver."* |

Note-se que nos exemplos **5, 6, 7, 8, 11 e 12**, a leitura humana confirma que **o texto expressa exatamente o sentimento previsto pelo modelo**, e não a nota registrada no banco — conectando diretamente à Pergunta 4.

---

### 11.4 Pergunta 4 — A regra de conversão de notas em sentimentos introduz ruído nos rótulos?

**Sim — de forma inequívoca e estatisticamente mensurável.** A conversão automática de `overall_rating` (1–5) em classes de sentimento funciona como *supervisão fraca*, introduzindo três fontes de ruído documentadas em `X_test`:

1. **Prova quantitativa via contradição com `recommend_to_a_friend` (`reports/figures/14_analise_erros_e_ruido.png`, Painel B):**
   - Entre todas as avaliações **Negativas** (`nota 1–2`) do teste, **`8,33%`** (`543 / 6.520`) marcaram `recommend_to_a_friend == "Yes"`. Porém, entre os `204` erros extremos em que o modelo previu **Positivo** para uma nota 1–2, essa contradição dispara para **`28,92%`** (`59 / 204`) — **3,5 vezes maior**!
   - Entre todas as avaliações **Positivas** (`nota 4–5`) do teste, apenas **`1,48%`** (`231 / 15.649`) marcaram `recommend_to_a_friend == "No"`. Já nos `578` erros extremos em que o modelo previu **Negativo** para uma nota 4–5, o índice sobe para **`8,48%`** (`49 / 578`) — **5,7 vezes maior**!
2. **Casos de erro de clique do consumidor e avaliações pré-uso:**
   - Detectamos **`99` casos com `nota == 1`** previstos como `Positivo` pelo modelo (ex.: *"Ótimo produto ...recomendo ..chegou antes do prazo"*, nota 1, `recommend = Yes`, confiança `86,4%`; *"produto excelente pra uso e ainda chegou antes do prazo muito obrigado recomendo."*, nota 1, `recommend = Yes`, confiança `79,2%`).
   - Detectamos **`283` casos com `nota == 5`** previstos como `Negativo` pelo modelo (ex.: *"Não recebi ainda Não recebi ainda Não recebi ainda"*, nota 5, confiança `85,4%`; *"Não consigo avaliar o produto, pois ele não foi entregue! As lojas Americanas ofertou como alternativa somente o cancelamento da compra."*, nota 5, confiança `86,4%`).
   - Adicionalmente, **`180` avaliações** com nota 4 ou 5 no teste contêm expressões como `"ainda não"`, `"não testei"`, `"não abri"` ou `"não recebi"` (`57` delas previstas como Negativo).
3. **Experimento complementar — Cenário Binário (Funcionalidade 2 da Seção 2.4):**
   Para quantificar o impacto da ambiguidade da **nota 3**, treinamos o mesmo classificador (`TF-IDF Uni+Bi + LinearSVC(C=0,1, balanced)`) removendo a nota 3 tanto em `X_train` (`88.678` registros) quanto em `X_test` (`22.169` registros), sem qualquer *data leakage*:

| Cenário de Rotulagem | Acurácia | F1-Macro | Precisão Neg. | Recall Neg. | F1 Neg. | F1 Pos. |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **3 Classes (Padrão: 1–2 / 3 / 4–5)** | 0,8378 | 0,7318 | 0,8249 | 0,9138 | 0,8671 | 0,9015 |
| **2 Classes (Binário: 1–2 vs. 4–5, sem nota 3)** | **0,9477** | **0,9388** | **0,8721** | **0,9633** | **0,9155** | **0,9621** |
| **Diferença** | **+10,99 p.p.** | **+0,2070** | +4,72 p.p. | +4,95 p.p. | +0,0484 | +0,0606 |

Esse salto de **+0,2070 em F1-Macro** (e **94,77% de acurácia**) comprova que o teto de desempenho no cenário ternário não se deve à incapacidade do modelo em distinguir polaridade, mas sim à **zona cinzenta intrínseca da nota 3** e ao ruído dos rótulos derivados de estrelas.

---

### 11.5 Pergunta 5 — O desempenho se mantém em avaliações de produtos ou categorias diferentes?

A análise segmentada por `site_category_lv1` nas **Top 10 categorias** do conjunto de teste (incluindo as Top 5 em volume e **`Móveis`**) mostra que a **Acurácia permanece relativamente estável** (`81,89%` a `85,55%`), mas o **F1-Macro varia `0,0720` pontos** (`0,6920` a `0,7640`), acompanhando diretamente o desempenho na classe Neutra (`reports/figures/13_desempenho_por_categoria.png`):

| Categoria (`site_category_lv1`) | n (Teste) | % Neg. | % Neu. | % Pos. | Acurácia | F1-Macro | F1 Neg. | F1 Neu. | F1 Pos. | Rec. Neg. |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Utilidades Domésticas | 1.546 | 25,1% | 13,3% | 61,6% | 0,8461 | **0,7640** | 0,8774 | **0,5099** | 0,9046 | 0,8943 |
| Brinquedos | 941 | 27,3% | 12,1% | 60,6% | **0,8555** | 0,7633 | 0,8787 | 0,4977 | 0,9135 | 0,9300 |
| **Móveis** | **1.303** | **42,7%** | **15,8%** | **41,5%** | **0,8289** | **0,7593** | **0,8983** | **0,5051** | **0,8744** | **0,9299** |
| Casa e Construção | 852 | 33,2% | 14,2% | 52,6% | 0,8439 | 0,7523 | 0,8938 | 0,4660 | 0,8972 | 0,9364 |
| Informática | 822 | 28,6% | 15,8% | 55,6% | 0,8224 | 0,7453 | 0,8687 | 0,4848 | 0,8824 | 0,9149 |
| Beleza e Perfumaria | 1.784 | 25,4% | 12,0% | 62,6% | 0,8380 | 0,7322 | 0,8574 | 0,4388 | 0,9004 | 0,9161 |
| Eletroportáteis | 2.317 | 16,9% | 11,4% | 71,6% | 0,8502 | 0,7232 | 0,8407 | 0,4134 | **0,9157** | 0,8954 |
| Informática e Acessórios | 1.447 | 31,8% | 13,0% | 55,2% | 0,8189 | 0,7147 | 0,8859 | 0,3871 | 0,8711 | 0,9196 |
| TV e Home Theater | 1.444 | 22,3% | 10,7% | 67,0% | 0,8255 | **0,6920** | 0,8309 | 0,3455 | 0,8995 | 0,8851 |
| Celulares e Smartphones | 4.055 | 19,2% | 11,5% | 69,3% | 0,8358 | **0,6920** | 0,8314 | **0,3350** | 0,9097 | 0,9038 |
| **Global (todo o conjunto de teste)** | **25.338** | **25,7%** | **12,5%** | **61,8%** | **0,8378** | **0,7318** | **0,8671** | **0,4269** | **0,9015** | **0,9138** |

**Interpretação crítica:**
- **Surpresa positiva em `Móveis`:** embora seja a categoria com maior índice de insatisfação (`42,7%` negativas e `15,8%` neutras no teste), `Móveis` alcançou o **3º maior F1-Macro (`0,7593`)**, o **maior F1 Negativo (`0,8983`)** e **Recall Negativo de `92,99%`**. Isso ocorre porque suas classes são muito mais equilibradas e o vocabulário de reclamações/ressalvas (montagem, avarias, peças faltantes, madeira frágil) é altamente discriminativo.
- **Queda relativa em eletrônicos (`Celulares e Smartphones` e `TV e Home Theater`):** ambas registraram o menor F1-Macro (`0,6920`) apesar do grande volume de treino (`Celulares` é a maior categoria do corpus!). O motivo está no **F1 Neutro mais baixo (`0,3350` e `0,3455`)**: avaliações de tecnologia são muito concentradas em notas 4–5 (~67–69%), contêm jargões técnicos/nomes de modelos e misturam elogios com queixas pontuais de bateria, tela ou sistema, dificultando separar nota 3 de nota 4.

---

## 12. Análise Crítica: Limitações, Vieses e Cuidados Éticos (Seção 2.8)

### 12.1 Limitações Técnicas e Linguísticas
1. **Ironia, sarcasmo e linguagem informal:** modelos lineares baseados em *Bag of N-grams* avaliam somas ponderadas de unigramas e bigramas; por isso, frases irônicas que empregam vocabulário positivo com sentido depreciativo (ex.: *"Parabéns pela transportadora incrível, só levou dois meses para entregar um produto quebrado"*) podem confundir o classificador quando o peso de `"parabéns"`/`"incrível"` compete com `"quebrado"`. Além disso, erros ortográficos e abreviações informais (`vc`, `q`, `exelente`, `nao`/`porem` sem acento) fragmentam a contagem de *features* no vocabulário.
2. **Avaliações multi-aspecto (produto vs. entrega):** uma única classificação global (`Positivo`/`Neutro`/`Negativo`) não separa a satisfação com o produto da insatisfação com a transportadora ou o marketplace. Para separar esses alvos seria necessária uma abordagem de *Aspect-Based Sentiment Analysis* (ABSA) ou modelos baseados em *Transformers* em português (ex.: BERTimbau), previstos como extensão futura na Seção 2.7.
3. **Escopo limitado de negação:** bigramas capturam negações imediatas (`não recomendo`, `não funciona`), mas não resolvem negações de longa distância (ex.: *"Não posso dizer que o atendimento foi ruim"*).
4. **Calibração das pontuações de confiança:** o classificador final (`LinearSVC`) maximiza margens e não implementa `predict_proba` nativo. Na aplicação Streamlit, convertemos `decision_function` via *softmax* apenas como indicador relativo de confiança, exibindo aviso técnico explícito ao usuário.

### 12.2 Vieses do Corpus e Generalização
- **Desequilíbrio de classes e supervisão fraca:** a dominância da classe Positiva (`61,8%`) e o ruído da regra de estrelas impõem um teto natural às métricas da classe Neutra (`F1 ≈ 0,43`).
- **Viés de domínio, plataforma e período:** todas as avaliações são da *Americanas.com* coletadas entre janeiro e maio de 2018. Termos específicos da plataforma (`americanas`, `submarino`, nomes de marcas/aparelhos de 2018) estão presentes no vocabulário; a aplicação do modelo a outros setores (ex.: saúde, política, redes sociais) ou períodos atuais exigiria retreinamento e validação em domínio alvo (como o corpus RePro).
- **Viés de autosseleção:** consumidores extremamente satisfeitos ou fortemente frustrados tendem a avaliar com mais frequência do que consumidores indiferentes.

### 12.3 Ética, Privacidade e Licença dos Dados
- **Privacidade estrita (Seção 2.8):** as quatro colunas potencialmente sensíveis ou demográficas (`reviewer_id`, `reviewer_birth_year`, `reviewer_gender`, `reviewer_state`) são descartadas imediatamente em `src/data_loader.py` e validadas por testes automatizados (`tests/test_pipeline.py`). Nenhum dado pessoal é gravado nos artefatos JSON/joblib nem exibido no Streamlit.
- **Respeito à licença `CC BY-NC-SA 4.0`:** o arquivo `B2W-Reviews01.csv` não é redistribuído no repositório Git (bloqueado no `.gitignore`); o `README.md` e o módulo `src/data_loader.py` documentam integralmente a origem, autoria, licença e instruções de download.
- **Uso responsável em produção:** o sistema deve atuar como apoio à triagem agregada e monitoramento de tendências, nunca para decisões automatizadas punitivas ou sem supervisão humana — especialmente em predições classificadas como `Neutro` ou com baixa margem de confiança.

---

## 13. Aplicação Demonstrável em Streamlit (`app.py`) e Testes Integrados

Para integrar os módulos de todo o trio e atender integralmente às **7 funcionalidades obrigatórias da Seção 2.4**, implementou-se a aplicação web local `app.py` (executada via `streamlit run app.py`), com carregamento instantâneo por cache (`@st.cache_data` e `@st.cache_resource`) dos resumos JSON e pipelines `.joblib`:

- **Aba 1 – Corpus, Rotulagem e Pré-processamento:** exibe KPIs da auditoria (`132.373 → 126.690`), funil de limpeza, **seletor interativo de regra de mapeamento** (comparando 3 classes, cenário binário com resultado experimental e regra customizada estrela por estrela), gráficos da EDA (`01..04`) e **comparador ao vivo de pré-processamento** (NLTK padrão sem proteção vs. nosso pipeline preservando acentos e negações vs. *negation marking*) + gráficos `05..08`.
- **Aba 2 – Comparação de Modelos, Métricas e Análise de Erros:** destaca o modelo final (`F1-Macro = 0,7318`), oferece tabela interativa filtrável das **20 configurações** (16 padrão + 4 ajustadas), **explorador dinâmico de Matriz de Confusão** para qualquer modelo, visualizações (`09`, `11`, `12`, `13`, `14`, `15`), desempenho por categoria (`site_category_lv1`) e explorador filtrável de erros reais e ruído de rótulos do `X_test`.
- **Aba 3 – Simulador em Tempo Real (Exemplos Inéditos):** permite digitar qualquer avaliação inédita (ou testar 6 exemplos pré-prontos de 1 clique: elogio, negação, mista, neutro, negação positiva e ironia), escolher qualquer modelo ajustado ou combinação livre da grade padrão, e visualizar: sentimento previsto, gráfico de confiança por classe, auditoria do texto normalizado (alertando quais negações foram salvas), **gráfico e tabela dos tokens/bigramas que mais pesaram na decisão** (`explain_prediction()`), e comparação simultânea entre os 4 classificadores ajustados.

Adicionalmente, implementou-se a suíte de testes integrados **`tests/test_pipeline.py`** (10 testes automatizados executados via `python -m unittest discover -s tests -v`), cobrindo remoção de colunas pessoais, limpeza, preservação de acentos/negações, carregamento de todos os modelos salvos, explicabilidade nas três famílias de classificadores (`LinearSVC`, `LogisticRegression`, `MultinomialNB`), integridade dos resumos/figuras e validação sintática de `app.py`.

---

## 14. Declaração Transparente sobre Uso de IA Generativa (Seção 8)

Ferramentas de Inteligência Artificial generativa foram utilizadas pela Pessoa 3 como apoio no desenvolvimento e revisão estrutural de `src/evaluation.py`, `notebooks/04_avaliacao_e_erros.ipynb`, `app.py` e `tests/test_pipeline.py`, bem como na revisão redacional deste relatório. Todas as métricas, matrizes de confusão, taxas de ruído, resultados por categoria e exemplos reais do `X_test` foram integralmente gerados, executados e verificados localmente sobre o corpus **B2W-Reviews01** e os modelos treinados pela equipe.

---

## 15. Referências

1. AMERICANAS TECH / B2W DIGITAL. **B2W-Reviews01 — Open Corpus of Product Reviews in Portuguese**. Disponível em: <https://github.com/americanas-tech/b2w-reviews01>. Licença CC BY-NC-SA 4.0.
2. REAL, L.; OGAWA, M.; COELHO, F.; COELHO, L. **B2W-Reviews01: An Open Dataset for Sentiment Analysis in Portuguese**. STIL — Symposium in Information and Human Language Technology, 2019.
3. CHAVES, R. **B2W-Reviews01 (Hugging Face Dataset Mirror)**. Disponível em: <https://huggingface.co/datasets/ruanchaves/b2w-reviews01>.
4. PEDREGOSA, F. et al. **Scikit-learn: Machine Learning in Python**. Journal of Machine Learning Research, v. 12, p. 2825–2830, 2011.
5. BIRD, S.; KLEIN, E.; LOPER, E. **Natural Language Processing with Python (NLTK)**. O'Reilly Media, 2009.
6. JURAFSKY, D.; MARTIN, J. H. **Speech and Language Processing** (3rd ed. draft). Stanford University, 2024.
7. STREAMLIT INC. **Streamlit Documentation**. Disponível em: <https://docs.streamlit.io/>.

---

## 16. Roteiro Sugerido para Apresentação Oral — Bloco da Pessoa 3 (3 a 4 minutos)
*(Não faz parte do corpo do relatório técnico.)*

1. **Avaliação crítica e Perguntas 1–2 (~45 s):** Mostrar `12_matrizes_confusao_comparacao.png` e `15_recall_negativo_comparacao.png`. Destacar que o modelo final (`TF-IDF uni+bi + Linear SVM C=0,1 balanced`) atinge **F1-Macro 0,7318**, **Acurácia 83,78%** e **Recall Negativo de 91,38%** (apenas 3,1% das negativas viram falso positivo).
2. **Análise de erros, ruído de rótulos e categorias — Perguntas 3–5 (~1 min 15 s):**
   - Mostrar `14_analise_erros_e_ruido.png`: **81,0% dos 4.111 erros envolvem a classe Neutra** e **26,4% são avaliações mistas** (produto bom + entrega ruim).
   - Provar o ruído de rótulo: nos erros extremos `Negativo → Positivo`, **28,9%** dos clientes marcaram `recomenda='Yes'` (vs. 8,3% no teste geral — 3,5× mais!) e citar um exemplo real nota 1 elogiando a loja e um nota 5 reclamando do PROCON.
   - Mencionar o experimento binário: sem a nota 3, o F1-Macro salta de **0,7318 para 0,9388** (acurácia **94,77%**).
   - Mostrar `13_desempenho_por_categoria.png`: **`Móveis`** teve F1-Macro **0,7593** (acima da média!), enquanto **`Celulares`** ficou em **0,6920** por causa do Neutro mais ambíguo.
3. **Limitações, ética e privacidade (~30 s):** Ironia, avaliações multi-aspecto, viés de domínio (Americanas 2018), confiança do `LinearSVC` via softmax (não calibrada), zero dados pessoais exibidos e respeito à licença `CC BY-NC-SA 4.0`.
4. **Demonstração ao vivo no Streamlit (`app.py`, ~1 min 30 s):**
   - Mostrar rapidamente a **Aba 1** (alternando 3 classes ↔ Binário) e a **Aba 2** (matriz de confusão interativa).
   - Ir para a **Aba 3 (Simulador)**, clicar em **Negação crítica** (mostrando a proteção do `"não"` e os bigramas `não recomendo` / `não funciona` no gráfico de tokens), depois em **Avaliação mista** (mostrando a disputa entre `ótimo` vs `mas` / `atrasou` / `rasgada`), e convidar o professor a ditar uma avaliação inédita ao vivo!
