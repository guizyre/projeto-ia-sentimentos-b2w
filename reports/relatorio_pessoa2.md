# Relatório Técnico — Parte da Pessoa 2 (Representação Textual e Modelagem)
**Projeto 2: Classificação Automática de Sentimentos em Avaliações de Produtos**
**Disciplina:** Inteligência Artificial – Unidade 3 (IFPB)
**Responsável por esta seção:** Anaíza (Representação vetorial, classificadores, ajuste de hiperparâmetros)

---

## 5. Metodologia de Modelagem

### 5.1 Divisão treino/teste e prevenção de data leakage
Sobre as **126.690** avaliações pré-processadas aplicou-se `train_test_split(test_size=0.20, stratify=y, random_state=42)`, resultando em **101.352** avaliações de treino e **25.338** de teste, com proporções de classe idênticas nos dois conjuntos (61,76% Positivo, 25,73% Negativo, 12,50–12,51% Neutro).

Para evitar vazamento (Seção 2.5.4 do edital):
- o `fit` de `CountVectorizer`/`TfidfVectorizer` (vocabulário, `min_df`, IDF) foi feito **somente em `X_train`**; em `X_test` usou-se apenas `transform`;
- o `GridSearchCV` rodou **somente em `X_train`**, dentro de um `Pipeline`, de modo que o vetorizador é refeito em cada fold de validação;
- o conjunto de teste foi usado apenas para a avaliação final das configurações.

Todos os números desta seção provêm de `reports/model_results.json`, gerado por `python -m src.models` sobre o corpus completo (`random_state=42`).

### 5.2 Representações textuais (4 configurações)
Todas com `min_df=3` e `token_pattern=r"[a-záàâãéêíóôõúüç_]+"` (preserva acentos do português e o `_` do *negation marking*):

| Código | Representação | Nº de atributos |
| :--- | :--- | ---: |
| `bow_uni` | Bag of Words, unigramas | 15.978 |
| `bow_bi` | Bag of Words, uni+bigramas | 78.918 |
| `tfidf_uni` | TF-IDF (`sublinear_tf=True`), unigramas | 15.978 |
| `tfidf_bi` | TF-IDF (`sublinear_tf=True`), uni+bigramas | 78.918 |

### 5.3 Classificadores (4 variantes)
- **Multinomial Naive Bayes** (baseline rápido, adequado a contagens);
- **Regressão Logística** (`max_iter=1000`), com `class_weight=None` e com `class_weight='balanced'`;
- **Linear SVM** (`LinearSVC(class_weight='balanced', random_state=42)`).

As variantes `balanced` atribuem peso maior à classe minoritária (Neutro), atacando o desequilíbrio descrito na Seção 3.2.

### 5.4 Métrica principal
Como a classe Positiva representa 61,8% do corpus, um modelo que ignore o Neutro ainda tem acurácia alta. Por isso o critério de comparação é o **F1-Macro** (média simples dos F1 das três classes), com acurácia, precisão, recall e F1 por classe como apoio.

---

## 6. Resultados Experimentais (16 combinações, conjunto de teste)

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

*(Figura: `reports/figures/09_f1_macro_comparacao.png`.)*

### 6.1 Leitura dos resultados
1. **Acurácia e F1-Macro apontam modelos diferentes.** A maior acurácia (0,8452, LogReg padrão em TF-IDF uni+bi) recupera apenas 22,6% dos Neutros; a melhor configuração por F1-Macro (LogReg `balanced`) recupera 54,8%, ao custo de ~4,6 p.p. de acurácia.
2. **Bigramas ajudam.** Em média, uni+bigramas obtém F1-Macro 0,6898 contra 0,6707 dos unigramas (+0,019). Bigramas como `não recomendo`, `ainda não` e `mas não` (Seção 4.1) carregam polaridade que os unigramas separados perdem.
3. **`class_weight='balanced'` eleva o F1-Macro.** Média da Regressão Logística: 0,7046 (`balanced`) contra 0,6853 (padrão); o F1 médio do Neutro sobe de 0,302 para 0,411.
4. **Naive Bayes depende da representação.** Com contagens (BoW uni+bi) é competitivo (0,7073); com TF-IDF o F1 do Neutro cai para ~0,05 (recall de 2,4% a 3,2%), ou seja, o modelo praticamente nunca prevê Neutro. Duas leituras são compatíveis com os dados: (i) o TF-IDF produz valores contínuos, enquanto o Naive Bayes multinomial foi concebido para contagens; (ii) a suavização padrão (`alpha=1,0`) é grande demais frente aos valores pequenos do TF-IDF, achatando as verossimilhanças e deixando a probabilidade a priori da classe majoritária dominar. A segunda é reforçada pelo ajuste de hiperparâmetros (Seção 7): com `alpha=0,1`, o F1-Macro do Naive Bayes em TF-IDF uni+bi sobe de 0,5969 para 0,6851. Não isolamos experimentalmente qual das duas explicações pesa mais.
5. **BoW vs TF-IDF:** nos modelos lineares (LogReg padrão, LogReg `balanced` e SVM), o TF-IDF supera o BoW nas 6 comparações pareadas (de +0,0015 a +0,0333 de F1-Macro; a maior diferença é no SVM com bigramas). Três dessas diferenças são menores que 0,01 (+0,0015, +0,0071 e +0,0079) e devem ser lidas como tendência, não como diferença firme (ver Seção 10). A média geral do BoW (0,6875 contra 0,6730) só é maior por causa do colapso do Naive Bayes em TF-IDF.

---

## 7. Ajuste de Hiperparâmetros

`GridSearchCV` (`scoring='f1_macro'`, `StratifiedKFold(3, shuffle=True, random_state=42)`), **somente em `X_train`**, na representação `tfidf_bi`:

| Classificador | Grade | Melhor valor | F1-Macro (CV) | F1-Macro (teste) |
| :--- | :--- | :--- | ---: | ---: |
| Naive Bayes | `alpha` ∈ {0,1; 0,3; 0,5; 1,0} | 0,1 | 0,6800 | 0,6851 |
| LogReg padrão | `C` ∈ {0,5; 1; 3; 10} | 3 | 0,7010 | 0,7006 |
| LogReg `balanced` | `C` ∈ {0,5; 1; 3; 10} | 1 | 0,7171 | 0,7233 |
| **SVM `balanced`** | `C` ∈ {0,03; 0,1; 0,3; 1} | **0,1** | **0,7248** | **0,7318** |

O modelo final foi escolhido pelo **F1-Macro de validação cruzada** (sem consultar o teste): **TF-IDF uni+bigramas + Linear SVM (`C=0,1`, `balanced`)**.

**Desempenho do modelo final no teste:** acurácia **0,8378**, F1-Macro **0,7318**.

| Classe | Precisão | Recall | F1 |
| :--- | ---: | ---: | ---: |
| Negativo | 0,8249 | 0,9138 | 0,8671 |
| Neutro | 0,4697 | 0,3913 | 0,4269 |
| Positivo | 0,9066 | 0,8965 | 0,9015 |

Matriz de confusão (linhas = real; colunas = previsto; ordem Negativo, Neutro, Positivo):

|  | Pred. Neg. | Pred. Neu. | Pred. Pos. |
| :--- | ---: | ---: | ---: |
| **Real Neg.** | 5.958 | 358 | 204 |
| **Real Neu.** | 687 | 1.240 | 1.242 |
| **Real Pos.** | 578 | 1.042 | 14.029 |

Dos 3.169 Neutros reais, 39,1% foram acertados, 21,7% foram classificados como Negativo e 39,2% como Positivo. A classe Neutra é a fronteira difícil, coerente com o que os bigramas da Seção 4.1 antecipavam (elogios misturados com ressalvas: `bom mas`, `mas não`).

---

## 8. Termos Mais Influentes (Regressão Logística `balanced`, TF-IDF uni+bi)
*(Figura: `reports/figures/11_termos_mais_influentes.png`.)*

- **Negativo:** `não recomendo`, `não`, `não gostei`, `péssimo`, `não funciona`, `ruim`, `nao`, `dinheiro`, `horrível`, `péssima`.
- **Neutro:** `bom`, `gostei`, `porém`, `produto bom`, `boa`, `mas`, `bom produto`, `gostando`, `bons`, `porem`.
- **Positivo:** `excelente`, `ótimo`, `ótima`, `muito bom`, `perfeito`, `amei`, `adorei`, `recomendo`, `maravilhoso`, `lindo`.

Os pesos confirmam a decisão da Pessoa 1: `não`, `não recomendo` e `não gostei` são os termos mais fortes da classe Negativa; teriam sido apagados pela lista padrão de stopwords. Para o Neutro, o modelo aprende conectivos adversativos (`porém`, `mas`) e elogios moderados (`bom`, `gostei`), em vez dos superlativos que marcam a classe Positiva (`excelente`, `amei`).

---

## 9. Experimento Adicional: *Negation Marking* em Unigramas
Comparou-se `não gostei` → `não_gostei` (unigramas) com o pré-processamento padrão, usando o mesmo conjunto de teste:

| Representação | Classificador | F1 unigr. padrão | F1 unigr. com marcação | Ganho |
| :--- | :--- | ---: | ---: | ---: |
| BoW | Naive Bayes | 0,6770 | 0,6888 | +0,0118 |
| BoW | LogReg padrão | 0,6684 | 0,6808 | +0,0124 |
| BoW | LogReg `balanced` | 0,6902 | 0,6962 | +0,0061 |
| BoW | SVM `balanced` | 0,6754 | 0,6840 | +0,0086 |
| TF-IDF | Naive Bayes | 0,5874 | 0,5986 | +0,0113 |
| TF-IDF | LogReg padrão | 0,6755 | 0,6852 | +0,0097 |
| TF-IDF | LogReg `balanced` | 0,6981 | 0,7054 | +0,0073 |
| TF-IDF | SVM `balanced` | 0,6937 | 0,7023 | +0,0086 |

*(Valores do notebook `03_treinamento_modelos.ipynb`, seção 5; os baselines coincidem com `reports/model_results.json`.)*

A marcação melhora todas as oito combinações (+0,006 a +0,012), mas, na maioria, **não alcança os bigramas** (ex.: TF-IDF LogReg `balanced` 0,7054 vs 0,7233 com bigramas). Há duas exceções: o SVM em BoW, em que unigramas com marcação (0,6840) superam os bigramas (0,6767), e o Naive Bayes em TF-IDF (0,5986 vs 0,5969), em que a diferença é desprezível num modelo que já colapsa na classe Neutra (Seção 6.1, item 4). Os bigramas capturam também outras combinações (`ainda não`, `muito bom`), não só negações.

---

## 10. Limitações desta etapa
- **Seleção sobre o teste na grade padrão.** As 16 combinações foram comparadas no mesmo conjunto de teste, e a representação `tfidf_bi` foi escolhida por esse resultado antes do tuning; isso pode inflar levemente o desempenho reportado. O modelo final, porém, foi escolhido por validação cruzada em `X_train`.
- **Divisão única e diferenças pequenas.** Todos os resultados vêm de uma única divisão treino/teste, sem intervalos de confiança. Duas execuções do mesmo código no mesmo corpus produziram F1-Macro que diferem em até ~0,002 para a Regressão Logística (provavelmente por não convergência do `lbfgs` em 1000 iterações, com avisos suprimidos). Diferenças dessa ordem de grandeza, ou até ~0,01, entre configurações não devem ser tratadas como superioridade firme.
- **Convergência do Linear SVM.** O `LinearSVC` em BoW levou cerca de 100 s (unigramas) e 200 s (uni+bigramas) para treinar, contra poucos segundos em TF-IDF, o que sugere que pode ter atingido o `max_iter` padrão. Se for o caso, parte da desvantagem do BoW no SVM pode ser falta de convergência, e não da representação em si. Não foi testado com `max_iter` maior.
- **Linear SVM sem probabilidades.** O `LinearSVC` não possui `predict_proba`; na aplicação, a "confiança" é o softmax do `decision_function`, que é uma pontuação relativa e não uma probabilidade calibrada.
- **Ruído de rótulo.** Os rótulos vêm da nota (Seção 3.2); parte dos erros do Neutro pode refletir ruído de rotulagem, não só falha do modelo. A análise de erros é da Pessoa 3.
- **Grade restrita.** Ajustou-se apenas um hiperparâmetro por modelo, na melhor representação, e com 3 folds, por custo computacional.

---

## 11. Roteiro Sugerido para Apresentação Oral — Bloco da Pessoa 2 (3 a 4 minutos)
*(Não faz parte do corpo do relatório.)*

1. **Pipeline e anti-vazamento (~45 s):** divisão estratificada 80/20 (101.352 treino / 25.338 teste); vetorizador ajustado só no treino; `GridSearchCV` também só no treino, via `Pipeline`.
2. **Representações e classificadores (~45 s):** BoW vs TF-IDF × unigramas vs uni+bigramas (15.978 vs 78.918 atributos); Naive Bayes, Regressão Logística (padrão e `balanced`) e Linear SVM.
3. **Resultado principal (~1 min):** mostrar a figura 09. Melhor grade padrão: TF-IDF uni+bi + LogReg `balanced` (F1-Macro 0,7233); após tuning, **TF-IDF uni+bi + SVM `C=0,1`** (CV 0,7248, teste 0,7318, acurácia 0,8378). Destacar que o modelo de maior acurácia (0,8452) tem F1-Macro menor porque acerta só 22,6% dos Neutros.
4. **Bigramas, balanceamento e Naive Bayes (~45 s):** bigramas +0,019 de F1-Macro em média; `balanced` leva o F1 médio do Neutro de 0,30 para 0,41; Naive Bayes colapsa em TF-IDF (F1 Neutro ≈ 0,05), mas com `alpha=0,1` recupera para 0,685.
5. **Palavras mais influentes e passagem de bastão (~45 s):** mostrar a figura 11 (`não recomendo`, `não gostei`, `péssimo` no Negativo; `porém`, `mas`, `bom` no Neutro) e passar para a **Pessoa 3**, que vai analisar onde o Neutro é confundido e demonstrar o simulador ao vivo.

---

## 12. Nota de Uso de IA Generativa (para a seção final do relatório do trio)
Ferramentas de IA generativa foram utilizadas como apoio na revisão do `src/models.py`, na conferência dos números do relatório contra as saídas do notebook e do `model_results.json`, e na revisão textual desta seção. Os experimentos foram executados localmente sobre o corpus real **B2W-Reviews01**.