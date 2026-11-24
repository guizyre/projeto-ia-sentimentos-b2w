from __future__ import annotations
import json
import time
import warnings
from pathlib import Path
import joblib
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from src.data_loader import ID_TO_SENTIMENT, PROJECT_ROOT, load_and_clean_data

SEED = 42
LABELS = [0, 1, 2]
LABEL_NAMES = [ID_TO_SENTIMENT[i] for i in LABELS]  # Negativo, Neutro, Positivo
TOKEN_PATTERN = r"[a-záàâãéêíóôõúüç_]+"  # preserva acentos e '_'

MODELS_DIR = PROJECT_ROOT / "models_saved"
REPORTS_DIR = PROJECT_ROOT / "reports"

REPRESENTATIONS = {
    "bow_uni": (CountVectorizer, dict(ngram_range=(1, 1))),
    "bow_bi": (CountVectorizer, dict(ngram_range=(1, 2))),
    "tfidf_uni": (TfidfVectorizer, dict(ngram_range=(1, 1), sublinear_tf=True)),
    "tfidf_bi": (TfidfVectorizer, dict(ngram_range=(1, 2), sublinear_tf=True)),
}

CLASSIFIERS = {
    "nb": lambda: MultinomialNB(),
    "logreg": lambda: LogisticRegression(max_iter=1000),
    "logreg_bal": lambda: LogisticRegression(max_iter=1000, class_weight="balanced"),
    "svm_bal": lambda: LinearSVC(class_weight="balanced", random_state=SEED),
}

# Grades de hiperparâmetros (só rodam dentro de X_train, via validação cruzada)
PARAM_GRIDS = {
    "nb": {"clf__alpha": [0.1, 0.3, 0.5, 1.0]},
    "logreg": {"clf__C": [0.5, 1, 3, 10]},
    "logreg_bal": {"clf__C": [0.5, 1, 3, 10]},
    "svm_bal": {"clf__C": [0.03, 0.1, 0.3, 1]},
}


# Dados e split
def get_split(sample_size=None, mark_negation=False):
    """Carrega o corpus e devolve (df, X_train, X_test, y_train, y_test).

    O split é determinístico (80/20, stratify=y, random_state=42) e preserva o
    índice de `df`, então `df.loc[X_test.index]` recupera texto original,
    nota e categoria (usado pela Pessoa 3).
    """
    kwargs = {"mark_negation": True} if mark_negation else None
    df = load_and_clean_data(sample_size=sample_size, apply_preprocessing=True,
                             preprocessing_kwargs=kwargs, auto_download=True)
    X, y = df["review_text_clean"], df["sentiment_id"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED)
    return df, X_tr, X_te, y_tr, y_te


def make_vectorizer(rep: str, min_df: int = 3):
    cls, params = REPRESENTATIONS[rep]
    return cls(min_df=min_df, token_pattern=TOKEN_PATTERN, lowercase=False, **params)

# Métricas e experimentos
def evaluate(y_true, y_pred) -> dict:
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, zero_division=0)
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
    }
    for i, name in enumerate(LABEL_NAMES):
        out[f"precision_{name}"], out[f"recall_{name}"], out[f"f1_{name}"] = p[i], r[i], f[i]
    return out


def run_experiments(X_train, X_test, y_train, y_test, reps=None, clfs=None, verbose=True):
    """Treina todas as combinações representação × classificador.

    O vetorizador é ajustado (fit) só em X_train; X_test recebe apenas transform.
    Retorna (lista de resultados, vetorizadores, modelos).
    """
    reps, clfs = reps or list(REPRESENTATIONS), clfs or list(CLASSIFIERS)
    results, vecs, models = [], {}, {}
    for rep in reps:
        vec = make_vectorizer(rep)
        Xtr = vec.fit_transform(X_train)   # fit APENAS no treino
        Xte = vec.transform(X_test)        # teste: só transform
        vecs[rep] = vec
        for name in clfs:
            t0 = time.time()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # avisos de convergência do LogReg em BoW
                model = CLASSIFIERS[name]().fit(Xtr, y_train)
            res = {"representation": rep, "classifier": name,
                   "n_features": Xtr.shape[1], "fit_seconds": round(time.time() - t0, 1),
                   **evaluate(y_test, model.predict(Xte))}
            results.append(res)
            models[(rep, name)] = model
            if verbose:
                print(f"{rep:<10} {name:<11} acc={res['accuracy']:.4f} "
                      f"F1-macro={res['f1_macro']:.4f} ({res['fit_seconds']}s)")
    return results, vecs, models


def tune(X_train, y_train, rep, clf, cv=3):
    """GridSearchCV (F1-Macro) sobre um Pipeline, somente com dados de treino.

    Usar Pipeline faz o vetorizador ser reajustado dentro de cada fold: sem vazamento.
    """
    pipe = Pipeline([("vec", make_vectorizer(rep)), ("clf", CLASSIFIERS[clf]())])
    gs = GridSearchCV(pipe, PARAM_GRIDS[clf], scoring="f1_macro",
                      cv=StratifiedKFold(cv, shuffle=True, random_state=SEED), n_jobs=-1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gs.fit(X_train, y_train)
    return gs


# Inferência 
def predict_with_confidence(vectorizer, model, texto_tratado: str) -> dict:
    """Prediz uma avaliação já passada por `preprocess_text()`.

    LinearSVC não tem predict_proba: usa-se softmax do decision_function como
    pontuação de confiança (não é probabilidade calibrada).
    """
    x = vectorizer.transform([texto_tratado])
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)[0]
    else:
        s = model.decision_function(x)[0]
        e = np.exp(s - s.max())
        proba = e / e.sum()
    pred = int(np.argmax(proba))
    return {"pred_id": pred, "sentimento": ID_TO_SENTIMENT[pred],
            "probabilidades": {n: float(p) for n, p in zip(LABEL_NAMES, proba)}}


def load_artifacts(rep: str, clf: str):
    return (joblib.load(MODELS_DIR / f"vec_{rep}.joblib"),
            joblib.load(MODELS_DIR / f"{rep}__{clf}.joblib"))

# Pipeline completo: treina, tuna, salva
def train_and_save(sample_size=None, tune_top=True):
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    df, X_tr, X_te, y_tr, y_te = get_split(sample_size)
    print(f"Treino: {len(X_tr):,} | Teste: {len(X_te):,}")

    results, vecs, models = run_experiments(X_tr, X_te, y_tr, y_te)
    best = max(results, key=lambda r: r["f1_macro"])
    print(f"\nMelhor (grade padrão): {best['representation']} + {best['classifier']} "
          f"F1-macro={best['f1_macro']:.4f}")

    tuned, final = [], None
    if tune_top:  # ajusta os 4 classificadores na melhor representação, só com X_train
        rep = best["representation"]
        for clf in CLASSIFIERS:
            gs = tune(X_tr, y_tr, rep, clf)
            gs.best_estimator_.named_steps["vec"].stop_words_ = None
            joblib.dump(gs.best_estimator_, MODELS_DIR / f"tuned__{clf}.joblib", compress=3)
            row = {"representation": rep, "classifier": clf, "best_params": gs.best_params_,
                   "cv_f1_macro": gs.best_score_, **evaluate(y_te, gs.best_estimator_.predict(X_te))}
            tuned.append(row)
            print(f"Tuning {clf:<11} {gs.best_params_} CV={gs.best_score_:.4f} | teste F1-macro={row['f1_macro']:.4f}")
        # escolha do modelo final pelo CV (nunca pelo teste)
        final = max(tuned, key=lambda r: r["cv_f1_macro"])
        joblib.dump(joblib.load(MODELS_DIR / f"tuned__{final['classifier']}.joblib"),
                    MODELS_DIR / "best_pipeline.joblib", compress=3)
        print(f"Modelo final (melhor CV): {final['classifier']} {final['best_params']}")

    for rep_, vec in vecs.items():
        vec.stop_words_ = None  # atributo enorme (termos podados por min_df), inútil na inferência
        joblib.dump(vec, MODELS_DIR / f"vec_{rep_}.joblib", compress=3)
    for (rep_, name), m in models.items():
        joblib.dump(m, MODELS_DIR / f"{rep_}__{name}.joblib", compress=3)
    if final is None:
        joblib.dump(Pipeline([("vec", vecs[best["representation"]]),
                              ("clf", models[(best["representation"], best["classifier"])])]),
                    MODELS_DIR / "best_pipeline.joblib", compress=3)

    payload = {"n_train": len(X_tr), "n_test": len(X_te), "seed": SEED, "labels": LABEL_NAMES,
               "results": results, "best_default": best, "tuned": tuned, "final": final}
    (REPORTS_DIR / "model_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Artefatos salvos em models_saved/ e reports/model_results.json")
    return payload


if __name__ == "__main__":
    import sys
    train_and_save(sample_size=int(sys.argv[1]) if len(sys.argv) > 1 else None)