"""
Testes integrados do pipeline completo (Pessoa 1 + Pessoa 2 + Pessoa 3).

Execução (na raiz do projeto):
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import (
    BINARY_SENTIMENT_MAP,
    ID_TO_SENTIMENT,
    PERSONAL_DATA_COLUMNS,
    PROJECT_ROOT,
    clean_and_label_dataset,
    remove_personal_data,
)
from src.evaluation import (
    EVALUATION_SUMMARY_PATH,
    FIGURES_DIR,
    MODEL_RESULTS_PATH,
    build_comparison_dataframe,
    compute_detailed_metrics,
    explain_prediction,
    load_best_pipeline,
    load_model_results,
    load_tuned_pipeline,
    predict_single_review,
)
from src.models import MODELS_DIR, predict_with_confidence
from src.preprocessing import (
    get_protected_terms_removed_from_nltk,
    preprocess_text,
)


class TestDataLoaderAndPrivacy(unittest.TestCase):
    """Valida limpeza, rotulagem e remoção obrigatória de dados pessoais (Seção 2.8)."""

    def _make_sample_raw_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "submission_date": "2018-02-01 10:00:00",
                    "reviewer_id": "secret_id_1",
                    "reviewer_birth_year": 1990,
                    "reviewer_gender": "F",
                    "reviewer_state": "PB",
                    "site_category_lv1": "Celulares e Smartphones",
                    "site_category_lv2": "Smartphones",
                    "review_title": "Excelente",
                    "review_text": "Produto ótimo, chegou antes do prazo!",
                    "overall_rating": 5,
                    "recommend_to_a_friend": "Yes",
                },
                {
                    "submission_date": "2018-02-02 11:00:00",
                    "reviewer_id": "secret_id_2",
                    "reviewer_birth_year": 1985,
                    "reviewer_gender": "M",
                    "reviewer_state": "SP",
                    "site_category_lv1": "Móveis",
                    "site_category_lv2": "Sofás",
                    "review_title": "Ruim",
                    "review_text": "Não gostei, veio com defeito e não recomendo.",
                    "overall_rating": 1,
                    "recommend_to_a_friend": "No",
                },
                {
                    "submission_date": "2018-02-03 12:00:00",
                    "reviewer_id": "secret_id_3",
                    "reviewer_birth_year": 1995,
                    "reviewer_gender": "F",
                    "reviewer_state": "RJ",
                    "site_category_lv1": "Eletroportáteis",
                    "site_category_lv2": "Cafeteiras",
                    "review_title": "Mais ou menos",
                    "review_text": "Produto bom, mas a entrega atrasou bastante.",
                    "overall_rating": 3,
                    "recommend_to_a_friend": "Yes",
                },
                # Duplicata exata da linha 1 (deve ser removida)
                {
                    "submission_date": "2018-02-04 12:00:00",
                    "reviewer_id": "secret_id_4",
                    "reviewer_birth_year": 1991,
                    "reviewer_gender": "M",
                    "reviewer_state": "MG",
                    "site_category_lv1": "Celulares e Smartphones",
                    "site_category_lv2": "Smartphones",
                    "review_title": "Excelente",
                    "review_text": "Produto ótimo, chegou antes do prazo!",
                    "overall_rating": 5,
                    "recommend_to_a_friend": "Yes",
                },
                # Texto nulo (deve ser removido)
                {
                    "submission_date": "2018-02-05 12:00:00",
                    "reviewer_id": "secret_id_5",
                    "reviewer_birth_year": 1980,
                    "reviewer_gender": "F",
                    "reviewer_state": "BA",
                    "site_category_lv1": "Móveis",
                    "site_category_lv2": "Mesas",
                    "review_title": "",
                    "review_text": None,
                    "overall_rating": 2,
                    "recommend_to_a_friend": "No",
                },
            ]
        )

    def test_personal_columns_removed(self) -> None:
        df_raw = self._make_sample_raw_df()
        df_safe = remove_personal_data(df_raw)
        for col in PERSONAL_DATA_COLUMNS:
            self.assertNotIn(col, df_safe.columns)

    def test_clean_and_label_default_and_binary(self) -> None:
        df_raw = self._make_sample_raw_df()
        df_clean, audit = clean_and_label_dataset(df_raw)
        self.assertEqual(len(df_clean), 3)
        self.assertIn("review_text_original", df_clean.columns)
        self.assertEqual(set(df_clean["sentiment"]), {"Negativo", "Neutro", "Positivo"})
        self.assertEqual(audit["cleaning_steps"]["removed_duplicates"], 1)

        df_bin, _ = clean_and_label_dataset(df_raw, rating_mapping=BINARY_SENTIMENT_MAP)
        self.assertEqual(len(df_bin), 2)
        self.assertNotIn("Neutro", set(df_bin["sentiment"]))


class TestPreprocessingPLN(unittest.TestCase):
    """Valida preservação de acentos, negações e negation marking (Seções 2.4.3 e 2.5.2)."""

    def test_preserves_accents_and_negations(self) -> None:
        raw = "O produto NÃO funciona, é péssimo e nunca mais recomendo! https://teste.com <b>fim</b>"
        clean = preprocess_text(raw)
        for token in ["não", "péssimo", "nunca", "mais", "recomendo"]:
            self.assertIn(token, clean.split())
        self.assertNotIn("http", clean)
        self.assertNotIn("<b>", clean)

    def test_protected_terms_removed_from_nltk_stopwords(self) -> None:
        protected = get_protected_terms_removed_from_nltk()
        for critical in ["não", "nem", "sem", "muito", "mas", "mais"]:
            self.assertIn(critical, protected)

        no_prot = preprocess_text("Não recomendo", keep_negations=False, keep_modifiers=False)
        with_prot = preprocess_text("Não recomendo", keep_negations=True, keep_modifiers=True)
        self.assertNotIn("não", no_prot.split())
        self.assertIn("não", with_prot.split())

    def test_negation_marking(self) -> None:
        marked = preprocess_text("Não gostei e nunca mais compro", mark_negation=True)
        self.assertIn("não_gostei", marked.split())
        self.assertIn("nunca_mais", marked.split())


class TestModelsAndEvaluationIntegration(unittest.TestCase):
    """Valida modelos salvos, inferência ao vivo, explicabilidade e artefatos finais."""

    def test_saved_artifacts_and_model_results_json(self) -> None:
        self.assertTrue((MODELS_DIR / "best_pipeline.joblib").exists())
        for clf in ["nb", "logreg", "logreg_bal", "svm_bal"]:
            self.assertTrue((MODELS_DIR / f"tuned__{clf}.joblib").exists())

        mr = load_model_results(MODEL_RESULTS_PATH)
        self.assertEqual(len(mr["results"]), 16)
        self.assertEqual(len(mr["tuned"]), 4)
        self.assertEqual(mr["final"]["representation"], "tfidf_bi")
        self.assertEqual(mr["final"]["classifier"], "svm_bal")
        self.assertGreater(mr["final"]["f1_macro"], 0.72)

        df_comp = build_comparison_dataframe(mr)
        self.assertEqual(len(df_comp), 20)
        self.assertTrue(df_comp.iloc[0]["is_final"])

    def test_live_inference_and_explainability_all_model_families(self) -> None:
        pos_text = "Produto excelente, super recomendo, ótimo acabamento e chegou antes do prazo!"
        neg_text = "Péssimo produto, não funciona direito, veio quebrado e não recomendo para ninguém."

        best_pipe = load_best_pipeline()
        res_pos = predict_single_review(pos_text, best_pipe, top_k=5)
        res_neg = predict_single_review(neg_text, best_pipe, top_k=5)

        self.assertEqual(res_pos["sentimento"], "Positivo")
        self.assertEqual(res_neg["sentimento"], "Negativo")
        self.assertAlmostEqual(sum(res_pos["probabilidades"].values()), 1.0, places=5)
        self.assertGreater(len(res_neg["top_tokens_predicted"]), 0)

        # Garantir explicabilidade em todas as famílias (LinearSVC, LogReg, MultinomialNB)
        for clf_key in ["svm_bal", "logreg_bal", "logreg", "nb"]:
            pipe = load_tuned_pipeline(clf_key)
            vec, clf = pipe.named_steps["vec"], pipe.named_steps["clf"]
            cleaned = preprocess_text(neg_text)
            pred_basic = predict_with_confidence(vec, clf, cleaned)
            self.assertIn(pred_basic["sentimento"], set(ID_TO_SENTIMENT.values()))

            exp = explain_prediction(vec, clf, cleaned, top_k=5)
            self.assertEqual(exp["sentimento"], pred_basic["sentimento"])
            self.assertGreater(exp["tokens_matched"], 0)
            self.assertGreater(len(exp["top_tokens_predicted"]), 0)

    def test_detailed_metrics_calculation(self) -> None:
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 1, 1, 2, 2, 2])
        m = compute_detailed_metrics(y_true, y_pred)
        self.assertAlmostEqual(m["accuracy"], 4 / 6, places=6)
        self.assertEqual(len(m["confusion_matrix"]), 3)
        self.assertEqual(m["support_Neutro"], 2)

    def test_evaluation_summary_and_figures_exist_without_personal_data(self) -> None:
        if not EVALUATION_SUMMARY_PATH.exists():
            self.skipTest("Execute `python -m src.evaluation` antes de validar evaluation_summary.json.")
        data = json.loads(EVALUATION_SUMMARY_PATH.read_text(encoding="utf-8"))
        raw_json_str = json.dumps(data, ensure_ascii=False)
        for pcol in PERSONAL_DATA_COLUMNS:
            self.assertNotIn(pcol, raw_json_str)

        self.assertGreaterEqual(len(data["category_performance"]), 5)
        self.assertIn("Móveis", [r["categoria"] for r in data["category_performance"]])
        self.assertIn("binary_scenario_comparison", data)

        for fig_name in [
            "12_matrizes_confusao_comparacao.png",
            "13_desempenho_por_categoria.png",
            "14_analise_erros_e_ruido.png",
            "15_recall_negativo_comparacao.png",
        ]:
            self.assertTrue((FIGURES_DIR / fig_name).exists(), f"Figura ausente: {fig_name}")

    def test_streamlit_app_syntax_valid(self) -> None:
        app_path = PROJECT_ROOT / "app.py"
        self.assertTrue(app_path.exists())
        source = app_path.read_text(encoding="utf-8")
        ast.parse(source)


if __name__ == "__main__":
    unittest.main()
