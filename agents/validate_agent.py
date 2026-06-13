from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class ValidateAgent:
    """Train downstream ML models and return validation feedback.

    During feature discovery, candidate substructures/fragments are judged on
    the validation split. The held-out test split is reported for monitoring
    only and is not used for memory consolidation or badcase selection.
    """

    def __init__(
        self,
        dataset: str,
        output_dir: str | Path,
        time_limit: int = 600,
        presets: str = "best_quality",
        num_cpus: int = 8,
        eval_metric: str = "roc_auc",
        backend: str = "tabular",
    ) -> None:
        self.dataset = dataset
        self.output_dir = Path(output_dir)
        self.time_limit = time_limit
        self.presets = presets
        self.num_cpus = num_cpus
        self.eval_metric = eval_metric
        self.backend = backend

    def evaluate_round(
        self,
        train: pd.DataFrame,
        valid: pd.DataFrame,
        test: pd.DataFrame,
        substructures: dict[str, Any],
        fragments: dict[str, Any],
        round_index: int,
    ) -> tuple[dict[str, float], pd.DataFrame]:
        if self.backend == "sklearn":
            return self._evaluate_round_sklearn(train, valid, test, substructures, fragments, round_index)
        if self.backend == "tabular":
            return self._evaluate_round_tabular(train, valid, test, substructures, fragments, round_index)
        raise ValueError(f"Unsupported validation backend: {self.backend}")

    @staticmethod
    def _pack_feedback(frame: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> pd.DataFrame:
        feedback = frame.copy()
        feedback["pred_label"] = y_pred
        feedback["pred_prob"] = y_prob
        feedback["_feedback_error"] = np.abs(y_true - y_prob)
        return feedback.sort_values("_feedback_error", ascending=False).reset_index(drop=True)

    @staticmethod
    def _merge_test_metrics(valid_metrics: dict[str, float], test_metrics: dict[str, float]) -> dict[str, float]:
        merged = dict(valid_metrics)
        merged.update({f"test_{key}": value for key, value in test_metrics.items()})
        return merged

    def _evaluate_round_tabular(
        self,
        train: pd.DataFrame,
        valid: pd.DataFrame,
        test: pd.DataFrame,
        substructures: dict[str, Any],
        fragments: dict[str, Any],
        round_index: int,
    ) -> tuple[dict[str, float], pd.DataFrame]:
        from downstream_ml.validation import compute_binary_metrics, extract_features, get_tabular_predictor

        x_train, y_train = extract_features(train, substructures, fragments, self.dataset)
        train_df = pd.concat([x_train, y_train], axis=1).fillna(0)

        model_path = self.output_dir / "models" / f"round_{round_index + 1:03d}"
        TabularPredictor = get_tabular_predictor()
        predictor = TabularPredictor(
            label="label",
            problem_type="binary",
            path=str(model_path),
            eval_metric=self.eval_metric,
        ).fit(
            train_data=train_df,
            time_limit=self.time_limit,
            presets=self.presets,
            num_cpus=self.num_cpus,
            excluded_model_types=["KNN"],
        )

        eval_frame = valid if len(valid) else test
        x_eval, y_eval = extract_features(eval_frame, substructures, fragments, self.dataset)
        y_eval_true = y_eval.values
        y_eval_pred = predictor.predict(x_eval).values
        y_eval_prob = predictor.predict_proba(x_eval).iloc[:, 1].values
        valid_metrics = compute_binary_metrics(y_eval_true, y_eval_pred, y_eval_prob)

        x_test, y_test = extract_features(test, substructures, fragments, self.dataset)
        y_test_true = y_test.values
        y_test_pred = predictor.predict(x_test).values
        y_test_prob = predictor.predict_proba(x_test).iloc[:, 1].values
        test_metrics = compute_binary_metrics(y_test_true, y_test_pred, y_test_prob)

        feedback = self._pack_feedback(eval_frame, y_eval_true, y_eval_pred, y_eval_prob)
        return self._merge_test_metrics(valid_metrics, test_metrics), feedback

    def _evaluate_round_sklearn(
        self,
        train: pd.DataFrame,
        valid: pd.DataFrame,
        test: pd.DataFrame,
        substructures: dict[str, Any],
        fragments: dict[str, Any],
        round_index: int,
    ) -> tuple[dict[str, float], pd.DataFrame]:
        from downstream_ml.validation import compute_binary_metrics, extract_features
        from sklearn.ensemble import RandomForestClassifier

        x_train, y_train = extract_features(train, substructures, fragments, self.dataset)
        eval_frame = valid if len(valid) else test
        x_eval, y_eval = extract_features(eval_frame, substructures, fragments, self.dataset)

        model = RandomForestClassifier(
            n_estimators=120,
            max_depth=12,
            min_samples_leaf=2,
            random_state=42 + max(round_index, 0),
            n_jobs=-1,
            class_weight="balanced",
        )
        model.fit(x_train, y_train)

        y_eval_prob = model.predict_proba(x_eval)[:, 1]
        y_eval_pred = (y_eval_prob >= 0.5).astype(int)
        valid_metrics = compute_binary_metrics(y_eval.values, y_eval_pred, y_eval_prob)

        x_test, y_test = extract_features(test, substructures, fragments, self.dataset)
        y_test_prob = model.predict_proba(x_test)[:, 1]
        y_test_pred = (y_test_prob >= 0.5).astype(int)
        test_metrics = compute_binary_metrics(y_test.values, y_test_pred, y_test_prob)

        feedback = self._pack_feedback(eval_frame, y_eval.values, y_eval_pred, y_eval_prob)
        return self._merge_test_metrics(valid_metrics, test_metrics), feedback
