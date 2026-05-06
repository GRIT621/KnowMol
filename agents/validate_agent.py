from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class ValidateAgent:
    """Train downstream ML models and return metrics plus feedback bad cases."""

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
        if len(valid):
            x_valid, y_valid = extract_features(valid, substructures, fragments, self.dataset)
            train_df = pd.concat(
                [pd.concat([x_train, x_valid], axis=0), pd.concat([y_train, y_valid], axis=0)],
                axis=1,
            )
        else:
            train_df = pd.concat([x_train, y_train], axis=1)
        train_df = train_df.fillna(0)

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

        x_test, y_test = extract_features(test, substructures, fragments, self.dataset)
        y_true = y_test.values
        y_pred = predictor.predict(x_test).values
        y_prob = predictor.predict_proba(x_test).iloc[:, 1].values
        metrics = compute_binary_metrics(y_true, y_pred, y_prob)

        x_feedback, y_feedback = extract_features(train, substructures, fragments, self.dataset)
        train_prob = predictor.predict_proba(x_feedback).iloc[:, 1].values
        feedback = train.copy()
        feedback["_feedback_error"] = np.abs(y_feedback.values - train_prob)
        feedback = feedback.sort_values("_feedback_error", ascending=False).reset_index(drop=True)
        return metrics, feedback

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

        train_valid = pd.concat([train, valid], axis=0).reset_index(drop=True) if len(valid) else train
        x_train, y_train = extract_features(train_valid, substructures, fragments, self.dataset)
        x_test, y_test = extract_features(test, substructures, fragments, self.dataset)

        model = RandomForestClassifier(
            n_estimators=120,
            max_depth=12,
            min_samples_leaf=2,
            random_state=42 + max(round_index, 0),
            n_jobs=-1,
            class_weight="balanced",
        )
        model.fit(x_train, y_train)

        y_prob = model.predict_proba(x_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        metrics = compute_binary_metrics(y_test.values, y_pred, y_prob)

        x_feedback, y_feedback = extract_features(train, substructures, fragments, self.dataset)
        train_prob = model.predict_proba(x_feedback)[:, 1]
        feedback = train.copy()
        feedback["_feedback_error"] = np.abs(y_feedback.values - train_prob)
        feedback = feedback.sort_values("_feedback_error", ascending=False).reset_index(drop=True)
        return metrics, feedback
