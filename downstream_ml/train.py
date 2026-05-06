from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from downstream_ml.validation import (
    compute_binary_metrics,
    get_tabular_predictor,
    print_metrics,
    read_or_split_dataset,
)


RAW_COLUMNS = ["drug", "target"]


def extract_raw_features(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    missing = [column for column in RAW_COLUMNS + ["label"] if column not in data.columns]
    if missing:
        raise ValueError(f"Raw baseline data is missing required columns: {missing}")

    x = data[RAW_COLUMNS].copy()
    x["drug"] = x["drug"].astype(str)
    x["target"] = x["target"].astype(str)
    y = data["label"].reset_index(drop=True)
    return x.reset_index(drop=True), y


def build_train_frame(train: pd.DataFrame, valid: pd.DataFrame) -> pd.DataFrame:
    x_train, y_train = extract_raw_features(train)
    if len(valid):
        x_valid, y_valid = extract_raw_features(valid)
        x = pd.concat([x_train, x_valid], axis=0).reset_index(drop=True)
        y = pd.concat([y_train, y_valid], axis=0).reset_index(drop=True)
    else:
        x = x_train
        y = y_train

    train_df = x.copy()
    train_df["label"] = y
    return train_df


def train_mode(args: argparse.Namespace) -> None:
    train, valid, test = read_or_split_dataset(args.data, args.dataset, args.seed)

    print("Baseline: Raw sequence baseline")
    print("Features: drug, target")
    print(f"Dataset: {args.dataset}")
    print(f"Train={len(train)} Valid={len(valid)} Test={len(test)}")

    train_df = build_train_frame(train, valid)

    TabularPredictor = get_tabular_predictor()
    predictor = TabularPredictor(
        label="label",
        problem_type="binary",
        path=args.model_path,
        eval_metric=args.eval_metric,
    ).fit(
        train_data=train_df,
        time_limit=args.time_limit,
        presets=args.presets,
        num_cpus=args.num_cpus,
        excluded_model_types=["KNN"],
    )

    print(f"Saved model: {predictor.path}")
    if args.evaluate_after_train:
        evaluate_predictor(predictor, test, args.leaderboard)


def evaluate_predictor(predictor: Any, test_data: pd.DataFrame, show_leaderboard: bool = False) -> dict[str, float]:
    x_test, y_test = extract_raw_features(test_data)
    y_true = y_test.values
    y_pred = predictor.predict(x_test).values
    y_prob = predictor.predict_proba(x_test).iloc[:, 1].values

    metrics = compute_binary_metrics(y_true, y_pred, y_prob)
    print_metrics(metrics)

    test_with_scores = test_data.copy()
    test_with_scores["pred_label"] = y_pred
    test_with_scores["pred_prob"] = y_prob
    test_with_scores["error_margin"] = np.abs(y_true - y_prob)
    print("\nTop 5 bad cases")
    print(test_with_scores.nlargest(5, "error_margin")[["drug", "target", "label", "pred_prob", "error_margin"]])

    if show_leaderboard:
        leaderboard_df = x_test.copy()
        leaderboard_df["label"] = y_test
        print("\nModel leaderboard on the selected test set")
        print(predictor.leaderboard(leaderboard_df, silent=True))

    return metrics


def test_mode(args: argparse.Namespace) -> None:
    _, _, test = read_or_split_dataset(args.data, args.dataset, args.seed)

    TabularPredictor = get_tabular_predictor()
    predictor = TabularPredictor.load(args.model_path)
    print(f"Loaded model: {args.model_path}")
    print("Baseline: Raw sequence baseline")
    print("Features: drug, target")
    print(f"Dataset: {args.dataset} | Test={len(test)}")
    evaluate_predictor(predictor, test, args.leaderboard)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Raw sequence baseline using only drug SMILES and target protein sequences"
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    train = subparsers.add_parser("train", help="Train raw baseline on drug,target only")
    train.add_argument("--dataset", choices=["davis", "drugbank", "kiba"], required=True)
    train.add_argument("--data", required=True, help="CSV file or split directory containing train/valid/test CSVs")
    train.add_argument("--model-path", required=True)
    train.add_argument("--time-limit", type=int, default=600)
    train.add_argument("--presets", default="best_quality")
    train.add_argument("--num-cpus", type=int, default=8)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--eval-metric", default="roc_auc")
    train.add_argument("--evaluate-after-train", action="store_true")
    train.add_argument("--leaderboard", action="store_true")
    train.set_defaults(func=train_mode)

    test = subparsers.add_parser("test", help="Evaluate a saved raw baseline model")
    test.add_argument("--dataset", choices=["davis", "drugbank", "kiba"], required=True)
    test.add_argument("--data", required=True, help="CSV file or split directory containing train/valid/test CSVs")
    test.add_argument("--model-path", required=True)
    test.add_argument("--seed", type=int, default=42)
    test.add_argument("--leaderboard", action="store_true")
    test.set_defaults(func=test_mode)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
