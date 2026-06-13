#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FEATURE_NAMES = [
    "MolWt",
    "LogP",
    "TPSA",
    "HBD",
    "HBA",
    "RotatableBonds",
    "AromaticRings",
    "RingCount",
    "HeteroAtoms",
    "HydrophobicAtoms",
    "FormalCharge",
    "LabuteASA",
    "PolarSurfaceFraction",
    "AromaticDensity",
    "HBondDensity",
    "FlexibilityRatio",
    "FractionCSP3",
    "BertzComplexity",
]
KIBA_FEATURE_NAMES = ["logP", "TPSA", "MolWt"] + [f"fp_{i}" for i in range(1024)]
STANDARD_AA = list("ACDEFGHIKLMNPQRSTVWY")
PROTEIN_BASE_NAMES = ["gravy", "aromaticity", "instability", "isoelectric_point", "mol_weight"]
PROTEIN_AAC_NAMES = [f"aac_{aa}" for aa in STANDARD_AA]


def require_shap():
    try:
        import shap
    except ImportError as exc:
        raise SystemExit("Install shap to run interpretability plots: pip install shap") from exc
    return shap


def get_tabular_predictor():
    from autogluon.tabular import TabularPredictor

    return TabularPredictor


def load_predictor(model_path: str):
    return get_tabular_predictor().load(model_path)


def predictor_features(predictor: Any) -> list[str]:
    if hasattr(predictor, "feature_metadata_in") and predictor.feature_metadata_in is not None:
        return list(predictor.feature_metadata_in.get_features())
    if hasattr(predictor, "features"):
        return list(predictor.features())
    raise ValueError("Cannot infer feature order from the loaded predictor.")


def load_frame(data_path: str, dataset: str, split: str, seed: int) -> pd.DataFrame:
    from downstream_ml.validation import read_dataset, read_or_split_dataset

    path = Path(data_path).expanduser()
    if path.is_dir():
        train, valid, test = read_or_split_dataset(path, dataset, seed)
        if split == "train":
            return train
        if split == "valid":
            return valid
        if split == "test":
            return test
        return pd.concat([train, valid, test], axis=0).reset_index(drop=True)
    return read_dataset(path, dataset)


def filter_frame(frame: pd.DataFrame, mode: str, drug_id: str | None, target_id: str | None) -> pd.DataFrame:
    filtered = frame.copy()
    if mode in {"drug", "pair"}:
        if drug_id is None:
            raise ValueError("--drug-id is required for mode=drug or mode=pair.")
        if "drug_id" not in filtered.columns:
            raise ValueError("The selected data does not contain a drug_id column.")
        filtered = filtered[filtered["drug_id"].astype(str).str.strip() == str(drug_id).strip()]
    if mode in {"target", "pair"}:
        if target_id is None:
            raise ValueError("--target-id is required for mode=target or mode=pair.")
        if "target_id" not in filtered.columns:
            raise ValueError("The selected data does not contain a target_id column.")
        filtered = filtered[filtered["target_id"].astype(str).str.strip() == str(target_id).strip()]
    if filtered.empty:
        raise ValueError(f"No records matched mode={mode}, drug_id={drug_id}, target_id={target_id}.")
    return filtered.reset_index(drop=True)


def sample_frame(frame: pd.DataFrame, max_rows: int | None, seed: int) -> pd.DataFrame:
    if max_rows and len(frame) > max_rows:
        return frame.sample(n=max_rows, random_state=seed).reset_index(drop=True)
    return frame.reset_index(drop=True)


def build_feature_matrix(
    frame: pd.DataFrame,
    dataset: str,
    predictor: Any,
    drug_dict: str,
    protein_dict: str,
) -> tuple[pd.DataFrame, pd.Series]:
    from downstream_ml.validation import extract_features, load_feature_dicts

    substructures, fragments = load_feature_dicts(drug_dict, protein_dict)
    x_data, y_data = extract_features(frame, substructures, fragments, dataset)
    model_features = predictor_features(predictor)
    return x_data.reindex(columns=model_features, fill_value=0).fillna(0), y_data


def predict_positive_probability(predictor: Any, x_data: pd.DataFrame) -> np.ndarray:
    probabilities = predictor.predict_proba(x_data)
    if isinstance(probabilities, pd.DataFrame):
        if 1 in probabilities.columns:
            return probabilities[1].to_numpy()
        return probabilities.iloc[:, -1].to_numpy()
    probabilities = np.asarray(probabilities)
    return probabilities[:, -1] if probabilities.ndim == 2 else probabilities


def compute_shap_values(predictor: Any, x_data: pd.DataFrame, background_size: int, seed: int):
    shap = require_shap()
    background = sample_frame(x_data, min(background_size, len(x_data)), seed)
    masker = shap.maskers.Independent(background)

    def model_fn(values):
        values_df = pd.DataFrame(values, columns=x_data.columns)
        return predict_positive_probability(predictor, values_df)

    explainer = shap.Explainer(model_fn, masker, algorithm="permutation")
    return explainer(x_data)


def feature_category(feature: str, substructure_names: set[str]) -> str:
    if feature in set(FEATURE_NAMES) | set(KIBA_FEATURE_NAMES) | substructure_names or feature.startswith("fp_"):
        return "drug"
    if feature in set(PROTEIN_BASE_NAMES) | set(PROTEIN_AAC_NAMES) or feature.startswith("frag_"):
        return "protein"
    return "other"


def select_features(
    x_data: pd.DataFrame,
    shap_values: Any,
    view: str,
    substructure_names: set[str],
) -> tuple[pd.DataFrame, Any]:
    if view == "all":
        return x_data, shap_values
    selected = [col for col in x_data.columns if feature_category(col, substructure_names) == view]
    if not selected:
        raise ValueError(f"No {view} features found for the selected data/model.")
    indices = [x_data.columns.get_loc(col) for col in selected]
    return x_data[selected], shap_values[:, indices]


def write_importance(shap_values: Any, x_data: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    values = np.asarray(shap_values.values)
    importance = pd.DataFrame(
        {
            "feature": x_data.columns,
            "mean_abs_shap": np.abs(values).mean(axis=0),
            "mean_shap": values.mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)
    importance.to_csv(output_dir / "shap_feature_importance.csv", index=False)
    return importance


def plot_importance_bar(importance: pd.DataFrame, output_dir: Path, top_k: int) -> None:
    top = importance.head(top_k).iloc[::-1]
    plt.figure(figsize=(9, max(4, 0.34 * len(top))))
    plt.barh(top["feature"], top["mean_abs_shap"], color="#4C78A8")
    plt.xlabel("mean(|SHAP value|)")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance_bar.png", dpi=300)
    plt.close()


def plot_summary(shap_values: Any, x_data: pd.DataFrame, output_dir: Path, top_k: int, name: str) -> None:
    shap = require_shap()
    plt.figure()
    shap.summary_plot(shap_values, x_data, max_display=top_k, show=False)
    plt.tight_layout()
    plt.savefig(output_dir / f"{name}_summary.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_waterfall(shap_values: Any, output_dir: Path, top_k: int, name: str, row_index: int = 0) -> None:
    shap = require_shap()
    row_index = min(row_index, len(shap_values) - 1)
    plt.figure()
    shap.plots.waterfall(shap_values[row_index], max_display=top_k, show=False)
    plt.tight_layout()
    plt.savefig(output_dir / f"{name}_waterfall.png", dpi=300, bbox_inches="tight")
    plt.close()


def default_summary_view(mode: str, requested: str) -> str:
    if requested != "auto":
        return requested
    if mode == "target":
        return "drug"
    if mode == "drug":
        return "protein"
    return "all"


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    predictor = load_predictor(args.model_path)
    frame = load_frame(args.data, args.dataset, args.split, args.seed)
    frame = filter_frame(frame, args.mode, args.drug_id, args.target_id)
    frame = sample_frame(frame, args.max_rows, args.seed)
    frame.to_csv(output_dir / "selected_records.csv", index=False)

    x_data, y_data = build_feature_matrix(
        frame,
        args.dataset,
        predictor,
        args.drug_dict,
        args.protein_dict,
    )
    pd.concat([x_data, y_data.rename("label")], axis=1).to_csv(output_dir / "selected_features.csv", index=False)

    shap_values = compute_shap_values(predictor, x_data, args.background_size, args.seed)
    importance = write_importance(shap_values, x_data, output_dir)
    plot_importance_bar(importance, output_dir, args.top_k)

    from downstream_ml.validation import load_feature_dicts

    substructures, _ = load_feature_dicts(args.drug_dict, args.protein_dict)
    view = default_summary_view(args.mode, args.summary_view)
    x_summary, shap_summary = select_features(x_data, shap_values, view, set(substructures))
    plot_summary(shap_summary, x_summary, output_dir, args.top_k, f"{args.mode}_{view}")

    if args.mode in {"target", "drug", "pair"}:
        plot_waterfall(shap_values, output_dir, args.top_k, args.mode, args.row_index)

    metadata = {
        "mode": args.mode,
        "dataset": args.dataset,
        "data": args.data,
        "model_path": args.model_path,
        "drug_id": args.drug_id,
        "target_id": args.target_id,
        "rows": len(frame),
        "features": x_data.shape[1],
        "summary_view": view,
    }
    pd.Series(metadata).to_json(output_dir / "metadata.json", indent=2)
    print(f"Saved interpretability outputs to {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Draw KnowMol SHAP interpretability plots.")
    parser.add_argument("--mode", choices=["global", "target", "drug", "pair"], required=True)
    parser.add_argument("--dataset", required=True, help="Dataset name accepted by downstream_ml.validation")
    parser.add_argument("--data", required=True, help="CSV file or split directory")
    parser.add_argument("--model-path", required=True, help="Trained AutoGluon TabularPredictor path")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--drug-id")
    parser.add_argument("--target-id")
    parser.add_argument("--split", choices=["all", "train", "valid", "test"], default="all")
    parser.add_argument("--summary-view", choices=["auto", "all", "drug", "protein"], default="auto")
    parser.add_argument("--drug-dict", default=str(ROOT / "downstream_ml" / "drug_dict.txt"))
    parser.add_argument("--protein-dict", default=str(ROOT / "downstream_ml" / "protein_dict.txt"))
    parser.add_argument("--max-rows", type=int, default=200)
    parser.add_argument("--background-size", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
