from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import pandas as pd

try:
    from validation import (
        FEATURE_NAMES,
        KIBA_FEATURE_NAMES,
        STANDARD_AA,
        get_tabular_predictor,
        load_feature_dicts,
        load_feature_vocab,
        print_metrics,
        read_or_split_dataset,
        compute_binary_metrics,
        smiles_to_features_enhanced,
        smiles_to_features_optimized,
        substructure_features,
        protein_substructure_features,
        protein_to_features_optimized,
    )
except ImportError:
    from downstream_ml.validation import (
        FEATURE_NAMES,
        KIBA_FEATURE_NAMES,
        STANDARD_AA,
        get_tabular_predictor,
        load_feature_dicts,
        load_feature_vocab,
        print_metrics,
        read_or_split_dataset,
        compute_binary_metrics,
        smiles_to_features_enhanced,
        smiles_to_features_optimized,
        substructure_features,
        protein_substructure_features,
        protein_to_features_optimized,
    )


ABLATION_CHOICES = [
    "none",
    "no_protein_basic",
    "no_protein_fragment",
    "no_molecule_basic",
    "no_molecule_fragment",
]

ABLATION_LABELS = {
    "none": "Full features",
    "no_protein_basic": "Remove protein physicochemical/AAC features",
    "no_protein_fragment": "Remove protein structural fragment features",
    "no_molecule_basic": "Remove molecular descriptor/fingerprint features",
    "no_molecule_fragment": "Remove molecular substructure fragment features",
}


def build_feature_groups(
    data: pd.DataFrame,
    substructures: Dict[str, Any],
    fragments: Dict[str, Any],
    dataset_type: str,
) -> tuple[dict[str, pd.DataFrame], pd.Series]:
    data = data.reset_index(drop=True)
    dataset_type = dataset_type.lower()

    if dataset_type == "kiba":
        drug_basic_values = data["drug"].apply(smiles_to_features_optimized)
        drug_basic = pd.DataFrame(drug_basic_values.tolist(), columns=KIBA_FEATURE_NAMES)
    else:
        drug_basic_values = data["drug"].apply(smiles_to_features_enhanced)
        drug_basic = pd.DataFrame(drug_basic_values.tolist(), columns=FEATURE_NAMES)

    drug_fragment_values = data["drug"].apply(lambda s: substructure_features(s, substructures))
    drug_fragment = pd.DataFrame(drug_fragment_values.tolist(), columns=list(substructures.keys()))

    protein_basic_values = data["target"].apply(protein_to_features_optimized)
    prot_base_names = ["gravy", "aromaticity", "instability", "isoelectric_point", "mol_weight"]
    prot_aac_names = [f"aac_{aa}" for aa in STANDARD_AA]
    protein_basic = pd.DataFrame(protein_basic_values.tolist(), columns=prot_base_names + prot_aac_names)

    protein_fragment_values = data["target"].apply(lambda s: protein_substructure_features(s, fragments))
    protein_fragment = pd.DataFrame(
        protein_fragment_values.tolist(),
        columns=[f"frag_{i + 1}" for i in range(len(fragments))],
    )

    y = data["label"].reset_index(drop=True)
    return {
        "molecule_basic": drug_basic,
        "molecule_fragment": drug_fragment,
        "protein_basic": protein_basic,
        "protein_fragment": protein_fragment,
    }, y


def extract_ablation_features(
    data: pd.DataFrame,
    substructures: Dict[str, Any],
    fragments: Dict[str, Any],
    dataset_type: str,
    ablation: str,
) -> tuple[pd.DataFrame, pd.Series]:
    groups, y = build_feature_groups(data, substructures, fragments, dataset_type)

    selected = {
        "molecule_basic": ablation != "no_molecule_basic",
        "molecule_fragment": ablation != "no_molecule_fragment",
        "protein_basic": ablation != "no_protein_basic",
        "protein_fragment": ablation != "no_protein_fragment",
    }
    frames = [groups[name] for name, keep in selected.items() if keep]
    if not frames:
        raise ValueError("Ablation removed every feature group; at least one group must remain.")

    x = pd.concat(frames, axis=1).fillna(0)
    return x, y


def train_and_evaluate_ablation(
    train_data: pd.DataFrame,
    valid_data: pd.DataFrame,
    test_data: pd.DataFrame,
    substructures: Dict[str, Any],
    fragments: Dict[str, Any],
    args: argparse.Namespace,
    ablation: str,
) -> dict[str, float]:
    print("\n" + "=" * 72)
    print(f"Ablation: {ablation} | {ABLATION_LABELS[ablation]}")
    print("=" * 72)

    x_train, y_train = extract_ablation_features(train_data, substructures, fragments, args.dataset, ablation)
    if len(valid_data):
        x_valid, y_valid = extract_ablation_features(valid_data, substructures, fragments, args.dataset, ablation)
        train_df = pd.concat([pd.concat([x_train, x_valid], axis=0), pd.concat([y_train, y_valid], axis=0)], axis=1)
    else:
        train_df = pd.concat([x_train, y_train], axis=1)
    train_df = train_df.fillna(0)

    model_path = Path(args.output_dir) / ablation
    TabularPredictor = get_tabular_predictor()
    predictor = TabularPredictor(
        label="label",
        problem_type="binary",
        path=str(model_path),
        eval_metric=args.eval_metric,
    ).fit(
        train_data=train_df,
        time_limit=args.time_limit,
        presets=args.presets,
        num_cpus=args.num_cpus,
        excluded_model_types=["KNN"],
    )

    x_test, y_test = extract_ablation_features(test_data, substructures, fragments, args.dataset, ablation)
    y_true = y_test.values
    y_pred = predictor.predict(x_test).values
    y_prob = predictor.predict_proba(x_test).iloc[:, 1].values
    metrics = compute_binary_metrics(y_true, y_pred, y_prob)
    print_metrics(metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="KnowMol downstream ML ablation experiments")
    parser.add_argument("--dataset", choices=["davis", "drugbank", "kiba"], required=True)
    parser.add_argument("--data", required=True, help="CSV file or split directory containing train/valid/test CSVs")
    parser.add_argument("--output-dir", required=True, help="Directory for ablation validation models")
    parser.add_argument(
        "--vocab-py",
        default="/data/lsj/KnowMol/EviDTI_dataset/knowmol_36.py",
        help="Legacy combined Python/txt file defining substructure_patterns and binding_fragments",
    )
    parser.add_argument("--drug-dict", default=str(Path(__file__).with_name("drug_dict.txt")))
    parser.add_argument("--protein-dict", default=str(Path(__file__).with_name("protein_dict.txt")))
    parser.add_argument("--ablations", nargs="+", choices=ABLATION_CHOICES, default=ABLATION_CHOICES)
    parser.add_argument("--time-limit", type=int, default=600)
    parser.add_argument("--presets", default="best_quality")
    parser.add_argument("--num-cpus", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-metric", default="roc_auc")
    args = parser.parse_args()

    substructures, fragments = load_feature_dicts(args.drug_dict, args.protein_dict, args.vocab_py)
    train_data, valid_data, test_data = read_or_split_dataset(args.data, args.dataset, args.seed)
    print(f"Dataset: {args.dataset}")
    print(f"Train={len(train_data)} Valid={len(valid_data)} Test={len(test_data)}")
    print(f"Substructures={len(substructures)} Protein fragments={len(fragments)}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    rows = []
    for ablation in args.ablations:
        metrics = train_and_evaluate_ablation(
            train_data,
            valid_data,
            test_data,
            substructures,
            fragments,
            args,
            ablation,
        )
        rows.append({"ablation": ablation, "description": ABLATION_LABELS[ablation], **metrics})

    summary = pd.DataFrame(rows)
    summary_path = Path(args.output_dir) / "ablation_summary.csv"
    summary.to_csv(summary_path, index=False)
    print("\nAblation summary")
    print(summary.to_string(index=False))
    print(f"\nSaved summary: {summary_path}")


if __name__ == "__main__":
    main()
