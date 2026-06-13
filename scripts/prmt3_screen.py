#!/usr/bin/env python
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from downstream_ml.validation import (
    compute_binary_metrics,
    extract_features,
    load_feature_dicts,
    print_metrics,
)


DEFAULT_PRMT3_SEQ = (
    "MCSLASGATGGRGAVENEEDLPELSDSGDEAAWEDEDDADLPHGKQQTPCLFCNRLFTSA"
    "EETFSHCKSEHQFNIDSMVHKHGLEFYGYIKLINFIRLKNPTVEYMNSIYNPVPWEKEEY"
    "LKPVLEDDLLLQFDVEDLYEPVSVPFSYPNGLSENTSVVEKLKHMEARALSAEAALARAR"
    "EDLQKMKQFAQDFVMHTDVRTCSSSTSVIADLQEDEDGVYFSSYGHYGIHEEMLKDKIRT"
    "ESYRDFIYQNPHIFKDKVVLDVGCGTGILSMFAAKAGAKKVLGVDQSEILYQAMDIIRLN"
    "KLEDTITLIKGKIEEVHLPVEKVDVIISEWMGYFLLFESMLDSVLYAKNKYLAKGGSVYP"
    "DICTISLVAVSDVNKHADRIAFWDDVYGFKMSCMKKAVIPEAVVEVLDPKTLISEPCGIK"
    "HIDCHTTSISDLEFSSDFTLKITRTSMCTAIAGYFDIYFEKNCHNRVVFSTGPQSTKTHW"
    "KQTVFLLEKPFSVKAGEALKGKVTVHKNKKDPRSLTVTLTLNNSTQTYGLQ"
)


def normalize_prmt3_frame(frame: pd.DataFrame, default_target: str | None = None) -> pd.DataFrame:
    frame = frame.copy()
    rename_map = {
        "smile": "drug",
        "smiles": "drug",
        "SMILES": "drug",
        "sequence": "target",
        "seq": "target",
    }
    frame = frame.rename(columns={k: v for k, v in rename_map.items() if k in frame.columns})

    if "drug" not in frame.columns:
        raise ValueError("PRMT3 data must contain a drug/smiles column.")
    if "target" not in frame.columns:
        if not default_target:
            raise ValueError("PRMT3 data must contain target/seq, or provide --protein-seq/--protein-seq-file.")
        frame["target"] = default_target
    if "label" not in frame.columns:
        frame["label"] = 0

    return frame.dropna(subset=["drug", "target"]).reset_index(drop=True)


def load_protein_sequence(args: argparse.Namespace) -> str | None:
    if args.protein_seq:
        return args.protein_seq.replace("\n", "").strip()
    if args.protein_seq_file:
        return Path(args.protein_seq_file).read_text().replace("\n", "").strip()
    return DEFAULT_PRMT3_SEQ if args.use_default_prmt3_seq else None


def build_classifier(args: argparse.Namespace) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        class_weight=args.class_weight,
        random_state=args.seed,
        n_jobs=args.num_cpus,
    )


def evaluate_holdout(
    train_data: pd.DataFrame,
    substructures: dict[str, Any],
    fragments: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, float]:
    labels = train_data["label"].astype(int)
    stratify = labels if labels.nunique() == 2 and labels.value_counts().min() >= 2 else None
    train_split, test_split = train_test_split(
        train_data,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=stratify,
    )

    x_train, y_train = extract_features(train_split, substructures, fragments, dataset_type="prmt3")
    x_test, y_test = extract_features(test_split, substructures, fragments, dataset_type="prmt3")

    clf = build_classifier(args)
    clf.fit(x_train, y_train.astype(int))

    y_prob = clf.predict_proba(x_test)[:, 1]
    y_pred = (y_prob >= args.threshold).astype(int)
    metrics = compute_binary_metrics(y_test.astype(int).values, y_pred, y_prob)
    print_metrics(metrics)
    return metrics


def train_full_model(
    train_data: pd.DataFrame,
    substructures: dict[str, Any],
    fragments: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[RandomForestClassifier, list[str]]:
    x_train, y_train = extract_features(train_data, substructures, fragments, dataset_type="prmt3")
    clf = build_classifier(args)
    clf.fit(x_train, y_train.astype(int))
    return clf, list(x_train.columns)


def score_ligands(
    clf: RandomForestClassifier,
    feature_columns: list[str],
    ligands: pd.DataFrame,
    substructures: dict[str, Any],
    fragments: dict[str, Any],
    args: argparse.Namespace,
) -> pd.DataFrame:
    x_score, _ = extract_features(ligands, substructures, fragments, dataset_type="prmt3")
    x_score = x_score.reindex(columns=feature_columns, fill_value=0)
    probabilities = clf.predict_proba(x_score)[:, 1]
    predictions = (probabilities >= args.threshold).astype(int)

    result = ligands.copy()
    result["pred_label"] = predictions
    result["binding_score"] = probabilities
    if "label" in result.columns:
        result["label"] = result["label"].astype(int, errors="ignore")
    return result.sort_values("binding_score", ascending=False).reset_index(drop=True)


def save_model(path: str | None, clf: RandomForestClassifier, feature_columns: list[str]) -> None:
    if not path:
        return
    model_path = Path(path).expanduser()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as handle:
        pickle.dump({"model": clf, "feature_columns": feature_columns}, handle)
    print(f"Saved PRMT3 model: {model_path}")


def load_model(path: str) -> tuple[RandomForestClassifier, list[str]]:
    with Path(path).expanduser().open("rb") as handle:
        payload = pickle.load(handle)
    return payload["model"], payload["feature_columns"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and score the PRMT3 case study with KnowMol feature extraction."
    )
    parser.add_argument("--train-csv", help="PRMT3 training CSV with drug,target,label columns")
    parser.add_argument("--ligands-csv", help="Candidate ligand CSV to score")
    parser.add_argument("--output", help="Output CSV for ranked PRMT3 scores")
    parser.add_argument("--model-path", help="Pickle path for saving/loading the PRMT3 RandomForest model")
    parser.add_argument("--load-model", action="store_true", help="Load --model-path instead of training a new model")
    parser.add_argument("--drug-dict", default=str(ROOT / "downstream_ml" / "drug_dict.txt"))
    parser.add_argument("--protein-dict", default=str(ROOT / "downstream_ml" / "protein_dict.txt"))
    parser.add_argument("--protein-seq")
    parser.add_argument("--protein-seq-file")
    parser.add_argument("--use-default-prmt3-seq", action="store_true", help="Use the PRMT3 sequence from the original case study")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    parser.add_argument("--class-weight", default="balanced")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-cpus", type=int, default=-1)
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--top-k", type=int, default=20)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    default_target = load_protein_sequence(args)
    substructures, fragments = load_feature_dicts(args.drug_dict, args.protein_dict)

    if args.load_model:
        if not args.model_path:
            raise ValueError("--model-path is required with --load-model.")
        clf, feature_columns = load_model(args.model_path)
        print(f"Loaded PRMT3 model: {args.model_path}")
    else:
        if not args.train_csv:
            raise ValueError("--train-csv is required unless --load-model is set.")
        train_data = normalize_prmt3_frame(pd.read_csv(args.train_csv), default_target)
        print(f"PRMT3 train rows: {len(train_data)}")
        print(f"Train label counts: {train_data['label'].value_counts().to_dict()}")
        print(f"KnowMol features: drug={len(substructures)} protein={len(fragments)}")
        if not args.skip_eval and train_data["label"].nunique() == 2:
            evaluate_holdout(train_data, substructures, fragments, args)
        clf, feature_columns = train_full_model(train_data, substructures, fragments, args)
        save_model(args.model_path, clf, feature_columns)

    if args.ligands_csv:
        ligands = normalize_prmt3_frame(pd.read_csv(args.ligands_csv), default_target)
        scores = score_ligands(clf, feature_columns, ligands, substructures, fragments, args)
        display_columns = [
            column
            for column in ["drug_id", "target_id", "drug", "label", "pred_label", "binding_score", "relative_similarity"]
            if column in scores.columns
        ]
        print("\nTop PRMT3 candidates")
        print(scores.head(args.top_k)[display_columns].to_string(index=False))
        if args.output:
            output = Path(args.output).expanduser()
            output.parent.mkdir(parents=True, exist_ok=True)
            scores.to_csv(output, index=False)
            print(f"\nSaved PRMT3 scores: {output}")


if __name__ == "__main__":
    main()
