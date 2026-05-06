#!/usr/bin/env python
from __future__ import annotations

import argparse
import datetime as dt
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import RDLogger

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents import FeatureAggregateAgent, MolecularAgent, ProteinAgent

RDLogger.DisableLog("rdApp.*")


@dataclass
class DatasetContext:
    dataset_name: str


def choose_samples(
    train: pd.DataFrame,
    badcases: pd.DataFrame | None,
    round_index: int,
    feedback_strategy: str,
    sample_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    if round_index > 0 and feedback_strategy == "badcase" and badcases is not None and len(badcases):
        return badcases.head(sample_size)[["drug", "target", "label", "_feedback_error"]].to_dict("records")

    sampled = train.sample(n=min(sample_size, len(train)), random_state=seed + round_index)
    return sampled[["drug", "target", "label"]].to_dict("records")


def evaluate_vocab(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    feature_aggregate: FeatureAggregateAgent,
    args: argparse.Namespace,
    round_index: int,
) -> tuple[dict[str, float], pd.DataFrame]:
    from downstream_ml.validation import compute_binary_metrics, extract_features, get_tabular_predictor

    x_train, y_train = extract_features(train, feature_aggregate.substructures, feature_aggregate.fragments, args.dataset)
    if len(valid):
        x_valid, y_valid = extract_features(
            valid,
            feature_aggregate.substructures,
            feature_aggregate.fragments,
            args.dataset,
        )
        train_df = pd.concat(
            [pd.concat([x_train, x_valid], axis=0), pd.concat([y_train, y_valid], axis=0)],
            axis=1,
        )
    else:
        train_df = pd.concat([x_train, y_train], axis=1)
    train_df = train_df.fillna(0)

    model_path = Path(args.output_dir) / "models" / f"round_{round_index + 1:03d}"
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

    x_test, y_test = extract_features(test, feature_aggregate.substructures, feature_aggregate.fragments, args.dataset)
    y_true = y_test.values
    y_pred = predictor.predict(x_test).values
    y_prob = predictor.predict_proba(x_test).iloc[:, 1].values
    metrics = compute_binary_metrics(y_true, y_pred, y_prob)

    x_feedback, y_feedback = extract_features(
        train,
        feature_aggregate.substructures,
        feature_aggregate.fragments,
        args.dataset,
    )
    train_prob = predictor.predict_proba(x_feedback).iloc[:, 1].values
    feedback = train.copy()
    feedback["_feedback_error"] = np.abs(y_feedback.values - train_prob)
    feedback = feedback.sort_values("_feedback_error", ascending=False).reset_index(drop=True)
    return metrics, feedback


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Iterative KnowMol feature discovery for drug-target interaction datasets."
    )
    parser.add_argument("--dataset", choices=["davis", "drugbank", "kiba"], required=True)
    parser.add_argument("--data", required=True, help="CSV file or split directory containing train/valid/test CSVs")
    parser.add_argument("--output-dir", default="outputs/knowmol_discovery")
    parser.add_argument("--existing-vocab", help="Optional vocab file to resume from")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--feedback-strategy", choices=["random", "badcase"], default="badcase")
    parser.add_argument("--mode", choices=["both", "molecule", "protein"], default="both")
    parser.add_argument("--model", default="pro-deepseek-r1")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--api-base", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--time-limit", type=int, default=600)
    parser.add_argument("--presets", default="best_quality")
    parser.add_argument("--num-cpus", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-metric", default="roc_auc")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    from downstream_ml.validation import load_feature_vocab, print_metrics, read_or_split_dataset

    random.seed(args.seed)
    np.random.seed(args.seed)

    train, valid, test = read_or_split_dataset(args.data, args.dataset, args.seed)
    initial_substructures: dict[str, Any] = {}
    initial_fragments: dict[str, Any] = {}
    if args.existing_vocab:
        initial_substructures, initial_fragments = load_feature_vocab(args.existing_vocab)

    feature_aggregate = FeatureAggregateAgent(initial_substructures, initial_fragments)
    context = DatasetContext(dataset_name=args.dataset)
    molecular_agent = MolecularAgent("MolecularAgent", context, args.model, args.api_key, args.api_base)
    protein_agent = ProteinAgent("ProteinAgent", context, args.model, args.api_key, args.api_base)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    badcases: pd.DataFrame | None = None
    rows: list[dict[str, Any]] = []

    print(f"Dataset: {args.dataset} | Train={len(train)} Valid={len(valid)} Test={len(test)}")
    print(
        "Initial drug features="
        f"{len(feature_aggregate.substructures)} target features={len(feature_aggregate.fragments)}"
    )

    for round_index in range(args.rounds):
        print("\n" + "=" * 72)
        print(f"KnowMol discovery round {round_index + 1}/{args.rounds}")
        print("=" * 72)
        samples = choose_samples(train, badcases, round_index, args.feedback_strategy, args.sample_size, args.seed)

        added_molecule = 0
        added_protein = 0
        if args.mode in {"both", "molecule"}:
            molecule_text = molecular_agent.generate_substructure(
                samples,
                feature_aggregate.substructures,
                feedback_strategy=args.feedback_strategy if round_index > 0 else "random",
            )
            added_molecule = feature_aggregate.add_drug_features(molecule_text)
            print(f"Added drug SMARTS features: {added_molecule}")

        if args.mode in {"both", "protein"}:
            protein_text = protein_agent.generate_substructure(samples)
            added_protein = feature_aggregate.add_target_features(protein_text)
            print(f"Added target fragment features: {added_protein}")

        vocab_path = output_dir / f"knowmol_vocab_round_{round_index + 1:03d}.py"
        feature_aggregate.write_vocab(vocab_path)
        print(f"Saved vocab: {vocab_path}")

        metrics, badcases = evaluate_vocab(train, valid, test, feature_aggregate, args, round_index)
        print_metrics(metrics)

        rows.append(
            {
                "round": round_index + 1,
                "added_molecule_smarts": added_molecule,
                "added_protein_fragments": added_protein,
                "total_drug_features": len(feature_aggregate.substructures),
                "total_target_features": len(feature_aggregate.fragments),
                **metrics,
            }
        )

    summary = pd.DataFrame(rows)
    summary_path = output_dir / f"discovery_summary_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    summary.to_csv(summary_path, index=False)
    feature_aggregate.write_vocab(output_dir / "knowmol_vocab_final.py")
    print("\nDiscovery summary")
    print(summary.to_string(index=False))
    print(f"\nSaved summary: {summary_path}")
    print(f"Saved final vocab: {output_dir / 'knowmol_vocab_final.py'}")


if __name__ == "__main__":
    main()
