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

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from agents import AnalysisAgent, FeatureAggregator, MolecularAgent, ProteinAgent, ShortMemory, ValidateAgent

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
    sample_columns = [column for column in ["drug", "target", "label"] if column in train.columns]
    if round_index > 0 and feedback_strategy == "badcase" and badcases is not None and len(badcases):
        badcase_columns = sample_columns + ["_feedback_error"]
        return badcases.head(sample_size)[badcase_columns].to_dict("records")

    sampled = train.sample(n=min(sample_size, len(train)), random_state=seed + round_index)
    return sampled[sample_columns].to_dict("records")


def mock_target_fragments(samples: list[dict[str, Any]], excluded_fragments: dict[str, Any], limit: int = 10) -> str:
    fragments: list[str] = []
    excluded = {str(fragment).upper() for fragment in excluded_fragments}
    for sample in samples:
        target = "".join(ch for ch in str(sample.get("target", "")).upper() if ch.isalpha())
        for start in range(0, max(len(target) - 11, 0), 17):
            fragment = target[start : start + 14]
            if len(fragment) < 8 or fragment in excluded or fragment in fragments:
                continue
            fragments.append(fragment)
            if len(fragments) >= limit:
                break
        if len(fragments) >= limit:
            break
    return "\n".join(f"{idx}. {fragment}" for idx, fragment in enumerate(fragments, start=1))


def mock_drug_smarts(excluded_substructures: dict[str, Any], limit: int = 10) -> str:
    candidates = ["[#6]-[#7]", "[#6]=[#8]", "c1ccccc1", "[#7]-[#6]=[#8]", "[#16](=[#8])=[#8]"]
    excluded = {str(smarts) for smarts in excluded_substructures}
    selected = [smarts for smarts in candidates if smarts not in excluded][:limit]
    return "\n".join(f"{idx}. {smarts}" for idx, smarts in enumerate(selected, start=1))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Iterative KnowMol feature discovery for drug-target interaction datasets."
    )
    parser.add_argument(
        "--dataset",
        choices=[
            "davis",
            "drugbank",
            "kiba",
            "prmt3",
            "bace",
            "bbbp",
            "hiv",
            "clintox",
            "sider",
            "tox21",
            "toxcast",
            "muv",
            "freesolv",
            "esol",
            "lipo",
        ],
        required=True,
    )
    parser.add_argument("--data", required=True, help="CSV file or split directory containing train/valid/test CSVs")
    parser.add_argument("--output-dir", default="discovered_fragments/runs/default")
    parser.add_argument("--drug-dict", default=str(ROOT / "discovered_fragments" / "drug_dict.txt"))
    parser.add_argument("--protein-dict", default=str(ROOT / "discovered_fragments" / "protein_dict.txt"))
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--max-train-rows", type=int, help="Optional row cap for train data")
    parser.add_argument("--max-valid-rows", type=int, help="Optional row cap for validation data")
    parser.add_argument("--max-test-rows", type=int, help="Optional row cap for test data")
    parser.add_argument("--feedback-strategy", choices=["random", "badcase"], default="badcase")
    parser.add_argument("--mode", choices=["both", "drug", "target", "molecule", "protein"], default="both")
    parser.add_argument("--model", default="pro-deepseek-r1")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--api-base", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--time-limit", type=int, default=600)
    parser.add_argument("--presets", default="best_quality")
    parser.add_argument("--num-cpus", type=int, default=8)
    parser.add_argument(
        "--validator-backend",
        choices=["tabular", "xgboost"],
        default="tabular",
        help="Validation backend. Use xgboost for the manuscript-style surrogate validator.",
    )
    parser.add_argument("--mock-agents", action="store_true", help="Use deterministic local agent outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-metric", default="roc_auc")
    parser.add_argument("--memory-metric", default="auc_roc")
    parser.add_argument("--analysis-report", default="multi_level_analysis.md")
    parser.add_argument("--long-drug-dict", help="Optional long-memory drug_dict.txt path")
    parser.add_argument("--long-protein-dict", help="Optional long-memory protein_dict.txt path")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    from downstream_ml.validation import (
        is_molecule_only_dataset,
        load_feature_dicts,
        print_metrics,
        read_or_split_dataset,
    )

    random.seed(args.seed)
    np.random.seed(args.seed)

    train, valid, test = read_or_split_dataset(args.data, args.dataset, args.seed)
    if args.max_train_rows:
        train = train.sample(n=min(args.max_train_rows, len(train)), random_state=args.seed).reset_index(drop=True)
    if args.max_valid_rows:
        valid = valid.sample(n=min(args.max_valid_rows, len(valid)), random_state=args.seed).reset_index(drop=True)
    if args.max_test_rows:
        test = test.sample(n=min(args.max_test_rows, len(test)), random_state=args.seed).reset_index(drop=True)
    molecule_only = is_molecule_only_dataset(args.dataset) or "target" not in train.columns
    if molecule_only and args.mode in {"target", "protein"}:
        raise ValueError("Molecule-only datasets do not contain protein targets; use --mode molecule or --mode drug.")
    effective_mode = "molecule" if molecule_only and args.mode == "both" else args.mode
    output_dir = Path(args.output_dir)
    if output_dir == Path("/private/tmp/knowmol_api_sklearn_target"):
        output_dir = ROOT / "outputs" / "knowmol_api_sklearn_target"
        args.output_dir = str(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    memory_dir = output_dir / "memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    long_drug_dict_path = Path(args.long_drug_dict) if args.long_drug_dict else memory_dir / "drug_dict.txt"
    long_protein_dict_path = Path(args.long_protein_dict) if args.long_protein_dict else memory_dir / "protein_dict.txt"

    initial_substructures: dict[str, Any] = {}
    initial_fragments: dict[str, Any] = {}
    if long_drug_dict_path.exists() and long_protein_dict_path.exists():
        initial_substructures, initial_fragments = load_feature_dicts(long_drug_dict_path, long_protein_dict_path)
    if not initial_substructures and not initial_fragments and args.drug_dict and args.protein_dict:
        initial_substructures, initial_fragments = load_feature_dicts(args.drug_dict, args.protein_dict)
    if molecule_only:
        initial_fragments = {}

    feature_aggregate = FeatureAggregator(initial_substructures, initial_fragments)
    context = DatasetContext(dataset_name=args.dataset)
    drug_agent = MolecularAgent("DrugAgent", context, args.model, args.api_key, args.api_base)
    target_agent = ProteinAgent("TargetAgent", context, args.model, args.api_key, args.api_base)
    validate_agent = ValidateAgent(
        dataset=args.dataset,
        output_dir=args.output_dir,
        time_limit=args.time_limit,
        presets=args.presets,
        num_cpus=args.num_cpus,
        eval_metric=args.eval_metric,
        backend=args.validator_backend,
    )
    analysis_agent = AnalysisAgent()
    short_memory = ShortMemory.load(memory_dir / "short_memory_all.json")

    badcases: pd.DataFrame | None = None
    rows: list[dict[str, Any]] = []

    print(f"Dataset: {args.dataset} | Train={len(train)} Valid={len(valid)} Test={len(test)}")
    if molecule_only:
        print("Molecule-only mode: target/protein agents and protein features are disabled.")
    print(
        "Initial drug features="
        f"{len(feature_aggregate.substructures)} target features={len(feature_aggregate.fragments)}"
    )

    previous_score = 0.0
    if args.rounds > 0:
        print("\nEvaluating initial long memory state")
        baseline_metrics, badcases = validate_agent.evaluate_round(
            train,
            valid,
            test,
            feature_aggregate.substructures,
            feature_aggregate.fragments,
            -1,
        )
        previous_score = float(baseline_metrics.get(args.memory_metric, 0.0))
        print_metrics(baseline_metrics)

    for round_index in range(args.rounds):
        print("\n" + "=" * 72)
        print(f"KnowMol discovery round {round_index + 1}/{args.rounds}")
        print("=" * 72)
        samples = choose_samples(train, badcases, round_index, args.feedback_strategy, args.sample_size, args.seed)
        memory_context = short_memory.build_agent_context(
            feature_aggregate.substructures,
            feature_aggregate.fragments,
            badcases,
        )
        short_memory.start_round(round_index, samples)
        candidate_aggregate = feature_aggregate.clone()

        added_drug = 0
        added_target = 0
        if effective_mode in {"both", "drug", "molecule"}:
            if args.mock_agents:
                drug_text = mock_drug_smarts(feature_aggregate.substructures)
            else:
                drug_text = drug_agent.generate_substructure(
                    samples,
                    feature_aggregate.substructures,
                    feedback_strategy=args.feedback_strategy if round_index > 0 else "random",
                    memory_context=memory_context,
                )
            added_drug = candidate_aggregate.add_drug_features(drug_text)
            short_memory.set_agent_output("drug_agent_output", drug_text)
            print(f"Added drug SMARTS features: {added_drug}")

        if effective_mode in {"both", "target", "protein"}:
            if args.mock_agents:
                target_text = mock_target_fragments(samples, feature_aggregate.fragments)
            else:
                target_text = target_agent.generate_substructure(
                    samples,
                    feature_aggregate.fragments,
                    feedback_strategy=args.feedback_strategy if round_index > 0 else "random",
                    memory_context=memory_context,
                )
            added_target = candidate_aggregate.add_target_features(target_text)
            short_memory.set_agent_output("target_agent_output", target_text)
            print(f"Added target fragment features: {added_target}")
        short_memory.set_candidate_features(candidate_aggregate.last_drug_keys, candidate_aggregate.last_target_keys)

        candidate_drug_dict_path = output_dir / f"candidate_drug_dict_round_{round_index + 1:03d}.txt"
        candidate_protein_dict_path = output_dir / f"candidate_protein_dict_round_{round_index + 1:03d}.txt"
        candidate_aggregate.write_dicts(candidate_drug_dict_path, candidate_protein_dict_path)
        print(f"Saved candidate dicts: {candidate_drug_dict_path}, {candidate_protein_dict_path}")

        metrics, badcases = validate_agent.evaluate_round(
            train,
            valid,
            test,
            candidate_aggregate.substructures,
            candidate_aggregate.fragments,
            round_index,
        )
        current_score = float(metrics.get(args.memory_metric, 0.0))
        validation_gain = current_score - previous_score
        consolidated = validation_gain > 0
        if consolidated:
            feature_aggregate.consolidate_from(
                candidate_aggregate,
                candidate_aggregate.last_drug_keys,
                candidate_aggregate.last_target_keys,
                validation_gain,
            )
            previous_score = current_score
            drug_dict_path = output_dir / f"drug_dict_round_{round_index + 1:03d}.txt"
            protein_dict_path = output_dir / f"protein_dict_round_{round_index + 1:03d}.txt"
            feature_aggregate.write_dicts(drug_dict_path, protein_dict_path)
            feature_aggregate.write_dicts(long_drug_dict_path, long_protein_dict_path)
            print(f"Validation gain {validation_gain:.6f} > 0; consolidated into long memory.")
            print(f"Saved dicts: {drug_dict_path}, {protein_dict_path}")
            print(f"Updated long-memory dicts: {long_drug_dict_path}, {long_protein_dict_path}")
        else:
            print(f"Validation gain {validation_gain:.6f} <= 0; discarded short-memory candidates.")

        short_memory.set_consolidation(validation_gain, consolidated)
        short_memory.set_validation_feedback(metrics, badcases)
        short_memory.write_round(memory_dir / f"short_memory_round_{round_index + 1:03d}.json")
        print_metrics(metrics)

        rows.append(
            {
                "round": round_index + 1,
                "added_drug_features": added_drug,
                "added_target_features": added_target,
                "validation_gain": validation_gain,
                "consolidated": consolidated,
                "total_drug_features": len(feature_aggregate.substructures),
                "total_target_features": len(feature_aggregate.fragments),
                **metrics,
            }
        )

    summary = pd.DataFrame(rows)
    summary_path = output_dir / f"discovery_summary_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    summary.to_csv(summary_path, index=False)
    feature_aggregate.write_dicts(output_dir / "drug_dict.txt", output_dir / "protein_dict.txt")
    feature_aggregate.write_dicts(long_drug_dict_path, long_protein_dict_path)
    short_memory.write_all(memory_dir / "short_memory_all.json")
    analysis_path = output_dir / args.analysis_report
    analysis_agent.write_round_report(
        analysis_path,
        args.dataset,
        rows,
        len(feature_aggregate.substructures),
        len(feature_aggregate.fragments),
    )
    print("\nDiscovery summary")
    print(summary.to_string(index=False))
    print(f"\nSaved summary: {summary_path}")
    print(f"Saved final dicts: {output_dir / 'drug_dict.txt'}, {output_dir / 'protein_dict.txt'}")
    print(f"Saved analysis report: {analysis_path}")


if __name__ == "__main__":
    main()
