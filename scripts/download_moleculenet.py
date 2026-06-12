#!/usr/bin/env python
from __future__ import annotations

import argparse
import gzip
import shutil
import urllib.request
from pathlib import Path

import pandas as pd


BASE_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets"

DATASETS = {
    "bace": {
        "file": "bace.csv",
        "smiles": ["mol", "smiles", "drug"],
        "default_task": "Class",
    },
    "bbbp": {
        "file": "BBBP.csv",
        "smiles": ["smiles", "mol", "drug"],
        "default_task": "p_np",
    },
    "hiv": {
        "file": "HIV.csv",
        "smiles": ["smiles", "mol", "drug"],
        "default_task": "HIV_active",
    },
    "clintox": {
        "file": "clintox.csv.gz",
        "smiles": ["smiles", "mol", "drug"],
        "default_task": "CT_TOX",
    },
    "sider": {
        "file": "sider.csv.gz",
        "smiles": ["smiles", "mol", "drug"],
        "default_task": "Hepatobiliary disorders",
    },
    "tox21": {
        "file": "tox21.csv.gz",
        "smiles": ["smiles", "mol", "drug"],
        "default_task": "NR-AR",
    },
    "toxcast": {
        "file": "toxcast_data.csv.gz",
        "smiles": ["smiles", "mol", "drug"],
        "default_task": None,
    },
    "muv": {
        "file": "muv.csv.gz",
        "smiles": ["smiles", "mol", "drug"],
        "default_task": "MUV-466",
    },
    "freesolv": {
        "file": "SAMPL.csv",
        "smiles": ["smiles", "mol", "drug"],
        "default_task": "expt",
        "regression": True,
    },
    "esol": {
        "file": "delaney-processed.csv",
        "smiles": ["smiles", "mol", "drug"],
        "default_task": "measured log solubility in mols per litre",
        "regression": True,
    },
    "lipo": {
        "file": "Lipophilicity.csv",
        "smiles": ["smiles", "mol", "drug"],
        "default_task": "exp",
        "regression": True,
    },
}


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        print(f"Using cached file: {destination}")
        return
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, destination)
    print(f"Saved raw file: {destination}")


def maybe_decompress(path: Path) -> Path:
    if path.suffix != ".gz":
        return path
    output = path.with_suffix("")
    if output.exists():
        return output
    with gzip.open(path, "rb") as src, output.open("wb") as dst:
        shutil.copyfileobj(src, dst)
    return output


def choose_column(frame: pd.DataFrame, candidates: list[str], role: str) -> str:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    raise ValueError(f"Cannot find {role} column. Candidates: {candidates}. Available: {list(frame.columns)}")


def choose_task(frame: pd.DataFrame, dataset: str, requested_task: str | None) -> str:
    if requested_task:
        if requested_task not in frame.columns:
            raise ValueError(f"Task column {requested_task!r} not found. Available columns: {list(frame.columns)}")
        return requested_task

    default_task = DATASETS[dataset].get("default_task")
    if default_task and default_task in frame.columns:
        return str(default_task)

    excluded = {"smiles", "mol", "drug", "name", "compound", "id"}
    candidate_tasks = [column for column in frame.columns if column.lower() not in excluded]
    if len(candidate_tasks) == 1:
        return candidate_tasks[0]
    raise ValueError(
        f"{dataset} has multiple possible task columns. Pass --task. "
        f"Examples: {candidate_tasks[:10]}"
    )


def standardize_dataset(args: argparse.Namespace) -> Path:
    info = DATASETS[args.dataset]
    raw_dir = Path(args.raw_dir).expanduser()
    processed_dir = Path(args.output_dir).expanduser()
    raw_path = raw_dir / info["file"]
    download_file(f"{BASE_URL}/{info['file']}", raw_path)
    csv_path = maybe_decompress(raw_path)

    frame = pd.read_csv(csv_path)
    smiles_col = choose_column(frame, info["smiles"], "SMILES")
    task_col = choose_task(frame, args.dataset, args.task)
    result = frame[[smiles_col, task_col]].rename(columns={smiles_col: "drug", task_col: "label"})
    result = result.dropna(subset=["drug", "label"]).reset_index(drop=True)

    if info.get("regression"):
        if args.binarize_threshold is None:
            raise ValueError(
                f"{args.dataset} is a regression dataset. Pass --binarize-threshold to export binary labels, "
                "or use a regression-specific downstream script."
            )
        result["label"] = (result["label"].astype(float) >= args.binarize_threshold).astype(int)
    else:
        result["label"] = result["label"].astype(int)

    task_suffix = task_col.replace("/", "_").replace(" ", "_")
    output = processed_dir / f"{args.dataset}_{task_suffix}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(f"Saved standardized MoleculeNet CSV: {output}")
    print(f"Rows={len(result)} label_counts={result['label'].value_counts().to_dict()}")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download MoleculeNet CSV files and export KnowMol-compatible drug,label data."
    )
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--task", help="Task/label column to export for multi-task datasets")
    parser.add_argument("--raw-dir", default="data/moleculenet/raw")
    parser.add_argument("--output-dir", default="data/moleculenet/processed")
    parser.add_argument(
        "--binarize-threshold",
        type=float,
        help="Required for regression datasets such as FreeSolv, ESOL, or Lipophilicity.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    standardize_dataset(args)


if __name__ == "__main__":
    main()
