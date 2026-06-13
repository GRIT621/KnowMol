from __future__ import annotations

import argparse
import ast
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("AG_DISABLE_RAY", "1")

import numpy as np
import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem import rdPartialCharges
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.utils import shuffle

RDLogger.DisableLog("rdApp.*")

STANDARD_AA = list("ACDEFGHIKLMNPQRSTVWY")

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

KIBA_FEATURE_NAMES = [
    "logP",
    "TPSA",
    "MolWt",
] + [f"fp_{i}" for i in range(1024)]

DTI_DATASETS = {"davis", "drugbank", "kiba", "prmt3"}
MOLECULENET_DATASETS = {
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
}
SUPPORTED_DATASETS = sorted(DTI_DATASETS | MOLECULENET_DATASETS)


def is_molecule_only_dataset(dataset_type: str | None) -> bool:
    return bool(dataset_type and dataset_type.lower() in MOLECULENET_DATASETS)


def _load_vocab_assignments(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}

    path = Path(path).expanduser().resolve()
    source = path.read_text()

    parsed = ast.parse(source, filename=str(path))
    found: dict[str, Any] = {}
    for node in parsed.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in {"substructure_patterns", "binding_fragments"}:
                try:
                    found[target.id] = ast.literal_eval(node.value)
                except Exception:
                    pass
    return found


def load_feature_dicts(
    drug_dict: str | Path | None = None,
    protein_dict: str | Path | None = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load KnowMol feature dictionaries from separate drug/protein txt files."""
    drug_found = _load_vocab_assignments(drug_dict) if drug_dict else {}
    protein_found = _load_vocab_assignments(protein_dict) if protein_dict else {}
    return drug_found.get("substructure_patterns", {}), protein_found.get("binding_fragments", {})


def smiles_to_features_enhanced(smiles: str) -> list[float]:
    """Kinase-binding-oriented molecular descriptors from the user's validation code."""
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return [0.0] * len(FEATURE_NAMES)

    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    rot_bonds = Lipinski.NumRotatableBonds(mol)
    aromatic_rings = Lipinski.NumAromaticRings(mol)
    aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
    ring_count = mol.GetRingInfo().NumRings()
    hydrophobic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() in [6, 16])
    hetero_atoms = Lipinski.NumHeteroatoms(mol)
    formal_charge = Chem.GetFormalCharge(mol)
    labute_asa = rdMolDescriptors.CalcLabuteASA(mol)
    num_atoms = mol.GetNumAtoms()
    num_bonds = mol.GetNumBonds()

    polar_fraction = tpsa / labute_asa if labute_asa > 0 else 0.0
    aromatic_density = aromatic_atoms / num_atoms if num_atoms else 0.0
    hbond_density = (hbd + hba) / num_atoms if num_atoms else 0.0
    flex_ratio = rot_bonds / num_bonds if num_bonds else 0.0

    return [
        mw,
        logp,
        tpsa,
        hbd,
        hba,
        rot_bonds,
        aromatic_rings,
        ring_count,
        hetero_atoms,
        hydrophobic_atoms,
        formal_charge,
        labute_asa,
        polar_fraction,
        aromatic_density,
        hbond_density,
        flex_ratio,
        Descriptors.FractionCSP3(mol),
        Descriptors.BertzCT(mol),
    ]


def smiles_to_features_enhanced30(smiles: str) -> list[float]:
    """Reserved 30D molecular descriptor interface from the original experiment."""
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return [0.0] * 30

    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    rot_bonds = Lipinski.NumRotatableBonds(mol)
    ring_count = mol.GetRingInfo().NumRings()
    aromatic_rings = Lipinski.NumAromaticRings(mol)
    atom_nums = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    total_atoms = mol.GetNumAtoms()
    aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
    hydrophobic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() in [6, 16])
    hetero_atoms = Lipinski.NumHeteroatoms(mol)
    formal_charge = Chem.GetFormalCharge(mol)
    labute_asa = rdMolDescriptors.CalcLabuteASA(mol)

    try:
        rdPartialCharges.ComputeGasteigerCharges(mol)
        charges = [abs(float(atom.GetProp("_GasteigerCharge"))) for atom in mol.GetAtoms()]
        max_charge = max(charges)
        min_charge = min(charges)
    except Exception:
        max_charge = 0.0
        min_charge = 0.0

    return [
        mw,
        logp,
        tpsa,
        hbd,
        hba,
        rot_bonds,
        rot_bonds / mol.GetNumBonds() if mol.GetNumBonds() else 0.0,
        ring_count,
        aromatic_rings,
        atom_nums.count(6),
        atom_nums.count(7),
        atom_nums.count(8),
        atom_nums.count(16),
        atom_nums.count(9) + atom_nums.count(17) + atom_nums.count(35) + atom_nums.count(53),
        hetero_atoms,
        hydrophobic_atoms,
        formal_charge,
        max_charge,
        min_charge,
        labute_asa,
        tpsa / labute_asa if labute_asa else 0.0,
        Descriptors.BertzCT(mol),
        Descriptors.FractionCSP3(mol),
        aromatic_atoms,
        aromatic_atoms / total_atoms if total_atoms else 0.0,
        (hbd + hba) / total_atoms if total_atoms else 0.0,
        hetero_atoms / total_atoms if total_atoms else 0.0,
        hydrophobic_atoms / total_atoms if total_atoms else 0.0,
        tpsa / mw if mw else 0.0,
        logp / mw if mw else 0.0,
    ]


def smiles_to_features_optimized(smiles: str, fp_radius: int = 2, fp_bits: int = 1024) -> list[float]:
    """Molecular descriptors plus Morgan fingerprint used by the best KIBA model."""
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return [0.0] * (3 + fp_bits)

    basic_feats = [
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),
        Descriptors.MolWt(mol),
    ]
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, fp_radius, nBits=fp_bits)
    fp_list = [int(bit) for bit in fp.ToBitString()]
    return basic_feats + fp_list


def substructure_features(smiles: str, substructure_patterns: Dict[str, Any]) -> list[int]:
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return [0] * len(substructure_patterns)

    features = []
    for sma in substructure_patterns.keys():
        patt = Chem.MolFromSmarts(sma)
        features.append(int(patt is not None and mol.HasSubstructMatch(patt)))
    return features


def protein_substructure_features(seq: str, binding_fragments: Dict[str, Any]) -> list[int]:
    seq = str(seq).upper()
    return [int(frag in seq) for frag in binding_fragments.keys()]


def protein_to_features_optimized(seq: str) -> list[float]:
    """Protein physicochemical properties plus 20D amino-acid composition."""
    clean_seq = "".join(aa for aa in str(seq).upper() if aa in STANDARD_AA)
    if not clean_seq:
        return [0.0] * 25

    try:
        analysis = ProteinAnalysis(clean_seq)
        base_feats = [
            analysis.gravy(),
            analysis.aromaticity(),
            analysis.instability_index(),
            analysis.isoelectric_point(),
            analysis.molecular_weight(),
        ]
        aac = analysis.get_amino_acids_percent()
        return base_feats + [aac.get(aa, 0.0) for aa in STANDARD_AA]
    except Exception:
        return [0.0] * 25


def extract_features(
    data: pd.DataFrame,
    substructure_patterns: Dict[str, Any],
    binding_fragments: Dict[str, Any],
    dataset_type: str | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Feature extraction order matches the tabular training pipeline."""
    data = data.reset_index(drop=True)

    if dataset_type and dataset_type.lower() == "kiba":
        drug_combined = data["drug"].apply(smiles_to_features_optimized)
        drug_df = pd.DataFrame(drug_combined.tolist(), columns=KIBA_FEATURE_NAMES)
    else:
        drug_combined = data["drug"].apply(smiles_to_features_enhanced)
        drug_df = pd.DataFrame(drug_combined.tolist(), columns=FEATURE_NAMES)

    drug_sub = data["drug"].apply(lambda s: substructure_features(s, substructure_patterns))
    drug_df_sub = pd.DataFrame(drug_sub.tolist(), columns=list(substructure_patterns.keys()))

    feature_frames = [drug_df, drug_df_sub]
    has_target = "target" in data.columns and not is_molecule_only_dataset(dataset_type)
    if has_target:
        protein_combined = data["target"].apply(protein_to_features_optimized)
        prot_base_names = ["gravy", "aromaticity", "instability", "isoelectric_point", "mol_weight"]
        prot_aac_names = [f"aac_{aa}" for aa in STANDARD_AA]
        protein_df = pd.DataFrame(protein_combined.tolist(), columns=prot_base_names + prot_aac_names)

        protein_sub = data["target"].apply(lambda s: protein_substructure_features(s, binding_fragments))
        protein_df_sub = pd.DataFrame(
            protein_sub.tolist(),
            columns=[f"frag_{i + 1}" for i in range(len(binding_fragments))],
        )
        feature_frames.extend([protein_df, protein_df_sub])

    x = pd.concat(feature_frames, axis=1)
    y = data["label"].reset_index(drop=True) if "label" in data.columns else pd.Series([0] * len(data))
    return x.fillna(0), y


def normalize_dataset(df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
    """Normalize DTI or molecule-only CSV columns to drug,target,label."""
    dataset_type = dataset_type.lower()
    df = df.copy()

    rename_map = {
        "mol": "drug",
        "molecule": "drug",
        "SMILES": "drug",
        "smile": "drug",
        "smiles": "drug",
        "seq": "target",
        "sequence": "target",
        "cid": "drug_id",
        "uid": "target_id",
        "Class": "label",
        "class": "label",
        "Label": "label",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    if "drug" not in df.columns:
        if dataset_type == "drugbank" and df.shape[1] >= 5:
            cols = list(df.columns)
            df = df.rename(columns={cols[-4]: "drug_id", cols[-3]: "drug", cols[-2]: "target_id", cols[-1]: "target"})
        else:
            raise ValueError("Cannot find molecule column. Expected drug, smiles, smile, mol, molecule, or SMILES.")

    molecule_only = is_molecule_only_dataset(dataset_type)
    if not molecule_only and "target" not in df.columns:
        if dataset_type == "drugbank" and df.shape[1] >= 5:
            cols = list(df.columns)
            df = df.rename(columns={cols[-4]: "drug_id", cols[-3]: "drug", cols[-2]: "target_id", cols[-1]: "target"})
        else:
            raise ValueError(
                "Cannot find drug/target columns. Expected drug+target, smiles+seq, or a known DrugBank layout."
            )

    if "label" not in df.columns:
        if dataset_type == "drugbank":
            df["label"] = 0
            positive_stop = min(16532, len(df))
            if positive_stop > 1:
                df.iloc[1:positive_stop, df.columns.get_loc("label")] = 1
        else:
            raise ValueError(f"{dataset_type} data must include a label column.")

    required = ["drug", "label"] if molecule_only else ["drug", "target", "label"]
    return df.dropna(subset=required).reset_index(drop=True)


def read_drugbank_dataset(data_path: str | Path) -> pd.DataFrame:
    """Read DrugBank exactly like the original validation script.

    The original DrugBank experiment treated the CSV as headerless and assigned
    labels from the raw row positions: rows 1..16531 are positive, all others
    are negative. Keep that behavior so saved models are evaluated on the same
    reconstructed split.
    """
    df = pd.read_csv(data_path, header=None, sep=None, engine="python")
    if df.shape[1] < 5:
        raise ValueError(f"DrugBank file must have at least 5 columns, got {df.shape[1]}.")

    df = df.iloc[:, :5].copy()
    df.columns = ["id", "drug_id", "drug", "target_id", "target"]
    df["label"] = 0
    positive_stop = min(16532, len(df))
    if positive_stop > 1:
        df.iloc[1:positive_stop, df.columns.get_loc("label")] = 1
    return df.dropna(subset=["drug", "target", "label"]).reset_index(drop=True)


def read_dataset(data_path: str | Path, dataset_type: str) -> pd.DataFrame:
    path = Path(data_path).expanduser()
    if path.is_dir():
        raise ValueError("read_dataset expects a CSV file. Use read_or_split_dataset for split directories.")
    if dataset_type.lower() == "drugbank":
        return read_drugbank_dataset(path)
    df = pd.read_csv(path, sep=None, engine="python")
    return normalize_dataset(df, dataset_type)


def read_or_split_dataset(
    data_path: str | Path,
    dataset_type: str,
    random_state: int = 42,
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    path = Path(data_path).expanduser()
    if path.is_dir() and (path / "train.csv").exists() and (path / "test.csv").exists():
        train = normalize_dataset(pd.read_csv(path / "train.csv"), dataset_type)
        valid_file = path / "valid.csv"
        valid = normalize_dataset(pd.read_csv(valid_file), dataset_type) if valid_file.exists() else train.iloc[0:0].copy()
        test = normalize_dataset(pd.read_csv(path / "test.csv"), dataset_type)
        print_split_diagnostics(train, valid, test)
        return train, valid, test

    df = read_dataset(path, dataset_type)
    df = shuffle(df, random_state=random_state).reset_index(drop=True)
    n_total = len(df)
    n_train = int(train_ratio * n_total)
    n_valid = int(valid_ratio * n_total)
    train = df.iloc[:n_train].reset_index(drop=True)
    valid = df.iloc[n_train : n_train + n_valid].reset_index(drop=True)
    test = df.iloc[n_train + n_valid :].reset_index(drop=True)
    print(
        "Split label counts | "
        f"train: {train['label'].value_counts().to_dict()} | "
        f"valid: {valid['label'].value_counts().to_dict()} | "
        f"test: {test['label'].value_counts().to_dict()}"
    )
    print_split_diagnostics(train, valid, test)
    return train, valid, test


def print_split_diagnostics(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame) -> None:
    """Print split sanity checks without changing the split."""
    frames = {"train": train, "valid": valid, "test": test}
    for name, frame in frames.items():
        if "label" in frame.columns:
            print(f"{name} labels: {frame['label'].value_counts().to_dict()}")

    if "drug_id" in train.columns and "drug_id" in test.columns:
        train_drugs = set(train["drug_id"].astype(str))
        test_drugs = set(test["drug_id"].astype(str))
        print(f"Train/test drug_id overlap: {len(train_drugs & test_drugs)} / {len(test_drugs)} test drugs")

    if "target_id" in train.columns and "target_id" in test.columns:
        train_targets = set(train["target_id"].astype(str))
        test_targets = set(test["target_id"].astype(str))
        print(f"Train/test target_id overlap: {len(train_targets & test_targets)} / {len(test_targets)} test targets")

    if {"drug_id", "target_id"}.issubset(train.columns) and {"drug_id", "target_id"}.issubset(test.columns):
        train_pairs = set(zip(train["drug_id"].astype(str), train["target_id"].astype(str)))
        test_pairs = set(zip(test["drug_id"].astype(str), test["target_id"].astype(str)))
        print(f"Train/test pair overlap: {len(train_pairs & test_pairs)} / {len(test_pairs)} test pairs")

    if "target" not in train.columns:
        print("Molecule-only dataset detected: protein descriptors and protein fragments are disabled.")


def get_tabular_predictor():
    from autogluon.tabular import TabularPredictor

    return TabularPredictor


def compute_binary_metrics(y_true: Iterable[int], y_pred: Iterable[int], y_prob: Iterable[float]) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    has_both_classes = len(np.unique(y_true)) == 2
    return {
        "auc_roc": roc_auc_score(y_true, y_prob) if has_both_classes else float("nan"),
        "auc_pr": average_precision_score(y_true, y_prob) if has_both_classes else float("nan"),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }


def print_metrics(metrics: dict[str, float]) -> None:
    print("\nMetrics")
    print("-" * 32)
    for key in ["auc_roc", "auc_pr", "accuracy", "precision", "recall", "specificity", "f1", "mcc"]:
        print(f"{key:12s}: {metrics[key]:.4f}")
    print("-" * 32)


def train_mode(args: argparse.Namespace) -> None:
    substructures, fragments = load_feature_dicts(args.drug_dict, args.protein_dict)
    train, valid, test = read_or_split_dataset(args.data, args.dataset, args.seed)

    print(f"Dataset: {args.dataset}")
    print(f"Train={len(train)} Valid={len(valid)} Test={len(test)}")
    x_train, y_train = extract_features(train, substructures, fragments, args.dataset)
    x_valid, y_valid = extract_features(valid, substructures, fragments, args.dataset) if len(valid) else (None, None)

    if x_valid is not None:
        train_df = pd.concat([pd.concat([x_train, x_valid], axis=0), pd.concat([y_train, y_valid], axis=0)], axis=1)
    else:
        train_df = pd.concat([x_train, y_train], axis=1)
    train_df = train_df.fillna(0)

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
        evaluate_predictor(predictor, test, substructures, fragments, show_leaderboard=False, dataset_type=args.dataset)


def evaluate_predictor(
    predictor: Any,
    test_data: pd.DataFrame,
    substructures: Dict[str, Any],
    fragments: Dict[str, Any],
    show_leaderboard: bool,
    dataset_type: str | None = None,
) -> dict[str, float]:
    x_test, y_test = extract_features(test_data, substructures, fragments, dataset_type)
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
    badcase_columns = [
        column for column in ["drug", "target", "label", "pred_prob", "error_margin"] if column in test_with_scores.columns
    ]
    print(test_with_scores.nlargest(5, "error_margin")[badcase_columns])

    if show_leaderboard:
        print("\nModel leaderboard on the selected test set")
        print(predictor.leaderboard(pd.concat([x_test, y_test], axis=1), silent=True))

    return metrics


def test_mode(args: argparse.Namespace) -> None:
    substructures, fragments = load_feature_dicts(args.drug_dict, args.protein_dict)
    _, _, test = read_or_split_dataset(args.data, args.dataset, args.seed)

    TabularPredictor = get_tabular_predictor()
    predictor = TabularPredictor.load(args.model_path)
    print(f"Loaded model: {args.model_path}")
    print(f"Dataset: {args.dataset} | Test={len(test)}")
    evaluate_predictor(predictor, test, substructures, fragments, args.leaderboard, args.dataset)


def predict_single(
    predictor: Any,
    ligand_smiles: str,
    protein_seq: str,
    substructures: Dict[str, Any],
    fragments: Dict[str, Any],
    dataset_type: str | None = None,
) -> tuple[int, float]:
    single_data = pd.DataFrame([{"drug": ligand_smiles, "target": protein_seq, "label": 0}])
    x_single, _ = extract_features(single_data, substructures, fragments, dataset_type)
    prob = float(predictor.predict_proba(x_single).iloc[0, 1])
    label = int(predictor.predict(x_single).iloc[0])
    return label, prob


def prmt3_mode(args: argparse.Namespace) -> None:
    substructures, fragments = load_feature_dicts(args.drug_dict, args.protein_dict)
    TabularPredictor = get_tabular_predictor()
    predictor = TabularPredictor.load(args.model_path)

    if args.protein_seq:
        protein_seq = args.protein_seq
    elif args.protein_seq_file:
        protein_seq = Path(args.protein_seq_file).read_text().replace("\n", "").strip()
    else:
        ligands_df_for_seq = pd.read_csv(args.ligands_csv)
        if "target" not in ligands_df_for_seq.columns:
            raise ValueError("Provide --protein-seq/--protein-seq-file, or a ligands CSV containing a target column.")
        protein_seq = str(ligands_df_for_seq["target"].iloc[0])

    ligands = pd.read_csv(args.ligands_csv)
    if "drug" not in ligands.columns:
        if "smiles" in ligands.columns:
            ligands = ligands.rename(columns={"smiles": "drug"})
        elif "smile" in ligands.columns:
            ligands = ligands.rename(columns={"smile": "drug"})
        else:
            raise ValueError("Ligand CSV must contain drug, smiles, or smile column.")

    name_col = args.name_col if args.name_col in ligands.columns else None
    rows = []
    for i, row in ligands.iterrows():
        name = row[name_col] if name_col else f"ligand_{i + 1}"
        try:
            label, prob = predict_single(predictor, row["drug"], protein_seq, substructures, fragments)
            rows.append({"name": name, "drug": row["drug"], "pred_label": label, "binding_score": prob})
        except Exception as exc:
            rows.append({"name": name, "drug": row["drug"], "pred_label": "ERROR", "binding_score": np.nan, "error": str(exc)})

    result = pd.DataFrame(rows).sort_values("binding_score", ascending=False, na_position="last")
    print(result.to_string(index=False))
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output, index=False)
        print(f"\nSaved PRMT3 scores: {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="KnowMol downstream ML validation")
    default_dict_dir = Path(__file__).resolve().parents[1] / "discovered_fragments"
    parser.add_argument("--drug-dict", default=str(default_dict_dir / "drug_dict.txt"))
    parser.add_argument("--protein-dict", default=str(default_dict_dir / "protein_dict.txt"))

    subparsers = parser.add_subparsers(dest="mode", required=True)

    train = subparsers.add_parser("train", help="Train a tabular validation model")
    train.add_argument("--dataset", choices=SUPPORTED_DATASETS, required=True)
    train.add_argument("--data", required=True, help="CSV file or split directory containing train/valid/test CSVs")
    train.add_argument("--model-path", required=True)
    train.add_argument("--time-limit", type=int, default=600)
    train.add_argument("--presets", default="best_quality")
    train.add_argument("--num-cpus", type=int, default=8)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--eval-metric", default="roc_auc")
    train.add_argument("--evaluate-after-train", action="store_true")
    train.set_defaults(func=train_mode)

    test = subparsers.add_parser("test", help="Evaluate a saved model on the dataset test split")
    test.add_argument("--dataset", choices=SUPPORTED_DATASETS, required=True)
    test.add_argument("--data", required=True, help="CSV file or split directory containing train/valid/test CSVs")
    test.add_argument("--model-path", required=True)
    test.add_argument("--seed", type=int, default=42)
    test.add_argument("--leaderboard", action="store_true")
    test.set_defaults(func=test_mode)

    prmt3 = subparsers.add_parser("prmt3", help="Score many ligands against one locked PRMT3 protein sequence")
    prmt3.add_argument("--model-path", required=True)
    prmt3.add_argument("--ligands-csv", required=True)
    prmt3.add_argument("--protein-seq")
    prmt3.add_argument("--protein-seq-file")
    prmt3.add_argument("--name-col", default="name")
    prmt3.add_argument("--output")
    prmt3.set_defaults(func=prmt3_mode)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
