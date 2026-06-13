# KnowMol

KnowMol is an iterative feature-discovery workflow for drug-target interaction
datasets. It uses molecular and protein agents to propose interpretable
substructure features, validates them with downstream machine-learning models,
and keeps useful features in long memory.

## Core Entrypoint

```bash
python knowmol_discovery.py \
  --dataset davis \
  --data data/raw/davis_total_cid_unid.csv \
  --output-dir discovered_fragments/runs/davis \
  --rounds 30 \
  --sample-size 2 \
  --mode both \
  --model deepseek-v3.2 \
  --api-base "$OPENAI_BASE_URL" \
  --api-key "$OPENAI_API_KEY" \
  --validator-backend sklearn
```

This command runs KnowMol discovery on a DTI dataset and writes the discovered
molecular substructures and protein fragments to `discovered_fragments/runs/`.
The released feature dictionaries are provided in
`discovered_fragments/drug_dict.txt` and `discovered_fragments/protein_dict.txt`.

## MoleculeNet Data

MoleculeNet molecule-only tasks use the same `knowmol_discovery.py` workflow.
After preparing the task CSV, switch the dataset name:

```bash
python knowmol_discovery.py \
  --dataset bace \
  --data data/moleculenet/processed/bace_Class.csv \
  --output-dir discovered_fragments/runs/bace \
  --rounds 5 \
  --mode molecule
```

Replace `bace` with another MoleculeNet dataset such as `bbbp`, `hiv`,
`clintox`, `sider`, `tox21`, `toxcast`, `muv`, `freesolv`, `esol`, or `lipo`.
Protein agents and protein features are disabled automatically for
molecule-only datasets.

## PRMT3 Case Study

The PRMT3 case study uses the included 20 candidate ligands in
`data/prmt3/prmt3_candidates_20.csv`. Train the case-study model and score the
candidates with:

```bash
python scripts/prmt3_screen.py \
  --train-csv data/prmt3/prmt3_drugbank_merged_training.csv \
  --ligands-csv data/prmt3/prmt3_candidates_20.csv \
  --output outputs/prmt3/prmt3_scores.csv \
  --use-default-prmt3-seq
```

## Interpretability

The manuscript visualizations can be regenerated with
`interpretability/shap_visualization.py`. Four modes are supported:

- `global`: all drug-target records.
- `target`: one fixed target against all drugs.
- `drug`: one fixed drug against all targets.
- `pair`: one selected drug-target pair.

Example:

```bash
python interpretability/shap_visualization.py \
  --mode target \
  --dataset davis \
  --data data/raw/davis_total_cid_unid.csv \
  --model-path /path/to/AutogluonModels/davis_realwant1 \
  --target-id ABL1 \
  --output-dir outputs/interpretability/davis_ABL1
```

## Included Data

Raw Davis, DrugBank, and KIBA DTI CSV files matching the EviDTI data are
included under `data/raw/`:

- `data/raw/davis_total_cid_unid.csv`
- `data/raw/drugbank_total_cid_unid.csv`
- `data/raw/KIBA_total_cid_unid.csv`

These files use `cid,uid,smiles,seq,label` columns; the training and discovery
loaders normalize them to `drug,target,label` automatically.

## Repository Layout

- `knowmol_discovery.py`: iterative KnowMol discovery runner.
- `agents/`: LLM agents, feature aggregation, memory, validation, and analysis.
- `downstream_ml/`: feature extraction, validation helpers, and baseline assets.
- `discovered_fragments/`: checked-in feature dictionaries plus ignored discovery-run outputs.
- `scripts/`: auxiliary downstream, MoleculeNet download, mining, and PRMT3 case-study scripts.

Runtime artifacts such as `.env`, `outputs/`, trained models, logs, and Python
caches are intentionally ignored by Git.

## License

This project is released under the [BSD 3-Clause License](https://opensource.org/license/BSD-3-Clause).
