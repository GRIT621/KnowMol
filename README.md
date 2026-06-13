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
  --output-dir outputs/knowmol_api_sklearn_target \
  --rounds 5 \
  --sample-size 2 \
  --mode both \
  --model deepseek-v3.2 \
  --api-base "$OPENAI_BASE_URL" \
  --api-key "$OPENAI_API_KEY" \
  --validator-backend sklearn
```

Use `--mock-agents` for local smoke tests that do not call an LLM API.

## MoleculeNet Data

MoleculeNet molecule-only tasks use the same `knowmol_discovery.py` workflow.
After preparing the task CSV, switch the dataset name:

```bash
python knowmol_discovery.py \
  --dataset bace \
  --data data/moleculenet/processed/bace_Class.csv \
  --output-dir outputs/discovery/bace \
  --rounds 5 \
  --mode molecule
```

Replace `bace` with another MoleculeNet dataset such as `bbbp`, `hiv`,
`clintox`, `sider`, `tox21`, `toxcast`, `muv`, `freesolv`, `esol`, or `lipo`.
Protein agents and protein features are disabled automatically for
molecule-only datasets.

## PRMT3 Case Study

The PRMT3 case-study scores were generated with the framework-compatible
screening script:

```bash
python scripts/prmt3_screen.py \
  --train-csv ../EviDTI_dataset/data/case_studies/prmt3/PRMT3_Final_Training_Set.csv \
  --ligands-csv ../EviDTI_dataset/data/case_studies/prmt3/PRMT3_drugbank_Matches.csv \
  --output outputs/prmt3/prmt3_scores.csv \
  --use-default-prmt3-seq
```

## Interpretability

The manuscript visualizations are organized under `interpretability/`:
`global` for all drug-target records, `target` for one target against all drugs,
`drug` for one drug against all targets, and `pair` for one drug-target pair.
Model paths and selected drug/target IDs are passed as command-line parameters.

## Included Data

Raw Davis, DrugBank, and KIBA DTI CSV files are included under `data/raw/`:

- `data/raw/davis_total_cid_unid.csv`
- `data/raw/drugbank_total_cid_unid.csv`
- `data/raw/KIBA_total_cid_unid.csv`

These files use `cid,uid,smiles,seq,label` columns; the training and discovery
loaders normalize them to `drug,target,label` automatically.

## Repository Layout

- `knowmol_discovery.py`: iterative KnowMol discovery runner.
- `agents/`: LLM agents, feature aggregation, memory, validation, and analysis.
- `downstream_ml/`: feature extraction, validation helpers, and baseline assets.
- `scripts/`: auxiliary downstream, MoleculeNet download, mining, and PRMT3 case-study scripts.

Runtime artifacts such as `.env`, `outputs/`, trained models, logs, and Python
caches are intentionally ignored by Git.
