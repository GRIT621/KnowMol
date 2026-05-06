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
- `scripts/`: auxiliary downstream and mining scripts.

Runtime artifacts such as `.env`, `outputs/`, trained models, logs, and Python
caches are intentionally ignored by Git.
