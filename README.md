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

## PRMT3 Case Study

The PRMT3 natural-compound screen is exposed as a framework-compatible script
that reuses the KnowMol feature dictionaries and downstream metrics:

```bash
python scripts/prmt3_screen.py \
  --train-csv ../EviDTI_dataset/data/case_studies/prmt3/PRMT3_Final_Training_Set.csv \
  --ligands-csv ../EviDTI_dataset/data/case_studies/prmt3/PRMT3_drugbank_Matches.csv \
  --output outputs/prmt3/prmt3_scores.csv \
  --model-path models/prmt3/prmt3_rf.pkl \
  --use-default-prmt3-seq
```

To reuse a trained PRMT3 model for rescoring:

```bash
python scripts/prmt3_screen.py \
  --load-model \
  --model-path models/prmt3/prmt3_rf.pkl \
  --ligands-csv ../EviDTI_dataset/data/case_studies/prmt3/PRMT3_drugbank_Matches.csv \
  --output outputs/prmt3/prmt3_scores.csv \
  --use-default-prmt3-seq
```

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
- `scripts/`: auxiliary downstream, mining, and PRMT3 case-study scripts.

Runtime artifacts such as `.env`, `outputs/`, trained models, logs, and Python
caches are intentionally ignored by Git.
