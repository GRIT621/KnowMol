# KnowMol

**A Knowledge-Guided Multi-Agent Framework for Interpretable Drug–Target Interaction Prediction.**


## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.9+, RDKit, AutoGluon, and an OpenAI-compatible LLM endpoint
(`OPENAI_BASE_URL`, `OPENAI_API_KEY`).

## Repository structure

```
knowmol_discovery.py        Closed-loop feature-discovery runner
agents/                     Drug / target / validate / analysis agents and memory
downstream_ml/              Feature extraction, training, evaluation, ablation
interpretability/           SHAP visualization for multi-level interpretability
scripts/                    MoleculeNet download, feature mining, PRMT3 screening
data/raw/                   Davis, DrugBank, KIBA DTI datasets
data/prmt3/                 PRMT3 candidate ligand set
discovered_fragments/       Seed feature dictionaries (run outputs are git-ignored)
```

## Data

- **DTI benchmarks** — Davis, DrugBank, and KIBA are provided under `data/raw/`
  (`cid,uid,smiles,seq,label`; loaders normalize to `drug,target,label`).
- **MoleculeNet** — downloaded on demand (see below).
- **PRMT3** — `data/prmt3/prmt3_candidates_20.csv` holds the 20 candidate ligands.

## Usage

### Drug–target interaction discovery

```bash
python knowmol_discovery.py \
  --dataset davis \
  --data data/raw/davis_total_cid_unid.csv \
  --output-dir discovered_fragments/runs/davis \
  --rounds 30 \
  --mode both \
  --model deepseek-r1 \
  --validator-backend tabular
```

Use `--dataset {davis,drugbank,kiba}`. `--validator-backend sklearn` runs a
faster RandomForest validator instead of AutoGluon.

### Molecular property prediction (MoleculeNet)

Download and standardize a dataset, then run discovery in molecule-only mode:

```bash
python scripts/download_moleculenet.py --dataset bace

python knowmol_discovery.py \
  --dataset bace \
  --data data/moleculenet/processed/bace_Class.csv \
  --output-dir discovered_fragments/runs/bace \
  --rounds 30 \
  --mode molecule
```

Supported datasets: `bbbp, tox21, toxcast, sider, clintox, muv, hiv, bace`.
Protein agents and protein features are disabled automatically for
molecule-only datasets.

### Downstream training and evaluation

```bash
python scripts/knowmol_downstream.py train \
  --dataset davis --data data/raw/davis_total_cid_unid.csv \
  --model-path discovered_fragments/runs/davis/model

python scripts/knowmol_downstream.py test \
  --dataset davis --data data/raw/davis_total_cid_unid.csv \
  --model-path discovered_fragments/runs/davis/model
```

Feature-group ablations:

```bash
python downstream_ml/ablation_validation.py \
  --dataset davis --data data/raw/davis_total_cid_unid.csv \
  --output-dir discovered_fragments/runs/davis/ablation
```

### Interpretability

Generate SHAP-based explanations at four levels — `global` (all pairs),
`target` (one target vs. all drugs), `drug` (one drug vs. all targets), and
`pair` (a single drug–target pair):

```bash
python interpretability/shap_visualization.py \
  --mode target \
  --dataset davis --data data/raw/davis_total_cid_unid.csv \
  --model-path discovered_fragments/runs/davis/model \
  --target-id ABL1 \
  --output-dir discovered_fragments/runs/davis/shap_ABL1
```

### PRMT3 case study

Train the target-specific model and rank the candidate ligands against the
PRMT3 sequence:

```bash
python scripts/prmt3_screen.py \
  --train-csv path/to/prmt3_training_set.csv \
  --ligands-csv data/prmt3/prmt3_candidates_20.csv \
  --use-default-prmt3-seq \
  --output discovered_fragments/runs/prmt3/scores.csv
```

`--train-csv` is the target-specific training set built for the PRMT3 case
study (`drug,target,label`); pass `--load-model --model-path ...` to score with
a previously saved model.

## License

This project is released under the [BSD 3-Clause License](https://opensource.org/license/BSD-3-Clause).
