# Downstream Machine-Learning Validation

This module keeps the original KnowMol downstream validation logic, but exposes
three explicit modes:

```bash
python scripts/knowmol_downstream.py train --dataset davis --data ../EviDTI_dataset/data/splits/davis --model-path models/davis
python scripts/knowmol_downstream.py test --dataset davis --data ../EviDTI_dataset/data/splits/davis --model-path models/davis
python scripts/knowmol_downstream.py prmt3 --model-path models/drugbank --ligands-csv ../EviDTI_dataset/data/case_studies/prmt3/PRMT3_drugbank_Matches.csv --output outputs/prmt3_scores.csv
```

Datasets are selected with `--dataset davis`, `--dataset drugbank`, or
`--dataset kiba`. `--data` may be a raw CSV or a directory containing
`train.csv`, `valid.csv`, and `test.csv`.

The test mode reports AUC-ROC, AUC-PR, Accuracy, Precision, Recall,
Specificity, F1, and MCC, then prints the top five high-confidence bad cases.

KIBA uses the Morgan fingerprint profile from the best saved model:
`logP`, `TPSA`, `MolWt`, and `fp_0` through `fp_1023`. Davis and DrugBank keep
the original compact molecular descriptor profile so their saved models remain
reproducible.

## Raw-AutoGluon Baseline

Raw-AutoGluon trains directly on the original `drug` and `target` columns,
without RDKit descriptors, protein physicochemical features, substructure
features, or KnowMol feature mining:

```bash
python scripts/raw_autogluon.py train \
  --dataset davis \
  --data ../EviDTI_dataset/data/splits/davis \
  --model-path models/raw_autogluon/davis \
  --evaluate-after-train
```

Evaluate a saved Raw-AutoGluon model with:

```bash
python scripts/raw_autogluon.py test \
  --dataset davis \
  --data ../EviDTI_dataset/data/splits/davis \
  --model-path models/raw_autogluon/davis
```

## Traditional-Feature Baseline

Traditional-AutoGluon uses standard handcrafted features: molecular Morgan
fingerprint bits and protein amino-acid composition. It does not use molecular
descriptors, protein physicochemical descriptors, KnowMol substructure
patterns, or protein binding fragments.

```bash
python scripts/traditional_autogluon.py train \
  --dataset davis \
  --data ../EviDTI_dataset/data/splits/davis \
  --model-path models/traditional_autogluon/davis \
  --presets medium_quality \
  --evaluate-after-train
```

Evaluate a saved Traditional-AutoGluon model with:

```bash
python scripts/traditional_autogluon.py test \
  --dataset davis \
  --data ../EviDTI_dataset/data/splits/davis \
  --model-path models/traditional_autogluon/davis
```

## Ablation Experiments

Run feature-group ablations with:

```bash
python ablation_validation.py \
  --dataset davis \
  --data /data/lsj/KnowMol/EviDTI_dataset/davis_total_cid_unid.csv \
  --output-dir /data/lsj/KnowMol/EviDTI_dataset/AutogluonModels/ablations/davis
```

By default this trains and evaluates:

- `none`: full feature set
- `no_protein_basic`: remove protein physicochemical and AAC features
- `no_protein_fragment`: remove protein structural fragment features
- `no_molecule_basic`: remove molecular descriptor/fingerprint features
- `no_molecule_fragment`: remove molecular substructure fragment features

To run only selected ablations:

```bash
python ablation_validation.py \
  --dataset kiba \
  --data /data/lsj/KnowMol/EviDTI_dataset/KIBA_total_cid_unid.csv \
  --output-dir /data/lsj/KnowMol/EviDTI_dataset/AutogluonModels/ablations/kiba \
  --ablations no_protein_basic no_molecule_fragment
```

Each run saves an AutoGluon model under the output directory and writes
`ablation_summary.csv`.

Feature vocabularies are loaded from a Python file defining
`substructure_patterns` and `binding_fragments`. The default is
`/data/lsj/KnowMol/EviDTI_dataset/knowmol_36.py`. You can override it with
`--vocab-py` when running on another machine.

## KnowMol Agent Feature Mining

The KnowMol fragment features come from the LLM agents now included under
`agents/`:

- `agents/molecular_agent.py` proposes RDKit SMARTS molecular substructures.
- `agents/protein_agent.py` proposes protein sequence fragments.
- `agents/base_agent.py` wraps the OpenAI-compatible chat API and tracks token
  usage.

For the protein-target DTI setting, use the iterative discovery entrypoint. It
is the `KnowMol_GitHub` version of the old `bace_mol.py` loop: sample
`drug,target,label` records, ask the molecular agent for RDKit SMARTS, ask the
protein agent for target-sequence fragments, train a downstream model, then feed
high-error bad cases into the next round.

```bash
python scripts/knowmol_discovery.py \
  --dataset davis \
  --data ../EviDTI_dataset/data/splits/davis \
  --output-dir outputs/discovery/davis \
  --rounds 3 \
  --sample-size 10 \
  --model pro-deepseek-r1 \
  --api-base "$OPENAI_BASE_URL" \
  --api-key "$OPENAI_API_KEY"
```

Each round writes `knowmol_vocab_round_XXX.py`, trains an AutoGluon model under
`outputs/discovery/davis/models/`, and saves badcase-guided metrics in
`discovery_summary_*.csv`. The final downstream vocabulary is:

```bash
outputs/discovery/davis/knowmol_vocab_final.py
```

Then run the regular downstream validation with that vocabulary:

```bash
python scripts/knowmol_downstream.py train \
  --dataset davis \
  --data ../EviDTI_dataset/data/splits/davis \
  --model-path models/davis_agent_vocab \
  --vocab-py outputs/discovery/davis/knowmol_vocab_final.py \
  --evaluate-after-train
```

For a one-shot vocabulary without the bace-style feedback loop, use the smaller
feature-mining helper:

```bash
python scripts/mine_knowmol_features.py \
  --dataset davis \
  --data ../EviDTI_dataset/data/splits/davis \
  --output outputs/davis_agent_vocab.py \
  --model pro-deepseek-r1 \
  --api-base "$OPENAI_BASE_URL" \
  --api-key "$OPENAI_API_KEY"
```

The generated file defines:

- `substructure_patterns`: molecule SMARTS keys used by
  `substructure_features()`.
- `binding_fragments`: protein fragment keys used by
  `protein_substructure_features()`.

To continue mining without repeating existing fragments, pass the current
vocabulary as an exclusion list:

```bash
python scripts/mine_knowmol_features.py \
  --dataset davis \
  --data ../EviDTI_dataset/data/splits/davis \
  --existing-vocab outputs/davis_agent_vocab.py \
  --output outputs/davis_agent_vocab_round2.py
```
