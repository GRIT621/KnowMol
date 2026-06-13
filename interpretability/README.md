# KnowMol Interpretability

This folder organizes the four visualization settings used in the manuscript:

- `global`: all drugs against all targets, corresponding to `all_all`.
- `target`: one fixed target against all drugs, corresponding to `all_target`.
- `drug`: one fixed drug against all targets, corresponding to `drug_all`.
- `pair`: one drug-target pair, corresponding to `one_by_one`.

All modes reuse the same KnowMol feature extraction code as downstream
validation. Pass the trained AutoGluon model path and the dataset file at run
time.

```bash
python interpretability/shap_visualization.py \
  --mode global \
  --dataset davis \
  --data data/raw/davis_total_cid_unid.csv \
  --model-path /path/to/AutogluonModels/davis_realwant1 \
  --output-dir outputs/interpretability/davis_global
```

```bash
python interpretability/shap_visualization.py \
  --mode target \
  --dataset davis \
  --data data/raw/davis_total_cid_unid.csv \
  --model-path /path/to/AutogluonModels/davis_realwant1 \
  --target-id ABL1 \
  --output-dir outputs/interpretability/davis_ABL1
```

```bash
python interpretability/shap_visualization.py \
  --mode drug \
  --dataset davis \
  --data data/raw/davis_total_cid_unid.csv \
  --model-path /path/to/AutogluonModels/davis_realwant1 \
  --drug-id 44259 \
  --output-dir outputs/interpretability/davis_44259
```

```bash
python interpretability/shap_visualization.py \
  --mode pair \
  --dataset davis \
  --data data/raw/davis_total_cid_unid.csv \
  --model-path /path/to/AutogluonModels/davis_realwant1 \
  --drug-id 44259 \
  --target-id ABL1 \
  --output-dir outputs/interpretability/davis_44259_ABL1
```

Use `--summary-view drug`, `--summary-view protein`, or `--summary-view all`
to control which feature group is shown in the SHAP summary plot. By default,
`target` focuses on drug-side features, `drug` focuses on protein-side features,
and `global`/`pair` use all features.
