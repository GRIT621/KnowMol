# AutoGluon Checkpoints

This directory stores compressed, split AutoGluon checkpoints. The archives are split into parts below GitHub's 100 MB single-file limit.

## Contents

- `drugbank_realwant1/`: DrugBank checkpoint parts for `drugbank_realwant1.tar.gz`
- `davis_realwant1/`: Davis checkpoint parts for `davis_realwant1.tar.gz`

## Restore

From the repository root:

```bash
mkdir -p EviDTI_dataset/AutogluonModels
cat checkpoints/autogluon/drugbank_realwant1/drugbank_realwant1.tar.gz.part-* > /tmp/drugbank_realwant1.tar.gz
tar -xzf /tmp/drugbank_realwant1.tar.gz -C EviDTI_dataset/AutogluonModels

cat checkpoints/autogluon/davis_realwant1/davis_realwant1.tar.gz.part-* > /tmp/davis_realwant1.tar.gz
tar -xzf /tmp/davis_realwant1.tar.gz -C EviDTI_dataset/AutogluonModels
```

## SHA256

```text
61a3c0a2a6c811b14069644a101848b40731e63380ea010d1ee7749ce49b4353  drugbank_realwant1.tar.gz
eead64bf4a65334dcaf984748adfaa5d42c6676ab137b8fde671222f960b9718  davis_realwant1.tar.gz
```
