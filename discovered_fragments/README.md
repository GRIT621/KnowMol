# Discovered Fragments

This directory is the default location for newly mined KnowMol molecular
substructures and protein fragments.

The repository starts from the public seed dictionaries in:

- `downstream_ml/drug_dict.txt`
- `downstream_ml/protein_dict.txt`

Discovery runs write candidate dictionaries, consolidated dictionaries, memory,
metrics, and per-round models under `discovered_fragments/runs/`. Generated run
artifacts are ignored by Git; this README is kept so the directory purpose is
clear.
