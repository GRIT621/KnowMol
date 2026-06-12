#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
out_dir="$root_dir/EviDTI_dataset/AutogluonModels"
mkdir -p "$out_dir"

drugbank_archive="${TMPDIR:-/tmp}/drugbank_realwant1.tar.gz"
davis_archive="${TMPDIR:-/tmp}/davis_realwant1.tar.gz"

cat "$root_dir"/checkpoints/autogluon/drugbank_realwant1/drugbank_realwant1.tar.gz.part-* > "$drugbank_archive"
echo "61a3c0a2a6c811b14069644a101848b40731e63380ea010d1ee7749ce49b4353  $drugbank_archive" | sha256sum -c -
tar -xzf "$drugbank_archive" -C "$out_dir"

cat "$root_dir"/checkpoints/autogluon/davis_realwant1/davis_realwant1.tar.gz.part-* > "$davis_archive"
echo "eead64bf4a65334dcaf984748adfaa5d42c6676ab137b8fde671222f960b9718  $davis_archive" | sha256sum -c -
tar -xzf "$davis_archive" -C "$out_dir"
