#!/usr/bin/env bash
# Train 6 rxrx1 mappers: whit_geom loss + center+norm, γ=0.75 SigLIP target.
set -euo pipefail

OUT=outputs/posthoc_alignment/mappers_whit_geom
COMMON_ARGS=(output_dir="$OUT" "+loss.kind=whit_geom" +loss.lambda_whit=1.0 +loss.lambda_geom=0.1 +loss.whit_gamma=0.75 +loss.whit_cond=1000.0 "+condition_keys=[cell_type_id,sirna_id]" +center_norm=true)

MODELS_FULL="rxrx1_vanilla_full_v1 rxrx1_repa_full_v1 rxrx1_repa_siglip_full_v1"
MODELS_MARG="rxrx1_vanilla_marginal_v1 rxrx1_repa_marginal_v1 rxrx1_repa_siglip_marginal_v1"

SIG_FULL=outputs/real_rxrx1_siglip_meanpatch_full/train_features.pt
SIG_MARG=outputs/real_rxrx1_siglip_meanpatch/train_features.pt

gpu=0
pids=()

for m in $MODELS_FULL; do
  echo "[GPU $gpu] Training $m (full, whit_geom)"
  CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=src uv run python scripts/posthoc_alignment/train_mapper.py \
    model_key=$m \
    hidden_dir=outputs/posthoc_alignment/raw_hidden/$m \
    siglip_path=$SIG_FULL \
    "${COMMON_ARGS[@]}" &
  pids+=($!)
  gpu=$((gpu + 1))
done

for m in $MODELS_MARG; do
  echo "[GPU $gpu] Training $m (marginal, whit_geom)"
  CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=src uv run python scripts/posthoc_alignment/train_mapper.py \
    model_key=$m \
    hidden_dir=outputs/posthoc_alignment/raw_hidden/$m \
    siglip_path=$SIG_MARG \
    "${COMMON_ARGS[@]}" &
  pids+=($!)
  gpu=$((gpu + 1))
done

echo "Waiting for ${#pids[@]} training jobs..."
for pid in "${pids[@]}"; do
  wait "$pid"
done
echo "All done."
