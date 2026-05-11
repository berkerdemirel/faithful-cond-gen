#!/usr/bin/env bash
# Train 6 celeba mappers: whit_geom loss, NO center+norm, SigLIP target.
set -euo pipefail

OUT=outputs/posthoc_alignment/mappers_whit_geom_nocn
SIG=outputs/real_celeba_siglip_meanpatch/train_features.pt
COMMON_ARGS=(output_dir="$OUT" siglip_path="$SIG" "+loss.kind=whit_geom" +loss.lambda_whit=1.0 +loss.lambda_geom=0.1 +loss.whit_gamma=0.75 +loss.whit_cond=1000.0)

MODELS="celeba_vanilla_full_v1 celeba_vanilla_marginal_v1 celeba_repa_full_v1 celeba_repa_marginal_v1 celeba_repa_siglip_full_v1 celeba_repa_siglip_marginal_v1"

gpu=0
pids=()

for m in $MODELS; do
  echo "[GPU $gpu] Training $m (whit_geom, no CN)"
  CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=src uv run python scripts/posthoc_alignment/train_mapper.py \
    model_key=$m \
    hidden_dir=outputs/posthoc_alignment/raw_hidden/$m \
    "${COMMON_ARGS[@]}" &
  pids+=($!)
  gpu=$((gpu + 1))
done

echo "Waiting for ${#pids[@]} training jobs..."
for pid in "${pids[@]}"; do
  wait "$pid"
done
echo "All done."
