"""
Ablation scorers for trust evaluation.

Each scorer here outputs a single scalar per sample (lower = better) that the
main pipeline routes as `faithfulness_margin_z = trust_updated = score`,
`realism_global_z = zeros`. This keeps the rest of the eval layers
(ranking, FPR@95, decile binning, downstream) unchanged.

Implementations:
- Linear probe per attribute, energy score on logits, summed over attrs.
- CLIP text-image alignment (CelebA only, 16 joint-combo prompts).
- (Per-attribute kNN is added in a subsequent step.)
"""

from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from scipy.special import logsumexp
from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import NearestNeighbors

from faithful_cond_gen.eval.trust_eval.condition_utils import get_condition_key
from faithful_cond_gen.eval.trust_eval.scoring_core import normalize_features


def _attr_labels(meta: Dict, attr_key: str) -> np.ndarray:
    v = meta[attr_key]
    if isinstance(v, torch.Tensor):
        return v.cpu().numpy().astype(np.int64)
    return np.asarray(v).astype(np.int64)


def fit_linear_probe_per_attr(
    real_feats: torch.Tensor,
    real_meta: Dict,
    condition_keys: List[str],
) -> Dict[str, LogisticRegression]:
    """Fit one multinomial logistic regression per attribute on L2-normalised real features.

    Binary attributes become 2-class so logits are 2D and energy is well defined.
    Attributes with a single observed class are skipped (returned as None).
    """
    X = normalize_features(real_feats).cpu().numpy().astype(np.float64)
    probes: Dict[str, LogisticRegression] = {}
    for attr_key in condition_keys:
        y = _attr_labels(real_meta, attr_key)
        if len(np.unique(y)) < 2:
            probes[attr_key] = None
            continue
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X, y)
        probes[attr_key] = clf
    return probes


def _logits_2d(clf: LogisticRegression, X: np.ndarray) -> np.ndarray:
    """Return (N, n_classes) logits even when sklearn collapses binary to 1D."""
    z = clf.decision_function(X)
    if z.ndim == 1:
        # Binary: sklearn returns the score for the positive class.
        # Pair it with a zero column for the negative class so logsumexp is well-defined.
        z = np.stack([np.zeros_like(z), z], axis=1)
    return z


def score_linear_probe_energy(
    probes: Dict[str, LogisticRegression],
    gen_feats: torch.Tensor,
    condition_keys: List[str],
) -> np.ndarray:
    """Sum of per-attribute free energies E_a(x) = -logsumexp(logits_a). Lower = better."""
    X = normalize_features(gen_feats).cpu().numpy().astype(np.float64)
    N = X.shape[0]
    score = np.zeros(N, dtype=np.float64)
    for attr_key in condition_keys:
        clf = probes.get(attr_key)
        if clf is None:
            continue
        logits = _logits_2d(clf, X)
        score += -logsumexp(logits, axis=1)
    return score


# ============================================================================
# CLIP text-image alignment (CelebA only)
# ============================================================================

CELEBA_CONDITION_ORDER = ["Male", "Smiling", "Blond_Hair", "Eyeglasses"]

_CELEBA_PROMPT_PARTS = {
    # (attr_name): {value: phrase fragment}
    "Male": {0: "woman", 1: "man"},
    "Smiling": {0: "neutral", 1: "smiling"},
    "Blond_Hair": {0: "with dark hair", 1: "with blond hair"},
    "Eyeglasses": {0: "without eyeglasses", 1: "wearing eyeglasses"},
}


def build_celeba_text_prompts() -> Dict[Tuple[int, int, int, int], str]:
    """Return one prompt per (Male, Smiling, Blond_Hair, Eyeglasses) combo."""
    prompts: Dict[Tuple[int, int, int, int], str] = {}
    for male in (0, 1):
        for smile in (0, 1):
            for blond in (0, 1):
                for glasses in (0, 1):
                    parts = (
                        _CELEBA_PROMPT_PARTS["Smiling"][smile],
                        _CELEBA_PROMPT_PARTS["Male"][male],
                        _CELEBA_PROMPT_PARTS["Blond_Hair"][blond],
                        _CELEBA_PROMPT_PARTS["Eyeglasses"][glasses],
                    )
                    prompts[(male, smile, blond, glasses)] = (
                        f"a photo of a {parts[0]} {parts[1]} {parts[2]} {parts[3]}"
                    )
    return prompts


_CLIP_CACHE: Dict[str, Any] = {}


def fit_clip_alignment(
    model_name: str = "openai/clip-vit-base-patch16",
    device: str = "cpu",
) -> Dict[str, Any]:
    """Load CLIP, encode the 16 CelebA joint-combo prompts, return text+visual-projection."""
    cache_key = f"{model_name}|{device}"
    if cache_key in _CLIP_CACHE:
        return _CLIP_CACHE[cache_key]

    from transformers import CLIPModel, CLIPTokenizer

    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    tokenizer = CLIPTokenizer.from_pretrained(model_name)

    prompts = build_celeba_text_prompts()
    combos = list(prompts.keys())
    texts = [prompts[c] for c in combos]
    inputs = tokenizer(texts, padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        text_emb = model.get_text_features(**inputs)
    text_emb = text_emb / (text_emb.norm(dim=1, keepdim=True) + 1e-12)

    visual_proj = model.visual_projection.weight.detach().cpu().numpy().astype(np.float64)

    stats = {
        "model_name": model_name,
        "combos": combos,
        "text_emb": text_emb.cpu().numpy().astype(np.float64),  # (16, D_joint)
        "visual_proj": visual_proj,                              # (D_joint, D_vision)
    }
    _CLIP_CACHE[cache_key] = stats
    return stats


def score_clip_alignment(
    stats: Dict[str, Any],
    gen_feats: torch.Tensor,
    gen_meta: Dict,
    condition_keys: List[str],
) -> np.ndarray:
    """Score = -cos(image_proj, text_target). Lower = better.

    gen_feats are CLIP vision_model.pooler_output (D_vision); we apply CLIP's
    visual_projection to map them into the joint embedding space.
    """
    img = gen_feats.cpu().numpy().astype(np.float64)
    img_proj = img @ stats["visual_proj"].T  # (N, D_joint)
    img_proj /= np.linalg.norm(img_proj, axis=1, keepdims=True) + 1e-12

    combo_to_row = {c: i for i, c in enumerate(stats["combos"])}
    text_emb = stats["text_emb"]

    N = img_proj.shape[0]
    score = np.zeros(N, dtype=np.float64)
    missing = 0
    for i in range(N):
        cond = get_condition_key(gen_meta, condition_keys, i)
        row = combo_to_row.get(tuple(cond))
        if row is None:
            score[i] = 0.0
            missing += 1
            continue
        score[i] = -float(img_proj[i] @ text_emb[row])
    if missing:
        # Surface unexpected gen conditions (CelebA must be 4-bit binary).
        import logging
        logging.getLogger(__name__).warning(
            "CLIP scorer: %d/%d gen samples had no matching combo prompt", missing, N
        )
    return score


def compute_scores_clip(
    real_used_feats: torch.Tensor,
    real_used_meta: Dict,
    gen_feats: torch.Tensor,
    gen_meta: Dict,
    condition_keys: List[str],
    dataset: str,
    clip_model: str = "openai/clip-vit-base-patch16",
    **_: Any,
) -> Dict[str, Any]:
    """CLIP image-text alignment scorer (CelebA only)."""
    if dataset != "celeba":
        raise ValueError(
            f"clip scorer only supports celeba (got dataset={dataset!r}). "
            "Per-condition text prompts are CelebA-specific."
        )
    stats = fit_clip_alignment(model_name=clip_model)
    score = score_clip_alignment(stats, gen_feats, gen_meta, condition_keys)
    n = score.shape[0]
    return {
        "realism_global_z": np.zeros(n, dtype=np.float64),
        "faithfulness_margin_z": score,
        "trust_updated": score,
        "margin_calib": {},
        "global_stats_summary": {
            "n_samples": int(real_used_feats.shape[0]),
            "scoring_method": "clip",
            "clip_model": clip_model,
        },
    }


# ============================================================================
# Per-attribute kNN (k-th nearest neighbour distance, summed across attrs)
# ============================================================================


def fit_knn_per_attr(
    real_feats: torch.Tensor,
    real_meta: Dict,
    condition_keys: List[str],
    k: int = 5,
) -> Dict[str, Any]:
    """For each attribute, fit a per-class kNN index on L2-normalised real features.

    Returns:
        Dict with:
        - "per_attr": Dict[attr_key, Dict[value, NearestNeighbors]]
        - "global":   Dict[attr_key, NearestNeighbors] for fallback when target value missing
        - "k":        kept for reference (per-class k may shrink if subset is too small)
    """
    X = normalize_features(real_feats).cpu().numpy().astype(np.float64)
    per_attr: Dict[str, Dict[int, Any]] = {}
    global_attr: Dict[str, Any] = {}
    for attr_key in condition_keys:
        labels = _attr_labels(real_meta, attr_key)
        per_attr[attr_key] = {}
        for v in np.unique(labels):
            idx = np.where(labels == int(v))[0]
            n = len(idx)
            k_eff = min(k, max(n - 1, 1))
            nn = NearestNeighbors(n_neighbors=k_eff + 1, metric="cosine", algorithm="auto")
            nn.fit(X[idx])
            per_attr[attr_key][int(v)] = {"nn": nn, "k": k_eff, "n": n}
        # Global fallback uses the full pool so unseen target classes still score.
        nn_global = NearestNeighbors(n_neighbors=k + 1, metric="cosine", algorithm="auto")
        nn_global.fit(X)
        global_attr[attr_key] = {"nn": nn_global, "k": k, "n": X.shape[0]}
    return {"per_attr": per_attr, "global": global_attr, "k": k}


def score_knn_per_attr(
    stats: Dict[str, Any],
    gen_feats: torch.Tensor,
    gen_meta: Dict,
    condition_keys: List[str],
) -> np.ndarray:
    """Sum across attrs of cosine distance to k-th NN within target-class real subset."""
    Xq = normalize_features(gen_feats).cpu().numpy().astype(np.float64)
    N = Xq.shape[0]
    score = np.zeros(N, dtype=np.float64)
    for attr_key in condition_keys:
        labels = _attr_labels(gen_meta, attr_key)
        attr_score = np.zeros(N, dtype=np.float64)
        # Group sample indices by target value to query each NN model in batches.
        for v in np.unique(labels):
            mask = labels == int(v)
            entry = stats["per_attr"].get(attr_key, {}).get(int(v))
            if entry is None:
                entry = stats["global"][attr_key]
            k_eff = entry["k"]
            dists, _ = entry["nn"].kneighbors(Xq[mask], n_neighbors=k_eff + 1)
            # k_eff-th neighbour distance (we asked for k+1 but fitted only on
            # real samples, so query=gen never matches identically — index k_eff-1).
            attr_score[mask] = dists[:, k_eff - 1] if k_eff >= 1 else 0.0
        score += attr_score
    return score


def compute_scores_knn_per_attr(
    real_used_feats: torch.Tensor,
    real_used_meta: Dict,
    gen_feats: torch.Tensor,
    gen_meta: Dict,
    condition_keys: List[str],
    dataset: str,
    knn_k: int = 5,
    **_: Any,
) -> Dict[str, Any]:
    """Per-attribute kNN scorer (k-th NN distance summed across attributes)."""
    stats = fit_knn_per_attr(real_used_feats, real_used_meta, condition_keys, k=knn_k)
    score = score_knn_per_attr(stats, gen_feats, gen_meta, condition_keys)
    n = score.shape[0]
    return {
        "realism_global_z": np.zeros(n, dtype=np.float64),
        "faithfulness_margin_z": score,
        "trust_updated": score,
        "margin_calib": {},
        "global_stats_summary": {
            "n_samples": int(real_used_feats.shape[0]),
            "scoring_method": "knn_per_attr",
            "knn_k": knn_k,
        },
    }


def compute_scores_linear_probe(
    real_used_feats: torch.Tensor,
    real_used_meta: Dict,
    gen_feats: torch.Tensor,
    gen_meta: Dict,
    condition_keys: List[str],
    dataset: str,
    **_: Any,
) -> Dict[str, Any]:
    """Linear-probe energy scorer (registered into _SCORERS)."""
    probes = fit_linear_probe_per_attr(real_used_feats, real_used_meta, condition_keys)
    score = score_linear_probe_energy(probes, gen_feats, condition_keys)
    n = score.shape[0]
    return {
        "realism_global_z": np.zeros(n, dtype=np.float64),
        "faithfulness_margin_z": score,
        "trust_updated": score,
        "margin_calib": {},
        "global_stats_summary": {
            "n_samples": int(real_used_feats.shape[0]),
            "scoring_method": "linear_probe",
            "fitted_attrs": [k for k, v in probes.items() if v is not None],
        },
    }
