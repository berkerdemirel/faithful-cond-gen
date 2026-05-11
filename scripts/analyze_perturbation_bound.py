"""
Empirical perturbation-bound test for Cor.~oracle-pooled-decision (App.~A.4).

Per attribute k and dataset (CelebA full, RxRx1 full), we measure how often
the pooled-prototype margin sign agrees with the reference-anchored oracle sign
on real samples, and check whether the Buffered-Pooled-Oracle bound
    B_k^ref(y) = 4 * eps_k^ref * R_k^ref(y) + 2 * (eps_k^ref)^2
is informative (|M_k^ref| > B_k^ref) in practice.

Definitions (matches paper Lemma `pooled-reference-decomposition' and
Proposition `pooled-margin-perturbation'):
    eta_{k,v}    : pooled marginal mean of features over {a_k = v}.
    mu^ref_{k,v} : mean over {a_k = v AND a_{-k} = bar_a_{-k}}, where
                   bar_a_{-k} is the most populous joint of the other attrs.
    gamma_k^ref  : average over v of (eta_{k,v} - mu^ref_{k,v}).
    e_{k,v}^ref  : (eta_{k,v} - mu^ref_{k,v}) - gamma_k^ref.
    eps_k^ref    : max_v ||e_{k,v}^ref||_{P_k}.
    P_k          : LDA-style pooled within-class precision (from fit_factorized_stats).
    M_k(y;t)     : min_{v != t} d_k(y;v) - d_k(y;t),  d_k = ||y - eta_{k,v}||_{P_k}^2.
    M_k^ref(y;t) : same with mu^ref and y_tilde = y - gamma_k^ref.
    R_k^ref(y)   : max_v ||y_tilde - mu^ref_{k,v}||_{P_k}.
    B_k^ref(y)   : 4*eps*R + 2*eps^2.

Outputs (per dataset, per attribute):
    |D_k|, num values, eps_k^ref, median/quantile B_k^ref,
    Pr[|M_k^ref| > B_k^ref], sign-agreement (overall and conditional).

Usage:
    PYTHONPATH=src uv run python scripts/analyze_perturbation_bound.py \
        --datasets celeba rxrx1 --output-dir outputs/perturbation_bound_test
"""
import argparse
import csv
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from faithful_cond_gen.eval.trust_eval.config import (
    CONDITION_ATTRS,
    REAL_FEATURE_PATHS,
)
from faithful_cond_gen.eval.trust_eval.feature_io import (
    apply_normalization,
)
from faithful_cond_gen.eval.trust_eval.scoring_core import (
    fit_factorized_stats,
)
from faithful_cond_gen.eval.trust_eval.subset_io import (
    filter_rxrx1_real_to_scoring_pool,
    load_rxrx1_subset,
)

# SigLIP real features are not registered in REAL_FEATURE_PATHS; map them locally
# so this diagnostic can run in either embedding space without touching config.
SIGLIP_REAL_FEATURE_PATHS: Dict[str, str] = {
    "celeba": "outputs/real_celeba_siglip_meanpatch/train_features.pt",
    "rxrx1": "outputs/real_rxrx1_siglip_meanpatch_full/train_features.pt",
}

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _to_int_array(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.int64)
    return np.asarray(x, dtype=np.int64)


def _mahal_norm(diff: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Vectorised ||x||_P over rows of diff, with P symmetric PSD."""
    return np.sqrt(np.maximum(np.einsum("...i,ij,...j->...", diff, P, diff), 0.0))


def _mahal_chebyshev_center(
    D: np.ndarray, P: np.ndarray, max_iter: int = 5000, tol: float = 1e-12,
) -> Tuple[np.ndarray, float]:
    """Min-max-norm shift under Mahalanobis metric P.

    Returns (gamma_star, eps_star) where
        gamma_star = argmin_g max_v ||D[v] - g||_P
        eps_star   = max_v ||D[v] - gamma_star||_P (= radius of the minimum
                     enclosing Mahalanobis ball of {D[v]}).

    Algorithm: whiten with Cholesky (P = L L^T → u_v = L^T D[v]), then run
    Frank-Wolfe with line search on the SEB dual
        max_w  sum_i w_i ||u_i||^2 - w^T G w,    w in the simplex,
    where G = U U^T. Center c = U^T w; γ = L^{-T} c; eps = sqrt(2 f(w*)).
    For V=2 this returns the midpoint (matches mean), and for V>2 it is
    strictly tighter than the mean unless all D[v] are colinear.
    """
    V = D.shape[0]
    if V == 1:
        return D[0].copy(), 0.0
    L = np.linalg.cholesky(P)            # P = L L^T
    U = D @ L                            # (V, d) whitened: row v is L^T D[v]
    G = U @ U.T                          # (V, V) Gram in whitened space
    norms2 = np.diag(G).copy()           # ||u_i||^2

    # Init: put all weight on the point farthest from origin (Bădoiu-Clarkson).
    w = np.zeros(V)
    w[int(np.argmax(norms2))] = 1.0
    for _ in range(max_iter):
        Gw = G @ w
        # Gradient of f(w) = norms2·w - w^T G w  is  norms2 - 2 G w.
        # On the simplex, ||u_i - c||^2 = norms2_i - 2(Gw)_i + w^T G w (constant in i),
        # so the Frank-Wolfe vertex is the farthest point from current center.
        i_star = int(np.argmax(norms2 - 2.0 * Gw))
        d = -w.copy()
        d[i_star] += 1.0
        a = float((norms2 - 2.0 * Gw) @ d)   # df/dt at t=0 along d
        if a <= tol:
            break
        b = float(d @ G @ d)                  # second-order coeff: f changes as a t - b t^2
        gamma_step = 1.0 if b <= 0 else min(max(a / (2.0 * b), 0.0), 1.0)
        w = w + gamma_step * d

    Gw = G @ w
    radii2 = np.maximum(norms2 - 2.0 * Gw + float(w @ Gw), 0.0)
    eps_star = float(np.sqrt(radii2.max()))
    c_white = U.T @ w                            # (d,) center in whitened space
    gamma_star = np.linalg.solve(L.T, c_white)   # γ such that L^T γ = c_white
    return gamma_star, eps_star


def _per_attribute_perturbation(
    feats_fit: np.ndarray,
    attr_vals_fit: Dict[str, np.ndarray],
    feats_eval: np.ndarray,
    attr_vals_eval: Dict[str, np.ndarray],
    attr_key: str,
    other_keys: List[str],
    eta: Dict[int, np.ndarray],
    P: np.ndarray,
    min_ref_per_value: int = 5,
    min_total_per_value: int = 2,
    gamma_estimator: str = "chebyshev",
    bound_kind: str = "pair",
    forced_reference: Optional[Tuple[int, ...]] = None,
) -> Dict:
    """Run the perturbation-bound diagnostic for one attribute k.

    feats_fit / attr_vals_fit drive estimation of mu^ref and gamma (broader
    scoring pool).  feats_eval / attr_vals_eval are the canonical eval samples
    on which M_k, M_k^ref, and sign-agreement are measured.
    """
    a_k_fit = attr_vals_fit[attr_key]
    if other_keys:
        a_other_fit = np.stack([attr_vals_fit[k] for k in other_keys], axis=1)
    else:
        a_other_fit = np.zeros((len(a_k_fit), 0), dtype=np.int64)

    # Reference context bar_a_{-k}.  If forced_reference is provided, use it
    # (per-attribute projection of a single global bar_a, e.g. (0,0,0,0) for
    # CelebA); otherwise pick the most populous slice from the fitting pool.
    if a_other_fit.shape[1] > 0:
        if forced_reference is not None:
            assert len(forced_reference) == a_other_fit.shape[1], (
                f"forced_reference length {len(forced_reference)} != "
                f"#other-attrs {a_other_fit.shape[1]} for k={attr_key}"
            )
            bar_other = tuple(int(x) for x in forced_reference)
        else:
            keys = [tuple(row) for row in a_other_fit.tolist()]
            bar_other = Counter(keys).most_common(1)[0][0]
        bar_other_arr = np.array(bar_other, dtype=np.int64)
        ref_mask_all = np.all(a_other_fit == bar_other_arr, axis=1)
    else:
        bar_other = ()
        ref_mask_all = np.ones(len(a_k_fit), dtype=bool)

    # mu^ref_{k,v}: conditional mean at the reference context, estimated from
    # the same pool used to compute eta.
    values_all = sorted({int(v) for v in a_k_fit.tolist()})
    values_kept: List[int] = []
    mu_ref: Dict[int, np.ndarray] = {}
    n_ref_per_v: Dict[int, int] = {}
    for v in values_all:
        v_mask = a_k_fit == v
        n_total = int(v_mask.sum())
        ref_mask = v_mask & ref_mask_all
        n_ref = int(ref_mask.sum())
        if n_total < min_total_per_value or n_ref < min_ref_per_value:
            continue
        if v not in eta:
            continue
        values_kept.append(v)
        mu_ref[v] = feats_fit[ref_mask].mean(axis=0)
        n_ref_per_v[v] = n_ref

    if len(values_kept) < 2:
        return {
            "attr": attr_key,
            "ref_context": str(bar_other),
            "n_values_total": len(values_all),
            "n_values_kept": len(values_kept),
            "n_samples": 0,
            "skipped_reason": "insufficient values with reference support",
        }

    # Stack means
    eta_stack = np.stack([eta[v] for v in values_kept], axis=0)        # (V, D)
    mu_ref_stack = np.stack([mu_ref[v] for v in values_kept], axis=0)  # (V, D)

    diff_eta_mu = eta_stack - mu_ref_stack      # (V, D)
    if gamma_estimator == "chebyshev":
        # γ̂ = argmin_γ max_v ||(η_v − μ_v^ref) − γ||_{P_k}: tightest possible ε.
        gamma, eps = _mahal_chebyshev_center(diff_eta_mu, P)
    else:
        gamma = diff_eta_mu.mean(axis=0)
        eps = float(_mahal_norm(diff_eta_mu - gamma[None, :], P).max())
    # Per-value leakage norms ε_v = ‖e_{k,v}^ref‖_{P_k}; the global ε we already have
    # is just max_v of these.  Used by the value-specific bounds below.
    e_kv = diff_eta_mu - gamma[None, :]                 # (V, D)
    eps_per_v = _mahal_norm(e_kv, P)                    # (V,)

    # Inter-prototype reference scale (A1 diagnostic): pairwise
    # ||mu_ref_u - mu_ref_t||_{P_k} for u != t. Ratio eps / typical_gap << 1
    # means A1 approximately holds (leakage is tiny vs. value-signal gap).
    V = len(values_kept)
    if V >= 2:
        diff_pairs = mu_ref_stack[:, None, :] - mu_ref_stack[None, :, :]   # (V,V,D)
        iu = np.triu_indices(V, k=1)
        gap_norms = _mahal_norm(diff_pairs[iu], P)                          # (V*(V-1)/2,)
        proto_gap_min = float(gap_norms.min())
        proto_gap_median = float(np.median(gap_norms))
        ratio_eps_to_min = eps / proto_gap_min if proto_gap_min > 0 else float("nan")
        ratio_eps_to_median = eps / proto_gap_median if proto_gap_median > 0 else float("nan")
    else:
        proto_gap_min = float("nan")
        proto_gap_median = float("nan")
        ratio_eps_to_min = float("nan")
        ratio_eps_to_median = float("nan")

    # Sample-level evaluation: keep eval rows whose true t is in values_kept.
    a_k_eval = attr_vals_eval[attr_key]
    keep_t = np.isin(a_k_eval, values_kept)
    y = feats_eval[keep_t]                       # (N, D)
    t_arr = a_k_eval[keep_t]
    n_samples = len(y)

    if n_samples == 0:
        return {
            "attr": attr_key,
            "ref_context": str(bar_other),
            "n_values_total": len(values_all),
            "n_values_kept": len(values_kept),
            "n_samples": 0,
            "skipped_reason": "no samples for kept values",
        }

    # Map value -> index in values_kept
    v2i = {v: i for i, v in enumerate(values_kept)}
    t_idx = np.array([v2i[int(t)] for t in t_arr.tolist()], dtype=np.int64)

    # tilde_y = y - gamma
    y_tilde = y - gamma[None, :]

    # Kernelised squared Mahalanobis: ||y - mu||_P^2 = yPy - 2 yPmu + muPmu.
    # Avoids the (N, V, D) intermediate.
    Py = y @ P                                  # (N, D)
    Pyt = y_tilde @ P                           # (N, D)
    yPy = np.einsum("ni,ni->n", y, Py)          # (N,)
    ytPyt = np.einsum("ni,ni->n", y_tilde, Pyt) # (N,)
    eta_Peta = np.einsum("vi,vi->v", eta_stack @ P, eta_stack)        # (V,)
    mu_Pmu = np.einsum("vi,vi->v", mu_ref_stack @ P, mu_ref_stack)    # (V,)
    yPe = Py @ eta_stack.T                      # (N, V)
    ytPm = Pyt @ mu_ref_stack.T                 # (N, V)
    d_k = np.maximum(yPy[:, None] - 2.0 * yPe + eta_Peta[None, :], 0.0)
    d_ref = np.maximum(ytPyt[:, None] - 2.0 * ytPm + mu_Pmu[None, :], 0.0)

    # Pairwise diagnostic at the closest pooled alternative.
    # hat_t(y) := argmin_{t != a_k^*(y)} ||y - eta_{k,t}||^2_{P_k}.
    # Both pooled and reference margins use this same (a_k^*, hat_t) pair, so
    # the diagnostic is a direct empirical check of the pairwise corollary
    # (the perturbation bound holds for any single t, and B(y) is unchanged).
    arange_n = np.arange(n_samples)
    d_k_true = d_k[arange_n, t_idx]
    d_ref_true = d_ref[arange_n, t_idx]

    d_k_masked = d_k.copy()
    d_k_masked[arange_n, t_idx] = np.inf
    hatt_idx = d_k_masked.argmin(axis=1)

    M_k = d_k[arange_n, hatt_idx] - d_k_true        # > 0 iff pooled puts y closer to a^*
    M_ref = d_ref[arange_n, hatt_idx] - d_ref_true  # > 0 iff oracle agrees, restricted to {a^*, hat_t}

    # Buffered-bound variants (all proof-compatible, monotone in tightness):
    # Per-value primitive: b_v(y) = 2 ε_v ρ_v(y) + ε_v²,  ρ_v(y) = ‖ỹ−μ_v^ref‖_P.
    # (1) pooled (paper): B = 4ε R + 2ε²,  ε = max_v ε_v,  R = max_v ρ_v.
    # (2) split (global-ε, target/alt asymmetric R):
    #         B = 2ε (ρ_t + max_{v≠t} ρ_v) + 2ε²
    # (3) val   (value-specific, full margin, proof-compatible tightening of paper):
    #         B = b_t(y) + max_{v≠t} b_v(y)
    # (4) pair  (value-specific, fixed competitor t̂):
    #         B = b_t(y) + b_{t̂}(y)
    # The script already fixes t̂ = argmin_{v≠t} d_k(y;v) before comparing pooled vs.
    # reference margins, so for THIS diagnostic the pair bound is the tightest valid
    # certificate; val is the tightest one that bounds the full deployed margin.
    rho = np.sqrt(d_ref)                                  # (N, V)
    R_full = rho.max(axis=1)
    rho_t = rho[arange_n, t_idx]                          # ρ_t = ‖ỹ−μ_t^ref‖_P
    rho_hatt = rho[arange_n, hatt_idx]                    # ρ_t̂
    rho_alt = rho.copy()
    rho_alt[arange_n, t_idx] = -np.inf
    R_alt = np.maximum(rho_alt.max(axis=1), 0.0)          # max_{v≠t} ρ_v

    b_v = 2.0 * eps_per_v[None, :] * rho + eps_per_v[None, :] ** 2   # (N, V)
    b_t = b_v[arange_n, t_idx]
    b_hatt = b_v[arange_n, hatt_idx]
    b_alt = b_v.copy()
    b_alt[arange_n, t_idx] = -np.inf
    b_alt_max = np.maximum(b_alt.max(axis=1), 0.0)        # max_{v≠t} b_v

    B_pooled = 4.0 * eps * R_full + 2.0 * eps * eps
    B_split = 2.0 * eps * (rho_t + R_alt) + 2.0 * eps * eps
    B_val = b_t + b_alt_max
    B_pair = b_t + b_hatt
    B_all = {"pooled": B_pooled, "split": B_split, "val": B_val, "pair": B_pair}
    B = B_all[bound_kind]

    sign_match = np.sign(M_k) == np.sign(M_ref)
    informative_mask = np.abs(M_ref) > B
    overall_agree = float(np.mean(sign_match))
    pr_informative = float(np.mean(informative_mask))
    if informative_mask.any():
        cond_agree = float(np.mean(sign_match[informative_mask]))
    else:
        cond_agree = float("nan")
    n_informative = int(informative_mask.sum())

    # Side-by-side stats for the other variants (always reported for transparency).
    extra_bound_stats: Dict[str, float] = {}
    for name, Bv in B_all.items():
        m = np.abs(M_ref) > Bv
        extra_bound_stats[f"B_{name}_median"] = float(np.median(Bv))
        extra_bound_stats[f"pr_informative_{name}"] = float(np.mean(m))
        extra_bound_stats[f"sign_agreement_informative_{name}"] = (
            float(np.mean(sign_match[m])) if m.any() else float("nan")
        )

    out = {
        "attr": attr_key,
        "ref_context": str(bar_other),
        "ref_keys": "|".join(other_keys),
        "n_values_total": len(values_all),
        "n_values_kept": len(values_kept),
        "n_ref_min": int(min(n_ref_per_v.values())) if n_ref_per_v else 0,
        "n_ref_median": int(np.median(list(n_ref_per_v.values()))) if n_ref_per_v else 0,
        "n_samples": n_samples,
        "eps_k_ref": eps,
        "eps_v_min": float(eps_per_v.min()),
        "eps_v_median": float(np.median(eps_per_v)),
        "eps_v_max": float(eps_per_v.max()),
        "proto_gap_min": proto_gap_min,
        "proto_gap_median": proto_gap_median,
        "ratio_eps_to_min_gap": ratio_eps_to_min,
        "ratio_eps_to_median_gap": ratio_eps_to_median,
        "bound_kind": bound_kind,
        "B_median": float(np.median(B)),
        "B_q25": float(np.quantile(B, 0.25)),
        "B_q75": float(np.quantile(B, 0.75)),
        "B_q95": float(np.quantile(B, 0.95)),
        "M_ref_median_abs": float(np.median(np.abs(M_ref))),
        "M_k_median_abs": float(np.median(np.abs(M_k))),
        "pr_informative": pr_informative,
        "sign_agreement_overall": overall_agree,
        "sign_agreement_informative": cond_agree,
        "n_informative": n_informative,
    }
    out.update(extra_bound_stats)
    return out


def _load_posthoc_real_features(
    dataset: str, model_key: str
) -> Tuple[torch.Tensor, Dict]:
    """Load real raw_hidden at t=0.01 and apply the posthoc mapper for this model.

    Mirrors `_load_posthoc_mapped_features` (real branch only) without the gen
    side, so the diagnostic operates entirely on real samples in the mapper's
    output geometry.
    """
    import json

    from faithful_cond_gen.posthoc_alignment.mapper import ResidualAlignmentMapper
    from faithful_cond_gen.eval.trust_eval.feature_io import l2_normalize_features

    if dataset == "rxrx1" and model_key.endswith("_full_v1"):
        mapper_root = "mappers_whit075"
    else:
        mapper_root = "mappers_whitened"
    mapper_dir = Path(f"outputs/posthoc_alignment/{mapper_root}/{model_key}")
    mapper_path = mapper_dir / "best_mapper.pt"
    if not mapper_path.exists():
        raise FileNotFoundError(f"Mapper not found at {mapper_path}")

    with open(mapper_dir / "training_config.json") as f:
        cfg = json.load(f)["mapper"]
    mapper = ResidualAlignmentMapper(
        int(cfg.get("in_dim", 768)),
        int(cfg.get("out_dim", 1152)),
        hidden_dim=int(cfg.get("hidden_dim", 2048)),
    )
    mapper.load_state_dict(torch.load(mapper_path, map_location="cpu", weights_only=True))
    mapper.eval()

    stats_path = mapper_dir / "preprocessing_stats.pt"
    src_mean = (
        torch.load(stats_path, map_location="cpu", weights_only=False)["src_mean"]
        if stats_path.exists() else None
    )

    raw_path = Path(
        f"outputs/posthoc_alignment/raw_hidden/{model_key}/t0.01_hidden.pt"
    )
    if not raw_path.exists():
        raise FileNotFoundError(f"Real raw_hidden missing: {raw_path}")
    raw = torch.load(raw_path, map_location="cpu", weights_only=False)
    real_hidden = raw["features"]
    real_meta = {k: v for k, v in raw["metadata"].items()}
    logger.info(f"[posthoc] real raw_hidden {tuple(real_hidden.shape)} from {raw_path}")

    if src_mean is not None:
        real_hidden = l2_normalize_features(real_hidden - src_mean)
    with torch.no_grad():
        real_feats = mapper(real_hidden)
    return real_feats, real_meta


def _run_dataset(
    dataset: str,
    output_dir: Path,
    normalize_mode: str = "l2",
    feature_type: str = "dinov3",
    gamma_estimator: str = "chebyshev",
    bound_kind: str = "pair",
    posthoc_model: Optional[str] = None,
    forced_reference: Optional[Tuple[int, ...]] = None,
) -> List[Dict]:
    if feature_type == "posthoc_mapped":
        if posthoc_model is None:
            raise ValueError("--posthoc-model is required for feature_type=posthoc_mapped")
        feats_raw, meta_raw = _load_posthoc_real_features(dataset, posthoc_model)
        real_path = Path(f"posthoc:{posthoc_model}")
    else:
        if feature_type == "siglip":
            real_path = Path(SIGLIP_REAL_FEATURE_PATHS[dataset])
        else:
            real_path = Path(REAL_FEATURE_PATHS[(dataset, feature_type)])
        if dataset == "rxrx1":
            scoring_sibling = real_path.with_name(real_path.stem + "_subset_scoring.pt")
            if scoring_sibling.exists():
                real_path = scoring_sibling
        if not real_path.exists():
            raise FileNotFoundError(f"Real features not found: {real_path}")
        logger.info(f"[{dataset}/{feature_type}] Loading real features from {real_path}")
        data = torch.load(real_path, map_location="cpu", weights_only=False)
        feats_raw = data["features"]
        meta_raw = data.get("metadata", {})

    # Fit pool: matches the main pipeline (sirna-column scoring pool for rxrx1,
    # full real for celeba).
    if dataset == "rxrx1":
        feats_fit_t, meta_fit = filter_rxrx1_real_to_scoring_pool(feats_raw, meta_raw)
    else:
        feats_fit_t, meta_fit = feats_raw, meta_raw

    feats_fit_t = apply_normalization(feats_fit_t, normalize_mode, f"fit_{dataset}_{feature_type}")
    feats_fit = feats_fit_t.detach().cpu().numpy().astype(np.float64)
    cond_keys = CONDITION_ATTRS[dataset]
    attr_vals_fit = {k: _to_int_array(meta_fit[k]) for k in cond_keys}
    logger.info(f"[{dataset}] fit pool N={feats_fit.shape[0]}, D={feats_fit.shape[1]}")

    # Eval pool: canonical eval set (50-pair subset for rxrx1, all real for celeba).
    if dataset == "rxrx1":
        subset = load_rxrx1_subset()
        ct = _to_int_array(meta_fit["cell_type_id"])
        sr = _to_int_array(meta_fit["sirna_id"])
        eval_mask = np.array([(int(a), int(b)) in subset for a, b in zip(ct, sr)])
        feats_eval = feats_fit[eval_mask]
        attr_vals_eval = {k: v[eval_mask] for k, v in attr_vals_fit.items()}
    else:
        feats_eval = feats_fit
        attr_vals_eval = attr_vals_fit
    logger.info(f"[{dataset}] eval pool N={feats_eval.shape[0]}")

    # Fit factorized stats once on the fit pool (pooled means and shared P_k).
    stats = fit_factorized_stats(feats_fit_t, meta_fit, cond_keys, use_shared_cov=True)

    rows = []
    for k in cond_keys:
        eta = {int(v): val_stats["mu"].cpu().numpy().astype(np.float64)
               for v, val_stats in stats[k].items()}
        P = stats[k][next(iter(stats[k]))]["precision"].cpu().numpy().astype(np.float64)
        other_keys = [kk for kk in cond_keys if kk != k]
        logger.info(f"[{dataset}] Attribute '{k}': fitting μ^ref over {len(other_keys)} other attrs ...")
        if forced_reference is not None:
            assert len(forced_reference) == len(cond_keys), (
                f"--reference must have {len(cond_keys)} entries for {dataset} "
                f"(got {len(forced_reference)})"
            )
            forced_for_k = tuple(
                int(forced_reference[cond_keys.index(kk)]) for kk in other_keys
            )
        else:
            forced_for_k = None
        res = _per_attribute_perturbation(
            feats_fit, attr_vals_fit,
            feats_eval, attr_vals_eval,
            k, other_keys, eta, P,
            min_ref_per_value=(5 if dataset == "celeba" else 3),
            min_total_per_value=2,
            gamma_estimator=gamma_estimator,
            bound_kind=bound_kind,
            forced_reference=forced_for_k,
        )
        res["dataset"] = dataset
        rows.append(res)
        logger.info(
            f"[{dataset}/{k}] V={res.get('n_values_kept')}/{res.get('n_values_total')}, "
            f"|D|={res.get('n_samples')}, eps={res.get('eps_k_ref', float('nan')):.4f}, "
            f"medB[{bound_kind}]={res.get('B_median', float('nan')):.4f}, "
            f"Pr[|Mref|>B] pooled/split/val/pair="
            f"{res.get('pr_informative_pooled', float('nan')):.3f}/"
            f"{res.get('pr_informative_split', float('nan')):.3f}/"
            f"{res.get('pr_informative_val', float('nan')):.3f}/"
            f"{res.get('pr_informative_pair', float('nan')):.3f}, "
            f"agree(all)={res.get('sign_agreement_overall', float('nan')):.3f}"
        )

    # Save CSV per dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if feature_type == "dinov3" else f"_{feature_type}"
    csv_path = output_dir / f"perturbation_bound_{dataset}{suffix}.csv"
    cols = [
        "dataset", "attr", "ref_keys", "ref_context",
        "n_values_total", "n_values_kept", "n_ref_min", "n_ref_median",
        "n_samples", "eps_k_ref",
        "proto_gap_min", "proto_gap_median",
        "ratio_eps_to_min_gap", "ratio_eps_to_median_gap",
        "bound_kind",
        "B_median", "B_q25", "B_q75", "B_q95",
        "M_ref_median_abs", "M_k_median_abs",
        "pr_informative", "n_informative",
        "sign_agreement_overall", "sign_agreement_informative",
        "B_pooled_median", "B_split_median", "B_val_median", "B_pair_median",
        "pr_informative_pooled", "pr_informative_split",
        "pr_informative_val", "pr_informative_pair",
        "sign_agreement_informative_pooled",
        "sign_agreement_informative_split",
        "sign_agreement_informative_val",
        "sign_agreement_informative_pair",
        "eps_v_min", "eps_v_median", "eps_v_max",
        "skipped_reason",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    logger.info(f"[{dataset}] Saved {csv_path}")
    return rows


def _write_markdown_summary(all_rows: List[Dict], output_dir: Path, feature_type: str = "dinov3"):
    md = []
    md.append("# Perturbation-bound diagnostic\n")
    md.append("Empirical validation of Cor.~`oracle-pooled-decision`.\n\n")
    md.append("Bound variants (all proof-compatible, monotone in tightness):\n"
              "  - **pooled** (paper): 4εR + 2ε², ε = max_v ε_v, R = max_v ρ_v.\n"
              "  - **split**: 2ε(ρ_t + max_{v≠t} ρ_v) + 2ε² (target/alt asymmetry, global ε).\n"
              "  - **val**:  b_t + max_{v≠t} b_v with b_v = 2ε_v ρ_v + ε_v² "
              "(value-specific, full-margin tightening).\n"
              "  - **pair**: b_t + b_t̂ (value-specific, fixed-competitor — tightest valid "
              "bound for this script's diagnostic).\n\n")
    md.append("| Dataset | Attribute | V | |D| | ε(max) | "
              "med B(pooled) | med B(split) | med B(val) | med B(pair) | "
              "Pr[cert] pooled | split | val | pair | "
              "Agree(all) | Agree(cert) pooled | split | val | pair |\n")
    md.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")

    def _fnum(x, fmt=".3f"):
        try:
            return format(float(x), fmt)
        except (TypeError, ValueError):
            return "—"

    for r in all_rows:
        if r.get("skipped_reason"):
            md.append(f"| {r['dataset']} | {r['attr']} | "
                      f"{r['n_values_kept']}/{r['n_values_total']} | "
                      "0 | — | — | — | — | — | — | — | — | — | — | — | — | — | — |\n")
            continue
        md.append(
            f"| {r['dataset']} | {r['attr']} | "
            f"{r['n_values_kept']}/{r['n_values_total']} | "
            f"{r['n_samples']} | {_fnum(r['eps_k_ref'], '.4f')} | "
            f"{_fnum(r['B_pooled_median'], '.2f')} | {_fnum(r['B_split_median'], '.2f')} | "
            f"{_fnum(r['B_val_median'], '.2f')} | {_fnum(r['B_pair_median'], '.2f')} | "
            f"{_fnum(r['pr_informative_pooled'])} | {_fnum(r['pr_informative_split'])} | "
            f"{_fnum(r['pr_informative_val'])} | {_fnum(r['pr_informative_pair'])} | "
            f"{_fnum(r['sign_agreement_overall'])} | "
            f"{_fnum(r['sign_agreement_informative_pooled'])} | "
            f"{_fnum(r['sign_agreement_informative_split'])} | "
            f"{_fnum(r['sign_agreement_informative_val'])} | "
            f"{_fnum(r['sign_agreement_informative_pair'])} |\n"
        )

    md.append("\n\n# A1 context-balance diagnostic\n")
    md.append("Operational test for assumption A1: under A1+A2 the leakage e_{k,v}^ref "
              "is the only v-dependent term in (η_{k,v} − μ^ref_{k,v}); we therefore "
              "compare ε = max_v ‖e_{k,v}^ref‖_{P_k} to the typical inter-prototype "
              "scale ‖μ^ref_u − μ^ref_t‖_{P_k}. ratio = ε / gap; "
              "ratio « 1 means A1 approximately holds.\n\n")
    md.append("| Dataset | Attribute | V | ε | min ‖μ_u−μ_t‖_P | median ‖μ_u−μ_t‖_P | ε/min | ε/median |\n")
    md.append("|---|---|---:|---:|---:|---:|---:|---:|\n")
    for r in all_rows:
        if r.get("skipped_reason"):
            md.append(f"| {r['dataset']} | {r['attr']} | {r['n_values_kept']} | — | — | — | — | — |\n")
            continue
        md.append(
            f"| {r['dataset']} | {r['attr']} | {r['n_values_kept']} | "
            f"{r['eps_k_ref']:.4f} | "
            f"{r['proto_gap_min']:.4f} | {r['proto_gap_median']:.4f} | "
            f"{r['ratio_eps_to_min_gap']:.3f} | {r['ratio_eps_to_median_gap']:.3f} |\n"
        )

    suffix = "" if feature_type == "dinov3" else f"_{feature_type}"
    out = output_dir / f"perturbation_bound_summary{suffix}.md"
    out.write_text("".join(md))
    logger.info(f"Markdown summary: {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["celeba", "rxrx1"])
    p.add_argument("--output-dir", default="outputs/perturbation_bound_test")
    p.add_argument("--normalize", default="l2", choices=["none", "l2"])
    p.add_argument("--feature-type", default="dinov3",
                   choices=["dinov3", "siglip", "posthoc_mapped"])
    p.add_argument("--posthoc-model", default=None,
                   help="Posthoc model_key (e.g. celeba_vanilla_marginal_v1) "
                        "required when --feature-type=posthoc_mapped.")
    p.add_argument("--reference", default=None,
                   help="Comma-separated reference context bar_a in CONDITION_ATTRS "
                        "order (e.g. '0,0,0,0' for celeba). If unset, the most populous "
                        "slice is used per attribute (matches the paper's appendix).")
    p.add_argument("--gamma-estimator", default="chebyshev",
                   choices=["mean", "chebyshev"],
                   help="Estimator for the common shift γ. 'chebyshev' minimises "
                        "max_v ||(η_v−μ_v^ref)−γ||_{P_k} (tighter ε); 'mean' uses "
                        "the average residual (looser, but identical for V=2).")
    p.add_argument("--bound", default="pair",
                   choices=["pooled", "split", "val", "pair"],
                   help="Buffered bound variant for the headline B (all four are "
                        "always reported in the CSV/Markdown). pooled = paper's "
                        "4εR+2ε²; split = global-ε target/alt asymmetric; "
                        "val = value-specific full-margin (b_t + max_{v≠t} b_v); "
                        "pair = value-specific fixed-competitor (b_t + b_t̂).")
    args = p.parse_args()

    forced_reference = (
        tuple(int(x) for x in args.reference.split(","))
        if args.reference is not None else None
    )

    output_dir = Path(args.output_dir)
    all_rows: List[Dict] = []
    for d in args.datasets:
        rows = _run_dataset(d, output_dir, normalize_mode=args.normalize,
                            feature_type=args.feature_type,
                            gamma_estimator=args.gamma_estimator,
                            bound_kind=args.bound,
                            posthoc_model=args.posthoc_model,
                            forced_reference=forced_reference)
        all_rows.extend(rows)
    _write_markdown_summary(all_rows, output_dir, feature_type=args.feature_type)


if __name__ == "__main__":
    main()
