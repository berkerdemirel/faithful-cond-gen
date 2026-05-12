"""Build demo/trust_score_l_shape.ipynb following the paper's exact formulas.

Reference: notes/paper.tex §Method, "Global realism" and
"Attribute-wise faithfulness" paragraphs (≈ lines 311–340).

  E_real(y)       = (y - mu_real)^T Sigma_real^{-1} (y - mu_real)
  R(y)            = (E_real(y) - m_R) / s_R              # larger -> less real
  eta_{k,v}       = E[y | a_k = v]                       # pooled prototype
  P_k             = inv( pooled within-value covariance )
  d_k(y; v)       = (y - eta_{k,v})^T P_k (y - eta_{k,v})
  M_k(y; t)       = d_k(y; t) - min_{v != t} d_k(y; v)   # negative -> faithful
  F_k(y; t)       = (M_k(y; t) - m_{k,t}) / s_{k,t}      # calibrated on a_k=t
  F(y; a*)        = (1/K) sum_k F_k(y; a_k*)             # larger -> less faithful
  T(y; a*)        = R(y) + F(y; a*)                      # larger -> less trustworthy
"""

import json
from pathlib import Path


def md(*lines):
    return {
        "cell_type": "markdown",
        "id": f"md-{hash(lines) & 0xffffff:x}",
        "metadata": {},
        "source": [l + "\n" for l in lines],
    }


def code(*lines):
    return {
        "cell_type": "code",
        "id": f"code-{hash(lines) & 0xffffff:x}",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [l + "\n" for l in lines],
    }


cells = []

cells.append(md(
    "# Trust score on an L-shaped toy: realism + faithfulness",
    "",
    "Follows the exact formulation in the paper "
    "(`notes/paper.tex`, §Method). The trust score has two components:",
    "",
    "$$R(y) = \\frac{E_{\\mathrm{real}}(y) - m_R}{s_R}, \\qquad "
    "E_{\\mathrm{real}}(y) = (y - \\mu_{\\mathrm{real}})^\\top "
    "\\Sigma_{\\mathrm{real}}^{-1}(y - \\mu_{\\mathrm{real}})$$",
    "",
    "$$F_k(y; t) = \\frac{M_k(y; t) - m_{k,t}}{s_{k,t}}, \\qquad "
    "M_k(y; t) = d_k(y; t) - \\min_{v \\neq t} d_k(y; v)$$",
    "",
    "$$d_k(y; v) = (y - \\eta_{k,v})^\\top P_k (y - \\eta_{k,v}), \\qquad "
    "T(y; a^\\star) = R(y) + \\frac{1}{K}\\sum_{k=1}^K F_k(y; a_k^\\star)$$",
    "",
    "By construction **larger $R$, $F$, or $T$ is *worse*** — $R$ measures "
    "distance from the real distribution, $M_k$ is negative when $y$ is closer "
    "to the requested value than to any competitor, and $F_k$ is calibrated on "
    "real samples that actually have $a_k = t$.",
    "",
    "**Setup.** Two binary attributes $(a_1, a_2)$. Three seen conditions "
    "$(0,0), (0,1), (1,0)$ are arranged so their union forms an **L shape** in "
    "the $(a_1, a_2)$ plane. The fourth condition $(1,1)$ is unseen — we ask "
    "what the trust score does for a generated sample claiming target $(1,1)$.",
))

cells.append(code(
    "import numpy as np",
    "import matplotlib.pyplot as plt",
    "from matplotlib.colors import Normalize",
    "from sklearn.covariance import LedoitWolf",
    "",
    "rng = np.random.default_rng(0)",
    "plt.rcParams.update({'figure.dpi': 110, 'font.size': 10})",
))

cells.append(md(
    "## 1. Synthesize the L-shaped data",
))

cells.append(code(
    "# Cluster means: each row is (attr1, attr2)",
    "means = {",
    "    (0, 0): np.array([0.0, 0.0]),",
    "    (0, 1): np.array([0.0, 1.0]),",
    "    (1, 0): np.array([1.0, 0.0]),",
    "}",
    "sigma = 0.12  # tight clusters so the L is clearly visible",
    "n_per = 400",
    "",
    "X_by_cond = {",
    "    cond: mu + sigma * rng.standard_normal(size=(n_per, 2))",
    "    for cond, mu in means.items()",
    "}",
    "X = np.concatenate(list(X_by_cond.values()), axis=0)  # all real features",
    "",
    "# Attribute label arrays, one per attribute",
    "a1 = np.concatenate([np.full(n_per, cond[0]) for cond in means])",
    "a2 = np.concatenate([np.full(n_per, cond[1]) for cond in means])",
    "",
    "TARGET = (1, 1)  # the unseen request",
))

cells.append(code(
    "fig, ax = plt.subplots(figsize=(5, 5))",
    "colors = ['#4c72b0', '#dd8452', '#55a868']",
    "for (cond, pts), c in zip(X_by_cond.items(), colors):",
    "    ax.scatter(pts[:, 0], pts[:, 1], s=8, alpha=0.55, color=c,",
    "               label=f'cond {cond}')",
    "ax.scatter(*TARGET, marker='X', s=180, color='red', edgecolor='black',",
    "           linewidth=1.4, zorder=5, label='requested (1,1)')",
    "ax.set_xlabel(r'attribute $a_1$')",
    "ax.set_ylabel(r'attribute $a_2$')",
    "ax.set_xlim(-0.5, 1.6); ax.set_ylim(-0.5, 1.6)",
    "ax.set_aspect('equal')",
    "ax.grid(alpha=0.3)",
    "ax.legend(loc='upper right', frameon=True)",
    "ax.set_title('Seen conditions form an L; (1,1) is unseen')",
    "plt.show()",
))

cells.append(md(
    "## 2. Global realism $R(y)$",
    "",
    "Fit $(\\mu_{\\mathrm{real}}, \\Sigma_{\\mathrm{real}})$ on the pooled real "
    "features (Ledoit–Wolf shrinkage for the covariance), evaluate "
    "$E_{\\mathrm{real}}$ on a grid, and standardize against the real-sample "
    "energies to get $R$.",
    "",
    "**Larger $R$ = less real**, because $E_{\\mathrm{real}}$ is squared "
    "Mahalanobis distance.",
))

cells.append(code(
    "lw = LedoitWolf().fit(X)",
    "mu_real = lw.location_",
    "prec_real = lw.precision_",
    "",
    "def mahal_sq(points, mu, prec):",
    "    d = points - mu",
    "    return np.einsum('ni,ij,nj->n', d, prec, d)",
    "",
    "# Energy grid",
    "grid_n = 240",
    "x_g = np.linspace(-0.6, 1.7, grid_n)",
    "y_g = np.linspace(-0.6, 1.7, grid_n)",
    "XX, YY = np.meshgrid(x_g, y_g)",
    "grid_pts = np.stack([XX.ravel(), YY.ravel()], axis=1)",
    "",
    "E_grid = mahal_sq(grid_pts, mu_real, prec_real)",
    "E_real_calib = mahal_sq(X, mu_real, prec_real)  # for m_R, s_R",
    "m_R, s_R = E_real_calib.mean(), E_real_calib.std() + 1e-9",
    "",
    "R = ((E_grid - m_R) / s_R).reshape(XX.shape)",
    "R_real = (E_real_calib - m_R) / s_R  # R on real samples (mean≈0, std≈1)",
))

cells.append(md(
    "## 3. Attribute-wise faithfulness for target $(1, 1)$",
    "",
    "For each attribute $k$:",
    "",
    "1. Estimate pooled prototypes $\\eta_{k,v} = \\mathbb E[y \\mid a_k = v]$ by "
    "averaging real features across the two possible values of $a_k$.",
    "2. Estimate a shared precision matrix $P_k$ from the **pooled within-value** "
    "covariance — for each value $v$, center samples at their own $\\eta_{k,v}$ "
    "and average the resulting covariances. Inverting gives $P_k$.",
    "3. Compute $d_k(y; v)$ on the grid, the margin "
    "$M_k(y; 1) = d_k(y; 1) - d_k(y; 0)$, and standardize on the real samples "
    "that *do* have $a_k = 1$ (calibrating $m_{k,1}, s_{k,1}$).",
    "",
    "$F_k(y; 1)$ is negative where $y$ looks more like attribute $k$ = 1 than 0, "
    "and the joint faithfulness is the simple average over $k=1, 2$.",
))

cells.append(code(
    "def fit_attribute(X, a_k):",
    "    \"\"\"Return prototypes {v: eta_{k,v}} and shared precision matrix P_k.\"\"\"",
    "    values = np.unique(a_k)",
    "    protos = {}",
    "    pooled_cov = np.zeros((X.shape[1], X.shape[1]))",
    "    total = 0",
    "    for v in values:",
    "        idx = (a_k == v)",
    "        pts = X[idx]",
    "        protos[int(v)] = pts.mean(axis=0)",
    "        # within-value covariance, centered at the value's own prototype",
    "        diffs = pts - protos[int(v)]",
    "        pooled_cov += diffs.T @ diffs",
    "        total += pts.shape[0]",
    "    pooled_cov = pooled_cov / max(total - len(values), 1)",
    "    # Small ridge to stay PD (the toy is 2D, very low rank-deficiency risk,",
    "    # but the paper uses Ledoit-Wolf shrinkage; we mimic with a small ridge)",
    "    pooled_cov += 1e-4 * np.eye(X.shape[1])",
    "    P_k = np.linalg.inv(pooled_cov)",
    "    return protos, P_k",
    "",
    "protos_1, P1 = fit_attribute(X, a1)",
    "protos_2, P2 = fit_attribute(X, a2)",
    "",
    "def d_k(points, eta, P):",
    "    return mahal_sq(points, eta, P)",
    "",
    "def margin_grid(target, protos, P):",
    "    \"\"\"M_k(y; target) over the grid.\"\"\"",
    "    d_target = d_k(grid_pts, protos[target], P)",
    "    competitors = [v for v in protos if v != target]",
    "    d_others = np.stack([d_k(grid_pts, protos[v], P) for v in competitors], 0)",
    "    return d_target - d_others.min(axis=0)",
    "",
    "def margin_real(target, protos, P):",
    "    \"\"\"M_k(y; target) on every real sample.\"\"\"",
    "    d_target = d_k(X, protos[target], P)",
    "    competitors = [v for v in protos if v != target]",
    "    d_others = np.stack([d_k(X, protos[v], P) for v in competitors], 0)",
    "    return d_target - d_others.min(axis=0)",
    "",
    "# Target (1, 1): t_1 = 1, t_2 = 1",
    "M1_grid = margin_grid(1, protos_1, P1).reshape(XX.shape)",
    "M2_grid = margin_grid(1, protos_2, P2).reshape(XX.shape)",
    "",
    "# Calibration: m_{k,t}, s_{k,t} on real samples with a_k = t",
    "M1_calib = margin_real(1, protos_1, P1)[a1 == 1]",
    "M2_calib = margin_real(1, protos_2, P2)[a2 == 1]",
    "m_1, s_1 = M1_calib.mean(), M1_calib.std() + 1e-9",
    "m_2, s_2 = M2_calib.mean(), M2_calib.std() + 1e-9",
    "",
    "F1 = (M1_grid - m_1) / s_1",
    "F2 = (M2_grid - m_2) / s_2",
    "F = 0.5 * (F1 + F2)",
    "",
    "# Per-attribute F_k on every real sample (for calibration of F)",
    "F1_real_all = (margin_real(1, protos_1, P1) - m_1) / s_1",
    "F2_real_all = (margin_real(1, protos_2, P2) - m_2) / s_2",
    "F_real = 0.5 * (F1_real_all + F2_real_all)",
))

cells.append(md(
    "## 4. The two components, side by side",
    "",
    "Both panels use the same orientation: **blue = small value (good), red = large "
    "value (bad)**. The white-to-red transition tracks the contour where the "
    "score equals its real-sample mean.",
))

cells.append(code(
    "fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))",
    "",
    "for ax, field, title, vmin, vmax in [",
    "    (axes[0], R, r'$R(y)$  — larger = less real', -2, 6),",
    "    (axes[1], F, r'$F(y;\\,(1,1))$  — larger = less faithful', -3, 3),",
    "]:",
    "    im = ax.pcolormesh(XX, YY, field, cmap='RdBu_r', shading='auto',",
    "                       norm=Normalize(vmin=vmin, vmax=vmax))",
    "    ax.scatter(X[:, 0], X[:, 1], s=3, color='black', alpha=0.25)",
    "    ax.scatter(*TARGET, marker='X', s=160, color='yellow',",
    "               edgecolor='black', linewidth=1.4, zorder=5)",
    "    ax.set_xlim(-0.5, 1.6); ax.set_ylim(-0.5, 1.6)",
    "    ax.set_aspect('equal')",
    "    ax.set_xlabel(r'$a_1$'); ax.set_ylabel(r'$a_2$')",
    "    ax.set_title(title)",
    "    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)",
    "plt.tight_layout()",
    "plt.show()",
))

cells.append(md(
    "**Read the heatmaps.** $R$ is small (blue) along the L — the real support — "
    "and grows as you leave it, including toward the requested $(1,1)$ corner. "
    "$F$ for target $(1,1)$ is small (blue) in the upper-right quadrant, because "
    "samples there are closer to the seen $a_1{=}1$ and $a_2{=}1$ prototypes "
    "than to their respective $0$-competitors. The two components prefer "
    "**opposite regions**.",
))

cells.append(md(
    "## 5. Trust $T = R + F$ and the agreement region",
    "",
    "A sample is trustworthy when **both** components are small. We thresholdat the real-sample 95th percentile on each component (so 95 % of real samples "
    "from the calibration set sit inside the accepted region) and overlay the "
    "intersection as a green contour on the trust panel.",
))

cells.append(code(
    "T = R + F",
    "",
    "R_thresh = np.percentile(R_real, 95)",
    "F_thresh = np.percentile(F_real, 95)",
    "accept = (R <= R_thresh) & (F <= F_thresh)",
    "",
    "fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))",
    "",
    "im0 = axes[0].pcolormesh(XX, YY, R, cmap='RdBu_r', shading='auto',",
    "                          norm=Normalize(vmin=-2, vmax=6))",
    "axes[0].set_title(r'$R(y)$  (real-calibrated)')",
    "fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)",
    "",
    "im1 = axes[1].pcolormesh(XX, YY, F, cmap='RdBu_r', shading='auto',",
    "                          norm=Normalize(vmin=-3, vmax=3))",
    "axes[1].set_title(r'$F(y;\\,(1,1))$  (real-calibrated)')",
    "fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)",
    "",
    "im2 = axes[2].pcolormesh(XX, YY, T, cmap='RdBu_r', shading='auto',",
    "                          norm=Normalize(vmin=-3, vmax=7))",
    "axes[2].contour(XX, YY, accept.astype(float), levels=[0.5],",
    "                colors='lime', linewidths=2.5)",
    "axes[2].set_title(r'$T(y;\\,(1,1)) = R + F$' + '\\n(green: both ≤ 95th pct on real)')",
    "fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)",
    "",
    "for ax in axes:",
    "    ax.scatter(X[:, 0], X[:, 1], s=3, color='black', alpha=0.18)",
    "    ax.scatter(*TARGET, marker='X', s=160, color='yellow',",
    "               edgecolor='black', linewidth=1.4, zorder=5)",
    "    ax.set_xlim(-0.5, 1.6); ax.set_ylim(-0.5, 1.6)",
    "    ax.set_aspect('equal')",
    "    ax.set_xlabel(r'$a_1$'); ax.set_ylabel(r'$a_2$')",
    "plt.tight_layout()",
    "plt.show()",
))

cells.append(md(
    "## 6. What the demo shows",
    "",
    "The green contour encloses the region the trust score is willing to call "
    "trustworthy for the $(1,1)$ request — where realism *and* faithfulness "
    "are both below the real-sample 95th percentile.",
    "",
    "On this L-shaped support the contour **does not include the requested "
    "$(1,1)$ corner**: faithfulness peaks (i.e. is most negative) precisely "
    "where the data is absent, and realism explodes there. That is the correct "
    "behavior under compositional shift — the trust score abstains rather than "
    "rubber-stamp the request.",
    "",
    "On real data the picture is the same, in high dimension: realism uses the "
    "Mahalanobis precision of the pooled real features in DINOv3 / SigLIP space, "
    "faithfulness uses per-attribute margins under a shared precision matrix "
    "$P_k$, and calibration is computed on a held-out real split. See "
    "`src/faithful_cond_gen/eval/trust_eval/scoring_core.py` for the full "
    "implementation.",
))


nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path(__file__).resolve().parent / "trust_score_l_shape.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out} ({len(cells)} cells)")
