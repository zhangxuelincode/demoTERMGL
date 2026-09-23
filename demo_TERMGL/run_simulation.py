# -*- coding: utf-8 -*-
"""Paper-style simulation study for the Python TERMGL re-implementation.

Grid (following demo.m of the MATLAB repository and the paper's figures):
  * four output-noise families: normal / t / chisq / exp, with and without
    20% additive outliers;
  * partial annotation (50% of tasks labelled) and noisy dimensions (30% of
    feature columns corrupted in trn/val) as additional stress factors on the
    normal-noise scenario.

Methods compared per scenario:
  * TERMGL  : bilevel-learned structure (this work);
  * ridge   : uniform structure (no group learning), i.e. plain ridge;
  * oracle  : lower solve at the true indicator structure thetastar.

Metrics: TD (vs noiseless responses), ASE (vs noisy test responses),
group-assignment accuracy, interval coverage. Results -> results/results.csv.
"""
import os
import time
import numpy as np
import pandas as pd

from termgl.config import TERMGLConfig
from termgl.data import synthesize_dataset, apply_partial_annotation, add_noisy_dimensions
from termgl.bilevel import run_termgl
from termgl.evaluation import evaluate_tasks, structure_metrics
from termgl.dfbb import term_hq_dfbb


def ridge_solve(X, y, mu):
    n, P = X.shape
    return np.linalg.solve(X.T @ X / n + (mu / 2.0) * np.eye(P), X.T @ y / n)


def run_scenario(noise, outlier_frac, annotation_ratio, noisy_frac, tag,
                 mode="unroll", N=50, P=50, T=200, L=5, G=2, seed=0):
    cfg = TERMGLConfig(
        N=N, P=P, T=T, L=L, G=G, sigma=1.0, lam=0.5, mu=1e-3,
        tilt=-1e-3, noise_distrib=noise,
        outlier_number=int(round(outlier_frac * N)), outlier_type=2,
        modal_iter=2, inner_itermax=600, inner_tol=1e-10,
        outer_itermax=300, batch_size=4,
        annotation_ratio=annotation_ratio,
        noisy_dims_fraction=noisy_frac, noisy_dims_level=1.0, seed=seed)

    rng = np.random.default_rng(cfg.seed)
    data = synthesize_dataset(cfg.N, cfg.P, cfg.T, cfg.L, cfg.G, cfg, rng)
    data = add_noisy_dimensions(data, cfg, rng)
    data, annotated = apply_partial_annotation(data, cfg, rng)

    t0 = time.time()
    res = run_termgl(data, cfg, annotated, mode=mode)
    print(f"[{tag}] bilevel done in {time.time()-t0:.0f}s "
          f"({len(res['history'])} iters)", flush=True)

    rows = []
    for method, W in [
        ("TERMGL", res["W"]),
        ("ridge", np.stack([ridge_solve(data["X"]["trn"][t],
                                        data["y"]["trn"][t], cfg.mu)
                            for t in range(cfg.T)], axis=1)),
        ("oracle", np.stack([term_hq_dfbb(data["X"]["trn"][t],
                                          data["y"]["trn"][t],
                                          data["thetastar"], cfg)[0]
                             for t in range(cfg.T)], axis=1)),
    ]:
        ev = evaluate_tasks(data, W, cfg)
        sm = structure_metrics(res["theta"] if method == "TERMGL"
                               else data["thetastar"], data["thetastar"])
        rows.append({
            "tag": tag, "noise": noise, "outlier_frac": outlier_frac,
            "annotation": annotation_ratio, "noisy_dims": noisy_frac,
            "method": method,
            "td_mean": ev["td_mean"], "td_std": float(ev["td"].std()),
            "ase_mean": ev["ase_mean"],
            "ci_cover": ev["ci_cover_mean"],
            "struct_acc": sm["assignment_accuracy"],
            "n_outer": len(res["history"]),
        })
    return rows


def main():
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(out_dir, exist_ok=True)
    all_rows = []

    # ---- core grid: 4 noises x {no outliers, 20% outliers} ----
    for noise in ("normal", "t", "chisq", "exp"):
        for out_frac in (0.0, 0.2):
            tag = f"{noise}-out{int(out_frac*100)}"
            all_rows += run_scenario(noise, out_frac, 1.0, 0.0, tag)

    # ---- stress factors on normal noise ----
    all_rows += run_scenario("normal", 0.0, 0.5, 0.0, "normal-partial50")
    all_rows += run_scenario("normal", 0.0, 1.0, 0.3, "normal-noisydims30")
    all_rows += run_scenario("normal", 0.2, 0.5, 0.3, "normal-out20-partial50-noisy30")

    df = pd.DataFrame(all_rows)
    csv_path = os.path.join(out_dir, "results.csv")
    df.to_csv(csv_path, index=False)
    print("saved", csv_path)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
