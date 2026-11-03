#!/usr/bin/env python3
"""E3 扩展标签：新增 12 组织曲线 → labels/e3_labels_ext.csv（固定 k，不重标定）

前提（已验证）：k_metal,pot 在基准 40 组织上按 (metal, potential) 恒定（中位数
标定对样本集不敏感），故新组织标签直接沿用基准 k_metal，不重标定——避免新样本
集改变 k 而把"组织差异"与"标定漂移"混在一起。

输出列与 e3_labels.csv 完全一致（E4b/E6 直接 concat）。
自检：贯通率与基准同金属量级一致、首步堆点 0%、k 列与基准逐位一致。
"""
import csv
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D6 = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from e3_postprocess import (load_e1_spectra, damage_labels,  # noqa: E402
                            GAMMA0, GAMMA_P)

EXT_SAMPLES = [34, 28, 15, 55, 115, 83, 8, 171, 103, 38, 193, 39]


def main():
    # ---- 基准 k_metal（每 (metal,pot) 应恒定）----
    with open(os.path.join(D6, "labels", "e3_labels.csv")) as fh:
        base = list(csv.DictReader(fh))
    k_base = {}
    for r in base:
        k_base.setdefault((r["metal"], r["potential"]), set()).add(float(r["k_metal"]))
    for key, vals in k_base.items():
        assert len(vals) == 1, f"基准 k_metal 不恒定 {key}: {vals}"
    k_fix = {key: vals.pop() for key, vals in k_base.items()}
    print(f"[k] 基准 k_metal 恒定，沿用 {len(k_fix)} 组 (metal,pot)")

    spec = load_e1_spectra()
    pots = sorted(spec.keys())
    rows = []
    for s in EXT_SAMPLES:
        d = np.load(os.path.join(D6, "networks", "library", f"sample_{int(s):03d}.npz"),
                    allow_pickle=True)
        et = d["etype"]
        for metal in ("Cu", "Ni", "Fe"):
            tag = f"{int(s):03d}_{metal}"
            p = os.path.join(D6, "labels", "e3_curves", f"{tag}.npz")
            if not os.path.exists(p):
                raise FileNotFoundError(f"缺曲线 {tag}（DAMASK 批量未完？）")
            c = dict(np.load(p, allow_pickle=True))
            g0 = GAMMA0[metal]
            for pot in pots:
                rng = np.random.default_rng(10_000 + int(s))     # 与 E3 同规则
                tw = spec[pot][metal]["twin"] or spec[pot][metal]["general"]
                gn = spec[pot][metal]["general"]
                g = np.maximum(
                    np.array([rng.choice(tw) if e == "twin" else rng.choice(gn)
                              for e in et]) + g0, 1e-6) ** GAMMA_P
                k = k_fix[(metal, pot)]
                ef, frac, cens, _ = damage_labels(
                    c["sigma_n"].astype(np.float64), c["eps"], g,
                    c["edge_g1"], c["edge_g2"], c["edge_n"],
                    c["edge_wx"], c["edge_wy"], int(c["n_grains"]), k)
                rows.append({"sample": int(s), "metal": metal, "potential": pot,
                             "n_grains": int(c["n_grains"]),
                             "eps_fail": round(float(ef), 5) if not np.isnan(ef) else "",
                             "failed_GB_len_frac": round(float(frac), 5),
                             "censored": int(cens),
                             "k_metal": round(k, 3)})

    out = os.path.join(D6, "labels", "e3_labels_ext.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"→ {out}（{len(rows)} 行 = {len(EXT_SAMPLES)} 组织 × 3 金属 × 3 势）")

    # ---- 自检：与基准同金属量级一致 ----
    n_fail = sum(r["censored"] == 0 for r in rows)
    print(f"[ext] 贯通率 {n_fail}/{len(rows)} = {n_fail / len(rows):.3f}"
          f"（基准 0.431）")
    per = [float(r["eps_fail"]) for r in rows if r["eps_fail"] != ""]
    pileup = np.mean(np.array(per) <= 0.0031) if per else np.nan
    print(f"[ext] eps_fail {min(per):.4f}–{max(per):.4f} 中位 {np.median(per):.4f}"
          f"；首步堆点 {pileup:.3f}")
    assert 0.25 <= n_fail / len(rows) <= 0.65, "扩展贯通率偏离基准转变带"
    assert pileup == 0.0, "扩展标签出现首步堆点"
    for metal in ("Cu", "Ni", "Fe"):
        sub = [r for r in rows if r["metal"] == metal]
        f_ = sum(r["censored"] == 0 for r in sub)
        print(f"[ext] {metal}: 贯通 {f_}/{len(sub)}")
        assert f_ >= 0.2 * len(sub), f"{metal} 贯通过少（标签退化）"
    print("[check] 通过：扩展标签生成（固定 k、无堆点、贯通带内）")


if __name__ == "__main__":
    main()
