#!/usr/bin/env python3
"""E4c 探针：per-metal (Mondrian) split conformal 的可行代价（审稿 2.6 备答）

问题：§3.5 的条件覆盖未验证，Cu TG-RF 覆盖 0.74（Wilson CI 不含 0.90）。
能否用 Mondrian 分组校准补上？每金属校准行数 = 12，higher-method 下
lvl = ceil(13×0.9)/12 = 1.0 → 组内最大残差，区间会显著变宽。

做法：与 E4b 完全同配方（同 seed、同 24/8/8 组划分、同模型）训练 6 个
reg 模型；对每个模型算两种 q：
  pooled   —— 全部 36 校准行（论文现行口径，自检须复现 e4b 覆盖）
  mondrian —— 每金属 12 行各自 conformal_q
在 20 组盲测（100 行）上报池合覆盖、per-metal 覆盖与半宽度。

自检：pooled q 下 6 组合的覆盖率逐一对齐 results/e4b_prediction_audit.csv。

产物：results/e4c_mondrian.csv
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
D6 = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

import e4_blindtest as E4  # noqa: E402
import e4b_blindtest_ext as E4B  # noqa: E402


def main():
    base, ext = E4B.load_all()
    gtr, gca, gte = E4B.split_orig40(base)
    df = pd.concat([base, ext], ignore_index=True)
    mtr = df["sample"].isin(gtr).to_numpy()
    mca = df["sample"].isin(gca).to_numpy()
    mte = df["sample"].isin(list(gte) + sorted(ext["sample"].unique())).to_numpy()
    reg = ~np.isnan(df["eps_fail"].to_numpy(float))
    eps = df["eps_fail"].to_numpy(float)
    print(f"[split] train/cal/test 行 = {mtr.sum()}/{(mca & reg).sum()}/{(mte & reg).sum()}")
    print(f"[cal] 每金属校准行数: "
          f"{dict(df[mca & reg].metal.value_counts())}")

    # 自检基准：e4b 已发布 pooled 覆盖（scope=all20 的 6 组合全在 blind_ci 表）
    ref_tab = pd.read_csv(os.path.join(D6, "results", "e4b_blind_ci.csv"))
    ref_tab = ref_tab[(ref_tab.scope == "all20") & (ref_tab.target == "reg")
                      & (ref_tab.metric == "coverage")]

    recs = []
    for fset in ("T", "G", "TG"):
        X = E4.design(df, E4.SETS[fset])
        for mn, mo in E4.make_models("reg", E4.SEED).items():
            tr, ca = mtr & reg, mca & reg
            mo.fit(X[tr], eps[tr])
            resid = np.abs(eps[ca] - mo.predict(X[ca]))
            q_pool = E4.conformal_q(resid)
            ca_df = df[ca]
            q_metal = {m: E4.conformal_q(resid[ca_df.metal.to_numpy() == m])
                       for m in ("Cu", "Ni", "Fe")}

            te = mte & reg
            y, p = eps[te], mo.predict(X[te])
            met = df[te].metal.to_numpy()
            abs_e = np.abs(y - p)

            # 自检：pooled 覆盖复现 e4b blind_ci
            ref = ref_tab[(ref_tab.fset == fset) & (ref_tab.model == mn)]
            cov_ref = float(ref["value"].iloc[0])
            cov_pool = float((abs_e <= q_pool).mean())
            assert abs(cov_pool - cov_ref) < 5e-4, \
                f"{fset}/{mn}: pooled {cov_pool:.4f} != e4b {cov_ref:.4f}"

            cov_m = float(np.mean([abs_e[i] <= q_metal[met[i]]
                                   for i in range(len(abs_e))]))
            rec = dict(fset=fset, model=mn, n_cal_metal=12,
                       halfwidth_pool=round(q_pool, 4),
                       halfwidth_mondrian_median=round(
                           float(np.median(list(q_metal.values()))), 4),
                       cov_pool=round(cov_pool, 3),
                       cov_mondrian=round(cov_m, 3),
                       **{f"q_{m}": round(q_metal[m], 4) for m in q_metal},
                       **{f"cov_{m}": round(float(
                           (abs_e[met == m] <= q_metal[m]).mean()), 3)
                          for m in q_metal},
                       **{f"covpool_{m}": round(float(
                           (abs_e[met == m] <= q_pool).mean()), 3)
                          for m in q_metal})
            recs.append(rec)
            print(f"[{fset}-{mn}] pool q={q_pool:.4f} 池合覆盖={cov_pool:.3f} "
                  f"(复现✓) | mondrian q: "
                  + " ".join(f"{m}={q_metal[m]:.3f}" for m in q_metal)
                  + f" | 池合 Mondrian 覆盖={cov_m:.3f}")

    out = os.path.join(D6, "results", "e4c_mondrian.csv")
    pd.DataFrame(recs).to_csv(out, index=False)
    print(f"→ {out}")
    print("[check] 通过：6 组合 pooled 覆盖逐一对齐 e4b")


if __name__ == "__main__":
    main()
