#!/usr/bin/env python3
"""E4d 扩校准集敏感性分析：条件覆盖的可改善空间（审稿意见 r3）

问题：§3.5 的 split-conformal 校准仅 36 行（每金属 12 行），higher-method
下 Mondrian 每金属 lvl = ceil(13×0.9)/12 = 1.0 → 组内最大残差；池合口径下
Cu TG-RF 盲测覆盖 0.74（Wilson CI 不含 0.90）。若校准池扩大，条件覆盖
与区间宽度能改善多少？

做法（敏感性/探索性分析，不动冻结协议）：
  - 划分、模型、特征与 E4/E4b/E4c 完全同配方（同 seed、同 24/8/8 组）
  - 盲测 = 20 组织 / 100 非删失行（orig8 + ext12），与 e4b all20 一致
  - 校准口径四个 regime：
      pool36       基线：36 校准行池合 q（论文现行口径）
      mondrian36   基线：每金属 12 行（e4c，q = 组内最大残差）
      pool_oof     扩池：36 校准行 + train 24 组 5 折 OOF 残差，池合 q
      mondrian_oof 扩池：同上按金属分组（每金属 ~48 行，q 不再是最大值）
  - OOF：train 行上 StratifiedGroupKFold(5, shuffle, seed)（stratify=fail,
    group=sample），折内用非删失行拟合、对非删失验证行预测 → 残差
  - 报告盲测边际覆盖（组聚类 bootstrap 95% CI, B=2000）、分金属覆盖、
    半宽度；量化 Cu 覆盖 0.74 → ? 与区间宽度代价

自检：pool36 的 6 组合（T/G/TG × RF/Lin）盲测覆盖逐一对齐
results/e4b_blind_ci.csv（scope=all20, target=reg, metric=coverage）。

产物：results/e4d_cal_sensitivity.csv
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
D6 = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

import e4_blindtest as E4  # noqa: E402
import e4b_blindtest_ext as E4B  # noqa: E402

FSETS = ("T", "G", "TG")
METALS = ("Cu", "Ni", "Fe")


def make_oof(df, X, mtr, eps, make_model):
    """train 行的 5 折 OOF 绝对残差（仅非删失行有值）。"""
    tr = np.where(mtr)[0]
    y_cls = df["fail"].to_numpy()
    groups = df["sample"].to_numpy()
    sgkf = StratifiedGroupKFold(5, shuffle=True, random_state=E4.SEED)
    resid = np.full(len(df), np.nan)
    Xtr, ytr_cls, gtr = X[tr], y_cls[tr], groups[tr]
    for p_tr, p_va in sgkf.split(Xtr, ytr_cls, gtr):
        fit = tr[p_tr][~np.isnan(eps[tr[p_tr]])]
        va = tr[p_va][~np.isnan(eps[tr[p_va]])]
        if len(fit) < 30 or len(va) < 5:
            continue
        mo = make_model()
        mo.fit(X[fit], eps[fit])
        resid[va] = np.abs(eps[va] - mo.predict(X[va]))
    assert not np.isnan(resid[mtr & ~np.isnan(eps)]).any(), "OOF 残差有缺口"
    return resid


def main():
    base, ext = E4B.load_all()
    gtr, gca, gte = E4B.split_orig40(base)
    df = pd.concat([base, ext], ignore_index=True)
    mtr = df["sample"].isin(gtr).to_numpy()
    mca = df["sample"].isin(gca).to_numpy()
    mte = df["sample"].isin(list(gte) + sorted(ext["sample"].unique())).to_numpy()
    reg = ~np.isnan(df["eps_fail"].to_numpy(float))
    eps = df["eps_fail"].to_numpy(float)
    n_cal = int((mca & reg).sum())
    n_tr_reg = int((mtr & reg).sum())
    print(f"[split] train/cal/test 行 = {mtr.sum()}/{(mca & reg).sum()}/"
          f"{(mte & reg).sum()}（非删失 cal={n_cal} train={n_tr_reg}）")

    ref_tab = pd.read_csv(os.path.join(D6, "results", "e4b_blind_ci.csv"))
    ref_tab = ref_tab[(ref_tab.scope == "all20") & (ref_tab.target == "reg")
                      & (ref_tab.metric == "coverage")]

    recs = []
    for fset in FSETS:
        X = E4.design(df, E4.SETS[fset])
        for mn in ("RF", "Lin"):
            factory = (lambda mn=mn: E4.make_models("reg", E4.SEED)[mn])
            mo = factory()
            tr_r, ca_r = mtr & reg, mca & reg
            mo.fit(X[tr_r], eps[tr_r])

            # ---- 校准残差池 ----
            resid_ca = np.abs(eps[ca_r] - mo.predict(X[ca_r]))
            oof = make_oof(df, X, mtr, eps, factory)
            ca_df = df[ca_r]

            regimes = {}
            q_pool36 = E4.conformal_q(resid_ca)
            regimes["pool36"] = {m: q_pool36 for m in METALS}
            regimes["mondrian36"] = {
                m: E4.conformal_q(resid_ca[ca_df.metal.to_numpy() == m])
                for m in METALS}
            resid_pool = np.concatenate([resid_ca, oof[mtr & reg]])
            q_pool_oof = E4.conformal_q(resid_pool)
            regimes["pool_oof"] = {m: q_pool_oof for m in METALS}
            met_allcal = np.concatenate([
                ca_df.metal.to_numpy(), df[mtr & reg].metal.to_numpy()])
            resid_all = resid_pool
            regimes["mondrian_oof"] = {
                m: E4.conformal_q(resid_all[met_allcal == m]) for m in METALS}
            n_metal_regime = {
                "pool36": {m: 12 for m in METALS},
                "mondrian36": {m: int((ca_df.metal.to_numpy() == m).sum())
                               for m in METALS},
                "pool_oof": {m: 12 + int((df[mtr & reg].metal == m).sum())
                             for m in METALS},
                "mondrian_oof": {m: int((met_allcal == m).sum())
                                 for m in METALS},
            }

            # ---- 盲测评估 ----
            te = mte & reg
            y, p = eps[te], mo.predict(X[te])
            met = df[te].metal.to_numpy()
            abs_e = np.abs(y - p)
            grp = df.loc[te, "sample"].to_numpy()

            for rname, q_of in regimes.items():
                q_row = np.array([q_of[m] for m in met])
                covered = abs_e <= q_row
                def f_marg(idx, q_row=q_row, abs_e=abs_e):
                    return float(np.mean(abs_e[idx] <= q_row[idx]))
                boots = np.asarray(E4B.cluster_boot(grp, f_marg), float)
                lo, hi = np.quantile(boots, [0.025, 0.975])

                if rname == "pool36":
                    ref = ref_tab[(ref_tab.fset == fset) & (ref_tab.model == mn)]
                    cov_ref = float(ref["value"].iloc[0])
                    cov_pool = float((abs_e <= q_pool36).mean())
                    assert abs(cov_pool - cov_ref) < 5e-4, \
                        f"{fset}/{mn}: pool36 {cov_pool:.4f} != e4b {cov_ref:.4f}"

                rec = dict(
                    fset=fset, model=mn, regime=rname,
                    n_cal_total=sum(n_metal_regime[rname].values()),
                    q_Cu=round(q_of["Cu"], 4), q_Ni=round(q_of["Ni"], 4),
                    q_Fe=round(q_of["Fe"], 4),
                    halfwidth_median=round(float(np.median(q_row)), 4),
                    cov_marginal=round(float(covered.mean()), 3),
                    ci_lo=round(float(lo), 3), ci_hi=round(float(hi), 3),
                    **{f"cov_{m}": round(float(covered[met == m].mean()), 3)
                       for m in METALS},
                    **{f"n_cal_{m}": n_metal_regime[rname][m]
                       for m in METALS},
                )
                recs.append(rec)
                print(f"[{fset}-{mn}/{rname}] n={rec['n_cal_total']} "
                      f"q={q_of['Cu']:.3f}/{q_of['Ni']:.3f}/{q_of['Fe']:.3f} "
                      f"边际={rec['cov_marginal']:.3f} "
                      f"[{rec['ci_lo']:.3f},{rec['ci_hi']:.3f}] "
                      f"Cu/Ni/Fe={rec['cov_Cu']:.2f}/{rec['cov_Ni']:.2f}/"
                      f"{rec['cov_Fe']:.2f} "
                      f"半宽中位={rec['halfwidth_median']:.3f}")

    out = os.path.join(D6, "results", "e4d_cal_sensitivity.csv")
    pd.DataFrame(recs).to_csv(out, index=False)
    print(f"→ {out}")
    print("[check] 通过：pool36 六组合盲测覆盖逐一对齐 e4b all20")


if __name__ == "__main__":
    main()
