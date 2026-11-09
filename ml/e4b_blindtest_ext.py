#!/usr/bin/env python3
"""E4b 扩展盲测：新增 12 组织只进测试集 + 组聚类 bootstrap 置信区间

动机：原盲测仅 8 组织 / 72 行，点估计无区间。本脚本：
  - train/cal 完全不动（E4 同 seed 同 24/8 划分 → 已发表 CV/盲测数字可比）
  - 新增 12 组织（labels/e3_labels_ext.csv，缺席时退化为原 8 组织自检）
    仅作测试 → 盲测 20 组织 / 180 行
  - 指标 AUC / R² / conformal 覆盖率 报告组聚类 bootstrap 95% CI：
    以组织为单位有放回重采样（组内 9 行整体进出，保持跨金属/跨势相关结构），
    B=2000，percentile 2.5/97.5；AUC 单类重复记无效并报告有效重复数

范围（scope）：
  orig8   原 8 组织（与 E4 已发表点估计对账）
  ext12   新 12 组织（纯外推）
  all20   合并 20 组织（论文主报）

产物：
  results/e4b_blind_ci.csv            scope × target × model × fset × metric + CI
  results/e4b_prediction_audit.csv    all20 逐行预测（区间 + 域标志，供图）
  results/e4b_domain.csv              各 scope 适用域 OOD 行数
  （图在 e4b 出图步骤里由 make_figs / 本脚本追加）

自检：ext 缺席时 all20 ≡ orig8，且 TG-RF cls AUC = 0.8508、reg R²/覆盖
      与 results/e4_blindtest.csv 逐位一致。
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, r2_score

import e4_blindtest as E4

B_BOOT = 2000
BOOT_SEED = E4.SEED + 1
METRICS = ("auc", "r2", "mae", "coverage")


def load_all():
    """基准 360 行 + 扩展标签（若有）。split 只由基准 40 组织决定。"""
    base = E4.load()
    ext_path = os.path.join(E4.D6, "labels", "e3_labels_ext.csv")
    ext = None
    if os.path.exists(ext_path):
        lab = pd.read_csv(ext_path)
        fea = pd.read_csv(os.path.join(E4.D6, "results", "network_features.csv"))
        ext = lab.merge(fea.drop(columns=["n_grains"]),
                        on=["sample", "metal", "potential"])
        ext["fail"] = 1 - ext["censored"]
        dup = set(ext["sample"].unique()) & set(base["sample"].unique())
        assert not dup, f"扩展标签与基准组织重叠 {dup}"
        assert ext["eps_fail"].isna().eq(ext["censored"].eq(1)).all(), "扩展 NaN 与删失不一致"
    return base, ext


def split_orig40(base):
    g = base["sample"].unique().copy()
    np.random.default_rng(E4.SEED).shuffle(g)
    gtr = g[:E4.N_TRAIN_G]
    gca = g[E4.N_TRAIN_G:E4.N_TRAIN_G + E4.N_CAL_G]
    gte = g[E4.N_TRAIN_G + E4.N_CAL_G:]
    return list(gtr), list(gca), list(gte)


def cluster_boot(groups_of_row, stat_fn, b=B_BOOT, seed=BOOT_SEED):
    """组聚类 bootstrap：有放回抽组 → 组内行整体进入 → stat_fn → percentile CI。"""
    ug = np.array(sorted(set(groups_of_row)))
    idx_of = {g: np.where(groups_of_row == g)[0] for g in ug}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(b):
        pick = rng.choice(ug, size=len(ug), replace=True)
        idx = np.concatenate([idx_of[g] for g in pick])
        v = stat_fn(idx)
        if v is not None and np.isfinite(v):
            vals.append(v)
    return vals


def main():
    base, ext = load_all()
    gtr, gca, gte = split_orig40(base)
    n_ext_g = 0 if ext is None else ext["sample"].nunique()
    print(f"[load] 基准 {len(base)} 行 / 40 组织；扩展 "
          f"{0 if ext is None else len(ext)} 行 / {n_ext_g} 组织")
    print(f"[split] train/cal/test(原) = {len(gtr)}/{len(gca)}/{len(gte)} 组织")

    df = base if ext is None else pd.concat([base, ext], ignore_index=True)
    ext_g = [] if ext is None else sorted(ext["sample"].unique().tolist())
    scopes = {"orig8": list(gte), "all20": list(gte) + ext_g}
    if ext_g:
        scopes["ext12"] = ext_g

    mtr = df["sample"].isin(gtr).to_numpy()
    mca = df["sample"].isin(gca).to_numpy()
    assert not (df["sample"].isin(gca).to_numpy() & mtr).any(), "train/cal 泄漏"
    for g_ in ext_g:
        assert g_ not in gtr and g_ not in gca, f"扩展组织 {g_} 泄漏进 train/cal"

    eps = df["eps_fail"].to_numpy(float)
    y_cls = df["fail"].to_numpy()
    recs, audit = [], []

    # ---- 适用域（TG 全特征，E4 同配方）----
    X_all = {k: E4.design(df, v) for k, v in E4.SETS.items()}
    d_cal, d_te, _ = E4.domain_distance(X_all["TG"][mtr], X_all["TG"][mca],
                                        X_all["TG"])
    tau = float(np.quantile(d_cal, 0.95))
    in_dom_all = d_te <= tau
    dom_rows = []

    for fset in ("T", "G", "TG"):
        X = X_all[fset]
        for target in ("cls", "reg"):
            ok = np.ones(len(df), bool) if target == "cls" else ~np.isnan(eps)
            for mn, mo in E4.make_models(target, E4.SEED).items():
                tr = mtr & ok
                ca = mca & ok
                ytr = (y_cls if target == "cls" else eps)[tr]
                mo.fit(X[tr], ytr)
                if target == "cls":
                    p_all = mo.predict_proba(X)[:, 1]
                    q = np.nan
                else:
                    q = E4.conformal_q(np.abs(eps[ca] - mo.predict(X[ca])))
                    p_all = mo.predict(X)

                for sc, glist in scopes.items():
                    mte = df["sample"].isin(glist).to_numpy() & ok
                    idx_all = np.where(mte)[0]
                    if len(idx_all) == 0:
                        continue
                    y = (y_cls if target == "cls" else eps)[idx_all]
                    p = p_all[idx_all]
                    grp = df["sample"].to_numpy()[idx_all]
                    n_g = len(glist)
                    full = np.arange(len(y))

                    if target == "cls":
                        fns = {"auc": lambda idx: (roc_auc_score(y[idx], p[idx])
                               if len(np.unique(y[idx])) == 2 else None)}
                    else:
                        def f_r2(idx):
                            yy = y[idx]
                            return (r2_score(yy, p[idx])
                                    if len(yy) > 2 and np.unique(yy).size > 2
                                    else None)
                        fns = {"r2": f_r2,
                               "mae": lambda idx: float(np.mean(np.abs(y[idx] - p[idx]))),
                               "coverage": lambda idx: float(
                                   np.mean(np.abs(y[idx] - p[idx]) <= q))}

                    for met, fn in fns.items():
                        v = fn(full)
                        boots = cluster_boot(grp, fn)
                        arr = np.asarray(boots, float)
                        lo, hi = (np.quantile(arr, [0.025, 0.975]) if len(arr)
                                  else (np.nan, np.nan))
                        recs.append(dict(scope=sc, target=target, model=mn,
                                         fset=fset, metric=met, value=v,
                                         ci_lo=lo, ci_hi=hi, n_groups=n_g,
                                         n_rows=len(idx_all),
                                         n_boot_valid=len(arr)))
                    if sc == "all20" and fset == "TG":
                        for j in idx_all:
                            r = df.iloc[j]
                            audit.append(dict(
                                sample=int(r["sample"]), metal=r["metal"],
                                potential=r["potential"], fset=fset, model=mn,
                                target=target,
                                y_true=(y_cls[j] if target == "cls" else eps[j]),
                                y_pred=p_all[j],
                                lo=(np.nan if target == "cls" else p_all[j] - q),
                                hi=(np.nan if target == "cls" else p_all[j] + q),
                                covered=(np.nan if target == "cls"
                                         else float(abs(eps[j] - p_all[j]) <= q)),
                                d5=d_te[j], in_domain=bool(in_dom_all[j])))

    res = pd.DataFrame(recs)
    res.round(4).to_csv(os.path.join(E4.D6, "results", "e4b_blind_ci.csv"),
                        index=False)
    aud = pd.DataFrame(audit)
    aud.round(6).to_csv(
        os.path.join(E4.D6, "results", "e4b_prediction_audit.csv"), index=False)

    for sc, glist in scopes.items():
        mte = df["sample"].isin(glist).to_numpy()
        dom_rows.append(dict(scope=sc, n_groups=len(glist), n_rows=int(mte.sum()),
                             n_ood=int((~in_dom_all[mte]).sum()), tau=tau))
    pd.DataFrame(dom_rows).round(4).to_csv(
        os.path.join(E4.D6, "results", "e4b_domain.csv"), index=False)

    show = res[res.metric.isin(["auc", "r2", "coverage"])].copy()
    show["ci"] = show.apply(lambda r: f"[{r.ci_lo:.3f}, {r.ci_hi:.3f}]", axis=1)
    print("\n[e4b_blind_ci]")
    print(show.pivot_table(index=["target", "fset", "model"],
                           columns=["scope", "metric"],
                           values="value").round(3).to_string())
    print("\n[CI 95%]")
    print(show.pivot_table(index=["target", "fset", "model"],
                           columns=["scope", "metric"], values="ci",
                           aggfunc="first").to_string())
    print("\n[e4b_domain]")
    print(pd.DataFrame(dom_rows).to_string(index=False))

    figs(res, aud, scopes)

    # ---- 自检 ----
    def get(t, f, m, met, sc):
        r = res[(res.scope == sc) & (res.target == t) & (res.fset == f)
                & (res.model == m) & (res.metric == met)]
        return float(r["value"].iloc[0])

    auc_tg = get("cls", "TG", "RF", "auc", "orig8")
    assert abs(auc_tg - 0.8508) < 5e-4, f"orig8 TG-RF AUC {auc_tg} 未复现 0.8508"
    aud_pub = pd.read_csv(os.path.join(E4.D6, "results", "prediction_audit.csv"))
    ar = aud_pub[(aud_pub.target == "reg") & (aud_pub.fset == "TG")
                 & (aud_pub.model == "RF") & aud_pub.y_true.notna()]
    r2_pub = r2_score(ar["y_true"], ar["y_pred"])
    cov_pub = float(ar["covered"].mean())
    assert abs(get("reg", "TG", "RF", "r2", "orig8") - r2_pub) < 5e-4, "reg R² 未复现"
    assert abs(get("reg", "TG", "RF", "coverage", "orig8") - cov_pub) < 5e-4, \
        "coverage 未复现"
    if ext is None:
        for key in (("cls", "auc"), ("reg", "r2"), ("reg", "coverage")):
            a, b = get(key[0], "TG", "RF", key[1], "orig8"), \
                   get(key[0], "TG", "RF", key[1], "all20")
            assert a == b, "无扩展标签时 all20 应等于 orig8"
        print("\n[check] 通过：AUC/R²/覆盖逐位复现 E4（ext 未到，all20≡orig8）")
    else:
        assert res[res.scope == "ext12"]["n_groups"].iloc[0] == 12
        print(f"\n[check] 通过：E4 复现 + 扩展 {n_ext_g} 组织盲测 "
              f"{res[res.scope == 'all20']['n_rows'].iloc[0]} 行，CI 已出")


def figs(res, aud, scopes):
    """Fig 8：三指标分组柱状图（点估计 + 组聚类 bootstrap 95% CI，orig8 灰 ×）+ TG-RF 盲测散点。"""
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK HK", "Droid Sans Fallback",
                                              "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["font.size"] = 11.5
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.6))
    # 分组柱状图：三指标各一格，x = T/G/TG，柱色 = RF/Lin，误差棒 = 95% CI
    forest = [("cls", "auc", "ROC-AUC", (0.45, 0.97),
               [0.5, 0.6, 0.7, 0.8, 0.9], 0.5),
              ("reg", "r2", "$R^{2}$", (-0.25, 0.65),
               [-0.2, 0.0, 0.2, 0.4, 0.6], 0.0),
              ("reg", "coverage", "coverage", (0.70, 1.03),
               [0.7, 0.8, 0.9, 1.0], 0.9)]
    fsets, models = ["T", "G", "TG"], ["RF", "Lin"]
    cols = {"RF": "#4C72B0", "Lin": "#DD8452"}
    xoff = {"RF": -0.18, "Lin": 0.18}
    bw = 0.34
    for ax, (tgt, met, ylab, ylim, yticks, ref) in zip(axes.ravel()[:3], forest):
        for i, f in enumerate(fsets):
            for m in models:
                s = res[(res.scope == "all20") & (res.target == tgt)
                        & (res.metric == met) & (res.fset == f) & (res.model == m)]
                if len(s) != 1:
                    continue
                v = float(s["value"].iloc[0])
                clo = float(s["ci_lo"].iloc[0])
                chi = float(s["ci_hi"].iloc[0])
                x = i + xoff[m]
                ax.bar(x, v - ylim[0], bottom=ylim[0], width=bw, color=cols[m],
                       alpha=0.9, edgecolor="white", lw=0.8, zorder=3)
                ax.errorbar(x, v, yerr=[[v - clo], [chi - v]], fmt="none",
                            ecolor="0.25", elinewidth=1.2, capsize=3, zorder=4)
        s0 = res[(res.scope == "orig8") & (res.target == tgt)
                 & (res.metric == met) & (res.model == "RF")].set_index("fset")
        for i, f in enumerate(fsets):
            if f in s0.index:
                ax.plot(i + xoff["RF"], float(s0.loc[f, "value"]), "x",
                        ms=7, color="0.3", mew=1.6, zorder=5)
        ax.axhline(ref, color="0.45", ls=":", lw=1, zorder=6)
        ax.set_ylim(*ylim)
        ax.set_yticks(yticks)
        ax.set_ylabel(ylab, fontsize=11.5)
        ax.set_xticks(range(len(fsets)))
        ax.set_xticklabels(fsets, fontsize=11.5)
        ax.set_xlabel("feature set", fontsize=11.5)
        ax.set_xlim(-0.55, len(fsets) - 0.45)
        ax.tick_params(axis="y", labelsize=11.5)
        ax.grid(axis="y", alpha=0.3)

    # 散点：TG-RF reg，all20，按 orig/ext 着色；90% conformal 区间 = 对角线 ±q 阴影带
    ax = axes.ravel()[3]
    a = aud[(aud.target == "reg") & aud.y_true.notna() & (aud.model == "RF")
            & (aud["sample"].isin(scopes["all20"]))]
    ext_g = set(scopes.get("ext12", []))
    q = float((a.hi - a.y_pred).median())   # split-conformal q，各行相同
    lo, hi = float(a.y_true.min()), float(a.y_true.max())
    ax.fill_between([lo, hi], [lo - q, hi - q], [lo + q, hi + q],
                    color="0.55", alpha=0.22, lw=0, zorder=1)
    for mask, c, lb in ((~a["sample"].isin(ext_g), "#4C72B0", "orig."),
                        (a["sample"].isin(ext_g), "#DD8452", "added")):
        s = a[mask]
        if len(s):
            ax.scatter(s.y_true, s.y_pred, s=24, color=c, alpha=0.9,
                       edgecolors="white", linewidths=0.5, zorder=3,
                       label=f"{lb} (n={s['sample'].nunique()})")
    # OOD 标记：盲测回归行中超出 τ（d5）的行用空心黑圈标出
    if "in_domain" in a.columns:
        o = a[~a.in_domain.astype(bool)]
        if len(o):
            ax.scatter(o.y_true, o.y_pred, s=80, facecolors="none",
                       edgecolors="k", lw=1.4, zorder=5)
            ax.annotate("OOD", xy=(0.0235, 0.0305), xytext=(0.0105, 0.006),
                        ha="left", va="center", fontsize=11.5, color="k",
                        arrowprops=dict(arrowstyle="-", lw=0.8, color="0.3"),
                        bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5),
                        zorder=6)
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, zorder=4)
    ax.set_xlabel("True $\\varepsilon_{fail}$")
    ax.set_ylabel("Predicted $\\varepsilon_{fail}$", fontsize=11.5)
    ax.set_ylim(-0.028, 0.078)
    ax.set_yticks([0, 0.02, 0.04, 0.06])
    ax.tick_params(axis="y", labelsize=11.5)
    ax.legend(fontsize=11.5, loc="lower right", frameon=True, facecolor="white",
              framealpha=0.92, edgecolor="0.85", borderpad=0.6)
    ax.grid(alpha=0.3)

    hs = [plt.Rectangle((0, 0), 1, 1, fc="#4C72B0", alpha=0.9, label="RF (n=20)"),
          plt.Rectangle((0, 0), 1, 1, fc="#DD8452", alpha=0.9, label="Lin (n=20)"),
          plt.Line2D([], [], ls="", marker="x", color="0.3", ms=7, mew=1.6,
                     label="orig. 8 (RF)")]
    fig.legend(hs, [h.get_label() for h in hs], loc="upper center",
               bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False,
               fontsize=11.5, columnspacing=1.4, handletextpad=0.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(os.environ.get("FIG_OUT", os.path.join(E4.D6, "figures")),
                             "e4b_blind_ci.png"), dpi=int(os.environ.get("FIG_DPI", "200")))
    plt.close(fig)
    print("[fig] figures/e4b_blind_ci.png 已出")


if __name__ == "__main__":
    main()
