#!/usr/bin/env python3
"""E4 ML 盲测：特征消融（纯拓扑 vs +GB 能）→ conformal UQ → 适用域

数据（全部已有，零新计算）：
  labels/e3_labels.csv           360 标签 = 40 组织 × 3 金属 × 3 势
  results/network_features.csv   E2 特征（按 sample,metal,potential 对齐）

设计：
  - 防泄漏分组：group = sample id（同一组织跨金属/跨势 9 行恒同组）
  - 盲测固定划分 seed=20261023：24 组织 train / 8 组织 cal / 8 组织 test
  - 两阶段目标：
      ① 二分类 y=1−censored（ε_max=0.06 内是否贯通，360 行全用）
      ② 回归 eps_fail（仅非删失 155 行；判失后再预测失效应变）
  - 特征消融三组（金属 one-hot 恒含作对照；势不入模——势间标签差异应由
    γ 特征吸收，这正是"GB 能特征吸收 MLIP 势间差异"的可学信号）：
      T  纯拓扑：n_grains,twin_prob,n_edges,mean_degree,avg_clustering,twin_length_frac
      G  纯 GB 能：mean_gamma,gamma_std,gamma_perc,failed_cluster_*（γ×网络渗流复合）
      TG 全量
  - 模型：RF（主）+ Ridge/Logistic（线性基线）
  - 消融指标：StratifiedGroupKFold 5 折 × 4 重复 → mean±std（结果稳健性）
  - conformal UQ：split conformal，|残差| 经验分位（method=higher），α=0.1 → 名义 90%
  - 适用域：train 标准化全特征空间 5-NN 距离 d5；τ = cal 集 d5 的 95 分位，
    test 行 d5>τ 判 OOD，分域报告 coverage / MAE

产物：
  results/e4_ablation.csv        消融折间指标（target × model × fset）
  results/e4_blindtest.csv       盲测 + conformal 汇总（fset × model）
  results/prediction_audit.csv   test 逐行预测（区间 + 域标志）
  figures/e4_ablation.png        消融条形图（AUC / R²）
  figures/e4_conformal_domain.png 区间一致性图 + 残差-域距离图

注意（论文 caveat）：E3 标签经 σ_c=k(γ+γ₀)^0.65 生成，γ 对标签有构造内耦合，
G 组特征天然占优；T 组独立于该耦合，其消融差值即"拓扑的净贡献"下界。
"""
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, brier_score_loss, mean_absolute_error, r2_score

HERE = os.path.dirname(os.path.abspath(__file__))
D6 = os.path.abspath(os.path.join(HERE, ".."))
SEED = 20261023
ALPHA = 0.1                 # conformal 名义误覆盖率 → 90% 区间
N_TRAIN_G, N_CAL_G = 24, 8   # 40 组织 → 24/8/8
K_DOMAIN = 5

TOPO = ["n_grains", "twin_prob", "n_edges", "mean_degree", "avg_clustering",
        "twin_length_frac"]
GAMMA_F = ["mean_gamma", "gamma_std", "gamma_perc_mJm2", "gamma_perc_rel",
           "failed_cluster_auc", "failed_frac_at_mean"]
SETS = {"T": TOPO, "G": GAMMA_F, "TG": TOPO + GAMMA_F}


def load():
    lab = pd.read_csv(os.path.join(D6, "labels", "e3_labels.csv"))
    fea = pd.read_csv(os.path.join(D6, "results", "network_features.csv"))
    df = lab.merge(fea.drop(columns=["n_grains"]), on=["sample", "metal", "potential"])
    assert len(df) == 360 and df.isna().sum().sum() == df["eps_fail"].isna().sum()
    assert (df["eps_fail"].isna() == (df["censored"] == 1)).all(), "NaN 与删失不一致"
    df["fail"] = 1 - df["censored"]
    return df


def design(df, cols):
    """特征矩阵：消融列 + 金属 one-hot（对照恒含）。"""
    met = pd.get_dummies(df["metal"], dtype=float)
    return np.hstack([df[cols].to_numpy(float), met.to_numpy(float)])


def make_models(target, seed):
    if target == "cls":
        return {"RF": RandomForestClassifier(
                    n_estimators=500, min_samples_leaf=2, random_state=seed, n_jobs=-1),
                "Lin": make_pipeline(StandardScaler(),
                                     LogisticRegression(C=1.0, max_iter=2000))}
    return {"RF": RandomForestRegressor(
                n_estimators=500, min_samples_leaf=2, random_state=seed, n_jobs=-1),
            "Lin": make_pipeline(StandardScaler(), Ridge(alpha=10.0))}


def cv_ablation(df, Xs):
    """5 折 × 4 重复 StratifiedGroupKFold（group=sample）→ 折间指标。"""
    recs = []
    y_cls = df["fail"].to_numpy()
    eps = df["eps_fail"].to_numpy()
    groups = df["sample"].to_numpy()
    for target in ("cls", "reg"):
        for fset in ("T", "G", "TG"):
            X = Xs[fset]
            for rep in range(4):
                sgkf = StratifiedGroupKFold(5, shuffle=True, random_state=SEED + rep)
                for tr, te in sgkf.split(X, y_cls, groups):
                    if target == "reg":
                        tr = tr[~np.isnan(eps[tr])]
                        te = te[~np.isnan(eps[te])]
                        if len(tr) < 30 or len(te) < 5:
                            continue
                        ytr, yte = eps[tr], eps[te]
                    else:
                        ytr, yte = y_cls[tr], y_cls[te]
                        if len(np.unique(yte)) < 2:
                            continue
                    for mn, m in make_models(target, SEED + rep).items():
                        m.fit(X[tr], ytr)
                        if target == "cls":
                            p = m.predict_proba(X[te])[:, 1]
                            recs.append(dict(target=target, model=mn, fset=fset,
                                             rep=rep, fold=len(recs),
                                             auc=roc_auc_score(yte, p),
                                             brier=brier_score_loss(yte, p),
                                             r2=np.nan, mae=np.nan))
                        else:
                            p = m.predict(X[te])
                            recs.append(dict(target=target, model=mn, fset=fset,
                                             rep=rep, fold=len(recs),
                                             auc=np.nan, brier=np.nan,
                                             r2=r2_score(yte, p),
                                             mae=mean_absolute_error(yte, p)))
    return pd.DataFrame(recs)


def conformal_q(resid, alpha=ALPHA):
    """split conformal 分位数：ceil((n+1)(1-α))/n，method=higher。"""
    n = len(resid)
    lvl = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(resid, lvl, method="higher"))


def domain_distance(Xtr, Xcal, Xte, k=K_DOMAIN):
    """train 空间标准化 → 各行到 train 的 k-NN 平均距离。"""
    sc = StandardScaler().fit(Xtr)

    def d5(A):
        Z, Ztr = sc.transform(A), sc.transform(Xtr)
        dist = np.sqrt(((Z[:, None, :] - Ztr[None, :, :]) ** 2).sum(-1))
        return np.sort(dist, axis=1)[:, :k].mean(1)
    return d5(Xcal), d5(Xte), sc


def main():
    df = load()
    Xs = {k: design(df, v) for k, v in SETS.items()}
    g = df["sample"].unique().copy()
    rng = np.random.default_rng(SEED)
    rng.shuffle(g)
    gtr, gca, gte = g[:N_TRAIN_G], g[N_TRAIN_G:N_TRAIN_G + N_CAL_G], g[N_TRAIN_G + N_CAL_G:]
    mtr = df["sample"].isin(gtr).to_numpy()
    mca = df["sample"].isin(gca).to_numpy()
    mte = df["sample"].isin(gte).to_numpy()
    assert not (mtr & mca).any() and not (mtr & mte).any() and not (mca & mte).any(), "分组泄漏"
    print(f"[split] train/cal/test 组织数 {N_TRAIN_G}/{N_CAL_G}/{len(gte)}"
          f"  行数 {mtr.sum()}/{mca.sum()}/{mte.sum()}"
          f"  test 贯通率 {df.loc[mte, 'fail'].mean():.3f}")

    # ---- 消融 CV ----
    ab = cv_ablation(df, Xs)
    summ = (ab.groupby(["target", "model", "fset"])
              .agg(auc=("auc", "mean"), auc_sd=("auc", "std"),
                   brier=("brier", "mean"),
                   r2=("r2", "mean"), r2_sd=("r2", "std"),
                   mae=("mae", "mean"), n_folds=("rep", "size"))
              .reset_index())
    summ.round(4).to_csv(os.path.join(D6, "results", "e4_ablation.csv"), index=False)
    print("\n[e4_ablation]（5 折 × 4 重复 mean±std）")
    print(summ.round(3).to_string(index=False))

    # ---- 盲测 + conformal + 适用域 ----
    eps = df["eps_fail"].to_numpy()
    audit, blind = [], []
    d_cal, d_te, _ = domain_distance(Xs["TG"][mtr], Xs["TG"][mca], Xs["TG"][mte])
    tau = float(np.quantile(d_cal, 0.95))
    in_domain = d_te <= tau
    print(f"\n[domain] τ(cal d5 p95)={tau:.3f}  test OOD 行 {int((~in_domain).sum())}/{mte.sum()}")

    for fset in ("T", "G", "TG"):
        X = Xs[fset]
        for target in ("cls", "reg"):
            ok = np.ones(len(df), bool) if target == "cls" else ~np.isnan(eps)
            for mn, m in make_models(target, SEED).items():
                tr = mtr & ok
                ca = mca & ok
                te = mte & ok
                ytr = (df["fail"].to_numpy() if target == "cls" else eps)[tr]
                m.fit(X[tr], ytr)
                if target == "cls":
                    p_te = m.predict_proba(X[mte])[:, 1]
                    yte = df["fail"].to_numpy()[mte]
                    blind.append(dict(target=target, model=mn, fset=fset,
                                      auc=roc_auc_score(yte, p_te),
                                      brier=brier_score_loss(yte, p_te),
                                      coverage=np.nan, width=np.nan, n_test=len(yte)))
                    for i, idx in enumerate(np.where(mte)[0]):
                        audit.append(dict(sample=int(df.at[idx, "sample"]),
                                          metal=df.at[idx, "metal"],
                                          potential=df.at[idx, "potential"],
                                          fset=fset, model=mn, target=target,
                                          y_true=df.at[idx, "fail"], y_pred=p_te[i],
                                          lo=np.nan, hi=np.nan, covered=np.nan))
                else:
                    p_ca = m.predict(X[ca])
                    q = conformal_q(np.abs(eps[ca] - p_ca))
                    p_te = m.predict(X[te])
                    yte = eps[te]
                    cov = float(np.mean(np.abs(yte - p_te) <= q))
                    blind.append(dict(target=target, model=mn, fset=fset,
                                      auc=np.nan, brier=np.nan,
                                      coverage=cov, width=2 * q, n_test=len(yte)))
                    sub = np.where(te)[0]
                    for j, idx in enumerate(sub):
                        audit.append(dict(sample=int(df.at[idx, "sample"]),
                                          metal=df.at[idx, "metal"],
                                          potential=df.at[idx, "potential"],
                                          fset=fset, model=mn, target=target,
                                          y_true=eps[idx], y_pred=p_te[j],
                                          lo=p_te[j] - q, hi=p_te[j] + q,
                                          covered=float(abs(eps[idx] - p_te[j]) <= q)))
    blind_df = pd.DataFrame(blind)
    # 域标志（仅 TG 组织定义，回填全表）
    dom = df.loc[mte, ["sample", "metal", "potential"]].copy()
    dom["d5"], dom["in_domain"] = d_te, in_domain
    aud = pd.DataFrame(audit).merge(dom, on=["sample", "metal", "potential"], how="left")
    blind_df.round(4).to_csv(os.path.join(D6, "results", "e4_blindtest.csv"), index=False)
    aud.round(6).to_csv(os.path.join(D6, "results", "prediction_audit.csv"), index=False)
    print("\n[e4_blindtest]（盲测 8 组织）")
    print(blind_df.round(3).to_string(index=False))

    # 分域指标（TG-RF 主模型）
    a = aud[(aud.fset == "TG") & (aud.model == "RF")]
    ar = a[(a.target == "reg") & (a.y_true.notna())]
    print(f"[domain] TG-RF reg 分域：in {ar[ar.in_domain]['covered'].mean():.3f}"
          f"(n={ar.in_domain.sum()}) vs OOD {ar[~ar.in_domain]['covered'].mean():.3f}"
          f"(n={(~ar.in_domain).sum()})；MAE in {ar[ar.in_domain].apply(lambda r: abs(r.y_true-r.y_pred), axis=1).mean():.4f}")

    figs(summ, aud, tau)
    # 验收自检
    cov_tg = blind_df[(blind_df.target == "reg") & (blind_df.fset == "TG")
                      & (blind_df.model == "RF")]["coverage"].iloc[0]
    auc = {(r.fset): r.auc for r in blind_df.itertuples() if r.target == "cls" and r.model == "RF"}
    print(f"\n[check] conformal 覆盖 TG-RF={cov_tg:.3f}（名义 0.90，n=36~42 容差 0.80+）；"
          f"盲测 AUC T={auc['T']:.3f} G={auc['G']:.3f} TG={auc['TG']:.3f}")
    assert cov_tg >= 0.80, "conformal 覆盖不足"
    print("[check] 通过")


def figs(summ, aud, tau):
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK HK", "Droid Sans Fallback",
                                              "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (target, met, met_sd, title, ylim) in zip(axes, [
            ("cls", "auc", "auc_sd", "分类：是否贯通 (ROC-AUC)", (0.5, 1.0)),
            ("reg", "r2", "r2_sd", "回归：eps_fail (R²)", None)]):
        w = 0.26
        x = np.arange(3)
        for k, mn in enumerate(["RF", "Lin"]):
            s = summ[(summ.target == target) & (summ.model == mn)].set_index("fset").loc[["T", "G", "TG"]]
            ax.bar(x + (k - 0.5) * w, s[met], w, yerr=s[met_sd], capsize=3,
                   label=mn, color=["#4C72B0", "#DD8452"][k])
        ax.set_xticks(x, ["T 纯拓扑", "G 纯GB能", "TG 全量"])
        ax.set_title(title)
        if ylim:
            ax.set_ylim(*ylim)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(D6, "figures", "e4_ablation.png"), dpi=150)
    plt.close(fig)

    a = aud[(aud.fset == "TG") & (aud.model == "RF") & (aud.target == "reg")
            & aud.y_true.notna()]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    ax = axes[0]
    for mask, c in ((a.in_domain, "#4C72B0"), (~a.in_domain, "#C44E52")):
        s = a[mask]
        if len(s):
            ax.errorbar(s.y_true, s.y_pred,
                        yerr=np.vstack([s.y_pred - s.lo, s.hi - s.y_pred]),
                        fmt="o", ms=4, capsize=2, color=c, alpha=0.85,
                        label="in-domain" if c == "#4C72B0" else "OOD")
    ax.legend(fontsize=8)
    lo, hi = a.y_true.min(), a.y_true.max()
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel("ε_fail 真值"); ax.set_ylabel("预测 + 90% conformal 区间")
    ax.set_title(f"盲测 8 组织（TG-RF）覆盖 {a.covered.mean():.2f}")
    ax.grid(alpha=0.3)
    ax = axes[1]
    res = np.abs(a.y_true - a.y_pred)
    ax.scatter(a["d5"], res, c=np.where(a["in_domain"], "#4C72B0", "#C44E52"))
    ax.axvline(tau, color="k", ls="--", lw=1, label=f"τ={tau:.2f} (cal p95)")
    ax.set_xlabel("适用域距离 d5（train 标准化空间 5-NN）")
    ax.set_ylabel("|残差|")
    ax.set_title("残差 vs 适用域距离")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(D6, "figures", "e4_conformal_domain.png"), dpi=150)
    plt.close(fig)
    print(f"[fig] e4_ablation.png / e4_conformal_domain.png 已出")


if __name__ == "__main__":
    main()
