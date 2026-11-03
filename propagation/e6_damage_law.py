#!/usr/bin/env python3
"""E6 标签独立性验证：替代损伤律重标签 + γ 无关解耦对照 + GBE 实验代理

问题（对 E3 标签的半独立验证）：E3 标签由单一损伤律 σ_c=k(γ+γ₀)^0.65, m=2 生成，
γ 对标签有构造内耦合 → G 通道优势部分是"构造出来的"。本脚本回答：
  1) 换损伤律（m、p 变体）后，标签判定/排序与 ML 消融结论稳不稳？
  2) 把 γ 从强度里完全拿掉（dec 解耦对照），G 通道优势还剩多少
     → 构造耦合份额的直接量化（讨论区 caveat 的定量支撑）。
  3) GBE 实验代理：twin_length_frac 与失效的符号是否与 Σ3 阻断叙事 [1] 相符。

数据（零新模拟，复用 E3 的 σ_n(ε) 曲线）：
  labels/e3_curves/*.npz         每 (sample,metal) 一条 CPFEM 边应力历史
  labels/e3_labels.csv           基准标签（前 360 行固定使用，与已发表数字可比）
  results/network_features.csv   E2 特征（消融复用）

损伤律变体（γ₀ 只在耦合变体里出现）：
  base  p=0.65, m=2    现行 v2 定稿（须与 e3_labels.csv 逐行一致 → assert）
  m1    p=0.65, m=1    损伤积累指数减半（线性积累）
  p05   p=0.5,  m=2    Griffith 型弱耦合（文献侧 p 区间下沿）
  p10   p=1.0,  m=2    线性强耦合（v1 邻域，预期首步堆点 → 边界负对照）
  dec   γ 无关对照      σ_c,e ≡ 常数（每 metal 跨势恒同），逐边 γ 结构被完全切断

  贯通率匹配重标定（与 E3 中 κ 的角色同源：κ 是等效模型的常规标定量）：每个非 base
  变体对每金属求单一增益 a_metal 乘在 σ_c 上，使该金属贯通计数与 base 尽量一致
  （Cu 64 / Ni 56 / Fe 35；dec 因跨势标签全同、每样本三势同翻，计数步长为 3，
  取最近可达整）。不匹配则律间差异混淆"律形"与平凡强度尺度（实测原 κ 配方下
  dec 贯通率 0.000、m1 0.039 → 标签退化，ML 不可比）。a_metal 入表报告；
  对数二分求解（行级贯通对 a 单调 ⇒ 计数为 a 的单调阶梯函数）。

  dec 的意义：同等贯通率下，标签只由 σ_n(ε) 应力历史驱动——γ 与标签的构造耦合
  被切断。G 通道在 base 下相对 T 的增量若在 dec 下消失，差值即构造耦合份额。

每变体报告：
  标签层：贯通率、首步堆点率、与 base 的判定一致率、eps_fail Spearman（双方贯通行）
  ML 层：分组 CV（5×4，T/G/TG × RF/Lin）AUC/R² + 原 8 组织盲测 AUC/R²/覆盖
  GBE 代理：spearman(twin_length_frac, fail) 全体/分金属

产物：
  results/e6_label_stats.csv     变体 × 标签统计 + 重标定增益 + 代理符号
  results/e6_law_ml.csv          变体 × (CV/盲测) × fset × model 指标
  results/e6_gbe_proxy.csv       基准标签 GBE 代理
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
D6 = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(D6, "ml"))
from e3_postprocess import (load_e1_spectra, damage_labels, GAMMA0, EPS_REF)  # noqa: E402
from e4_blindtest import (design, make_models, conformal_q,  # noqa: E402
                          cv_ablation, SETS, SEED, N_TRAIN_G, N_CAL_G)
from sklearn.metrics import roc_auc_score, r2_score, mean_absolute_error  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

KAPPA = 0.7
POTS = ["chgnet", "mace-mp0-medium", "mace-mp0-small"]
METALS = ("Cu", "Ni", "Fe")
VARIANTS = {                      # name: (p, m, decoupled)
    "base": (0.65, 2.0, False),
    "m1":   (0.65, 1.0, False),
    "p05":  (0.50, 2.0, False),
    "p10":  (1.00, 2.0, False),
    "dec":  (0.65, 2.0, True),
}


def load_labels():
    """基准 + 扩展标签 → 特征合并表。E6 只用前 len(e3_labels) 行。"""
    lab = pd.read_csv(os.path.join(D6, "labels", "e3_labels.csv"))
    ext = os.path.join(D6, "labels", "e3_labels_ext.csv")
    n_ext = 0
    if os.path.exists(ext):
        lab = pd.concat([lab, pd.read_csv(ext)], ignore_index=True)
        n_ext = lab.shape[0] - 360
    fea = pd.read_csv(os.path.join(D6, "results", "network_features.csv"))
    df = lab.merge(fea.drop(columns=["n_grains"]), on=["sample", "metal", "potential"])
    df["fail"] = 1 - df["censored"]
    return df, n_ext


def load_curves():
    cur = {}
    root = os.path.join(D6, "labels", "e3_curves")
    for fn in sorted(os.listdir(root)):
        if fn.endswith(".npz"):
            cur[fn[:-4]] = dict(np.load(os.path.join(root, fn), allow_pickle=True))
    return cur


def build_gammas(spec, samples, p):
    """{(s,metal,pot): (γ+γ₀)^p 向量}，采样规则与 E3 stage_labels 完全一致。"""
    gam = {}
    for s in samples:
        d = np.load(os.path.join(D6, "networks", "library", f"sample_{int(s):03d}.npz"),
                    allow_pickle=True)
        et = d["etype"]
        for metal in METALS:
            g0 = GAMMA0[metal]
            for pot in POTS:
                rng = np.random.default_rng(10_000 + int(s))   # 每 (s,metal,pot) 重建
                tw = spec[pot][metal]["twin"] or spec[pot][metal]["general"]
                gn = spec[pot][metal]["general"]
                gam[(int(s), metal, pot)] = np.maximum(
                    np.array([rng.choice(tw) if e == "twin" else rng.choice(gn)
                              for e in et]) + g0, 1e-6) ** p
    return gam


def calibrate_k(curves, gam, samples, decoupled):
    """k(metal,pot) = κ·median_runs[σ_ref/⟨g̃⟩]；decoupled 时 g̃≡1。"""
    k = {}
    for metal in METALS:
        for pot in POTS:
            ratios = []
            for s in samples:
                c = curves.get(f"{int(s):03d}_{metal}")
                if c is None:
                    continue
                eps, sn = c["eps"], c["sigma_n"].astype(np.float64)
                if eps[-1] < EPS_REF:
                    continue
                sig_ref = float(np.interp(EPS_REF, eps, sn.mean(axis=0)))
                gbar = 1.0 if decoupled else float(np.mean(gam[(int(s), metal, pot)]))
                ratios.append(sig_ref / gbar)
            k[(metal, pot)] = float(np.median(ratios)) * KAPPA if ratios else np.nan
    return k


def run_rows(df_rows, curves, gam, k, gains, m_exp, dec):
    """对 df_rows 逐行跑损伤律 → (eps, censored) 列表。σ_c 增益按金属乘。"""
    eps_out, cens_out = [], []
    for r in df_rows.itertuples():
        c = curves[f"{int(r.sample):03d}_{r.metal}"]
        g = (np.ones_like(gam[(int(r.sample), r.metal, r.potential)]) if dec
             else gam[(int(r.sample), r.metal, r.potential)])
        ef, _fr, ce, _ = damage_labels(
            c["sigma_n"].astype(np.float64), c["eps"], g,
            c["edge_g1"], c["edge_g2"], c["edge_n"],
            c["edge_wx"], c["edge_wy"], int(c["n_grains"]),
            k[(r.metal, r.potential)] * gains[r.metal], m=m_exp)
        eps_out.append(ef)
        cens_out.append(ce)
    return eps_out, cens_out


def solve_gains(df, curves, gam, k, m_exp, dec, targets, iters=40):
    """每金属对数二分求增益 a，使该金属贯通计数最接近 base 目标。"""
    gains = {}
    for metal in METALS:
        sub = df[df.metal == metal]
        tgt = int(targets[metal])

        def count_at(a):
            g = {mm: (a if mm == metal else 1.0) for mm in METALS}
            _, cens = run_rows(sub, curves, gam, k, g, m_exp, dec)
            return int(np.sum(1 - np.array(cens)))

        lo, hi, best = 1e-3, 1e3, None
        # 确保计数包住目标（a↑ ⇒ 计数↓）
        while count_at(hi) > tgt and hi < 1e6:
            hi *= 10
        while count_at(lo) < tgt and lo > 1e-6:
            lo /= 10
        for _ in range(iters):
            mid = float(np.sqrt(lo * hi))
            c = count_at(mid)
            if best is None or abs(c - tgt) < abs(best[1] - tgt):
                best = (mid, c)
            if c == tgt:
                break
            if c > tgt:      # 太多贯通 → 提高 σ_c
                lo = mid
            else:
                hi = mid
        gains[metal] = best[0]
        print(f"  [gain] {metal}: a={best[0]:.4f} → count {best[1]}（目标 {tgt}）")
    return gains


def split_orig(df_base):
    """原 8 组织盲测划分（与 e4 完全同配方）。"""
    g = df_base["sample"].unique().copy()
    np.random.default_rng(SEED).shuffle(g)
    gtr, gca, gte = g[:N_TRAIN_G], g[N_TRAIN_G:N_TRAIN_G + N_CAL_G], g[N_TRAIN_G + N_CAL_G:]
    return gtr, gca, gte


def blind_metrics(dfv, gtr, gca, gte):
    """原协议盲测：cls AUC + reg R²/MAE/conformal 覆盖（per fset × model）。"""
    eps = dfv["eps_fail"].to_numpy(float)
    mtr = dfv["sample"].isin(gtr).to_numpy()
    mca = dfv["sample"].isin(gca).to_numpy()
    mte = dfv["sample"].isin(gte).to_numpy()
    recs = []
    for fset in ("T", "G", "TG"):
        X = design(dfv, SETS[fset])
        for target in ("cls", "reg"):
            ok = np.ones(len(dfv), bool) if target == "cls" else ~np.isnan(eps)
            for mn, mo in make_models(target, SEED).items():
                tr, ca, te = mtr & ok, mca & ok, mte & ok
                ytr = (dfv["fail"].to_numpy() if target == "cls" else eps)[tr]
                if tr.sum() < 30 or (target == "reg" and ca.sum() < 5):
                    recs.append(dict(target=target, model=mn, fset=fset,
                                     auc=np.nan, r2=np.nan, mae=np.nan,
                                     cov=np.nan, n=int(te.sum())))
                    continue
                mo.fit(X[tr], ytr)
                if target == "cls":
                    p = mo.predict_proba(X[mte])[:, 1]
                    y = dfv["fail"].to_numpy()[mte]
                    auc = (roc_auc_score(y, p)
                           if len(np.unique(y)) == 2 else np.nan)
                    recs.append(dict(target=target, model=mn, fset=fset,
                                     auc=auc, r2=np.nan, mae=np.nan,
                                     cov=np.nan, n=len(y)))
                else:
                    q = conformal_q(np.abs(eps[ca] - mo.predict(X[ca])))
                    p = mo.predict(X[te])
                    y = eps[te]
                    recs.append(dict(target=target, model=mn, fset=fset, auc=np.nan,
                                     r2=r2_score(y, p) if len(y) > 2 else np.nan,
                                     mae=mean_absolute_error(y, p) if len(y) else np.nan,
                                     cov=float(np.mean(np.abs(y - p) <= q))
                                     if len(y) else np.nan, n=len(y)))
    return pd.DataFrame(recs)


def main():
    (df_all, n_ext), curves = load_labels(), load_curves()
    lab0 = pd.read_csv(os.path.join(D6, "labels", "e3_labels.csv"))
    df = df_all.iloc[:len(lab0)].copy()      # E6 固定在原 360 标签（与已发表数字可比）
    samples = sorted(df["sample"].unique())
    print(f"[load] {len(df)} 行（原 360；扩展标签 {n_ext} 行仅供 E4b 盲测）、"
          f"{len(samples)} 组织、{len(curves)} 条曲线")

    # ---- 基准复现 assert ----
    assert (df["censored"].to_numpy() == lab0["censored"].to_numpy()).all()
    assert np.allclose(df["eps_fail"].fillna(-1), lab0["eps_fail"].fillna(-1),
                       atol=1e-5), "载入标签与 e3_labels.csv 不一致"

    gtr, gca, gte = split_orig(df)
    print(f"[split] 原协议 train/cal/test = {len(gtr)}/{len(gca)}/{len(gte)} 组织")

    # ---- GBE 实验代理（基准标签） ----
    proxy = []
    for scope, sub in [("ALL", df)] + [(m, df[df.metal == m]) for m in METALS]:
        r1, p1 = spearmanr(sub["twin_length_frac"], sub["fail"])
        sp = sub[sub["fail"] == 1]
        r2, p2 = (spearmanr(sp["twin_length_frac"], sp["eps_fail"])
                  if len(sp) > 5 else (np.nan, np.nan))
        proxy.append(dict(scope=scope, r_fail=r1, p_fail=p1,
                          r_eps=r2, p_eps=p2, n=len(sub)))
    proxy = pd.DataFrame(proxy)
    print("\n[e6 GBE 代理] spearman(twin_length_frac, …) 基准标签")
    print(proxy.round(3).to_string(index=False))

    # ---- base 贯通计数（重标定目标） ----
    targets = df.groupby("metal")["fail"].sum().astype(int).to_dict()
    print(f"[target] 每金属贯通计数 {targets}")

    # ---- 每变体 ----
    spec = load_e1_spectra()
    stats, ml_rows = [], []
    for name, (p, m_exp, dec) in VARIANTS.items():
        gam = build_gammas(spec, samples, p)
        k = calibrate_k(curves, gam, samples, dec)
        if dec:    # 解耦 k 必须跨势恒同（g̃≡1 ⇒ 只剩 σ_ref 中位）
            for metal in METALS:
                vals = [k[(metal, pot)] for pot in POTS]
                assert max(vals) - min(vals) < 1e-9 * abs(vals[0]), \
                    f"dec k 跨势漂移 {metal}"

        if name == "base":
            gains = {mm: 1.0 for mm in METALS}
        else:
            print(f"[solve] {name}:")
            gains = solve_gains(df, curves, gam, k, m_exp, dec, targets)

        eps_out, cens_out = run_rows(df, curves, gam, k, gains, m_exp, dec)
        dfv = df.copy()
        dfv["eps_fail"] = eps_out
        dfv["censored"] = cens_out
        dfv["fail"] = 1 - dfv["censored"]

        if name == "base":     # 复现 assert
            assert (dfv["censored"].to_numpy() == lab0["censored"].to_numpy()).all(), \
                "base 变体未复现 censored"
            assert np.allclose(dfv["eps_fail"].fillna(-1), lab0["eps_fail"].fillna(-1),
                               atol=1e-5), "base 变体未复现 eps_fail"
        if dec:    # 解耦 ⇒ 标签跨势完全不变
            piv = dfv.pivot_table(index=["sample", "metal"], columns="potential",
                                  values="censored")
            assert (piv.nunique(axis=1) == 1).all(), "dec 标签仍随势变化"

        eps_v = dfv["eps_fail"].to_numpy(float)
        per = ~np.isnan(eps_v)
        pileup = float(np.mean(eps_v[per] <= 0.0031)) if per.any() else np.nan
        agree = float((df["censored"].to_numpy() == dfv["censored"].to_numpy()).mean())
        both = (~np.isnan(df["eps_fail"].to_numpy(float))) & per
        # round 合并输出网格并列值：CSV 往返把 0.009 截成 0.0089999999 会打乱
        # 并列秩，使精确复现（max|Δ|~3e-15）的 rho 仍降到 0.997
        rho = (spearmanr(df["eps_fail"].to_numpy(float)[both].round(6),
                         eps_v[both].round(6)).statistic if both.sum() > 5 else np.nan)

        # GBE 代理在该变体标签下的符号
        r1, _ = spearmanr(dfv["twin_length_frac"], dfv["fail"])
        sig = {}
        for m in METALS:
            sm_ = dfv[dfv.metal == m]
            if sm_["fail"].nunique() > 1:
                sig[m] = spearmanr(sm_["twin_length_frac"], sm_["fail"]).statistic
            else:
                sig[m] = np.nan

        counts = dfv.groupby("metal")["fail"].sum().astype(int).to_dict()
        stats.append(dict(variant=name, p=p, m=m_exp, decoupled=int(dec),
                          a_Cu=gains["Cu"], a_Ni=gains["Ni"], a_Fe=gains["Fe"],
                          count_Cu=counts["Cu"], count_Ni=counts["Ni"],
                          count_Fe=counts["Fe"],
                          percolate_rate=float(dfv["fail"].mean()),
                          pileup_rate=pileup, verdict_agree=agree, rho_eps=rho,
                          gbe_r_all=r1, gbe_r_Cu=sig["Cu"], gbe_r_Fe=sig["Fe"]))

        # ---- ML：分组 CV + 原 8 组织盲测 ----
        Xs = {k_: design(dfv, v) for k_, v in SETS.items()}
        ab = cv_ablation(dfv, Xs)
        sm = (ab.groupby(["target", "model", "fset"])
                .agg(auc=("auc", "mean"), r2=("r2", "mean")).reset_index())
        bl = blind_metrics(dfv, gtr, gca, gte)
        for target in ("cls", "reg"):
            met = "auc" if target == "cls" else "r2"
            for fset in ("T", "G", "TG"):
                for model in ("RF", "Lin"):
                    cvv = sm[(sm.target == target) & (sm.fset == fset)
                             & (sm.model == model)]
                    bb = bl[(bl.target == target) & (bl.fset == fset)
                            & (bl.model == model)]
                    ml_rows.append(dict(
                        variant=name, target=target, model=model, fset=fset,
                        cv=cvv[met].iloc[0] if len(cvv) else np.nan,
                        blind=bb["auc"].iloc[0] if target == "cls"
                        else bb["r2"].iloc[0],
                        blind_cov=bb["cov"].iloc[0], blind_n=bb["n"].iloc[0]))
        s_row = stats[-1]
        print(f"[{name}] rate={s_row['percolate_rate']:.3f} pileup={pileup:.3f} "
              f"agree={agree:.3f} rho_eps={rho:.3f}")

    stats = pd.DataFrame(stats)
    ml = pd.DataFrame(ml_rows)
    stats.to_csv(os.path.join(D6, "results", "e6_label_stats.csv"), index=False)
    ml.round(4).to_csv(os.path.join(D6, "results", "e6_law_ml.csv"), index=False)
    proxy.round(4).to_csv(os.path.join(D6, "results", "e6_gbe_proxy.csv"), index=False)

    print("\n[e6_label_stats]")
    print(stats.drop(columns=["p", "m"]).round(3).to_string(index=False))
    piv = ml[ml.model == "RF"].pivot_table(index=["target", "fset"],
                                            columns="variant",
                                            values=["cv", "blind"])
    print("\n[e6_law_ml RF] cv / blind")
    print(piv.round(3).to_string())

    # ---- 验收断言 ----
    s = stats.set_index("variant")
    assert abs(s.loc["base", "percolate_rate"] - 0.431) < 0.01, "base 贯通率偏离 43%"
    assert s.loc["base", "verdict_agree"] == 1.0 and s.loc["base", "rho_eps"] > 0.9999, \
        "base 变体未逐行复现基准标签"
    for v in ("m1", "p05", "p10", "dec"):     # 重标定须把计数拉回目标附近
        dev = max(abs(s.loc[v, f"count_{m}"] - targets[m]) for m in METALS)
        assert dev <= 3, f"{v} 重标定后计数偏离目标 {dev} > 3"
    assert s.loc["dec", "verdict_agree"] < 1.0, "解耦对照与基准完全同标签，失去意义"
    print("\n[check] 通过：base 复现、四变体重标定命中目标（±3）、dec 解耦生效")


if __name__ == "__main__":
    main()
