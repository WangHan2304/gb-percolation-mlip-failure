#!/usr/bin/env python3
"""E5 误差传播（RQ3）：GB 能扰动 → 网络特征漂移 → 失效预测漂移

RQ3：GB 能的 uMLIP 预测误差沿"GB 能 → 网络特征 → 失效预测"传播多少——
     拓扑特征是否吸收/放大原子级误差？

误差源（实测，零人工噪声模型）：E1 三势能谱池。"一条边的 γ 属于哪个势"是
该边能量预测不确定性的天然实现——势间差异即 MLIP 误差结构（E1：高角层离散
2–3×、孪晶层符号可翻转）。

扰动协议：
  比例 p ∈ {0, 0.25, 0.5, 0.75, 1}——每条边以概率 p 从"另一随机势"的同族池
  重采样（整次 trial 只选一个 alt 势）。p=1 = 完整势间误差（实证锚点：
  E2 γ_perc 跨势分离 1.7×、E3 标签势间翻转）。p=0 为机器零对照。
  基准 γ 向量由 E2 的 rng 序列精确重放（default_rng(42+sample)，与 E2 网格
  生成共流；已全 40 样本验证 mean_gamma 与 network_features.csv 相对差 ≤7e-5，
  仅表值 round(0.1) 舍入）。

链路与对照：
  γ 特征（mean_gamma/gamma_std/gamma_perc×2/failed_cluster×2）→ 重算漂移；
  拓扑特征跨势恒定 → 零传播对照（结构上不可能携带原子级误差）。
  预测层：E4 同配方模型（同 seed 同 24/8/8 划分，train 24 组织）：
    TG-RF：原子级误差经 γ 特征通道 → 预测漂移 Δp̂/Δε̂
    T-RF ：纯拓扑通道 → 漂移恒 0（吸收的结构上限）
  增益（RQ3 量化）：G = 中位相对预测漂移 / 中位相对 γ 输入漂移
    G<1 预测层吸收、G>1 放大；T 模型 G=0（全吸收）。

产物：
  results/error_propagation.csv   tidy 长表：p × (feature|model) × 指标
  figures/e5_propagation.png      特征漂移 / 分类翻转率(+E3 锚) / 回归漂移

自检（main 末尾 assert）：
  重放基准 = E2 表（≤1e-3）；基线 AUC 复现 E4（0.8508）；p=0 与 T 模型零漂移。
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
D6 = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(D6, "networks"))
sys.path.insert(0, HERE)
from rve_networks import (load_e1_spectra, gen_microstructure, edge_types,  # noqa: E402
                          percolation_features)
from e4_blindtest import (load, design, make_models, TOPO, GAMMA_F,  # noqa: E402
                          SEED, N_TRAIN_G)

P_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
MC = 8                       # 每档扰动蒙特卡洛重复数
MC_SEED = 20261023           # 与 E4 同源，扰动流独立派生
REL_COLS = GAMMA_F           # γ 特征列（与 E4 设计同序）


def replay_gamma(s, spec, metals, pots):
    """精确重放 E2 的 rng 序列 → (et, {(metal,pot): γ 向量})。"""
    rng = np.random.default_rng(42 + s)
    tp = float(rng.uniform(0, 0.65))
    ng = int(rng.integers(40, 201))
    ms = gen_microstructure(rng, M=256, n_grains=ng, twin_prob=tp)
    et = edge_types(ms)
    out = {}
    for metal in metals:
        for pot in pots:
            tw = spec[pot][metal]["twin"]
            gn = spec[pot][metal]["general"]
            if not tw:
                tw = gn
            out[(metal, pot)] = np.array(
                [rng.choice(tw) if t == "twin" else rng.choice(gn) for t in et])
    return et, out


def feats_of(gam, n_g, edges, et):
    """6 特征（与 E2 表同定义）。"""
    f = percolation_features(n_g, edges, et, gam)
    return dict(mean_gamma=float(np.mean(gam)), gamma_std=float(np.std(gam)), **f)


def main():
    spec = load_e1_spectra()
    metals = sorted(next(iter(spec.values())).keys())
    pots = sorted(spec.keys())
    df = load()                      # 360 行（E2 特征表 + E3 标签）
    lab = pd.read_csv(os.path.join(D6, "labels", "e3_labels.csv"))

    # ---- 缓存：重放 γ、边表、基准特征/矩阵 ----
    et_s, gam_base, edges_s, ng_s = {}, {}, {}, {}
    for s in sorted(df["sample"].unique()):
        et, gdict = replay_gamma(int(s), spec, metals, pots)
        et_s[int(s)] = et
        d = np.load(os.path.join(D6, "networks", "library", f"sample_{int(s):03d}.npz"),
                    allow_pickle=True)
        edges_s[int(s)], ng_s[int(s)] = d["edges"], int(d["grid"].max()) + 1
        for k, v in gdict.items():
            gam_base[(int(s),) + k] = v

    # 基准特征重算 vs E2 表（重放自检）
    F0 = np.empty((len(df), len(REL_COLS)))
    for i, r in enumerate(df.itertuples()):
        s = int(r.sample)
        f = feats_of(gam_base[(s, r.metal, r.potential)], ng_s[s], edges_s[s], et_s[s])
        F0[i] = [f[c] for c in REL_COLS]
    tbl = df[REL_COLS].to_numpy(float)
    replay_err = float(np.max(np.abs(F0 - tbl) / np.maximum(np.abs(tbl), 1e-9)))
    print(f"[replay] 基准特征 vs E2 表最大相对差 {replay_err:.2e}（round 舍入级）")
    assert replay_err < 1e-3, "重放失败：基准特征与 E2 表不一致"

    X_base = design(df, TOPO + REL_COLS)          # TG 设计（γ 列 = 6..11）
    gi = slice(len(TOPO), len(TOPO) + len(REL_COLS))
    y_cls = df["fail"].to_numpy()
    eps = df["eps_fail"].to_numpy()
    reg_rows = ~np.isnan(eps)
    groups = df["sample"].unique().copy()
    np.random.default_rng(SEED).shuffle(groups)
    gtr = groups[:N_TRAIN_G]
    mtr = df["sample"].isin(gtr).to_numpy()

    # ---- 模型：E4 同配方同划分（基线复现 E4 盲测 AUC）----
    mdl = {}
    mdl[("TG", "cls")] = make_models("cls", SEED)["RF"].fit(X_base[mtr], y_cls[mtr])
    mdl[("T", "cls")] = make_models("cls", SEED)["RF"].fit(
        design(df, TOPO)[mtr], y_cls[mtr])
    X_T = design(df, TOPO)
    mdl[("TG", "reg")] = make_models("reg", SEED)["RF"].fit(X_base[mtr & reg_rows],
                                                             eps[mtr & reg_rows])
    mdl[("T", "reg")] = make_models("reg", SEED)["RF"].fit(X_T[mtr & reg_rows],
                                                            eps[mtr & reg_rows])
    mte = df["sample"].isin(groups[N_TRAIN_G + 8:]).to_numpy()
    from sklearn.metrics import roc_auc_score
    auc0 = roc_auc_score(y_cls[mte], mdl[("TG", "cls")].predict_proba(X_base[mte])[:, 1])
    print(f"[model] 基线 TG cls 盲测 AUC = {auc0:.4f}（E4 = 0.8508）")
    assert abs(auc0 - 0.8508) < 5e-4, "基线未复现 E4"
    p0_cls = mdl[("TG", "cls")].predict_proba(X_base)[:, 1]
    p0_reg = mdl[("TG", "reg")].predict(X_base)
    t0_cls = mdl[("T", "cls")].predict_proba(X_T)[:, 1]
    t0_reg = mdl[("T", "reg")].predict(X_T)

    # ---- 扰动 MC ----
    recs = []
    rel_store = {(p, c): [] for p in P_GRID for c in REL_COLS}
    d_store = {(p, mn, tg): [] for p in P_GRID for mn in ("TG", "T")
               for tg in ("cls", "reg")}
    flip = {(p, mn): [] for p in P_GRID for mn in ("TG", "T")}

    for t in range(MC):
        for p in P_GRID:
            rng = np.random.default_rng(MC_SEED + 104729 * t + int(round(p * 100)))
            Xp = X_base.copy()
            Fp = F0.copy()
            if p > 0:
                for i, r in enumerate(df.itertuples()):
                    s, m, pot = int(r.sample), r.metal, r.potential
                    et, edges, ng = et_s[s], edges_s[s], ng_s[s]
                    base = gam_base[(s, m, pot)]
                    others = [q for q in pots if q != pot]
                    alt = others[int(rng.integers(len(others)))]
                    mask = rng.random(len(et)) < p
                    if mask.any():
                        tw = spec[alt][m]["twin"]
                        gn = spec[alt][m]["general"]
                        if not tw:
                            tw = gn
                        g = base.copy()
                        for j in np.where(mask)[0]:
                            g[j] = rng.choice(tw if et[j] == "twin" else gn)
                        f = feats_of(g, ng, edges, et)
                        Fp[i] = [f[c] for c in REL_COLS]
                        Xp[i, gi] = Fp[i]
            rel = np.abs(Fp - F0) / np.maximum(np.abs(F0), 1e-9)
            for k, c in enumerate(REL_COLS):
                rel_store[(p, c)].extend(rel[:, k].tolist())
            pc = mdl[("TG", "cls")].predict_proba(Xp)[:, 1]
            pr = mdl[("TG", "reg")].predict(Xp)
            tc = mdl[("T", "cls")].predict_proba(X_T)[:, 1]
            tr = mdl[("T", "reg")].predict(X_T)
            d_store[(p, "TG", "cls")].extend(np.abs(pc - p0_cls).tolist())
            d_store[(p, "TG", "reg")].extend(np.abs(pr[reg_rows] - p0_reg[reg_rows]).tolist())
            d_store[(p, "T", "cls")].extend(np.abs(tc - t0_cls).tolist())
            d_store[(p, "T", "reg")].extend(np.abs(tr[reg_rows] - t0_reg[reg_rows]).tolist())
            flip[(p, "TG")].extend(((pc >= .5) != (p0_cls >= .5)).tolist())
            flip[(p, "T")].extend(((tc >= .5) != (t0_cls >= .5)).tolist())
        print(f"[mc] trial {t + 1}/{MC} 完成", flush=True)

    # ---- 汇总 tidy 表 ----
    def q(a, v):
        return float(np.quantile(a, v)) if len(a) else np.nan

    for p in P_GRID:
        for c in REL_COLS:
            recs.append(dict(p=p, scope="feature", name=c, target="",
                             med=q(rel_store[(p, c)], .5), p90=q(rel_store[(p, c)], .9),
                             flip_rate=np.nan, gain=np.nan))
        for mn in ("TG", "T"):
            for tg in ("cls", "reg"):
                arr = d_store[(p, mn, tg)]
                gain = np.nan
                if tg == "reg" and mn == "TG":
                    # arr 按 trial 主序池化（每 trial 155 行）→ 基准值按 trial 平铺对齐
                    rel_pred = arr / np.maximum(np.tile(p0_reg[reg_rows], MC), 1e-6)
                    g_gamma = np.median(rel_store[(p, "mean_gamma")])
                    gain = float(np.median(rel_pred) / g_gamma) if g_gamma > 0 else np.nan
                recs.append(dict(p=p, scope="model", name=mn, target=tg,
                                 med=q(arr, .5), p90=q(arr, .9),
                                 flip_rate=float(np.mean(flip[(p, mn)])) if tg == "cls" else np.nan,
                                 gain=gain))
    out = pd.DataFrame(recs)
    out.round(6).to_csv(os.path.join(D6, "results", "error_propagation.csv"), index=False)

    # E3 实测势间失效判翻转锚点（censored 跨势不一致——与模型二值翻转同类指标）
    flips = int(lab.groupby(["sample", "metal"])["censored"].nunique().gt(1).sum())
    anchor = flips / 120
    print(f"\n[anchor] E3 实测势间失效判翻转 {flips}/120 = {anchor:.3f}"
          f"（censored 跨势不一致；广义标签不稳定 42/120 见 E3 台账）")

    # ---- 验收自检 ----
    z_feat = out[(out.p == 0) & (out.scope == "feature")]["med"].abs().max()
    z_t = out[(out.p == 0) & (out.scope == "model")]["med"].abs().max()
    t_all = out[(out.scope == "model") & (out.name == "T")]["med"].abs().max()
    mono = bool(np.all(np.diff([out[(out.p == p) & (out.scope == "model") &
                                   (out.name == "TG") & (out.target == "cls")]["med"].iloc[0]
                               for p in P_GRID]) >= -0.002))
    f1 = out[(out.p == 1) & (out.name == "gamma_perc_rel")]["med"].iloc[0]
    flip1 = out[(out.p == 1) & (out.scope == "model") & (out.name == "TG")
                & (out.target == "cls")]["flip_rate"].iloc[0]
    gain1 = out[(out.p == 1) & (out.scope == "model") & (out.name == "TG")
                & (out.target == "reg")]["gain"].iloc[0]
    print(f"[check] p=0 零漂移 feat={z_feat:.1e}/model={z_t:.1e}；T 模型最大漂移={t_all:.1e}")
    print(f"[check] γ_perc_rel p=1 中位相对漂移={f1:.3f}；TG 分类翻转率 p=1={flip1:.3f}"
          f"（E3 锚 {anchor:.3f}）；回归传播增益 G(p=1)={gain1:.3f}")
    assert z_feat < 1e-12 and z_t < 1e-12 and t_all < 1e-12, "p=0/T 模型应零漂移"
    assert mono, "TG 分类漂移不随 p 单调（异常）"
    print("[check] 通过")
    fig(out, anchor, flips)


def fig(out, anchor, flips):
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK HK", "Droid Sans Fallback",
                                              "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    ax = axes[0]
    for c in ["mean_gamma", "gamma_std", "gamma_perc_mJm2", "gamma_perc_rel",
              "failed_cluster_auc"]:
        s = out[(out.scope == "feature") & (out.name == c)].sort_values("p")
        ax.plot(s.p, s.med, "o-", label=c)
    ax.axhline(0, color="k", lw=1, ls=":", label="拓扑特征（恒 0）")
    ax.set_xlabel("扰动比例 p"); ax.set_ylabel("中位相对漂移 |Δf|/f")
    ax.set_title("γ 特征漂移 vs 扰动比例")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = axes[1]
    for mn, c in (("TG", "#4C72B0"), ("T", "#DD8452")):
        s = out[(out.scope == "model") & (out.name == mn) & (out.target == "cls")]
        ax.plot(s.p, s.flip_rate, "o-", color=c, label=f"{mn}-RF")
    ax.axhline(anchor, color="k", ls=":", lw=1.2,
               label=f"E3 势间失效判翻转 {anchor:.0%}（{flips}/120）")
    ax.set_xlabel("扰动比例 p"); ax.set_ylabel("分类翻转率")
    ax.set_title("失效判预测翻转 vs 扰动比例")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[2]
    for mn, c in (("TG", "#4C72B0"), ("T", "#DD8452")):
        s = out[(out.scope == "model") & (out.name == mn) & (out.target == "reg")]
        ax.plot(s.p, s.med, "o-", color=c, label=f"{mn}-RF")
    ax.set_xlabel("扰动比例 p"); ax.set_ylabel("中位 |Δ 预测 eps_fail|")
    ax.set_title("回归预测漂移 vs 扰动比例")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(D6, "figures", "e5_propagation.png"), dpi=150)
    plt.close(fig)
    print("[fig] e5_propagation.png 已出")


if __name__ == "__main__":
    main()
