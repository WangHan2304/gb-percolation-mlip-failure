#!/usr/bin/env python3
"""Fe γ0 探针：twin-fraction proxy 金属依赖性的归因小消融（Sec. 3.7 补强）

问题（审稿 2.4）：基准标签中 spearman(twin_length_frac, percolation verdict)
Fe = -0.303 与 Cu(+0.658) / Ni(+0.587) 反号。两个候选解释：
  (i)  γ0(Fe)=0 使 Σ3 相对高角边不构成显著弱链（能量通道弱 → 归因 γ0）；
  (ii) 低渗流率下的拓扑分割主导（γ0 换大也不翻转 → 归因拓扑）。

做法：仅把 Fe 的 γ0 从 0 换成 22（Cu 锚）/ 70（Ni 锚），按 E3 同流程
重采样 γ̃、重标定 k、重跑损伤积分（复用 e6_damage_law，无需新 CPFEM），
并按 E6 协议对每金属 count-matched（对齐 Fe 35，与 p05/p10/dec 同口径），
同时记录自然增益（a=1）下的贯通计数作参照。Cu/Ni 的 γ0 不动，其
verdict 与 Spearman 应逐位不变（探针只碰 Fe）。

自检（不通过即退出）：
  γ0=0 + a=1 逐行复现 labels/e3_labels.csv 的 censored / eps_fail，
  Fe 贯通计数 35/120，Fe Spearman ≈ -0.303。

产物：results/e6_fe_gamma0_probe.csv
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
D6 = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(D6, "ml"))

from scipy.stats import spearmanr  # noqa: E402
import e6_damage_law as E6  # noqa: E402

G0_VARIANTS = [0.0, 22.0, 70.0]


def recompute(df, curves, samples, g0_fe, count_matched, targets):
    """改 Fe γ0 → 重采样 γ̃ → 重标定 k →（可选 count-match）→ 逐行损伤积分。"""
    E6.GAMMA0["Fe"] = g0_fe                     # build_gammas 读此 dict
    gam = E6.build_gammas(E6.load_e1_spectra(), samples, p=0.65)
    k = E6.calibrate_k(curves, gam, samples, decoupled=False)
    gains = (E6.solve_gains(df, curves, gam, k, 2.0, False, targets)
             if count_matched else {m: 1.0 for m in E6.METALS})
    eps_out, cens_out = E6.run_rows(df, curves, gam, k, gains, 2.0, False)
    dfv = df.copy()
    dfv["eps_fail"] = eps_out
    dfv["censored"] = cens_out
    dfv["fail"] = 1 - dfv["censored"]
    return dfv, gains


def main():
    (df_all, _), curves = E6.load_labels(), E6.load_curves()
    lab0 = pd.read_csv(os.path.join(D6, "labels", "e3_labels.csv"))
    df = df_all.iloc[:len(lab0)].copy()          # 原 360 标签，与 E6 同口径
    samples = sorted(df["sample"].unique())
    targets = df.groupby("metal")["fail"].sum().astype(int).to_dict()
    assert targets["Fe"] == 35, f"Fe 基准计数 {targets['Fe']} != 35"
    print(f"[load] {len(df)} 行 / {len(samples)} 组织；targets={targets}")

    # ---- 基准（γ0=0, a=1）逐行复现 ----
    base, _ = recompute(df, curves, samples, 0.0, False, targets)
    assert (base["censored"].to_numpy() == lab0["censored"].to_numpy()).all(), \
        "γ0=0 未复现 censored"
    assert np.allclose(base["eps_fail"].fillna(-1), lab0["eps_fail"].fillna(-1),
                       atol=1e-5), "γ0=0 未复现 eps_fail"
    fe_base = float(spearmanr(*_pair(base, "Fe")).statistic)
    print(f"[check] γ0=0 逐行复现 e3_labels.csv；Fe Spearman={fe_base:.3f}")
    assert abs(fe_base - (-0.303)) < 0.01, f"Fe 基准 Spearman {fe_base:.3f} != -0.303"

    # ---- 变体扫描 ----
    rows = []
    for g0 in G0_VARIANTS:
        for matched in (False, True):
            tag = f"g0={g0:.0f}" + ("+match" if matched else "+natural")
            dfv, gains = recompute(df, curves, samples, g0, matched, targets)
            agree = float((dfv["censored"].to_numpy()
                           == base["censored"].to_numpy()).mean())
            rec = dict(gamma0_fe=g0, count_matched=matched, gain_fe=gains["Fe"],
                       agree_vs_base=agree)
            for m in E6.METALS:
                sub = dfv[dfv.metal == m]
                rec[f"percolated_{m}"] = int(sub["fail"].sum())
                rec[f"spearman_{m}"] = float(
                    spearmanr(sub["twin_length_frac"], sub["fail"]).statistic)
            rows.append(rec)
            print(f"[{tag}] Fe 计数 {rec['percolated_Fe']}/120 "
                  f"a={gains['Fe']:.3f} 同意率={agree:.3f} "
                  f"ρ: Fe={rec['spearman_Fe']:+.3f} "
                  f"Cu={rec['spearman_Cu']:+.3f} Ni={rec['spearman_Ni']:+.3f}")

    out = os.path.join(D6, "results", "e6_fe_gamma0_probe.csv")
    pd.DataFrame(rows).round(4).to_csv(out, index=False)
    print(f"→ {out}")

    # ---- Cu/Ni 不应被扰动（γ0 只改 Fe，k 标定按金属隔离）----
    r = pd.DataFrame(rows)
    ref = r[(r.gamma0_fe == 0) & (~r.count_matched)].iloc[0]
    for m in ("Cu", "Ni"):
        assert (r[f"percolated_{m}"] == ref[f"percolated_{m}"]).all(), f"{m} 计数被扰动"
        assert np.allclose(r[f"spearman_{m}"], ref[f"spearman_{m}"]), \
            f"{m} Spearman 被扰动"
    print("[check] 通过：γ0=0 逐行复现 + Cu/Ni 全程不变")


def _pair(dfv, metal):
    sub = dfv[dfv.metal == metal]
    return sub["twin_length_frac"], sub["fail"]


if __name__ == "__main__":
    main()
