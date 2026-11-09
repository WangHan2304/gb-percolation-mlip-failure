#!/usr/bin/env python3
"""D6 论文图 Fig1–7（英文、统一风格）→ paper/fig1..7_*.png

全部基于已验收产物（results/、labels/、networks/library、e4/e5 CSV），
不重算任何科学量；数据对账以台账登记值为准。
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
D6 = os.path.abspath(os.path.join(HERE, ".."))
C_POT = {"chgnet": "#4C72B0", "mace-mp0-medium": "#DD8452", "mace-mp0-small": "#55A868"}
C_MET = {"Cu": "#C44E52", "Ni": "#4C72B0", "Fe": "#2CA02C"}
M_POT = {"chgnet": "o", "mace-mp0-medium": "s", "mace-mp0-small": "^"}
plt.rcParams.update({"font.size": 11.5, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150})


FIG_DPI = int(os.environ.get("FIG_DPI", "200"))
FIG_OUT = os.path.abspath(os.environ["FIG_OUT"]) if os.environ.get("FIG_OUT") else HERE


def save(fig, name):
    p = os.path.join(FIG_OUT, name)
    fig.savefig(p, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"→ {name}")


def fig1():
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.axis("off")
    boxes = [
        ("E1  Atomic scale\nMLIP GB-energy\nspectra: 17 cells\n$\\times$3 potentials\n(50 valid)",
         "#DCE9F7"),
        ("E2  Micro-\nstructure\n200 Voronoi RVEs\npercolation\nfeatures:\n($\\gamma_{perc}$,\ncluster AUC)",
         "#E4F0DC"),
        ("E3  Failure\nlabels\nDAMASK CPFEM\n+ CDM damage\n360 labels\n(2-stage)",
         "#FBE6D8"),
        ("E4  ML blind\ntest: feature\nablation, split-\nconformal UQ,\napplicability\ndomain",
         "#EEE3F5"),
    ]
    w, h, gap, y = 2.5, 2.0, 0.45, 1.35
    for i, (txt, fc) in enumerate(boxes):
        x = i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                                    fc=fc, ec="0.4", lw=1.1))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=9)
        if i < 3:
            ax.annotate("", xy=(x + w + gap - 0.02, y + h / 2), xytext=(x + w + 0.02, y + h / 2),
                        arrowprops=dict(arrowstyle="->", lw=1.5, color="0.3"))
    # E5 bottom band
    bw = 3 * (w + gap) - gap
    bx = (w + gap)
    ax.add_patch(FancyBboxPatch((bx, -0.30), bw, 1.35, boxstyle="round,pad=0.05",
                                fc="#FDECEC", ec="#C44E52", lw=1.1))
    ax.text(bx + bw / 2, 0.38,
            "E5  Error-propagation audit:\n"
            "GB-energy perturbation $\\rightarrow$ network features\n"
            "$\\rightarrow$ prediction (gain G < 1)",
            ha="center", va="center", fontsize=9, color="#8B2E2E")
    ax.add_patch(FancyArrowPatch((w / 2, y - 0.02), (bx + 0.4, 1.15),
                                 arrowstyle="-|>", mutation_scale=22,
                                 lw=1.8, color="#C44E52"))
    ax.add_patch(FancyArrowPatch((bx + bw * 0.62, 1.15), (3 * (w + gap) + w / 2, y - 0.02),
                                 arrowstyle="-|>", mutation_scale=22,
                                 lw=1.8, color="#C44E52"))
    # post-hoc audits (dashed, hanging off E3 / E4 without changing the pipeline)
    ay, ah = 3.85, 1.75
    audits = [
        (2, "E6  Post-hoc\naudit: re-label\n360 stress\nhistories under\nother damage\nlaws (Sec. 3.7)",
         "#FBE6D8"),
        (3, "E4b Post-hoc\naudit: blind\nset $\\rightarrow$ 20\nunseen\nmicrostructures\n"
            "+ 95% CIs (Sec. 3.4)",
         "#EEE3F5"),
    ]
    for i, txt, fc in audits:
        x = i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, ay), w, ah, boxstyle="round,pad=0.05",
                                    fc=fc, ec="0.55", lw=1.0, ls="--"))
        ax.text(x + w / 2, ay + ah / 2, txt, ha="center", va="center", fontsize=8)
        ax.add_patch(FancyArrowPatch((x + w / 2, y + h + 0.06), (x + w / 2, ay - 0.06),
                                     arrowstyle="-|>", mutation_scale=18,
                                     lw=1.5, color="0.45"))
    ax.set_xlim(-0.25, 4 * (w + gap) - gap + 0.25)
    ax.set_ylim(-0.65, 5.75)
    save(fig, "fig1_pipeline.png")


def fig2():
    d = pd.read_csv(os.path.join(D6, "results", "gb_energy_library_merged.csv"))
    d = d[d["flag"] == "ok"]
    fams = [(3, 111), (3, 112), (5, 130), (5, 210), (5, 310), (9, 221), (11, 113)]
    labs = ["$\\Sigma$3 (111)", "$\\Sigma$3 (112)", "$\\Sigma$5 (130)", "$\\Sigma$5 (210)",
            "$\\Sigma$5 (310)", "$\\Sigma$9 (221)", "$\\Sigma$11 (113)"]
    pots = ["chgnet", "mace-mp0-medium", "mace-mp0-small"]
    ylims = {"Cu": (-25, 4000), "Ni": (-4, 7000), "Fe": (60, 4500)}
    yticks = {"Cu": ([-10, 0, 10, 100, 1000], ["$-10$", "0", "10", "100", "1000"]),
              "Ni": ([0, 10, 100, 1000], ["0", "10", "100", "1000"]),
              "Fe": ([100, 1000], ["100", "1000"])}
    fig = plt.figure(figsize=(7.0, 7.1))
    gs = fig.add_gridspec(2, 4, wspace=0.30, hspace=0.66)
    axes = [fig.add_subplot(gs[0, 0:2]), fig.add_subplot(gs[0, 2:4]),
            fig.add_subplot(gs[1, 1:3])]
    for ax, metal in zip(axes, ["Cu", "Ni", "Fe"]):
        sub = d[d["metal"] == metal]
        for k, (sg, hk) in enumerate(fams):
            vals = []
            for j, pot in enumerate(pots):
                v = sub[(sub["sigma"] == sg) & (sub["hkl"] == hk) & (sub["potential"] == pot)]
                if len(v):
                    g = float(v["gamma_mJ_per_m2"].iloc[0])
                    vals.append(g)
                    ax.plot(k + (j - 1) * 0.16, g, M_POT[pot], color=C_POT[pot],
                            ms=5.5, mec="white", mew=0.5, zorder=3)
            if len(vals) > 1:
                ax.vlines(k, min(vals), max(vals), color="0.5", lw=1.5, zorder=2)
        ax.axhline(0, color="0.55", lw=0.8, ls="--")
        ax.set_yscale("symlog", linthresh=10)
        tk, tl = yticks[metal]
        ax.set_yticks(tk)
        ax.set_yticklabels(tl, fontsize=11.5)
        ax.set_ylim(*ylims[metal])
        ax.set_xticks(range(len(fams)), labs, fontsize=11, rotation=90)
        ax.set_xlim(-0.6, len(fams) - 0.4)
        ax.set_title(metal, fontsize=13, fontweight="bold")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("GB energy $\\gamma$ (mJ m$^{-2}$)")
    axes[0].annotate("twin-layer sign flip\n$-14.4$ (chgnet)", xy=(-0.16, -14.35),
                     xytext=(0.5, 60), va="top", fontsize=11.5, color="#C44E52",
                     arrowprops=dict(arrowstyle="->", color="#C44E52", lw=0.9))
    axes[2].axhspan(340, 510, facecolor="#C44E52", alpha=0.10, zorder=1)
    axes[2].annotate("DFT 340\u2013510", xy=(6.55, 430), ha="right", va="center",
                     fontsize=11, color="#C44E52")
    axes[2].annotate("3-potential consensus\n114\u2013177", xy=(1.0, 145),
                     xytext=(1.55, 65), va="bottom", fontsize=11.5, color="#8172B2",
                     arrowprops=dict(arrowstyle="->", color="#8172B2", lw=0.9))
    hs = [plt.Line2D([], [], marker=M_POT[p], color=C_POT[p], ls="", ms=5.5,
                     label=p) for p in pots]
    hs.append(plt.Line2D([], [], color="0.5", lw=1.5,
                         label="3-potential range"))
    fig.legend(handles=hs, loc="upper center", ncol=4, frameon=False, fontsize=11,
               bbox_to_anchor=(0.5, 1.035), columnspacing=1.0, handletextpad=0.4,
               labelspacing=0.4)
    fig.tight_layout()
    save(fig, "fig2_energy_spectra.png")


def fig3():
    d = np.load(os.path.join(D6, "networks", "library", "sample_002.npz"),
                allow_pickle=True)
    f = pd.read_csv(os.path.join(D6, "results", "network_features.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 4.6))
    ax = axes[0]
    im = ax.imshow(d["grid"], cmap="tab20", interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("grain ID", fontsize=11.5)
    cb.ax.tick_params(labelsize=11.5)
    ax = axes[1]
    M_MET = {"Cu": "o", "Ni": "s", "Fe": "^"}
    from matplotlib import patheffects as pe
    for m in ["Cu", "Ni", "Fe"]:
        s = f[f["metal"] == m]
        ax.scatter(s["twin_length_frac"], s["gamma_perc_rel"], s=16, marker=M_MET[m],
                   alpha=0.5, color=C_MET[m], linewidths=0, zorder=2)
        k, b = np.polyfit(s["twin_length_frac"], s["gamma_perc_rel"], 1)
        xs = np.array([f["twin_length_frac"].min(), f["twin_length_frac"].max()])
        ax.plot(xs, k * xs + b, color=C_MET[m], lw=2.0, zorder=4,
                path_effects=[pe.withStroke(linewidth=3.6, foreground="white")])
    r_all = np.corrcoef(f["twin_length_frac"], f["gamma_perc_rel"])[0, 1]
    rs = {m: np.corrcoef(g["twin_length_frac"], g["gamma_perc_rel"])[0, 1]
          for m, g in f.groupby("metal")}
    ax.text(0.03, 0.97, f"Pearson $r$ = {r_all:.2f}\n"
                        f"(per metal: {rs['Cu']:.2f} / {rs['Ni']:.2f} / "
                        f"{rs['Fe']:.2f})",
            transform=ax.transAxes, va="top", fontsize=11.5, linespacing=1.15,
            bbox=dict(fc="white", ec="none", alpha=0.85, pad=2.0), zorder=5)
    ax.set_xlabel("twin-boundary length fraction")
    ax.set_ylabel("$\\gamma_{perc} / \\langle\\gamma\\rangle$")
    hs = [plt.Line2D([], [], ls="-", lw=2.0, color=C_MET[m], marker=M_MET[m],
                     ms=6, label=m) for m in ["Cu", "Ni", "Fe"]]
    ax.legend(handles=hs, fontsize=11.5, frameon=True, facecolor="white",
              framealpha=0.9, edgecolor="0.85", loc="upper left",
              bbox_to_anchor=(0.03, 0.84))
    ax.grid(alpha=0.25)
    fig.tight_layout()
    save(fig, "fig3_network_features.png")


def fig4():
    lab = pd.read_csv(os.path.join(D6, "labels", "e3_labels.csv"))
    pots = ["chgnet", "mace-mp0-medium", "mace-mp0-small"]
    short = {"chgnet": "chg", "mace-mp0-medium": "med", "mace-mp0-small": "sml"}
    fig = plt.figure(figsize=(7.0, 6.7))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 1], hspace=0.95, wspace=0.26)
    ax = fig.add_subplot(gs[0, :])
    cens = {m: lab[lab["metal"] == m]["censored"].mean() for m in ("Cu", "Ni", "Fe")}
    data, pos, seq = [], [], []
    for m in ["Cu", "Ni", "Fe"]:
        for p in pots:
            v = lab[(lab["metal"] == m) & (lab["potential"] == p)]["eps_fail"].dropna()
            data.append(v.values)
            pos.append(len(pos))
            seq.append(p)
    bp = ax.boxplot(data, positions=pos, widths=0.55, patch_artist=True,
                    showfliers=True)
    for patch, p in zip(bp["boxes"], seq):
        patch.set(facecolor=C_POT[p], alpha=0.6, edgecolor=C_POT[p], lw=1.3)
    for el in ("whiskers", "caps"):
        for ln in bp[el]:
            ln.set(color="0.4", lw=1.1)
    for ln in bp["medians"]:
        ln.set(color="0.15", lw=1.6)
    for fl in bp["fliers"]:
        fl.set(marker="o", markersize=3.5, markerfacecolor="0.55",
               markeredgecolor="none", alpha=0.8)
    ax.set_xticks(range(9), [short[p] for p in pots] * 3, fontsize=11, rotation=90)
    ax.set_xlim(-0.7, 8.7)
    for x in (1, 4, 7):
        ax.axvline(x + 0.5, color="0.85", lw=0.8)
    from matplotlib.transforms import blended_transform_factory
    bt = blended_transform_factory(ax.transAxes, ax.transAxes)
    for x, m in zip((1, 4, 7), ("Cu", "Ni", "Fe")):
        ax.text((x + 0.7) / 9.4, -0.37, m, transform=bt, ha="center", fontsize=13,
                fontweight="bold", clip_on=False)
        ax.text(x, 0.0735, f"censored {cens[m]:.0%}", ha="center", fontsize=10.5,
                color="0.35", clip_on=False,
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
    ax.set_ylabel("$\\varepsilon_{fail}$ at percolation")
    ax.set_ylim(0, 0.08)
    ax.grid(axis="y", alpha=0.25)
    med = lab[lab["eps_fail"].notna()]["failed_GB_len_frac"].median()
    for j, m in enumerate(["Cu", "Ni", "Fe"]):
        ax = fig.add_subplot(gs[1, j])
        s = lab[(lab["metal"] == m) & lab["eps_fail"].notna()]
        ax.scatter(s["eps_fail"], s["failed_GB_len_frac"], s=11, alpha=0.85,
                   color=C_MET[m], linewidths=0)
        ax.axhline(med, color="0.4", ls="--", lw=1)
        if j == 0:
            ax.text(0.0115, med + 0.011, f"median = {med:.3f}", fontsize=11,
                    color="0.35", va="bottom",
                    bbox=dict(fc="white", ec="none", alpha=0.9, pad=1.5))
            ax.set_ylabel("failed GB length fraction")
        ax.set_xlim(0.005, 0.063)
        ax.set_ylim(0.04, 0.29)
        ax.set_title(m, fontsize=12, fontweight="bold")
        ax.set_xlabel("$\\varepsilon_{fail}$")
        ax.grid(alpha=0.25)
    save(fig, "fig4_labels.png")


def fig5():
    ab = pd.read_csv(os.path.join(D6, "results", "e4_ablation.csv"))
    aud = pd.read_csv(os.path.join(D6, "results", "prediction_audit.csv"))
    fig = plt.figure(figsize=(7.0, 5.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1.0],
                          wspace=0.34, hspace=0.48)
    # 森林图：y = 6 行（T/G/TG × RF/Lin），x = 5×4 group-CV mean ± sd
    forest = [("cls", "auc", "auc_sd", "ROC-AUC", (0.67, 0.92),
               [0.7, 0.75, 0.8, 0.85, 0.9], None),
              ("reg", "r2", "r2_sd", "$R^{2}$", (-0.1, 0.70),
               [0.0, 0.2, 0.4, 0.6], 0.0)]
    fsets, models = ["T", "G", "TG"], ["RF", "Lin"]
    pos = {(f, m): 2 * i + j for i, f in enumerate(fsets)
           for j, m in enumerate(models)}
    cols = {"RF": "#4C72B0", "Lin": "#DD8452"}
    ylabels = [f"{f} · {m}" for f in fsets for m in models]
    for row, (tgt, met, sd, xlab, xlim, xticks, ref) in zip((0, 1), forest):
        ax = fig.add_subplot(gs[row, 0])
        for (f, m), y in pos.items():
            s = (ab[(ab["target"] == tgt) & (ab["model"] == m)]
                 .set_index("fset").loc[f])
            v, e = float(s[met]), float(s[sd])
            ax.plot([v - e, v + e], [y, y], "-", color=cols[m], lw=1.4,
                    solid_capstyle="round", zorder=2)
            ax.plot(v, y, "o", color=cols[m], ms=5.5, zorder=3)
        if ref is not None:
            ax.axvline(ref, color="0.55", ls=":", lw=1, zorder=1)
        ax.set_xlim(*xlim)
        ax.set_xticks(xticks)
        ax.set_xlabel(xlab, fontsize=11.5)
        ax.set_yticks(list(pos.values()))
        ax.set_yticklabels(ylabels, fontsize=11.5)
        ax.set_ylim(-0.75, 5.75)
        ax.invert_yaxis()
        ax.tick_params(axis="x", labelsize=11.5)
        ax.grid(axis="x", alpha=0.3)
    # 右列：40 行逐行 90% conformal 区间覆盖图（组=8 个微结构）
    ax = fig.add_subplot(gs[:, 1])
    a = aud[(aud["fset"] == "TG") & (aud["model"] == "RF") & (aud["target"] == "reg")
            & aud["y_true"].notna()]
    a = a.sort_values(["sample", "metal", "potential"]).reset_index(drop=True)
    ys, y, last = [], 0.0, None
    for _, r in a.iterrows():
        if last is not None and r["sample"] != last:
            y += 1.0                      # 组间空一行距
        ys.append(y)
        y += 1.0
        last = r["sample"]
    a["y"] = ys
    for i, (smp, g) in enumerate(a.groupby("sample")):
        if i % 2 == 0:                    # 交替浅灰组带
            ax.axhspan(g["y"].min() - 0.5, g["y"].max() + 0.5,
                       facecolor="0.93", zorder=0)
        ax.text(-0.02, g["y"].mean(), f"S{smp}", ha="right", va="center",
                fontsize=11.5, color="0.15", transform=ax.get_yaxis_transform(),
                clip_on=False)
    for mask, c, lb in ((a["in_domain"], "#4C72B0", "90% interval (in-domain)"),
                        (~a["in_domain"], "#C44E52", "90% interval (OOD)")):
        s = a[mask]
        ax.hlines(s["y"], s["lo"], s["hi"], color=c, lw=3.2, alpha=0.75,
                  label=lb, zorder=2)
    ax.plot(a["y_true"], a["y"], "|", color="k", ms=9, mew=1.7, zorder=3,
            label="true value")
    ax.set_yticks([])                      # 行号无意义，去掉
    ax.set_xlim(0, 0.078)
    ax.set_xticks([0, 0.02, 0.04, 0.06])
    ax.tick_params(axis="x", labelsize=11.5)
    ax.set_ylim(a["y"].max() + 1.4, -1.4)   # 组 S9 在顶
    ax.set_xlabel("$\\varepsilon_{fail}$")
    ax.grid(axis="x", alpha=0.3)
    hs = [plt.Line2D([], [], ls="-", lw=1.4, color="#4C72B0", marker="o",
                     ms=5.5, label="Random forest"),
          plt.Line2D([], [], ls="-", lw=1.4, color="#DD8452", marker="o",
                     ms=5.5, label="Linear"),
          plt.Line2D([], [], ls="", marker="|", ms=11, mew=1.8, color="k",
                     label="true value"),
          plt.Line2D([], [], ls="-", lw=3.2, color="#4C72B0", alpha=0.75,
                     label="90% interval"),
          plt.Line2D([], [], ls="-", lw=3.2, color="#C44E52", alpha=0.75,
                     label="OOD row")]
    fig.legend(hs, [h.get_label() for h in hs], loc="upper center",
               bbox_to_anchor=(0.5, 1.0), ncol=5, frameon=False, fontsize=11.5,
               columnspacing=1.0, handletextpad=0.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save(fig, "fig5_ablation_blind.png")


def fig6():
    aud = pd.read_csv(os.path.join(D6, "results", "prediction_audit.csv"))
    a = aud[(aud["fset"] == "TG") & (aud["model"] == "RF") & (aud["target"] == "reg")
            & aud["y_true"].notna()].copy()
    # 精确复算 E4 的 τ（cal d5 的 95 分位），避免用子集内 in-domain 最大值误标
    import sys
    sys.path.insert(0, os.path.join(D6, "ml"))
    from e4_blindtest import load, design, SETS, SEED, N_TRAIN_G, N_CAL_G, domain_distance
    df = load()
    X = design(df, SETS["TG"])
    g = df["sample"].unique().copy()
    rng = np.random.default_rng(SEED)
    rng.shuffle(g)
    gtr, gca = g[:N_TRAIN_G], g[N_TRAIN_G:N_TRAIN_G + N_CAL_G]
    d_cal, _, _ = domain_distance(X[df["sample"].isin(gtr).to_numpy()],
                                  X[df["sample"].isin(gca).to_numpy()],
                                  X[df["sample"].isin(g[N_TRAIN_G + N_CAL_G:]).to_numpy()])
    tau = float(np.quantile(d_cal, 0.95))
    assert a.loc[a["in_domain"], "d5"].max() <= tau <= a.loc[~a["in_domain"], "d5"].min(), \
        "τ 与 audit 域标志不一致"
    a["res"] = (a["y_true"] - a["y_pred"]).abs()
    fig, ax = plt.subplots(figsize=(6.4, 4.5))
    xlo, xhi = a["d5"].min() - 0.1, a["d5"].max() + 0.12
    ax.set_xlim(xlo, xhi)
    ax.axvspan(tau, xhi, facecolor="#C44E52", alpha=0.10, zorder=0)
    ax.scatter(a[a["in_domain"]]["d5"], a[a["in_domain"]]["res"], s=20,
               color="#4C72B0", edgecolors="white", linewidths=0.5, alpha=0.9,
               label="in-domain")
    ax.scatter(a[~a["in_domain"]]["d5"], a[~a["in_domain"]]["res"], s=32,
               color="#C44E52", marker="^", edgecolors="white", linewidths=0.5,
               label="OOD")
    ax.axvline(tau, color="k", ls="--", lw=1.1)
    ax.text(tau + 0.05, 0.022, f"$\\tau$ = {tau:.2f}", rotation=90,
            va="center", ha="left", fontsize=11, color="0.25")
    ax.set_xlabel("applicability-domain distance $d_5$")
    ax.set_ylabel("absolute residual")
    ax.set_ylim(0, 0.035)
    ax.legend(fontsize=11.5, frameon=False, loc="upper center", ncol=2)
    ax.grid(alpha=0.25)
    save(fig, "fig6_domain.png")


def fig7():
    out = pd.read_csv(os.path.join(D6, "results", "error_propagation.csv"))
    anchor = 13 / 120
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 4.4))
    ax = axes[0]
    hs1 = []
    for c, col, lb in zip(
            ["mean_gamma", "gamma_std", "gamma_perc_mJm2", "gamma_perc_rel",
             "failed_cluster_auc"],
            ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"],
            ["mean $\\gamma$", "std $\\gamma$", "$\\gamma_{perc}$ (mJ m$^{-2}$)",
             "$\\gamma_{perc}$ / $\\langle\\gamma\\rangle$", "cluster AUC"]):
        s = out[(out["scope"] == "feature") & (out["name"] == c)].sort_values("p")
        ln, = ax.plot(s["p"], s["med"], "o-", color=col, ms=4, label=lb)
        hs1.append(ln)
    ln0 = ax.axhline(0, color="k", lw=1, ls=":", label="topology features (=0)")
    hs1.append(ln0)
    ax.set_xlabel("perturbed GB fraction $p$")
    ax.set_ylabel("median relative drift |$\\Delta f$| / $f$")
    ax.grid(alpha=0.3)
    ax = axes[1]
    for mn, c in (("TG", "#4C72B0"), ("T", "#DD8452")):
        s = out[(out["scope"] == "model") & (out["name"] == mn) & (out["target"] == "cls")]
        ax.plot(s["p"], s["flip_rate"], "o-", color=c, ms=4,
                label=f"{mn}-RF")
    ax.axhline(anchor, color="k", ls=":", lw=1.2)
    ax.text(0.02, 0.101, "E3 label flips (11%)", va="top", fontsize=11.5, color="0.15")
    ax.set_xlabel("perturbed GB fraction $p$")
    ax.set_ylabel("failure-verdict flip rate")
    ax.legend(fontsize=11.5, frameon=False, loc="lower right", bbox_to_anchor=(1.0, 0.03))
    ax.grid(alpha=0.3)
    ax = axes[2]
    for mn, c in (("TG", "#4C72B0"), ("T", "#DD8452")):
        s = out[(out["scope"] == "model") & (out["name"] == mn) & (out["target"] == "reg")]
        ax.plot(s["p"], s["med"], "o-", color=c, ms=4, label=f"{mn}-RF")
    g1 = out[(out["p"] == 1) & (out["scope"] == "model") & (out["name"] == "TG")
             & (out["target"] == "reg")]["gain"].iloc[0]
    ax.set_xlabel("perturbed GB fraction $p$")
    ax.set_ylabel("median |$\\Delta\\hat{\\varepsilon}_{fail}$|")
    ax.legend(fontsize=11.5, frameon=False, loc="upper left")
    ax.annotate(f"$G$ at $p$=1:\n{g1:.2f} $<$ 1", xy=(0.97, 0.0005),
                ha="right", va="bottom", fontsize=11.5, color="0.15",
                linespacing=1.2)
    ax.grid(alpha=0.3)
    fig.tight_layout(rect=[0, 0, 1, 0.85])
    fig.legend(handles=hs1, loc="upper center", ncol=3, frameon=False, fontsize=11.5,
               bbox_to_anchor=(0.5, 1.0), columnspacing=1.0, handletextpad=0.4,
               labelspacing=0.4)
    save(fig, "fig7_propagation.png")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6(); fig7()
    print("全部论文图完成")
