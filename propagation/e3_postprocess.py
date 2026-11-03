#!/usr/bin/env python3
"""E3 后处理：DAMASK 结果 → GB 法向应力历史 → 网格损伤 CDM → 沿晶失效标签

两阶段：
  --stage curves  单 run：读 result.hdf5 → 每边法向应力 σ_n(ε) 曲线 → labels/e3_curves/<run>.npz
                  （提取后 hdf5 可删除；同时校验 cell 排序与网格轴序约定）
  --stage labels  全局：k_metal 标定 + 损伤演化 + 渗流贯通 → labels/e3_labels.csv + sanity 图

损伤当量模型（v2 定稿；每势一套标签，无需重复 CPFEM）：
  σ_c,e = k_metal,pot · (γ_e + γ₀(metal))^p     GB 强度次线性耦合界面能
  dD_e = (⟨σ_n⟩₊/σ_c,e)² dε_xx       m=2，D≥1 永久失效
  失效事件 = 失效边网络沿 y 绕回      DSU winding ⇒ 裂纹面 ⊥ x 贯通 RVE
  标签：eps_fail（贯通时宏观应变）、failed_GB_len_frac、censored
  参数：p=0.65、γ₀=(Cu22/Ni70/Fe0 mJ/m²)、κ=0.7——见下方常量注释

v2 修订依据（2026-10-23 验收发现，v1 σ_c=k·γ 已废弃）：
  - v1 使 γ 动态范围（孪晶 1.4~25 vs 高角 577~3840 mJ/m²，30-60×）直接变成
    强度比：Cu 孪晶 σ_c≈2 MPa ≪ σ_n≈64 MPa → 孪晶首步必死，120/360 标签
    堆死在首个输出点 ε=0.003（占贯通事件 57%），不具可学性；
  - E1 验收报告 §2/§3.5 已记录孪晶层病态值（chgnet Cu γ=−14.4、small Ni −0.39
    负值；三势全低估文献 20-30）并处方"孪晶层作独立不确定度通道"：γ₀ 偏移
    吸收负值与低估，偏移后势间孪晶差（如 chgnet −14.4 vs mace 5.4 → σ_c 差
    ~1.9×）保留为有界不确定度通道；
  - v2 验收（360 标签）：贯通率 43%、首步堆点 0%、eps_fail 铺开 0.019-0.06
    （中位 0.036）、分金属 Cu 64/Ni 56/Fe 35（均 ≥29%），sanity 图已出。
k_metal,pot = κ·median_runs( σ̄_net(ε_ref) / mean_e((γ_e+γ₀)^p) )，κ=0.7 转变区定标
  E2 采样规则不变：twin→E1 Σ3 值，general/twin_related→其余值，seed=10_000+sample。

轴序约定（唯一标记弹性探针实锤，match=1.0000）：
  hdf5 行号 k = iy + M·ix + M²·iz（y 最快、x 次之、z 最慢），第 k 行晶粒 = grid2d[iy,ix]；
  E2 grid2d[i,j] 中 i=row(y)、j=col(x)。多相时 row 须经 (label,entry) 二级映射（单相 run 不需要）。
"""
import argparse, csv, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D6 = os.path.abspath(os.path.join(HERE, ".."))
EPS_REF = 0.05        # k_metal 标定参考应变（N=40 时 ε_max=0.06，留边界余量）
M_EXP = 2.0           # 损伤指数
VOIGT_XX, VOIGT_YY = 0, 1   # 假设 Cauchy 6 分量序 [11,22,33,23,13,12]
# v2：σ_c = k·(γ+γ₀)^p。γ₀ 吸收 E1 孪晶层负值/低估（文献锚点）：
#   Cu 22（含 −14.4 负值需 >14.4）、Ni 70（含 −0.39）、Fe 0（无负值，
#   且偏移会误抬其唯一低能弱链 Σ3(112)=113-177——E1 三势收敛锚点）。
# p=0.65：p=1（正比）首步堆点、p=0.5（Griffith）Fe 网络不贯通（0-2/120），
#   由验收区间定标于 0.65。
GAMMA0 = {"Cu": 22.0, "Ni": 70.0, "Fe": 0.0}
GAMMA_P = 0.65


def load_e1_spectra():
    """{potential: {metal: {"twin":[γ...], "general":[γ...]}}}——与 E2 同规则。"""
    spec = {}
    with open(os.path.join(D6, "results", "gb_energy_library_merged.csv")) as fh:
        for r in csv.DictReader(fh):
            if r["flag"] != "ok":
                continue
            pot, metal = r["potential"], r["metal"]
            g = float(r["gamma_mJ_per_m2"])
            d = spec.setdefault(pot, {}).setdefault(metal, {"twin": [], "general": []})
            (d["twin"] if r["hkl"].strip() == "111" and r["sigma"].strip() == "3"
             else d["general"]).append(g)
    return spec


def edge_segments(grid2d, nz):
    """重扫 2D 网格 → 每条边的界面段：法向 Voigt 分量 + 3D 平铺索引。

    平铺序已用唯一标记弹性探针实锤（/tmp/e3_order，match=1.0000）：
    hdf5 行号 k = iy + M·ix + M²·iz（y 最快、x 次之、z 最慢），第 k 行晶粒 = grid2d[iy,ix]。
    返回 seg：{(g_lo,g_hi): [(voigt_comp, flat_idx ndarray), ...]}。"""
    M = grid2d.shape[0]
    r = grid2d
    d = np.roll(r, -1, axis=0)   # +y 邻居（i+1，绕回）
    rr = np.roll(r, -1, axis=1)  # +x 邻居（j+1，绕回）

    def flat(iy, ix):
        base = iy + M * ix
        return (base[None, :] + (M * M * np.arange(nz))[:, None]).ravel()

    seg = {}
    for a, b, comp, shifty, shiftx in ((r, d, VOIGT_YY, 1, 0), (r, rr, VOIGT_XX, 0, 1)):
        mask = a != b
        iy, ix = np.where(mask)
        idx_p = flat(iy, ix)
        idx_q = flat((iy + shifty) % M, (ix + shiftx) % M)
        ga, gb = a[iy, ix].astype(int), b[iy, ix].astype(int)
        keys = np.stack([np.minimum(ga, gb), np.maximum(ga, gb)], 1)
        for k in range(iy.size):
            seg.setdefault(tuple(keys[k]), []).append(
                (comp, np.concatenate([idx_p[2 * k: 2 * k + 2], idx_q[2 * k: 2 * k + 2]])))
    return seg


def extract_curves(run_dir, out_npz, nz=4):
    """读 result.hdf5 → 每边 σ_n(ε) 与宏观曲线。

    Cauchy 应力由 P、F 自算（本 build 的 Cauchy 输出损坏）：σ = P·Fᵀ/det(F)。
    降采样 run（grid_small.npy 存在）在 64² 网格上扫界面段，再映射回 E2 边表；
    E2 边表中未出现在降采样网络的边按同时刻中位数插补（占比 <1%）。"""
    import h5py
    tag = os.path.basename(run_dir.rstrip("/"))
    d = np.load(os.path.join(D6, "networks", "library", f"sample_{tag[:3]}.npz"),
                allow_pickle=True)
    gs = os.path.join(run_dir, "grid_small.npy")
    grid_scan = np.load(gs) if os.path.exists(gs) else d["grid"]
    M = grid_scan.shape[0]
    edges = d["edges"]
    n_edges = len(edges)
    key2id = {(int(a), int(b)): k for k, (a, b, *_n) in enumerate(edges)}

    f = h5py.File(os.path.join(run_dir, "result.hdf5"), "r")
    cells = f["geometry/cells"][:]
    assert list(cells[:2]) == [M, M], f"cells {cells} 前两维 != [{M},{M}]"
    nz = int(cells[2])   # z 层数以 hdf5 为准（campaign 传过空串 --z，不能信任入参）
    entry = f["cell_to/phase"]["entry"][:].ravel()
    # 单相 run：entry 应为平凡 cell 序号；平铺序 y-fastest 已由标记探针实锤
    assert np.all(entry == np.arange(entry.size)), "cell_to/phase 非平凡序，需复核"
    seg = edge_segments(grid_scan, nz)

    comps, idxs, eids, n_miss = [], [], [], 0
    for key, items in seg.items():
        eid = key2id.get(key, key2id.get((key[1], key[0])))
        if eid is None:                 # 降采样新邻接对：不属于 E2 边表，跳过
            n_miss += 1
            continue
        for comp, idx in items:
            comps.append(np.full(idx.size, comp))
            idxs.append(idx)
            eids.append(np.full(idx.size, eid))
    comps = np.concatenate(comps)
    idxs = np.concatenate(idxs)
    eids = np.concatenate(eids)
    if n_miss:
        print(f"[curves] {tag}: 跳过 {n_miss} 个降采样新增邻接对")
    xx_mask, yy_mask = comps == VOIGT_XX, comps == VOIGT_YY
    counts = np.bincount(eids, minlength=n_edges).astype(float)

    incs = sorted([k for k in f.keys() if k.startswith("increment_")],
                  key=lambda s: int(s.split("_")[1]))
    ph = f["cell_to/phase"]["label"][:].ravel()[0].decode()
    eps, sig_nom = [], []
    sn = np.zeros((n_edges, len(incs)), dtype=np.float32)
    for t, inc in enumerate(incs):
        mech = f[inc]["phase"][ph]["mechanical"]
        F, P = mech["F"][:], mech["P"][:]   # (n,3,3)
        sig = np.einsum("nij,nkj->nik", P, F) / np.linalg.det(F)[:, None, None]
        eps.append(float(np.mean(F[:, 0, 0])) - 1.0)
        sig_nom.append(float(np.mean(P[:, 0, 0])))   # 名义应力（P_xx）
        sxx_cell, syy_cell = sig[:, 0, 0], sig[:, 1, 1]
        s = np.zeros(n_edges)
        for mask, vals in ((xx_mask, sxx_cell), (yy_mask, syy_cell)):
            if mask.any():
                s += np.bincount(eids[mask], weights=vals[idxs[mask]],
                                 minlength=n_edges)
        with np.errstate(invalid="ignore"):
            edge_sig = s / counts
        dead = counts == 0
        if dead.any():                   # 消失边：按同时刻存活边中位数插补
            edge_sig[dead] = np.median(edge_sig[~dead]) if (~dead).any() else 0.0
        sn[:, t] = edge_sig.astype(np.float32)
        del F, P, sig
    f.close()
    eps = np.array(eps)
    np.savez_compressed(out_npz, sigma_n=sn, eps=eps, sig_nominal=np.array(sig_nom),
                        edge_n=edges[:, 2].astype(np.float64),
                        edge_wx=edges[:, 3].astype(int), edge_wy=edges[:, 4].astype(int),
                        edge_g1=edges[:, 0].astype(int), edge_g2=edges[:, 1].astype(int),
                        n_grains=d["grid"].max() + 1, sample=int(tag[:3]), metal=tag[4:])
    return sn, eps


def damage_labels(sn, eps, gam, g1, g2, edge_n, wx, wy, n_grains, k, m=M_EXP):
    """损伤演化 + y 向绕回检测 → (eps_fail, failed_frac, censored)。"""
    n_edges = sn.shape[0]
    sigma_c = np.maximum(k * gam, 1e-9)
    D = np.zeros(n_edges)
    failed = np.zeros(n_edges, dtype=bool)

    parent = list(range(n_grains))
    off = [(0, 0)] * n_grains

    def find(a):
        if parent[a] == a:
            return a, (0, 0)
        r, (dx, dy) = find(parent[a])
        parent[a] = r
        off[a] = (off[a][0] + dx, off[a][1] + dy)
        return r, off[a]

    def union(a, b, wxi, wyi):
        """返回 (wound_x, wound_y)。"""
        ra, (ax, ay) = find(a)
        rb, (bx, by) = find(b)
        if ra == rb:
            return (ax + wxi - bx) != 0, (ay + wyi - by) != 0
        parent[ra] = rb
        off[ra] = (bx - wxi - ax, by - wyi - ay)
        return False, False

    total_len = float(edge_n.sum())
    for t in range(1, len(eps)):
        dt = eps[t] - eps[t - 1]
        alive = ~failed
        sp = np.maximum(sn[alive, t - 1], 0.0) / sigma_c[alive]
        s = np.maximum(sn[alive, t], 0.0) / sigma_c[alive]
        D[alive] += 0.5 * (sp ** m + s ** m) * dt
        newly = np.where(alive & (D >= 1.0))[0]
        for i in newly:
            failed[i] = True
            _, wound_y = union(int(g1[i]), int(g2[i]), int(wx[i]), int(wy[i]))
            if wound_y:
                return eps[t], float(edge_n[failed].sum() / total_len), False, D
    return np.nan, float(edge_n[failed].sum() / total_len), True, D


def sanity_figure(grid2d, D_at_fail, edges, eps, sig_nom, eps_fail, pot, path, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    M = grid2d.shape[0]
    gkey = {(int(a), int(b)): dv for (a, b, *_), dv in zip(edges, D_at_fail)}
    r = grid2d
    d = np.roll(r, -1, axis=0)
    rr = np.roll(r, -1, axis=1)
    dmap = np.full((M, M), np.nan)
    for a, b, mp in ((r, d, dmap), (r, rr, dmap)):
        k1 = np.minimum(a, b).astype(int)
        k2 = np.maximum(a, b).astype(int)
        vec = np.vectorize(lambda u, v: gkey.get((int(u), int(v)), np.nan))
        mp[:, :] = np.where(a != b, vec(k1, k2), mp)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    im = ax[0].imshow(dmap, cmap="inferno", interpolation="nearest", vmin=0, vmax=1)
    ax[0].set_title(f"damage D at failure ({pot})")
    fig.colorbar(im, ax=ax[0], fraction=0.046)
    ax[1].plot(eps, sig_nom, lw=1.5)
    if not np.isnan(eps_fail):
        ax[1].axvline(eps_fail, color="crimson", ls="--", label=f"eps_fail={eps_fail:.4f}")
        ax[1].legend()
    ax[1].set_xlabel("ε_xx"); ax[1].set_ylabel("σ_nominal (Pa)")
    ax[1].set_title("macro response")
    fig.suptitle(title)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def stage_labels(runs_root):
    """k_metal 标定 → 损伤/渗流 → labels/e3_labels.csv + sanity 图。"""
    spec = load_e1_spectra()
    pots = sorted(spec.keys())
    manifest = []
    with open(os.path.join(runs_root, "manifest.csv")) as fh:
        manifest = list(csv.DictReader(fh))

    curves = {}
    for row in manifest:
        tag = f"{int(row['sample']):03d}_{row['metal']}"
        p = os.path.join(D6, "labels", "e3_curves", f"{tag}.npz")
        if os.path.exists(p):
            curves[tag] = dict(np.load(p, allow_pickle=True))

    # γ 采样：每样本独立 rng，逐势逐边；v2 变换 γ̃ = √(γ+γ₀)（损伤与标定同空间）
    gam = {}
    for row in manifest:
        s = int(row["sample"])
        d = np.load(os.path.join(D6, "networks", "library", f"sample_{s:03d}.npz"),
                    allow_pickle=True)
        et = d["etype"]
        g0 = GAMMA0[row["metal"]]
        for pot in pots:
            rng = np.random.default_rng(10_000 + s)
            tw = spec[pot][row["metal"]]["twin"] or spec[pot][row["metal"]]["general"]
            gn = spec[pot][row["metal"]]["general"]
            gam[(s, row["metal"], pot)] = np.maximum(
                np.array([rng.choice(tw) if e == "twin" else rng.choice(gn) for e in et])
                + g0, 1e-6) ** GAMMA_P

    # k_metal：median_runs( κ·σ̄_net(ε_ref) / mean_e[γ̃] )，σ̄_net 为边平均法向应力（代数值）。
    # κ=0.7 为转变区定标：损伤等效模型按"典型组织在 ε_max=0.06 内处于贯通概率
    # 转变带"标定（验收实测贯通率 43%、首步堆点 0%）；κ=1 时损伤积分在 ε 内仅
    # 累积 D≈0.16，几乎不贯通（8.6%）——κ 是等效模型的常规标定量，非独立物理参数。
    # 按 (metal, potential) 各自标定：损伤演化用该势自己的 γ̃，标定也须用同势 γ̃̄，
    # 否则某势能谱系统性偏移会跨势污染 σ_c 尺度。
    KAPPA = 0.7
    k_metal = {}
    for metal in sorted({r["metal"] for r in manifest}):
        for pot in pots:
            ratios = []
            for row in manifest:
                if row["metal"] != metal:
                    continue
                c = curves.get(f"{int(row['sample']):03d}_{metal}")
                if c is None:
                    continue
                eps, sn = c["eps"], c["sigma_n"].astype(np.float64)
                if eps[-1] < EPS_REF:
                    continue
                sig_net = sn.mean(axis=0)
                sig_ref = float(np.interp(EPS_REF, eps, sig_net))
                gbar = float(np.mean(gam[(int(row["sample"]), metal, pot)]))
                ratios.append(sig_ref / gbar)
            k_metal[(metal, pot)] = float(np.median(ratios)) * KAPPA if ratios else np.nan
            print(f"[k] {metal}/{pot}: k = {k_metal[(metal, pot)]:.3f} Pa  "
                  f"(n={len(ratios)})")

    # 损伤 + 渗流 → 标签
    rows, D_store = [], {}
    for row in manifest:
        s, metal = int(row["sample"]), row["metal"]
        c = curves.get(f"{s:03d}_{metal}")
        if c is None:
            print(f"[labels] 缺 {s:03d}_{metal} 曲线（run 失败？），跳过该 run 的标签")
            continue
        for pot in pots:
            g = gam[(s, metal, pot)]
            eps_fail, frac, cens, D = damage_labels(
                c["sigma_n"].astype(np.float64), c["eps"], g,
                c["edge_g1"], c["edge_g2"], c["edge_n"],
                c["edge_wx"], c["edge_wy"], int(c["n_grains"]),
                k_metal[(metal, pot)])
            rows.append({"sample": s, "metal": metal, "potential": pot,
                         "n_grains": int(c["n_grains"]),
                         "eps_fail": round(float(eps_fail), 5) if not np.isnan(eps_fail) else "",
                         "failed_GB_len_frac": round(float(frac), 5),
                         "censored": int(cens), "k_metal": round(k_metal[(metal, pot)], 3)})
            D_store[(s, metal, pot)] = D

    out = os.path.join(D6, "labels", "e3_labels.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"→ {out}（{len(rows)} 行 = {len(manifest)} runs × {len(pots)} 势）")

    # 统计摘要
    eps_all = [float(r["eps_fail"]) for r in rows if r["eps_fail"] != ""]
    if eps_all:
        print(f"贯通率 = {len(eps_all)}/{len(rows)}；eps_fail 范围 "
              f"{min(eps_all):.4f}–{max(eps_all):.4f}")
    else:
        print("无贯通事件！k 标定或加载幅度需复核")

    # sanity 图：每金属取 twin_prob 最低/最高样本各 1（首个势）
    figdir = os.path.join(D6, "figures")
    os.makedirs(figdir, exist_ok=True)
    tw = {}
    with open(os.path.join(D6, "results", "network_features.csv")) as fh:
        for r in csv.DictReader(fh):
            if r["metal"] == "Cu":
                tw[int(r["sample"])] = float(r["twin_prob"])
    pots0 = pots[0]
    for metal in sorted({r["metal"] for r in manifest}):
        samples = sorted({int(r["sample"]) for r in manifest if r["metal"] == metal})
        for tag_word, s in (("lowtw", min(samples, key=lambda x: tw[x])),
                            ("hitw", max(samples, key=lambda x: tw[x]))):
            d = np.load(os.path.join(D6, "networks", "library", f"sample_{s:03d}.npz"),
                        allow_pickle=True)
            c = curves[f"{s:03d}_{metal}"]
            eps_fail = next((float(r["eps_fail"]) for r in rows
                             if r["sample"] == s and r["metal"] == metal
                             and r["potential"] == pots0 and r["eps_fail"] != ""), np.nan)
            sanity_figure(d["grid"], D_store[(s, metal, pots0)], d["edges"],
                          c["eps"], c["sig_nominal"], eps_fail, pots0,
                          os.path.join(figdir, f"e3_sanity_{metal}_{tag_word}_{s:03d}.png"),
                          title=f"sample {s:03d} {metal}/{pots0} twin_prob={tw[s]:.2f}")
    n_new = sum(1 for _ in sorted({r["metal"] for r in manifest})
                for _ in ("lowtw", "hitw"))
    print(f"→ sanity 图 ×{n_new} 于 {figdir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["curves", "labels"], required=True)
    ap.add_argument("--run-dir", default="", help="curves 阶段：单个 run 目录")
    ap.add_argument("--runs-root", default=os.path.join(HERE, "e3_runs"))
    ap.add_argument("--z", type=lambda s: int(s) if str(s).strip() else 0,
                    default=4, help="z 层数（可为空；实际以 result.hdf5 cells 为准）")
    args = ap.parse_args()

    outdir = os.path.join(D6, "labels")
    os.makedirs(os.path.join(outdir, "e3_curves"), exist_ok=True)

    if args.stage == "curves":
        tag = os.path.basename(args.run_dir.rstrip("/"))
        out_npz = os.path.join(outdir, "e3_curves", f"{tag}.npz")
        if os.path.exists(out_npz):
            print(f"[skip] {tag} 已有曲线")
            return
        sn, eps = extract_curves(args.run_dir, out_npz, nz=args.z)
        print(f"[curves] {tag}: {sn.shape[0]} 边 × {sn.shape[1]} 输出点, ε_max={eps[-1]:.4f}")
    else:
        stage_labels(args.runs_root)


if __name__ == "__main__":
    main()
