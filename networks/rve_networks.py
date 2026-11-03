#!/usr/bin/env python3
"""E2 微观组织库：周期 Voronoi 多晶生成 + 晶界网络渗流特征（v0）

管线：
  1) 生成 2D 周期多晶网格（seeds + cKDTree 周期指派，M×M）
  2) 晶粒取向：parent 随机 + 受控 Σ3 孪晶子晶粒（twin prevalence 可调）
  3) 晶界网络：4-邻域周期邻接 → 边表（界面长度 px、绕向标志、边类型）
  4) 边能量：从 E1 库引导采样（twin→Σ3 值；general→高角 Σ5/Σ9/Σ11 经验分布），每势一套实现
  5) 渗流特征：γ_th 扫描（高能边界先失效）→ union-find 带 mod 偏移检测 winding
     → γ_perc（起渗阈值）、失效簇分数曲线 AUC 等特征

输出：
  networks/library/sample_XXX.npz   （seeds/取向/parent/twin/边表——供 E3 CPFEM 复用同一拓扑）
  results/network_features.csv      （样本 × 金属 × 势 的特征表）
"""
import argparse, csv, os
import networkx as nx
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

# ---------------- E1 能谱引导（来自 gb_energy_library_merged.csv） ----------------
E1_CSV = os.path.join(os.path.dirname(__file__), "..", "results", "gb_energy_library_merged.csv")

def load_e1_spectra():
    """返回 {potential: {metal: {"twin": [γ...], "general": [γ...]}}}
    twin: |hkl|=111 的 Σ3；general: 其余全部有效行。"""
    spec = {}
    with open(E1_CSV) as fh:
        for r in csv.DictReader(fh):
            if r["flag"] != "ok":
                continue
            pot, metal, hkl = r["potential"], r["metal"], r["hkl"].strip()
            g = float(r["gamma_mJ_per_m2"])
            d = spec.setdefault(pot, {}).setdefault(metal, {"twin": [], "general": []})
            (d["twin"] if hkl == "111" and r["sigma"].strip() == "3" else d["general"]).append(g)
    return spec

# ---------------- 组织生成 ----------------
def gen_microstructure(rng, M=256, n_grains=80, twin_prob=0.4):
    """返回 dict：grain 网格、seeds、每晶粒 (parent, twin_variant)、旋转矩阵、边表"""
    seeds = rng.random((n_grains, 2))
    yy, xx = np.mgrid[0:M, 0:M]
    pts = np.stack([xx.ravel() + 0.5, yy.ravel() + 0.5], axis=1) / M
    _, gid = cKDTree(seeds, boxsize=1.0).query(pts)
    grid = gid.reshape(M, M).astype(np.int32)

    # 取向簇：层级 Voronoi——粗种子把细晶粒划成空间紧致的簇，簇内才可能共邻出孪晶界
    n_clusters = max(1, n_grains // 8)
    coarse = rng.random((n_clusters, 2))
    _, parent = cKDTree(coarse, boxsize=1.0).query(seeds)
    parent = parent.astype(int)
    is_twin = rng.random(n_grains) < twin_prob
    # 孪晶错向：60° 绕 <111>（fcc Σ3）；不同子晶粒用 4 个等价轴之一
    AX111 = np.array([[1, 1, 1], [1, 1, -1], [1, -1, 1], [-1, 1, 1]]) / np.sqrt(3.0)
    R_parent = Rotation.random(n_grains, random_state=rng)
    Rm = R_parent.as_matrix()
    tv = rng.integers(0, 4, size=n_grains)
    for i in np.where(is_twin)[0]:
        D = Rotation.from_rotvec(np.pi / 3 * AX111[tv[i]]).as_matrix()
        Rm[i] = D @ Rm[i]

    # 周期 4-邻域边表（右、下两个方向，含绕回；方向约定 b = a + w）
    r = grid; d = np.roll(grid, -1, axis=0); rr = np.roll(grid, -1, axis=1)
    edges = {}
    for (a, b, wx, wy) in ((r, d, 0, 1), (r, rr, 1, 0)):
        m = a != b
        pairs = np.stack([a[m].ravel(), b[m].ravel(), np.full(m.sum(), wx), np.full(m.sum(), wy)], 1)
        for g1, g2, x, y in pairs:
            k = (min(g1, g2), max(g1, g2))
            e = edges.setdefault(k, {"n": 0, "wx": 0, "wy": 0})
            e["n"] += 1; e["wx"] |= int(x); e["wy"] |= int(y)
    E = [(g1, g2, e["n"], e["wx"], e["wy"]) for (g1, g2), e in edges.items()]
    return {"grid": grid, "seeds": seeds, "R": Rm, "parent": parent, "is_twin": is_twin,
            "edges": np.array(E, dtype=np.float64) if E else np.zeros((0, 5))}

def edge_types(ms):
    """(g1,g2)→'twin' 同簇且恰一方为孪晶子晶粒；'twin_related' 同簇双子；'general' 异簇"""
    p, tw = ms["parent"], ms["is_twin"]
    out = []
    for g1, g2, _n, _wx, _wy in ms["edges"]:
        g1, g2 = int(g1), int(g2)
        if p[g1] == p[g2] and (tw[g1] != tw[g2]):
            out.append("twin")
        elif p[g1] == p[g2]:
            out.append("twin_related")
        else:
            out.append("general")
    return np.array(out)

# ---------------- 渗流（union-find 带 mod 偏移 → winding 检测） ----------------
class DSU:
    """union-find 带整数偏移势：边 (a,b,w) 约束 off[b]=off[a]+w（不取模）。
    同根但偏移约束冲突 ⇒ 存在非可缩回路 ⇒ 绕回（渗流）。"""
    def __init__(self, n):
        self.f = list(range(n)); self.off = [(0, 0)] * n
    def find(self, a):
        if self.f[a] == a: return a, (0, 0)
        r, (dx, dy) = self.find(self.f[a])
        self.f[a] = r; self.off[a] = (self.off[a][0] + dx, self.off[a][1] + dy)
        return r, self.off[a]
    def union(self, a, b, wx, wy):
        """返回 False = 合并产生绕回（winding）"""
        ra, (ax, ay) = self.find(a); rb, (bx, by) = self.find(b)
        if ra == rb:
            return (ax + wx - bx) != 0 or (ay + wy - by) != 0
        self.f[ra] = rb
        self.off[ra] = (bx - wx - ax, by - wy - ay)
        return True

def percolation_features(n_grains, edges, etypes, gammas, nthresh=40):
    """gammas: 每条边能量（与 edges 同序）。
    γ_perc：按能量降序加边（高能边界先失效），首次形成绕回回路时的边能量——
    "γ ≥ γ_perc 的子网即可贯穿"，值越高越耐沿晶开裂。"""
    gvals = np.unique(gammas)
    if gvals.size < 2 or len(edges) == 0:
        return {"gamma_perc_mJm2": np.nan, "gamma_perc_rel": np.nan,
                "failed_cluster_auc": np.nan, "failed_frac_at_mean": np.nan}
    gx = edges[:, 0].astype(int); gy = edges[:, 1].astype(int)
    # 精确临界值：降序加边
    dsu = DSU(n_grains); gamma_perc = np.nan
    for i in np.argsort(-gammas):
        if not dsu.union(gx[i], gy[i], int(edges[i, 3]), int(edges[i, 4])):
            gamma_perc = float(gammas[i]); break
    # 失效簇分数曲线（γ>th 扫描，供 AUC）
    ths = np.linspace(gvals.min(), gvals.max() + 1e-9, nthresh)
    frac_curve = []
    for th in ths:
        failed = gammas > th
        d2 = DSU(n_grains)
        for i in np.where(failed)[0]:
            d2.union(gx[i], gy[i], int(edges[i, 3]), int(edges[i, 4]))
        cnt = {}
        for g in range(n_grains):
            r, _ = d2.find(g); cnt[r] = cnt.get(r, 0) + 1
        frac_curve.append((max(cnt.values()) / n_grains) if cnt else 0.0)
    frac_curve = np.array(frac_curve)
    mean_g = float(np.mean(gammas))
    return {"gamma_perc_mJm2": gamma_perc,
            "gamma_perc_rel": (gamma_perc / mean_g) if not np.isnan(gamma_perc) else np.nan,
            "failed_cluster_auc": float(np.trapezoid(frac_curve, ths) / (ths[-1] - ths[0])),
            "failed_frac_at_mean": float(frac_curve[np.argmin(np.abs(ths - mean_g))])}

# ---------------- 主流程 ----------------
def sanity_figure(ms, et, gam, path, title=""):
    """两联 sanity 图：左=晶粒簇归属，右=晶界按 γ 着色（孪晶应显低能亮线）"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    M = ms["grid"].shape[0]
    gkey = {(min(int(a), int(b)), max(int(a), int(b))): g
            for (a, b, _n, _x, _y), g in zip(ms["edges"], gam)}
    r = ms["grid"]; d = np.roll(r, -1, axis=0); rr = np.roll(r, -1, axis=1)
    gam_map = np.full((M, M), np.nan)
    for (a, b, mp) in ((r, d, gam_map), (r, rr, gam_map)):
        k1 = np.minimum(a, b); k2 = np.maximum(a, b)
        vec = np.vectorize(lambda u, v: gkey.get((int(u), int(v)), np.nan))
        mp[:, :] = np.where(a != b, vec(k1, k2), mp)
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    ax[0].imshow(ms["parent"][ms["grid"]], cmap="tab20", interpolation="nearest")
    ax[0].set_title("parent clusters")
    im = ax[1].imshow(gam_map, cmap="inferno", interpolation="nearest")
    ax[1].set_title("GB energy gamma (mJ/m2)")
    fig.colorbar(im, ax=ax[1], fraction=0.046)
    fig.suptitle(title)
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--grid", type=int, default=256)
    ap.add_argument("--grains", type=int, default=0, help="0=每样本随机 40..200")
    ap.add_argument("--twin-prob", type=float, default=-1.0, help="<0=每样本随机 0..0.65")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fig-samples", default="", help="逗号分隔样本号，出 sanity 图")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "results", "network_features.csv"))
    args = ap.parse_args()

    spec = load_e1_spectra()
    lib = os.path.join(os.path.dirname(__file__), "library"); os.makedirs(lib, exist_ok=True)
    figdir = os.path.join(os.path.dirname(__file__), "..", "figures"); os.makedirs(figdir, exist_ok=True)
    metals = sorted(next(iter(spec.values())).keys())
    pots = sorted(spec.keys())
    rows = []
    for s in range(args.n_samples):
        rng = np.random.default_rng(args.seed + s)
        twin_prob = args.twin_prob if args.twin_prob >= 0 else float(rng.uniform(0, 0.65))
        n_g = args.grains if args.grains > 0 else int(rng.integers(40, 201))
        ms = gen_microstructure(rng, M=args.grid, n_grains=n_g, twin_prob=twin_prob)
        et = edge_types(ms)
        tw_frac = float(et[et == "twin"].size) / max(1, et.size)
        np.savez_compressed(os.path.join(lib, f"sample_{s:03d}.npz"),
                            grid=ms["grid"], seeds=ms["seeds"], R=ms["R"],
                            parent=ms["parent"], is_twin=ms["is_twin"],
                            edges=ms["edges"], etype=et, twin_prob=twin_prob)
        # 拓扑特征（networkx，无权图）
        Gnx = nx.Graph()
        Gnx.add_nodes_from(range(n_g))
        Gnx.add_edges_from([(int(a), int(b)) for a, b, *_ in ms["edges"]])
        deg = dict(Gnx.degree())
        mean_deg = float(np.mean([deg[g] for g in Gnx.nodes])) if Gnx.number_of_nodes() else np.nan
        avg_clu = nx.average_clustering(Gnx) if Gnx.number_of_nodes() > 2 else np.nan
        for metal in metals:
            for pot in pots:
                tw_vals = spec[pot][metal]["twin"]; gn_vals = spec[pot][metal]["general"]
                if not tw_vals: tw_vals = gn_vals
                gam = np.array([rng.choice(tw_vals) if t == "twin" else rng.choice(gn_vals)
                                for t in et]) if len(et) else np.zeros(0)
                f = percolation_features(n_g, ms["edges"], et, gam)
                rows.append({"sample": s, "metal": metal, "potential": pot,
                             "n_grains": n_g, "twin_prob": round(twin_prob, 4),
                             "n_edges": len(et), "mean_degree": round(mean_deg, 3),
                             "avg_clustering": round(avg_clu, 4),
                             "twin_length_frac": round(tw_frac, 4),
                             "mean_gamma": round(float(np.mean(gam)), 1) if len(gam) else np.nan,
                             "gamma_std": round(float(np.std(gam)), 1) if len(gam) else np.nan,
                             **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in f.items()}})
        if str(s) in args.fig_samples.split(","):
            gam0 = np.array([rng.choice(spec["chgnet"]["Cu"]["twin"]) if t == "twin"
                             else rng.choice(spec["chgnet"]["Cu"]["general"]) for t in et])
            sanity_figure(ms, et, gam0, os.path.join(figdir, f"e2_sanity_sample_{s:03d}.png"),
                          title=f"sample {s:03d}  n_grains={n_g}  twin_prob={twin_prob:.2f}  twin_frac={tw_frac:.3f}")
        print(f"sample {s:03d} n_grains={n_g} twin_prob={twin_prob:.3f} 边数={len(et)} 孪晶边占比={tw_frac:.3f}", flush=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"→ {args.out}（{len(rows)} 行 = {args.n_samples} 样本 × {len(metals)} 金属 × {len(pots)} 势）")

if __name__ == "__main__":
    main()
