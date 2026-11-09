#!/usr/bin/env python3
"""E7 相场断裂排序交叉验证：独立断裂模型的失效排序 vs 等效损伤规则标签

动机：标签 = 等效损伤规则（σ_c ∝ (γ+γ₀)^0.65 + CPFEM 应力史 + DSU 渗流），
属合成标签。本实验用一个独立的力学引擎——2D 平面应力准静态相场断裂（全弹性、
无塑性、无 CPFEM、无逐边损伤积分）——对同一组微结构重新生成失效排序，
与 labels/e3_labels.csv 的 eps_fail 做 Spearman ρ 对照。

模型（显式写全，不依赖命名）：
  能量  E[u,d] = ∫ (1−d)²·H(x)  +  Gc(x)/(2·l0)·d² + Gc(x)·l0/2·|∇d|² dx
  其中 H = max_ε ψ⁺(ε(u)) 历史场（不可逆损伤），二次退化 + 二次 crack-density
  正则（AT2 型泛函）；交错求解：u-PCG（能量一致前向差分算子、平均刚度 FFT
  预条件），d-PCG（对称边系数 Laplace、FFT-Helmholtz 预条件），
  每步后投影 d ≥ d_prev 保证不可逆。
  边界：周期；宏观 ε_xx 逐步加载 0→0.06（Δε=0.001），侧向 ε_yy 迭代至
  ⟨σ_yy⟩=0（对 DAMASK 单轴拉伸、横向自由的等效；2D 平面应力 ↔ 柱状 RVE
  z 向自由）。
  Gc 场：GB 像素 Gc_e = G_REF · (γ_e+γ₀)^0.65 / median_pool((γ+γ₀)^0.65)，
  与标签规则同向同形的 γ→强度耦合（比较对象是力学引擎与失效判据，不是
  耦合符号）；γ_e 按标签管线同源复采（E1 池、default_rng(10_000+sample)、
  twin→Σ3、其余→general、同序）；G_REF=1 J/m²（池中位数处 GB 断裂能取
  物理量级 1 J/m²），晶内 G_BULK=100 J/m² 压住跨晶断裂（论文范围=沿晶）。
  弹性常数：与 e3_prepare 相同的立方 Cij → Voigt-Reuss-Hill 各向同性 E,ν。
  l0 = 2 格（128² 解、h=0.1 μm、l0=0.2 μm；Gc 场 256²→128² 取块 min，
  保住 1px GB 线连通）。
  失效判据：d≥0.5 像素带的周期连通分量在 y 向绕回（↔ 裂纹面 ⊥ x 贯通，
  与标签 DSU wound_y 等价）；ε≤0.06 未绕回 → censored。

子集（确定性）：labels 中三势均非删失的全部样本，按 (n_grains, twin_prob,
twin_length_frac) 贪心 max-min（twin_length_frac 最高起步）排序：
Cu 20 / Ni 18 / Fe 8（Fe 候选恰 8）→ 46 组织 × 3 金属 × 3 势 = 138 run。
（贪心为前缀稳定：先前 Cu/Ni 各取前 10 的 84-run 结果可直接复用。）

输出：
  results/e7_phasefield_runs.csv        逐 run 明细（相场 + 标签并排）
  results/e7_phasefield_summary.csv     每 (metal, potential) ρ 汇总
  results/e7_phasefield_summary_micro.csv  微结构级 ρ（势平均，每金属一个）

用法：
  python e7_phasefield.py --smoke        # 单 run 冒烟
  python e7_phasefield.py --jobs 10      # 全量
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse, csv, math, sys, time
from multiprocessing import Pool

import numpy as np
from scipy.ndimage import label as cc_label
from scipy.sparse.linalg import LinearOperator, cg
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
D6 = os.path.abspath(os.path.join(HERE, ".."))

# ---- 与 e3_postprocess 相同的 γ→强度耦合约定（v2 定稿）----
GAMMA0 = {"Cu": 22.0, "Ni": 70.0, "Fe": 0.0}
GAMMA_P = 0.65

# ---- 与 e3_prepare 相同的立方弹性常数（DAMASK run 即用此组）----
CUBIC = {"Cu": (168.9e9, 121.8e9, 75.8e9),
         "Ni": (251.0e9, 150.0e9, 123.7e9),
         "Fe": (232.2e9, 136.4e9, 117.0e9)}

DELTA_UM = 0.05          # 256² 网格步长 μm（RVE 12.8 μm）
DOWN = 2                 # 求解分辨率 128²，h = 0.1 μm
L0_CELLS = 2.0           # 相场长度尺度（格）
G_REF = 1.0              # J/m²：池中位数 GB 断裂能（物理量级，非对标签标定）
G_BULK = 100.0           # J/m²：晶内（压住跨晶，聚焦沿晶）
G_FLOOR = 0.02           # J/m²：数值下限（负 γ 病态值兜底）
G_RESID = 0.01           # 退化下限：g ≥ (1−d)² 截断于 1e-4，避免裂纹带零刚度奇异性
EPS_MAX, DEPS = 0.06, 0.001
D_CRACK = 0.5            # 裂纹判据阈值
N_SUB = {"Cu": 20, "Ni": 18, "Fe": 8}   # 取全部候选（三势标签均非删失）
CG_RTOL = 1e-7
RUN_FIELDS = ["sample", "metal", "potential", "n_grains", "twin_length_frac",
              "eps_label", "censored_label", "eps_pf", "censored_pf", "runtime_s"]


def isotropic_elastic(metal):
    """立方 Cij → Voigt-Reuss-Hill → 平面应力 D 系数与剪切模量。"""
    C11, C12, C44 = CUBIC[metal]
    Kv = (C11 + 2.0 * C12) / 3.0
    Gv = (C11 - C12 + 3.0 * C44) / 5.0
    den = (C11 - C12) * (C11 + 2.0 * C12)
    s11, s12, s44 = (C11 + C12) / den, -C12 / den, 1.0 / C44
    Gr = 5.0 / (4.0 * (s11 - s12) + 3.0 * s44)
    K, G = Kv, 0.5 * (Gv + Gr)                      # KV = KR（立方）
    E = 9.0 * K * G / (3.0 * K + G)
    nu = (3.0 * K - 2.0 * G) / (2.0 * (3.0 * K + G))
    return E, nu


def load_e1_spectra():
    """{potential: {metal: {"twin":[γ...], "general":[γ...]}}}——与 E2/E3 同规则。"""
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


def draw_edge_gamma(sample, metal, pot, spec, etype):
    """与 e3_postprocess 逐字同口径：rng(10_000+s) 逐边选择（原始 γ，不做变换）。"""
    s = sample
    tw = spec[pot][metal]["twin"] or spec[pot][metal]["general"]
    gn = spec[pot][metal]["general"]
    rng = np.random.default_rng(10_000 + s)
    return np.array([rng.choice(tw) if e == "twin" else rng.choice(gn)
                     for e in etype], dtype=np.float64)


def select_subsets():
    """三势均非删失样本 → 每金属贪心 max-min（tlf 起步）→ 确定性子集。"""
    labels = {}
    for r in csv.DictReader(open(os.path.join(D6, "labels", "e3_labels.csv"))):
        labels.setdefault((int(r["sample"]), r["metal"]), {})[r["potential"]] = (
            r["eps_fail"], int(r["censored"]))
    feats = {int(r["sample"]): r for r in
             csv.DictReader(open(os.path.join(D6, "results", "network_features.csv")))
             if r["potential"] == sorted(
                 {x["potential"] for x in csv.DictReader(
                     open(os.path.join(D6, "results", "network_features.csv")))})[0]}
    out = {}
    for metal, n_want in N_SUB.items():
        cand = sorted(s for (s, m), d in labels.items()
                      if m == metal and len(d) == 3 and all(v[1] == 0 for v in d.values()))
        X = np.array([[float(feats[s]["n_grains"]) / 200.0,
                       float(feats[s]["twin_prob"]) / 0.65,
                       float(feats[s]["twin_length_frac"])] for s in cand])
        sel = [int(np.argmax(X[:, 2]))]             # 孪晶分数最高起步
        while len(sel) < min(n_want, len(cand)):
            dmin = np.min([np.linalg.norm(X - X[i], axis=1) for i in sel], axis=0)
            dmin[sel] = -1
            sel.append(int(np.argmax(dmin)))
        out[metal] = sorted(cand[i] for i in sel)
        print(f"[subset] {metal}: 候选 {len(cand)} → 选 {out[metal]}")
    return out


# ------------------------------------------------------------------ 网格算子
def build_gc_field(sample, metal, pot, spec):
    """256² GB 断裂能场（晶内 G_BULK），块 min 降到 128²。"""
    d = np.load(os.path.join(D6, "networks", "library", f"sample_{sample:03d}.npz"),
                allow_pickle=True)
    grid, edges, etype = d["grid"], d["edges"], d["etype"]
    gam = draw_edge_gamma(sample, metal, pot, spec, etype)

    # 与标签同形的 γ→Gc：池中位数归一到 G_REF（池 = twin+general 全池）
    pool = np.maximum(np.array(spec[pot][metal]["twin"] + spec[pot][metal]["general"])
                      + GAMMA0[metal], 1e-6)
    med = float(np.median(pool ** GAMMA_P))
    gce = G_REF * np.maximum((gam + GAMMA0[metal]) ** GAMMA_P / med, G_FLOOR / G_REF)

    key = {(int(min(a, b)), int(max(a, b))): g
           for (a, b), g in zip(edges[:, :2], gce)}
    # 同对边取 min（平行段取更弱者，保守）
    for (a, b), g in zip(edges[:, :2], gce):
        k = (int(min(a, b)), int(max(a, b)))
        key[k] = min(key[k], g)

    gc = np.full(grid.shape, G_BULK, dtype=np.float64)
    rb, cb = np.roll(grid, -1, axis=0), np.roll(grid, -1, axis=1)
    for nb, mask in ((rb, grid != rb), (cb, grid != cb)):
        pairs_a, pairs_b = grid[mask], nb[mask]
        ka = np.minimum(pairs_a, pairs_b)
        kb = np.maximum(pairs_a, pairs_b)
        miss = 0
        for a, b, ia, ib in zip(ka, kb, *np.nonzero(mask)):
            g = key.get((int(a), int(b)))
            if g is None:
                miss += 1
                continue
            gc[ia, ib] = min(gc[ia, ib], g)
        if miss:
            print(f"  [warn] sample {sample}: {miss} 个相邻对不在边表（按晶内处理）")
    # 边界像素双侧取 min 已由逐对 min 覆盖
    small = gc.reshape(gc.shape[0] // DOWN, DOWN, gc.shape[1] // DOWN, DOWN
                       ).min(axis=(1, 3))
    return small, grid, edges, etype, gam


def winds_y(mask):
    """d≥0.5 带在 y 向周期绕回（pad-wrap 连通分量法）。"""
    if not mask.any():
        return False
    p = np.pad(mask, 1, mode="wrap")
    lab, _ = cc_label(p)
    top = set(np.unique(lab[0, :])) - {0}
    bot = set(np.unique(lab[-1, :])) - {0}
    return bool(top & bot)


# ------------------------------------------------------------------ 相场求解
class PhaseField:
    def __init__(self, gc, metal, eps_max=EPS_MAX, deps=DEPS):
        self.gc = gc                      # 128² Gc 场 J/m²
        self.N = gc.shape[0]
        self.h = DELTA_UM * DOWN * 1e-6   # m
        self.l0 = L0_CELLS * self.h
        E, nu = isotropic_elastic(metal)
        self.E, self.nu = E, nu
        self.cp = E / (1.0 - nu * nu)     # 平面应力：σxx = cp(εxx+νεyy)
        self.mu = E / (2.0 * (1.0 + nu))
        self.eps_max, self.deps = eps_max, deps
        N = self.N
        self.d = np.zeros((N, N))
        self.H = np.zeros((N, N))
        self.eps_yy = 0.0
        self.gbar = 1.0
        # FFT 预条件符号：轴0=row=y、轴1=col=x（与 grid2d 约定一致）
        kx = 2.0 * math.pi * np.fft.fftfreq(N, d=self.h)   # x 频率（轴1）
        ky = 2.0 * math.pi * np.fft.fftfreq(N, d=self.h)   # y 频率（轴0）
        KX, KY = np.meshgrid(kx, ky, indexing="xy")        # shape (N,N)：KX 随轴1、KY 随轴0
        self.KX, self.KY = KX, KY
        self._build_u_symbol()

    # --- 应变/应力（能量一致前向差分，周期）---
    def _grad(self, ux, uy):
        """轴0=y、轴1=x：exx=∂/∂x(轴1)、eyy=∂/∂y(轴0)。前向差分（周期）。"""
        hx = self.h
        exx = (np.roll(ux, -1, axis=1) - ux) / hx
        eyy = (np.roll(uy, -1, axis=0) - uy) / hx
        exy = 0.5 * ((np.roll(ux, -1, axis=0) - ux) / hx +
                     (np.roll(uy, -1, axis=1) - uy) / hx)
        return exx, eyy, exy

    def _stress(self, exx, eyy, exy, g):
        sxx = g * (self.cp * (exx + self.nu * eyy))
        syy = g * (self.cp * (self.nu * exx + eyy))
        sxy = g * (2.0 * self.mu * exy)
        return sxx, syy, sxy

    def _div_adj(self, sxx, syy, sxy):
        """与 _grad 伴随（−div，反向差分）：K = ∇_bᵀ·g·C·∇_f 为 SPD。
        x=轴1：∂sxx/∂x + ∂τ/∂y(轴0) → fx；y=轴0 对称。"""
        hx = self.h
        fx = -(sxx - np.roll(sxx, 1, axis=1)) / hx - (sxy - np.roll(sxy, 1, axis=0)) / hx
        fy = -(syy - np.roll(syy, 1, axis=0)) / hx - (sxy - np.roll(sxy, 1, axis=1)) / hx
        return fx, fy

    def _build_u_symbol(self):
        """常系数算子的 FFT 符号（2×2，逐波数求逆）→ 平均刚度预条件。"""
        kx, ky = self.KX, self.KY
        cp, mu, nu = self.cp, self.mu, self.nu
        # ε = B·u，B: (ux,uy) → [exx,eyy,exy]（去掉 i）
        B = np.zeros((3, 2) + kx.shape)
        B[0, 0], B[1, 1] = kx, ky
        B[2, 0], B[2, 1] = 0.5 * ky, 0.5 * kx
        C = np.array([[cp, cp * nu, 0.0],
                      [cp * nu, cp, 0.0],
                      [0.0, 0.0, 2.0 * mu]])
        A = np.einsum('auxy,ab,bvxy->uvxy', B, C, B)
        det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
        det[np.abs(det) < 1e-30] = np.inf
        self._sym_inv = (A[1, 1] / det, -A[0, 1] / det, -A[1, 0] / det, A[0, 0] / det)

    def _precond_u(self, rx, ry):
        FX, FY = np.fft.fft2(rx), np.fft.fft2(ry)
        a, b, c, dd = self._sym_inv
        wx = a * FX + b * FY
        wy = c * FX + dd * FY
        wx[0, 0] = wy[0, 0] = 0.0
        return np.real(np.fft.ifft2(wx)), np.real(np.fft.ifft2(wy))

    def solve_u(self, exx_bar, eyy_bar):
        """给定宏观应变，解周期 u（g 退化）；返回 (⟨σyy⟩, ψ场, ⟨g⟩)。"""
        gc = None
        g = np.maximum(1.0 - self.d, G_RESID) ** 2
        N2 = self.N * self.N

        def op(v):
            ux, uy = v[:N2].reshape(g.shape), v[N2:].reshape(g.shape)
            exx, eyy, exy = self._grad(ux, uy)
            fx, fy = self._div_adj(*self._stress(exx, eyy, exy, g))
            return np.concatenate([fx.ravel(), fy.ravel()])

        def rhs():
            # −K(affine)：affine 应变恒为 (exx̄, eyȳ, 0)
            sxx = g * (self.cp * (exx_bar + self.nu * eyy_bar))
            syy = g * (self.cp * (self.nu * exx_bar + eyy_bar))
            z = np.zeros_like(sxx)
            fx, fy = self._div_adj(sxx, syy, z)
            return -np.concatenate([fx.ravel(), fy.ravel()])

        b = rhs()
        b[:N2] -= b[:N2].mean()
        b[N2:] -= b[N2:].mean()

        def precond(r):
            rx, ry = self._precond_u(r[:N2].reshape(g.shape) - r[:N2].reshape(g.shape).mean(),
                                     r[N2:].reshape(g.shape) - r[N2:].reshape(g.shape).mean())
            return np.concatenate([rx.ravel(), ry.ravel()])

        x, info = cg(LinearOperator((2 * N2, 2 * N2), matvec=op), b,
                     rtol=1e-6, maxiter=6000,
                     M=LinearOperator((2 * N2, 2 * N2), matvec=precond))
        if info != 0:
            raise RuntimeError(
                f"u-CG 未收敛 info={info} gmean={g.mean():.4f} dmax={self.d.max():.3f} "
                f"eps_xx={exx_bar:.4f} eps_yy={eyy_bar:.4f} |b|={np.linalg.norm(b):.3e}")
        ux, uy = x[:N2].reshape(g.shape), x[N2:].reshape(g.shape)
        exx, eyy, exy = self._grad(ux, uy)
        exx = exx + exx_bar
        eyy = eyy + eyy_bar
        sxx, syy, sxy = self._stress(exx, eyy, exy, g)
        psi = (0.5 * (self.cp * (exx + self.nu * eyy) * exx
                      + self.cp * (self.nu * exx + eyy) * eyy)
               + 2.0 * self.mu * exy ** 2)
        return float(syy.mean()), psi, float(g.mean())

    def solve_d(self):
        """[−∇·(Gc l0 ∇) + Gc/l0 + 2H] d = 2H，边系数对称 → SPD。"""
        N, h, l0 = self.N, self.h, self.l0
        gce, H = self.gc, self.H
        gcx = 0.5 * (gce + np.roll(gce, -1, axis=0))   # +y 通量系数
        gcy = 0.5 * (gce + np.roll(gce, -1, axis=1))   # +x 通量系数
        a_y = l0 * gcx / h ** 2                          # 面向 (i,i+1)
        a_x = l0 * gcy / h ** 2

        def op(v):
            v = v.reshape(N, N)
            lap = (a_y * (v - np.roll(v, -1, axis=0)) +
                   np.roll(a_y, 1, axis=0) * (v - np.roll(v, 1, axis=0)) +
                   a_x * (v - np.roll(v, -1, axis=1)) +
                   np.roll(a_x, 1, axis=1) * (v - np.roll(v, 1, axis=1)))
            return (lap + (gce / l0 + 2.0 * H) * v).ravel()

        rhs = (2.0 * H).ravel()
        # FFT-Helmholtz 预条件（平均系数；DC 用局域项均值）
        ay_m, ax_m = float(np.mean(a_y)), float(np.mean(a_x))
        loc = float(np.mean(gce) / l0 + 2.0 * np.mean(H))
        sym = (ax_m * (2 - 2 * np.cos(self.KX * h)) +
               ay_m * (2 - 2 * np.cos(self.KY * h))) + loc
        sym[0, 0] = loc

        def precond(r):
            rr = np.real(np.fft.ifft2(np.fft.fft2(r.reshape(N, N)) / sym))
            return rr.ravel()

        dd, info = cg(LinearOperator((N * N, N * N), matvec=op), rhs,
                      rtol=CG_RTOL, maxiter=2000,
                      M=LinearOperator((N * N, N * N), matvec=precond))
        if info != 0:
            raise RuntimeError(f"d-CG 未收敛 info={info}")
        # 不可逆 + 有界
        self.d = np.clip(np.maximum(dd.reshape(N, N), self.d), 0.0, 1.0)

    def step(self, eps_xx):
        # 横向自由：割线法至 ⟨σ_yy⟩≈0（固定 d 下 εyy 线性 → 两点即收敛）
        tol = 1e-5 * self.cp
        prev_s, prev_e, syy = None, None, None
        for _ in range(5):
            e_now = self.eps_yy
            syy, psi, gbar = self.solve_u(eps_xx, e_now)
            if abs(syy) < tol:
                break
            if prev_s is None:
                self.eps_yy -= syy / max(self.cp * gbar, 1e-6)
            else:
                M = (syy - prev_s) / (e_now - prev_e)
                self.eps_yy -= syy / (M if abs(M) > 1e-6 * self.cp else self.cp * gbar)
            prev_s, prev_e = syy, e_now
        self.gbar = gbar
        self.H = np.maximum(self.H, psi)               # 历史场（弹性 ψ⁺）
        self.solve_d()
        return winds_y(self.d >= D_CRACK)

    def run(self):
        t0 = time.time()
        n_steps = int(round(self.eps_max / self.deps))
        for k in range(1, n_steps + 1):
            eps = k * self.deps
            if self.step(eps):
                return eps, False, time.time() - t0
        return float("nan"), True, time.time() - t0


# ------------------------------------------------------------------ 驱动
def run_one(args):
    sample, metal, pot = args
    spec = load_e1_spectra()
    gc, grid, edges, etype, gam = build_gc_field(sample, metal, pot, spec)
    # 注：不检查 twin 池均值 < general 池均值。Fe（γ₀=0）在 chgnet /
    # mace-mp0-small 下 twin 池（Σ3(112) 113--177）均值反而略高于 general 池
    # （如 92.9 > 84.9）——标签管线用同一对池、同一变换，这是数据既有性质
    #（与论文 Fe 拓扑主导的讨论一致），非本脚本错误。
    pf = PhaseField(gc, metal)
    eps_pf, cens, rt = pf.run()
    lab = {}
    for r in csv.DictReader(open(os.path.join(D6, "labels", "e3_labels.csv"))):
        if int(r["sample"]) == sample and r["metal"] == metal and r["potential"] == pot:
            lab = r
    return dict(sample=sample, metal=metal, potential=pot,
                n_grains=int(grid.max()) + 1,
                twin_length_frac="", eps_label=lab["eps_fail"], censored_label=0,
                eps_pf=("" if cens else round(eps_pf, 4)), censored_pf=int(cens),
                runtime_s=round(rt, 1))


def twin_frac(sample):
    s = int(sample)
    for r in csv.DictReader(open(os.path.join(D6, "results", "network_features.csv"))):
        if int(r["sample"]) == s and r["potential"] == "chgnet":
            return r["twin_length_frac"]
    return ""


def winding_unit_tests():
    N = 32
    m = np.zeros((N, N), bool)
    m[:, 5] = True                              # 贯通列 → y 绕回
    assert winds_y(m), "贯通列应绕回"
    m2 = np.zeros((N, N), bool)
    m2[3:10, 8:9] = True                        # 内部弧段 → 不绕回
    assert not winds_y(m2), "内部弧段不应绕回"
    m3 = np.zeros((N, N), bool)
    m3[0, 4] = True
    m3[N - 1, 4] = True                         # 只触两侧不连通 → 不绕回
    assert not winds_y(m3), "两侧孤立点不应绕回"
    m4 = np.zeros((N, N), bool)
    m4[0, 4] = True
    m4[1:5, 4] = True
    m4[5:N - 1, 4] = True
    m4[N - 1, 4] = True                         # 经内部连通两侧 → 绕回
    assert winds_y(m4), "跨两侧连通应绕回"
    print("[test] winds_y 单测 4 项通过")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="单 run 冒烟")
    ap.add_argument("--jobs", type=int, default=10)
    args = ap.parse_args()

    winding_unit_tests()
    subsets = select_subsets()
    if args.smoke:
        s = subsets["Cu"][0]
        rows = [run_one((s, "Cu", "chgnet"))]
        print("[smoke]", rows[0])
        return

    tasks = [(s, m, p) for m in ["Cu", "Ni", "Fe"] for s in subsets[m]
             for p in ["chgnet", "mace-mp0-medium", "mace-mp0-small"]]
    # 断点续跑：partial 文件里已完成的 (sample, metal, pot) 直接跳过
    partial = os.path.join(D6, "results", "e7_phasefield_runs_partial.csv")
    done = {}
    if os.path.exists(partial):
        for r in csv.DictReader(open(partial)):
            done[(int(r["sample"]), r["metal"], r["potential"])] = r
    todo = [t for t in tasks if t not in done]
    print(f"[run] {len(tasks)} runs, {len(done)} 已完成, 待跑 {len(todo)}, "
          f"jobs={args.jobs}", flush=True)
    t0 = time.time()
    fh_p = open(partial, "a", newline="") if todo else None
    if fh_p and not done:
        csv.DictWriter(fh_p, fieldnames=RUN_FIELDS).writeheader()
    with Pool(args.jobs) as pool:
        n_done = len(done)
        for r in pool.imap_unordered(run_one, todo):
            n_done += 1
            if fh_p:
                csv.DictWriter(fh_p, fieldnames=RUN_FIELDS).writerow(r)
                fh_p.flush()
            print(f"[{n_done}/{len(tasks)}] {r['sample']:>3} {r['metal']:>2} "
                  f"{r['potential']:<16} pf={'CENS' if int(r['censored_pf']) else r['eps_pf']}"
                  f" lab={r['eps_label']} {r['runtime_s']}s", flush=True)
    if fh_p:
        fh_p.close()
    rows = list(csv.DictReader(open(partial)))
    print(f"[run] 完成 {time.time() - t0:.0f}s")

    for r in rows:
        r["twin_length_frac"] = twin_frac(r["sample"])
    rows.sort(key=lambda r: (r["metal"], r["sample"], r["potential"]))
    out = os.path.join(D6, "results", "e7_phasefield_runs.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"→ {out}")

    # 汇总 ρ：每 (metal, potential)
    summ = []
    for metal in ["Cu", "Ni", "Fe"]:
        for pot in ["chgnet", "mace-mp0-medium", "mace-mp0-small"]:
            sub = [r for r in rows if r["metal"] == metal and r["potential"] == pot]
            y_lab = np.array([float(r["eps_label"]) for r in sub])
            y_pf = np.array([np.nan if int(r["censored_pf"]) else float(r["eps_pf"])
                             for r in sub])
            ok = ~np.isnan(y_pf)
            rho_s = (spearmanr(y_lab[ok], y_pf[ok]).statistic if ok.sum() >= 4 else float("nan"))
            # 删失按最差秩（0.061 挂尾）
            y_w = np.where(np.isnan(y_pf), EPS_MAX + DEPS, y_pf)
            rho_w = spearmanr(y_lab, y_w).statistic
            summ.append(dict(metal=metal, potential=pot, n=len(sub),
                             n_pf_censored=int((~ok).sum()),
                             rho_strict=round(float(rho_s), 4) if rho_s == rho_s else "",
                             rho_cens_worst=round(float(rho_w), 4)))
            print(f"[rho] {metal:>2} {pot:<16} n={len(sub)} cens={int((~ok).sum())} "
                  f"ρ_strict={summ[-1]['rho_strict']} ρ_worst={summ[-1]['rho_cens_worst']}")
    out2 = os.path.join(D6, "results", "e7_phasefield_summary.csv")
    with open(out2, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summ[0].keys()))
        w.writeheader()
        w.writerows(summ)
    print(f"→ {out2}")

    # ---- 微结构级 ρ（势平均，避免同微结构跨势伪重复）----
    # 每样本：非删失势的 eps_pf 均值 vs 标签均值；全删失 → 挂最差秩
    rows_micro = []
    for metal in ["Cu", "Ni", "Fe"]:
        sub = [r for r in rows if r["metal"] == metal]
        sids = sorted({int(r["sample"]) for r in sub})
        y_lab, y_pf = [], []
        for s in sids:
            rr = [r for r in sub if int(r["sample"]) == s]
            y_lab.append(np.mean([float(r["eps_label"]) for r in rr]))
            ok = [float(r["eps_pf"]) for r in rr if int(r["censored_pf"]) == 0]
            y_pf.append(np.mean(ok) if ok else EPS_MAX + DEPS)
        res = spearmanr(y_lab, y_pf)
        rows_micro.append(dict(metal=metal, n_micro=len(sids),
                               n_any_censored=sum(
                                   1 for s in sids
                                   if any(int(r["censored_pf"]) for r in sub
                                          if int(r["sample"]) == s)),
                               rho_micro=round(float(res.statistic), 4),
                               p_value=round(float(res.pvalue), 4)))
        print(f"[rho-micro] {metal:>2} n={len(sids)} "
              f"ρ={rows_micro[-1]['rho_micro']:+.3f} p={rows_micro[-1]['p_value']:.3f}")
    out3 = os.path.join(D6, "results", "e7_phasefield_micro.csv")
    with open(out3, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_micro[0].keys()))
        w.writeheader()
        w.writerows(rows_micro)
    print(f"→ {out3}")


if __name__ == "__main__":
    main()
