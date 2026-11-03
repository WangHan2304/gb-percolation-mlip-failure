"""E1: GB 能谱审计 —— uMLIP 晶界能（单 GB slab + 表面参考扣除）。

构造法（统一代码路径，免晶体学查表）：
  grain1 = (hkl) 面切割的立方晶格平板；grain2 = 跨镜面 z=z_m 的镜像 + 刚性平移态 (t, z_frac)。
  镜像 ≡ 绕法向 [hkl] 转 180°（反演不改变格子取向）⇒ 失配 180°[hkl]，
  经典等价：(111)→Σ3 共格孪晶、(112)→Σ3 非共格孪晶、(210)/(310)→Σ5、(221)→Σ9、(113)→Σ11。
  单 GB + 真空（pbc z 关闭）：
    γ_GB = (E_GB − N_GB·e_atom)/A − (E_ref − N_ref·e_atom)/A
    其中 ref = 同厚度同终止的纯表面 slab（2 个自由表面）——表面能成对扣除。
  平移态扫描（z_frac × 侧向 t，t 相对扩胞原胞、网格 1/4 覆盖原始面内原胞）。
  E2 教训落地：layers≥6 保证正确堆垛（layers=1 会铺成 AAA）；无环回结点；
  min_dist 自检（<2.0 Å 候选淘汰）。

运行（用 D5 venv）：
  python audit/gb_energy_audit.py --metals Cu --potentials chgnet --planes 1 1 1     # 冒烟
  python audit/gb_energy_audit.py --metals Cu                                       # 单金属全跑
"""
from __future__ import annotations

import argparse
import csv
import math
from fractions import Fraction
from itertools import product
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import bulk, surface
from ase.filters import FrechetCellFilter
from ase.geometry import get_distances
from ase.optimize import FIRE

import sys
sys.path.insert(0, str(Path(__file__).parent))
from potentials import get_calculator, POTENTIALS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

FMAX = 0.02               # eV/Å 弛豫判据
MAX_STEPS = 150           # FIRE 步数上限（防病态结构静默烧机时）
Z_SCALES = (1.00, 1.05)   # z 向标度扫描（面积固定，吸收 GB/表面弛豫）
N_RELAX = 1               # 预筛后弛豫的候选数（提速：2→1）
MIN_INPLANE = 9.0         # Å，面内最小边长
MIN_THICK = 8.0           # Å，半晶块最小厚度
VAC = 9.0                 # Å，真空层
DUP_TOL = 0.3             # Å，共点原子去重容差
MIN_DIST_OK = 2.0         # Å，界面最小原子间距验收线（可用 --min-dist 覆盖；
                          # 小面间距晶面侧向周期=1，保形镜像上限 ~P/2，降阈值后靠弛豫自愈，
                          # 弛豫后 dmin 会写入 CSV 供验收裁决）
EV_A2_TO_MJM2 = 16021.766

FCC_PLANES = [(1, 1, 1), (1, 1, 2), (2, 1, 0), (3, 1, 0), (2, 2, 1), (1, 1, 3)]
BCC_PLANES = [(1, 1, 2), (1, 1, 1), (2, 1, 0), (3, 1, 0), (1, 3, 0)]
Z_FRACS = (0.0, 0.25, 0.5, 0.75)   # v2：4 值网格（v2 修复小面间距晶面 d≤1.3 Å 全灭问题）
# 相对原始面内原胞的 1/4 网格平移态
T_STATES = ((0, 0), (0.25, 0), (0.5, 0), (0.75, 0), (0, 0.25),
            (0, 0.5), (0, 0.75), (0.25, 0.25), (0.5, 0.5), (0.25, 0.75))

PRIM = {
    "fcc": np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]),
    "bcc": np.array([[0.5, 0.5, -0.5], [0.5, -0.5, 0.5], [-0.5, 0.5, 0.5]]),
}


def sigma_formula(hkl) -> int:
    """镜像跨 (hkl) ⇒ 失配 180°[hkl] ⇒ Σ = s(奇) 或 s/2(偶)。
    核对：(111)→3、(112)→3、(210)/(310)→5、(221)→9、(113)→11。"""
    h, k, l = hkl
    s = h * h + k * k + l * l
    return s if s % 2 else s // 2


def sigma_count(hkl, structure: str) -> int:
    """纯取向 CSL 计数（交叉核对用；镜像含仿射平移，实际 Σ 可低于此值）。"""
    h, k, l = hkl
    s = h * h + k * k + l * l
    n = np.array(hkl, dtype=float) / math.sqrt(s)
    O_cubic = 2.0 * np.outer(n, n) - np.eye(3)
    B = PRIM[structure]
    O_prim = np.linalg.inv(B) @ O_cubic @ B
    fracs = [Fraction(float(x)).limit_denominator(400) for x in O_prim.flat]
    q = math.lcm(*[f.denominator for f in fracs])
    Op = np.array([[float(fracs[i * 3 + j]) * q for j in range(3)] for i in range(3)])
    M = (Op - q * np.eye(3)).astype(int)
    return sum(1 for i, j, kk in product(range(q), repeat=3)
               if np.all((M @ np.array([i, j, kk])) % q == 0))


def relax_bulk(pot: str, symbol: str, structure: str, a_guess: float):
    at = bulk(symbol, structure, a=a_guess, cubic=True)
    at.calc = get_calculator(pot)
    FIRE(FrechetCellFilter(at), logfile=None).run(fmax=FMAX, steps=MAX_STEPS)
    return at.cell.lengths()[0], at.get_potential_energy() / len(at)


def _min_dist(pos, cell, pbc) -> float:
    _, dlen = get_distances(pos, cell=cell, pbc=pbc)
    np.fill_diagonal(dlen, np.inf)
    return float(dlen.min())


def _slab_positions(bulk_at, hkl):
    """公共部分：正确堆垛的 (hkl) 平板 → (pos, syms, cell2d, lz, rep)。"""
    h, k, l = hkl
    slab = surface(bulk_at, (h, k, l), layers=6, vacuum=None, periodic=True)
    cell_arr = slab.cell.array.copy()
    assert abs(cell_arr[0, 2]) < 1e-6 and abs(cell_arr[1, 2]) < 1e-6 \
        and abs(cell_arr[2, 0]) < 1e-6 and abs(cell_arr[2, 1]) < 1e-6, \
        f"surface 胞取向异常：{cell_arr}"
    lx, ly, lz = slab.cell.lengths()
    rep = (max(1, round(MIN_INPLANE / lx)), max(1, round(MIN_INPLANE / ly)), 1)
    slab = slab.repeat((rep[0], rep[1], 1))
    lx, ly, lz = slab.cell.lengths()
    pos = slab.get_positions().copy()
    pos[:, 2] -= pos[:, 2].min()
    syms = list(slab.get_chemical_symbols())
    while (pos[:, 2].max() - pos[:, 2].min() + 1e-9) < MIN_THICK:
        pos = np.vstack([pos, pos + np.array([0.0, 0.0, lz])])
        syms = syms + syms
        lz *= 2.0
    return pos, syms, slab.cell.array[:2].copy(), lz, rep[:2]


def _vacuum_at(allp, allsyms, cell2d, c_new) -> tuple[Atoms, float]:
    cell_full = np.vstack([cell2d, [0.0, 0.0, c_new]])
    keep = np.ones(len(allp), dtype=bool)
    _, dlen = get_distances(allp, cell=cell_full, pbc=[True, True, False])
    for i in range(len(allp)):
        if not keep[i]:
            continue
        for j in np.where(dlen[i] < DUP_TOL)[0]:
            if j > i:
                keep[j] = False
    allp = allp[keep]
    allsyms = [s for s, kf in zip(allsyms, keep) if kf]
    at = Atoms(symbols=allsyms, positions=allp, cell=cell_full, pbc=[True, True, False])
    return at, _min_dist(at.get_positions(), at.cell, [True, True, False])


def build_reference(bulk_at, hkl) -> tuple[Atoms, float]:
    """纯表面参考 slab（同厚度同终止，2 自由表面）。"""
    pos, syms, cell2d, _, _ = _slab_positions(bulk_at, hkl)
    area = abs(np.linalg.norm(np.cross(cell2d[0], cell2d[1])))
    z_top = pos[:, 2].max()
    at, _ = _vacuum_at(pos, syms, cell2d, z_top + VAC)
    return at, area


def build_bicrystal(bulk_at, hkl, z_frac: float, t_prim):
    """单 GB slab：grain1 + 镜像 grain2(+平移态)，z 向真空。
    t_prim：相对原始面内原胞的分数平移（网格 1/4），内部换算到扩胞坐标。"""
    pos, syms, cell2d, lz, rep = _slab_positions(bulk_at, hkl)
    area = abs(np.linalg.norm(np.cross(cell2d[0], cell2d[1])))
    period = lz / 6.0
    z_top = pos[:, 2].max()
    z_m = z_top + z_frac * period
    p2 = pos.copy()
    p2[:, 2] = 2.0 * z_m - p2[:, 2]
    p2 += (t_prim[0] / rep[0]) * cell2d[0] + (t_prim[1] / rep[1]) * cell2d[1]
    at, dmin = _vacuum_at(np.vstack([pos, p2]), syms + syms, cell2d, 2.0 * z_m + VAC)
    return at, dmin, area


def _relax_zscan(at: Atoms, calc) -> tuple[float, float]:
    """原子弛豫 × z 标度扫描 → (E_best, dmin)。"""
    best = (np.inf, np.nan)
    for s in Z_SCALES:
        trial = at.copy()
        cell = trial.cell.array.copy()
        cell[2] = cell[2] * s
        trial.set_cell(cell, scale_atoms=True)
        trial.calc = calc
        FIRE(trial, logfile=None).run(fmax=FMAX, steps=MAX_STEPS)
        e = trial.get_potential_energy()
        if e < best[0]:
            best = (e, _min_dist(trial.get_positions(), trial.cell, [True, True, False]))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metals", nargs="+", default=["Cu", "Ni", "Fe"])
    ap.add_argument("--potentials", nargs="+", default=list(POTENTIALS))
    ap.add_argument("--planes", nargs="+", type=int, default=None,
                    help="展平 hkl 列表，如 1 1 1 2 1 0")
    ap.add_argument("--out", default=None, help="CSV 输出路径（并行作业分文件防写冲突）")
    ap.add_argument("--min-dist", type=float, default=None,
                    help="覆盖 MIN_DIST_OK（小面间距晶面自救用；flag 标 ok_lowdmin）")
    args = ap.parse_args()
    if args.min_dist is not None:
        global MIN_DIST_OK
        MIN_DIST_OK = args.min_dist

    struct_map = {"Cu": "fcc", "Ni": "fcc", "Al": "fcc", "Fe": "bcc"}
    a_guess = {"Cu": 3.615, "Ni": 3.52, "Al": 4.05, "Fe": 2.855}

    out_csv = Path(args.out) if args.out else RESULTS / "gb_energy_library.csv"
    new_file = not out_csv.exists()
    fout = open(out_csv, "a", newline="")
    wr = csv.writer(fout)
    if new_file:
        wr.writerow(["potential", "metal", "structure", "hkl", "sigma", "state",
                     "gamma_mJ_per_m2", "a0_A", "natoms", "min_dist_A", "flag"])

    for metal in args.metals:
        st = struct_map[metal]
        plist = [tuple(args.planes[i:i + 3]) for i in range(0, len(args.planes), 3)] \
            if args.planes else (FCC_PLANES if st == "fcc" else BCC_PLANES)
        for pot in args.potentials:
            print(f"\n=== {metal} ({st}) × {pot} ===", flush=True)
            a0, e_atom = relax_bulk(pot, metal, st, a_guess[metal])
            print(f"  a0={a0:.4f} Å  E_atom={e_atom:.4f} eV", flush=True)
            bulk_at = bulk(metal, st, a=a0, cubic=True)
            calc = get_calculator(pot)
            for hkl in plist:
                sig = sigma_formula(hkl)
                cnt = sigma_count(hkl, st)
                if cnt != sig:
                    print(f"  [σ核对] {hkl}: 公式 Σ{sig} vs 取向计数 Σ{cnt}（以公式为准）", flush=True)
                # 表面参考
                ref, area = build_reference(bulk_at, hkl)
                e_ref, _ = _relax_zscan(ref, calc)
                gamma_surf2 = (e_ref - len(ref) * e_atom) / area * EV_A2_TO_MJM2  # = 2γ_surf
                print(f"  [{hkl}] 2γ_surf={gamma_surf2:.0f} mJ/m² (N_ref={len(ref)})", flush=True)
                # 平移态扫描 → 验收 → 预筛（含表面扣除）→ 弛豫最优 2 个
                cands = []
                for zf, tf in product(Z_FRACS, T_STATES):
                    try:
                        at, dmin0, area2 = build_bicrystal(bulk_at, hkl, zf, tf)
                    except Exception as exc:  # noqa: BLE001
                        print(f"  [{hkl}|zf={zf} t={tf}] 构造失败: {type(exc).__name__}", flush=True)
                        continue
                    if dmin0 < MIN_DIST_OK:
                        continue
                    at.calc = calc
                    g0 = (at.get_potential_energy() - len(at) * e_atom) / area2 * EV_A2_TO_MJM2 - gamma_surf2
                    cands.append((g0, zf, tf, at, dmin0, area2))
                if not cands:
                    print(f"  [{hkl}] 全部平移态构造不合格（dmin < {MIN_DIST_OK}）", flush=True)
                    wr.writerow([pot, metal, st, "".join(map(str, hkl)), sig, "-", "",
                                 f"{a0:.4f}", 0, "", "failed"])
                    fout.flush()
                    continue
                cands.sort(key=lambda t: t[0])
                relaxed = []
                for g0, zf, tf, at, dmin0, area2 in cands[:N_RELAX]:
                    e_best, dmin = _relax_zscan(at, calc)
                    g = (e_best - len(at) * e_atom) / area2 * EV_A2_TO_MJM2 - gamma_surf2
                    state = f"z{zf:.1f}_t{tf[0]:.2f}{tf[1]:.2f}"
                    relaxed.append((state, g, dmin, len(at)))
                    print(f"  [{hkl}] Σ{sig} {state}: γ={g:.1f} mJ/m²  (N={len(at)}, dmin={dmin:.3f}, γ0={g0:.0f})", flush=True)
                relaxed.sort(key=lambda t: t[1])
                for i, (state, g, d, n) in enumerate(relaxed):
                    healed = d >= 2.0
                    flg = ("ok" if i == 0 else "alt") if healed else ("ok_lowdmin" if i == 0 else "alt_lowdmin")
                    wr.writerow([pot, metal, st, "".join(map(str, hkl)), sig, state,
                                 f"{g:.2f}", f"{a0:.4f}", n, f"{d:.3f}", flg])
                fout.flush()
    fout.close()
    print(f"\n写入 {out_csv}")


if __name__ == "__main__":
    main()
