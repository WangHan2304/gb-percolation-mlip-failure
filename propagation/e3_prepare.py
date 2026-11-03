#!/usr/bin/env python3
"""E3 沿晶失效标签：CPFEM 输入准备（样本子集 × 金属 → DAMASK run 目录）

流程：
  1) 从 E2 特征表选 40 样本子集（(n_grains, twin_prob, twin_length_frac) 空间贪心 max-min 覆盖）
  2) 每样本：2D 256² 晶粒网格多数票降采样 4× → 64²（晶粒/边存活率 ≥99%），
     柱状外推 Z=4 层 → 3D GeomGrid → grid.vti + grid_small.npy（后处理用）
     （cell 步长×4，RVE 物理尺寸不变；--down 1 可关闭降采样）
  3) material.yaml：每晶粒一条 material（phase=金属，O=E2 取向矩阵），homogenization=pass
  4) load.yaml：单轴拉伸沿 x（dot_F=1.5e-3/s，t=40s → ε_xx=0.06，N=40，f_out=2 → 21 个输出）

用法：
  python e3_prepare.py --n-select 40 --z 8
输出：
  propagation/e3_runs/<sample:03d>_<metal>/{grid.vti, material.yaml, load.yaml}
  propagation/e3_runs/manifest.csv
"""
import argparse, csv, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D6 = os.path.abspath(os.path.join(HERE, ".."))
VENV_PY = "/home/cl/20261016MLIP传播论文/.venv/bin/python"

# ---- 每金属本构（DAMASK 20240629 alpha7 schema；参数取镜像内官方示例 + 文献）----
# Cu: Hooke_Cu.yaml 原样；plastic 模板取 phenopowerlaw_Cu.yaml（Takeuchi 单晶拟合），
#     但 xi_0=1.6 MPa 为单晶 CRSS，多晶 RVE 首增量会产生 (τ/ξ)^20 子步进风暴
#     （实测 512 单元瞬态 ~2 min、64² ~10 min 级）。改用多晶标定值 xi_0=12 MPa
#     （Taylor 3.06× → 多晶屈服 ≈37 MPa，与退火 Cu 文献一致）；其余键原样。
# Ni: fcc 结构同 Cu 模板，τ 值取纯 Ni 文献量级（xi_0≈30, xi_inf≈210 MPa）
# Fe: examples/config/.../phenopowerlaw_polygonal-ferrite.yaml + Hooke_Fe.yaml（原样采用）
METALS = {
    "Cu": {"lattice": "cF", "C_11": 168.9e9, "C_12": 121.8e9, "C_44": 75.8e9,
           "N_sl": [12], "dot_gamma_0_sl": [3.0e-3], "n_sl": [20], "a_sl": [0.6],
           "xi_0_sl": [12.0e6], "xi_inf_sl": [96.4e6], "h_0_sl-sl": [0.35e9],
           "h_sl-sl": [1, 1, 5.123, 0.574, 1.123, 1.123, 1]},
    "Ni": {"lattice": "cF", "C_11": 251.0e9, "C_12": 150.0e9, "C_44": 123.7e9,
           "N_sl": [12], "dot_gamma_0_sl": [3.0e-3], "n_sl": [20], "a_sl": [0.6],
           "xi_0_sl": [30.0e6], "xi_inf_sl": [210.0e6], "h_0_sl-sl": [0.9e9],
           "h_sl-sl": [1, 1, 5.123, 0.574, 1.123, 1.123, 1]},
    "Fe": {"lattice": "cI", "C_11": 232.2e9, "C_12": 136.4e9, "C_44": 117.0e9,
           "N_sl": [12, 12, 24], "dot_gamma_0_sl": [4.0e-2] * 3, "n_sl": [20] * 3,
           "a_sl": [2.1] * 3, "xi_0_sl": [61e6, 54e6, 52e6],
           "xi_inf_sl": [277e6, 248e6, 237e6], "h_0_sl-sl": [1.0e9] * 3,
           "h_sl-sl": [1.0] + [1.4] * 27},
}
DELTA_UM = 0.05  # 网格步长 μm → 256×256×8 = 12.8×12.8×0.4 μm 柱状 RVE

DOT_F = 1.5e-3   # /s 沿 x
T_END, N_INC, F_OUT = 40.0, 40, 2   # ε_total = 0.06，输出 21 个增量点（Δε=0.003）
# 注：N=100（ε→0.15）版本实测 ~67 min/run（4096 cells/rank，含他任务抢占 7.5 核）；
# 失效/标定集中在 ε≤0.06，压缩到 N=40 使 120 runs 在 4 并发下 ~14 h 可完成。


def load_e1_spectra():
    """{potential: {metal: {"twin":[γ...], "general":[γ...]}}}——与 E2 同规则。"""
    import csv as _csv
    spec = {}
    with open(os.path.join(D6, "results", "gb_energy_library_merged.csv")) as fh:
        for r in _csv.DictReader(fh):
            if r["flag"] != "ok":
                continue
            pot, metal = r["potential"], r["metal"]
            g = float(r["gamma_mJ_per_m2"])
            d = spec.setdefault(pot, {}).setdefault(metal, {"twin": [], "general": []})
            (d["twin"] if r["hkl"].strip() == "111" and r["sigma"].strip() == "3"
             else d["general"]).append(g)
    return spec


def select_subset(n_select):
    """从 E2 特征表按 (n_grains, twin_prob, twin_length_frac) 贪心 max-min 选样本。"""
    with open(os.path.join(D6, "results", "network_features.csv")) as fh:
        all_rows = list(csv.DictReader(fh))
    first_pot = sorted({r["potential"] for r in all_rows})[0]
    rows = [r for r in all_rows if r["metal"] == "Cu" and r["potential"] == first_pot]
    X, ids = [], []
    for r in rows:
        X.append([float(r["n_grains"]) / 200.0, float(r["twin_prob"]) / 0.65,
                  float(r["twin_length_frac"])])
        ids.append(int(r["sample"]))
    X = np.array(X)
    sel = [int(np.argmax(X[:, 1]))]                     # 从孪晶占比最高处起步
    while len(sel) < min(n_select, len(ids)):
        dmin = np.min([np.linalg.norm(X - X[i], axis=1) for i in sel], axis=0)
        dmin[sel] = -1
        sel.append(int(np.argmax(dmin)))
    return sorted(ids[i] for i in sel), len(ids)


def downsample_majority(grid2d, down):
    """f×f 块多数票降采样：40 样本实测晶粒/边存活率 ≥99%（保真度检查见会话记录）。"""
    M = grid2d.shape[0] // down
    blk = grid2d.reshape(M, down, M, down)
    small = np.zeros((M, M), np.int32)
    for i in range(M):
        for j in range(M):
            small[i, j] = np.bincount(blk[i, :, j, :].ravel()).argmax()
    return small


def write_run(sample, metal, z_layers, runs_root, down=1):
    import damask
    d = np.load(os.path.join(D6, "networks", "library", f"sample_{sample:03d}.npz"),
                allow_pickle=True)
    grid2d, R = d["grid"], d["R"]
    n_grains = int(grid2d.max()) + 1
    assert n_grains == len(d["seeds"]), f"sample {sample}: grain id 不连续？"
    assert n_grains == R.shape[0], f"sample {sample}: 取向数与晶粒数不一致"

    if down > 1:
        grid2d = downsample_majority(grid2d, down)
    ny, nx = grid2d.shape                       # grid2d[i, j]：i=row(y)，j=col(x)
    # 平铺序已由唯一标记弹性探针实锤（match=1.0000）：M3[iy,ix,iz] 写入 VTI 后，
    # hdf5 行号 k = iy + M·ix + M²·iz ↔ 晶粒 g[iy,ix]。故按 (y,x,z) 升维不加转置。
    mat3d = np.broadcast_to(grid2d[:, :, None], (ny, nx, z_layers)).copy()
    # 降采样只改分辨率，不改 RVE 物理尺寸（cell 步长 = DELTA_UM × down）
    size = np.array([nx, ny, z_layers], dtype=float) * DELTA_UM * down * 1e-6
    g = damask.GeomGrid(material=mat3d, size=size, origin=np.zeros(3))

    wd = os.path.join(runs_root, f"{sample:03d}_{metal}")
    os.makedirs(wd, exist_ok=True)
    g.save(os.path.join(wd, "grid.vti"))
    if down > 1:
        np.save(os.path.join(wd, "grid_small.npy"), grid2d)

    mp = METALS[metal]
    cm = damask.ConfigMaterial()
    cm["homogenization"]["SX"] = {"N_constituents": 1, "mechanical": {"type": "pass"}}
    # alpha7 schema：elastic/plastic 挂在 phase.<名>.mechanical 下；
    # output=[F,P,Cauchy] → hdf5 中 F/P (n,3,3)、Cauchy (n,6) Voigt 序
    cm["phase"][metal] = {
        "lattice": mp["lattice"],
        "mechanical": {
            "output": ["F", "P", "Cauchy"],
            "elastic": {"type": "Hooke", "C_11": mp["C_11"], "C_12": mp["C_12"],
                        "C_44": mp["C_44"]},
            "plastic": {"type": "phenopowerlaw", "N_sl": mp["N_sl"],
                        "dot_gamma_0_sl": mp["dot_gamma_0_sl"], "n_sl": mp["n_sl"],
                        "xi_0_sl": mp["xi_0_sl"], "xi_inf_sl": mp["xi_inf_sl"],
                        "h_0_sl-sl": mp["h_0_sl-sl"], "h_sl-sl": mp["h_sl-sl"],
                        "a_sl": mp["a_sl"], "atol_xi": 1.0},
        },
    }
    ori = damask.Rotation.from_matrix(R)
    cm = cm.material_add(phase=[metal] * n_grains, O=ori, homogenization="SX")
    cm.save(os.path.join(wd, "material.yaml"))

    load = damask.YAML(f"""---
solver:
  mechanical: spectral_basic

loadstep:
  - boundary_conditions:
      mechanical:
        dot_F: [[{DOT_F}, 0, 0],
                [0,      x, 0],
                [0,      0, x]]
        P: [[x, x, x],
            [x, 0, x],
            [x, x, 0]]
    discretization:
      t: {T_END}
      N: {N_INC}
    f_out: {F_OUT}
""")
    load.save(os.path.join(wd, "load.yaml"))
    return {"sample": sample, "metal": metal, "workdir": os.path.relpath(wd, D6),
            "n_grains": n_grains}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-select", type=int, default=40)
    ap.add_argument("--z", type=int, default=4)
    ap.add_argument("--down", type=int, default=4, help="多数票降采样倍数（1=保原 256²）")
    ap.add_argument("--metals", default="Cu,Ni,Fe")
    ap.add_argument("--samples", default="", help="逗号分隔覆盖自动选样（冒烟用）")
    ap.add_argument("--runs-root", default=os.path.join(HERE, "e3_runs"))
    args = ap.parse_args()

    spec_cache = load_e1_spectra()  # 预加载校验 E1 库可用
    assert spec_cache, "E1 能谱库为空"
    metals = args.metals.split(",")

    if args.samples:
        samples = [int(s) for s in args.samples.split(",")]
    else:
        samples, n_total = select_subset(args.n_select)
        print(f"从 {n_total} 个 E2 样本中选出 {len(samples)} 个：{samples[:10]}...")

    runs_root = args.runs_root
    os.makedirs(runs_root, exist_ok=True)
    manifest = []
    for s in samples:
        for metal in metals:
            row = write_run(s, metal, args.z, runs_root, down=args.down)
            manifest.append(row)
            print(f"[prep] sample {s:03d} {metal}: {row['n_grains']} grains → {row['workdir']}",
                  flush=True)
    with open(os.path.join(runs_root, "manifest.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(manifest[0].keys()))
        w.writeheader()
        w.writerows(manifest)
    print(f"→ {os.path.join(runs_root, 'manifest.csv')}（{len(manifest)} runs）")


if __name__ == "__main__":
    main()
