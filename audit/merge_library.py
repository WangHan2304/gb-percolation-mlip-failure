#!/usr/bin/env python3
"""E1 全库合并：主库 3 CSV + 两轮重试 CSV → 单一 gb_energy_library.csv
策略：按 (potential, metal, hkl) 分组，取 flag 最优行（ok > ok_lowdmin > alt > alt_lowdmin > failed）；
同 flag 内取 γ 较低者。输出附 source 列标记来源文件。"""
import csv, glob, os

RANK = {"ok": 0, "ok_lowdmin": 1, "alt": 2, "alt_lowdmin": 3, "failed": 4}
RES = os.path.join(os.path.dirname(__file__), "..", "results")

def main():
    files = [f for f in sorted(glob.glob(os.path.join(RES, "gb_energy_library*.csv"))) +
             sorted(glob.glob(os.path.join(RES, "gb_retry*.csv")))
             if "partial" not in os.path.basename(f) and "merged" not in os.path.basename(f)]
    best = {}
    for f in files:
        src = os.path.basename(f)
        with open(f) as fh:
            for r in csv.DictReader(fh):
                key = (r["potential"], r["metal"], r["hkl"])
                if "flag" not in r or r["flag"] not in RANK:
                    continue
                rank = RANK[r["flag"]]
                try:
                    g = float(r["gamma_mJ_per_m2"]) if r["gamma_mJ_per_m2"] not in ("-", "") else 1e9
                except ValueError:
                    g = 1e9
                # 合理性闸门：弛豫塌缩（dmin<1.0 Å 或 |γ|>1e5）视为无效，降级到 failed 之后
                try:
                    dmin = float(r["min_dist_A"]) if r["min_dist_A"] not in ("-", "") else None
                except ValueError:
                    dmin = None
                if (dmin is not None and dmin < 1.0) or g < -1e5 or g > 1e5:
                    rank = 5
                    r = dict(r, flag="collapsed" if r["flag"] != "failed" else r["flag"])
                cur = best.get(key)
                cand = (rank, g, r, src)
                if cur is None or (rank, g) < (cur[0], cur[1]):
                    best[key] = cand
    rows = sorted(best.values(), key=lambda t: (t[2]["potential"], t[2]["metal"], t[2]["hkl"]))
    out_m = os.path.join(RES, "gb_energy_library_merged.csv")
    cols = ["potential", "metal", "structure", "hkl", "sigma", "state",
            "gamma_mJ_per_m2", "a0_A", "natoms", "min_dist_A", "flag", "source"]
    with open(out_m, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for rank, g, r, src in rows:
            r2 = dict(r); r2["source"] = src
            w.writerow({c: r2.get(c, "") for c in cols})
    n_ok = sum(1 for t in rows if t[0] <= 1)
    print(f"合并 {len(files)} 个文件 -> {out_m}")
    print(f"总组合 {len(rows)}，其中可用(ok/ok_lowdmin) {n_ok}")
    for rank, g, r, src in rows:
        mark = "OK " if rank <= 1 else ("alt" if rank <= 3 else "FAIL")
        print(f"  [{mark}] {r['potential']:16s} {r['metal']:3s} {r['hkl']:>10s} Σ{r['sigma']:>2s} "
              f"γ={r['gamma_mJ_per_m2']:>8s} dmin={r['min_dist_A']:>5s} flag={r['flag']:<11s} src={src}")

if __name__ == "__main__":
    main()
