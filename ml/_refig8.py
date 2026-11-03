"""临时驱动：从既有 CSV 重出 Fig.8（跳过 bootstrap）。"""
import sys
sys.path.insert(0, "/home/cl/20241015论文/ml")
import pandas as pd
import e4_blindtest_ext as X

res = pd.read_csv("/home/cl/20241015论文/results/e4b_blind_ci.csv")
aud = pd.read_csv("/home/cl/20241015论文/results/e4b_prediction_audit.csv")
base, ext = X.load_all()
gtr, gca, gte = X.split_orig40(base)
ext_g = [] if ext is None else sorted(ext["sample"].unique().tolist())
scopes = {"orig8": list(gte), "all20": list(gte) + ext_g}
if ext_g:
    scopes["ext12"] = ext_g
X.figs(res, aud, scopes)
