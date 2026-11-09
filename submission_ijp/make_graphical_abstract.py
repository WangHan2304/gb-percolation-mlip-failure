#!/usr/bin/env python3
"""Graphical abstract for the IJP submission.

Rendered at final physical size (13.6 x 5.4 cm) and 300 dpi, so the exported
PNG satisfies Elsevier's minimum of 531 x 1328 px (h x w) and stays readable
when printed at 5 x 13 cm. Quantitative panels use the values reported in the
manuscript; the network panel is a schematic Voronoi cell, not a result.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "graphical_abstract.png")

C_POT = ["#4C72B0", "#DD8452", "#55A868"]
C_MET = {"Cu": "#C44E52", "Ni": "#4C72B0", "Fe": "#2CA02C"}
GREY = "#9a9a9a"

fig = plt.figure(figsize=(5.35, 2.14))
bg = fig.add_axes([0, 0, 1, 1])
bg.set_xlim(0, 1)
bg.set_ylim(0, 1)
bg.axis("off")

# --- title and take-home line ---------------------------------------------
bg.text(0.5, 0.965, "Boundary energy $\\times$ network topology $\\rightarrow$ intergranular failure ranking",
        ha="center", va="top", fontsize=8.6, fontweight="bold")
bg.text(0.5, 0.055,
        "Topology alone blind-predicts percolation at AUC 0.77 (95% CI 0.62\u20130.90); the equivalent-damage\n"
        "rule is more permissive than a fracture criterion \u2014 the label is bounded, not certified.",
        ha="center", va="bottom", fontsize=5.8, style="italic", linespacing=1.6)

# --- four stage boxes ------------------------------------------------------
X0, W, Y0, H, GAP = 0.025, 0.176, 0.315, 0.525, 0.072
xs = [X0 + i * (W + GAP) for i in range(4)]

for x in xs:
    bg.add_patch(FancyBboxPatch((x, Y0), W, H, boxstyle="round,pad=0.008,rounding_size=0.02",
                                linewidth=0.8, edgecolor="#b8b8b8", facecolor="#fbfbfb"))

for i in range(3):
    xa, xb = xs[i] + W + 0.008, xs[i + 1] - 0.008
    bg.add_patch(FancyArrowPatch((xa, Y0 + H / 2), (xb, Y0 + H / 2), arrowstyle="-|>",
                                 mutation_scale=7, linewidth=0.9, color="#555555"))

# caption under each box (two lines)
caps = ["Boundary-energy spectra\n3 potentials, 3 metals",
        "Energy-decorated\nVoronoi network",
        "Crystal-plasticity\nfailure labels (360)",
        "Phase-field cross-check\nbounds the label"]
for x, c in zip(xs, caps):
    bg.text(x + W / 2, Y0 - 0.03, c, ha="center", va="top", fontsize=5.8, linespacing=1.4)

# --- panel 1: spectra (schematic bar groups) -------------------------------
ax1 = fig.add_axes([xs[0] + 0.018, Y0 + 0.055, W - 0.036, H - 0.10])
groups = ["Cu", "Ni", "Fe"]
vals = np.array([[120, 140, 110],   # chgnet
                 [180, 205, 170],   # mace-mp0-medium
                 [165, 190, 150]])  # mace-mp0-small
idx = np.arange(len(groups))
for k in range(3):
    ax1.bar(idx + (k - 1) * 0.26, vals[k], width=0.24, color=C_POT[k], edgecolor="none")
ax1.set_xticks(idx)
ax1.set_xticklabels(groups, fontsize=5.4)
ax1.set_yticks([])
ax1.tick_params(length=0, pad=1)
for s in ax1.spines.values():
    s.set_visible(False)
ax1.set_ylabel("$\\gamma$ (mJ m$^{-2}$)", fontsize=5.2, labelpad=1)

# --- panel 2: schematic Voronoi network with high-energy sub-network --------
ax2 = fig.add_axes([xs[1] + 0.018, Y0 + 0.072, W - 0.036, H - 0.117])
rng = np.random.default_rng(7)
pts = rng.uniform(0, 1, (26, 2))
from scipy.spatial import Voronoi
vor = Voronoi(pts)
for (a, b), rv in zip(vor.ridge_vertices, vor.ridge_points):
    if a < 0 or b < 0:
        continue
    p, q = vor.vertices[a], vor.vertices[b]
    if np.any(np.abs([p, q]) > 1.15):
        continue
    e = abs(rng.normal())
    hot = e > 1.25
    ax2.plot([p[0], q[0]], [p[1], q[1]], lw=1.7 if hot else 0.55,
             color="#C44E52" if hot else GREY, zorder=3 if hot else 2)
ax2.set_xlim(-0.02, 1.02)
ax2.set_ylim(-0.02, 1.02)
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_frame_on(False)
bg.text(xs[1] + W / 2, 0.352, "high-$\\gamma$ sub-network", ha="center", va="center",
        fontsize=4.8, color="#C44E52")

# --- panel 3: label balance -----------------------------------------------
ax3 = fig.add_axes([xs[2] + 0.018, Y0 + 0.055, W - 0.036, H - 0.10])
per = 155
tot = 360
ax3.barh([0], [per], color="#C44E52", height=0.55)
ax3.barh([0], [tot - per], left=[per], color="#dddddd", height=0.55)
ax3.set_xlim(0, tot)
ax3.set_ylim(-0.65, 0.75)
ax3.set_yticks([])
ax3.set_xticks([0, 180, 360])
ax3.set_xticklabels(["0", "180", "360"], fontsize=5.2)
ax3.tick_params(length=0, pad=1)
for s in ax3.spines.values():
    s.set_visible(False)
ax3.text(per / 2, 0.34, "percolated", ha="center", va="bottom", fontsize=4.5, color="#8c2f32")
ax3.text(per + (tot - per) / 2, 0.34, "censored", ha="center", va="bottom", fontsize=4.5, color="#777777")

# --- panel 4: rule versus phase field ------------------------------------
ax4 = fig.add_axes([xs[3] + 0.038, Y0 + 0.095, W - 0.058, H - 0.160])
metals = ["Cu", "Ni", "Fe"]
rule = [1.0, 1.0, 1.0]
pf = [37 / 60, 42 / 54, 15 / 24]
ii = np.arange(3)
ax4.bar(ii - 0.18, rule, width=0.32, color="#bbbbbb", edgecolor="none")
ax4.bar(ii + 0.18, pf, width=0.32, color="#4C72B0", edgecolor="none")
ax4.set_xticks(ii)
ax4.set_xticklabels(metals, fontsize=5.4)
ax4.set_yticks([0, 0.5, 1.0])
ax4.set_yticklabels(["0", "50", "100%"], fontsize=5.0)
ax4.tick_params(length=0, pad=1)
for s in ax4.spines.values():
    s.set_visible(False)
ax4.text(0.02, 1.06, "rule", transform=ax4.transAxes, fontsize=4.8, color="#888888")
ax4.text(0.98, 1.06, "phase field", transform=ax4.transAxes, ha="right",
         fontsize=4.8, color="#4C72B0")

fig.savefig(OUT, dpi=300, facecolor="white")
plt.close(fig)
print("\u2192", OUT)
