# Grain-boundary network percolation meets machine-learned grain-boundary energies

**Archived DOI:** https://doi.org/10.5281/zenodo.23912047

Code, data and manuscript sources for the manuscript:

> **Grain-boundary network percolation meets machine-learned grain-boundary energies: blind prediction of intergranular failure susceptibility**
> Han Wang\(^{1}\), Mengjia Chen\(^{2,1}\) — \({}^{1}\)Zhejiang Lab, \({}^{2}\)School of Materials Science and Engineering, Zhejiang University

The pipeline composes machine-learned grain-boundary energies (Cu, Ni, Fe × chgnet/MACE-MP), energy-decorated periodic Voronoi microstructures with percolation descriptors, crystal-plasticity-derived intergranular-failure labels, and a microstructure-grouped blind test with split-conformal uncertainty quantification and an error-propagation audit.

## Repository contents

| Directory | Content |
|---|---|
| `ml/` | Blind machine-learning protocol (E4): grouped ablation, blind test, conformal UQ, applicability domain, error-propagation audit (E5) |
| `propagation/` | Label generation and damage-law robustness audit (E3/E6). Raw HDF5 campaign runs (`e3_runs/`, 5.2 GB) are **not** included; derived labels are in `labels/` |
| `networks/` | Periodic Voronoi RVE generation and network extraction (E2) |
| `results/` | Boundary-energy spectra and audit (E1), network features, ablation/blind-test tables, propagation tables (CSV) |
| `labels/` | Crystal-plasticity failure labels, extended label set (CSV) |
| `figures/` | Figure PNGs used in the manuscript |
| `paper/` | Full manuscript source (`manuscript.tex`, `manuscript.md`) and `make_figs.py` |
| `submission_npj_comp_mat/` | npj Computational Materials submission source, cover letter |
| `audit/`, `logs/` | Verification notes and campaign logs |

## Reproducing

- **Figures**: `python paper/make_figs.py`
- **Manuscript PDFs**: `cd paper && pdflatex manuscript.tex && pdflatex manuscript.tex` (same for `submission_npj_comp_mat/`)

## Data availability

All derived datasets supporting this work are in `results/` and `labels/` (file-level mapping is given in the Data availability statement of the manuscript).

## Funding

This work was supported by the `Pioneers and Leaders + X' R&D Plan of the Zhejiang Provincial Major Sci-Tech Innovation Platform Project (Grant No. 2025SSYS0006), corresponding to the Key Research Project of Zhejiang Lab (Grant No. 3700-3AA250100)`.

## Correspondence

Han Wang — wanghan2304@zhejianglab.org
