#!/usr/bin/env python3
"""Plot the footprint matrix produced by bindercluster.py as a heatmap with its
dendrogram attached, so cluster structure and contact pattern are read together.

Rows are binder designs, ordered by the dendrogram so that neighbouring rows are
similar. Columns are target residues in sequence order. Cell colour is the contact
score from score.csv. An optional colour strip marks cluster membership at
one chosen k.

This script only draws; all numbers come from the bindercluster.py output
directory. Nothing here is needed to interpret the analysis - cluster_assignments.csv
already states which design belongs to which cluster at every k.

USAGE
    python '/The path where you store the script/binderplot.py'\
    '/Where contains the score.csv and cluster_assignments.csv as the output of bindercluster.py/'\
    '/Input the output path you want for figures here'

INPUTS (inside the first argument, as written by bindercluster.py)
    score.csv                required, the matrix that was clustered
    cluster_assignments.csv  optional, needed for --k

Requires: numpy, pandas, scipy, matplotlib.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#4a3aa7", "#7a7a76"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#dcdcd8", "#fcfcfb"


def parse_args():
    p = argparse.ArgumentParser(
        prog="binderplot.py",
        description="Heatmap of binder footprints with the clustering dendrogram attached.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("indir", help="a bindercluster.py output directory")
    p.add_argument("outdir", nargs="?", help="where to write the figure "
                                             "(default: same as indir)")
    p.add_argument("--k", type=int,
                   help="draw a cluster colour strip at this k "
                        "(needs cluster_assignments.csv)")
    p.add_argument("--title", help="figure title, usually the target name")
    p.add_argument("--metric", default="cosine-average",
                   choices=["cosine-average", "euclidean-ward"],
                   help="must match the bindercluster.py run")
    p.add_argument("--no-tree", action="store_true",
                   help="drop the dendrogram; rows stay in tree order")
    p.add_argument("--cmap", default="viridis",
                   help="matplotlib colormap for the heatmap")
    p.add_argument("--figsize", type=float, nargs=2, metavar=("W", "H"),
                   default=[16.0, 10.0], help="figure size in inches")
    p.add_argument("--boundary-color", default="white",
                   help="colour of the lines drawn between clusters; "
                        "'none' removes them")
    p.add_argument("--label-rows", action="store_true",
                   help="write the design name on every row; only legible for small sets")
    p.add_argument("--tick-step", type=int, default=0,
                   help="label every Nth column; 0 picks a readable step automatically")
    a = p.parse_args()
    a.metric, a.linkage = a.metric.split("-")
    return a


def main():
    a = parse_args()
    indir = Path(a.indir)
    outdir = Path(a.outdir) if a.outdir else indir
    outdir.mkdir(parents=True, exist_ok=True)

    score = pd.read_csv(indir / "score.csv").set_index("pdb_file")
    H = score.to_numpy(float)
    n, m = H.shape

    D = pdist(H, metric=a.metric)
    Z = linkage(np.nan_to_num(D, nan=1.0), method=a.linkage)
    order = dendrogram(Z, no_plot=True)["leaves"]

    labels = None
    if a.k:
        single = indir / f"clusters_k{a.k}.csv"
        sweep = indir / "cluster_assignments.csv"
        if single.exists():
            assign = pd.read_csv(single).set_index("pdb_file")
            labels = assign.loc[score.index, "cluster"].to_numpy()
        elif sweep.exists():
            assign = pd.read_csv(sweep).set_index("pdb_file")
            if f"k{a.k}" not in assign.columns:
                raise SystemExit(f"k={a.k} is not in {sweep.name}")
            labels = assign.loc[score.index, f"k{a.k}"].to_numpy()
        else:
            raise SystemExit(f"--k needs {single.name} or {sweep.name}")

    # layout: dendrogram | cluster strip | heatmap | colourbar
    widths = [0.0 if a.no_tree else 1.6,
              0.34 if labels is not None else 0.0, 8.0, 0.22]
    fig = plt.figure(figsize=tuple(a.figsize), facecolor=SURFACE)
    gs = fig.add_gridspec(1, 4, width_ratios=widths, wspace=0.015,
                          left=0.055, right=0.95, top=0.91, bottom=0.17)

    if not a.no_tree:
        ax_tree = fig.add_subplot(gs[0, 0])
        dendrogram(Z, ax=ax_tree, orientation="left", no_labels=True,
                   color_threshold=0, link_color_func=lambda _: INK2)
        ax_tree.invert_yaxis()
        ax_tree.set_facecolor(SURFACE)
        ax_tree.set_xticks([])
        ax_tree.set_yticks([])
        for s in ax_tree.spines.values():
            s.set_visible(False)
        ax_tree.set_ylabel(f"{n} designs, ordered by the tree", fontsize=10, color=INK2)

    if labels is not None:
        ax_strip = fig.add_subplot(gs[0, 1])
        uniq = {c: i for i, c in enumerate(pd.unique(labels[order]))}
        strip = np.array([[uniq[c]] for c in labels[order]])
        ax_strip.imshow(strip, aspect="auto", interpolation="nearest",
                        cmap=matplotlib.colors.ListedColormap(
                            [SERIES[i % len(SERIES)] for i in range(len(uniq))]))
        ax_strip.set_xticks([])
        ax_strip.set_yticks([])
        for s in ax_strip.spines.values():
            s.set_visible(False)
        ax_strip.set_title(f"k={a.k}", fontsize=8, color=INK2, pad=4)
        # cluster size written inside its block; blocks are contiguous in tree order
        ordered = labels[order]
        start = 0
        for i in range(1, n + 1):
            if i == n or ordered[i] != ordered[start]:
                size = i - start
                if size / n >= 0.03:
                    ax_strip.text(0, (start + i - 1) / 2, str(size), ha="center",
                                  va="center", fontsize=8, color="white",
                                  fontweight="bold")
                start = i
        if a.no_tree:
            ax_strip.set_ylabel(f"{n} designs, ordered by the tree",
                                fontsize=10, color=INK2)

    ax = fig.add_subplot(gs[0, 2])
    im = ax.imshow(H[order], aspect="auto", interpolation="nearest", cmap=a.cmap,
                   vmin=np.nanmin(H), vmax=np.nanmax(H))
    ax.set_facecolor(SURFACE)

    # cluster boundaries; fcluster blocks are contiguous in leaf order
    if labels is not None and a.boundary_color.lower() != "none":
        ordered = labels[order]
        for i in range(1, n):
            if ordered[i] != ordered[i - 1]:
                ax.axhline(i - 0.5, color=a.boundary_color, lw=1.0, alpha=0.9)

    step = a.tick_step or max(1, m // 30)
    ticks = list(range(0, m, step))
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(score.columns[i]).split(":")[0] for i in ticks],
                       rotation=90, fontsize=7, color=INK2)
    ax.set_xlabel("target site", fontsize=10, color=INK2)
    if a.label_rows and n <= 80:
        ax.set_yticks(range(n))
        ax.set_yticklabels([score.index[i] for i in order], fontsize=6, color=INK2)
    else:
        ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(GRID)

    cax = fig.add_subplot(gs[0, 3])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("contact score", fontsize=9, color=INK2)
    cb.ax.tick_params(labelsize=8, colors=INK2)
    cb.outline.set_edgecolor(GRID)

    fig.suptitle(a.title or "Binder footprints", color=INK, fontsize=12,
                 x=0.055, ha="left")
    path = outdir / "heatmap_with_tree.png"
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    main()
