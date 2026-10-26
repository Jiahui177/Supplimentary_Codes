#!/usr/bin/env python3
"""Cluster de novo binder designs by the footprint they make on a shared target.

No structural superposition is required. Contact distances are measured inside
each complex, so they are invariant to rigid-body placement, and the feature
columns are target sites (heavy atoms, or residues), which are shared by construction.

TYPICAL USE
    # input and output can be positional; a .csv is read as a distance matrix,
    # anything else as a directory of PDB files
    python '/The path where you store the script/bindercluster.py'\
    '/the folder you store pdb files/'\
    '/Your output path'\
    --title "Your job name"

PIPELINE
    PDB complexes
      -> per target site, minimum distance to the binder          (--level)
      -> contact score in [0, 1]                                  (--cutoff, --soft-width)
      -> optional spatial smoothing over the target surface       (--smooth-radius)
      -> optional per-site centring                               (--z)
      -> hierarchical clustering over a k range                   (--metric, --linkage)

OUTPUTS (written to --out)
    diagnostics.png              gain in effective clusters and in cut height vs k
    clustree.png                 cluster tree across k; node area = size, edge width = members moved
    mindist.csv                  structures x sites, minimum distance to the binder (A)
    score.csv                    the matrix actually clustered, after scoring/smoothing/centring
    hit_frequency_*.csv          mean contact score per site
    k_metrics.csv                one row per k: eff_n, d_eff_n, cut_height, d_cut_height,
                                 silhouette, n_singletons, largest_cluster, what_split_next
    cluster_assignments.csv      cluster id of every structure at every k
    cluster_profiles.csv         per (k, cluster): size, mean score, top contacted sites
    target_conformation_qc.csv   per structure, mean RMSD of the target chain to all others
    skipped.csv                  structures that could not be used, with the reason

READING THE DIAGNOSTICS
    d_eff_n        exp(Shannon entropy of cluster sizes) gained at this k. A large
                   cluster splitting in half gives a peak; peeling off a singleton
                   does not. Scale-free in the number of designs, so it is
                   comparable between targets.
    d_cut_height   how far apart the two separated groups are. A distant outlier
                   can make this large while regrouping almost nothing, so it is
                   read together with d_eff_n, not instead of it.

Requires: numpy, pandas, scipy, biopython, matplotlib.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#4a3aa7", "#7a7a76"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#dcdcd8", "#fcfcfb"


def parse_args():
    p = argparse.ArgumentParser(
        prog="bindercluster.py",
        description="Cluster binder designs by their footprint on a shared target "
                    "(no superposition needed).",
        epilog="Run with --pdb-dir once, then reuse mindist.csv via --matrix "
               "for all further parameter changes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument("input", nargs="?",
                   help="input: a directory of complex PDB files, or a "
                        "mindist.csv from a previous run")
    p.add_argument("output", nargs="?", help="output directory")
    p.add_argument("--pdb-dir", help="same as a directory given positionally")
    p.add_argument("--matrix", help="same as a .csv given positionally")
    p.add_argument("--out", help="same as the second positional argument")
    p.add_argument("--title", help="figure title, usually the target name")

    g = p.add_argument_group("chains and contacts")
    g.add_argument("--target-chain", default="A")
    g.add_argument("--binder-chain", default="B")
    g.add_argument("--level", choices=["atom", "residue"], default="atom",
                   help="one column per target heavy atom, or one per target residue")
    g.add_argument("--cutoff", type=float, default=4.5,
                   help="distance at which the contact score is still 1 (A)")
    g.add_argument("--soft-width", type=float, default=1.5,
                   help="width of the ramp from score 1 to score 0 (A)")
    g.add_argument("--binary", action="store_true",
                   help="hard 0/1 threshold instead of the ramp")
    g.add_argument("--min-freq", type=float, default=0.0,
                   help="drop residues whose mean score is below this")

    g = p.add_argument_group("spatial smoothing")
    g.add_argument("--smooth-radius", type=float, default=3.5,
                   help="gaussian sigma, or hard radius under --smooth-kernel box (A); "
                        "0 disables")
    g.add_argument("--smooth-kernel", choices=["box", "gaussian"], default="gaussian",
                   help="box weights all neighbours equally inside the radius; "
                        "gaussian decays as exp(-d^2 / 2 sigma^2), truncated at 3 sigma")
    g.add_argument("--alpha", type=float, default=0.5,
                   help="weight a residue keeps for itself; 1 disables smoothing")
    g.add_argument("--smooth-per-complex", action="store_true",
                   help="build the neighbourhood graph from each complex's own target "
                        "conformation instead of one shared reference")
    g.add_argument("--ref", default="medoid", choices=["medoid", "first"],
                   help="which structure supplies the shared neighbourhood graph")

    g = p.add_argument_group("centring and clustering")
    g.add_argument("--z", nargs="?", const="robust", default="none",
                   choices=["none", "standard", "robust"],
                   help="centre each residue column to suppress consensus positions; "
                        "bare --z means robust")
    g.add_argument("--z-sigma", "--sigma", "--tau", dest="z_sigma", type=float,
                   default=0.10,
                   help="stabiliser in robust centring: divide by sqrt(sd^2 + z_sigma^2)")
    g.add_argument("--metric", default="cosine",
                   choices=["cosine", "euclidean", "jaccard"])
    g.add_argument("--linkage", default="average",
                   choices=["average", "complete", "ward"])
    g.add_argument("--k-range", type=int, nargs=2, metavar=("KMIN", "KMAX"),
                   default=[2, 15], help="k values to sweep; produces the figures")
    g.add_argument("--k", type=int,
                   help="emit this single partition only, no sweep and no figures")
    g.add_argument("--mark-k", type=int,
                   help="highlight this one k in both figures, after you have chosen it")

    a = p.parse_args()
    if a.input:
        if a.input.lower().endswith(".csv"):
            a.matrix = a.matrix or a.input
        else:
            a.pdb_dir = a.pdb_dir or a.input
    if a.output:
        a.out = a.out or a.output
    if bool(a.pdb_dir) == bool(a.matrix):
        p.error("give exactly one input: a PDB directory or a mindist.csv")
    if not a.out:
        p.error("give an output directory (second positional argument or --out)")
    return a


# --- 1. PDB complexes -> minimum heavy-atom distance per target residue ---

def mindist_from_pdbs(pdb_dir, target_chain, binder_chain, level="atom"):
    """Minimum distance from every target site to the binder, per structure.

    A site is one target heavy atom (level="atom") or one target residue
    (level="residue", taking the closest of its heavy atoms).
    Returns (distance matrix, reference coordinates, skipped, per-structure coords).
    """
    from Bio.PDB import PDBParser
    from Bio.PDB.Atom import DisorderedAtom
    from scipy.spatial import cKDTree

    def chosen(atom):
        return atom.selected_child if isinstance(atom, DisorderedAtom) else atom

    def heavy(res):
        return [chosen(a) for a in res
                if str(a.element).strip().upper() != "H"
                and not a.get_name().strip().startswith("H")]

    files = sorted(Path(pdb_dir).glob("*.pdb"))
    if not files:
        sys.exit(f"no .pdb files in {pdb_dir}")

    parser = PDBParser(QUIET=True)
    rows, skipped, all_coords, ref_coords, ref_name = {}, [], {}, None, None

    for i, f in enumerate(files, 1):
        model = next(parser.get_structure(f.stem, str(f)).get_models())
        if target_chain not in model or binder_chain not in model:
            skipped.append({"pdb_file": f.name, "reason": "missing chain"})
            continue
        target_res = [r for r in model[target_chain] if r.id[0] == " "]
        binder_atoms = [a for r in model[binder_chain] if r.id[0] == " " for a in heavy(r)]
        if not binder_atoms:
            skipped.append({"pdb_file": f.name, "reason": "binder has no heavy atoms"})
            continue

        tree = cKDTree(np.array([a.coord for a in binder_atoms]))
        dists, coords = {}, {}
        for r in target_res:
            atoms = heavy(r)
            if not atoms:
                continue
            stem = f"{r.id[1]}{r.id[2].strip()}:{r.resname}"
            d = tree.query(np.array([a.coord for a in atoms]), k=1)[0]
            if level == "atom":
                for atom, di in zip(atoms, d):
                    key = f"{stem}:{atom.get_name()}"
                    dists[key] = float(di)
                    coords[key] = atom.coord
            else:
                dists[stem] = float(d.min())
                coords[stem] = (r["CA"].coord if "CA" in r
                                else np.array([a.coord for a in atoms]).mean(axis=0))
        rows[f.name] = dists
        all_coords[f.name] = coords
        if ref_coords is None:
            ref_coords, ref_name = coords, f.name
        if i % 25 == 0 or i == len(files):
            print(f"  parsed {i}/{len(files)}")

    if not rows:
        sys.exit("no usable structure")

    M = pd.DataFrame(rows).T.sort_index()
    print(f"used {len(M)} structures, {M.shape[1]} target {level}s "
          f"(coords from {ref_name})")
    if skipped:
        print(f"skipped {len(skipped)}: "
              f"{pd.DataFrame(skipped)['reason'].value_counts().to_dict()}")

    C = pd.DataFrame(ref_coords, index=["x", "y", "z"]).T.reindex(M.columns)
    long = [{"pdb_file": f, "site_key": k, "x": v[0], "y": v[1], "z": v[2]}
            for f, d in all_coords.items() for k, v in d.items()]
    return M, C, pd.DataFrame(skipped), pd.DataFrame(long)


# --- 2. Quality control and find the mediod reference---

def kabsch_rmsd(P, Q):
    """RMSD after optimal superposition. Used for QC only."""
    Pc, Qc = P - P.mean(axis=0), Q - Q.mean(axis=0)
    V, _, Wt = np.linalg.svd(Pc.T @ Qc)
    d = np.sign(np.linalg.det(V @ Wt))
    R = V @ np.diag([1.0, 1.0, d]) @ Wt
    return float(np.sqrt((((Pc @ R) - Qc) ** 2).sum() / len(P)))


def target_conformation_qc(coords_long, site_keys, out_csv=None):
    """Backbone spread of the target across structures; picks the medoid conformation.
    Uses CA atoms only when the sites are atoms, so side-chain rotamers do not
    inflate the RMSD."""
    ca = [k for k in site_keys if str(k).endswith(":CA")]
    if ca:
        site_keys = ca
        coords_long = coords_long[coords_long["site_key"].isin(set(ca))]
    g = coords_long.set_index(["pdb_file", "site_key"]).sort_index()
    names = coords_long["pdb_file"].unique().tolist()
    X = np.stack([g.loc[n].reindex(site_keys)[["x", "y", "z"]].to_numpy(float) for n in names])
    usable = ~np.isnan(X).any(axis=(1, 2))
    names = [n for n, ok in zip(names, usable) if ok]
    X = X[usable]

    n = len(names)
    R = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            R[i, j] = R[j, i] = kabsch_rmsd(X[i], X[j])
    mean_to_others = R.sum(axis=1) / max(n - 1, 1)
    medoid = int(np.argmin(mean_to_others))

    vals = R[np.triu_indices(n, 1)] if n > 1 else np.array([0.0])
    print(f"  target conformation spread: median RMSD {np.median(vals):.2f} A, "
          f"q95 {np.quantile(vals, 0.95):.2f} A, max {vals.max():.2f} A")
    print(f"  medoid = {names[medoid]} (mean RMSD to others {mean_to_others[medoid]:.2f} A)")
    if np.quantile(vals, 0.95) > 2.0:
        print("  [warn] the target is not in one conformation across structures; "
              "a single shared neighbourhood graph may not be appropriate")
    if out_csv is not None:
        pd.DataFrame({"pdb_file": names, "mean_rmsd_to_others": mean_to_others,
                      "is_medoid": [i == medoid for i in range(n)]}).to_csv(out_csv, index=False)
    return names[medoid]


# --- 3. scoring, smoothing, centring ---

def neighbour_report(coords, radius, kernel="box"):
    """Split the neighbourhood into sequence neighbours and genuine spatial ones.
    Under a gaussian kernel the effective reach is taken as 2 sigma."""
    import re
    from scipy.spatial.distance import cdist
    reach = 2.0 * radius if kernel == "gaussian" else radius
    X = coords[["x", "y", "z"]].to_numpy(float)
    resid = np.array([int(re.sub(r"\D", "", str(k).split(":")[0])) for k in coords.index])
    near = cdist(X, X) <= reach
    seq = (near & (np.abs(resid[:, None] - resid[None, :]) <= 2)).sum(axis=1).mean()
    total = near.sum(axis=1).mean()
    print(f"    neighbourhood within {reach:.1f}A: {total:.1f} sites = "
          f"{seq:.1f} sequence + {total - seq:.1f} spatial")
    if total - seq < 1.0:
        print("    [warn] fewer than one spatial neighbour; this radius only averages "
              "along the backbone")


def _graph(X, radius, kernel="box"):
    """Row-normalised neighbourhood weights; each row sums to 1."""
    from scipy.spatial.distance import cdist
    D = cdist(X, X)
    if kernel == "gaussian":
        A = np.exp(-(D ** 2) / (2.0 * radius ** 2))
        A[D > 3.0 * radius] = 0.0
    else:
        A = (D <= radius).astype(float)
    np.fill_diagonal(A, 1.0)
    return A / A.sum(axis=1, keepdims=True)


def smooth(H, coords, radius, alpha, kernel="box"):
    """Smooth every row against one shared neighbourhood graph. Recommended."""
    X = coords[["x", "y", "z"]].to_numpy(float)
    A = _graph(X, radius, kernel)
    effective = 1.0 / (A ** 2).sum(axis=1)
    name = "sigma" if kernel == "gaussian" else "r"
    print(f"  smoothing {kernel} {name}={radius}A (shared graph): "
          f"{effective.mean():.1f} effective neighbours, "
          f"self weight {np.median(np.diag(A)):.2f}")
    return alpha * H + (1 - alpha) * (H @ A.T)


def smooth_per_complex(H, names, site_keys, coords_long, radius, alpha,
                       ref_coords=None, kernel="box"):
    """Smooth each row against the neighbourhood graph of its own complex."""
    g = coords_long.set_index(["pdb_file", "site_key"]).sort_index()
    out = np.empty_like(H)
    edge_diff = []
    A_ref = (_graph(ref_coords[["x", "y", "z"]].to_numpy(float), radius, kernel) > 0
             if ref_coords is not None else None)
    for i, name in enumerate(names):
        X = g.loc[name].reindex(site_keys)[["x", "y", "z"]].to_numpy(float)
        A = _graph(X, radius, kernel)
        out[i] = alpha * H[i] + (1 - alpha) * (A @ H[i])
        if A_ref is not None:
            edge_diff.append(((A > 0) != A_ref).sum() / A.size)
    if edge_diff:
        print(f"  smoothing {kernel} {radius}A (per complex): graphs differ from the "
              f"reference by {np.median(edge_diff):.2%} of edges (max {np.max(edge_diff):.2%})")
    return out


def standardize(H, how, sigma):
    """Centre each residue column, optionally with a variance stabiliser."""
    if how == "none":
        return H
    mean = H.mean(axis=0, keepdims=True)
    sd = H.std(axis=0, keepdims=True)
    if how == "standard":
        return (H - mean) / np.where(sd == 0, 1.0, sd)
    return (H - mean) / np.sqrt(sd ** 2 + sigma ** 2)


# --- 4. cluster quality ---

def silhouette_from_D(Dsq, labels):
    """Silhouette from a precomputed distance matrix."""
    labels = np.asarray(labels)
    unique = np.unique(labels)
    if len(unique) < 2:
        return np.zeros(len(labels))
    s = np.zeros(len(labels))
    for i in range(len(labels)):
        own = labels == labels[i]
        if own.sum() <= 1:
            continue
        a = Dsq[i, own].sum() / (own.sum() - 1)
        b = min(Dsq[i, labels == c].mean() for c in unique if c != labels[i])
        s[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
    return s


def eff_n_clusters(labels):
    """exp(Shannon entropy of the cluster size distribution)."""
    p = np.array(list(pd.Series(labels).value_counts(normalize=True)))
    return float(np.exp(-(p * np.log(p)).sum()))


# --- 5. figures ---

def clustree(labels_by_k, ks, leaf_pos, out, n, height=None, mark_k=None):
    """Cluster tree across k. Nodes sit at the mean leaf position of their members,
    which keeps the edges from crossing."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pos, size = {}, {}
    for k in ks:
        for c in np.unique(labels_by_k[k]):
            members = labels_by_k[k] == c
            pos[(k, c)] = float(leaf_pos[members].mean())
            size[(k, c)] = int(members.sum())

    lineage = {(ks[0], int(c)): int(c) for c in np.unique(labels_by_k[ks[0]])}
    for k0, k1 in zip(ks, ks[1:]):
        for c1 in np.unique(labels_by_k[k1]):
            members = labels_by_k[k1] == c1
            parent = pd.Series(labels_by_k[k0][members]).mode().iloc[0]
            lineage[(k1, c1)] = lineage[(k0, parent)]
    colour_of = {r: i for i, r in enumerate(sorted(set(lineage.values())))}

    width = float(np.clip(0.35 * n, 9.0, 18.0))
    fig, ax = plt.subplots(figsize=(width, 0.80 * len(ks) + 2.0))
    ax.set_facecolor(SURFACE)
    fig.patch.set_facecolor(SURFACE)

    big_steps = []
    for k0, k1 in zip(ks, ks[1:]):
        ct = pd.crosstab(labels_by_k[k0], labels_by_k[k1])
        for c0 in ct.index:
            children = [c1 for c1 in ct.columns if ct.loc[c0, c1] > 0]
            major = [c1 for c1 in children if ct.loc[c0, c1] >= max(2, 0.08 * n)]
            if len(children) > 1 and len(major) > 1 and size[(k0, c0)] >= max(3, 0.15 * n):
                big_steps.append((k0, k1, int(c0), size[(k0, c0)],
                                  [int(ct.loc[c0, c1]) for c1 in children]))
            for c1 in children:
                ax.plot([pos[(k0, c0)], pos[(k1, c1)]], [k0, k1], color=GRID,
                        lw=0.8 + 4.5 * ct.loc[c0, c1] / n, alpha=0.85,
                        zorder=1, solid_capstyle="round")

    for (k, c), x in pos.items():
        area = 110 + 1000 * size[(k, c)] / n
        ax.scatter(x, k, s=area, color=SERIES[colour_of[lineage[(k, c)]] % len(SERIES)],
                   edgecolor=SURFACE, lw=2, zorder=3)
        label = str(size[(k, c)])
        if area >= 150 + 55 * len(label):
            ax.text(x, k, label, ha="center", va="center", fontsize=7.5,
                    color="white", zorder=4, fontweight="bold")
        else:
            ax.text(x, k - 0.085, label, ha="center", va="bottom", fontsize=7,
                    color=INK2, zorder=4)

    if height:
        xlim = ax.get_xlim()
        for k0, k1 in zip(ks, ks[1:]):
            if k0 in height and k1 in height:
                ax.text(xlim[0] - 0.012 * (xlim[1] - xlim[0]), (k0 + k1) / 2,
                        f"Δh {height[k0] - height[k1]:.3f}",
                        ha="right", va="center", fontsize=7.5, color=INK2)
        ax.set_xlim(xlim[0] - 0.16 * (xlim[1] - xlim[0]), xlim[1])

    if mark_k in ks:
        ax.axhspan(mark_k - 0.26, mark_k + 0.26, color=SERIES[3], alpha=0.16, zorder=0)
    ax.set_yticks(ks)
    ax.set_yticklabels([f"k={k}  *" if k == mark_k else f"k={k}" for k in ks],
                       fontsize=9, color=INK2)
    ax.invert_yaxis()
    ax.set_xticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    ax.set_title("Cluster tree across k\n"
                 "node = cluster (area ~ size, number = members), "
                 "edge width ~ members moved, left margin = cut-height gap",
                 color=INK, fontsize=10, loc="left")
    fig.tight_layout()
    path = out / "clustree.png"
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return path, big_steps


def diagnostics_plot(ks, d_eff, d_height, out, mark_k=None, title=None):
    """One panel, two y axes: how much is regrouped, and how distinct the split is."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax_left = plt.subplots(figsize=(8.6, 5.2))
    fig.patch.set_facecolor(SURFACE)
    ax_left.set_facecolor(SURFACE)
    ax_left.grid(color=GRID, lw=0.6, alpha=0.8)
    ax_left.set_axisbelow(True)
    ax_right = ax_left.twinx()
    for ax in (ax_left, ax_right):
        for spine in ax.spines.values():
            spine.set_color(GRID)
    if mark_k is not None:
        ax_left.axvline(mark_k, color=SERIES[3], lw=10, alpha=0.20, zorder=0)

    kx = sorted(d_eff)
    line_left, = ax_left.plot(kx, [d_eff[k] for k in kx], lw=2.4, color=SERIES[0],
                              marker="o", ms=8, mec=SURFACE, mew=2,
                              label="gain in effective clusters")
    kh = sorted(d_height)
    line_right, = ax_right.plot(kh, [d_height[k] for k in kh], lw=2.4, color=SERIES[1],
                                marker="s", ms=7, mec=SURFACE, mew=2,
                                label="gain in cut height")

    ax_left.set_ylabel("$\\Delta$ effective clusters  (how much is regrouped)",
                       color=SERIES[0], fontsize=10)
    ax_right.set_ylabel("$\\Delta$ cut height  (how distinct the split is)",
                        color=SERIES[1], fontsize=10)
    ax_left.tick_params(axis="y", colors=SERIES[0], labelsize=9)
    ax_right.tick_params(axis="y", colors=SERIES[1], labelsize=9)
    ax_left.tick_params(axis="x", colors=INK2, labelsize=9)
    ax_left.set_xlabel("number of clusters (k)", color=INK2, fontsize=10)
    ax_left.set_xticks(ks)
    ax_left.set_ylim(0, max(d_eff.values()) * 1.18)
    ax_right.set_ylim(0, max(d_height.values()) * 1.18 if d_height else 1)
    ax_left.legend(handles=[line_left, line_right], frameon=False, fontsize=9,
                   loc="upper right", labelcolor=[SERIES[0], SERIES[1]])
    ax_left.set_title(title or "Where the designs split", color=INK, fontsize=12,
                      loc="left", pad=12)
    fig.tight_layout()
    path = out / "diagnostics.png"
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return path


# --- 6. clustering ---

def cluster(H, metric, linkage_method, k):
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist
    D = (pdist((H > 0).astype(bool), metric="jaccard") if metric == "jaccard"
         else pdist(H, metric=metric))
    D = np.nan_to_num(D, nan=1.0)
    Z = linkage(D, method=linkage_method)
    return fcluster(Z, k, "maxclust"), Z, D


def sweep_report(H, names, site_keys, metric, linkage_method, kmin, kmax, out,
                 mark_k=None, title=None):
    """Cluster over a range of k, write the figures and the tables."""
    from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
    from scipy.spatial.distance import pdist, squareform

    D = np.nan_to_num(pdist((H > 0).astype(bool), metric="jaccard")
                      if metric == "jaccard" else pdist(H, metric=metric), nan=1.0)
    Dsq = squareform(D)
    Z = linkage(D, method=linkage_method)
    n = H.shape[0]

    ks = list(range(kmin, min(kmax, n - 1) + 1))
    labels_by_k = {k: fcluster(Z, k, "maxclust") for k in ks}
    silhouette = {k: silhouette_from_D(Dsq, labels_by_k[k]).mean() for k in ks}
    effective = {k: eff_n_clusters(labels_by_k[k]) for k in ks}
    height = {k: float(Z[-(k - 1), 2]) for k in ks if k >= 2}

    # increments are recorded on the k that was reached
    d_eff = {k: effective[k] - effective[k - 1] for k in ks[1:]}
    d_height = {k: height[k - 1] - height[k] for k in ks[1:] if k - 1 in height}

    split_desc = {}
    for k in ks[:-1]:
        ct = pd.crosstab(labels_by_k[k], labels_by_k[k + 1])
        n_children = (ct > 0).sum(axis=1)
        split_desc[k] = ";".join(
            f"c{c}(n={int((labels_by_k[k] == c).sum())})->{int(n_children[c])}"
            for c in n_children.index if n_children[c] > 1) or "-"

    diag_png = diagnostics_plot(ks, d_eff, d_height, out, mark_k, title)
    leaf_pos = np.empty(n)
    leaf_pos[dendrogram(Z, no_plot=True)["leaves"]] = np.arange(n)
    tree_png, big_steps = clustree(labels_by_k, ks, leaf_pos, out, n, height, mark_k)

    table = pd.DataFrame({
        "k": ks,
        "eff_n": [round(effective[k], 3) for k in ks],
        "d_eff_n": [round(d_eff[k], 3) if k in d_eff else np.nan for k in ks],
        "cut_height": [round(height.get(k, np.nan), 4) for k in ks],
        "d_cut_height": [round(d_height[k], 4) if k in d_height else np.nan for k in ks],
        "silhouette": [round(silhouette[k], 4) for k in ks],
        "n_singletons": [int((pd.Series(labels_by_k[k]).value_counts() == 1).sum())
                         for k in ks],
        "largest_cluster": [int(pd.Series(labels_by_k[k]).value_counts().iloc[0])
                            for k in ks],
        "what_split_next": [split_desc.get(k, "") for k in ks],
    })
    table.to_csv(out / "k_metrics.csv", index=False)

    assignments = pd.DataFrame({"pdb_file": names})
    for k in ks:
        assignments[f"k{k}"] = labels_by_k[k]
    assignments.to_csv(out / "cluster_assignments.csv", index=False)

    profiles = []
    for k in ks:
        for c in np.unique(labels_by_k[k]):
            members = labels_by_k[k] == c
            profile = H[members].mean(axis=0)
            top = np.argsort(-profile)[:12]
            profiles.append({
                "k": k, "cluster": int(c), "n": int(members.sum()),
                "mean_score": round(float(profile.mean()), 4),
                "top_sites": ";".join(f"{site_keys[i]}={profile[i]:.2f}" for i in top),
            })
    pd.DataFrame(profiles).to_csv(out / "cluster_profiles.csv", index=False)

    print(table.to_string(index=False))
    print(f"\nfigures: {diag_png.name}, {tree_png.name}")
    print("tables:  k_metrics.csv, cluster_assignments.csv, cluster_profiles.csv")
    if big_steps:
        print("\nsteps where a large cluster splits:")
        for k0, k1, c0, sz, children in big_steps:
            print(f"  k={k0}->{k1}: cluster {c0} (n={sz}) -> {children}")
    return table


def main():
    a = parse_args()
    a.soft = not a.binary
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    if a.pdb_dir:
        M, C, skipped, coords_long = mindist_from_pdbs(a.pdb_dir, a.target_chain,
                                                       a.binder_chain, a.level)
        M.rename_axis("pdb_file").reset_index().to_csv(out / "mindist.csv",
                                                       index=False)
        C.rename_axis("site_key").reset_index().to_csv(out / "coords.csv",
                                                      index=False)
        coords_long.to_csv(out / "coords_all.csv", index=False)
        if len(skipped):
            skipped.to_csv(out / "skipped.csv", index=False)
    else:
        M = pd.read_csv(a.matrix).set_index("pdb_file")
        ref_path = Path(a.matrix).with_name("coords.csv")
        C = (pd.read_csv(ref_path).set_index("site_key").reindex(M.columns)
             if ref_path.exists() else None)
        long_path = Path(a.matrix).with_name("coords_all.csv")
        coords_long = pd.read_csv(long_path) if long_path.exists() else None

    if a.soft:
        scores = (1.0 - (M - a.cutoff) / a.soft_width).clip(0.0, 1.0)
        print(f"contact score: 1 at d<={a.cutoff}A, 0 at d>={a.cutoff + a.soft_width}A")
    else:
        scores = (M <= a.cutoff).astype(float)

    freq = scores.mean(axis=0)
    if a.min_freq > 0:
        keep = freq[freq >= a.min_freq].index
        print(f"keeping {len(keep)}/{len(freq)} residues at --min-freq {a.min_freq}")
        scores, freq = scores[keep], freq[keep]
        if C is not None:
            C = C.reindex(keep)
        if coords_long is not None:
            coords_long = coords_long[coords_long["site_key"].isin(set(keep))]
    pd.DataFrame({"site_key": freq.index, "mean_score": freq.values}) \
        .sort_values("mean_score", ascending=False) \
        .to_csv(out / f"hit_frequency_cutoff{a.cutoff}.csv", index=False)

    H = scores.to_numpy(float)

    medoid = None
    if coords_long is not None:
        medoid = target_conformation_qc(coords_long, list(scores.columns),
                                        out_csv=out / "target_conformation_qc.csv")
        if a.ref == "medoid":
            Cm = (coords_long[coords_long["pdb_file"] == medoid]
                  .set_index("site_key").reindex(scores.columns)[["x", "y", "z"]])
            if not Cm.isna().any().any():
                C = Cm

    if a.smooth_radius > 0:
        if a.ref == "medoid" and medoid is not None:
            print(f"  shared neighbourhood graph from {medoid} (medoid)")
        if C is not None:
            neighbour_report(C, a.smooth_radius, a.smooth_kernel)
        if a.smooth_per_complex:
            if coords_long is None:
                sys.exit("--smooth-per-complex needs coords_all.csv")
            H = smooth_per_complex(H, scores.index.to_list(), list(scores.columns),
                                   coords_long, a.smooth_radius, a.alpha, C,
                                   a.smooth_kernel)
        else:
            if C is None:
                sys.exit("--smooth-radius needs coords.csv")
            H = smooth(H, C, a.smooth_radius, a.alpha, a.smooth_kernel)

    if a.z != "none":
        print(f"centring: {a.z}" + (f" (z_sigma={a.z_sigma})" if a.z == "robust" else "")
              + "; comparison is now of deviations from the per-residue mean")
    H = standardize(H, a.z, a.z_sigma)

    names = scores.index.to_list()
    site_keys = list(scores.columns)
    pd.DataFrame(H, index=names, columns=site_keys).rename_axis("pdb_file") \
        .reset_index().to_csv(out / "score.csv", index=False)

    if a.k:
        labels, _, _ = cluster(H, a.metric, a.linkage, a.k)
        pd.DataFrame({"pdb_file": names, "cluster": labels}).to_csv(
            out / f"clusters_k{a.k}.csv", index=False)
        print(f"k={a.k} cluster sizes: "
              f"{pd.Series(labels).value_counts().sort_index().to_dict()}")
    else:
        sweep_report(H, names, site_keys, a.metric, a.linkage,
                     a.k_range[0], a.k_range[1], out, a.mark_k, a.title)

    print("\noutput directory:", out)


if __name__ == "__main__":
    main()
