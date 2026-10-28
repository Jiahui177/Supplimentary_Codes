# Clustering by target–binder footprint -Section 2.2.3
Groups de novo binder designs by where they contact the target, not by binder sequence or prediction-confidence score. Contact distances are measured inside each predicted complex, so they are invariant to how the complex is placed in space, and the feature columns are shared target sites.
  - bindercluster.py: reads the predicted complexes, builds the footprint matrix, clusters it, and writes all tables and the diagnostic figures.
  - binderplot.py: a visualization companion compatible with bindercluster.py; it draws the footprint heatmap with the clustering dendrogram attached, from the output of bindercluster.py.

## Requirements:
Python 3.8+ with:
```
pip install numpy pandas scipy biopython matplotlib
```

## Usage:
### 1. Input:
You need to prepare a directory of complex PDB files, one per design, each containing the target chain (default A) and the binder chain (default B). The protein sequence of the target must be identical across all files.

### 2. Scoring, clustering and generating diagnostic diagrams:
Use bindercluster.py to conduct the target–binder contact footprint analysis.

Default:
```
python bindercluster.py /path to complex_pdbs/ /path to output/ --title "title"
```
The default analysis follows the thesis settings:
  - atom-level contacts,
  - c = 4.5 Å, ω = 1.5 Å,
  - Gaussian smoothing with σ = 3.5 Å and α = 0.5,
  - cosine distance with average linkage.

#### Advanced settings:

Run
```
python bindercluster.py --help
```
to obtain the full list. Every option has a default, so you only set the ones you want to change. In the lists below, the shape after each option tells you what it expects:

  - `<...>` input a **value** - the placeholder says what kind (a number, a chain ID, etc.).
  - `{a,b}` takes a **choice** - you pass exactly one of the listed words.
  - an option shown **on its own is a switch** - writing it turns the behaviour on; leaving it out keeps it off.

**Chains and contacts**
  - `--target-chain <ID>` / `--binder-chain <ID>`  -> The chain IDs to use. Default A / B.
  - `--level {atom,residue}`  -> Calculate contact scores using atomic/residue precision. Residue is coarser and builds its smoothing graph from Cα positions, so if you use it, it is recommened to raise `--smooth-radius` to about 6 Å.
  - `--cutoff <Å>`  -> Distance given a full contact score, c. Default 4.5.
  - `--soft-width <Å>`  -> Width of the ramp from score 1 down to 0, ω. Default 1.5.
  - `--binary`  -> Switch. Use a hard 0/1 contact at the cutoff instead of the soft ramp. Turning on this switch has the same effect as setting ω = 0. Default off.
  - `--min-freq <0–1>`  -> drop target sites whose mean score across all designs is below this. Default 0.

**Spatial smoothing**
  - `--smooth-radius <Å>`  -> Gaussian σ (or the hard radius under the box kernel). Default 3.5. Set to 0 to turn smoothing off.
  - `--smooth-kernel {box,gaussian}`  -> Gaussian (default) weights nearer atoms more; box weights every neighbour inside the radius equally, as long as the atom (or residue, when you use `--level residue`) is within the smooth radius, it will have the same value regardless of its distance.
  - `--alpha <0–1>`  -> The weight a site keeps for itself during smoothing. Default 0.5; 1 means disables smoothing.
  - `--ref {medoid,first}`  -> Which structure supplies the shared neighbourhood graph: the medoid conformation (default) or the first file. If you chose `--ref first` , the function will use `sorted()` for sorting, which means that the smallest model in ASCII order will be placed at the top. You can achieve this by naming (such as 0_reference) the structure you wish to use as the reference to placing it at the first position and enable the autonomous selection of the template.
  - `--smooth-per-complex`  -> Switch. Build the neighbourhood from each complex's own target geometry instead of one shared reference. Defult off.

**Centring and clustering**
  - `--z`  -> Switch. Centre each target site across designs. Default off.
    - Why use `--z`:  Some target sites are contacted by almost every design. Those columns say little about how designs *differ*, yet they still pull the cosine distances toward zero and can make all designs look alike. `--z` subtracts each site's mean across designs, so the clustering is driven by deviations from the consensus rather than the consensus itself. Use it when a shared hotspot is drowning out the distinctive contacts. However, because centring produces negative values, it changes what cosine distance means, but pairs more naturally with the Euclidean metric.
  - `--z-sigma <number>`  -> Stabiliser used by robust centring, prevent the denominator from approaching zero, dividing by sqrt(sd² + z_sigma²). Default 0.10.
  - `--metric {cosine-average,euclidean-ward}`  -> Distance and linkage as one fixed, valid pair. Default cosine-average (the thesis choice).
  - `--k-range <KMIN> <KMAX>`  -> Two numbers; the range of k to sweep. Default 2 15. This is the mode that produces the figures.
  - `--k <K>`  -> Emit a single partition at this k only, with no sweep and no dignose figures.
  - `--mark-k <K>`  -> Highlight one chosen k in both clustree and dignose figures. You can use this to track the number of clusters you are interested in.

### 3. Output:
Everything is written into the output directory. The two diagnostic figures are:
  - `diagnostics.png`  - For each k, the gain in effective cluster number (how much is regrouped) and the gain in cut height (how distinct the split is). Read these together to find the k values where the designs genuinely reorganise.
  - `clustree.png` - The cluster tree across k; node area ∝ cluster size, edge width ∝ number of designs moved.

The tables are:
  - `mindist.csv` - Designs * sites, minimum distance to the binder (Å). Reuse it as the input via `--matrix` to re-run with different settings without re-parsing the PDBs.
  - `score.csv` - The matrix actually clustered, after scoring, smoothing and any centring.
  - `hit_frequency_*.csv` - Mean contact score per site.
  - `k_metrics.csv` - One row per k: effective number and its increase, cut height and its drop, silhouette, number of singletons, largest cluster, and what splits next.
  - `cluster_assignments.csv` - The cluster id of every design at every k.
  - `cluster_profiles.csv` - Per (k, cluster): size, mean score, and the most contacted sites.
  - `target_conformation_qc.csv` - Each model's mean target-backbone RMSD to the others, with the medoid marked.
  - `coords.csv`, `coords_all.csv` - Reference and per-complex site coordinates (used for smoothing and QC).
  - `skipped.csv` - Designs that could not be used, with the reason.

QC note: a single shared neighbourhood only makes sense if the target is in essentially one conformation across the models. `target_conformation_qc.csv` is the check — the script also prints the spread of pairwise RMSDs and warns if the 95th percentile exceeds 2 Å. If this warning is triggered, it is recommended to manually check whether the target has undergone a conformational change.

### 4. Visualization with binderplot.py:
This is the second step, run after bindercluster.py, because it reads that output directory. It draws the footprint heatmap: rows are designs ordered by the dendrogram so similar designs sit together, columns are target sites, and cell colour is the contact score from `score.csv`. With `--k` it also adds a colour strip marking cluster membership at that k.

If this warning is triggered, it is recommended to manually check whether the target has undergone a conformational change. Every number comes from bindercluster.py. Thus, `--metric` and `--k` must match the bindercluster.py run that produced the folder.

```
python binderplot.py /path to output/ --k <k> --title "title"
```
Here the first argument is the bindercluster.py output directory. <k> means the number of cluster you chose. For example, `--k 4` reads the k = 4 partition (from `cluster_assignments.csv`, or from `clusters_k4.csv` if that run used single-k mode) and draws the cluster strip; omit `--k` to draw the heatmap and tree with no strip. The figure is written as `heatmap_with_tree.png`.

Options (same value / choice / switch convention as above):
  - `--k <K>`  -> draw the cluster strip at this k. Must match a k present in the folder.
  - `--metric {cosine-average,euclidean-ward}`  -> must match the bindercluster.py run. Default cosine-average.
  - `--no-tree`  -> switch. Drop the dendrogram; rows stay in tree order.
  - `--cmap <name>`  -> matplotlib colormap for the heatmap. Default viridis.
  - `--figsize <W> <H>`  -> two numbers, figure size in inches. Default 16 10.
  - `--boundary-color <name>`  -> colour of the lines drawn between clusters; `none` removes them. Default white.
  - `--label-rows`  -> switch. Write each design name on its row; only legible for small sets (≤ 80).
  - `--tick-step <N>`  -> label every Nth column. Default 0, which picks a readable step automatically.
  - `--title <text>`  -> figure title, usually the target name.
