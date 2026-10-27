# Clustering by target–binder footprint -Section 2.2.3
Groups de novo binder designs by where they contact the target, not by binder sequence or prediction-confidence score. Contact distances are measured inside each predicted complex, so they are invariant to how the complex is placed in space, and the feature columns are shared target sites.
  - bindercluster.py: The script to reads the predicted complexes, builds the footprint matrix, clusters it, and writes all tables and the diagnostic figures.
  - binderplot.py: The script used for visualization compatible with bindercluster.py, draws the footprint heatmap with the clustering dendrogram attached, from the output of bindercluster.py.
## Requirements:
Python 3.8+ with:
```
pip install numpy pandas scipy biopython matplotlib
```
## Input:
A directory of complex PDB files, one per design, each containing the target chain (default A) and the binder chain (default B). The protein sequences of target must be identical.
