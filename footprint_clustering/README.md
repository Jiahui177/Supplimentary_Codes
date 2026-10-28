# Clustering by target–binder footprint -Section 2.2.3
Groups de novo binder designs by where they contact the target, not by binder sequence or prediction-confidence score. Contact distances are measured inside each predicted complex, so they are invariant to how the complex is placed in space, and the feature columns are shared target sites.
  - bindercluster.py: The script to reads the predicted complexes, builds the footprint matrix, clusters it, and writes all tables and the diagnostic figures.
  - binderplot.py: The script used for visualization compatible with bindercluster.py, draws the footprint heatmap with the clustering dendrogram attached, from the output of bindercluster.py.
## Requirements:
Python 3.8+ with:
```
pip install numpy pandas scipy biopython matplotlib
```
## Usage:
### 1. Input:
You need prepare a directory of complex PDB files, one per design, each containing the target chain (default A) and the binder chain (default B). The protein sequences of target must be identical.
### 2. Scoring, clustering and generate diagnostic diagrams:
Use bindercluster.py to conduct the target-binder contact footprint analysis

Defult: 
```
python bindercluster.py /path to complex_pdbs/ /path to output/ --title "title"
```
The defult analysis following the thesis settings:
  - atom-level contacts, 
  - c = 4.5 Å, ω = 1.5 Å, 
  - Gaussian smoothing with σ = 3.5 Å and α = 0.5, 
  - cosine distance with average linkage

Advance setting: 
Run
```
python bindercluster.py --help
```
to obtain the full list.
- Chains and contacts.
  - --target-chain / --binder-chain   set the chain IDs (A / B).
