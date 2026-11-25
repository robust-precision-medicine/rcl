# rcl

Robust contrastive learning (RCL): Given a foreground group of samples (G1; e.g., disease cases) and a background group of samples (G0; e.g., healthy controls), rcl learns components that capture variation enriched in the foreground group compared to the background group, while accounting for sample-level heterogeneity by learning a solution that is robust across small subsets of samples.

Version 1.0.0

The following environment was tested:
```
conda create -n rcl python=3.9
conda activate rcl
pip install scanpy
conda install -c conda-forge qpsolvers
conda install dask -c conda-forge
```

Example:
```
import sys
sys.path.insert(1, "path_to_rcl") # the folder with rcl.py and logger.py
from rcl import rcl

# adata is an anndata; adata.X has to be a csr_matrix (e.g., log-normalized expression)
G1 = [...] # the foreground samples (values of the sample column in adata.obs)
G0 = [...] # the background samples
mdl = rcl(adata,
  sample_key = "sample_id", # the name of the column in adata.obs that maps to the sample identifier
  G0 = G0,
  G1 = G1,
  subset_size = 2, # the number of samples from each group in each subset
  logfile = "rcl.log")
mdl.train(n_components = 2, rand_seed = 1)
```
After training, `adata.varm["RCL"]` includes the components (a features by components matrix) and `adata.obsm["RCL"]` includes the projection of the data onto these components (an observations by components matrix, calculated as the product of the data matrix and the components matrix).

For example, when applied to single-cell RNA-seq data, the features are genes and the observations are cells: each component is a vector of gene loadings, and each cell receives one score per component, the weighted sum of its expression values with the loadings as weights. Observations from another dataset with the same features can be projected onto the components in the same way, with `np.dot(X_other, adata.varm["RCL"])`.
