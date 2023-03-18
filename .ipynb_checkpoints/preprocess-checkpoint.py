import scanpy as sc, numpy as np, pandas as pd, anndata as ad
from scipy import sparse

### [OPTION ONE] ###
# new = sc.read_h5ad('./data/recount3/mouse_FULL_sparse_with_geneIDs_with_metadata.h5ad') # for pre-training
######

### [OPTION TWO] ###
panglao = sc.read_h5ad('./data/recount3/preprocessed_mouse_gene2vec_15117_compatible_bulk_only.h5ad')
data = sc.read_h5ad('./data/osd105/osd105_raw_ensembl_all_genes.h5ad')
counts = sparse.lil_matrix((data.X.shape[0],panglao.X.shape[1]),dtype=np.float32)
ref = panglao.var_names.tolist()
obj = data.var_names.tolist()

with open('reports/preprocessing_progress.txt', 'w') as f:
    f.write('done with loading data')

for i in range(len(ref)):
    if ref[i] in obj:
        loc = obj.index(ref[i])
        counts[:,i] = data.X[:,loc]
    if i % 1000 == 0:
        with open('reports/preprocessing_progress.txt', 'w') as f:
            f.write('\n' + str(i))
            
with open('reports/preprocessing_progress.txt', 'w') as f:
    f.write('\ndone with matching')

counts = counts.tocsr()
new = ad.AnnData(X=counts)
new.var_names = ref
new.obs_names = data.obs_names
new.obs = data.obs
new.uns = panglao.uns

######

with open('reports/preprocessing_progress.txt', 'w') as f:
    f.write('\nbegin filtering cells')
        
sc.pp.filter_cells(new, min_genes=200)

with open('reports/preprocessing_progress.txt', 'a') as f:
    f.write('\nbegin normalizing cells')

sc.pp.normalize_total(new, target_sum=1e4)

with open('reports/preprocessing_progress.txt', 'a') as f:
    f.write('\nbegin log transform cells')

sc.pp.log1p(new, base=2)

with open('reports/preprocessing_progress.txt', 'a') as f:
    f.write('\nbegin writing result')

new.write('./data/osd105/osd105_preprocessed_ensembl_15117.h5ad')