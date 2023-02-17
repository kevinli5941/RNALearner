import pandas as pd
import h5py
import scanpy as sc
from scipy import sparse

data0 = pd.read_hdf('recount3/mouse_0_1000_transpose.h5').T
data1 = pd.read_hdf('recount3/mouse_1001_5000_transpose.h5').T
data2 = pd.read_hdf('recount3/mouse_5001_8001_transpose.h5').T
data3 = pd.read_hdf('recount3/mouse_8002_10087_transpose.h5').T
data = pd.concat([data0, data1, data2, data3])

print("done concat")
with open("reports/conversion_updates.txt", "a") as file_object:
            file_object.write("done concat")
        
adata = sc.AnnData(data, data.index.to_frame(), data.columns.to_frame())

print("anndata created")
with open("reports/conversion_updates.txt", "a") as file_object:
            file_object.write("AnnData created")

adata.obs.columns = ['Sample ID']
adata.var.columns = ['Gene ID']

adata.X = sparse.csr_matrix(adata.X)

print("sparsified")
with open("reports/conversion_updates.txt", "a") as file_object:
            file_object.write("AnnData Sparsified")
        
adata.write_h5ad("recount3/mouse_FULL_sparse.h5ad")


