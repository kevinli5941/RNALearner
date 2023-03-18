import pandas as pd
import numpy as np
import warnings
import os
warnings.simplefilter(action='ignore', category=FutureWarning)

# Retrieve links for all projects
projects = pd.read_csv("projects.csv")

organism = 'mouse'
# organism = 'human'

selected_projects = projects[projects['organism'] == organism]
selected_projects = selected_projects.reset_index()
n_proj = selected_projects.shape[0]
print("Total number of projects targeted: ", n_proj)

start = 8002 # start project for aggregation
#end = 8001 # end project for aggregation
end = n_proj - 1 # full run
print("Start: ", start)
print("End: ", end)

full = pd.read_csv(selected_projects['gene'][start], skiprows = 2, sep="\t", header=0)
full = full.set_index('gene_id')
index = full.index
n_samples = full.shape[1]
meta_row = np.ones(n_samples) * start
full = full.append(pd.Series(meta_row, index = full.columns), ignore_index = True)
index = list(index) + ['project']
full.index = index
full = full.astype(pd.SparseDtype("float", 0.0))

with open('reports/most_recent.txt', 'w') as f:
    f.write('Starting!')
    f.write("\nStart: " + str(start))
    f.write("\nEnd: " + str(end))
    f.write("\nDimensions: " + str(full.shape))
for i in range(start + 1, end + 1):
    file_link = selected_projects['gene'][i]
    try:
        counts = pd.read_csv(file_link, skiprows = 2, sep="\t", header=0)
        counts = counts.set_index('gene_id')
        counts = counts.astype(pd.SparseDtype("float", 0.0))
        n_samples = counts.shape[1]
        meta_row = np.ones(n_samples) * i
        counts = counts.append(pd.Series(meta_row, index = counts.columns), ignore_index = True)
        counts.index = index
        full = pd.concat([full, counts], axis = 1)
        full.index = index
    except Exception as e:
        logging.error(traceback.format_exc())
    # Logs the error appropriately. 
    if i % 1000 == 0 or i == 1:
        print("Number of most recent project: ", i)
        with open("reports/most_recent.txt", "a") as file_object:
            file_object.write("\n")
            s = "Number of most recent project: " + str(i + 1)
            file_object.write(s)
            file_object.write("\nDimensions: " + str(full.shape))

print("Dimensions of (appended) aggregate table: ", full.shape)
print("Dimensions of (appended) transpose aggregate table: ", full.T.shape)

#output_path=f"recount3/{organism}_transpose.csv.gz"
#full.T.to_csv(output_path, mode='a', header=not os.path.exists(output_path), compression = 'gzip')

output_path=f"recount3/{organism}_{start}_{end}_transpose.h5"
full.sparse.to_dense().to_hdf(output_path, key = f"mouse{start}to{end}")