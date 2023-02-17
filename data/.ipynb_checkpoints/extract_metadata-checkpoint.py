import pandas as pd
import numpy as np
import warnings
import os
warnings.simplefilter(action='ignore', category=FutureWarning)

# Retrieve links for all project metadata links
metadata = pd.read_csv("metadata.csv")

col = 'recount_pred'
selected_col = projects[col]

start = 0
end = len(selected_col) - 1

link = selected_col[start]
all_metadata = pd.read_csv(link, skiprows = 0, sep="\t", header=0)
for i in range(start, end + 1):
    link = selected_col[i]
    temp = pd.read_csv(link, skiprows = 0, sep="\t", header=0)
    all_metadata = pd.concat([all_metadata, temp])
    
