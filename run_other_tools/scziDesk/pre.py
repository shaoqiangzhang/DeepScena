import numpy as np
import pandas as pd
import scanpy as sc  
adata=pd.read_csv("bhatdrop90.txt",header=0,index_col=0,sep='\s+')
print(adata.shape)

data=sc.AnnData(adata)
sc.pp.filter_genes(data, min_cells=3)   
sc.pp.normalize_total(data, target_sum=1e4) 
sc.pp.log1p(data) 
sc.pp.highly_variable_genes(data,n_top_genes=784) 
data = data[:, data.var.highly_variable]  
data.to_df().to_csv("bhat90784.csv")  