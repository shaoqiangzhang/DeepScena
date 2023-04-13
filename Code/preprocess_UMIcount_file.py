##preprocess UMI-counting data before clustering
import numpy as np
import pandas as pd
import scanpy as sc  
#adata=pd.read_csv("/data/bhat.csv",header=0,index_col=0)
adata=pd.read_csv("/data/pbmc4340.txt",header=0,index_col=0,sep='\t')
print(adata.shape)
adata=adata.T # transpose adata if it is gene*cell
data=sc.AnnData(adata,dtype=np.float32)
sc.pp.filter_genes(data, min_cells=3)   
sc.pp.normalize_total(data, target_sum=1e4) 
sc.pp.log1p(data) 
 
 
sc.pp.highly_variable_genes(data, n_top_genes=784)#select top highly variable genes
 
data = data[:, data.var.highly_variable]  
data.to_df().to_csv("data/pbmcpre.csv")