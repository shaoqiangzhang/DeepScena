"""
@Author: Yu Huang
@Email: yuhuang-cst@foxmail.com
"""

import numpy as np
import scipy.sparse as sp
import scanpy as sc
from sklearn.utils import check_array
 
from aide import AIDE,AIDEConfig
#from rph_kmeans import RPHKMeans
from sklearn.cluster import KMeans

from rph import RPHKMeans
def to_adata(features, labels=None):
	adata = sc.AnnData(features)
	if labels is not None:
		adata.obs['Group'] = labels
	return adata


def from_adata(adata):
	features, labels = adata.X, None
	if 'Group' in adata.obs:
		labels = adata.obs['Group'].values
	return features, labels


def preprocess(X, y_true=None, x_dtype=np.float32, y_dtype=np.int32):
	"""
	Args:
		X (np.ndarray or sp.csr_matrix): raw counts data
			dtype = (np.float32 | np.float64 | np.int32 | np.int64)
			shape = (cell_num, gene_num)
		y_true (np.ndarray or None): labels of X
			dtype = (np.int32 | np.int64)
			shape = (cell_num,)
		x_dtype (type): dtype of preprocessed X; should be one of (np.float32, np.float64)
		y_dtype (type): dtype of preprocessed Y; should be one of (np.int32, np.int64)
	Returns:
		np.ndarray or sp.csr_matrix: preprocessed X
			type = type(X)
			dtype = x_dtype
			shape = (filtered_cell_num, filtered_gene_num)
		np.ndarray or None: preprocessed y_true
			dtype = y_dtype
			shape = (filtered_cell_num,)
	"""
	assert x_dtype == np.float32 or x_dtype == np.float64
	assert y_dtype == np.int32 or y_dtype == np.int64
	adata = to_adata(X, y_true)
	sc.pp.filter_cells(adata, min_counts=1)
	sc.pp.filter_genes(adata, min_counts=1)
	sc.pp.normalize_per_cell(adata)
	sc.pp.log1p(adata)
	X, y_true = from_adata(adata)

	X = check_array(X, accept_sparse="csr", order='C', dtype=[x_dtype])
	if sp.issparse(X):
		X.sorted_indices()
	if y_true is not None:
		y_true = check_array(y_true, ensure_2d=False, order='C', dtype=[y_dtype])
	return X, y_true

 

if __name__ == '__main__':
	import h5py
	import os
	import itertools
	from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
	for g in range(0, 10):
		print("Start to process g=", g)
		DEMO_FOLDER = os.path.dirname(os.path.realpath(__file__))
		f = h5py.File("/data/zsq/lty/DCSS/pbmc4k.h5", 'r')   # https://github.com/ttgump/scDeepCluster/tree/master/scRNA-seq%20data
		X, y_true = sp.csr_matrix(f['X']), np.array(f['Y'])
		f.close()
		print(f'Type of X = {type(X)}; Shape of X = {X.shape}; Data type of X = {X.dtype}')
		print(f'Type of y_true = {type(y_true)}; Shape of X = {y_true.shape}; Data type of X = {y_true.dtype}')
    
		X, y_true = preprocess(X, y_true)
		print(f'Preprocess: Type of X = {type(X)}; Shape of X = {X.shape}; Data type of X = {X.dtype}')
		print(f'Preprocess: Type of y_true = {type(y_true)}; Shape of X = {y_true.shape}; Data type of X = {y_true.dtype}')
    
		aide_model = AIDE(name='pbmc4k', save_folder='pbmc4k')
		config = AIDEConfig()   # Run with default config
		embedding = aide_model.fit_transform(X, config)
		print(f'Type of embedding = {type(embedding)}; Shape of embedding = {embedding.shape}; Data type of embedding = {embedding.dtype}')
    
		n_clusters = len(np.unique(y_true))
    		# rph-kmeans
		for n_init in [1, 10]:
			ari_list, nmi_list = [], []
			for i in range(10):
				clt = RPHKMeans(n_clusters=n_clusters, n_init=n_init, verbose=0)
				y_pred = clt.fit_predict(embedding)
				ari_list.append(adjusted_rand_score(y_true, y_pred))
				nmi_list.append(normalized_mutual_info_score(y_true, y_pred))
			print('RPH-KMeans (n_init = {}): ARI = {:.4f} ({:.4f}), NMI = {:.4f} ({:.4f})'.format(
    				n_init, np.mean(ari_list), np.std(ari_list), np.mean(nmi_list), np.std(nmi_list)
    			))
    
    		# kmeans
		for n_init, init_type in itertools.product([1, 10], ['k-means++']):
			ari_list, nmi_list = [], []
			for i in range(10):
				clt = KMeans(n_clusters=n_clusters, init=init_type, n_init=n_init)
				y_pred = clt.fit_predict(embedding)
				ari_list.append(adjusted_rand_score(y_true, y_pred))
				nmi_list.append(normalized_mutual_info_score(y_true, y_pred))
			print('KMeans (init = {}, n_init = {}): ARI = {:.4f} ({:.4f}), NMI = {:.4f} ({:.4f})'.format(
    				init_type, n_init, np.mean(ari_list), np.std(ari_list), np.mean(nmi_list), np.std(nmi_list)
    			))
