"""
@Author: Yu Huang
@Email: yuhuang-cst@foxmail.com
"""

import json
import numpy as np
import scipy.sparse as sp
import scanpy.api as sc
from sklearn.utils import check_array

from aide import AIDE, AIDEConfig
from rph_kmeans import RPHKMeans
from sklearn.cluster import KMeans


class LargeDataPreprocessor(object):
	"""Handle extremely large data. Maybe read and write to disk multiple times to avoid being out of memory.
	"""
	def __init__(self, output_folder, input_feature_path, input_label_path=None):
		"""
		Args:
			output_folder (str): folder that stores preprocessed data
			input_feature_path (str): file path of raw count feature
				'*.npz': sparse feature; load with 'sp.load_npz' and save with 'sp.save_npz'; (cell_num, gene_num)
				'*.npy': dense feature; load with 'np.load' and save with 'np.save'; (cell_num, gene_num)
			input_label_path (str or None): file path of true labels
				'*.npy': dense labels; load with 'np.load' and save with 'np.save'; (cell_num,)
				None: true labels is not provieded
		"""
		super(LargeDataPreprocessor, self).__init__()
		self.SAVE_FOLDER = output_folder; os.makedirs(output_folder, exist_ok=True)
		self.raw_feature_path, self.raw_label_path = input_feature_path, input_label_path

		self.is_dense = self.raw_feature_path.endswith('.npy')
		self.load_feature = np.load if self.is_dense else sp.load_npz
		self.save_feature = np.save if self.is_dense else sp.save_npz

		postfix = 'npy' if self.is_dense else 'npz'
		self.FEATURE_FILTER_CELL = os.path.join(self.SAVE_FOLDER, f'feature_filter_cell.{postfix}')
		self.LABEL_FILTER_CELL = os.path.join(self.SAVE_FOLDER, f'label_filter_cell.npy')
		self.FEATURE_FILTER_GENE = os.path.join(self.SAVE_FOLDER, f'feature_filter_gene.{postfix}')
		self.FEATURE_NORM = os.path.join(self.SAVE_FOLDER, f'feature_norm.{postfix}')
		self.FEATURE_LOG1P = os.path.join(self.SAVE_FOLDER, f'feature_log1p.{postfix}')
		self.FEATURE_PP_PATH = os.path.join(self.SAVE_FOLDER, f'feature_preprocess.{postfix}')
		self.LABEL_PP_PATH = os.path.join(self.SAVE_FOLDER, 'label_preprocess.npy')

		self.GENE_MASK_PATH = os.path.join(self.SAVE_FOLDER, 'gene_mask.npy')
		self.CELL_MASK_PATH = os.path.join(self.SAVE_FOLDER, 'cell_mask.npy')

		self.HISTORY_JSON = os.path.join(self.SAVE_FOLDER, 'history.json')
		self.history = self.get_history()


	def to_adata(self, features, labels=None):
		adata = sc.AnnData(features)
		if labels is not None:
			adata.obs['Group'] = labels
		return adata


	def from_adata(self, adata):
		features, labels = adata.X, None
		if 'Group' in adata.obs:
			labels = adata.obs['Group'].values
		return features, labels


	def has_been_processed(self):
		return os.path.exists(self.FEATURE_PP_PATH)


	def get_history(self):
		if os.path.exists(self.HISTORY_JSON):
			return json.load(open(self.HISTORY_JSON))
		return dict()


	def save_history(self, history):
		json.dump(history, open(self.HISTORY_JSON, 'w'), indent=2)


	def process_base(self, key, feature_save_path, label_save_path, handle_func):
		if key in self.history:
			return
		features, labels = handle_func()
		if sp.issparse(features):
			features.sort_indices()
		self.save_feature(feature_save_path, features)
		if labels is not None:
			np.save(label_save_path, labels)
		self.history[key] = True
		self.save_history(self.history)


	def cell_filter(self):
		self.process_base('cell_filter', self.FEATURE_FILTER_CELL, self.LABEL_FILTER_CELL, self.cell_filter_handle_func)


	def cell_filter_handle_func(self):
		features = self.load_feature(self.raw_feature_path)
		labels = np.load(self.raw_label_path) if os.path.exists(self.raw_label_path) else None
		cell_mask_ary, _ = sc.pp.filter_cells(features, min_counts=1, inplace=False)
		np.save(self.CELL_MASK_PATH, cell_mask_ary)
		features = features[cell_mask_ary]
		if labels is not None:
			labels = labels[cell_mask_ary]
		return features, labels


	def gene_filter(self):
		self.process_base('gene_filter', self.FEATURE_FILTER_GENE, None, self.gene_filter_handle_func)


	def gene_filter_handle_func(self):
		features = self.load_feature(self.FEATURE_FILTER_CELL)
		gene_mask_ary, _ = sc.pp.filter_genes(features, min_counts=1, inplace=False)
		np.save(self.GENE_MASK_PATH, gene_mask_ary)
		features = features[:, gene_mask_ary]
		return features, None


	def normalize_per_cell(self):
		self.process_base('normalize_per_cell', self.FEATURE_NORM, None, self.normalize_per_cell_handle_func)


	def normalize_per_cell_handle_func(self):
		features = self.load_feature(self.FEATURE_FILTER_GENE)
		adata = self.to_adata(features, None)
		sc.pp.normalize_per_cell(adata)
		return self.from_adata(adata)


	def log1p(self):
		self.process_base('log1p', self.FEATURE_LOG1P, None, self.log1p_handle_func)


	def log1p_handle_func(self):
		features = self.load_feature(self.FEATURE_NORM)
		adata = self.to_adata(features, None)
		sc.pp.log1p(adata)
		return self.from_adata(adata)


	def final(self):
		self.process_base('final', self.FEATURE_PP_PATH, self.LABEL_PP_PATH, self.final_handle_func)


	def final_handle_func(self):
		features = self.load_feature(self.FEATURE_LOG1P)
		features = check_array(features, accept_sparse="csr", order='C', dtype=[np.float32])
		if sp.issparse(features):
			features.sort_indices()

		labels = None
		if os.path.exists(self.LABEL_FILTER_CELL):
			labels = np.load(self.LABEL_FILTER_CELL)
			print(labels.shape, type(labels), labels.dtype)
			labels = check_array(labels, ensure_2d=False, order='C', dtype=[np.int32])
		return features, labels


	def process(self):
		"""
		Returns:
			str: file path of preprocessed features; will be one of ('*.npz', '*.npy')
				'*.npz': saved as sparse matrix if the input feature_path ends with '.npz'
				'*.npy': saved as dense matrix if the input feature_path ends with '.npy'
			str or None: file path of preprocessed labels; will be one of ('*.npy', None)
				'*.npy': saved as dense array if the input label_path is not None
				None: will be
		"""
		if not self.has_been_processed():
			print('cell_filter...')
			self.cell_filter()
			print('cell_filter done')

			print('gene_filter...')
			self.gene_filter()
			print('gene_filter done')

			print('normalize_per_cell...')
			self.normalize_per_cell()
			print('normalize_per_cell done')

			print('log1p...')
			self.log1p()
			print('log1p done')

			print('final check...')
			self.final()
			print('final check done.')

			print('preprocess done!')
		return self.FEATURE_PP_PATH, self.LABEL_PP_PATH if os.path.exists(self.LABEL_PP_PATH) else None


def get_tfrecord(x_path, train_data_folder=None, pred_data_path=None):
	"""
	Args:
		x_path (str): file path of preprocessed feature
			'*.npz': sparse feature; (cell_num, gene_num)
			'*.npy': dense feature; (cell_num, gene_num)
	Returns:
		(str, str): (data folder for training, data path for predicting/generating embedding)
			Training data will be shuffled; both dtypes will be tf.float32
		dict: {'n_samples': int, 'n_features': int, 'issparse': bool}.
	"""
	print('get_tfrecord...')
	if x_path.endswith('.npz'):
		from aide.utils_tf import write_csr_to_tfrecord, write_csr_shards_to_tfrecord
		X = sp.load_npz(x_path)
		train_data_folder = train_data_folder or os.path.join(os.path.dirname(x_path), 'train_csr_shards')
		pred_data_path = pred_data_path or os.path.join(os.path.dirname(x_path), 'pred_csr.tfrecord')
		write_csr_shards_to_tfrecord(X, tf_folder=train_data_folder, shard_num=10, shuffle=True)
		write_csr_to_tfrecord(X, tf_path=pred_data_path, shuffle=False)
		info_dict = {'n_samples': X.shape[0], 'n_features': X.shape[1], 'issparse':True}
		return (train_data_folder, pred_data_path), info_dict
	elif x_path.endswith('.npy'):
		from aide.utils_tf import write_ary_to_tfrecord, write_ary_shards_to_tfrecord
		X = np.load(x_path)
		train_data_folder = train_data_folder or os.path.join(os.path.dirname(x_path), 'train_ary_shards')
		pred_data_path = pred_data_path or os.path.join(os.path.dirname(x_path), 'pred_ary.tfrecord')
		write_ary_shards_to_tfrecord(X, tf_folder=train_data_folder, shard_num=10, shuffle=True)
		write_ary_to_tfrecord(X, tf_path=pred_data_path, shuffle=False)
		info_dict = {'n_samples':X.shape[0], 'n_features':X.shape[1], 'issparse':False}
		return (train_data_folder, pred_data_path), info_dict
	else:
		raise RuntimeError('Unknown type of X: {}'.format(x_path))


if __name__ == '__main__':
	import h5py
	import os
	import itertools
	from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

	def h5_to_csr(h5path, feature_npz, label_npy):
		f = h5py.File(h5path, 'r')
		sp.save_npz(feature_npz, sp.csr_matrix(f['X']))
		np.save(label_npy, np.array(f['Y']))
		f.close()

	DEMO_FOLDER = os.path.dirname(os.path.realpath(__file__))
	PREPROCESS_FOLDER = os.path.join(DEMO_FOLDER, 'data', 'Shekhar_mouse_retina_PP')

	h5path = os.path.join(DEMO_FOLDER, 'data', 'Shekhar_mouse_retina.h5')   # https://github.com/ttgump/scDeepCluster/tree/master/scRNA-seq%20data/large%20real%20datasets
	raw_feature_npz = os.path.join(DEMO_FOLDER, 'data', 'Shekhar_mouse_retina_feature.npz')
	raw_label_npy = os.path.join(DEMO_FOLDER, 'data', 'Shekhar_mouse_retina_label.npy')
	h5_to_csr(h5path, raw_feature_npz, raw_label_npy)

	pp = LargeDataPreprocessor(
		output_folder=PREPROCESS_FOLDER,
		input_feature_path=raw_feature_npz,
		input_label_path=raw_label_npy
	)
	pp_feature_npz, pp_label_npy = pp.process()
	aide_input = get_tfrecord(pp_feature_npz)
	print('AIDE Input: ', aide_input)

	aide_config = AIDEConfig()
	aide_model = AIDE(name='aide_for_retina', save_folder='aide_for_retina')
	embedding = aide_model.fit_transform(aide_input, aide_config)
	print(f'Type of embedding = {type(embedding)}; Shape of embedding = {embedding.shape}; Data type of embedding = {embedding.dtype}')

	y_true = np.load(pp_label_npy)
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
