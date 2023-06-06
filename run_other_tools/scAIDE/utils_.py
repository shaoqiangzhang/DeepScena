"""
@Author: Yu Huang
@Email: yuhuang-cst@foxmail.com
"""

import os
from tqdm import tqdm
import h5py
import sys
import scipy.sparse as sp
import numpy as np

from scipy.sparse import save_npz, load_npz, csr_matrix
import json
import pickle
import time
import joblib
import logging, logging.config

from constant import JSON_FILE_FORMAT, PKL_FILE_FORMAT, NPY_FILE_FORMAT, SPARSE_NPZ_FILE_FORMAT, JOBLIB_FILE_FORMAT, NPZ_FILE_FORMAT

def timer(func):
	def wrapper(*args, **kwargs):
		print('{0} starts running...'.format(func.__name__))
		startTime = time.time()
		ret = func(*args, **kwargs)
		print('Function {0} finished. Total time cost: {1} seconds'.format(func.__name__, time.time()-startTime))
		return ret
	return wrapper


def process_timer(func):
	def wrapper(*args, **kwargs):
		print('{0} starts running...'.format(func.__name__))
		startTime = time.process_time()
		ret = func(*args, **kwargs)
		print('Function {0} finished. Total time cost: {1} seconds'.format(func.__name__, time.process_time()-startTime))
		return ret
	return wrapper


def isJsonable(x):
	try:
		json.dumps(x)
		return True
	except:
		return False


def getsizeof(obj, unit='M'):
	mem = sys.getsizeof(obj)
	return mem / 1024 if unit == 'K' else (mem / 1048576 if unit == 'M' else mem / 1073741824)


def save_h5_csr(h5path, X, y=None):
	with h5py.File(h5path, 'w') as f:
		g = f.create_group('X')
		g.create_dataset('data', data=X.data)
		g.create_dataset('indptr', data=X.indptr)
		g.create_dataset('indices', data=X.indices)
		g.attrs['shape'] = X.shape

		if y is not None:
			f.create_dataset('y', data=y)


def save_h5_ary(h5path, X, y=None):
	with h5py.File(h5path, 'w') as f:
		f.create_dataset('X', data=X)
		if y is not None:
			f.create_dataset('y', data=y)


def is_sparse_h5(h5path):
	with h5py.File(h5path, 'r') as f:
		return isinstance(f['X'], h5py.Group)


def sample_h5_mat_rows(h5path, sample_num, verbose=0):
	shape = get_h5_mat_shape(h5path)
	sorted_sample_rows = sorted(np.random.choice(shape[0], sample_num))
	return get_h5_mat_rows(h5path, sorted_sample_rows, verbose), sorted_sample_rows


def get_h5_mat_rows(h5path, sorted_sample_rows, verbose=0):
	if isinstance(sorted_sample_rows, np.ndarray):
		sorted_sample_rows = list(sorted_sample_rows)
	with h5py.File(h5path, 'r') as f:
		is_sparse = isinstance(f['X'], h5py.Group)
		if is_sparse:
			data, indices, indptr, shape = f['X']['data'], f['X']['indices'], f['X']['indptr'], f['X'].attrs['shape']
			return get_h5_csr_rows(data, indices, indptr, shape, sorted_sample_rows, verbose=verbose)
		else:
			return f['X'][sorted_sample_rows, :]


def get_h5_mat(h5path):
	with h5py.File(h5path, 'r') as f:
		is_sparse = isinstance(f['X'], h5py.Group)
		if is_sparse:
			data, indices, indptr, shape = f['X']['data'][:], f['X']['indices'][:], f['X']['indptr'][:], f['X'].attrs['shape']
			return sp.csr_matrix((data, indices, indptr), shape=shape)
		else:
			return f['X'][:]


def get_h5_labels(h5path):
	with h5py.File(h5path, 'r') as f:
		if 'y' in f:
			return f['y'][:]
		else:
			return None


def get_h5_labels_rows(h5path, sorted_sample_rows):
	with h5py.File(h5path, 'r') as f:
		if 'y' in f:
			return f['y'][sorted_sample_rows]
		else:
			return None


def get_h5_mat_shape(h5path):
	"""
	Returns:
		tuple
	"""
	with h5py.File(h5path, 'r') as f:
		is_sparse = isinstance(f['X'], h5py.Group)
		shape = f['X'].attrs['shape'] if is_sparse else f['X'].shape
		return (int(shape[0]), int(shape[1]))


def get_h5_csr_rows(data, indices, indptr, shape, sorted_sample_rows, verbose=0):
	"""
	Args:
		data (h5py.Dataset):
		indices (h5py.Dataset):
		indptr (h5py.Dataset):
		sample_rows (list or np.ndarray):
	Returns:
		csr_matrix
	"""
	new_data, new_indices, new_indptr = [], [], [0]
	it = tqdm(enumerate(sorted_sample_rows), total=len(sorted_sample_rows)) if verbose else enumerate(sorted_sample_rows)
	for i, r in it:
		b, e = indptr[r], indptr[r+1]
		new_indptr.append(new_indptr[i] + e - b)
		new_data.append(data[b: e])
		new_indices.append(indices[b: e])
	return csr_matrix((np.hstack(new_data), np.hstack(new_indices), new_indptr), shape=(len(sorted_sample_rows), shape[1]))


def get_mat_memuse(m):
	"""get matrix's memory;  unit: GB
	"""
	assert isinstance(m, sp.csr_matrix) or isinstance(m, np.ndarray)
	if sp.issparse(m):
		nbytes = m.data.nbytes + m.indptr.nbytes + m.indices.nbytes
	else:
		nbytes = m.nbytes
	return nbytes / 1073741824  # 1024**3; G


def get_load_func(file_format):
	if file_format == JSON_FILE_FORMAT:
		return lambda path: json.load(open(path))
	if file_format == PKL_FILE_FORMAT:
		return lambda path: pickle.load(open(path, 'rb'))
	if file_format == NPY_FILE_FORMAT or file_format == NPZ_FILE_FORMAT:
		return lambda path: np.load(path)
	if file_format == SPARSE_NPZ_FILE_FORMAT:
		return lambda path: load_npz(path)
	if file_format == JOBLIB_FILE_FORMAT:
		return lambda path: joblib.load(path)
	assert False


def get_save_func(file_format):
	if file_format == JSON_FILE_FORMAT:
		return lambda obj, path: json.dump(obj, open(path, 'w'), indent=2, ensure_ascii=False)
	if file_format == PKL_FILE_FORMAT:
		return lambda obj, path: pickle.dump(obj, open(path, 'wb'))
	if file_format == NPY_FILE_FORMAT:
		return lambda obj, path: np.save(path, obj)
	if file_format == NPZ_FILE_FORMAT:
		return lambda obj, path: np.savez_compressed(path, obj)
	if file_format == SPARSE_NPZ_FILE_FORMAT:
		return lambda obj, path: save_npz(path, obj)
	if file_format == JOBLIB_FILE_FORMAT:
		return lambda obj, path: joblib.dump(obj, path)
	assert False


def check_return(attrCollector):
	def outerWrapper(func):
		def wrapper(cls, *args, **kwargs):
			coll = getattr(cls, attrCollector, None)
			if coll is not None:
				return coll
			coll = func(cls, *args, **kwargs)
			setattr(cls, attrCollector, coll)
			return coll
		return wrapper
	return outerWrapper


def get_logger(name, logPath=None, level=logging.DEBUG, mode='a'):
	"""
	Args:
		name (str or None): None means return root logger
		logPath (str or None): log文件路径
	"""
	formatter = logging.Formatter(fmt="%(message)s", datefmt='%Y-%m-%d %H:%M:%S')
	logger = logging.getLogger(name)
	if len(logger.handlers) != 0:
		return logger
	logger.setLevel(level)
	if logPath is not None:
		fh = logging.FileHandler(logPath, mode=mode)
		fh.setFormatter(formatter)
		logger.addHandler(fh)
	ch = logging.StreamHandler()
	ch.setFormatter(formatter)
	logger.addHandler(ch)
	return logger


def delete_logger(logger):
	while logger.handlers:
		logger.handlers.pop()


def sparse2tuple(mx):
	"""Convert sparse matrix to tuple representation.
		ref: https://github.com/tkipf/gcn/blob/master/gcn/utils.py
	"""
	if not sp.isspmatrix_coo(mx):
		mx = mx.tocoo()
	coords = np.vstack((mx.row, mx.col)).transpose()
	values = mx.data
	shape = mx.shape
	return coords, values, shape


def x_to_input(X):
	return sparse2tuple(X) if sp.issparse(X) else X


def sparse_x_to_input(X):
	X = sparse2tuple(X)
	return X, X[1].shape


def read_file_folder(path, handle_func, recursive=True):
	"""
	Args:
		path (string): path of file or file folder
		handleFunc (function): paras = (file_path)
		recursive (bool): Whether to recursively traverse the sub folders
	"""
	if os.path.isfile(path):
		handle_func(path)
	elif recursive:
		for file_name in os.listdir(path):
			file_dir = os.path.join(path, file_name)
			read_file_folder(file_dir, handle_func, recursive)


def get_file_list(path, filter):
	"""
	Args:
		dir (string): path of file or file folder
		filter (function): paras = (file_path); i.e. filter=lambda file_path: file_path.endswith('.json')
	Returns:
		list: [file_path1, file_path2, ...]
	"""
	def handle_func(file_path):
		if filter(file_path):
			file_list.append(file_path)
	file_list = []
	read_file_folder(path, handle_func)
	return file_list


def l2_normalize(X):
	"""
	Args:
		X (np.array or sp.csr_matrix): (n_samples, n_features)
	Returns:
		np.array or sp.csr_matrix: (n_samples, n_features)
	"""
	row_inv_norm = 1. / l2_norm(X).reshape((-1, 1))
	return X.multiply(row_inv_norm) if sp.issparse(X) else X * row_inv_norm


def l2_norm(X):
	"""
	Args:
		X (np.array or sp.csr_matrix): (n_samples, n_features)
	Returns:
		np.array: (n_samples,)
	"""
	return sparse_l2_norm(X) if sp.issparse(X) else ary_l2_norm(X)


def ary_l2_norm(X):
	return np.sqrt(np.square(X).sum(axis=1))


def sparse_l2_norm(X):
	return np.sqrt(X.power(2).sum(axis=-1)).A


def get_sample_ranks(sample_range, sample_num):
	return np.random.choice(sample_range, sample_num)


def sample_dist_from(X, sample_num, dist_name):
	from aide.utils_dist import get_dist_func
	dist_func = get_dist_func(dist_name)
	ranks1, ranks2 = get_sample_ranks(X.shape[0], sample_num), get_sample_ranks(X.shape[0], sample_num)
	X1, X2 = X[ranks1], X[ranks2]
	sample_dist = dist_func(X1, X2)
	return sample_dist
