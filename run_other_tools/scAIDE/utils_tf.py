"""
@Author: Yu Huang
@Email: yuhuang-cst@foxmail.com
"""

import tensorflow as tf
import numpy as np
from tqdm import tqdm
import os
import shutil
from scipy.stats import spearmanr

from utils_ import timer
from constant import OPTIMIZER_SGD, OPTIMIZER_ADAM, OPTIMIZER_RMS, RELU, LEAKY_RELU, TANH, SIGMOID


def get_sess_config():
	sess_config = tf.ConfigProto()
	sess_config.gpu_options.allow_growth = True
	return sess_config


def get_optimizer(type, lr, **kwargs):
	if type == OPTIMIZER_SGD:
		return tf.train.GradientDescentOptimizer(lr, **kwargs)
	if type == OPTIMIZER_ADAM:
		return tf.train.AdamOptimizer(lr, **kwargs)
	if type == OPTIMIZER_RMS:
		return tf.train.RMSPropOptimizer(lr, **kwargs)
	assert False


def get_placeholder(sparse, dtype, shape, name=None):
	return tf.sparse_placeholder(dtype, shape=shape, name=name) if sparse else tf.placeholder(dtype, shape=shape, name=name)


def concat(tensors, axis, sparse):
	return tf.sparse_concat(sp_inputs=tensors, axis=axis) if sparse else tf.concat(tensors, axis=axis)


def get_active_func(funcName):
	if funcName == RELU:
		return tf.nn.relu
	elif funcName == LEAKY_RELU:
		return tf.nn.leaky_relu
	elif funcName == TANH:
		return tf.tanh
	elif funcName == SIGMOID:
		return tf.sigmoid
	elif funcName is None:
		return tf.identity
	else:
		assert False


def get_mask_retain_mat(X, keep_prob, dtype):
	random_tensor = tf.random_uniform(tf.shape(X))
	random_tensor += keep_prob
	to_retain = tf.cast(tf.floor(random_tensor), dtype=dtype)
	return to_retain


def get_zifa_mask_retain_mat(X, lambda_, dtype):
	keep_prob = 1. - tf.math.exp(-lambda_ * X)
	random_tensor = tf.random_uniform(tf.shape(X))
	random_tensor += keep_prob
	to_retain = tf.cast(tf.floor(random_tensor), dtype=dtype)
	return to_retain


def sparse_mask(X, keep_prob, training=False):
	if (not training) or keep_prob == 1.0:
		return X
	to_retain = get_mask_retain_mat(X.values, keep_prob, tf.bool)
	return tf.sparse_retain(X, to_retain)


def sparse_zifa_mask(X, keep_prob, lambda_, training=False):
	if (not training) or keep_prob == 1.0:
		return X
	to_retain = tf.logical_or(
		get_mask_retain_mat(X.values, keep_prob, tf.bool),
		get_zifa_mask_retain_mat(X.values, lambda_, tf.bool))
	return tf.sparse_retain(X, to_retain)


def dense_mask(X, keep_prob, training=False):
	if (not training) or keep_prob == 1.0:
		return X
	to_retain = get_mask_retain_mat(X, keep_prob, X.dtype)
	return tf.multiply(X, to_retain)


def dense_zifa_mask(X, keep_prob, lambda_, training=False):
	if (not training) or keep_prob == 1.0:
		return X, tf.constant(1.0, dtype=X.dtype)
	to_retain = tf.logical_or(
		get_mask_retain_mat(X, keep_prob, tf.bool),
		get_zifa_mask_retain_mat(X, lambda_, tf.bool))
	to_retain = tf.cast(to_retain, X.dtype)
	X_non_zero = tf.cast(tf.not_equal(X, 0.), X.dtype)
	return tf.multiply(X, to_retain), tf.reduce_mean(tf.reduce_sum(X_non_zero * to_retain, axis=1) / tf.reduce_sum(X_non_zero, axis=1))


def dense_zifa_dropout(X, keep_prob, lambda_, training=False):
	if (not training) or keep_prob == 1.0:
		return X, tf.constant(1.0, dtype=X.dtype)
	to_retain = tf.logical_or(
		get_mask_retain_mat(X, keep_prob, tf.bool),
		get_zifa_mask_retain_mat(X, lambda_, tf.bool))
	to_retain = tf.cast(to_retain, X.dtype)
	X_non_zero = tf.cast(tf.not_equal(X, 0.), X.dtype)
	keep_prob = tf.reduce_sum(X_non_zero * to_retain, axis=1, keepdims=True) / tf.reduce_sum(X_non_zero, axis=1, keepdims=True)  # (batch_size, 1)
	return tf.multiply(X, to_retain) * (1./keep_prob), tf.reduce_mean(keep_prob)


def sparse_dropout(X, keep_prob):
	"""Dropout for sparse tensors.
	copy from https://github.com/tkipf/gcn/blob/master/gcn/layers.py
	"""
	pre_out = sparse_mask(X, keep_prob)
	return pre_out * (1./keep_prob)


def dropout(X, keep_prob):
	return sparse_dropout(X, keep_prob) if type(X) == tf.SparseTensor else tf.nn.dropout(X, keep_prob)


# dist ================================================================================================================
def sparse_euclidean_dist(X, Y, eps=0.0):
	"""
	Args:
		X (tf.SparseTensor): (sample_num, feature_num)
		Y (tf.SparseTensor): (sample_num, feature_num)
	Returns:
		tf.tensor: (sample_num,)
	"""
	return tf.sqrt(tf.sparse_reduce_sum(tf.math.square(tf.sparse_add(X, tf.math.negative(Y), threshold=1e-12)), axis=-1) + eps)


def euclidean_dist2(X, Y):
	return tf.reduce_sum(tf.square(X - Y), axis=-1)


def euclidean_dist(X, Y, eps=0.0):
	"""range: (0, inf)
	Args:
		X (tf.tensor): (sample_num, feature_num)
		Y (tf.tensor): (sample_num, feature_num)
	Returns:
		tf.tensor: (sample_num,)
	"""
	return tf.sqrt(tf.reduce_sum(tf.square(X - Y), axis=-1) + eps)


def cosine_dist(X, Y):
	"""range: (0, 2)
	"""
	X = tf.math.l2_normalize(X, axis=-1)
	Y = tf.math.l2_normalize(Y, axis=-1)
	return 1. - tf.reduce_sum(tf.multiply(X, Y), axis=-1)


def sparse_l2_normalize(X, axis=-1, epsilon=1e-12):
	"""
	Args:
		X (tf.SparseTensor): (sample_num, feature_num)
		axis (int):
		epsilon (float):
	Returns:
		tf.SparseTensor: (sample_num, feature_num)
	"""
	squad_sum = tf.sparse_reduce_sum(tf.math.square(X), axis=axis, keepdims=True)    # tf.tensor; (sample_num, 1)
	x_inv_norm = tf.math.rsqrt(tf.maximum(squad_sum, epsilon))  # tf.tensor; (sample_num, 1)
	return X.__mul__(x_inv_norm)


def pearson_corr(X, Y):
	"""\frac{\sum (x - m_x) (y - m_y)}{\sqrt{\sum (x - m_x)^2 \sum (y - m_y)^2}}
	Args:
		X (tf.tensor): (sample_num, feature_num)
		Y (tf.tensor): (sample_num, feature_num)
	Returns:
		tf.tensor: (sample_num,)
	"""
	X_mean, Y_mean = tf.reduce_mean(X, axis=-1, keepdims=True), tf.reduce_mean(Y, axis=-1, keepdims=True) # (sample_num, 1)
	a = tf.reduce_sum(tf.multiply(X - X_mean, Y - Y_mean), axis=-1) # (sample_num,)
	b = tf.sqrt(tf.multiply(tf.reduce_sum(tf.square(X - X_mean), axis=-1), tf.reduce_sum(tf.square(Y - Y_mean), axis=-1)))  # (sample_num,)
	return a / b


def pearson_dist(X, Y):
	"""(0, 2)
	"""
	return 1 - pearson_corr(X, Y)


def spearman_corr_wrap(X, Y):
	sample_num, dtype = X.shape[0], X.dtype
	ret = np.zeros(sample_num, dtype=dtype.name)
	for i in range(sample_num):
		ret[i] = spearmanr(X[i], Y[i])[0]
	return ret


def spearman_corr(X, Y):
	"""FIXME:
	Args:
		X (tf.Tensor): (sample_num, feature_num), dtype=tf.float32 or tf.float64
		Y (tf.Tensor): (sample_num, feature_num), dtype=tf.float32 or tf.float64
	Returns:
		tf.tensor: (sample_num,)
	"""
	return tf.py_function(spearman_corr_wrap, [X, Y], Tout=X.dtype)


def spearman_dist(X, Y):
	"""(0, 2)
	"""
	return 1. - spearman_corr(X, Y)


def sparse_multiply_sparse(X, Y):
	return tf.SparseTensor(X.indices, tf.gather_nd(Y, X.indices) * X.values, X.dense_shape)


def manhattan_dist(X, Y):
	return tf.reduce_sum(tf.abs(X - Y), axis=-1)


def sparse_manhattan_dist(X, Y):
	return tf.sparse_reduce_sum(tf.abs(tf.sparse_add(X, tf.math.negative(Y), threshold=1e-12)), axis=-1)


def chebyshev_dist(X, Y):
	return tf.reduce_max(tf.abs(X - Y), axis=-1)


def sparse_chebyshev_dist(X, Y):
	return tf.sparse_reduce_max(tf.abs(tf.sparse_add(X, tf.math.negative(Y), threshold=1e-12)), axis=-1)


distname2func_dense = {
	'euclidean': euclidean_dist,
	'pearson': pearson_dist,
	'spearman': spearman_dist,
	'manhattan': manhattan_dist,
	'cosine': cosine_dist,
	'chebyshev': chebyshev_dist,
}

distname2func_sparse = {
	'euclidean': sparse_euclidean_dist,
	'manhattan': sparse_manhattan_dist,
	'chebyshev': sparse_chebyshev_dist,
}

def get_dist_func_(dist_name, distname2func, mark):
	if dist_name in distname2func:
		return distname2func[dist_name]
	raise RuntimeError(f'Not support {mark} version of {dist_name}')


def get_dist_func(dist_name, sparse=False):
	return get_dist_func_(dist_name, distname2func_sparse, 'sparse') if sparse else get_dist_func_(dist_name, distname2func_dense, 'dense')
# dist related end ====================================================================================================


def bytes_feature(value):
	"""Returns a bytes_list from a string / byte."""
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_feature(value):
	"""Returns a float_list from a float / double."""
	return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def int64_feature(value):
	"""Returns an int64_list from a bool / enum / int / uint."""
	return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def write_csr_to_tfrecord(X, tf_path, shuffle=False):
	"""
	Args:
		X (sp.csr_matrix)
		save_path (str)
	"""
	if X.dtype != np.float32:
		X = X.astype(np.float32)
	row_ary = np.arange(0, X.shape[0])
	if shuffle:
		np.random.shuffle(row_ary)
	with tf.io.TFRecordWriter(tf_path) as tfwriter:
		for i in tqdm(row_ary):
			row_x = X[i]
			feature_dict = {
				'idx': int64_feature(row_x.indices),
				'values': float_feature(row_x.data),
				'col_num': int64_feature([X.shape[1]])
			}
			row_example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
			tfwriter.write(row_example.SerializeToString())


def tfrow_to_sparse_tensor(el):
	feature_dict = {
		'idx': tf.VarLenFeature(dtype=tf.int64),
		'values': tf.VarLenFeature(dtype=tf.float32),
		'col_num': tf.FixedLenFeature(shape=(), dtype=tf.int64)
	}
	parsed_example = tf.io.parse_single_example(el, features=feature_dict)
	col_idx = tf.expand_dims(tf.sparse_tensor_to_dense(parsed_example['idx']), axis=0)
	values = tf.sparse_tensor_to_dense(parsed_example['values'])
	row_idx = tf.zeros_like(col_idx)
	indices = tf.transpose(tf.concat([row_idx, col_idx], axis=0))
	ret = tf.SparseTensor(indices, values, (1, parsed_example['col_num']))
	return ret


def write_ary_to_tfrecord(X, tf_path, shuffle=False):
	"""
	Args:
		X (np.ndarray)
		save_path (str)
	"""
	if X.dtype != np.float32:
		X = X.astype(np.float32)
	row_ary = np.arange(0, X.shape[0])
	if shuffle:
		np.random.shuffle(row_ary)
	with tf.io.TFRecordWriter(tf_path) as tfwriter:
		for i in tqdm(row_ary):
			row_x = X[i]
			feature_dict = {'values': float_feature(row_x)}
			row_example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
			tfwriter.write(row_example.SerializeToString())


def tfrow_to_dense_tensor(el):
	feature_dict = {'values': tf.VarLenFeature(dtype=tf.float32)}
	parsed_example = tf.io.parse_single_example(el, features=feature_dict)
	values = tf.sparse_tensor_to_dense(parsed_example['values'])
	return values


def write_shards_to_tfrecord(write_func, X, tf_folder, shard_num=10, shuffle=True):
	"""
	Args:
		write_func (function)
		X (sp.csr_matrix)
		tf_folder (str)
		shard_num (int)
		shuffle (bool)
	"""
	shutil.rmtree(tf_folder, ignore_errors=True); os.makedirs(tf_folder)
	if X.dtype != np.float32:
		X = X.astype(np.float32)
	row_ary = np.arange(0, X.shape[0])
	if shuffle:
		np.random.shuffle(row_ary)
	shard_size = X.shape[0] // shard_num
	if X.shape[0] % shard_num != 0:
		shard_size += 1
	for i in range(shard_num):
		write_func(
			X[row_ary[i*shard_size:(i+1)*shard_size]],
			os.path.join(tf_folder, '{:0>4d}.tfrecord'.format(i)), shuffle=False)


def write_csr_shards_to_tfrecord(X, tf_folder, shard_num=10, shuffle=True):
	write_shards_to_tfrecord(write_csr_to_tfrecord, X, tf_folder, shard_num, shuffle)


def write_ary_shards_to_tfrecord(X, tf_folder, shard_num=10, shuffle=True):
	write_shards_to_tfrecord(write_ary_to_tfrecord, X, tf_folder, shard_num, shuffle)
