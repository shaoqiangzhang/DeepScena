"""
@Author: Yu Huang
@Email: yuhuang-cst@foxmail.com
"""

import numpy as np
from scipy.stats import pearsonr, spearmanr, entropy
from scipy.spatial.distance import braycurtis, chebyshev
from sklearn.metrics.pairwise import paired_distances
from multiprocessing import Pool

def euclidean_dist(X, Y):
	"""range: (0, inf)
	Args:
		X (scipy.sparse.csr_matrix): (sample_num, feature_num)
		Y (scipy.sparse.csr_matrix): (sample_num, feature_num)
	Returns:
		np.ndarray: (sample_num,)
	"""
	return paired_distances(X, Y, 'euclidean')


def manhattan_dist(X, Y):
	return paired_distances(X, Y, 'manhattan')


def cosine_dist(X, Y):
	"""range: (0, 2)
	"""
	return paired_distances(X, Y, 'cosine')  # 1 - cos(x, y)


def jaccard_dist(X, Y):
	"""range: (0, 1)
	"""
	x_size = X.sum(axis=1).A.flatten()  # np.ndarray; (sample_num,)
	y_size = Y.sum(axis=1).A.flatten()  # np.ndarray; (sample_num,)
	x_inter_y = X.multiply(Y).sum(axis=1).A.flatten()  # np.ndarray; (sample_num,)
	return 1 - x_inter_y / (x_size + y_size - x_inter_y)


def divide_XY(X, Y, cpu=12, min_chunk=200, max_chunk=1000):
	sample_size = X.shape[0]
	chunk_size = max(min(sample_size // cpu, max_chunk), min_chunk)
	intervals = list(range(0, sample_size, chunk_size)) + [sample_size]
	paraList = [(X[intervals[i]: intervals[i + 1]], Y[intervals[i]: intervals[i + 1]]) for i in
		range(len(intervals) - 1)]
	return paraList


def cal_dist_parallel(X, Y, f, cpu=12):
	paraList = divide_XY(X, Y, cpu=cpu)
	if cpu == 1:
		return np.hstack(list(map(pearson_dist_wrapper, paraList)))
	with Pool(cpu) as pool:
		return np.hstack(pool.map(f, paraList))


def pearson_dist_wrapper(args):
	X, Y = args
	X, Y = X.A, Y.A
	return 1 - np.array([pearsonr(X[i], Y[i])[0] for i in range(X.shape[0])])


def pearson_dist(X, Y, cpu=12):
	"""range: (0, 2)
	"""
	return cal_dist_parallel(X, Y, pearson_dist_wrapper, cpu)


def spearman_dist_wrapper(args):
	X, Y = args
	X, Y = X.A, Y.A
	return 1 - np.array([spearmanr(X[i], Y[i])[0] for i in range(X.shape[0])])


def spearman_dist( X, Y, cpu=12):
	"""range: (0, 2)
	"""
	return cal_dist_parallel(X, Y, spearman_dist_wrapper, cpu)


def bray_curtis_dist_wraper(args):
	X, Y = args
	X, Y = X.A, Y.A
	return np.array([braycurtis(X[i], Y[i]) for i in range(X.shape[0])])


def bray_curtis_dist(X, Y, cpu=12):
	"""range: [0, 1] if all coordinates are positive
	"""
	return cal_dist_parallel(X, Y, bray_curtis_dist_wraper, cpu)


def chebyshev_dist_wraper(args):
	X, Y = args
	X, Y = X.A, Y.A
	return np.array([chebyshev(X[i], Y[i]) for i in range(X.shape[0])])


def chebyshev_dist(X, Y, cpu=12):
	"""range: [0, inf]
	"""
	return cal_dist_parallel(X, Y, chebyshev_dist_wraper, cpu)


def kl_dist_wraper(args):
	X, Y = args
	X, Y = X.A + 1e-6, Y.A + 1e-6  # smooth
	kl1 = np.array([entropy(X[i], Y[i]) for i in range(X.shape[0])])
	kl2 = np.array([entropy(Y[i], X[i]) for i in range(X.shape[0])])
	return (kl1 + kl2) / 2


def kl_dist(X, Y, cpu=12):
	"""range: [0, inf]; X、Y should be positive
	"""
	return cal_dist_parallel(X, Y, kl_dist_wraper, cpu)


distname2func = {
	'euclidean': euclidean_dist,
	'manhattan': manhattan_dist,
	'cosine': cosine_dist,
	'pearson': pearson_dist,
	'spearman': spearman_dist,
	'bray_curtis': bray_curtis_dist,
	'chebyshev': chebyshev_dist,
	'kl_divergence': kl_dist,

	'jaccard': jaccard_dist,
}

def get_dist_func(dist_name):
	return distname2func[dist_name]


def get_all_dist_name():
	return list(distname2func.keys())


if __name__ == '__main__':
	pass
	def test_dist():
		from scipy.sparse import csr_matrix
		x = csr_matrix([
			[1., 0., 2., 0.],
			[3., 0., 3., 0.],
			[5., 6., 1., 0.]])
		y = csr_matrix([
			[1., 0., 2., 0.],
			[1., 0., 3., 0.],
			[5., 3., 6., 7.]])
		print('euclidean_dist', euclidean_dist(x, y))
		print('manhattan_dist', manhattan_dist(x, y))
		print('cosine_dist', cosine_dist(x, y))
		print('pearson_dist', pearson_dist(x, y))
		print('spearman_dist', spearman_dist(x, y))
		print('bray_curtis_dist', bray_curtis_dist(x, y))
		print('chebyshev_dist', chebyshev_dist(x, y))
		print('kl_dist', kl_dist(x, y))

		bx, by = x.copy(), y.copy()
		bx[bx > 0] = 1; bx[bx < 0] = 0
		by[by > 0] = 1; by[by < 0] = 0
		print('jaccard_dist', jaccard_dist(bx, by))


	test_dist()





