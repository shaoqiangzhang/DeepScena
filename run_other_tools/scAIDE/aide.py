"""
@Author: Yu Huang
@Email: yuhuang-cst@foxmail.com
"""

import json
import os
import tensorflow as tf
import numpy as np
import scipy.sparse as sp
from copy import copy
from sklearn.utils import check_array

from model_config import Config
from utils_ import timer
from utils_ import get_logger, delete_logger, x_to_input, get_file_list
from utils_draw import simple_multi_line_plot
from utils_tf import get_optimizer, get_active_func, get_sess_config, get_placeholder, get_dist_func, euclidean_dist, euclidean_dist2
from utils_tf import tfrow_to_sparse_tensor, tfrow_to_dense_tensor
from constant import OPTIMIZER_ADAM, RELU, SIGMOID, LEAKY_RELU, TANH, TRAIN_MODE, EVAL_MODE, PREDICT_MODE
from constant import DATA_MAT, DATA_TFRECORD, MDS_LOSS_S_STRESS, MDS_LOSS_SAMMON, MDS_LOSS_SQUARE_STRESS_1
from constant import MDS_LOSS_ABS_STRESS, MDS_LOSS_ELASTIC, MDS_LOSS_NORM_STRESS, MDS_LOSS_RAW_STRESS


class AIDEConfig(Config):
	def __init__(self, path=None, assign_dict=None):
		super(AIDEConfig, self).__init__()
		self.optimizer = OPTIMIZER_ADAM
		self.lr = 0.0001                        # Learning rate
		self.optimizer_kwargs = {}

		self.alpha = 12.0                       # Weight of MDS loss: L = reconstruct_loss + mds_loss * alpha + l2_loss * w_decay
		self.w_decay = 0.0                      # Weight of l2 norm loss
		self.ae_drop_out_rate = 0.4             # Dropout rate of autoencoder
		self.mds_drop_out_rate = 0.0            # Dropout rate of mds encoder

		self.ae_units = [1024, 512, 256]        # Units of Autoencoder: n_features*1024 - relu - 1024*512 - relu - 512*256 - relu - 256*512 - relu - 512*1024 - relu - 1024*n_features - relu
		self.ae_acts = [RELU, RELU, RELU]

		self.mds_units = [1024, 512, 256]       # Units of MDS Encoder: n_features*1024 - relu - 1024*512 - relu - 512*256 - none
		self.mds_acts = [RELU, RELU, None]

		self.dist_name = 'euclidean'            # 'euclidean' | 'manhattan' | 'chebyshev' | 'cosine' | 'pearson'
		self.mds_loss = MDS_LOSS_ABS_STRESS     # MDS_LOSS_ABS_STRESS | MDS_LOSS_S_STRESS | MDS_LOSS_RAW_STRESS | MDS_LOSS_NORM_STRESS | MDS_LOSS_SQUARE_STRESS_1 | MDS_LOSS_ELASTIC | MDS_LOSS_SAMMON
		self.dist_eps = 1e-6                    # Avoid 'nan' during back propagation

		self.pretrain_step_num = 1000           # The autoencoder will be pretrained with reconstruction loss by feeding (pretrain_step_num * batch_size * 2) samples.
		self.max_step_num = 20000               # Maximize Number of batches to run
		self.min_step_num = 4000                # Minimize number of batches to run
		self.early_stop_patience = 6            # None | int: Training will stop when no improvement is shown during (early_stop_patience * val_freq) steps. Set to None if early stopping is not used.

		self.print_freq = 50                    # Print train loss every print_freq steps.
		self.val_freq = 100                   # Calculate validation loss every val_freq steps (Note that it is used for early stopping)
		self.draw_freq = 500                    # Draw
		self.save_model = False                 # Whether to save model
		self.fix_ae = False                     # Whether to fix parameters of autoencoder when training MDS encoder
		self.verbose = True

		self.batch_size = 256                   # (batch_size * 2) samples will be fed in each batch during training
		self.validate_size = 2560               # validate_size samples will be used as validation set
		self.embed_batch_size = 2560            # embed_batch_size samples will be fed in each batch during generating embeddings
		self.train_shuffle_buffer = self.batch_size * 10
		self.train_interleave_cycle = 2

		# Will be set automatically
		self.n_samples = None
		self.n_features = None
		self.issparse = None
		self.dtype = None
		self.feed_type = None
		self.train_tfrecord_path = None
		self.pred_tfrecord_path = None

		if path is not None:
			self.load(path)

		if assign_dict is not None:
			self.assign(assign_dict)


class AIDEModel(object):
	def __init__(self, mode, config, batch):
		"""
		Args:
			config (DistAEConfig)
		"""
		self.mode = mode
		self.config = config
		self.forward(batch)


	def forward(self, batch):
		X = batch
		c = self.config
		self.cal_dist = get_dist_func(c.dist_name, sparse=False)

		if type(X) == tf.SparseTensor:
			X = tf.sparse_tensor_to_dense(X, validate_indices=False)

		if X.get_shape().as_list()[-1] is None:
			X = tf.reshape(X, (-1, c.n_features))

		# encoder
		with tf.variable_scope('AE'):
			self.ae_h = self.encoder(X, c.ae_units, c.ae_acts, c.ae_drop_out_rate)
			units, acts = self.get_decoder_acts_units(c.ae_units, c.ae_acts, c.n_features)
			X_hat = self.decoder(self.ae_h, units, acts)

		with tf.variable_scope('MDS'):
			self.mds_h = self.encoder(X, c.mds_units, c.mds_acts, c.mds_drop_out_rate)

		if self.mode == PREDICT_MODE:
			return

		# loss
		self.reconstruct_loss = self.mds_loss = self.l2_loss = tf.constant(0., dtype=X.dtype)

		self.reconstruct_loss = self.get_reconstruct_loss(X, X_hat)

		pair_num = tf.cast(tf.shape(self.mds_h)[0] / 2, tf.int32)
		h1, h2 = self.mds_h[:pair_num], self.mds_h[pair_num:]
		dist = self.cal_dist(X_hat[:pair_num], X_hat[pair_num:])
		self.mds_loss = self.get_mds_loss(c.mds_loss, h1, h2, dist, c.dist_eps)

		if c.w_decay > 1e-8:   # l2 loss
			self.l2_loss = self.get_l2_loss(c.w_decay)

		self.pretrain_loss = self.reconstruct_loss
		self.loss = self.reconstruct_loss + self.mds_loss * c.alpha + self.l2_loss
		self.all_loss = [self.reconstruct_loss, self.mds_loss, self.l2_loss]

		if self.mode == EVAL_MODE:
			return

		# optimize
		self.global_step = tf.Variable(0, trainable=False, name='global_step')

		optimizer = get_optimizer(c.optimizer, c.lr, **c.optimizer_kwargs)
		self.pretrain_op = optimizer.minimize(self.pretrain_loss)

		optimizer = get_optimizer(c.optimizer, c.lr, **c.optimizer_kwargs)
		if c.fix_ae:
			scope_name = f'{tf.get_variable_scope().name}/MDS'
			var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope_name)
			fix_ae_loss = self.mds_loss * c.alpha + self.l2_loss
			self.train_op = optimizer.minimize(fix_ae_loss, global_step=self.global_step, var_list=var_list)
		else:
			self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)

		self.init_op = tf.global_variables_initializer()


	def encoder(self, X, units, acts, drop_out_rate):
		"""
		Args:
			X (tf.Tensor): (batch_size, feature_num)
			units (list)
			acts (list)
		"""
		h = X
		for i in range(0, len(units)):
			h = tf.layers.dropout(h, rate=drop_out_rate, training=(self.mode == TRAIN_MODE), name=f'encoder_dropout_{i}')
			h = tf.layers.dense(h, units[i],
				activation=get_active_func(acts[i]),
				kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True),
				name=f'encoder_layer_{i}',
			)
		return h


	def decoder(self, X, units, acts):
		h = X
		for i in range(len(units)):
			h = tf.layers.dense(
				h, units[i],
				activation=get_active_func(acts[i]),
				kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True),
				name = 'decoder_layer_{}'.format(i),
			)
		return h


	def get_decoder_acts_units(self, units, acts, x_size):
		units, acts = copy(units), copy(acts)  # [dim(1), ..., h_size]
		units = [x_size] + units[:-1]; units.reverse()  # [dim(-2), ..., feature_size]
		acts = [None] + acts[:-1]; acts.reverse()
		return units, acts


	def sparse_layer(self, X, x_size, units, activation, kernel_initializer, name):
		with tf.variable_scope(name):
			W = tf.get_variable('W', shape=(x_size, units), dtype=X.dtype, initializer=kernel_initializer)
			b = tf.get_variable('b', shape=(units,), dtype=X.dtype, initializer=tf.zeros_initializer())
			return activation(tf.sparse_tensor_dense_matmul(X, W) + b)


	def get_mds_loss(self, loss_name, h1, h2, dist, eps):
		if loss_name == MDS_LOSS_RAW_STRESS:
			return self.get_raw_stress_loss(h1, h2, dist, eps)
		elif loss_name == MDS_LOSS_NORM_STRESS:
			return self.get_norm_stress_loss(h1, h2, dist, eps)
		elif loss_name == MDS_LOSS_SQUARE_STRESS_1:
			return self.get_square_stress_1_loss(h1, h2, dist, eps)
		elif loss_name == MDS_LOSS_ELASTIC:
			return self.get_elastic_scaling_loss(h1, h2, dist, eps)
		elif loss_name == MDS_LOSS_SAMMON:
			return self.get_sammon_loss(h1, h2, dist, eps)
		elif loss_name == MDS_LOSS_S_STRESS:
			return self.get_s_stress_loss(h1, h2, dist)
		elif loss_name == MDS_LOSS_ABS_STRESS:
			return self.get_abs_ss_stress_loss(h1, h2, dist)
		else:
			raise RuntimeError('Unknown Dist Loss Name: {}'.format(loss_name))


	def get_raw_stress_loss(self, h1, h2, dist, eps):
		"""Raw stress (Kruskal, 1964)
		"""
		return tf.reduce_mean(tf.square( euclidean_dist(h1, h2, eps) - dist ))


	def get_norm_stress_loss(self, h1, h2, dist, eps):
		return tf.reduce_sum(tf.square(euclidean_dist(h1, h2, eps) - dist)) / tf.reduce_sum(tf.square(dist) + eps)


	def get_square_stress_1_loss(self, h1, h2, dist, eps):
		"""Stress-1 (Kruskal, 1964); Note that the original stress-1 loss has 'sqrt'
		"""
		dist_h = euclidean_dist(h1, h2, eps)
		return tf.reduce_sum(tf.square(dist_h - dist)) / tf.reduce_sum(tf.square(dist_h))


	def get_elastic_scaling_loss(self, h1, h2, dist, eps):
		"""Elastic scaling loss (McGee, 1966)
		"""
		return tf.reduce_mean(tf.square( 1 - euclidean_dist(h1, h2, eps) / (dist + eps) ))


	def get_sammon_loss(self, h1, h2, dist, eps):
		"""Sammon loss (Sammon, 1969)
		"""
		return tf.reduce_mean(tf.square( euclidean_dist(h1, h2, eps) - dist ) / (dist + eps))


	def get_s_stress_loss(self, h1, h2, dist):
		"""S-Stress loss function (Takane, Young, and De Leeuw, 1977)
		"""
		return tf.reduce_mean( tf.square(euclidean_dist2(h1, h2) - tf.square(dist)) )


	def get_abs_ss_stress_loss(self, h1, h2, dist):
		return tf.reduce_mean( tf.abs(euclidean_dist2(h1, h2) - tf.square(dist)) )


	def get_reconstruct_loss(self, X, X_hat):
		return tf.reduce_mean(euclidean_dist2(X_hat, X))


	def get_l2_loss(self, wDecay):
		return tf.contrib.layers.apply_regularization(
			tf.contrib.layers.l2_regularizer(wDecay), tf.trainable_variables())


	def get_sparse_reg_loss(self, h, rho):
		h = tf.nn.sigmoid(h)  # (batch_size, hidden)
		rho_hat = tf.reduce_mean(h, axis=0)  # (hidden,)
		return self.kl_div(rho, rho_hat)


	def kl_div(self, rho, rho_hat):
		def log_op(p1, p2):
			p2 = tf.clip_by_value(p2, 1e-8, tf.reduce_max(p2))
			return p1 * (tf.log(p1) - tf.log(p2))
		return tf.reduce_mean(log_op(rho, rho_hat) + log_op(1 - rho, 1 - rho_hat))


class AIDE(object):
	def __init__(self, name=None, save_folder=None):
		self.name = name or 'AIDE'
		self.init_path(save_folder)


	def init_path(self, save_folder):
		self.SAVE_FOLDER = save_folder or self.name
		os.makedirs(self.SAVE_FOLDER, exist_ok=True)
		self.MODEL_PATH = self.SAVE_FOLDER + os.sep + 'model.ckpt'
		self.CONFIG_PATH = self.SAVE_FOLDER + os.sep + 'config.json'
		self.LOG_PATH = self.SAVE_FOLDER + os.sep + 'log'
		self.HISTORY_PATH = self.SAVE_FOLDER + os.sep + 'history.json'

		self.EMBEDDING_NPY = self.SAVE_FOLDER + os.sep + 'embedding.npy'
		self.EMBEDDING_TXT = self.SAVE_FOLDER + os.sep + 'embedding.txt'
		self.LOSS_FIG_PATH = self.SAVE_FOLDER + os.sep + 'loss.png'


	def __del__(self):
		if hasattr(self, 'sess'):
			self.sess.close()


	def get_feed_dict(self, ph, data):
		return None if ph is None else {ph: data}


	def fit_transform(self, X, config=None, from_last=False):
		"""
		Args:
			X (array-like or tuple):
				If X is array-like, if should has shape (n_samples, n_features).
				If X is tuple, it should be ((str, str), dict) where str represents path of file with '.tfrecord' postfix
				or path of file folder containing '.tfrecord' files. The 2 strs refer to data for training and
				prediction (generating embedding). The dict looks like {'n_samples': int, 'n_features': int, 'issparse': bool}.
				As for the format of '.tfrecord' file, See write_csr_to_tfrecord and write_ary_to_tfrecord in utils_tf.py for details.
				Note that the training data stored in '.tfrecord' need to be shuffled ahead of time. And only 'float32'
				is supported for '.tfrecord' data
			config (DistAEConfig or None):
				If config is set to None, default config will be used when from_last is False. Or the newest saved config
				will be used.
				If config is given, if will be used no matter from_last is True or False.
			from_last (bool):
				if set to False, model will be trained from scratch.
				If set to True, model will be trained by loading the newest saved model located in self.SAVE_FOLDER.
		Returns:
			np.ndarray: (n_samples, config.mds_units[-1])
		"""
		self.g = tf.Graph()
		with self.g.as_default():
			return self._fit_transform(X, config, from_last)


	def _fit_transform(self, X, c=None, from_last=False):
		X = self.check_input_X(X)
		logger = get_logger(self.name, logPath=self.LOG_PATH, mode='a' if from_last else 'w')
		c = self.get_config(X, c, from_last); logger.info(c)

		self.build(c)
		saver = tf.train.Saver()
		self.sess = sess = tf.Session(config=get_sess_config(), graph=self.g)
		if from_last:   # no need to run init_op
			logger.info('Loading from last...')
			saver.restore(self.sess, self.MODEL_PATH)
			history = self.load_history()
		else:
			history = self.init_history()
			sess.run(self.train_model.init_op)

		self.train_feed = self.pred_feed = self.get_train_feed(X, c)
		self.eval_feed = self.get_eval_feed(X, c)

		sess.run(self.train_data_init_op, feed_dict=self.get_feed_dict(self.train_data_ph, self.train_feed))
		self.pretrain(sess, logger, c)

		min_val_loss = np.inf; val_no_improve = 0; global_step = 0
		for i in range(1, c.max_step_num+1):
			_, global_step= sess.run([self.train_model.train_op, self.train_model.global_step])

			if c.verbose and i % c.print_freq == 0:
				loss, all_loss = sess.run([self.train_model.loss, self.train_model.all_loss])
				logger.info('Step {}({:.4}%); Global Step {}: Batch Loss={}; [Reconstruct, MDS, L2] Loss = {}'.format(
					i, 100 * i/c.max_step_num, global_step, loss, all_loss))
				history['step_list'].append(int(global_step)); history['loss_list'].append(float(loss))

			if i % c.val_freq == 0:
				val_loss, all_val_loss = self.get_validate_loss(sess)
				if val_loss < min_val_loss:
					min_val_loss = val_loss
					val_no_improve = 0
				else:
					val_no_improve += 1
					if c.early_stop_patience is not None and global_step > c.min_step_num and val_no_improve >= c.early_stop_patience:
						logger.info('No improve = {}, early stop!'.format(val_no_improve))
						break
				if c.verbose:
					logger.info('Step {}({:.4}%); Global Step {}: Validation Loss={}; [Reconstruct, MDS, L2] Loss = {}; Min Val Loss = {}; No Improve = {}; '.format(
						i, 100 * i / c.max_step_num, global_step, val_loss, all_val_loss, min_val_loss, val_no_improve))
				history['val_step_list'].append(int(global_step)); history['val_loss_list'].append(float(val_loss))
			#if i % c.draw_freq == 0:
				#self.draw_history(self.LOSS_FIG_PATH, history)

		logger.info('Training end. Total step = {}'.format(global_step))
		self.save(c, history, sess, saver, logger, save_model=c.save_model)
		delete_logger(logger)
		return self.get_embedding(sess)


	def pretrain(self, sess, logger, c):
		logger.info('Pretrain begin============================================')
		for i in range(1, c.pretrain_step_num + 1):
			sess.run(self.train_model.pretrain_op)
			if i % c.print_freq == 0:
				reconstruct_loss = sess.run(self.train_model.reconstruct_loss)
				logger.info(
					'Step {}({:.4}%): Batch Loss={}'.format(i, 100 * i / c.pretrain_step_num, reconstruct_loss))
		logger.info('Pretrain end.============================================')


	def build(self, config):
		with tf.name_scope(TRAIN_MODE):
			with tf.variable_scope('Model'):
				self.train_data, self.train_data_init_op, self.train_data_ph = self.get_train_data(config)
				self.train_model = AIDEModel(TRAIN_MODE, config, self.train_data)
		with tf.name_scope(EVAL_MODE):
			with tf.variable_scope('Model', reuse=True):
				self.eval_data, self.eval_data_init_op, self.eval_data_ph = self.get_eval_data(config)
				self.eval_model = AIDEModel(EVAL_MODE, config, self.eval_data)
		with tf.name_scope(PREDICT_MODE):
			with tf.variable_scope('Model', reuse=True):
				self.pred_data, self.pred_data_init_op, self.pred_data_ph = self.get_predict_data(config)
				self.pred_model = AIDEModel(PREDICT_MODE, config, self.pred_data)


	def get_embedding(self, sess=None):
		"""
		Args:
			sess (tf.Session)
		Returns:
			np.ndarray: (cell_num, embed_size)
		"""
		sess = sess or self.sess
		sess.run(self.pred_data_init_op, feed_dict=self.get_feed_dict(self.pred_data_ph, self.pred_feed))
		embed_list = []
		try:
			while True:
				embed_list.append(sess.run(self.pred_model.mds_h))
		except tf.errors.OutOfRangeError:
			pass
		return np.vstack(embed_list)


	def ds_to_el_op(self, ds):
		it = ds.make_initializable_iterator()
		return it.get_next(), it.initializer


	def get_train_mat_data(self, config):
		X_ph = get_placeholder(config.issparse, config.dtype, (None, config.n_features))
		ds = tf.data.Dataset.from_tensor_slices(X_ph).shuffle(config.n_samples).repeat().batch(config.batch_size*2)
		ds = ds.map(train_data_map, num_parallel_calls=tf.data.experimental.AUTOTUNE)
		ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
		return ds, X_ph


	def get_eval_mat_data(self, config):
		X_ph = get_placeholder(config.issparse, config.dtype, (None, config.n_features))
		ds = tf.data.Dataset.from_tensor_slices(X_ph).shuffle(config.validate_size).batch(config.batch_size * 2)
		ds = ds.map(train_data_map, num_parallel_calls=tf.data.experimental.AUTOTUNE)
		ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
		return ds, X_ph


	def get_pred_mat_data(self, config):
		X_ph = get_placeholder(config.issparse, config.dtype, (None, config.n_features))
		ds = tf.data.Dataset.from_tensor_slices(X_ph).batch(config.embed_batch_size)
		ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
		return ds, X_ph


	def get_parse_fn(self, config):
		return tfrow_to_sparse_tensor if config.issparse else tfrow_to_dense_tensor


	def get_train_tfrecord_data(self, config):
		shuffle_buffer_size = min(config.n_samples, config.train_shuffle_buffer)
		parse_fn = self.get_parse_fn(config)
		tfpath = config.train_tfrecord_path
		if os.path.isdir(config.train_tfrecord_path):
			ds = tf.data.Dataset.list_files(os.path.join(tfpath, '*.tfrecord')).interleave(
				tf.data.TFRecordDataset, cycle_length=config.train_interleave_cycle, num_parallel_calls=tf.data.experimental.AUTOTUNE)
		else:
			ds = tf.data.TFRecordDataset([tfpath])
		ds = ds.map(parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
		ds = ds.shuffle(shuffle_buffer_size).repeat().batch(config.batch_size*2)
		ds = ds.map(train_data_map, num_parallel_calls=tf.data.experimental.AUTOTUNE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
		return ds, None


	def get_eval_tfrecord_data(self, config):
		shuffle_buffer_size = min(config.validate_size, config.train_shuffle_buffer)
		parse_fn = self.get_parse_fn(config)
		tfpath = config.train_tfrecord_path
		file_list = get_file_list(tfpath, filter=lambda p:p.endswith('.tfrecord')) if os.path.isdir(tfpath) else [tfpath]
		ds = tf.data.TFRecordDataset(file_list).map(parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
		ds = ds.take(config.validate_size).shuffle(shuffle_buffer_size).batch(config.batch_size*2)
		ds = ds.map(train_data_map, num_parallel_calls=tf.data.experimental.AUTOTUNE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
		return ds, None


	def get_pred_tfrecord_data(self, config):
		parse_fn = self.get_parse_fn(config)
		tfpath = config.pred_tfrecord_path
		file_list = get_file_list(tfpath, filter=lambda p:p.endswith('.tfrecord')) if os.path.isdir(tfpath) else [tfpath]
		ds = tf.data.TFRecordDataset(file_list).map(parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
		ds = ds.batch(config.embed_batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
		return ds, None


	def get_train_data(self, config):
		if config.feed_type == DATA_MAT:
			ds, ph = self.get_train_mat_data(config)
		else:
			ds, ph = self.get_train_tfrecord_data(config)
		it = ds.make_initializable_iterator()
		return it.get_next(), it.initializer, ph


	def get_eval_data(self, config):
		if config.feed_type == DATA_MAT:
			ds, ph = self.get_eval_mat_data(config)
		else:
			ds, ph = self.get_eval_tfrecord_data(config)
		it = ds.make_initializable_iterator()
		return it.get_next(), it.initializer, ph


	def get_predict_data(self, config):
		if config.feed_type == DATA_MAT:
			ds, ph = self.get_pred_mat_data(config)
		else:
			ds, ph = self.get_pred_tfrecord_data(config)
		it = ds.make_initializable_iterator()
		return it.get_next(), it.initializer, ph


	def get_train_feed(self, X, config):
		return x_to_input(X) if config.feed_type == DATA_MAT else None


	def get_eval_feed(self, X, config):
		if config.feed_type == DATA_MAT:
			X = X[np.random.choice(X.shape[0], config.validate_size * 2)]
			if config.issparse:
				X.sort_indices()
			return x_to_input(X)
		return None


	def get_validate_loss(self, sess):
		sess.run(self.eval_data_init_op, feed_dict=self.get_feed_dict(self.eval_data_ph, self.eval_feed))
		loss_list = []
		recon_loss_list, mds_loss_list, l2_loss_list = [], [], []
		try:
			while True:
				loss, (recon_loss, mds_loss, l2_loss) = sess.run([self.eval_model.loss, self.eval_model.all_loss])
				loss_list.append(loss); recon_loss_list.append(recon_loss)
				mds_loss_list.append(mds_loss); l2_loss_list.append(l2_loss)
		except tf.errors.OutOfRangeError:
			pass
		return np.mean(loss_list), [np.mean(recon_loss_list), np.mean(mds_loss_list), np.mean(l2_loss_list)]


	def save(self, config, history, sess, saver, logger=None, save_model=True):
		if save_model:
			path = saver.save(sess, self.MODEL_PATH)
			if logger is not None:
				logger.info('Model saved in path: {}'.format(path))
		config.save(self.CONFIG_PATH)
		self.save_history(history)


	def init_history(self):
		return {
			'step_list': [],
			'loss_list': [],
			'val_step_list': [],
			'val_loss_list': []
		}


	def save_history(self, history):
		json.dump(history, open(self.HISTORY_PATH, 'w'))


	def load_history(self):
		return json.load(open(self.HISTORY_PATH))


	def get_data_feed_type(self, X):
		return DATA_TFRECORD if isinstance(X, tuple) else DATA_MAT


	def get_config(self, X, config, from_last):
		if config is None and from_last:
			return AIDEConfig(self.CONFIG_PATH)
		config = config or AIDEConfig()
		config.feed_type = x_feed_type = self.get_data_feed_type(X)

		if x_feed_type == DATA_MAT:
			config.n_samples, config.n_features = X.shape
			config.issparse = sp.issparse(X)
			config.dtype = X.dtype.name
		else:
			config.train_tfrecord_path, config.pred_tfrecord_path = X[0]
			info_dict = X[1]
			config.n_samples = info_dict['n_samples']
			config.n_features = info_dict['n_features']
			config.issparse = info_dict['issparse']
			config.dtype = 'float32'
		config.embed_batch_size = min(config.embed_batch_size, config.n_samples)
		return config


	def check_input_X(self, X):
		def legal_tfrecord_path(path):
			if path.endswith('.tfrecord'):
				return True
			if os.path.isdir(path):
				for p in os.listdir(path):
					if p.endswith('.tfrecord'):
						return True
			return False
		if isinstance(X, tuple):
			assert len(X) == 2 and len(X[0]) == 2
			(train_tfr, pred_tfr), info_dict = X
			if legal_tfrecord_path(train_tfr) and legal_tfrecord_path(pred_tfr) \
					and 'n_samples' in info_dict and 'n_features' in info_dict and 'issparse' in info_dict:
				return X
			raise RuntimeError('Illegal X: {}'.format(X))
		else:
			X = check_array(X, accept_sparse=True, dtype=[np.float64, np.float32])
			if sp.issparse(X):
				X.sort_indices()
		return X


	def draw_history(self, figpath, history):
		simple_multi_line_plot(
			figpath,
			[history['step_list'], history['val_step_list']],
			[history['loss_list'], history['val_loss_list']],
			line_names=['train_loss', 'val_loss'],
			x_label='Step', y_label='Loss',
			title='Loss Plot')


def train_data_map(X):
	"""
	Args:
		X (tf.tensor or tf.SparseTensor): (batch_size * 2, feature_size)
	Returns:
		tf.tensor or tf.SparseTensor: X
		tf.tensor: dist
	"""
	if type(X) == tf.SparseTensor:
		if len(X.shape) == 3:
			s = tf.shape(X)
			X = tf.sparse.reshape(X, (s[0], s[2]))
		X = tf.sparse_tensor_to_dense(X, validate_indices=False)
	return X


if __name__ == '__main__':
	encoder = AIDE('Test', 'TestFolder')
	X = np.random.rand(1000, 2000)
	embedding = encoder.fit_transform(X, from_last=False)
	print(embedding.shape, type(embedding))



