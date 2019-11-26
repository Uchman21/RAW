from __future__ import division
from __future__ import print_function

import os
os.environ['PYTHONHASHSEED'] = '2018'
glo_seed = 2018

import time
from scipy import sparse as sp
import random as rn
import itertools, tqdm
import subprocess
import tensorflow as tf
import numpy as np
import gc
from absl import app
from absl import flags

rn.seed(glo_seed)
np.random.seed(glo_seed)
tf.random.set_seed(glo_seed)

rng = np.random.RandomState(seed=glo_seed)
from trainer import Trainer


# Settings
FLAGS = flags.FLAGS

flags.DEFINE_string('dataset','dblp','Dataset name')
flags.DEFINE_string('dataset_dir','dblp/','Dataset directory.')
flags.DEFINE_float('lrate',  0.001, 'Learning rate.')
flags.DEFINE_float('drate',  0.0, 'Dropout rate.')
flags.DEFINE_float('gamma',  0.9, 'Reinforcement learning reward discount rate')
flags.DEFINE_integer('l_dim', 128, 'Dimension size of the latent vectors. Default is 128.')
flags.DEFINE_integer('batchsize', 128,'Size of batch input. Default is 32.')
flags.DEFINE_integer('max_neighbors', 40, 'maximum node neighbors to consider per step.')
flags.DEFINE_integer('num_walks', 10,'Number of walks per source. Default is 3.')
flags.DEFINE_integer('walk_len',10,'Number of nodes to tranverse per walk. Default is 20.')
flags.DEFINE_boolean('transductive', True, 'Boolean specifying if to train a transductive model. Default is False.')
flags.DEFINE_integer('epochs', 10, 'number of training epoch (10 default).')
flags.DEFINE_boolean('save_emb',False, 'Boolean specifying if to save trained embeddings only. Default is False.')
flags.DEFINE_boolean('save_path',False, 'Boolean specifying if to save trained paths only. Default is False.')
flags.DEFINE_boolean('verbose',True, 'display all outputs. Default is False.')
flags.DEFINE_boolean('has_edge_attr',True, 'The dataset has edge attributes. Default is True.')
flags.DEFINE_float('train_ratio',0.3, ' ratio of dataset to use as train set. Default is 0.5.')


def train_test_split( y, ratio):
	label = y.toarray()
	labeled_set = np.where(label.sum(-1) > 0)[0]
	train_mask = rng.choice(labeled_set, size=int(len(labeled_set) * ratio), replace=False)
	cur = np.setdiff1d(labeled_set,train_mask, assume_unique=True)
	test_mask = rng.choice(cur, size=int(len(labeled_set) * 0.3), replace=False)
	val_mask = np.setdiff1d(cur,test_mask, assume_unique=False)
	
	return train_mask, val_mask, test_mask #, u_mask


def main(args):
	'''
	Pipeline for representational learning for nodes in a graph.
	'''
	global_best = -1
	gb_config = []
	benchmark()


def benchmark():
	

	acc_nn, err = 0 , 0
	num_runs = 2
	labels = sp.load_npz("{}/{}_matrices/label.npz".format(FLAGS.dataset_dir,FLAGS.dataset)).astype(np.int32)
	labels = sp.vstack((np.zeros(labels.shape[-1]), labels)).tocsr()
	num_of_labels = labels.shape[1]
	dim_y = labels.shape[1]
	for i in range(num_runs):
		gc.collect()
		r_walk = Trainer(FLAGS, num_of_labels)
		if i == 0:
			print(r_walk)
		samples = train_test_split(labels, FLAGS.train_ratio)  
		acc = r_walk.build_eval(labels,samples)
		acc_nn += acc
		print("iter [{}] of [{}] : Acc {}".format(i, num_runs, acc))
		del r_walk

	print("test_acc_{} = {}".format(FLAGS.epochs, acc_nn/num_runs))
	# out_file.close()






if __name__ == "__main__":
	app.run(main)
