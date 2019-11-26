
# -*- coding: utf-8 -*-

'''
LICENSE: BSD 2-Clause

Summer 2017
'''

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os, time
os.environ['PYTHONHASHSEED'] = '2018'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

glo_seed = 2018

import random as rn
import tensorflow as tf
import numpy as np
from scipy.sparse import hstack, csr_matrix, vstack
from collections import OrderedDict
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from scipy.sparse import load_npz, save_npz
from sklearn.preprocessing import Normalizer
from sklearn.metrics.pairwise import cosine_distances
import tensorflow_probability as tfp
from model import RAW



class Trainer():
    '''
    RNN attention walk
    '''

    def __init__(self, FLAGS, num_of_labels):
        '''
        Initializer
        FLAGS -- input parameters
        num_of_labels -- total number of labels
        '''

        # For reproducibility
        rn.seed(glo_seed)
        self.rng = np.random.RandomState(seed=glo_seed)
        np.random.seed(glo_seed)
        tf.random.set_seed(glo_seed)

        #Initialize parameters
        self.dtype = tf.float32
        self.FLAGS = FLAGS

        # the hyper-parameters
        self.config = OrderedDict()
        self.config['l_dim'] = FLAGS.l_dim
        self.config['walk_len'] = FLAGS.walk_len
        self.config['lrate'] = FLAGS.lrate
        self.config['max_neighbors'] = FLAGS.max_neighbors
        self.config['gamma'] = FLAGS.gamma

        #other model parameters 
        self.config['num_walks'] = FLAGS.num_walks
        self.config['dim_y'] = num_of_labels
        self.config['transductive'] = FLAGS.transductive
        
        # identifier
        self.id = ('{0}_{1}_{2}_{3}_{4}'.format(
            self.FLAGS.dataset,
            self.config['walk_len'],
            self.config['l_dim'],
            self.config['max_neighbors'],
            time.time()))

        self.RAW_model = RAW(FLAGS, num_of_labels)

    def __str__(self):
        '''
        report configurations
        '''
        msg = []
        for key in self.config:
            msg.append('{0:15}:{1:>20}\n'.format(key, self.config[key]))
        return '\n'.join(msg)


    def fetch_batches(self, allidx, nbatches, batchsize, wind=True):
        '''
        allidx  -- 1D array of integers
        nbatches  -- #batches
        batchsize -- mini-batch size

        split allidx into batches, each of size batchsize
        '''
        N = allidx.size

        for i in range(nbatches):
            if wind:
                idx = [(_ % N) for _ in range(i * batchsize, (i + 1) * batchsize)]
            else:
                idx = [_ for _ in range(i * batchsize, (i + 1) * batchsize) if (_ < N)]

            yield allidx[idx]

    @tf.function
    def train_step(self, trueX, trueY):
        with tf.GradientTape() as tape:
            batch_cost = self.RAW_model.cost(trueX,trueY)
            # batch_cost += tf.add_n(self.RAW_model.losses)
            grads = tape.gradient( batch_cost, self.RAW_model.trainable_variables)
            self.train_op.apply_gradients(zip(grads, self.RAW_model.trainable_variables))
            
            return batch_cost

    def build_eval(self, labels, samples):


        train_mask, u_mask, test_mask = samples
        self.RAW_model.load_data(samples[-1])
        
        if self.FLAGS.save_path or self.FLAGS.save_emb:
            if not os.path.exists('output'):
                os.mkdir('output')
            train_mask = np.hstack((train_mask, u_mask, test_mask))
            save_npz('output/{}_idx'.format(self.id), csr_matrix(train_mask))
            if self.FLAGS.verbose: print("Indecies saved in file: %s" % "output/{}".format(self.FLAGS.dataset))
            allidx = np.arange(train_mask.shape[0])
            Y_train = labels[train_mask, :]
            nbatches = int(np.ceil(train_mask.shape[0] / self.FLAGS.batchsize))
        else:
            allidx = np.arange(train_mask.shape[0])
            Y_train = labels[train_mask, :]
            Y_test = labels[test_mask, :]
            Y_unlabeled = labels[u_mask, :]
            nbatches = int(np.ceil(train_mask.shape[0] / self.FLAGS.batchsize))

        #TF config setup
        # config = tf.compat.v1.ConfigProto()
        # config.gpu_options.allow_growth = True
        self.train_op = tf.keras.optimizers.Adam(self.config['lrate'])


        self.RAW_model.is_training.assign(1)
        self.RAW_model.get_path.assign(0)
        for e in range(self.FLAGS.epochs):
            tf.random.shuffle(allidx)

            epoch_cost = 0
            
            for batchl in self.fetch_batches(allidx, nbatches, self.FLAGS.batchsize):
                epoch_cost += self.train_step(train_mask[batchl],Y_train[batchl, :].toarray())
            
            epoch_cost /= nbatches
            if self.FLAGS.verbose:
                cost_only = True
                if cost_only:
                    print('[{0:5d}] E={1:.8f}\n'.format(e, epoch_cost))
                else:
                    # #output after each epoch
                    train_acc = self.acc(u_mask[0:self.batch_size],Y_u[0:self.batch_size, :].toarray())

                    print( '[{0:5d}]  TrainE={1:.4f}  TrainAcc={2:.4f}'.format( e, epoch_cost, train_acc ))

        if self.FLAGS.save_path:
            self.__save_path(train_mask, self.FLAGS.batchsize)
            if self.FLAGS.verbose: print("Path saved in file: %s" % "output/{}_node".format(self.FLAGS.dataset))
            exit()
        elif self.FLAGS.save_emb:
            self.get_emb(train_mask,self.FLAGS.batchsize)
            if self.FLAGS.verbose: print("Embeddings saved in file: %s" % "output/{}_emb".format(self.FLAGS.dataset))
            exit()
        else:
            predY = self.prediction(test_mask, largebatchsize=self.FLAGS.batchsize)
            Y_test = Y_test.toarray()

            return accuracy_score(Y_test.argmax(-1), predY)


    def get_emb(self, X, largebatchsize=200):
        '''
        X     -- 2D feature matrix (n_samples x DX)
        Y     -- 2D one-hot matrix (n_samples x DY) or 1D labels (n_samples)
        onehot -- True: Y is one-hot
                  False: Y is integer labels

        test the classsfication accuracy
        '''
        self.RAW_model.is_training.assign(0)
        self.RAW_model.get_path.assign(0)
        embX = []  # np.zeros(Y.shape)

        allidx = np.arange(len(X))
        nbatches = int(np.ceil(len(X) / largebatchsize))

        for batch in self.fetch_batches(allidx, nbatches, largebatchsize, wind=False):
            embX.append(self.RAW_model.get_embd(X[batch]))

        emb = np.vstack(embX)


    def __save_path(self, X, largebatchsize=200):
        '''
        X     -- 2D feature matrix (n_samples x DX)
        Y     -- 2D one-hot matrix (n_samples x DY) or 1D labels (n_samples)
        Save learned walk path
        '''
        self.RAW_model.is_training.assign(0)
        self.RAW_model.get_path.assign(1)
        pathX = [] 
        allidx = np.arange(len(X))
        nbatches = int(np.ceil(len(X) / largebatchsize))

        for batch in self.fetch_batches(allidx, nbatches, largebatchsize, wind=False):
            pathX.append(self.RAW_model.get_pt(X[batch]))

        path_node = np.vstack(pathX)
        np.save("output/{}_node".format(self.FLAGS.dataset), path_node)


    def prediction(self,X, largebatchsize=500):
        '''
        X     -- 2D feature matrix (n_samples x DX)
        Y     -- 2D one-hot matrix (n_samples x DY) or 1D labels (n_samples)
        predict node labels
        '''
        self.RAW_model.is_training.assign(0)
        self.RAW_model.get_path.assign(0)
        predY = []  # np.zeros(Y.shape)

        allidx = np.arange(len(X))
        nbatches = int(np.ceil(len(X) / largebatchsize))

        for batch in self.fetch_batches(allidx, nbatches, largebatchsize, wind=False):
            predY.append(self.RAW_model.pred(X[batch]))
        predY = np.hstack(predY).reshape(-1, 1)
        return predY