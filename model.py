
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


class RAWDense(tf.keras.layers.Layer):
  def __init__(self, output_dim ):
    super(RAWDense, self).__init__()
    self.output_dim = output_dim

  def build(self, input_shape):
    self.W = self.add_weight(
        shape=(input_shape[-1], self.output_dim),
        dtype=tf.float32,
        initializer=tf.keras.initializers.glorot_uniform(),
        trainable=True)

    self.b = self.add_weight(
        shape=(self.output_dim,),
        dtype=tf.float32,
        initializer=tf.keras.initializers.ones(),
        trainable=True)


  def call(self, input):
    if len(input.shape) > 3:
        return tf.nn.bias_add(tf.einsum('ablj, jk->ablk', input, self.W), self.b)
    else:
        return tf.nn.bias_add(tf.einsum('abi,ik->abk', input, self.W), self.b)



class RAW(tf.keras.Model):
    '''
    RNN attention walk
    '''

    def __init__(self, FLAGS, num_of_labels):
        '''
        Initializer
        FLAGS -- input parameters
        num_of_labels -- total number of labels
        '''
        super(RAW, self).__init__()

        # For reproducibility
        rn.seed(glo_seed)
        self.rng = np.random.RandomState(seed=glo_seed)
        np.random.seed(glo_seed)
        tf.random.set_seed(glo_seed)

        #Initialize parameters
        self.FLAGS = FLAGS

        # the hyper-parameters
        self.config = OrderedDict()

        #other model parameters 
        self.config['dim_y'] = num_of_labels
        self.FLAGS.transductive = FLAGS.transductive
        
        # place holders
        # self.X = tf.compat.v1.placeholder(tf.int64, shape=(None,))
        self.is_training = tf.Variable(1, trainable=False)
        self.get_path = tf.Variable(0, trainable=False)
        self.gamma = tf.constant(self.FLAGS.gamma)

        self.T = tf.constant(float(self.FLAGS.walk_len))
        self.range = tf.Variable(tf.range(0, self.FLAGS.walk_len, 1, dtype=self.dtype), trainable=False)

    def load_data(self, test_nodes):

        node_features = load_npz("{}/{}_matrices/node_attr.npz".format(self.FLAGS.dataset_dir,self.FLAGS.dataset)).toarray()
        node_features = Normalizer().fit_transform(node_features)
        self.node_features = np.vstack((np.zeros(node_features.shape[-1]), node_features))

        
        # self.node_emb_init= tf.compat.v1.placeholder(tf.float32, shape=self.node_features.shape)
        self.node_emb = tf.Variable(self.node_features, name="edge_emb", dtype=tf.float32, trainable=False)
                
        # self.node_emb = tf.compat.v1.get_variable(name="node_emb", shape=node_features.shape, initializer=tf.compat.v1.constant_initializer(node_features), dtype=self.dtype,
                                            # trainable=False)
        self.config['feature_dim'] = self.node_features.shape[-1]

        if self.FLAGS.has_edge_attr:
            try:
                edge_features = load_npz("{}/{}_matrices/edge_attr.npz".format(self.FLAGS.dataset_dir,self.FLAGS.dataset)).toarray()
                edge_features = np.vstack((np.zeros(edge_features.shape[-1]), edge_features))
                self.edge_features = Normalizer().fit_transform(edge_features)
                # self.edge_emb_init= tf.compat.v1.placeholder(tf.float32, shape=self.edge_features.shape)
                self.edge_emb = tf.Variable(self.edge_features, name="edge_emb", dtype=tf.float32, trainable=False)
                self.has_edge_attr = True

            except:
                print("Error reading the edge attributes. Defaulting to node attributes only")
                self.edge_emb = None
                self.has_edge_attr = False
        else:
            self.edge_emb = None
            self.has_edge_attr = False


        self.setup_lookup(test_nodes) #setup
        self.score = RAWDense(1)
        self.history_zr = RAWDense(self.FLAGS.l_dim*2)
        self.history_h = RAWDense(self.FLAGS.l_dim)
        self.history_zrb = RAWDense(self.FLAGS.l_dim*2)
        self.history_hb = RAWDense(self.FLAGS.l_dim)
        self.classify = RAWDense(self.config['dim_y'])
        self.pre_class = RAWDense(self.FLAGS.l_dim)

        # self.score = tf.keras.layers.Dense(1)
        # self.history_zr = tf.keras.layers.Dense(self.FLAGS.l_dim*2)
        # self.history_h = tf.keras.layers.Dense(self.FLAGS.l_dim)
        # self.history_zrb = tf.keras.layers.Dense(self.FLAGS.l_dim*2)
        # self.history_hb = tf.keras.layers.Dense(self.FLAGS.l_dim)
        # self.pre_class = tf.keras.layers.Dense(self.FLAGS.l_dim)
        # self.classify = tf.keras.layers.Dense(self.config['dim_y'])
    
        self.cost = self.__cost
        self.get_embd = self.get_embbeding
        self.get_pt = self.__get_walk
        self.pred = self.__classify
        self.acc = self.__accuracy


    

    def setup_lookup(self, test_nodes):
        '''
        Setup the lookup tables required for smooth run
        '''

        edgelist = open("{}/{}.edgelist".format(self.FLAGS.dataset_dir,self.FLAGS.dataset), "rU").read().split("\n")[:-1]
        neighbor = {}
        edge_tensor = [0,0]
        iter = 2
        neighbor[0] = [[], [[0, 1]]]
        neighbor[0] = [[[0, 1]], []]
        for edge in edgelist:
            edgei = edge.split('\t')
            s, t = map(int, edgei[:2])

            s, t = s+1, t+1
            if t in neighbor:
                neighbor[t][1].append([iter])
            else:
                neighbor[t] = [[],[[iter]]]

            iter += 1
            if s in neighbor:
                neighbor[s][0].append([iter])
            else:
                neighbor[s] = [[[iter]], []]


            edge_tensor.extend((s,t))
            iter += 1

        edges_per_node = np.zeros((len(neighbor), self.FLAGS.max_neighbors))
        for key, value in neighbor.items():
            value[0] = np.array(value[0])
            value[1] = np.array(value[1])
            half = int(self.FLAGS.max_neighbors / 2)
            if value[0].shape[0] > 0:
                if value[0].shape[0] <= half:
                    edges_per_node[key, :value[0].shape[0]] = value[0][:, 0]
                    space = self.FLAGS.max_neighbors - value[0].shape[0]
                    if value[1].shape[0] > 0:
                        others = value[1][:, 0]
                        if others.shape[0] >= space:
                            others_samp = self.rng.choice(others, size=space, replace=False)
                            edges_per_node[key, value[0].shape[0]:value[0].shape[0] + space] = others_samp
                        else:
                            edges_per_node[key, value[0].shape[0]:value[0].shape[0] + others.shape[0]] = others
                else:
                    rank = value[0][:, 0]
                    samp = self.rng.choice(rank, size=half, replace=False)
                    cur = np.setdiff1d(rank, samp).tolist()
                    edges_per_node[key, :half] = samp
                    if value[1].shape[0] < 1:
                        edges_per_node[key, half:half+len(cur)] = cur[:half]
                    else:
                        others = value[1][:, 0]
                        if others.shape[0] >= half:  
                            others_samp = self.rng.choice(others, size=half, replace=False)
                            edges_per_node[key, half:] = others_samp
                        else:
                            edges_per_node[key, half:(half + others.shape[0])] = others
                            space = self.FLAGS.max_neighbors - (half + others.shape[0])
                            space = len(cur) if len(cur) < space else space
                            edges_per_node[key, (half + others.shape[0]):(half + others.shape[0])+space] = cur[:space]
                    
            elif value[1].shape[0] > 0:
                others = value[1][:, 0]
                if others.shape[0] >= self.FLAGS.max_neighbors:
                    others_samp = self.rng.choice(others, size=self.FLAGS.max_neighbors, replace=False)
                    edges_per_node[key, :] = others_samp
                else:
                    edges_per_node[key, :others.shape[0]] = others
        # with tf.device('/cpu:0'):
        if self.FLAGS.transductive:
            test_mask = edges_per_node > 0
        else:
            test_mask = ~np.isin(np.array(edge_tensor)[edges_per_node.astype(np.int32)], test_nodes)
            null_neighboors = edges_per_node > 0
            test_mask = np.logical_and(test_mask, null_neighboors)

        self.edges_per_node_arr,self.edge_tensor_arr,self.test_mask_arr = np.array(edges_per_node), np.array(edge_tensor), np.array(test_mask)
       
        self.edges_per_node = tf.convert_to_tensor(self.edges_per_node_arr, name="edges_per_node", dtype=tf.int32)
        self.edge_tensor = tf.convert_to_tensor(self.edge_tensor_arr, name="edge_tensor", dtype=tf.int32)
        self.test_mask = tf.convert_to_tensor(self.test_mask_arr, name="test_mask", dtype=tf.bool)
              

    #@tf.function
    def sample_neighbor_walk(self, current_x, current_emb, h, t, reuse=tf.compat.v1.AUTO_REUSE):
        '''
            current_x -- current nodes (v^t)
            current_emb -- current node feature embedding (x^t)
            h -- history context
            t -- time step

            Sample the next nodes to visit (Step procedure)
        '''
        neighbors = tf.gather(self.edges_per_node, current_x)
        mask_neighbors = tf.gather(self.test_mask, current_x)
        mask = tf.cond(tf.math.equal(self.is_training,1), lambda : mask_neighbors, lambda : tf.greater(neighbors,0 ))
        # neighbor_emb = tf.nn.embedding_lookup(self.edge_emb,tf.div(neighbors,2))
        neighbor_node_emb = tf.nn.embedding_lookup(self.node_emb, tf.gather(self.edge_tensor, neighbors))
        current_emb = tf.tile(tf.expand_dims(current_emb, 2), [1,1, self.FLAGS.max_neighbors, 1])
        if self.has_edge_attr:
            neighbor_edge_emb = tf.nn.embedding_lookup(self.edge_emb,tf.cast(tf.math.ceil(tf.divide(neighbors,2)), tf.int32))
            neighbor_emb_act = tf.add_n((neighbor_edge_emb, current_emb, neighbor_node_emb))
        else:
            neighbor_emb_act = tf.add_n((current_emb, neighbor_node_emb))

        h = tf.tile(tf.expand_dims(h, 2), [1,1, self.FLAGS.max_neighbors, 1])
        att_emb = tf.concat((h, neighbor_emb_act), -1)
        # neighbor_node_emb = tf.nn.embedding_lookup(self.node_emb, tf.gather(self.edge_tensor, neighbors))

        neighbors_weight = tf.squeeze(tf.keras.backend.hard_sigmoid(self.score(att_emb)),-1)
        neighbors_weight = tf.multiply(neighbors_weight,tf.cast(mask, tf.float32))

        filter_neighbors = tf.greater_equal(neighbors_weight, 0.5)
        mask2 = tf.logical_and(filter_neighbors, mask)
        neighbors_weight = tf.math.divide_no_nan(neighbors_weight ,  tf.reduce_sum(input_tensor=neighbors_weight, axis=-1, keepdims=True))
        next_id_sample = tf.expand_dims(tfp.distributions.Categorical(probs=neighbors_weight).sample(),-1)
        # next_id_sample = tf.expand_dims(tf.math.argmax(neighbors_weight, -1),-1)

        next_id = tf.gather(neighbors, next_id_sample, batch_dims=-1)
        next_id= tf.nn.embedding_lookup(self.edge_tensor, next_id)

        # neighbor_emb = tf.reshape(tf.gather(neighbor_node_emb, next_id_sample, batch_dims=-1), [self.FLAGS.num_walks,-1,self.config['feature_dim']])
        # print(next_id_sample, neighbor_node_emb, neighbor_emb)
        # exit()
        neighbor_emb = tf.reduce_sum(tf.multiply(neighbor_node_emb, tf.expand_dims(tf.cast(mask2, tf.float32), -1)),2)
        is_sample_masked = tf.gather(mask, next_id_sample, batch_dims=-1)
        non_isolated_nodes = tf.logical_and(tf.reduce_any(mask, -1), tf.squeeze(is_sample_masked,-1))

        next_id = tf.add(tf.multiply(tf.squeeze(next_id,-1),tf.cast(non_isolated_nodes, tf.int32)) , tf.multiply(current_x,tf.cast(~non_isolated_nodes, tf.int32)))

        pi = tf.squeeze(tf.gather(neighbors_weight, next_id_sample, batch_dims=-1),[-1])#tf.add(tf.squeeze(self.gather_colsv2(neighbors_weight, next_id_sample),[-1]) , tf.cast(~non_isolated_nodes, tf.float32))
        pi = tf.multiply(tf.math.pow(self.FLAGS.gamma, (self.T - t)), -tf.math.log(tf.clip_by_value(pi,1e-10,1.0)))

        return tf.expand_dims(next_id,-1), neighbor_emb, tf.expand_dims(pi,-1)


    #@tf.function
    def GRU(self, trueX):


        #@tf.function
        def forward(input, t):
            """Perform a forward pass.

            Arguments
            ---------
            h_tm1: np.matrix
                The hidden state at the previous timestep (h_{t-1}).
            x_t: np.matrix
                The input vector.
            """

            h_tm1 = input[:,:,:self.FLAGS.l_dim]
            x = tf.cast(input[:,:,self.FLAGS.l_dim], tf.int32)
            h_tm1 = tf.cond(tf.math.equal(self.is_training,1), lambda : h_tm1 * self.dropout_recurrent, lambda : h_tm1)

            x_t = tf.nn.embedding_lookup(self.node_emb, x)
            next_x, c_t, likelihood = self.sample_neighbor_walk(x, x_t, h_tm1, t)
            x_t = tf.add(x_t, c_t)



            zr_t = tf.keras.backend.hard_sigmoid(self.history_zr(tf.concat([x_t, h_tm1],-1)))
            z_t, r_t = tf.split(value=zr_t, num_or_size_splits=2, axis=-1)
            r_state = r_t * h_tm1
            h_proposal = tf.tanh(self.history_h(tf.concat([x_t, r_state],-1)))


            # Compute the next hidden state
            h_t = tf.multiply(1 - z_t, h_tm1) + tf.multiply(z_t, h_proposal)
            return tf.concat([h_t, tf.cast(next_x, self.dtype), likelihood, x_t],-1)


        dummy_emb = tf.tile(tf.expand_dims(tf.cast(trueX, self.dtype),-1), [1,1,self.config['feature_dim']])
        shape = dummy_emb.get_shape().as_list()
        h_0 = tf.matmul(dummy_emb, tf.zeros(dtype=tf.float32, shape=(shape[0], self.config['feature_dim'], self.FLAGS.l_dim)),
                        name='h_0' )
        next_x0 = tf.expand_dims(tf.cast(trueX, self.dtype),-1)
        concat_tensor = tf.concat([h_0, next_x0, next_x0, dummy_emb], -1)

        self.dropout_recurrent = tf.nn.dropout(tf.ones_like(h_0[0,:,:]),self.FLAGS.drate)

        h_t = tf.scan(forward, self.range, initializer = concat_tensor,parallel_iterations=20,
                      name='h_t_transposed' )

        # Transpose the result back
        h_t_b = self.BGRU(tf.reverse(h_t[:,:,:,self.FLAGS.l_dim +2:], [0]))
        # ht =  tf.add(h_t[-1,:,:,:self.FLAGS.l_dim] , h_t_b)
        ht =  tf.add(h_t[-1,:,:,:self.FLAGS.l_dim] , h_t_b)
        output = tf.cond(tf.math.equal(self.get_path,1), lambda : tf.transpose(h_t[:,:, :,self.FLAGS.l_dim]), lambda : ht)
        
        return output, tf.reduce_mean(h_t[:-1,:, :,self.FLAGS.l_dim+1],0)


    #@tf.function
    def BGRU(self, node_emb):


        def backward(h_tm1, x_t):
            """Perform a forward pass.

            Arguments
            ---------
            h_tm1: np.matrix
                The hidden state at the previous timestep (h_{t-1}).
            x_t: np.matrix
                The input vector.
            """

            h_tm1 = tf.cond(tf.math.equal(self.is_training,1), lambda: h_tm1 * self.dropout_recurrent_b, lambda: h_tm1)
            zr_t = tf.keras.backend.hard_sigmoid(self.history_zrb(tf.concat([x_t, h_tm1],-1)))
            z_t, r_t = tf.split(value=zr_t, num_or_size_splits=2, axis=-1)
            r_state = r_t * h_tm1
            h_proposal = tf.tanh(self.history_hb(tf.concat([x_t, r_state],-1)))

            # Compute the next hidden state
            h_t = tf.multiply(1 - z_t, h_tm1) + tf.multiply(z_t, h_proposal)
            
            return h_t


        shape = node_emb.get_shape().as_list()
        h_0_b = tf.matmul(node_emb[0, :, :, :], tf.zeros(dtype=tf.float32, shape=(shape[1],self.config['feature_dim'], self.FLAGS.l_dim)),
                          name='h_0_b' )

        self.dropout_recurrent_b = tf.nn.dropout(tf.ones_like(h_0_b[0,:,:]),self.FLAGS.drate)

        h_t_transposed_b = tf.scan(backward, node_emb, initializer = h_0_b,parallel_iterations=20, name='h_t_transposed_b' )

        # shape = node_emb.get_shape().as_list()
        return h_t_transposed_b[-1,:,:,:]

    def __cost(self, trueX, trueY):
        '''
        compute the cost tensor

        trueX -- input X (2D tensor)
        trueY -- input Y (2D tensor)

        return 1D tensor of cost (batch_size)
        '''

        X = tf.expand_dims(trueX, 0)
        X = tf.tile(X, [self.FLAGS.num_walks,1])

        Y = tf.expand_dims(trueY,0)
        Y = tf.tile(Y, [self.FLAGS.num_walks, 1, 1])

        # X -> Z
        Z, pi = self.GRU(X)
        Z = tf.reshape(Z, [self.FLAGS.num_walks, -1,self.FLAGS.l_dim])

        log_pred_Y = self.classify(self.pre_class(Z))
        # regularizer = tf.keras.regularizers.l2(0.01)
        # reg_loss = regularizer(self.classify.W)

        reward = tf.cast(tf.equal(tf.argmax(tf.nn.softmax(log_pred_Y), -1), tf.argmax(Y,-1)), tf.float32)
        # reward = 2*(reward-0.5)
        # reward = (reward*9 + 1)

        _cost = tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=log_pred_Y)

        # print(pi, _cost)
        # exit()
        M_loss = tf.reduce_mean(_cost + (reward * pi))
        # M_loss = tf.reduce_mean(_cost)
        
        return M_loss# + reg_loss


    #@tf.function
    def __classify(self, trueX):
        '''
        classify input 2D tensor

        return 1D tensor (integer class labels)
        '''

        # X -> Z
        X = tf.expand_dims(trueX, 0)
        X = tf.tile(X, [self.FLAGS.num_walks, 1])

        Z, _ = self.GRU(X)
        Z = tf.reshape(Z, [self.FLAGS.num_walks, -1, self.FLAGS.l_dim])
        # Z = tf.reduce_sum(Z,0)
        # Z = tf.layers.dense(Z, self.FLAGS.l_dim, name='Z2Z', activation=self.config['activation'],
        #                     kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01), reuse=tf.compat.v1.AUTO_REUSE)
        log_pred_Y = self.classify(self.pre_class(Z))
        Y = tf.nn.softmax(log_pred_Y, -1)
        Y = tf.reduce_mean(Y, 0)
        return tf.argmax(Y, axis=-1)

    #@tf.function
    def __accuracy(self, trueX, trueY):
        '''
        measure the semi-supervised accuracy

        trueX -- input 2D tensor (features)
        trueY -- input 2D tensor (one-hot labels)

        return a scalar tensor
        '''

        eval_results = []
        # non_zero = tf.count_nonzero(predicted)

        predicted = self.__classify(trueX)
        actual = tf.argmax(trueY, axis=-1)

        corr = tf.equal(predicted, actual)
        acc = tf.reduce_mean(tf.cast(corr, tf.float32))

        return acc

    #@tf.function
    def get_embbeding(self, trueX):
        '''
        classify input 2D tensor

        return 1D tensor (integer class labels)
        '''

        # X -> Z
        X = tf.expand_dims(trueX, 0)
        X = tf.tile(X, [self.FLAGS.num_walks, 1])
        Z, _ = self.GRU(X)
        Z = tf.reshape(Z, [self.FLAGS.num_walks, -1, self.FLAGS.l_dim])
        Z = tf.reduce_mean(Z, 0)
            #
        return Z

    #@tf.function
    def __get_walk(self, trueX):
        '''
        classify input 2D tensor

        return 1D tensor (integer class labels)
        '''
        X = tf.expand_dims(trueX, 0)
        X = tf.tile(X, [self.FLAGS.num_walks, 1])
        walk,_ = self.GRU(X)
        return walk



#