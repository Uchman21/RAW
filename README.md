# Recurrent Attention Walk for Semi-supervised Classification

### Overview

This directory contains code necessary to run the RAW algorithm.
RAW is an algorithm for semi-supervised multi-class node classification under the framework of reinforcement learning, which aims at integrating information from both labeled and unlabeled nodes into the learning process of node embeddings in attributed graphs. The learned node
embeddings are used to conduct both transductive and inductive multi-class node classification.

See our [paper](https://arxiv.org/pdf/1910.10266.pdf) for details on the algorithm.

The dblp directory contains the preprocessed dblp data used in our experiments.
The raw dblp and delve datasets (used in the paper) will be provided soon ().

If you make use of this code or the MLGW algorithm in your work, please cite the following paper:

	@inproceedings{akujuclass,
	     author = {Akujuobi, Uchenna and Zhang, Qiannan and Yufei, Han and Zhang, Xiangliang},
	     title = {Recurrent Attention Walk for Semi-supervised Classification},
	     booktitle = {The Thirteenth ACM International Conference on Web Search and Data Mining},
	     year = {2020}
	  }

### Requirements

Recent versions of TensorFlow, numpy, scipy, sklearn are required. You can install all the required packages using the following command:

	$ pip install -r requirements.txt


### Running the code

Use `python RAW.py` to run using default settings. The parameters can be changed by passing during the command call (e.g., `python RAW.py --verbose`). Use `python RAW.py --help` to display the parameters.

#### Input format
The RAW model handles dataset with/without edge attributes but assumes no missing node attribute. Therefore, at minimums, the code requires that a `--dataset_dir` option is specified which specifies the following data files:

* <dataset_dir>/\<dataset>.edgelist -- An edgelist file describing the input graph. This is a two columned file with format `<source node>\t<target node>`.
* <dataset_dir>/\<dataset>_matrices/node_attr.npz -- A numpy-stored array of node features ordered according to the node index in the edgelist file.
* <dataset_dir>/\<dataset>_matrices/edge_attr.npz [optional] -- A numpy-stored array of edge features if available; ordered according to the edge appearance in the edgelist file.
* <dataset_dir>/\<dataset>_matrices/label.npz -- A numpy-stored binary array of node labels; ordered according to the node index in the edgelist file. A zero vector is used to represent unlabeled nodes. For instance [[0,1],[1,0],[0,0]] the last entry represents an unlabeled node.

To run the model on a new dataset, you need to make data files in the format described above.

#### Using the outputs

To save the node embeddings and walk paths, please use the `--save_emb` and `--save_path` options respectively. The ouput will be stored in an "output" folder.


