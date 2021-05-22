DESCRIPTION

This is the code implements the Falling Rule Lists algorithm described at the paper posted at http://arxiv.org/abs/1411.5899.

A falling rule list is a classifier in the form of a decision list, where the risk probability associated with each node decreases monotonically down the list.  An example can be found in the paper.

Right now, the algorithm uses fpgrowth to mine for rules.  As fpgrowth accepts only *binary* feature vectors for data, this code as is supports only binary feature vectors.  However as the algorithm works with any rule mining algorithm, we intend to add support for rule mining algorithms that accept scalar feature in the future.


INSTALLATION

This code requires fpgrowth, which is available as part of the pyFim package, available here: http://www.borgelt.net/pyfim.html

It also requires the standard Python data analysis stack: Numpy, Scipy, Pandas.

Please make sure that the folder monotonic in the same folder as this README file is placed in a folder that is on your python path.

USAGE

The classifier interface mimics that of the classifiers in the scikit-learn library.  An example data file uci_mammo_data.csv has been provided, as well as an example script test_script.py that trains the falling rule list algorithm on that dataset.  The available options are illustrated in the script.  Right now, only a constant temperature simulated annealing schedule is supported.
