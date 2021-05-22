import monotonic.monotonic.sklearn_wrappers as sklearn_wrappers
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split

import os

os.chdir(os.path.dirname(__file__))
data_file = './uci_mammo_data.csv'

data = pd.DataFrame.from_csv(data_file, index_col=None)

X = np.array(data.iloc[:, 0:-1])
y = np.array(data.iloc[:, -1])

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y)  # split

feature_names = data.columns

fitter = sklearn_wrappers.monotonic_sklearn_fitter(num_steps=5000, min_supp=5, max_clauses=2, prior_length_mean=8, prior_gamma_l_alpha=1., prior_gamma_l_beta=0.1, temperature=1)

# predictor = fitter.fit(Xtrain, ytrain, feature_names)
# predictor = fitter.fit(X,y) the feature_names argument is optional, but then you won't know the features used in each rule

# training_scores = predictor.decision_function(Xtest)

# print 'the MAP rule list'
# print predictor.train_info

# print 'some predictions:'

# print 'Accuracy:', predictor.score(Xtest, ytest)

from sklearn.model_selection import cross_val_score

scores = cross_val_score(fitter, X, y, cv=5)
print scores

r'''
    gamma     logprob           overall_support       positive_proportion     rule  \
0   1.614118  -96.351510        0.239334              85.2174                     8
1   1.587302  -33.620564        0.136316              78.1250                    30
2   1.298969  -24.072429        0.152966              69.2308                    16
3   2.632857  -100.490342       0.444329              63.3987                     9
4   4.763158  -42.317318        0.088450              39.6825                    39
5   0.138122  -152.285265            NaN              12.1359               default
6       NaN   -449.137429            NaN                  NaN                    43

                        rule_features  support      positive_proportion
0    ['IrregularShape', 'Age_geq_60']    230.0      85.2174
1  ['SpiculatedMargin', 'Age_geq_45']     64.0      78.1250
2  ['IllDefinedMargin', 'Age_geq_60']     39.0      69.2308
3                  ['IrregularShape']    153.0      63.3987
4   ['LobularShape', 'Density_geq_2']     63.0      39.6825
5                             default    412.0      12.1359
6                                 NaN    961.0          NaN

average accuracy:
0.763059313215
'''
