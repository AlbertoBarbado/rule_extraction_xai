from RuleListClassifier import *
from sklearn.datasets.mldata import fetch_mldata
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pprint
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn import svm

data = pd.DataFrame.from_csv('D:\\Scripts\\BRL\\Data sets\\turkiye-student-evaluation_generic.csv', index_col=None, sep=',')

features = list(data.columns[:2]) + list(data.columns[3:])
class1label = data.columns[2]

feature_data = data[features]
target_data = data[class1label]
# raw_input()
target_data[target_data > 1] = 1
# target_data = target_data.map({'L': 0, 'H': 1})
print target_data.head(100)
target_data = np.asarray(target_data)
print 'Size of dataset:', target_data.shape
raw_input('Press enter to continue...')

Xtrain, Xtest, ytrain, ytest = train_test_split(feature_data, target_data)  # split

# train classifier (allow more iterations for better accuracy; use BigDataRuleListClassifier for large datasets)
model = RuleListClassifier(max_iter=10000, class1label='performs well', verbose=False)
svm_model = svm.SVC(kernel='linear')
svm_model_2 = svm.SVC(kernel='rbf')
rf_model = RandomForestClassifier()

print '*' * 80
# print "RuleListClassifier Accuracy:", model.score(Xtest, ytest), "Learned interpretable model:\n", model
# print "RandomForestClassifier Accuracy:", RandomForestClassifier().fit(Xtrain, ytrain).score(Xtest, ytest)

num_partitions = 20

scores = cross_val_score(model, feature_data, target_data, cv=num_partitions)
print 'BRL accuracy'
print scores
print 'BRL average accuracy'
print '%f (+/- %f)' % (np.mean(scores), np.std(scores))
print 'Rules:'
model.fit(Xtrain, ytrain, feature_labels=features)
print model

# svm_scores = cross_val_score(svm_model, feature_data, target_data, cv=num_partitions)
# print 'SVM accuracy (linear)'
# print svm_scores
# print 'SVM average accuracy'
# print '%f (+/- %f)' % (np.mean(svm_scores), np.std(svm_scores))
#
# svm_scores_2 = cross_val_score(svm_model_2, feature_data, target_data, cv=num_partitions)
# print 'SVM accuracy (rbf)'
# print svm_scores_2
# print 'SVM average accuracy'
# print '%f (+/- %f)' % (np.mean(svm_scores_2), np.std(svm_scores_2))

rf_scores = cross_val_score(rf_model, feature_data, target_data, cv=num_partitions)
print 'Random forest accuracy (rbf)'
print rf_scores
print 'Random forest average accuracy'
print '%f (+/- %f)' % (np.mean(rf_scores), np.std(rf_scores))


r"""
**Output:**
********************************************************************************
RuleListClassifier Accuracy: 0.56846473029 Learned interpretable model:
Trained RuleListClassifier for detecting Malignant
===================================================
IF Age_geq_60 : 1 AND IrregularShape : 1 THEN probability of Malignant: 86.4% (80.9%-91.0%)
ELSE IF CircumscribedMargin : 1 THEN probability of Malignant: 12.2% (8.6%-16.2%)
ELSE IF OvalShape : 0 THEN probability of Malignant: 65.2% (58.8%-71.3%)
ELSE IF IllDefinedMargin : 0 THEN probability of Malignant: 12.5% (1.7%-31.9%)
ELSE probability of Malignant: 44.8% (27.5%-62.8%)
==================================================

RandomForestClassifier Accuracy: 0.784232365145
"""


r"""
BRL accuracy
[ 0.79591837  0.79166667  0.75        0.75        0.72916667  0.8125
  0.85416667  0.875       0.70833333  0.91666667  0.77083333  0.77083333
  0.75        0.8125      0.75        0.83333333  0.8125      0.875       0.8125
  0.70833333]
BRL average accuracy
0.793963 (+/- 0.055863)

SVM accuracy
[ 0.73469388  0.79591837  0.81632653  0.73469388  0.81632653  0.8125
  0.77083333  0.83333333  0.6875      0.85416667  0.77083333  0.75
  0.66666667  0.70833333  0.77083333  0.77083333  0.78723404  0.87234043
  0.70212766  0.76595745]
SVM average accuracy
0.771073 (+/- 0.053581)

SVM accuracy
[ 0.75510204  0.83673469  0.81632653  0.75510204  0.81632653  0.8125
  0.77083333  0.875       0.6875      0.875       0.79166667  0.75        0.8125
  0.77083333  0.77083333  0.79166667  0.80851064  0.89361702  0.74468085
  0.74468085]
SVM average accuracy
0.793971 (+/- 0.049776)

Random forest accuracy
[ 0.7755102   0.7755102   0.7755102   0.73469388  0.75510204  0.79166667
  0.75        0.85416667  0.70833333  0.85416667  0.83333333  0.77083333
  0.75        0.77083333  0.8125      0.85416667  0.80851064  0.89361702
  0.74468085  0.74468085]
Random forest average accuracy
0.787891 (+/- 0.047617)
"""
