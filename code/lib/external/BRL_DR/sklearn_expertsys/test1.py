from RuleListClassifier import *
from sklearn.datasets.mldata import fetch_mldata
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pprint
import numpy as np

feature_labels = ["#_times_Pregnant", "Glucose_concentration_test", "Blood_pressure(mmHg)", "Triceps_skin_fold_thickness(mm)",
                  "2-Hour_serum_insulin(mu_U/ml)", "Body_mass_index", "Diabetes_pedigree_function", "Age_(years)"]

data = fetch_mldata("diabetes")  # get dataset
# print data
# raw_input()

print data.data[0]
raw_input()
y = (data.target + 1) / 2  # target labels (0 or 1)
for i in range(10):
    print '=' * 80
    for key, val in zip(feature_labels, data.data[i]):
        print key, ':', val
    print 'Target:', y[i]

print '=' * 80

# print dir(y)
# print type(y)
# print np.count_nonzero(np.where(y == 1))
# print data['data'][0]
# print len(data['label'])

print data.data
print data.data.shape
print y
print y.shape

raw_input('Press enter to continue...')

Xtrain, Xtest, ytrain, ytest = train_test_split(data.data, y)  # split

# train classifier (allow more iterations for better accuracy; use BigDataRuleListClassifier for large datasets)
model = RuleListClassifier(max_iter=10000, class1label="diabetes", verbose=True)
model.fit(Xtrain, ytrain, feature_labels=feature_labels)

print "RuleListClassifier Accuracy:", model.score(Xtest, ytest), "Learned interpretable model:\n", model
print "RandomForestClassifier Accuracy:", RandomForestClassifier().fit(Xtrain, ytrain).score(Xtest, ytest)
"""
**Output:**
RuleListClassifier Accuracy: 0.776041666667 Learned interpretable model:
Trained RuleListClassifier for detecting diabetes
==================================================
IF Glucose concentration test : 157.5_to_inf THEN probability of diabetes: 81.1% (72.5%-72.5%)
ELSE IF Body mass index : -inf_to_26.3499995 THEN probability of diabetes: 5.2% (1.9%-1.9%)
ELSE IF Glucose concentration test : -inf_to_103.5 THEN probability of diabetes: 14.4% (8.8%-8.8%)
ELSE IF Age (years) : 27.5_to_inf THEN probability of diabetes: 59.6% (51.8%-51.8%)
ELSE IF Glucose concentration test : 103.5_to_127.5 THEN probability of diabetes: 15.9% (8.0%-8.0%)
ELSE probability of diabetes: 44.7% (29.5%-29.5%)
=================================================

RandomForestClassifier Accuracy: 0.729166666667
"""
