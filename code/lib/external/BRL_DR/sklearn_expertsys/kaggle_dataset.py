import sys
sys.path.append('D:\\Scripts\\BRL\\sklearn-expertsys')

from RuleListClassifier import *
import sklearn.ensemble
from sklearn.cross_validation import train_test_split
from sklearn.datasets.mldata import fetch_mldata
import numpy as np
import os

# data_labels = ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu',
#                'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime', 'failures',
#                'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher',
#                'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc',
#                'Walc', 'health', 'absences']
# data_labels_named = ['School name', 'Gender', 'Age', 'Locality', 'Family size',
#                      'Parents marital status', 'Mothers education', 'Fathers education',
#                      'Mothers job', 'Fathers job', 'Reason to join school',
#                      'Legal guardian', 'Travel time to school', 'Weekly study time',
#                      'Additional school support', 'Additional family support',
#                      'Extra tutoring', 'Extracurricular activities',
#                      'Attended nursery', 'Planning higher education', 'Access to internet',
#                      'Romantic status', 'Quality of family relationships', 'Free time post school',
#                      'Leisure time with friends', 'Daily Alcohol consumption',
#                      'Weekly alcohol consumption', 'Current health', '# of absences',
#                      'Score1', 'Score2', 'Score3']
#
# lookup_table = {
#     'school': {
#         'GP': 0,
#         'MS': 1
#     },
#     'sex': {
#         'F': 0,
#         'M': 1
#     },
#     'address': {
#         'U': 0,
#         'R': 1
#     },
#     'famsize': {
#         'LE3': 0,
#         'GT3': 1
#     },
#     'Pstatus': {
#         'T': 0,
#         'A': 1
#     },
#     'Mjob': {
#         'teacher': 0,
#         'health': 1,
#         'services': 2,
#         'at_home': 3,
#         'other': 4
#     },
#     'Fjob': {
#         'teacher': 0,
#         'health': 1,
#         'services': 2,
#         'at_home': 3,
#         'other': 4
#
#     },
#     'reason': {
#         'home': 0,
#         'reputation': 1,
#         'course': 2,
#         'other': 3
#     },
#     'guardian': {
#         'mother': 1,
#         'father': 2,
#         'other': 3
#     },
#     'schoolsup': {
#         'no': 0,
#         'yes': 1
#     },
#     'famsup': {
#         'no': 0,
#         'yes': 1
#     },
#     'paid': {
#         'no': 0,
#         'yes': 1
#     },
#     'activities': {
#         'no': 0,
#         'yes': 1
#     },
#     'nursery': {
#         'no': 0,
#         'yes': 1
#     },
#     'higher': {
#         'no': 0,
#         'yes': 1
#     },
#     'internet': {
#         'no': 0,
#         'yes': 1
#     },
#     'romantic': {
#         'no': 0,
#         'yes': 1
#     },
# }
#

# def conv_func(pos):
#     return lambda x: lookup_table[data_labels[pos]][x]
#
# converter_dict = {i: conv_func(i) for i in range(len(data_labels)) if data_labels[i] in lookup_table.keys()}
#
# # converter_dict.update({30: lambda x: int(x.replace('"', '')), 31: lambda x: int(x.replace('"', ''))})
#
# print converter_dict


def load_dataset():
    pass

# dataseturls = [
#
# ]

# datasets = [
#
# ]

# data_feature_labels = [
#     ["Sepal length", "Sepal width", "Petal length", "Petal width"],
#     ["#Pregnant", "Glucose concentration demo", "Blood pressure(mmHg)", "Triceps skin fold thickness(mm)", "2-Hour serum insulin (mu U/ml)",
#      "Body mass index", "Diabetes pedigree function", "Age (years)"],
# ]
data_set_path = 'D:\\Scripts\\BRL\\Data sets\\students-academic-performance-dataset\\'

fp = open(os.path.join(data_set_path, 'xAPI-Edu-Data-clean.csv'))
labels = fp.readline().strip().split(',')

print labels
print len(labels)
# data_array = []
# target_array = []

# full_array = np.asarray([line.strip().replace('"', '').split(';') for line in fp.readlines()])
# full_array = np.genfromtxt(fp, delimiter=',',
#                            #    skiprows=1, unpack=True,
#                            names=labels,
#                            dtype=None
#                            # dtype={'names': labels,
#                            #    'formats': ('|S15', '|S15', '|S15', '|S15', '|S15', '|S15', '|S15', '|S15', '|S15', np.float, np.float, np.float, np.float, '|S15', '|S15', '|S15', '|S15')},
#                            # dtype='|S15,|S15,int,|S15,|S15,|S15,int,int,|S15,|S15,|S15,|S15,int,int,int,|S15,|S15,|S15,|S15,|S15,|S15,|S15,|S15,int,int,int,int,int,int,int,int,int,int',
#                            # converters=converter_dict
#                            )

# full_array = np.reshape(full_array, (-1, 2))
# full_array = np.transpose(np.vstack(full_array))
#
#

# for row in full_array:
#     print row[labels[:-1]]
#     raw_input()

fp.seek(0)
data_array = np.genfromtxt(fp, delimiter=',',
                        #    names=labels[:-1],
                        #    usecols=labels[:-1],
                           usecols=range(17)[:-1],
                           dtype=None, skip_header=1)
# data_array = (np.copy(full_array))
# data_array = np.delete(data_array, 16, 1)
# target_array = np.asarray([row[-1] for row in full_array])
# fp = open(os.path.join(data_set_path, 'xAPI-Edu-Data-clean.csv'))
fp.seek(0)
target_array = np.genfromtxt(fp, delimiter=',',
                            #  names=labels[-1],
                            #  usecols=labels[-1],
                             usecols=(-1),
                             dtype=None, skip_header=1)

target_array[np.where(target_array == 'H')] = 1
target_array[np.where(target_array == 'L')] = 0
target_array = target_array.astype('int')

#
# target_array[np.where(target_array != 0)] = 1
#
# print data_array[1:5, :]
# print target_array
# raw_input()
#
# # target_array = np.sum(target_array, axis=1)
#
# # threshold = 0.65
# #
# # target_array[np.where(target_array <= threshold * 60)] = 0
# # target_array[np.where(target_array > threshold * 60)] = 1
# #
# # print target_array
# # print np.asarray(full_array)
# # for data_line in fp.readlines():
# #     data_row = data_line.strip().split(';')
# #
# #     data_row = map(lambda x: x.replace('"', ''), data_row)
# #     # for label, val in zip(labels, data_row):
# #     #     print label, '-', val
# #     #
# #     # print data_row
# #     parsed_row = []
# #     for index, label in enumerate(labels):
# #         if label in lookup_table.keys():
# #             parsed_row.append(lookup_table[label][data_row[index]])
# #         else:
# #             parsed_row.append(int(data_row[index]))
# #
# #     data_array.append(parsed_row[:-3])
# #     target_array.append(parsed_row[-3:])
# #
# # data_array = np.asarray(data_array)
# # target_array = np.sum(np.asarray(target_array), axis=1)
# #
# # threshold = 0.8
# #
# # print np.where(target_array > threshold * 60)
# # target_array[np.where(target_array <= threshold * 60)] = 0
# # target_array[np.where(target_array > threshold * 60)] = 1
# #
# # # print data_array.astype('float')
# # # print data_array.shape
# # # print target_array
# # # print target_array.shape
# #
# #
# #
# #
print "--------"
# # print "DATASET: ", datasets[i], "(", dataseturls[i], ")"
# # data = fetch_mldata(datasets[i])
# # y = data.target
# # y[y > 1] = 0
# # y[y < 0] = 0
#
d = fetch_mldata('iris')
print d
# print data_array
raw_input()
Xtrain, Xtest, ytrain, ytest = train_test_split(data_array, target_array)
#
# print Xtrain.shape
# print Xtest.shape
#
#

# print Xtrain.shape
# new_data
# print len(data_labels_named)

clf = RuleListClassifier(max_iter=50000, n_chains=3, class1label='Performs well',
                         listlengthprior=6, listwidthprior=3,
                         verbose=True)
clf.fit(Xtrain, ytrain,
        feature_labels=labels[:-1],
        # feature_labels=data_labels,
        # undiscretized_features=['School name', 'Gender', 'Age', 'Locality', 'Family size',
        #                      'Parents marital status', 'Mothers education', 'Fathers education',
        #                      'Mothers job', 'Fathers job', 'Reason to join school',
        #                      'Legal guardian', 'Additional school support', 'Additional family support',
        #                      'Extra tutoring', 'Extracurricular activities',
        #                      'Attended nursery', 'Planning higher education', 'Access to internet',
        #                      'Romantic status', 'Quality of family relationships', 'Free time post school',
        #                      'Leisure time with friends', 'Daily Alcohol consumption',
        #                      'Weekly alcohol consumption', 'Current health', '# of absences']
        )

print Xtrain.shape
print ytrain.shape
print Xtest.shape
print ytest.shape

print "rules:\n", clf
print "accuracy:", clf.score(Xtest, ytest)
print "Random Forest accuracy:", sklearn.ensemble.RandomForestClassifier().fit(Xtrain, ytrain).score(Xtest, ytest)

# data_class1_labels = ["Iris Versicolour", "No Diabetes"]
# for i in range(len(datasets)):
#     print "--------"
#     print "DATASET: ", datasets[i], "(", dataseturls[i], ")"
#     data = fetch_mldata(datasets[i])
#     y = data.target
#     y[y > 1] = 0
#     y[y < 0] = 0
#
#     Xtrain, Xtest, ytrain, ytest = train_test_split(data.data, y)
#
#     clf = RuleListClassifier(max_iter=50000, n_chains=3, class1label=data_class1_labels[i], verbose=False)
#     clf.fit(Xtrain, ytrain, feature_labels=data_feature_labels[i])
#
#     print "rules:\n", clf
#     print "accuracy:", clf.score(Xtest, ytest)
#     print "Random Forest accuracy:", sklearn.ensemble.RandomForestClassifier().fit(Xtrain, ytrain).score(Xtest, ytest)
