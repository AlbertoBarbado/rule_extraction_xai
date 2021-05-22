# -*- coding: utf-8 -*-
"""
Created on Tue May 18 20:03:21 2021

@author: alber
"""

import numpy as np
import pandas as pd
import os
import sys
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
from lib.xai_auxiliary_rule_extraction import generateRuleHypercubes
from lib.xai_rule_metrics import (checkRepresentativeness, 
                                  checkStability,
                                  checkDiversity)
from lib.xai_tools import plot_2D, loadDatasets

import warnings
warnings.filterwarnings("ignore")

# Load data
df_input, _, _ = loadDatasets(f_name = "seismic-bumps")
numerical_cols = ["gdenergy", "gdpuls"]  # 2D
categorical_cols = []
df_input = (df_input[numerical_cols + categorical_cols + ['target_class']]
            .drop_duplicates(subset = numerical_cols + categorical_cols)
            )

# Train model & Get predictions
df_train = df_input.copy().drop(columns=['target_class'])
clf = IsolationForest()
y_pred_train = clf.fit_predict(df_train)
dist = clf.decision_function(df_train)
df_anomalies = df_input.copy()
df_anomalies["predictions"] = y_pred_train
df_anomalies["dist"] = dist
df_anomalies["score"] = clf.decision_function(df_train)

# Paths
path_folder = "pruebasIF"
file_template = "DT_IF"

# Get Rules
method = "DecisionTree"
method = "RuleFit"
method = "FRL"
method = "SkopeRules"
method = "DecisionRuleList"
method = "brlg"
method = "logrr"

for method in [
    "DecisionTree",
    "RuleFit",
    "FRL",
    "SkopeRules",
    "DecisionRuleList",
    "brlg",
    "logrr",
]:
    print("Iter for method: ", method)
    
    ## 1. Get Rules (Hypercube)
    # Define hyperparams for the XAI model
    model_params = {}
    # Generate rules hypercubes
    df_rules_inliers, df_rules_outliers = generateRuleHypercubes(
        df_anomalies = df_anomalies,
        numerical_cols = numerical_cols,
        categorical_cols = categorical_cols,
        method = method,
        simplify_rules = True,
        model_params = model_params,
    )
    
    ## 2. Plot Results
    max_replace = df_input.max().max() + np.abs(df_input.max().max()) * 0.1
    min_replace = df_input.min().min() - np.abs(df_input.min().min()) * 0.1
    df_inliers_plot = (
        df_rules_inliers.copy()
        .replace(np.inf, max_replace)
        .replace(-np.inf, min_replace)
    ) 
    df_outliers_plot = (
        df_rules_outliers.copy()
        .replace(np.inf, max_replace)
        .replace(-np.inf, min_replace)
    )

    plot_2D(df_inliers_plot, df_anomalies, title = method + ' (Inliers)')
    plot_2D(df_outliers_plot, df_anomalies, title = method + ' (Outliers)')
    
    ## 3. Get Metrics
    # Precision (unsupervised model predictions vs ground truth)
    target_names = ['outliers', 'inliers']
    y_true = df_anomalies['target_class']
    y_pred = df_anomalies['predictions']
    print(classification_report(y_true, y_pred, target_names=target_names))
    
    # Comprehensibility
    n_rules_inliers = len(df_rules_inliers)
    n_rules_outliers = len(df_rules_outliers)
    mean_rule_size_inliers = df_rules_inliers['size_rules'].mean()
    mean_rule_size_outliers = df_rules_outliers['size_rules'].mean()
    
    # Representativeness (vs ground truth) [a.k.a XAI "precision"]
    df_rules_inliers_gt, df_rules_outliers_gt = checkRepresentativeness(
        df_anomalies, df_rules_inliers, df_rules_outliers, numerical_cols, 
        categorical_cols, col_predictions = "target_class"
        )
    
    # Representativeness (vs model)
    df_rules_inliers, df_rules_outliers = checkRepresentativeness(
        df_anomalies, df_rules_inliers, df_rules_outliers, numerical_cols, 
        categorical_cols, col_predictions = "predictions"
        )
    
    # Stability
    df_rules_inliers = checkStability(df_anomalies, df_rules_inliers, clf,
                                      numerical_cols, categorical_cols,
                                      using_inliers = True)
    df_rules_outliers = checkStability(df_anomalies, df_rules_outliers, clf,
                                       numerical_cols, categorical_cols,
                                       using_inliers = False)
   
    # Diversity
    df_rules_inliers, _ = checkDiversity(df_rules_inliers, 
                                         numerical_cols, 
                                         categorical_cols)
    df_rules_outliers, _ = checkDiversity(df_rules_outliers, 
                                          numerical_cols, 
                                          categorical_cols)
    
    
