# -*- coding: utf-8 -*-
"""
Created on Wed May 19 20:47:56 2021

@author: alber
"""
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from scipy.io import loadmat  # this is the SciPy module that loads mat-files
from common.config import PATH_DATASETS, PATH_RESULTS

def loadDatasets(f_name=""):
    """
    Function for loading one of the different datasets with the proper structure,
    as well as indicating whiche feature columns are numerical and which are
    categorical.

    Parameters
    ----------
    f_name : string, optional
        name of the dataset to load. The default is "".

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    df_input : dataframe
        dataframe with the input data, where each column represents one
        feature, and the final column, "target_class", represents whether
        the datapoint is an outlier (-1) or an inlier (1).
        Example:
                 col_0     col_1     col_2  target_class
            0  0.095310  7.095976  5.796362             1
            1  0.095310  7.426013  5.808443             1
            2 -2.302585  6.886634  5.799396            -1

    numerical_cols : list
        list of numerical columns from the total features.
    categorical_cols : list
        list of categorical columns from the total features..

    """

    list_datasets = [
        "forest_cover",
        "annthyroid",
        "mammography",
        "arrhythmia",
        "smtp",
        "seismic-bumps",
        "shuttle",
        "speech",
    ]

    if f_name not in list_datasets:
        msg_err = "Dataset {0} not included - use one from {1}".format(
            f_name, list_datasets
        )
        raise ValueError(msg_err)
        
    if f_name == "forest_cover":
        # Source dataset: http://odds.cs.stonybrook.edu/forestcovercovertype-dataset/
        file_raw = loadmat("{0}/cover.mat".format(PATH_DATASETS))
        X, y = file_raw["X"], file_raw["y"]
        df_input = pd.DataFrame(X)
        numerical_cols = df_input.columns
        numerical_cols = ["col_{0}".format(column) for column in list(df_input.columns)]
        df_input.columns = numerical_cols
        df_input["target_class"] = y
        df_input["target_class"] = df_input["target_class"].astype(int)
        df_input["target_class"] = df_input.apply(
            lambda x: -1 if x["target_class"] == 1 else 1, axis=1
        )
        categorical_cols = []
        
    elif f_name == "annthyroid":
        # Source dataset: http://odds.cs.stonybrook.edu/annthyroid-dataset/
        file_raw = loadmat("{0}/annthyroid.mat".format(PATH_DATASETS))
        X, y = file_raw["X"], file_raw["y"]
        df_input = pd.DataFrame(X)
        numerical_cols = df_input.columns
        numerical_cols = ["col_{0}".format(column) for column in list(df_input.columns)]
        df_input.columns = numerical_cols
        df_input["target_class"] = y
        df_input["target_class"] = df_input["target_class"].astype(int)
        df_input["target_class"] = df_input.apply(
            lambda x: -1 if x["target_class"] == 1 else 1, axis=1
        )
        categorical_cols = []
        
    elif f_name == "mammography":
        # Source dataset: http://odds.cs.stonybrook.edu/mammography-dataset/
        file_raw = loadmat("{0}/mammography.mat".format(PATH_DATASETS))
        X, y = file_raw["X"], file_raw["y"]
        df_input = pd.DataFrame(X)
        numerical_cols = df_input.columns
        numerical_cols = ["col_{0}".format(column) for column in list(df_input.columns)]
        df_input.columns = numerical_cols
        df_input["target_class"] = y
        df_input["target_class"] = df_input["target_class"].astype(int)
        df_input["target_class"] = df_input.apply(
            lambda x: -1 if x["target_class"] == 1 else 1, axis=1
        )
        categorical_cols = []
        
    elif f_name == "arrhythmia":
        # Source dataset: http://odds.cs.stonybrook.edu/arrhythmia-dataset/
        file_raw = loadmat("{0}/arrhythmia.mat".format(PATH_DATASETS))
        X, y = file_raw["X"], file_raw["y"]

        df_input = pd.DataFrame(X)
        df_input.columns = [
            "col_{0}".format(column) for column in list(df_input.columns)
        ]
        categorical_cols = [
            column
            for column in list(df_input.columns)
            if list(df_input[column].unique()) == [0.0, 1.0]
        ]
        numerical_cols = [
            column
            for column in list(df_input.columns)
            if column not in categorical_cols
        ]
        df_input["target_class"] = y
        df_input["target_class"] = df_input["target_class"].astype(int)
        df_input["target_class"] = df_input.apply(
            lambda x: -1 if x["target_class"] == 1 else 1, axis=1
        )
        
    elif f_name == "smtp":
        # Source dataset: http://odds.cs.stonybrook.edu/smtp-kddcup99-dataset/
        file_raw = h5py.File("{0}/smtp.mat".format(PATH_DATASETS))
        arrays = {}
        for k, v in file_raw.items():
            arrays[k] = np.array(v)
        X, y = arrays["X"].T, arrays["y"].T
        df_input = pd.DataFrame(X)
        numerical_cols = df_input.columns
        numerical_cols = ["col_{0}".format(column) for column in list(df_input.columns)]
        df_input.columns = numerical_cols
        df_input["target_class"] = y
        df_input["target_class"] = df_input["target_class"].astype(int)
        df_input["target_class"] = df_input.apply(
            lambda x: -1 if x["target_class"] == 1 else 1, axis=1
        )
        categorical_cols = []
        
    elif f_name == "seismic-bumps":
        # Source dataset (I): http://odds.cs.stonybrook.edu/seismic-dataset/
        # Source dataset (II): https://archive.ics.uci.edu/ml/datasets/seismic-bumps#
        # Hazard is the anomalous class: outlier if hazard>0

        # Load data
        file_raw = pd.read_csv("{0}/seismic-bumps.csv".format(PATH_DATASETS))

        # Label Encoding
        categorical_cols = ["seismic", "seismoacoustic", "shift", "hazard"]
        for column in categorical_cols:
            le = LabelEncoder()
            le.fit(file_raw[column])
            file_raw[column] = le.transform(file_raw[column])
        # Onehot encoding
        onehot_cols = ["seismoacoustic"]
        categorical_cols = [x for x in categorical_cols if x not in onehot_cols]
        for column in onehot_cols:
            y = pd.get_dummies(
                file_raw[column], prefix="{0}".format(column), drop_first=True
            )
            y = y.astype(int)
            file_raw = file_raw.drop(columns=[column], errors="ignore")
            file_raw[y.columns] = y
            categorical_cols += list(y.columns)
        # Define target column
        df_input = file_raw.copy()
        df_input = df_input.rename(columns={"hazard": "target_class"})
        categorical_cols = [column for column in categorical_cols if column != "hazard"]
        numerical_cols = [
            column
            for column in list(df_input.columns)
            if column not in categorical_cols + ["target_class"]
        ]

        # Rearrange columns
        df_input = df_input[numerical_cols + categorical_cols + ["target_class"]]

        # Specify target class value
        df_input["target_class"] = df_input.apply(
            lambda x: -1 if x["target_class"] > 0 else 1, axis=1
        )
        
    elif f_name == "shuttle":
        # Source dataset: http://odds.cs.stonybrook.edu/shuttle-dataset/
        file_raw = loadmat("{0}/shuttle.mat".format(PATH_DATASETS))
        X, y = file_raw["X"], file_raw["y"]

        df_input = pd.DataFrame(X)
        df_input.columns = [
            "col_{0}".format(column) for column in list(df_input.columns)
        ]
        categorical_cols = []
        numerical_cols = [
            column
            for column in list(df_input.columns)
            if column not in categorical_cols
        ]
        df_input["target_class"] = y
        df_input["target_class"] = df_input["target_class"].astype(int)
        df_input["target_class"] = df_input.apply(
            lambda x: -1 if x["target_class"] == 1 else 1, axis=1
        )
        df_input[numerical_cols] = df_input[numerical_cols].astype(float)
        
    elif f_name == "speech":
        # Source dataset: http://odds.cs.stonybrook.edu/speech-dataset/
        file_raw = loadmat("{0}/speech.mat".format(PATH_DATASETS))
        X, y = file_raw["X"], file_raw["y"]
        df_input = pd.DataFrame(X)
        df_input.columns = [
            "col_{0}".format(column) for column in list(df_input.columns)
        ]
        categorical_cols = []
        numerical_cols = [
            column
            for column in list(df_input.columns)
            if column not in categorical_cols
        ]
        df_input["target_class"] = y
        df_input["target_class"] = df_input["target_class"].astype(int)
        df_input["target_class"] = df_input.apply(
            lambda x: -1 if x["target_class"] == 1 else 1, axis=1
        )
        df_input[numerical_cols] = df_input[numerical_cols].astype(float)
        
    return df_input, numerical_cols, categorical_cols
        

def plot_2D(df_rules, df_anomalies, title=""):
    """
    Function to plot a 2D figure with the anomalies and the hypercubes. df_rules
    should be in a format like the one returned in turn_rules_to_df() but without
    np.inf values (they can be set instead to an arbitrary big/low enough value).

    Parameters
    ----------
    df_rules : TYPE
        DESCRIPTION.
    df_anomalies : TYPE
        DESCRIPTION.
    folder : TYPE, optional
        DESCRIPTION. The default is "".
    path_name : TYPE, optional
        DESCRIPTION. The default is "".

    Returns
    -------
    None.

    """
    
    ### Plot 2D
    plt.figure(figsize=(12, 8))
    
    # Add hypercubes
    for i in range(len(df_rules)):
        # Create a Rectangle patch
        x_1 = df_rules.iloc[i:i + 1]['gdenergy_min'].values[0]
        x_2 = df_rules.iloc[i:i + 1]['gdenergy_max'].values[0]
        y_1 = df_rules.iloc[i:i + 1]['gdpuls_min'].values[0]
        y_2 = df_rules.iloc[i:i + 1]['gdpuls_max'].values[0]
    
        # Add the patch to the Axes
        rect = patches.Rectangle(
            (x_1, y_1),
            x_2 - x_1,
            y_2 - y_1,
            linewidth=3,
            edgecolor='black',
            facecolor='none',
            zorder=15)
        currentAxis = plt.gca()
        currentAxis.add_patch(rect)
    
    # Plot points
    plt.plot(
        df_anomalies[df_anomalies['predictions'] == 1]['gdenergy'],
        df_anomalies[df_anomalies['predictions'] == 1]['gdpuls'],
        'o',
        color="blue",
        label='not anomaly',
        zorder=10)
    plt.plot(
        df_anomalies[df_anomalies['predictions'] == -1]['gdenergy'],
        df_anomalies[df_anomalies['predictions'] == -1]['gdpuls'],
        'o',
        color='red',
        label='anomaly',
        zorder=10)
    plt.legend(loc='upper left')
    plt.xlabel('gdenergy', fontsize=12)
    plt.ylabel('gdpuls', fontsize=12)
    
    x1, x2, y1, y2 = plt.axis()
    plt.axis((x1, x2, y1, y2))
    plt.title("Anomalies and Rules - {0}".format(title))