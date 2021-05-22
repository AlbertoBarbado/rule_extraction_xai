# -*- coding: utf-8 -*-
"""
Created on Tue May 18 21:25:39 2021

@author: alber
"""

import sys, os
import pandas as pd
import numpy as np
import time
import pickle

import six

sys.modules["sklearn.externals.six"] = six

from joblib import Parallel, delayed
from itertools import combinations, permutations, product
from shapely.geometry import Polygon
from sklearn.preprocessing import StandardScaler
from aix360.algorithms.protodash import ProtodashExplainer

N_JOBS = 1


def checkPointInside(
    data_point, df_rules, numerical_cols, categorical_cols, check_opposite=True
):
    """
    1 for the hypercubes where it's inside, 0 for when not. It checks differently
    whether its for scenarios where the rules are independent according to the
    different combination of categorical variables or whether everything is analyzed
    alltogether.

    Parameters
    ----------
    data_point : dataframe (row)
        One row from the dataframe.
    df_rules : dataframe
        Dataframe with the inlier or outlier rules. The features should appear as columns,
        indicating the maximum/minimum values associated to the vertices of the
        hypercubes. 
        For instance:
               gdenergy_max  gdenergy_min  gdpuls_max  gdpuls_min
        0         -13.5         -79.5       -64.5       -74.5
        1          11.0         -74.5        57.5       -64.5
    numerical_cols : list
        list with the numerical features.
    categorical_cols : list
        list with the caategorical feartures.
    check_opposite : TYPE
        It indicates whether to consider datapoints with >=/<= or strict >/<.
        Since we will see the rules in a counterfactual way (p.e what should
        happen for an outlier to be an inlier) we consider the datapoints of the
        target rules with >=/<=, and the ones from the other class as >/< (that
        means that we consider rules with P=1 even if they have points from the
        other class on the edges) [NOT USED]
    Returns
    -------
    df_plot : TYPE
        DESCRIPTION.

    """
    df_plot = df_rules.copy()

    if len(df_rules) == 0:
        df_plot["check"] = 0
        return df_plot
    # Default value
    df_plot["check"] = 1

    # Check for categorical
    if len(categorical_cols) > 0:
        for col in categorical_cols:
            value = data_point[col]
            df_plot["check"] = df_plot["check"] * (
                df_plot.apply(lambda x: 1 if (x[col] == value) else 0, axis=1)
            )
    # Check for numerical
    if len(numerical_cols) > 0:
        for col in numerical_cols:
            value = data_point[col]
            if check_opposite:
                df_plot["check"] = df_plot["check"] * (
                    df_plot.apply(
                        lambda x: 1
                        if ((x[col + "_max"] >= value) & (value >= x[col + "_min"]))
                        else 0,
                        axis=1,
                    )
                )
            else:
                df_plot["check"] = df_plot["check"] * (
                    df_plot.apply(
                        lambda x: 1
                        if ((x[col + "_max"] > value) & (value > x[col + "_min"]))
                        else 0,
                        axis=1,
                    )
                )
    return df_plot[["check"]]


def checkStability(
    df_anomalies, df_rules, model, numerical_cols, categorical_cols, using_inliers
):
    """
    Function that computes the "stability" metrics of the hypercubes.
    First, it obtains the prototypes from the dataset and generates random samples
    near them.

    Then, it obtains the prediction of the original model for those dummy samples
    and checks if when the prediction is inlier/outlier, there is at least one rule
    that includes that datapoint within it.

    Model agreement:
        - Choose N prototypes that represent the original hyperspace of data
        - Generate M samples close to each of those N prototypes; the hypothesis
        is that close points should be generally predicted belonging to the same class
        - For each of those N*M datapoints (M datapoints per each N prototype) check
        whether the rules (all of them) predict them as inliner or outlier; the datapoints
        that come into the function are either outliers or inliers. If they are inliers,
        then the rules identify an artificial datapoint (of those M*N) as inlier if it
        is outside every rule. If the datapoints are outliers it's the same reversed: a
        datapoint is an inlier if no rule includes it.
        - The % of predictions for the artificial datapoints aforementioned that
        are the same between the rules and the original OCSVM model.

    Parameters
    ----------
    df_anomalies : dataframe
        A dataframe with the original features, along with one column ("anomalies") 
        that indicates whether the datapoint is outlier (-1) or inlier (1),
        the distance to the decision function ("dist"), and the scoring,
        which is essentially the same as the distance.
        For instance:
             gdenergy  gdpuls  anomalies      dist  predictions     score
        0       -72     -72          1  0.022756            1  0.022756
        1       -70     -79         -1 -0.003314           -1 -0.003314
        
    df_rules : dataframe
        Dataframe with the inlier or outlier rules. The features should appear as columns,
        indicating the maximum/minimum values associated to the vertices of the
        hypercubes. 
        For instance:
               gdenergy_max  gdenergy_min  gdpuls_max  gdpuls_min
        0         -13.5         -79.5       -64.5       -74.5
        1          11.0         -74.5        57.5       -64.5
    model : binary
        trained model used as binary classifier/anomaly detection.
    numerical_cols : list
        list with the numerical features.
    categorical_cols : list
        list with the caategorical feartures.
    using_inliers : boolean
        True if df_rules are associated to the inliers, and False if they are
        associated to outliers.

    Returns
    -------
    df_rules : TYPE
        The original dataframe with a column named 'stability score' with 
        the corresponding metric.

    """

    # Ignore prints in this function
    ff = open(os.devnull, "w")
    xx = sys.stdout  # save sys.stdout
    sys.stdout = ff

    if len(df_rules) == 0:
        df_rules["precision_vs_model"] = 0
        df_rules["rules_agreement"] = 0

        return df_rules
    # Choose the type of datapoints and define params
    label = 1 if using_inliers else -1
    df_data = df_anomalies[df_anomalies["predictions"] == label].copy()
    n_samples = np.round(len(df_rules))
    n_samples = n_samples if n_samples > 20 else 20  # at least 20 samples

    if n_samples > len(df_data):
        n_samples = len(df_data)
    df_rules_aux = df_rules.copy()
    df_anomalies_aux = df_anomalies.copy()

    # Scaling
    if len(numerical_cols):
        sc = StandardScaler()
        sc.fit_transform(df_anomalies[numerical_cols])
        df_anomalies_aux[numerical_cols] = sc.transform(
            df_anomalies_aux[numerical_cols]
        )
        cols_max = [x + "_max" for x in numerical_cols]
        cols_min = [x + "_min" for x in numerical_cols]
    # Generate Prototypes
    explainer = ProtodashExplainer()
    list_cols = numerical_cols + categorical_cols
    (W, S, _) = explainer.explain(
        df_data[list_cols].values,
        df_data[list_cols].values,
        m=n_samples,
        kernelType="Gaussian",
        sigma=2,
    )
    df_prototypes = df_anomalies[df_anomalies.index.isin(list(S))][
        list_cols
    ].reset_index(drop=True)

    # Generate artificial samples around the prototypes
    df_samples_total = pd.DataFrame()
    base_size = len(df_anomalies)
    for i, row in df_prototypes.iterrows():
        iter_size = np.round(base_size * W[i])
        iter_size = iter_size if iter_size > 10 else 10
        iter_size = int(np.round(iter_size))
        df_samples = pd.DataFrame(
            {
                col: np.random.uniform(
                    low=row[col] * 0.9, high=row[col] * 1.1, size=(iter_size,)
                )
                for col in numerical_cols
            }
        )
        if len(categorical_cols) > 0:
            for col in categorical_cols:
                df_samples[col] = row[col]
        df_samples["prototype_id"] = i
        df_samples_total = df_samples_total.append(df_samples)
    df_samples_unscaled = df_samples_total.copy()
    df_samples_scaled = df_samples_total.copy()
    df_samples_scaled[numerical_cols] = sc.transform(df_samples_scaled[numerical_cols])

    # Check two things:
    # 1) Classifications are the same for all similar neighbours of a prototype
    # 2) Rules give the same output as the original model
    #    (-1 if not included in any rule, 1 if included in at least one of them)
    list_proto_id = list(df_samples_total["prototype_id"].unique())
    precision_rules = 0
    df_agree = pd.DataFrame()
    j = 0  # Number of datapoints inside the rules
    for proto_id in list_proto_id:
        print(proto_id)
        df_proto_subset_scaled = df_samples_scaled[
            df_samples_scaled["prototype_id"] == proto_id
        ][list_cols]
        df_proto_subset_unscaled = df_samples_unscaled[
            df_samples_unscaled["prototype_id"] == proto_id
        ][list_cols]

        for row_scaled, row_unscaled in zip(
            df_proto_subset_scaled.iterrows(), df_proto_subset_unscaled.iterrows()
        ):
            i = row_scaled[0]
            data_point_scaled = row_scaled[1]
            data_point_unscaled = row_unscaled[1]
            df_aux = pd.DataFrame(
                checkPointInside(
                    data_point_unscaled, df_rules, numerical_cols, categorical_cols
                )["check"]
            )

            # Only if the prediction of this datapoint belongs to the same class...
            rules_prediction = df_aux["check"].max()
            df_agree = df_agree.append(
                pd.DataFrame(
                    {"proto_id": proto_id, "rules_prediction": rules_prediction},
                    index=[0],
                )
            )

            # Check if the predictions are the same as the model
            y_model = model.predict(data_point_scaled.values.reshape(1, -1))[0]
            if using_inliers:
                # Model=inlier, Rules=inlier -> correct
                if df_aux["check"].max() == 1 and y_model == 1:
                    j += 1
                    # If inside any rule, check as correct if the model
                    # also predicted it
                    precision_rules += 1
                # Model=outlier, Rules=Inlier -> incorrect
                elif df_aux["check"].max() == 1:
                    j += 1
                # Model=outlier, Rules=outlier -> correct
                elif df_aux["check"].max() == 0 and y_model == -1:
                    j += 1
                    # If outside any rule, check as correct if the model also
                    # predicted it as outlier
                    precision_rules += 1
                # Model=inlier, Rules=outlier -> incorrect
                elif df_aux["check"].max() == 0 and y_model == 1:
                    j += 1
            else:
                # Model=outlier, Rules=outlier -> correct
                if df_aux["check"].max() == 1 and y_model == -1:
                    j += 1
                    # If inside any rule, check as correct if the model also
                    # predicted it
                    precision_rules += 1
                # Model=inlier, Rules=outlier -> incorrect
                elif df_aux["check"].max() == 1:
                    j += 1
                # Model=inlier, Rules=inlier -> correct
                elif df_aux["check"].max() == 0 and y_model == 1:
                    j += 1
                    # # If outside any rule, check as correct if the model also
                    # predicted it as inlier
                    precision_rules += 1
                # Model=outlier, Rules=Inlier -> incorrect
                elif df_aux["check"].max() == 0 and y_model == -1:
                    j += 1
    rules_0 = (
        df_agree[df_agree["rules_prediction"] == 0]
        .groupby(by=["proto_id"])
        .count()
        .reset_index()
        .rename(columns={"rules_prediction": "rules_0"})
    )
    rules_1 = (
        df_agree[df_agree["rules_prediction"] == 1]
        .groupby(by=["proto_id"])
        .count()
        .reset_index()
        .rename(columns={"rules_prediction": "rules_1"})
    )
    rules_agreement = rules_0.merge(rules_1, how="outer").fillna(0)
    rules_agreement["per_agree"] = rules_agreement.apply(
        lambda x: max([x["rules_0"], x["rules_1"]]) / (x["rules_0"] + x["rules_1"]),
        axis=1,
    )
    j = j if j != 0 else 1
    precision_vs_model = (
        precision_rules / j
    )  # % of points with the same values as the model
    final_agreement = np.round(np.mean(rules_agreement["per_agree"]), 4)

    df_rules[
        "precision_vs_model"
    ] = precision_vs_model  # % of datapoints with the same prediction as the original model
    # df_rules["rules_agreement"] = final_agreement  # % of agreement bewtween rules. [Deprecated]
    df_rules["precision_vs_model"] = df_rules["precision_vs_model"].round(2)

    # Revert print
    sys.stdout = xx
    
    df_rules = df_rules.rename(columns={'precision_vs_model':'stability_score'})

    return df_rules


def checkDiversity(df_rules, numerical_cols, categorical_cols):
    """
    Function to measure "Diversity" through an overlapping score;
    the rules are different with "few overlapping concepts".
    This is computed checking the area of the hypercubes of the rules that
    overlaps with another one.

    The way to check this is by seeing the 2D planes of each hypercube (by keeping
    two degrees of freedom for the features in the hyperplane coordinates; n-2 features
    are maintained and the other two are changed between their max/min values in order
    to obtain the vertices of that 2D plane). Then, it is computed the area of the
    2D planes for the rules that overlaps, adding for all possible 2D planes the total
    area overlapped for each rule.

    In order to compute a score, the features are normalized in order to have
    values between 0 and 1.


    DEBUG:
        numerical_cols = ['gdenergy', 'gdpuls']
        categorical_cols = []
        df_rules = pd.DataFrame({'gdenergy_max':[50, 30],
                                 'gdenergy_min':[-20, -10],
                                 'gdpuls_max':[84, 100],
                                 'gdpuls_min':[-50, 20],
                                 'rule_id':[1, 2]
                                }
                                )
        # Area intersect: 64*40 = 2560
        # Area union: 9380 + 3200 - 2560
        # Jaccard: 2560 / (9380 + 3200 - 2560) = 0.255


    DEBUG 2:
        df_rules = pd.read_csv("df_multi_debug.csv")
        numerical_cols = ['seismic','seismoacoustic', 'genergy','gplus',
                          'gdenergy','gdpuls','bumps','bumps2','bumps3',
                          'bumps4', 'bumps5','bumps6', 'bumps7', 'bumps8',
                          'energy', 'maxenergy', 'class']
        categorical_cols = ['hazard_1', 'hazard_2', 'shift_1']

    DEGUB 3:
        df_rules = pd.read_csv("df_multi_debug2.csv")
        numerical_cols = ['seismic','seismoacoustic', 'genergy','gplus',
                          'gdenergy','gdpuls','bumps','bumps2','bumps3',
                          'bumps4', 'bumps5','bumps6', 'bumps7', 'bumps8',
                          'energy', 'maxenergy', 'class']
        categorical_cols = ['hazard_1_max', 'hazard_2_max', 'hazard_1_min',
                            'hazard_2_min', 'shift_1_max', 'shift_1_min']

    DEGUB 4:
        df_rules = pd.read_csv("df_multi_debug3.csv")
        numerical_cols = ['seismic', 'seismoacoustic', 'genergy','gplus',
                          'gdenergy','gdpuls','bumps', 'bumps2', 'bumps3',
                          'bumps4', 'bumps5','bumps6', 'bumps7', 'bumps8',
                          'energy', 'maxenergy', 'class']
        categorical_cols = ['hazard_1_max', 'hazard_1_min','hazard_2_max',
                            'hazard_2_min', 'shift_1_max', 'shift_1_min']

    DEGUB 5:
        df_rules = pd.read_csv("df_multi_debug4.csv")
        numerical_cols = ['hours_speed_control', 'fuel_idle_day',
                          'fuel_idle_accumulated', 'max_engine_cool_temp',
                          'max_engine_oil_temp','total_odometer','count_drive',
                          'reverse','count_neutral','count_park', 'count_forward',
                          'count_idle_events','idle_time','hour_drive', 'hour_park',
                          'hour_reverse', 'hour_forward','hour_neutral',
                          'count_harsh_brakes', 'speed_over_120', 'count_harsh_turns',
                          'count_jackrabbit', 'cruise_control_on','lights_left_on',
                          'engine_oil_variation', 'fuel_filter_life_variation',
                          'fuel_exhaust_fluid_variation','mean_transmission_oil_temp',
                          'mean_forward_acc', 'mean_braking_acc', 'mean_oil_temp',
                          'with_passenger','trip_fuel_used','trip_kms',
                          'max_tire_pressure_rl', 'max_tire_pressure_rr',
                          'max_tire_pressure_fl', 'max_tire_pressure_fr',
                          'rpm_high', 'rpm_low','rpm_medium_low', 'rpm_medium',
                          'rpm_medium_high', 'rpm_stopped','diff_tire_pressure_rl',
                          'diff_tire_pressure_fl', 'diff_tire_pressure_fr',
                          'diff_tire_pressure_rr', 'avg_fuel_consumption']
        categorical_cols = ['vehicle_group_1','vehicle_group_10',
                            'vehicle_group_12', 'vehicle_group_14',
                            'vehicle_group_2','vehicle_group_3',
                            'vehicle_group_4', 'vehicle_group_5',
                            'vehicle_group_6', 'vehicle_group_7',
                            'vehicle_group_8','vehicle_group_9',
                            'vehicle_group_10','vehicle_group_12',
                            'vehicle_group_14']

    Parameters
    ----------
    df_rules : dataframe
        Dataframe with the inlier or outlierrules. The features should appear as columns,
        indicating the maximum/minimum values associated to the vertices of the
        hypercubes. 
        For instance:
               gdenergy_max  gdenergy_min  gdpuls_max  gdpuls_min
        0         -13.5         -79.5       -64.5       -74.5
        1          11.0         -74.5        57.5       -64.5
        
    df_rules_outliers : dataframe
        Same as "df_rules_inliers" but for outliers.
    numerical_cols : list
        list with the numerical features.
    categorical_cols : list
        list with the caategorical feartures.

    Returns
    -------
    df_rules : dataframe
        Original dataframe with a column with the global score.
    df_return : dataframe
        All the possible vertex combinations per rule with their individual 
        overlapping score

    """

    def sort_vertices(list_vertex):
        """
        TODO
        Sort values for the Shapely order needed
        """
        list_x = list(set([v[0] for v in list_vertex]))
        list_y = list(set([v[1] for v in list_vertex]))

        result = [
            (min(list_x), max(list_y)),
            (max(list_x), max(list_y)),
            (max(list_x), min(list_y)),
            (min(list_x), min(list_y)),
        ]
        return result
    
    df_original = df_rules.copy()
    max_replace = (df_rules.replace(np.inf, 0).max().max() +
                   np.abs(df_rules.replace(np.inf, 0).max().max()) * 0.1)
    min_replace = (df_rules.replace(-np.inf, 0).min().min() - 
                   np.abs(df_rules.replace(-np.inf, 0).min().min()) * 0.1)
    df_rules = (
        df_rules.copy()
        .replace(np.inf, max_replace)
        .replace(-np.inf, min_replace)
    ) 

    list_cols = numerical_cols
    # Obtain combinations of features to create the 2D planes
    comb_free_features = [c for c in combinations(list_cols, 2)]
    # Filter rules for each categorical combination state
    if len(categorical_cols) > 0:
        df_cat = df_rules[categorical_cols]
        df_cat_unique = df_cat.drop_duplicates()
    else:
        df_cat_unique = pd.DataFrame({"dummy": [1]})  # Only one iter
        df_cat = pd.DataFrame()
    # Simple copy
    df_rules_original = df_rules.copy()

    # If there is one rule, then there is no overlapping
    if len(df_rules_original) <= 1:
        df_rules_original["score"] = 0
        df_rules_original["n_intersects"] = 0
        return df_rules_original
    # Obtain the vectors (dataframe) for each combination of 2 free features and n-2 fixed ones (n size hyperspace)
    df_return = pd.DataFrame()
    k = 0
    for i, row in df_cat_unique.iterrows():
        k += 1
        print("Iter {0}/{1}".format(k, len(df_cat_unique)))

        # If no categorical features, all rules together
        if not df_cat.empty:
            # Obtain sub-hypercube (not outliers)
            list_index = (
                df_cat[df_cat[row.index] == row.values].dropna().index
            )  # index for that sub-hypercube
            df_rules_sub = df_rules[
                (df_rules.index.isin(list_index))
            ].copy()  # sub-hypercube
        else:
            df_rules_sub = df_rules
        # If no rules, skip
        if len(df_rules_sub) == 0:
            continue
        # If len == 1, then no overlapping for this iter
        elif len(df_rules_sub) == 1:
            df_final = pd.DataFrame(
                {
                    "rule_id": [df_rules_sub.index[0]],
                    "score": [1],
                    "n_intersects": [0],
                    "n": [1],
                }
            )
        # Generic case
        else:
            df_final = pd.DataFrame()
            for comb in comb_free_features:
                # Specify cols
                list_free = [col + "_max" for col in comb] + [
                    col + "_min" for col in comb
                ]  # Cols to change
                cols_fixed = [col + "_max" for col in list_cols] + [
                    col + "_min" for col in list_cols
                ]
                cols_fixed = [
                    col for col in cols_fixed if col not in list_free
                ]  # Cols to mantain
                # cols_fixed = cols_fixed + categorical_cols # categorical do not use _max or _min

                list_x1 = [comb[0] + "_max", comb[0] + "_min"]
                list_x2 = [comb[1] + "_max", comb[1] + "_min"]
                cols_free = list(product(list_x1, list_x2))
                # Vertices of the 2D planes
                comb_vertices = [
                    tuple(list(x) + [j for j in cols_fixed]) for x in cols_free
                ]

                # Obtain polygons for those 2D planes
                list_aux = [
                    [tuple(vector[list(x)].values) for x in comb_vertices]
                    for _, vector in df_rules_sub[list_free + cols_fixed].iterrows()
                ]
                list_aux = [sort_vertices(list_vertex) for list_vertex in list_aux]
                polys = [Polygon(x) for x in list_aux]

                # Compute intersections of parallel 2D planes
                df_polys = pd.DataFrame(
                    {
                        "rule_id": list(df_rules_sub.index),
                        "rule_vertices": [x.bounds for x in polys],
                    }
                )

                df_results = pd.DataFrame(
                    {
                        "rules": [c for c in combinations(list(df_rules_sub.index), 2)],
                        "area_inter": [
                            pair[0].intersection(pair[1]).area
                            for pair in combinations(polys, 2)
                        ],
                        "area_1": [pair[0].area for pair in combinations(polys, 2)],
                        "area_2": [pair[1].area for pair in combinations(polys, 2)],
                    }
                )

                # If some area is 0, there is no overlapping
                df_results["rule_1"] = df_results.apply(lambda x: x["rules"][0], axis=1)
                df_results["rule_2"] = df_results.apply(lambda x: x["rules"][1], axis=1)
                df_results["area_union"] = (
                    df_results["area_1"]
                    + df_results["area_2"]
                    - df_results["area_inter"]
                )
                df_results["area_union"] = df_results.apply(
                    lambda x: 1 if x["area_union"] == 0 else x["area_union"], axis=1
                )
                df_results["jaccard"] = (
                    df_results["area_inter"] / df_results["area_union"]
                )
                df_results["score"] = 1 - df_results["jaccard"]

                # Annotate the rules with intersections in this 2D subspace
                df_results["n_intersects"] = df_results.apply(
                    lambda x: 0 if x["jaccard"] == 0 else 1, axis=1
                )

                # If score negative, set to 1 (no overlapping)
                df_results["score"] = df_results.apply(
                    lambda x: 1 if x["score"] < 0 else x["score"], axis=1
                )
                # Keep iter results
                if df_final.empty:
                    n = 1
                    df_results["n"] = n
                    df_final = df_results
                else:
                    n += 1
                    df_final["n"] = n
                    df_final["score"] += df_results["score"]
                    df_final["n_intersects"] += df_results["n_intersects"]
        df_return = df_return.append(df_final)
    df_return["n"] = df_return.apply(lambda x: 1 if x["n"] == 0 else x["n"], axis=1)
    df_return["score"] = df_return["score"] / df_return["n"]
    df_return.drop(["n"], axis=1, inplace=True)
    
    # Score (mean)
    overlap_score = df_return['score'].mean()
    df_original['diversity_score'] = overlap_score

    return df_original, df_return


def checkRepresentativeness(
    df_anomalies, df_rules_inliers, df_rules_outliers, numerical_cols, categorical_cols,
    col_predictions="predictions"
):
    """
    This function computes metrics regarding the representativness of the rules.
    First, it obtains how many inliers/outliers are within each of the rules.
    Then, it sees how many total inlers/outliers are covered by all the rules,
    as well as only by P@1 rules.

    Parameters
    ----------
    df_anomalies : dataframe
        A dataframe with the original features, along with one column ("anomalies") 
        that indicates whether the datapoint is outlier (-1) or inlier (1),
        the distance to the decision function ("dist"), and the scoring,
        which is essentially the same as the distance.
        For instance:
             gdenergy  gdpuls  anomalies      dist  predictions     score
        0       -72     -72          1  0.022756            1  0.022756
        1       -70     -79         -1 -0.003314           -1 -0.003314
        
    df_rules_inliers : dataframe
        Dataframe with the inlier rules. The features should appear as columns,
        indicating the maximum/minimum values associated to the vertices of the
        hypercubes. 
        For instance:
               gdenergy_max  gdenergy_min  gdpuls_max  gdpuls_min
        0         -13.5         -79.5       -64.5       -74.5
        1          11.0         -74.5        57.5       -64.5
        
    df_rules_outliers : dataframe
        Same as "df_rules_inliers" but for outliers.
    numerical_cols : list
        list with the numerical features.
    categorical_cols : list
        list with the caategorical feartures.
    col_predictions : str
        name of the column from df_anomalies where the predictions are.

    Returns
    -------
    df_rules_inliers : dataframe
        Dataframe with the original columns from the inliers rules with additional 
        columns containing the metric results. These columns are:
        - 'n_inliers_included': number of inlier datapoints included within
        each rule.
        - 'n_outliers_included': number of outlier datapoints included within
        each rule.
        - 'n_inliers': number of total inlier datapoints.
        - 'n_inliers_p0': number of inlier datapoints included within all rules.
        - 'n_inliers_p1': number of inlier datapoints included within P@1 rules.
        
    df_rules_outliers : dataframe
        Dataframe with the original columns from the outliers rules with additional 
        columns containing the metric results. These columns are:
        - 'n_inliers_included': number of inlier datapoints included within
        each rule.
        - 'n_outliers_included': number of outlier datapoints included within
        each rule.
        - 'n_outliers': number of total outliers datapoints.
        - 'n_outliers_p0': number of outliers datapoints included within all rules.
        - 'n_outliers_p1': number of outliers datapoints included within P@1 rules.
    """
    
    # Init param
    feature_cols = numerical_cols + categorical_cols
    df_rules_inliers = df_rules_inliers.copy()
    df_rules_outliers = df_rules_outliers.copy()

    # =======================================================================
    # 1. Get Inliers/Outliers inside rules
    # =======================================================================
    print("Checking inliers inside rules for inliers/outliers...")
    df_rules_inliers["n_inliers_included"] = 0
    df_rules_outliers["n_inliers_included"] = 0
    for i, data_point in df_anomalies[df_anomalies[col_predictions] == 1].iterrows():
        df_rules_inliers["n_inliers_included"] += checkPointInside(
            data_point, df_rules_inliers, feature_cols, []
        )["check"]

        df_rules_outliers["n_inliers_included"] += checkPointInside(
            data_point, df_rules_outliers, feature_cols, []
        )["check"]
        
    print("Checking outliers inside rules for inliers/outliers...")
    df_rules_inliers["n_outliers_included"] = 0
    df_rules_outliers["n_outliers_included"] = 0
    for i, data_point in df_anomalies[df_anomalies[col_predictions] == -1].iterrows():
        df_rules_inliers["n_outliers_included"] += checkPointInside(
            data_point, df_rules_inliers, feature_cols, []
        )["check"]

        df_rules_outliers["n_outliers_included"] += checkPointInside(
            data_point, df_rules_outliers, feature_cols, []
        )["check"]
        
    # Check how many datapoints are included with the rules with Precision=1 
    # and in general
    print("Checking inliers/outliers inside hypercubes with Precision=1...")
    n_inliers_p1 = 0
    n_inliers_p0 = 0
    n_outliers_p1 = 0
    n_outliers_p0 = 0
    n_inliers = len(df_anomalies[df_anomalies[col_predictions] == 1])
    n_outliers = len(df_anomalies[df_anomalies[col_predictions] == -1])

    for i, data_point in df_anomalies.iterrows():
        df_rules_inliers["check"] = checkPointInside(
            data_point, df_rules_inliers, feature_cols, []
        )["check"]
        df_rules_outliers["check"] = checkPointInside(
            data_point, df_rules_outliers, feature_cols, []
        )["check"]
        
        # If inlier
        if data_point[col_predictions] == 1:
            # Rules with any P and that include this datapoint
            df_aux = df_rules_inliers[(df_rules_inliers["check"] == 1)]
            if len(df_aux) > 0:
                n_inliers_p0 += 1
            # Rules with P=1 and that include this datapoint
            df_aux = df_rules_inliers[
                (df_rules_inliers["n_outliers_included"] == 0)
                & (df_rules_inliers["check"] == 1)
            ]
            if len(df_aux) > 0:
                n_inliers_p1 += 1
                
        # If outlier
        elif data_point[col_predictions] == -1:
            # Rules with any P and that include this datapoint
            df_aux = df_rules_outliers[(df_rules_outliers["check"] == 1)]
            if len(df_aux) > 0:
                n_outliers_p0 += 1
            # Rules with P=1 and that include this datapoint
            df_aux = df_rules_outliers[
                (df_rules_outliers["n_inliers_included"] == 0)
                & (df_rules_outliers["check"] == 1)
            ]
            if len(df_aux) > 0:
                n_outliers_p1 += 1
                
    df_rules_inliers["n_inliers"] = n_inliers
    df_rules_inliers["n_inliers_p0"] = n_inliers_p0
    df_rules_inliers["n_inliers_p1"] = n_inliers_p1
    df_rules_outliers["n_outliers"] = n_outliers
    df_rules_outliers["n_outliers_p1"] = n_outliers_p1
    df_rules_outliers["n_outliers_p0"] = n_outliers_p0

    df_rules_outliers = df_rules_outliers.drop(columns=["check"], errors="ignore")
    df_rules_inliers = df_rules_inliers.drop(columns=["check"], errors="ignore")

    return df_rules_inliers, df_rules_outliers







