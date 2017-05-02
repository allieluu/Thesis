# -*- coding: utf-8 -*-
# @Author: allisonburton
# @Date:   2017-05-01 12:12:22
# @Last Modified by:   allieluu
# @Last Modified time: 2017-05-01 16:47:14

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


def main():
    np.set_printoptions(suppress=True)
    narac_file_path = "../../tigress/arburton/plink_data/narac_rf"
    csv_data = []
    for chunk in pd.read_csv(narac_file_path, delim_whitespace=True, index_col=0, chunksize=20000):
        csv_data.append(chunk)
    samples = pd.concat(csv_data, axis=0)
    del csv_data
    # TODO: pull out affection column as y
    affection = pd.DataFrame(samples, columns="Affection")
    samples = samples.drop(["Affection", "Sex", "DRB1_1",
                            "DRB1_2", "SENum", "SEStatus", "AntiCCP", "RFUW"], axis=1)
    samples = pd.get_dummies(samples, columns=(samples.columns != "ID"))
    sample_train, sample_test, affection_train, affection_test = train_test_split(
        samples, affection, test_size=0.8)
    # TODO: potentially make sample weights percentage of non ?? SNPs

    # RANDOM FOREST CLASSIFIER

    rf = RandomForestClassifier(n_estimators=5000, max_features=40, n_jobs=2)
    rf.fit(sample_train, affection_train)
    print("Random forest accuracy: {}".format(
        rf.score(sample_test, affection_test)))
    print("Random forest feature importances:")
    print(rf.feature_importances_)
    print("Random forest parameters:")
    print(rf.get_params())

    # LASSO CLASSIFIER
    lasso = Lasso()
    lasso.fit(sample_train, affection_train)
    print("LASSO accuracy: {}".format(lasso.score(sample_test, affection_test)))
    print("LASSO parameters:")
    print(lasso.get_params())

    # LOG REGRESSION
    log_reg = LogisticRegression(n_jobs=2)
    log_reg.fit(sample_train, affection_train)
    print("Log regression accuracy: {}".format(
        log_reg.score(sample_test, affection_test)))
    print("Log regression parameters:")
    print(log_reg.get_params())

    # NEURAL NETS
    mlp_classifier = MLPClassifier()
    mlp_classifier.fit(sample_train, affection_train)
    print("MLP Classifier accuracy: {}".format(mlp_classifier.score(sample_test, affection_test)))
    print("MLP Classifier parameters:")
    print(mlp_classifier.get_params())

if __name__ == '__main__':
    main()
