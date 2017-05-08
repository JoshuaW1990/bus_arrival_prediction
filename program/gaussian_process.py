"""
This file compare different settings for the gaussian process, especially for kernel function
"""

# import the modules
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn import linear_model, svm, neural_network, preprocessing
from sklearn.metrics import mean_squared_error as MSE
import json
import matplotlib.pyplot as plt
import GPy


def preprocess_dataset(origin_dataset):
    # preprocess to obtain the dataset
    full_dataset = pd.DataFrame()
    full_dataset['weather'] = origin_dataset['weather']
    full_dataset['rush_hour'] = origin_dataset['rush_hour']
    full_dataset['baseline_result'] = origin_dataset['baseline_result']
    full_dataset['ratio_baseline'] = origin_dataset['actual_arrival_time'] / origin_dataset['baseline_result']
    full_dataset['ratio_current_trip'] = origin_dataset['ratio_current_trip']
    full_dataset['ratio_prev_trip'] = origin_dataset['ratio_prev_trip']
    full_dataset['ratio_prev_seg'] = origin_dataset['actual_arrival_time'] / origin_dataset['prev_arrival_time']
    full_dataset['prev_arrival_time'] = origin_dataset['prev_arrival_time']
    full_dataset['actual_arrival_time'] = origin_dataset['actual_arrival_time']
    return full_dataset


def split_dataset(dataset, feature_list):
    X = dataset.as_matrix(columns=feature_list)
    y = dataset.as_matrix(columns=['ratio_baseline', 'baseline_result', 'actual_arrival_time'])

    # normalization
    X_normalized = preprocessing.normalize(X, norm='l2')

    # split the dataset
    X_train, X_test, output_train, output_test = train_test_split(X_normalized, y, test_size=0.33, random_state=42)

    output_train = output_train.transpose()
    output_test = output_test.transpose()

    return X_train, X_test, output_train, output_test


def generate_ratio_result(X_train, X_test, y_train, y_test):
    # generate the result for random samples
    ratio_result = pd.DataFrame(y_test, columns=['ratio_baseline'])

    print "Gaussian Process: RBF, ARD=True"
    kernel = GPy.kern.RBF(input_dim=6, ARD=True)
    model = GPy.models.SparseGPRegression(X_train, y_train.reshape(len(y_train), 1), kernel)
    model.optimize('bfgs')
    y_pred, y_var = model.predict(X_test)
    ratio_result['GP_RBF_ARD'] = y_pred

    print "Gaussian Process: RBF, ARD=False"
    kernel = GPy.kern.RBF(input_dim=6, ARD=False)
    model = GPy.models.SparseGPRegression(X_train, y_train.reshape(len(y_train), 1), kernel)
    model.optimize('bfgs')
    y_pred, y_var = model.predict(X_test)
    ratio_result['GP_RBF_NoARD'] = y_pred

    print "Gaussian Process: Matern32, ARD=True"
    kernel = GPy.kern.Matern32(input_dim=6, ARD=True)
    model = GPy.models.SparseGPRegression(X_train, y_train.reshape(len(y_train), 1), kernel)
    model.optimize('bfgs')
    y_pred, y_var = model.predict(X_test)
    ratio_result['GP_Matern32_ARD'] = y_pred

    print "Gaussian Process: Matern32, ARD=False"
    kernel = GPy.kern.Matern32(input_dim=6, ARD=False)
    model = GPy.models.SparseGPRegression(X_train, y_train.reshape(len(y_train), 1), kernel)
    model.optimize('bfgs')
    y_pred, y_var = model.predict(X_test)
    ratio_result['GP_Matern32_NoARD'] = y_pred

    print "Gaussian Process: Matern52, ARD=True"
    kernel = GPy.kern.Matern52(input_dim=6, ARD=True)
    model = GPy.models.SparseGPRegression(X_train, y_train.reshape(len(y_train), 1), kernel)
    model.optimize('bfgs')
    y_pred, y_var = model.predict(X_test)
    ratio_result['GP_Matern52_ARD'] = y_pred

    print "Gaussian Process: Matern52, ARD=False"
    kernel = GPy.kern.Matern52(input_dim=6, ARD=False)
    model = GPy.models.SparseGPRegression(X_train, y_train.reshape(len(y_train), 1), kernel)
    model.optimize('bfgs')
    y_pred, y_var = model.predict(X_test)
    ratio_result['GP_Matern52_NoARD'] = y_pred

    print "Gaussian Process: Linear, ARD=True"
    kernel = GPy.kern.Linear(input_dim=6, ARD=True)
    model = GPy.models.SparseGPRegression(X_train, y_train.reshape(len(y_train), 1), kernel)
    model.optimize('bfgs')
    y_pred, y_var = model.predict(X_test)
    ratio_result['GP_Linear_ARD'] = y_pred

    print "Gaussian Process: Linear, ARD=False"
    kernel = GPy.kern.Linear(input_dim=6, ARD=False)
    model = GPy.models.SparseGPRegression(X_train, y_train.reshape(len(y_train), 1), kernel)
    model.optimize('bfgs')
    y_pred, y_var = model.predict(X_test)
    ratio_result['GP_Linear_NoARD'] = y_pred

    return ratio_result


def check_performance(output_test, ratio_result):
    # process the result to obtain the pred arrival time with different models
    time_result = pd.DataFrame(output_test[2], columns=['actual'])
    time_result['baseline'] = output_test[1]
    time_result['GP_RBF_ARD'] = time_result['baseline'] * ratio_result['GP_RBF_ARD']
    time_result['GP_RBF_NoARD'] = time_result['baseline'] * ratio_result['GP_RBF_NoARD']
    time_result['GP_Matern32_ARD'] = time_result['baseline'] * ratio_result['GP_Matern32_ARD']
    time_result['GP_Matern32_NoARD'] = time_result['baseline'] * ratio_result['GP_Matern32_NoARD']
    time_result['GP_Matern52_ARD'] = time_result['baseline'] * ratio_result['GP_Matern52_ARD']
    time_result['GP_Matern52_NoARD'] = time_result['baseline'] * ratio_result['GP_Matern52_NoARD']
    time_result['GP_Linear_ARD'] = time_result['baseline'] * ratio_result['GP_Linear_ARD']
    time_result['GP_Linear_NoARD'] = time_result['baseline'] * ratio_result['GP_Linear_NoARD']

    # calculate the MSE of the arrival time
    columns = time_result.columns
    mse_time = dict()
    for column in columns:
        if column == 'actual':
            continue
        mse_time[column] = MSE(time_result['actual'], time_result[column])

    # process the result to obtain the ratio(actual_arrival_time / pred_arrival_time)
    ratio_result['GP_RBF_ARD'] = ratio_result['ratio_baseline'] / ratio_result['GP_RBF_ARD']
    ratio_result['GP_RBF_NoARD'] = ratio_result['ratio_baseline'] / ratio_result['GP_RBF_NoARD']
    ratio_result['GP_Matern32_ARD'] = ratio_result['ratio_baseline'] / ratio_result['GP_Matern32_ARD']
    ratio_result['GP_Matern32_NoARD'] = ratio_result['ratio_baseline'] / ratio_result['GP_Matern32_NoARD']
    ratio_result['GP_Matern52_ARD'] = ratio_result['ratio_baseline'] / ratio_result['GP_Matern52_ARD']
    ratio_result['GP_Matern52_NoARD'] = ratio_result['ratio_baseline'] / ratio_result['GP_Matern52_NoARD']
    ratio_result['GP_Linear_ARD'] = ratio_result['ratio_baseline'] / ratio_result['GP_Linear_ARD']
    ratio_result['GP_Linear_NoARD'] = ratio_result['ratio_baseline'] / ratio_result['GP_Linear_NoARD']

    # calculate the MSE of ratio
    columns = ratio_result.columns
    true_ratio = [1.0] * len(ratio_result)
    mse_ratio = dict()
    for column in columns:
        mse_ratio[column] = MSE(true_ratio, ratio_result[column])

    return time_result, mse_time, ratio_result, mse_ratio



plt.style.use('ggplot')

dataset = pd.read_csv("full_dataset.csv")
dataset.reset_index(inplace=True)
dataset.drop(['Unnamed: 0'], axis=1, inplace=True)

full_dataset = preprocess_dataset(dataset)

features = ['weather', 'rush_hour', 'baseline_result', 'ratio_current_trip', 'ratio_prev_trip', 'prev_arrival_time']

X_train, X_test, output_train, output_test = split_dataset(full_dataset, features)
y_train = output_train[0]
y_test = output_test[0]

ratio_result = generate_ratio_result(X_train, X_test, y_train, y_test)

time_result, mse_time, ratio_result, mse_ratio = check_performance(output_test, ratio_result)
