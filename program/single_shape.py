"""
Use each shape to train the model and predict the corresponding result
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
    full_dataset['shape_id'] = origin_dataset['shape_id']
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

    model1 = linear_model.LinearRegression()
    model1.fit(X_train, y_train)
    y_pred = model1.predict(X_test)
    ratio_result['linear_regression'] = y_pred

    model2 = svm.SVR()
    model2.fit(X_train, y_train)
    y_pred = model2.predict(X_test)
    ratio_result['SVM'] = y_pred

    model3 = neural_network.MLPRegressor(solver='lbfgs', max_iter=1000, learning_rate_init=0.005)
    model3.fit(X_train, y_train)
    y_pred = model3.predict(X_test)
    ratio_result['NN'] = y_pred

    kernel = GPy.kern.Matern32(input_dim=6, ARD=True)
    m_full = GPy.models.SparseGPRegression(X_train, y_train.reshape(len(y_train), 1), kernel)
    m_full.optimize('bfgs')
    y_pred, y_var = m_full.predict(X_test)
    ratio_result['GP'] = y_pred

    return ratio_result

def check_performance(output_test, ratio_result):
    # process the result to obtain the pred arrival time with different models
    time_result = pd.DataFrame(output_test[2], columns=['actual'])
    time_result['baseline'] = output_test[1]
    time_result['linear_regression'] = time_result['baseline'] * ratio_result['linear_regression']
    time_result['SVM'] = time_result['baseline'] * ratio_result['SVM']
    time_result['NN'] = time_result['baseline'] * ratio_result['NN']
    time_result['GP'] = time_result['baseline'] * ratio_result['GP']

    # calculate the MSE of the arrival time
    columns = time_result.columns
    mse_time = dict()
    for column in columns:
        if column == 'actual':
            continue
        mse_time[column] = MSE(time_result['actual'], time_result[column])

    # process the result to obtain the ratio(actual_arrival_time / pred_arrival_time)
    ratio_result['linear_regression'] = ratio_result['ratio_baseline'] / ratio_result['linear_regression']
    ratio_result['SVM'] = ratio_result['ratio_baseline'] / ratio_result['SVM']
    ratio_result['NN'] = ratio_result['ratio_baseline'] / ratio_result['NN']
    ratio_result['GP'] = ratio_result['ratio_baseline'] / ratio_result['GP']

    # calculate the MSE of ratio
    columns = ratio_result.columns
    true_ratio = [1.0] * len(ratio_result)
    mse_ratio = dict()
    for column in columns:
        mse_ratio[column] = MSE(true_ratio, ratio_result[column])

    return time_result, mse_time, ratio_result, mse_ratio


def export_files(time_result, mse_time, ratio_result, mse_ratio):
    path = 'result/single_shape/'

    print "prepare to export file: ", path

    if not os.path.exists(path):
        os.mkdir(path)

    # export figures
    columns = time_result.columns
    for column in columns:
        if column == 'actual':
            continue
        filename = column + '.png'
        figure = time_result.plot(kind='scatter', y=column, x='actual', xlim=(0, 6000), ylim=(0, 6000))
        fig = figure.get_figure()
        fig.savefig(path + filename)
        plt.close(fig)

    # export mse files
    with open(path + 'mse_ratio.json', 'w') as f:
        json.dump(mse_ratio, f)
    with open(path + 'mse_time.json', 'w') as f:
        json.dump(mse_time, f)

    # export the csv file
    time_result.to_csv(path + 'time_result.csv')
    ratio_result.to_csv(path + 'ratio_result.csv')


plt.style.use('ggplot')

dataset = pd.read_csv("full_dataset.csv")
dataset.reset_index(inplace=True)
dataset.drop(['Unnamed: 0'], axis=1, inplace=True)

full_dataset = preprocess_dataset(dataset)

features = ['weather', 'rush_hour', 'baseline_result', 'ratio_current_trip', 'ratio_prev_trip', 'prev_arrival_time']

# single shape result
grouped = full_dataset.groupby(['shape_id'])
ratio_result_list = []
output_test_list = []
for name, item in grouped:
    print "generate the result for ", name
    X_train, X_test, output_train, output_test = split_dataset(item, features)
    y_train = output_train[0]
    y_test = output_test[0]

    ratio_result = generate_ratio_result(X_train, X_test, y_train, y_test)

    ratio_result_list.append(ratio_result)
    output_test_list.append(output_test)

ratio_result = pd.concat(ratio_result_list, ignore_index=True)
output_test = np.concatenate(output_test_list, axis=1)
time_result, mse_time, ratio_result, mse_ratio = check_performance(output_test, ratio_result)

export_files(time_result, mse_time, ratio_result, mse_ratio)
