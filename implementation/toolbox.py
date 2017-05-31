"""
Provide tool functions for implementations
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from sklearn import linear_model, svm, neural_network, preprocessing
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


def split_dataset(dataset):

    feature_list = ['weather', 'rush_hour', 'baseline_result', 'ratio_current_trip', 'ratio_prev_trip', 'prev_arrival_time']

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
    """
    train and predict the result with the training set and the test set

    :param X_train: the features of the training set
    :param X_test: the output of the training set
    :param y_train: the features of the test set
    :param y_test: the output of the test set
    :return: dataframe of the predicted result under different models
    """
    # generate the result for random samples
    ratio_result = pd.DataFrame(y_test, columns=['ratio_baseline'])

    model1 = linear_model.LinearRegression()
    model1.fit(X_train, y_train)
    y_pred = model1.predict(X_test)
    ratio_result['single_linear_regression'] = y_pred

    model2 = svm.SVR()
    model2.fit(X_train, y_train)
    y_pred = model2.predict(X_test)
    ratio_result['single_SVM'] = y_pred

    model3 = neural_network.MLPRegressor(solver='lbfgs', max_iter=1000, learning_rate_init=0.005)
    model3.fit(X_train, y_train)
    y_pred = model3.predict(X_test)
    ratio_result['single_NN'] = y_pred

    kernel = GPy.kern.Matern32(input_dim=6, ARD=True)
    m_full = GPy.models.SparseGPRegression(X_train, y_train.reshape(len(y_train), 1), kernel)
    m_full.optimize('bfgs')
    y_pred, y_var = m_full.predict(X_test)
    ratio_result['single_GP'] = y_pred

    return ratio_result


def multiple_shape_learning(ratio_result, X_train_list, y_train_list, X_test_list):
    """
    the dataset is used without considering the shape id

    :param ratio_result: the predicted result from each model
    :param X_train_list: the list of input in the training set
    :param y_train_list: the list of output in the training set
    :param X_test_list: the list of input in the test set
    :return: dataframe of the predicted result under different models
    """

    # MTL GP learning
    kernel = GPy.kern.Matern32(input_dim=6, ARD=True)
    icm = GPy.util.multioutput.ICM(input_dim=6, num_outputs=len(X_train_list), kernel=kernel)
    model = GPy.models.SparseGPCoregionalizedRegression(X_list=X_train_list, Y_list=y_train_list, Z_list=[], kernel=icm)
    model.optimize('bfgs')
    X_test = np.concatenate(X_test_list)
    X_test = np.hstack((X_test, np.ones((len(X_test), 1))))
    noise_dict = {'output_index': X_test[:, -1:].astype(int)}
    y_pred, y_var = model.predict(X_test, Y_metadata=noise_dict)
    ratio_result['MTL_GP'] = y_pred

    X_train = np.concatenate(X_train_list)
    y_train = np.concatenate(y_train_list)
    y_train = y_train.reshape((len(y_train),))
    X_test = np.concatenate(X_test_list)

    model1 = linear_model.LinearRegression()
    model1.fit(X_train, y_train)
    y_pred = model1.predict(X_test)
    ratio_result['multiple_linear_regression'] = y_pred

    model2 = svm.SVR()
    model2.fit(X_train, y_train)
    y_pred = model2.predict(X_test)
    ratio_result['multiple_SVM'] = y_pred

    model3 = neural_network.MLPRegressor(solver='lbfgs', max_iter=1000, learning_rate_init=0.005)
    model3.fit(X_train, y_train)
    y_pred = model3.predict(X_test)
    ratio_result['multiple_NN'] = y_pred

    kernel = GPy.kern.Matern32(input_dim=6, ARD=True)
    m_full = GPy.models.SparseGPRegression(X_train, y_train.reshape(len(y_train), 1), kernel)
    m_full.optimize('bfgs')
    y_pred, y_var = m_full.predict(X_test)
    ratio_result['multiple_GP'] = y_pred

    return ratio_result


def single_shape_learning(full_dataset):

    X_train_list = []
    X_test_list = []
    output_train_list = []
    output_test_list = []
    y_train_list = []
    y_test_list = []

    grouped = full_dataset.groupby(['shape_id'])
    ratio_result_list = []
    ratio_result = None
    for name, item in grouped:
        print "generate the result for ", name
        X_train, X_test, output_train, output_test = split_dataset(item)
        if len(X_test) == 0 or len(X_train) == 0:
            continue
        y_train = output_train[0]
        y_test = output_test[0]

        ratio_result = generate_ratio_result(X_train, X_test, y_train, y_test)

        ratio_result_list.append(ratio_result)

        X_train_list.append(X_train)
        output_train_list.append(output_train)
        y_train_list.append(y_train.reshape(len(y_train), 1))

        X_test_list.append(X_test)
        output_test_list.append(output_test)
        y_test_list.append(y_test)

    if ratio_result_list != []:
        ratio_result = pd.concat(ratio_result_list, ignore_index=True)
    return ratio_result, X_train_list, X_test_list, output_train_list, output_test_list, y_train_list, y_test_list



def check_performance(output_test_list, ratio_result):
    output_test = np.concatenate(output_test_list, axis=1)
    # process the result to obtain the pred arrival time with different models
    time_result = pd.DataFrame(output_test[2], columns=['actual'])
    time_result['baseline'] = output_test[1]
    time_result['single_linear_regression'] = time_result['baseline'] * ratio_result['single_linear_regression']
    time_result['single_SVM'] = time_result['baseline'] * ratio_result['single_SVM']
    time_result['single_NN'] = time_result['baseline'] * ratio_result['single_NN']
    time_result['single_GP'] = time_result['baseline'] * ratio_result['single_GP']
    time_result['MTL_GP'] = time_result['baseline'] * ratio_result['MTL_GP']
    time_result['multiple_linear_regression'] = time_result['baseline'] * ratio_result['multiple_linear_regression']
    time_result['multiple_SVM'] = time_result['baseline'] * ratio_result['multiple_SVM']
    time_result['multiple_NN'] = time_result['baseline'] * ratio_result['multiple_NN']
    time_result['multiple_GP'] = time_result['baseline'] * ratio_result['multiple_GP']


    # calculate the MSE of the arrival time
    columns = time_result.columns
    mse_time = dict()
    for column in columns:
        if column == 'actual':
            continue
        mse_time[column] = MSE(time_result['actual'], time_result[column])

    # process the result to obtain the ratio(actual_arrival_time / pred_arrival_time)
    ratio_result['single_linear_regression'] = ratio_result['ratio_baseline'] / ratio_result['single_linear_regression']
    ratio_result['single_SVM'] = ratio_result['ratio_baseline'] / ratio_result['single_SVM']
    ratio_result['single_NN'] = ratio_result['ratio_baseline'] / ratio_result['single_NN']
    ratio_result['single_GP'] = ratio_result['ratio_baseline'] / ratio_result['single_GP']
    ratio_result['MTL_GP'] = ratio_result['ratio_baseline'] / ratio_result['MTL_GP']
    ratio_result['multiple_linear_regression'] = ratio_result['ratio_baseline'] / ratio_result['multiple_linear_regression']
    ratio_result['multiple_SVM'] = ratio_result['ratio_baseline'] / ratio_result['multiple_SVM']
    ratio_result['multiple_NN'] = ratio_result['ratio_baseline'] / ratio_result['multiple_NN']
    ratio_result['multiple_GP'] = ratio_result['ratio_baseline'] / ratio_result['multiple_GP']

    # calculate the MSE of ratio
    columns = ratio_result.columns
    true_ratio = [1.0] * len(ratio_result)
    mse_ratio = dict()
    for column in columns:
        mse_ratio[column] = MSE(true_ratio, ratio_result[column])

    return time_result, mse_time, ratio_result, mse_ratio
