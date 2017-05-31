"""
Compare the performance of different model parameters
"""

# import packages
import toolbox
import GPy
import json
import os
import pandas as pd
from sklearn.metrics import mean_squared_error as MSE
from sklearn import neural_network
import matplotlib.pyplot as plt

#################################################################################################################
#                                              neural network                                           #
#################################################################################################################
"""
Compare the performance under different solver function and activation function
"""


# solver function
def generate_nn_solver_ratio_result(X_train, X_test, y_train, y_test):
    # generate the result for random samples
    ratio_result = pd.DataFrame(y_test, columns=['ratio_baseline'])

    print "Solver Function: lbfgs"
    model = neural_network.MLPRegressor(solver='lbfgs', max_iter=1000, learning_rate_init=0.005)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    ratio_result['lbfgs'] = y_pred

    print "Solver Function: sgd"
    model = neural_network.MLPRegressor(solver='sgd', max_iter=1000, learning_rate_init=0.005)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    ratio_result['sgd'] = y_pred

    print "Solver Function: adam"
    model = neural_network.MLPRegressor(solver='adam', max_iter=1000, learning_rate_init=0.005)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    ratio_result['adam'] = y_pred

    return ratio_result


def check_nn_solver_performance(output_test, ratio_result):
    # process the result to obtain the pred arrival time with different models
    time_result = pd.DataFrame(output_test[2], columns=['actual'])
    time_result['baseline'] = output_test[1]
    time_result['lbfgs'] = time_result['baseline'] * ratio_result['lbfgs']
    time_result['sgd'] = time_result['baseline'] * ratio_result['sgd']
    time_result['adam'] = time_result['baseline'] * ratio_result['adam']

    # calculate the MSE of the arrival time
    columns = time_result.columns
    mse_time = dict()
    for column in columns:
        if column == 'actual':
            continue
        mse_time[column] = MSE(time_result['actual'], time_result[column])

    # process the result to obtain the ratio(actual_arrival_time / pred_arrival_time)
    ratio_result['lbfgs'] = ratio_result['ratio_baseline'] / ratio_result['lbfgs']
    ratio_result['sgd'] = ratio_result['ratio_baseline'] / ratio_result['sgd']
    ratio_result['adam'] = ratio_result['ratio_baseline'] / ratio_result['adam']

    # calculate the MSE of ratio
    columns = ratio_result.columns
    true_ratio = [1.0] * len(ratio_result)
    mse_ratio = dict()
    for column in columns:
        mse_ratio[column] = MSE(true_ratio, ratio_result[column])

    return time_result, mse_time, ratio_result, mse_ratio


# activation function
def generate_nn_activation_ratio_result(X_train, X_test, y_train, y_test):
    # generate the result for random samples
    ratio_result = pd.DataFrame(y_test, columns=['ratio_baseline'])

    print "Activation Function: identity"
    model = neural_network.MLPRegressor(activation='identity', max_iter=1000, learning_rate_init=0.005)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    ratio_result['identity'] = y_pred

    print "Activation Function: logistic"
    model = neural_network.MLPRegressor(activation='logistic', max_iter=1000, learning_rate_init=0.005)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    ratio_result['logistic'] = y_pred

    print "Activation Function: tanh"
    model = neural_network.MLPRegressor(activation='tanh', max_iter=1000, learning_rate_init=0.005)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    ratio_result['tanh'] = y_pred

    print "Activation Function: relu"
    model = neural_network.MLPRegressor(activation='relu', max_iter=1000, learning_rate_init=0.005)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    ratio_result['relu'] = y_pred

    return ratio_result


def check_nn_activation_performance(output_test, ratio_result):
    # process the result to obtain the pred arrival time with different models
    time_result = pd.DataFrame(output_test[2], columns=['actual'])
    time_result['baseline'] = output_test[1]
    time_result['identity'] = time_result['baseline'] * ratio_result['identity']
    time_result['logistic'] = time_result['baseline'] * ratio_result['logistic']
    time_result['tanh'] = time_result['baseline'] * ratio_result['tanh']
    time_result['relu'] = time_result['baseline'] * ratio_result['relu']

    # calculate the MSE of the arrival time
    columns = time_result.columns
    mse_time = dict()
    for column in columns:
        if column == 'actual':
            continue
        mse_time[column] = MSE(time_result['actual'], time_result[column])

    # process the result to obtain the ratio(actual_arrival_time / pred_arrival_time)
    ratio_result['identity'] = ratio_result['ratio_baseline'] / ratio_result['identity']
    ratio_result['logistic'] = ratio_result['ratio_baseline'] / ratio_result['logistic']
    ratio_result['tanh'] = ratio_result['ratio_baseline'] / ratio_result['tanh']
    ratio_result['relu'] = ratio_result['ratio_baseline'] / ratio_result['relu']

    # calculate the MSE of ratio
    columns = ratio_result.columns
    true_ratio = [1.0] * len(ratio_result)
    mse_ratio = dict()
    for column in columns:
        mse_ratio[column] = MSE(true_ratio, ratio_result[column])

    return time_result, mse_time, ratio_result, mse_ratio


#################################################################################################################
#                                              gaussian process                                    #
#################################################################################################################
"""
Compare the performance under different kernel function for gaussian process
"""


def generate_gaussian_ratio_result(X_train, X_test, y_train, y_test):
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


def check_gaussian_performance(output_test, ratio_result):
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


#################################################################################################################
#                                              main functions                                                   #
#################################################################################################################
"""
Main interface for users
"""


def compare_models(dataset, generate_ratio_result_function, check_performance_function, save_path=None):
    """
    Do model selection for the neural network and gaussian process
    
    :param dataset: the dataframe for the dataset table
    :param generate_ratio_result_function: the function for training and prediction in model selection. There are three different choices here:
    ['generate_nn_solver_ratio_result', 'generate_nn_activation_ratio_result', 'generate_gaussian_ratio_result']
    :param check_performance_function: the function for performance assessement in model selection. There are three different choices here:
    ['check_nn_solver_performance', 'check_nn_activation_performance', 'check_gaussian_performance']
    :param save_path: path of a csv file to store the baseline1 result
    :return: a list of result for model selection: time_result, mse_time, ratio_result, mse_ratio
    """
    plt.style.use('ggplot')

    dataset.reset_index(inplace=True)

    full_dataset = toolbox.preprocess_dataset(dataset)

    X_train, X_test, output_train, output_test = toolbox.split_dataset(full_dataset)
    y_train = output_train[0]
    y_test = output_test[0]

    ratio_result = generate_ratio_result_function(X_train, X_test, y_train, y_test)

    time_result, mse_time, ratio_result, mse_ratio = check_performance_function(output_test, ratio_result)

    if save_path is not None:
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        time_result.to_csv(save_path + 'time_result.csv')
        ratio_result.to_csv(save_path + 'ratio_result.csv')
        with open(save_path + 'mse_time.json', 'w') as f:
            json.dump(mse_time, f)
        with open(save_path + 'mse_ratio.json', 'w') as f:
            json.dump(mse_ratio, f)

    return time_result, mse_time, ratio_result, mse_ratio
