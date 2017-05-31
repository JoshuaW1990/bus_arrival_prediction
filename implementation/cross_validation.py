"""
Run cross validation for the dataset
"""

# import package
import toolbox
import pandas as pd
import numpy as np
import os
from sklearn import preprocessing


def split_dataset(dataset, fold, total_fold):
    """
    Split the dataset according to fold information
    
    :param dataset: dataframe for the dataset table
    :param fold: start from 0, indicate which fold will be used in the folds
    :param total_fold: total number of folds
    :return: split matrix of dataset
    """
    total_count = len(dataset)
    fold_count = int(total_count / total_fold)
    start_index = fold * fold_count
    end_index = (fold + 1) * fold_count

    X = dataset.as_matrix(columns=['weather', 'rush_hour', 'baseline_result', 'ratio_current_trip', 'ratio_prev_trip', 'prev_arrival_time'])
    y = dataset.as_matrix(columns=['ratio_baseline', 'baseline_result', 'actual_arrival_time'])

    # normalization
    X_normalized = preprocessing.normalize(X, norm='l2')

    # split the dataset
    X_train = np.concatenate([X_normalized[:start_index, :], X_normalized[end_index:, :]])
    X_test = X_normalized[start_index:end_index, :]

    output_train = np.concatenate([y[:start_index, :], y[end_index:, :]])
    output_test = y[start_index:end_index, :]

    output_train = output_train.transpose()
    output_test = output_test.transpose()

    return X_train, X_test, output_train, output_test


def single_shape_learning(full_dataset, fold, total_fold):
    """
    the dataset is used when considering the shape id

    :param full_dataset: the dataframe of the preprocessed dataset
    :param fold: start from 0, indicate which fold will be used in the folds
    :param total_fold: total number of folds
    :return: the list of prediction in different folds
    """

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
        X_train, X_test, output_train, output_test = split_dataset(item, fold, total_fold)
        if len(X_test) == 0 or len(X_train) == 0:
            continue
        y_train = output_train[0]
        y_test = output_test[0]

        ratio_result = toolbox.generate_ratio_result(X_train, X_test, y_train, y_test)

        ratio_result_list.append(ratio_result)

        X_train_list.append(X_train)
        output_train_list.append(output_train)
        y_train_list.append(y_train.reshape(len(y_train), 1))

        X_test_list.append(X_test)
        output_test_list.append(output_test)
        y_test_list.append(y_test)

    if ratio_result_list:
        ratio_result = pd.concat(ratio_result_list, ignore_index=True)
    return ratio_result, X_train_list, X_test_list, output_train_list, output_test_list, y_train_list, y_test_list


#################################################################################################################
#                                              main functions                                                   #
#################################################################################################################
"""
Main interface for users
"""


def cross_validation(origin_dataset, total_fold, save_path=None):
    """
    run cross validation
    
    :param origin_dataset: original dataset read from the dataset table
    :param total_fold: total number of folds
    :param save_path: the path to export result
    :return: dataframe of results under different models
    """
    dataset = toolbox.preprocess_dataset(origin_dataset)

    mse_time_result = pd.DataFrame(
        columns=['baseline', 'single_linear_regression', 'single_SVM', 'single_NN', 'single_GP', 'MTL_GP',
                 'multiple_linear_regression', 'multiple_SVM', 'multiple_NN', 'multiple_GP'])
    mse_ratio_result = pd.DataFrame(
        columns=['baseline', 'single_linear_regression', 'single_SVM', 'single_NN', 'single_GP', 'MTL_GP',
                 'multiple_linear_regression', 'multiple_SVM', 'multiple_NN', 'multiple_GP'])

    dataset = dataset.sample(frac=1).reset_index(drop=True)

    for fold in range(total_fold):
        print fold
        ratio_result, X_train_list, X_test_list, output_train_list, output_test_list, y_train_list, y_test_list = single_shape_learning(dataset, fold, total_fold)
        if ratio_result is None:
            continue
        ratio_result = toolbox.multiple_shape_learning(ratio_result, X_train_list, y_train_list, X_test_list)
        time_result, mse_time, ratio_result, mse_ratio = toolbox.check_performance(output_test_list, ratio_result)

        mse_time_result.loc[len(mse_time_result)] = [mse_time['baseline'], mse_time['single_linear_regression'],
                                                     mse_time['single_SVM'], mse_time['single_NN'],
                                                     mse_time['single_GP'], mse_time['MTL_GP'],
                                                     mse_time['multiple_linear_regression'], mse_time['multiple_SVM'],
                                                     mse_time['multiple_NN'], mse_time['multiple_GP']]

        mse_ratio_result.loc[len(mse_ratio_result)] = [mse_ratio['ratio_baseline'],
                                                       mse_ratio['single_linear_regression'], mse_ratio['single_SVM'],
                                                       mse_ratio['single_NN'], mse_ratio['single_GP'],
                                                       mse_ratio['MTL_GP'], mse_ratio['multiple_linear_regression'],
                                                       mse_ratio['multiple_SVM'], mse_ratio['multiple_NN'],
                                                       mse_ratio['multiple_GP']]

    if save_path is not None:
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        mse_time_result.to_csv(save_path + 'mse_time.csv')
        mse_ratio_result.to_csv(save_path + 'mse_ratio.csv')

    return mse_time_result, mse_ratio_result
