"""
Feature selection:
compare the mse result with different feature list
"""

# import the modules
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn import linear_model, svm, neural_network, preprocessing
from sklearn.metrics import mean_squared_error as MSE
import json
import matplotlib.pyplot as plt
import GPy


def preprocess_dataset(origin_dataset):
    """
    Preprocess the dataset for obtaining the input matrix
    
    :param origin_dataset: original dataset read from the dataset table
    :return: the preprocessed dataset which only extracts the necessary information
    """
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
    """
    Split the dataset into training set and test set
    
    :param dataset: preprcessed dataset which only contains the necessary information
    :param feature_list: the list of features 
    :return: list of matrix for training set and test set
    """
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
    Predict the ratio: true_arrival_time/baseline_arrival_time
    
    :param X_train: the features of the training set
    :param X_test: the output of the training set
    :param y_train: the features of the test set
    :param y_test: the output of the test set
    :return: dataframe of the predicted result under different models
    """
    # generate the result for random samples
    ratio_result = pd.DataFrame(y_test, columns=['ratio_baseline'])

    print 'linear regression'
    model1 = linear_model.LinearRegression()
    model1.fit(X_train, y_train)
    y_pred = model1.predict(X_test)
    ratio_result['linear_regression'] = y_pred

    print 'SVM'
    model2 = svm.SVR()
    model2.fit(X_train, y_train)
    y_pred = model2.predict(X_test)
    ratio_result['SVM'] = y_pred

    print 'NN'
    model3 = neural_network.MLPRegressor(solver='lbfgs', max_iter=1000, learning_rate_init=0.005)
    model3.fit(X_train, y_train)
    y_pred = model3.predict(X_test)
    ratio_result['NN'] = y_pred

    print "Gaussian Process"
    m_full = GPy.models.SparseGPRegression(X_train, y_train.reshape(len(y_train), 1))
    m_full.optimize('bfgs')
    y_pred, y_var = m_full.predict(X_test)
    ratio_result['GP'] = y_pred

    return ratio_result


def check_performance(output_test, ratio_result):
    """
    Convert the ratio result into actual values and calculate the MSE for the ratios and actual values
    
    :param output_test: the list of information for each predicted outputs including the predicted result from the baseline algorithm and the actual arrival time. Each record of these information are necessary to assess the performance of the predicted result
    :param ratio_result: the predicted result from each model
    :return: the predicted arrival time, predicted ratio and the corresponding MSEs
    """
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


def export_files(time_result, mse_time, ratio_result, mse_ratio, feature, feature_list, save_path):
    """
    Export the results: ratios, actual values, MSEs
    
    :param time_result: the predicted arrival time under different models
    :param mse_time: the corresponding MSE for time_result
    :param ratio_result: the predicted ratio under different models
    :param mse_ratio: the corresponding MSE for ratio_result
    :param feature: the feature need to be removed
    :param feature_list: the list of remained features
    :param save_path: the path to export the files
    :return: None
    """
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    path = save_path + str(len(feature_list)) + '/' + feature + '/'

    if not os.path.exists(save_path + str(len(feature_list)) + '/'):
        os.mkdir(save_path + str(len(feature_list)) + '/')

    print "prepare to export file: ", path

    if not os.path.exists(path):
        os.mkdir(path)

    with open(path + 'descrip.txt', 'w') as f:
        f.write(str(feature_list))

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


#################################################################################################################
#                                    main function                                                              #
#################################################################################################################

def run_feature_selection(dataset, tablename=None, save_path=None, engine=None):
    """
    run an incomplete feature selection for the models
    
    :param dataset: the dataframe for dataset table
    :param save_path: the path to export result
    :param engine: the database connector
    :return: the dataframe of the feature selection under different models
    """
    plt.style.use('ggplot')
    dataset.reset_index(inplace=True)
    full_dataset = preprocess_dataset(dataset)
    features = ['weather', 'rush_hour', 'baseline_result', 'ratio_current_trip', 'ratio_prev_trip', 'prev_arrival_time']
    mse_compare = pd.DataFrame(
        columns=['feature_removed', 'time_baseline', 'time_linear_regression', 'time_SVM', 'time_NN', 'time_GP',
                 'ratio_baseline', 'ratio_linear_regression', 'ratio_SVM', 'ratio_NN', 'ratio_GP'])

    # complete features
    X_train, X_test, output_train, output_test = split_dataset(full_dataset, features)
    y_train = output_train[0]
    y_test = output_test[0]
    ratio_result = generate_ratio_result(X_train, X_test, y_train, y_test)
    time_result, mse_time, ratio_result, mse_ratio = check_performance(output_test, ratio_result)
    removed_features = ['None']

    if save_path is not None:
        export_files(time_result, mse_time, ratio_result, mse_ratio, 'AND'.join(removed_features), features, save_path)

    mse_compare.loc[len(mse_compare)] = ['AND'.join(removed_features), mse_time['baseline'],
                                         mse_time['linear_regression'], mse_time['SVM'], mse_time['NN'], mse_time['GP'],
                                         mse_ratio['ratio_baseline'], mse_ratio['linear_regression'], mse_ratio['SVM'],
                                         mse_ratio['NN'], mse_ratio['GP']]

    # remove one feature
    for i in xrange(len(features)):
        feature1 = features[i]
        tmp_features = list(features)
        tmp_features.remove(feature1)

        X_train, X_test, output_train, output_test = split_dataset(full_dataset, tmp_features)
        y_train = output_train[0]
        y_test = output_test[0]
        ratio_result = generate_ratio_result(X_train, X_test, y_train, y_test)
        time_result, mse_time, ratio_result, mse_ratio = check_performance(output_test, ratio_result)
        removed_features = [feature1]

        if save_path is not None:
            export_files(time_result, mse_time, ratio_result, mse_ratio, 'AND'.join(removed_features), tmp_features, save_path)

        mse_compare.loc[len(mse_compare)] = ['AND'.join(removed_features), mse_time['baseline'],
                                             mse_time['linear_regression'], mse_time['SVM'], mse_time['NN'],
                                             mse_time['GP'], mse_ratio['ratio_baseline'],
                                             mse_ratio['linear_regression'], mse_ratio['SVM'], mse_ratio['NN'],
                                             mse_ratio['GP']]

    # remove two features
    for i in xrange(len(features)):
        for j in xrange(i + 1, len(features)):
            feature1 = features[i]
            feature2 = features[j]
            tmp_features = list(features)

            tmp_features.remove(feature1)
            tmp_features.remove(feature2)

            print tmp_features

            X_train, X_test, output_train, output_test = split_dataset(full_dataset, tmp_features)
            y_train = output_train[0]
            y_test = output_test[0]

            ratio_result = generate_ratio_result(X_train, X_test, y_train, y_test)

            time_result, mse_time, ratio_result, mse_ratio = check_performance(output_test, ratio_result)

            removed_features = [feature1, feature2]

            if save_path is not None:
                export_files(time_result, mse_time, ratio_result, mse_ratio, 'AND'.join(removed_features), tmp_features, save_path)

            mse_compare.loc[len(mse_compare)] = ['AND'.join(removed_features), mse_time['baseline'],
                                                 mse_time['linear_regression'], mse_time['SVM'], mse_time['NN'],
                                                 mse_time['GP'], mse_ratio['ratio_baseline'],
                                                 mse_ratio['linear_regression'], mse_ratio['SVM'], mse_ratio['NN'],
                                                 mse_ratio['GP']]
    if save_path is not None:
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        mse_compare.to_csv(save_path + tablename + '.csv')
    if engine is not None:
        mse_compare.to_sql(name=tablename, con=engine, if_exists='replace', index_label='id')
    return mse_compare
