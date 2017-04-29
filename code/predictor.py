"""
Predict the dataset with the dataset
"""

# import the modules
import pandas as pd
from sklearn import linear_model, svm, neural_network
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
import matplotlib.pyplot as plt


# Read and divide the dataset
def split_dataset(dataset, feature_list):
    training_set = dataset[dataset.service_date < 20160125].reset_index()
    test_set = dataset[dataset.service_date >= 20160125].reset_index()

    train_X = training_set.as_matrix(
        columns=feature_list)
    train_Y = training_set.as_matrix(columns=['actual_arrival_time'])

    test_X = test_set.as_matrix(
        columns=feature_list)
    test_Y = test_set.as_matrix(columns=['actual_arrival_time'])

    return train_X, train_Y, test_X, test_Y


# Use the linear model for regression
def build_result_dataset(dataset, feature_list):
    training_set = dataset[dataset.service_date < 20160125].reset_index()
    test_set = dataset[dataset.service_date >= 20160125].reset_index()

    train_X = training_set.as_matrix(columns=feature_list)
    train_Y = training_set.as_matrix(columns=['actual_arrival_time'])

    test_X = test_set.as_matrix(columns=feature_list)
    test_Y = test_set.as_matrix(columns=['actual_arrival_time'])

    if len(train_X) < len(test_X) or len(test_X) == 0:
        return None

    result = pd.DataFrame(test_Y, columns=['actual_arrival_time'])
    result['baseline'] = test_set['baseline_result']
    print 'linear regression'
    model = linear_model.LinearRegression()
    model.fit(train_X, train_Y)
    predict_Y = model.predict(test_X)
    result['linear_regression'] = predict_Y
    print 'SVM'
    model = svm.SVR()
    model.fit(train_X, train_Y.ravel())
    predict_Y = model.predict(test_X)
    result['SVM'] = predict_Y
    print 'NN'
    model = neural_network.MLPRegressor(max_iter=1000, learning_rate_init=0.005)
    model.fit(train_X, train_Y.ravel())
    predict_Y = model.predict(test_X)
    result['NN'] = predict_Y
    return result


dataset = pd.read_csv('full_dataset_1.csv')


def run_compare_function(feature_list, dataset, pathname):
    root_path = '/Users/junwang/PycharmProjects/bus_arrival_prediction/program/'

    with open(pathname + 'descrip.txt', 'w') as f:
        f.write(str(feature_list))


    # single trip result
    grouped = dataset.groupby(['trip_id'])
    result_list = []
    for name, item in grouped:
        current_dataset = item
        # train_X, train_Y, test_X, test_Y = split_dataset(current_dataset, feature_list)
        # if len(train_X) < len(test_X) or len(test_X) == 0:
        #     continue
        current_result = build_result_dataset(current_dataset, feature_list)
        if current_result is None:
            continue
        result_list.append(current_result)
    result = pd.concat(result_list, ignore_index=True)
    result.to_csv(os.path.join(root_path + pathname, 'single_trip_result.csv'))


    # single route result
    current_dataset = dataset
    # train_X, train_Y, test_X, test_Y = split_dataset(current_dataset, feature_list)
    result = build_result_dataset(current_dataset, feature_list)
    result.to_csv(os.path.join(root_path + pathname, 'single_route_result.csv'))


complete_feature_list = ['weather', 'rush_hour', 'baseline_result', 'delay_current_trip', 'delay_prev_trip', 'prev_arrival_time', 'delay_neighbor_stops']

run_compare_function(complete_feature_list[:5], dataset, 'result/orig/')


for feature in complete_feature_list:
    feature_list = list(complete_feature_list)
    feature_list.remove(feature)
    if not os.path.exists('./result/1/' + feature):
        os.mkdir('./result/1/' + feature)
    run_compare_function(feature_list, dataset, 'result/1/' + feature + '/')


