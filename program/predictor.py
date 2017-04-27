"""
Predict the dataset with the dataset
"""

# import the modules
import pandas as pd
from sklearn import linear_model, svm, neural_network
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
import matplotlib.pyplot as plt


# Read and divide the dataset
def split_dataset(dataset):
    training_set = dataset[dataset.service_date < 20160125].reset_index()
    test_set = dataset[dataset.service_date >= 20160125].reset_index()

    train_X = training_set.as_matrix(
        columns=['weather', 'rush_hour', 'baseline_result', 'delay_current_trip', 'delay_prev_trip', 'prev_arrival_time', 'delay_neighbor_stops'])
    train_Y = training_set.as_matrix(columns=['actual_arrival_time'])

    test_X = test_set.as_matrix(
        columns=['weather', 'rush_hour', 'baseline_result', 'delay_current_trip', 'delay_prev_trip', 'prev_arrival_time', 'delay_neighbor_stops'])
    test_Y = test_set.as_matrix(columns=['actual_arrival_time'])

    return train_X, train_Y, test_X, test_Y


# Use the linear model for regression
def build_result_dataset(train_X, train_Y, test_X, test_Y):
    result = pd.DataFrame(test_Y, columns=['actual_arrival_time'])
    result['baseline'] = test_X[:, 2]
    print 'linear regression'
    model = linear_model.LinearRegression()
    model.fit(train_X, train_Y)
    predict_Y = model.predict(test_X)
    print 'SVM'
    result['linear_regression'] = predict_Y
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
# single trip result
grouped = dataset.groupby(['trip_id'])
result_list = []
for name, item in grouped:
    single_trip = name
    current_dataset = item
    train_X, train_Y, test_X, test_Y = split_dataset(current_dataset)
    if len(train_X) < len(test_X) or len(test_X) == 0:
        continue
    current_result = build_result_dataset(train_X, train_Y, test_X, test_Y)
    result_list.append(current_result)
result = pd.concat(result_list, ignore_index=True)
result.to_csv('single_trip_result.csv')

# single route result
current_dataset = dataset
train_X, train_Y, test_X, test_Y = split_dataset(current_dataset)
result = build_result_dataset(train_X, train_Y, test_X, test_Y)
result.to_csv('single_route_result.csv')

