"""
Predict the dataset with the dataset
"""

# import the modules
import pandas as pd
from sklearn import linear_model, svm, neural_network, preprocessing
import os
from sklearn.model_selection import train_test_split
import GPy
from sklearn.metrics import mean_squared_error as MSE
import matplotlib.pyplot as plt


# Read and divide the dataset
def split_dataset(original_dataset, feature_list):
    origin_dataset = pd.read_csv("full_dataset.csv")
    origin_dataset.reset_index(inplace=True)
    origin_dataset.drop(['Unnamed: 0'], axis=1, inplace=True)


    # preprocess
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

    X = full_dataset.as_matrix(
        columns=feature_list)
    y = full_dataset.as_matrix(columns=['ratio_baseline']).reshape((1, len(full_dataset)))[0]

    X_normalized = preprocessing.normalize(X, norm='l2')

    X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.33, random_state=42)

    return X_train, X_test, y_train, y_test



# Use the linear model for regression
def build_result_dataset(dataset, feature_list):

    train_X, test_X, train_Y, test_Y = split_dataset(dataset, feature_list)

    if len(train_X) < len(test_X) or len(test_X) == 0:
        return None

    result = pd.DataFrame(test_Y, columns=['ratio_baseline'])
    # result['baseline'] = test_set['baseline_result']
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
    model = neural_network.MLPRegressor(activation='logistic',solver='sgd', max_iter=1000, learning_rate_init=0.005)
    model.fit(train_X, train_Y.ravel())
    predict_Y = model.predict(test_X)
    result['NN'] = predict_Y
    print "Gaussian Process"
    m_full = GPy.models.SparseGPRegression(train_X, train_Y.reshape(len(train_Y), 1))
    m_full.optimize('bfgs')
    y_pred, y_var = m_full.predict(test_X)
    # m_full.plot()
    result['GP'] = y_pred

    result['linear_regression'] = result['ratio_baseline'] / result['linear_regression']
    result['SVM'] = result['ratio_baseline'] / result['SVM']
    result['NN'] = result['ratio_baseline'] / result['NN']
    result['GP'] = result['ratio_baseline'] / result['GP']

    return result




def run_compare_function(feature_list, dataset, pathname):
    root_path = '/Users/junwang/PycharmProjects/bus_arrival_prediction/program/'

    with open(pathname + 'descrip.txt', 'w') as f:
        f.write(str(feature_list))


    # single shape result
    grouped = dataset.groupby(['shape_id'])
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
    result.to_csv(os.path.join(root_path + pathname, 'single_shape_result.csv'))


    # single route result
    current_dataset = dataset
    # train_X, train_Y, test_X, test_Y = split_dataset(current_dataset, feature_list)
    result = build_result_dataset(current_dataset, feature_list)
    result.to_csv(os.path.join(root_path + pathname, 'single_result.csv'))


complete_feature_list = ['weather', 'rush_hour', 'baseline_result', 'ratio_current_trip', 'ratio_prev_trip', 'prev_arrival_time']

# run_compare_function(complete_feature_list[:5], dataset, 'result/orig/')


dataset = pd.read_csv('full_dataset.csv')

# for feature in complete_feature_list:
#     feature_list = list(complete_feature_list)
#     feature_list.remove(feature)
#     if not os.path.exists('./result/1/' + feature):
#         os.mkdir('./result/1/' + feature)
#     run_compare_function(feature_list, dataset, 'result/1/' + feature + '/')

run_compare_function(complete_feature_list, dataset, 'result/0/')

