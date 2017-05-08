"""

Used for debug

Currently check the abnormal route in route_stop_dist
"""

#
# file_list = os.listdir('./')
#
# if 'preprocessed_full_baseline_result.csv' not in file_list:
#     baseline_result = pd.read_csv('full_baseline_result.csv')
#     trips = pd.read_csv('../data/GTFS/gtfs/trips.txt')
#     trips_dict = trips.set_index('trip_id').to_dict(orient='index')
#     baseline_result['shape_id'] = baseline_result['trip_id'].apply(lambda x: trips_dict[x]['shape_id'])
#     baseline_result.to_csv('preprocessed_full_baseline_result.csv')
# else:
#     baseline_result = pd.read_csv('preprocessed_full_baseline_result.csv')
#
#
# original_route_stop_dist = pd.read_csv('origin_route_stop_dist.csv')
# route_stop_dist = pd.read_csv('route_stop_dist.csv')
#
# original_grouped = original_route_stop_dist.groupby(['route_id'])
# grouped = route_stop_dist.groupby(['route_id'])
#
# shape_id_set = set()
#
# for route_id, item in grouped:
#     sub_grouped = item.groupby(['shape_id'])
#     original_single_route_stop_dist = original_grouped.get_group(route_id)
#     original_stop_list = list(original_single_route_stop_dist['stop_id'])
#     for shape_id, single_route_stop_dist in sub_grouped:
#         stop_list = list(single_route_stop_dist['stop_id'])
#         if original_stop_list == stop_list:
#             shape_id_set.add(shape_id)
#
# final_baseline_result = baseline_result[baseline_result.shape_id.isin(shape_id_set)]
# final_baseline_result.to_csv('full_baseline_result.csv')

# import pandas as pd
# import psycopg2
# from sqlalchemy import create_engine
#
# # db = pymysql.connect(host='localhost', user='root', password='Wj15029054380', db='bus_prediction')
#
#
# # read the full history data into database
#
#
#
# engine = create_engine('postgresql://joshuaw:Wj2080989@localhost:5432/bus_prediction', echo=False)
# full_api_data = pd.read_csv('full_api_data.csv')
# full_api_data.drop(['Unnamed: 0'], axis=1, inplace=True)
# full_api_data.to_sql(name='full_api_data', con=engine, if_exists='replace', index_label='id')
# print "complete save full_api_data into database"
#
# full_api_data_baseline3 = pd.read_csv('tmp.csv')
# full_api_data_baseline3.drop(['Unnamed: 0'], axis=1, inplace=True)
# full_api_data_baseline3.to_sql(name='full_api_baseline', con=engine, if_exists='replace', index_label='id')

# import the modules
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model, svm, neural_network, preprocessing
from sklearn.metrics import mean_squared_error as MSE
import json
import matplotlib.pyplot as plt
import GPy

# plt.style.use('ggplot')
#
# origin_dataset = pd.read_csv("full_dataset.csv")
# origin_dataset.reset_index(inplace=True)
# origin_dataset.drop(['Unnamed: 0'], axis=1, inplace=True)
#
# # shape_id = origin_dataset.iloc[0]['shape_id']
# # origin_dataset = origin_dataset[origin_dataset.shape_id == shape_id]
#
# # preprocess to obtain the dataset
# full_dataset = pd.DataFrame()
# full_dataset['weather'] = origin_dataset['weather']
# full_dataset['rush_hour'] = origin_dataset['rush_hour']
# full_dataset['baseline_result'] = origin_dataset['baseline_result']
# full_dataset['ratio_baseline'] = origin_dataset['actual_arrival_time'] / origin_dataset['baseline_result']
# full_dataset['ratio_current_trip'] = origin_dataset['ratio_current_trip']
# full_dataset['ratio_prev_trip'] = origin_dataset['ratio_prev_trip']
# full_dataset['ratio_prev_seg'] = origin_dataset['actual_arrival_time'] / origin_dataset['prev_arrival_time']
# full_dataset['prev_arrival_time'] = origin_dataset['prev_arrival_time']
# full_dataset['actual_arrival_time'] = origin_dataset['actual_arrival_time']
#
# X = full_dataset.as_matrix(
#     columns=['weather', 'rush_hour', 'baseline_result', 'ratio_current_trip', 'ratio_prev_trip', 'prev_arrival_time'])
# y = full_dataset.as_matrix(columns=['ratio_baseline', 'baseline_result', 'actual_arrival_time'])
#
# # normalization
# X_normalized = preprocessing.normalize(X, norm='l2')
#
# # split the dataset
# X_train, X_test, output_train, output_test = train_test_split(X_normalized, y, test_size=0.33, random_state=42)
#
# output_train = output_train.transpose()
# output_test = output_test.transpose()
# y_train = output_train[0]
# y_test = output_test[0]
#
#
# # generate the result for random samples
# ratio_result = pd.DataFrame(y_test, columns=['ratio_baseline'])
# print 'linear regression'
# model1 = linear_model.LinearRegression()
# model1.fit(X_train, y_train)
# y_pred = model1.predict(X_test)
# ratio_result['linear_regression'] = y_pred
# print 'SVM'
# model2 = svm.SVR()
# model2.fit(X_train, y_train)
# y_pred = model2.predict(X_test)
# ratio_result['SVM'] = y_pred
# print 'NN'
# model3 = neural_network.MLPRegressor(solver='lbfgs', max_iter=1000, learning_rate_init=0.005)
# model3.fit(X_train, y_train)
# y_pred = model3.predict(X_test)
# ratio_result['NN'] = y_pred
# print "Gaussian Process"
# m_full = GPy.models.SparseGPRegression(X_train, y_train.reshape(len(y_train), 1))
# m_full.optimize('bfgs')
# y_pred, y_var = m_full.predict(X_test)
# ratio_result['GP'] = y_pred
#
#
# # process the result to obtain the pred arrival time with different models
# time_result = pd.DataFrame(output_test[2], columns=['actual'])
# time_result['baseline'] = output_test[1]
# time_result['linear_regression'] = time_result['baseline'] * ratio_result['linear_regression']
# time_result['SVM'] = time_result['baseline'] * ratio_result['SVM']
# time_result['NN'] = time_result['baseline'] * ratio_result['NN']
# time_result['GP'] = time_result['baseline'] * ratio_result['GP']
#
# # calculate the MSE of the arrival time
# columns = time_result.columns
# mse_time = dict()
# for column in columns:
#     if column == 'actual':
#         continue
#     mse_time[column] = MSE(time_result['actual'], time_result[column])
#     filename = column + '.png'
#     figure = time_result.plot(kind='scatter', y=column, x='actual', xlim=(0, 6000), ylim=(0, 6000))
#     fig = figure.get_figure()
#     fig.savefig('result/compare_sanity/random_sample/' + filename)
#
# # export data
# time_result.to_csv('result/compare_sanity/random_sample/result_time.csv')
# with open('result/compare_sanity/random_sample/mse_time.json', 'w') as f:
#     json.dump(mse_time, f)
#
#
# # process the result to obtain the ratio(actual_arrival_time / pred_arrival_time)
# ratio_result['linear_regression'] = ratio_result['ratio_baseline'] / ratio_result['linear_regression']
# ratio_result['SVM'] = ratio_result['ratio_baseline'] / ratio_result['SVM']
# ratio_result['NN'] = ratio_result['ratio_baseline'] / ratio_result['NN']
# ratio_result['GP'] = ratio_result['ratio_baseline'] / ratio_result['GP']
#
# # calculate the MSE of ratio
# columns = ratio_result.columns
# true_ratio = [1.0] * len(ratio_result)
# mse_ratio = dict()
# for column in columns:
#     mse_ratio[column] = MSE(true_ratio, ratio_result[column])
#
# # export data
# ratio_result.to_csv('result/compare_sanity/random_sample/result_ratio.csv')
# with open('result/compare_sanity/random_sample/mse_ratio.json', 'w') as f:
#     json.dump(mse_ratio, f)


with open('result/1/feature_selection.json', 'r') as f:
    feature_select = json.load(f)


result = pd.DataFrame(columns=['feature_removed', 'time_baseline', 'time_linear_regression', 'time_SVM', 'time_NN', 'time_GP', 'ratio_baseline', 'ratio_linear_regression', 'ratio_SVM', 'ratio_NN', 'ratio_GP'])

for feature in feature_select.keys():
    result.loc[len(result)] = [feature, feature_select[feature]['mse_time']['baseline'], feature_select[feature]['mse_time']['linear_regression'], feature_select[feature]['mse_time']['SVM'], feature_select[feature]['mse_time']['NN'], feature_select[feature]['mse_time']['GP'], feature_select[feature]['mse_ratio']['ratio_baseline'], feature_select[feature]['mse_ratio']['linear_regression'], feature_select[feature]['mse_ratio']['SVM'], feature_select[feature]['mse_ratio']['NN'], feature_select[feature]['mse_ratio']['GP']]

result.to_csv('result/1/mse_compare.csv')
