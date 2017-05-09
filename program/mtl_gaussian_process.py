"""
Test the multi task learning of gaussian process with GPy module
"""

# import module
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error as MSE
import GPy
import json

plt.style.use('ggplot')


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

    return X_train, X_test, output_train, output_test


def obtain_data_matrix(dataset, features):
    X_train_list = []
    X_test_list = None
    output_train_list = []
    output_test_list = None
    y_train_list = []
    y_test_list = None

    grouped = full_dataset.groupby(['shape_id'])
    for name, item in grouped:
        X_train, X_test, output_train, output_test = split_dataset(item, features)

        y_train = output_train[:, 0]
        y_test = output_test[:, 0]

        X_train_list.append(X_train)
        output_train_list.append(output_train)
        y_train_list.append(y_train.reshape(len(y_train), 1))

        if X_test_list is None:
            X_test_list = X_test
            output_test_list = output_test
            y_test_list = y_test
        else:
            X_test_list = np.concatenate((X_test_list, X_test))
            output_test_list = np.concatenate((output_test_list, output_test))
            y_test_list = np.concatenate((y_test_list, y_test))

    return X_train_list, X_test_list, output_train_list, output_test_list, y_train_list, y_test_list


dataset = pd.read_csv('full_dataset.csv')
full_dataset = preprocess_dataset(dataset)
features = ['weather', 'rush_hour', 'baseline_result', 'ratio_current_trip', 'ratio_prev_trip', 'prev_arrival_time']

# X_train_list, X_test_list, output_train_list, output_test_list, y_train_list, y_test_list = obtain_data_matrix(full_dataset, features)

X_train_list = []
X_test_list = []
output_train_list = []
output_test_list = []
y_train_list = []
y_test_list = []

grouped = full_dataset.groupby(['shape_id'])
for name, item in grouped:
    X_train, X_test, output_train, output_test = split_dataset(item, features)

    y_train = output_train[:, 0]
    y_test = output_test[:, 0]

    X_train_list.append(X_train)
    output_train_list.append(output_train)
    y_train_list.append(y_train.reshape(len(y_train), 1))

    X_test_list.append(X_test)
    output_test_list.append(output_test)
    y_test_list.append(y_test)

    # if X_test_list is None:
    #     X_test_list = X_test
    #     output_test_list = output_test
    #     y_test_list = y_test
    # else:
    #     X_test_list = np.concatenate((X_test_list, X_test))
    #     output_test_list = np.concatenate((output_test_list, output_test))
    #     y_test_list = np.concatenate((y_test_list, y_test))




# train model
print "Gaussian Process"
kernel = GPy.kern.Matern32(input_dim=6, ARD=True)
icm = GPy.util.multioutput.ICM(input_dim=6, num_outputs=len(X_train_list), kernel=kernel)
model = GPy.models.SparseGPCoregionalizedRegression(X_list=X_train_list, Y_list=y_train_list, kernel=icm)
model.optimize('bfgs')

# dataframe for random sampling test
print "random sampling test"
ratio_result_random = pd.DataFrame(np.concatenate(y_test_list), columns=['ratio_baseline'])
X_test = np.concatenate(X_test_list)
X_test = np.hstack((X_test, np.ones((len(X_test), 1))))
noise_dict = {'output_index': X_test[:, -1:].astype(int)}
y_pred, y_var = model.predict(X_test, Y_metadata=noise_dict)
ratio_result_random['GP'] = y_pred

# dataframe for sanity check
print "sanity check"
ratio_result_sanity = pd.DataFrame(np.concatenate(y_train_list), columns=['ratio_baseline'])
X_train = np.concatenate(X_train_list)
X_train = np.hstack((X_train, np.ones((len(X_train), 1))))
noise_dict = {'output_index': X_train[:, -1:].astype(int)}
y_pred, y_var = model.predict(X_train, Y_metadata=noise_dict)
ratio_result_sanity['GP'] = y_pred

# calculate the time result and mse
print "calculate the result and mse"
# random sampling test
# time result
output_test = np.concatenate(output_test_list)
time_result_random = pd.DataFrame(output_test[:, 2], columns=['actual'])
time_result_random['baseline'] = output_test[:, 1]
time_result_random['GP'] = time_result_random['baseline'] * ratio_result_random['GP']
# calculate the MSE of the arrival time
columns = time_result_random.columns
mse_time_random = dict()
for column in columns:
    if column == 'actual':
        continue
    mse_time_random[column] = MSE(time_result_random['actual'], time_result_random[column])
# ratio result
ratio_result_random['GP'] = ratio_result_random['ratio_baseline'] / ratio_result_random['GP']
# calculate the MSE of ratio
columns = ratio_result_random.columns
true_ratio = [1.0] * len(ratio_result_random)
mse_ratio_random = dict()
for column in columns:
    mse_ratio_random[column] = MSE(true_ratio, ratio_result_random[column])
# export file
time_result_random.to_csv('result/mtl_gp/time_result_random.csv')
ratio_result_random.to_csv('result/mtl_gp/ratio_result_random.csv')


# sanity check
# time result
output_train = np.concatenate(output_train_list)
time_result_sanity = pd.DataFrame(output_train[:, 2], columns=['actual'])
time_result_sanity['baseline'] = output_train[:, 1]
time_result_sanity['GP'] = time_result_sanity['baseline'] * ratio_result_sanity['GP']
# calculate the MSE of the arrival time
columns = time_result_sanity.columns
mse_time_sanity = dict()
for column in columns:
    if column == 'actual':
        continue
    mse_time_sanity[column] = MSE(time_result_sanity['actual'], time_result_sanity[column])
# ratio result
ratio_result_sanity['GP'] = ratio_result_sanity['ratio_baseline'] / ratio_result_sanity['GP']
# calculate the MSE of ratio
columns = ratio_result_sanity.columns
true_ratio = [1.0] * len(ratio_result_sanity)
mse_ratio_sanity = dict()
for column in columns:
    mse_ratio_sanity[column] = MSE(true_ratio, ratio_result_sanity[column])
# export file
time_result_sanity.to_csv('result/mtl_gp/time_result_sanity.csv')
ratio_result_sanity.to_csv('result/mtl_gp/ratio_result_sanity.csv')

result_dict = {}
result_dict['random_sampling'] = {"ratio": mse_ratio_random, "time": mse_time_random}
result_dict['sanity_check'] = {"ratio": mse_ratio_sanity, "time": mse_time_sanity}
result_df = pd.DataFrame(columns=['sampling method', 'baseline_ratio', 'GP_ratio', 'baseline_time', 'GP_time'])
for key in result_dict.keys():
    result_df.loc[len(result_df)] = [key, result_dict[key]['ratio']['ratio_baseline'], result_dict[key]['ratio']['GP'], result_dict[key]['time']['baseline'], result_dict[key]['time']['GP']]

result_df.to_csv('result/mtl_gp/result.csv')

with open('result/mtl_gp/result.json', 'w') as f:
    json.dump(result_dict, f)

