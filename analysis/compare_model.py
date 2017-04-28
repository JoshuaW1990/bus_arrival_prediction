# import modules
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error as MSE
import json

# function for graphing
def compare_model(filename, path):
    # read the dataset
    plt.style.use('ggplot')
    output = pd.read_csv(path + filename)
    fig, axes = plt.subplots(nrows=2, ncols=2)
    # plot the scatter for the whole dataset
    output.plot(kind='scatter', y='baseline', x='actual_arrival_time', xlim=(0, 1500), ylim=(0, 1500), ax=axes[0,0])
    output.plot(kind='scatter', y='linear_regression', x='actual_arrival_time', xlim=(0, 1500), ylim=(0, 1500),
                ax=axes[0, 1])
    output.plot(kind='scatter', y='SVM', x='actual_arrival_time', xlim=(0, 1500), ylim=(0, 1500), ax=axes[1, 0])
    output.plot(kind='scatter', y='NN', x='actual_arrival_time', xlim=(0, 1500), ylim=(0, 1500), ax=axes[1, 1])
    fig.savefig(path + filename[:-4] + '.png')


def calculate_MSE(filename):
    result = dict()
    output = pd.read_csv(path + filename)
    columns = list(output.columns)
    columns.remove('actual_arrival_time')
    columns.remove('Unnamed: 0')
    for item in columns:
        tmp_result = MSE(output['actual_arrival_time'], output[item])
        result[item] = tmp_result
    return result


dir_list = os.listdir('./dataset/')
dir_list.remove('.DS_Store')
for dir_name in dir_list:
    print dir_name
    path = os.path.join('./dataset/' + dir_name + '/')
    file_list = os.listdir(path)
    result = []
    for filename in file_list:
        if filename.endswith('.csv'):
            print filename
            compare_model(filename, path)
            result.append({filename: calculate_MSE(filename)})
            print result
        else:
            continue

    with open(path + 'MSE_result.json', 'w') as f:
        json.dump(result, f)

# file_list = os.listdir('./')
# result = []
# for filename in file_list:
#     print filename
#     if filename.endswith('.csv'):
#         compare_model(filename)
#         result.append({filename: calculate_MSE(filename)})
#         print result
#     else:
#         continue
#
# with open('MSE_result.json', 'w') as f:
#     json.dump(result, f)

