# import modules
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error as MSE
import json

# function for graphing
def compare_model(filename):
    # read the dataset
    output = pd.read_csv(filename)
    fig, axes = plt.subplots(nrows=2, ncols=2)
    plt.style.use('ggplot')
    # plot the scatter for the whole dataset
    output.plot(kind='scatter', y='baseline', x='actual_arrival_time', xlim=(0, 1500), ylim=(0, 1500), ax=axes[0,0])
    output.plot(kind='scatter', y='linear_regression', x='actual_arrival_time', xlim=(0, 1500), ylim=(0, 1500),
                ax=axes[0, 1])
    output.plot(kind='scatter', y='SVM', x='actual_arrival_time', xlim=(0, 1500), ylim=(0, 1500), ax=axes[1, 0])
    output.plot(kind='scatter', y='NN', x='actual_arrival_time', xlim=(0, 1500), ylim=(0, 1500), ax=axes[1, 1])
    fig.savefig(filename[:-4] + '.png')


def calculate_MSE(filename):
    result = dict()
    output = pd.read_csv(filename)
    columns = list(output.columns)
    columns.remove('actual_arrival_time')
    columns.remove('Unnamed: 0')
    for item in columns:
        tmp_result = MSE(output['actual_arrival_time'], output[item])
        result[item] = tmp_result
    return result


file_list = os.listdir('./')
result = []
for filename in file_list:
    print filename
    if filename.endswith('.csv'):
        # compare_model(filename)
        result.append(calculate_MSE(filename))
        print result
    else:
        continue
# with open('MSE_result.json', 'w') as f:
#     json.dump(result, f)



