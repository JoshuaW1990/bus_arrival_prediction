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
        # # split the dataset according to the rush hour and plot the scatter
        # rush_hour = output['time_of_day'].apply(lambda x: x[11:19] < '20:00:00' and x[11:19] > '17:00:00')
        # output['rush_hour'] = rush_hour
        # grouped_output = output.groupby(['rush_hour'])
        # for name, item in grouped_output:
        #     filename = str(name) + model_name + '.png'
        #     figure = item.plot(kind='scatter', y='estimated_arrival_time', x='actual_arrival_time', xlim=(0, 10000),
        #                          ylim=(0, 10000))
        #     fig = figure.get_figure()
        #     fig.savefig(filename)
        # # split the dataset according to the rush hour and plot the scatter
        # weather_df = pd.read_csv('../program/weather.csv')
        # weather = output['service_date'].apply(lambda x: weather_df[weather_df.date == x].iloc[0].weather)
        # output['weather'] = weather
        # grouped_output = output.groupby(['weather'])
        # weather_dict = {}
        # weather_dict[0] = 'sunny'
        # weather_dict[1] = 'rainy'
        # weather_dict[2] = 'snowy'
        # for name, item, in grouped_output:
        #     filename = weather_dict[name] + model_name + '.png'
        #     figure = item.plot(kind='scatter', y='estimated_arrival_time', x='actual_arrival_time', xlim=(0, 10000),
        #                        ylim=(0, 10000))
        #     fig = figure.get_figure()
        #     fig.savefig(filename)

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
    if filename.endswith('.csv'):
        # compare_model(filename)
        result.append(calculate_MSE(filename))
    else:
        continue
with open('MSE_result.json', 'w') as f:
    json.dump(result, f)



