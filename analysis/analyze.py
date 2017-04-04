# import modules
import pandas as pd
import matplotlib.pyplot as plt
import os

#################################################################################################################
#                                    baseline algorithm                                                         #
#################################################################################################################
"""
Analyze for the output of the baseline algorithm:
1. simplest baseline
2. simple baseline
3. advanced baseline
"""

# function for graphing
def draw_scatter_graph(filename):
    # read the dataset
    output = pd.read_csv(filename)
    model_name = filename.split('.')[0]
    # plot the scatter for the whole dataset
    plt.style.use('ggplot')
    figure = output.plot(kind='scatter', y='estimated_arrival_time', x='actual_arrival_time', xlim=(0, 10000), ylim=(0, 10000))
    fig = figure.get_figure()
    fig.savefig(model_name + '.png')
    # split the dataset according to the rush hour and plot the scatter
    rush_hour = output['time_of_day'].apply(lambda x: x[11:19] < '20:00:00' and x[11:19] > '17:00:00')
    output['rush_hour'] = rush_hour
    grouped_output = output.groupby(['rush_hour'])
    for name, item in grouped_output:
        filename = str(name) + model_name + '.png'
        figure = item.plot(kind='scatter', y='estimated_arrival_time', x='actual_arrival_time', xlim=(0, 10000),
                             ylim=(0, 10000))
        fig = figure.get_figure()
        fig.savefig(filename)
    # split the dataset according to the rush hour and plot the scatter
    weather_df = pd.read_csv('../program/weather.csv')
    weather = output['service_date'].apply(lambda x: weather_df[weather_df.date == x].iloc[0].weather)
    output['weather'] = weather
    grouped_output = output.groupby(['weather'])
    weather_dict = {}
    weather_dict[0] = 'sunny'
    weather_dict[1] = 'rainy'
    weather_dict[2] = 'snowy'
    for name, item, in grouped_output:
        filename = weather_dict[name] + model_name + '.png'
        figure = item.plot(kind='scatter', y='estimated_arrival_time', x='actual_arrival_time', xlim=(0, 10000),
                           ylim=(0, 10000))
        fig = figure.get_figure()
        fig.savefig(filename)

file_list = os.listdir('./')
for filename in file_list:
    if filename.endswith('.csv'):
        draw_scatter_graph(filename)
    else:
        continue

