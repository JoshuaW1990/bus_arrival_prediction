import data_collection

import pandas as pd

# setting for path

save_path = '/Users/junwang/PycharmProjects/bus_arrival_prediction/preprocess/example/'
example_path = '/Users/junwang/PycharmProjects/bus_arrival_prediction/example_output/preprocessed_data/'
history_path = '/Users/junwang/PycharmProjects/bus_arrival_prediction/data/history/'
gtfs_path = '/Users/junwang/PycharmProjects/bus_arrival_prediction/data/GTFS/gtfs/'

# read files
weather_df = pd.read_csv(example_path + 'weather.csv')
stop_times = pd.read_csv(gtfs_path + 'stop_times.txt')
trips = pd.read_csv(gtfs_path + 'trips.txt')
history = pd.read_csv(example_path + 'history.csv')
route_stop_dist = pd.read_csv(example_path + 'route_stop_dist.csv')
date_list = [20160106, 20160109]
time_list = ['12:15:00', '12:20:00']

"""
User can export the files into csv format as follow
"""
# weather
# suppose api token is 'API'
weather = data_collection.obtain_weather('20160101', '20160106', 'API', save_path+'weather.csv')

# history
# download history data
data_collection.download_history_file(2016, 1, [1, 3, 5], history_path)
# export history table
history = data_collection.obtain_history(20160105, 20160106, trips, history_path, save_path+'history.csv')

# route_stop_dist
route_stop_dist = data_collection.obtain_route_stop_dist(trips, stop_times, history, save_path+'route_stop_dist.csv')

# segment
segment = data_collection.obtain_segment(weather_df, trips, stop_times, route_stop_dist, history, date_list, save_path+'segment.csv')

# api_data
api_data = data_collection.obtain_api_data(route_stop_dist, history, date_list, time_list, 3, save_path+'api_data')

"""
If user prefer to use database like postgres, please ignore the save_path attribute such that the program will not export the dataframe table into csv file. Then user can use the return value of each function to save them into database.
"""
# weather
# suppose api token is 'API'
weather = data_collection.obtain_weather('20160101', '20160106', 'API')

# history
# download history data
data_collection.download_history_file(2016, 1, [1, 3, 5], history_path)
# export history table
history = data_collection.obtain_history(20160105, 20160106, trips, history_path)

# route_stop_dist
route_stop_dist = data_collection.obtain_route_stop_dist(trips, stop_times, history)

# segment
segment = data_collection.obtain_segment(weather_df, trips, stop_times, route_stop_dist, history, date_list)

# api_data
api_data = data_collection.obtain_api_data(route_stop_dist, history, date_list, time_list, 3)
