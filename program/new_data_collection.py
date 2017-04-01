"""
This datacollection try to reuse the method in different places such that all the process are similar, and we can avoid the error
"""


# import module

import pandas as pd
import os
from datetime import datetime, timedelta
from dateutil.rrule import rrule, DAILY
import requests
import csv




path = '../'

#################################################################################################################
#                                weather.csv                                                                    #
#################################################################################################################
def get_precip(gooddate):
    """
    Download the weather information for a specific date
    :param gooddate: date for downloading
    :return: list of the data
    """
    urlstart = 'http://api.wunderground.com/api/d083880ff5428216/history_'
    urlend = '/q/NY/New_York.json'

    url = urlstart + str(gooddate) + urlend
    data = requests.get(url).json()
    result = None
    for summary in data['history']['dailysummary']:
        rain = summary['rain']
        snow = summary['snow']
        if snow == '1':
            weather = '2'
        elif rain == '1':
            weather = '1'
        else:
            weather = '0'
        result = [gooddate, rain, snow, weather]
    return result


def download_weather(date_start, date_end):
    """
    download the weather information for a date range
    :param date_start: start date, string, ex: '20160101'
    :param date_end: similar to date_start
    :return: list of the table record
    weather = 2: snow
    weather = 1: rain
    weather = 0: sunny
    """

    a = datetime.strptime(date_start, '%Y%m%d')
    b = datetime.strptime(date_end, '%Y%m%d')

    result = pd.DataFrame(columns=['date', 'rain', 'snow', 'weather'])
    for dt in rrule(DAILY, dtstart=a, until=b):
        current_data = get_precip(dt.strftime("%Y%m%d"))
        if current_data is None:
            continue
        else:
            result.loc[len(result)] = current_data
    return result


#################################################################################################################
#                                route_stop_dist.csv                                                            #
#################################################################################################################
"""
Calcualte the distance of each stops for a specific route from the initial stop.

It will read three different files: trips.txt, stop_times.txt and history file.
Use the stop_times.txt and trips.txt file to obtain the stop sequence for each route and use the historical data to calculate the actual distance for each stop.
If the specific stop has no records for the distance, we will use the average value as the result like calculating the travel duration.

Since the dist_along_route in the history data is actually the distance between the next_stop and the intial stop, which decrease the difficulty a lot.
"""


def read_data(route_num=None, direction_id=0):
    # type: (object, object) -> object
    """
    Read all the corresponding data according to the requirements: number of the routes we need to calcualte.
    Input: route_num
    Output: Three different dataframe:
    trips, stop_times, history. All of these three data should have been filtered according to the trip_id and route_id
    """
    trips = pd.read_csv(path + 'data/GTFS/gtfs/trips.txt')
    stop_times = pd.read_csv(path + 'data/GTFS/gtfs/stop_times.txt')
    # Obtain the filterd trips dataframe
    route_list = list(trips.route_id)
    non_dup_route_list = sorted(list(set(route_list)))
    if route_num is None:
        selected_routes = non_dup_route_list
    else:
        selected_routes = non_dup_route_list[:route_num]
    result_trips = trips[(trips.route_id.isin(selected_routes)) & (trips.direction_id == direction_id)]
    # Obtain the filtered stop_times dataframe
    selected_trips_var = set(list(result_trips.trip_id))
    result_stop_times = stop_times[stop_times.trip_id.isin(selected_trips_var)]
    # Obtain the filtered history dataframe
    file_list = os.listdir(path + 'data/history/')
    file_list.sort()
    history_list = []
    for single_file in file_list:
        if not single_file.endswith('.csv'):
            continue
        else:
            current_history = pd.read_csv(path + 'data/history/' + single_file)
            tmp_history = current_history[current_history.trip_id.isin(selected_trips_var)]
            if len(tmp_history) == 0:
                continue
            else:
                print "historical file name: ", single_file
                history_list.append(tmp_history)
    result_history = pd.concat(history_list)
    print "complete reading data"
    return result_trips, result_stop_times, result_history


def calculate_stop_distance(trips, stop_times, history, direction_id=0):
    """
    Calculate the distance of each stop with its initial stop. Notice that the dist_along_route is the distance between the next_stop and the initial stop
    Input: three filtered dataframe, trips, stop_times, history
    Output: One dataframe, route_stop_dist
    The format of the route_stop_dist:
    route_id    direction_id    stop_id    dist_along_route
    str         int             int        float
    """
    result = pd.DataFrame(columns=['route_id', 'direction_id', 'stop_id', 'dist_along_route'])
    selected_routes = set(trips.route_id)
    # Looping from each route to obtain the distance of each stops
    for single_route in selected_routes:
        print "route name: ", single_route
        selected_trips_var = set(trips[trips.route_id == single_route].trip_id)
        stop_sequence = list(stop_times[stop_times.trip_id == list(selected_trips_var)[0]].stop_id)
        result.loc[len(result)] = [single_route, int(direction_id), int(stop_sequence[0]), 0.0]
        selected_history = history[history.trip_id.isin(selected_trips_var)]
        for i in range(1, len(stop_sequence)):
            stop_id = stop_sequence[i]
            current_history = selected_history[selected_history.next_stop_id == stop_id]
            if float(stop_id) == float(result.iloc[-1].stop_id):
                continue
            elif len(current_history) == 0:
                dist_along_route = -1.0
            else:
                current_dist = []
                for j in range(len(current_history)):
                    current_dist.append(current_history.iloc[j].dist_along_route)
                dist_along_route = sum(current_dist) / float(len(current_dist))
            result.loc[len(result)] = [single_route, int(direction_id), int(stop_id), dist_along_route]
    result.to_csv('original_route_stop_dist.csv')
    # Since some of the stops might not record, it is necessary to check the dataframe again.
    # Because of the bug or other reasons, some of the routes have a long jump in the stop list, we should remove the corresponding stop list
    count = 1
    prev = 0
    remove_route_list = set()
    for i in range(1, len(result) - 1):
        if result.iloc[i].dist_along_route == -1:
            if result.iloc[i - 1].dist_along_route != -1:
                prev = result.iloc[i - 1].dist_along_route
            count += 1
        else:
            if count != 1:
                if count >= 4:
                    remove_route_list.add(result.iloc[i - 1].route_id)
                distance = (float(result.iloc[i].dist_along_route) - float(prev)) / float(count)
                while count > 1:
                    result.iloc[i - count + 1, result.columns.get_loc('dist_along_route')] = result.iloc[
                                                                                                 i - count].dist_along_route + float(
                        distance)
                    count -= 1
            else:
                continue
    result.to_csv('original_improve_route_stop_dist.csv')
    result = result[~result.route_id.isin(remove_route_list)]
    return result


#################################################################################################################
#                   helper function for api data, segment data, and other calcualtion                           #
#################################################################################################################
"""
Helper functions for generating api data, segment data and even the arrival time

calcualte_arrival_time are used in:
generate_segment_data
generate_actual_arrival_time


calculate_arrival_distance are used in:
generate_api_data
"""




def calcualte_arrival_time(stop_dist, prev_dist, next_dist, prev_timestamp, next_timestamp):
    """
     Calculate the arrival time according to the given tuple (prev_dist, next_dist), the current location, the timestamp of the prev location, and the timestamp of the next location

    Algorithm:
    distance_prev_next = next_dist - prev_dist
    distance_prev_stop = stop_distance - prev_dist
    ratio = distance_prev_stop / distance_prev_next
    duration_prev_next = next_timestamp - prev_timestamp
    duration_prev_stop = duration_prev_next * ratio
    stop_timestamp = prev_timestamp + duration_prev_stop
    return the stop_timestamp

    :Param stop_dist: the distance of the target stop between the prev and next tuple
    :Param prev_dist: the distance of the location of the bus at the previous record
    :Param next_dist: the distance of the location of the bus at the next record
    :Param prev_timestamp: the timestamp of the bus at the previous record
    :Param next_timestamp: the timestamp of the bus at the next record
    :Return result: the timestamp of the bus arrival the target stop
    """
    distance_prev_next = next_dist - prev_dist
    distance_prev_stop = stop_dist - prev_dist
    ratio = float(distance_prev_stop) / float(distance_prev_next)
    duration_prev_next = next_timestamp - prev_timestamp
    duration_prev_stop = ratio * duration_prev_next.total_seconds()
    duration_prev_stop = timedelta(0, duration_prev_stop)
    stop_timestamp = prev_timestamp + duration_prev_stop
    return stop_timestamp


"""
calculate arrival distance according to the given input: time_of_day, prev_dist, next_dist, prev_timestamp, next_timestamp

Algorithm:
distance_bus_bus = next_dist - prev_dist
duration_bus_bus = next_timestamp - prev_timestamp
duration_bus_time = time_of_day - prev_timestamp
ratio = duration_bus_time / duration_bus_bus
distance_bus_time = distance_bus_bus * ratio
dist_along_route = distance_bus_time + prev_dist
return the dist_along_route
"""
def calculate_arrival_distance(time_of_day, prev_dist, next_dist, prev_timestamp, next_timestamp):
    """
    calculate arrival distance according to the given input: time_of_day, prev_dist, next_dist, prev_timestamp, next_timestamp

    Algorithm:
    distance_prev_next = next_dist - prev_dist
    duration_prev_next = next_timestamp - prev_timestamp
    duration_prev_time = time_of_day - prev_timestamp
    ratio = duration_prev_time / duration_prev_next
    distance_prev_time = distance_prev_next * ratio
    dist_along_route = distance_prev_time + prev_dist
    return the dist_along_route

    :Param time_of_day: the given time for calculating the dist_along_route
    :Param prev_dist: the distance of the location of the bus for the previous record in historical data
    :Param next_dist: the distance of the location of the bus for the next record in historical data
    :Param prev_timestamp: the timestamp of the bus for the previous record in historical data
    :Param next_timestamp: the timestamp of the bus for the next record in historical data
    :Return result: dist_along_route for the bus at the given time_of_day
    """
    duration_prev_next = next_timestamp - prev_timestamp
    duration_prev_time = time_of_day - prev_timestamp
    duration_prev_next = duration_prev_next.total_seconds()
    duration_prev_time = duration_prev_time.total_seconds()
    ratio = duration_prev_time / duration_prev_next
    distance_prev_next = next_dist - prev_dist
    distance_prev_time = distance_prev_next * ratio
    dist_along_route = prev_dist + distance_prev_time
    return dist_along_route


















#################################################################################################################
#                                    debug section                                                              #
#################################################################################################################
weather_df = download_weather('20160101', '20160131')



#################################################################################################################
#                                    main function                                                              #
#################################################################################################################


if __name__ == '__main__':
    file_list = os.listdir('./')
    # download weather information
    if 'weather.csv' not in file_list:
        print "download weather.csv file"
        weather_df = download_weather('20160101', '20160131')
        weather_df.to_csv('weather.csv')
        print "complete downloading weather information"
#     # export the route dist data
#     if 'route_stop_dist.csv' not in file_list:
#         print "export route_stop_dist.csv file"
#         trips, stop_times, history = read_data()
#         route_stop_dist = calculate_stop_distance(trips, stop_times, history)
#         route_stop_dist.to_csv('route_stop_dist.csv')
#         print "complete exporting the route_stop_dist.csv file"
#     # export the segment data
#     if 'original_segment.csv' not in file_list:
#         print "export original_segment.csv file"
#         selected_trips = select_trip_list()
#         weather_df = pd.read_csv('weather.csv')
#         full_history = filter_history_data(20160104, 20160123, selected_trips)
#         stop_times = pd.read_csv(path + 'data/GTFS/gtfs/stop_times.txt')
#         segment_df = generate_original_segment(full_history, weather_df, stop_times)
#         segment_df.to_csv('original_segment.csv')
#         print "complete exporting the original_segement.csv file"
#     if 'segment.csv' not in file_list:
#         print "export segment.csv file"
#         segment_df = improve_dataset()
#         segment_df.to_csv('segment.csv')
#         print "complete exporting the segment.csv file"
#     if "final_segment.csv" not in file_list:
#         print "export final segment.csv file"
#         segment_df = pd.read_csv('segment.csv')
#         final_segment = fix_bug_segment(segment_df)
#         final_segment.to_csv('final_segment.csv')
#         print "complete exporting the final_segment.csv file"
#     # export the api data
#     if 'api_data.csv' not in file_list:
#         print "export api_data.csv file"
#         date_list = range(20160125, 20160130)
#         route_stop_dist = pd.read_csv('route_stop_dist.csv')
#         stop_num = 4
#         route_list = list(set(route_stop_dist.route_id))
#         history_list = []
#         for current_date in date_list:
#             filename = 'bus_time_' + str(current_date) + '.csv'
#             history_list.append(pd.read_csv(path + 'data/history/' + filename))
#         full_history = pd.concat(history_list)
#         api_data_list = []
#         time_list = ['12:00:00', '12:05:00', '12:10:00', '12:15:00', '12:20:00', '12:25:00', '12:30:00']
#         current_api_data = generate_api_data(date_list, time_list, route_list, stop_num, route_stop_dist, full_history)
#         api_data_list.append(current_api_data)
#         time_list = ['18:00:00', '18:05:00', '18:10:00', '18:15:00', '18:20:00', '18:25:00', '18:30:00']
#         current_api_data = generate_api_data(date_list, time_list, route_list, stop_num, route_stop_dist, full_history)
#         api_data_list.append(current_api_data)
#         api_data = pd.concat(api_data_list)
#         api_data.to_csv('api_data.csv')
#         print "complete exporting the api_data.csv file"
#     print "complete data collection"
