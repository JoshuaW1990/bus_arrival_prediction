# import modules
import os
import numpy as np
import requests
import csv
import random
import pandas as pd
from datetime import datetime, timedelta, date
from dateutil.rrule import rrule, DAILY

# set the path
path = '../'


#################################################################################################################
#                                        segment.csv                                                            #
#################################################################################################################


def select_trip_list(num_route=None, direction_id=0):
    """
    Generate the list of the trip id for the selected routes
    :param num_route: the number of the selected routes. If the num_route is None, then all the route id will be selected
    :param direction_id: the direction id can be 0 or 1
    :return: the list of the trip_id
    """
    # Read the GTFS data
    # data source: MTA, state island, Jan, 4, 2016
    trips = pd.read_csv(path + 'data/GTFS/gtfs/trips.txt')
    route_stop_dist = pd.read_csv('route_stop_dist.csv')

    # select a specific route and the corresponding trips
    route_list = list(route_stop_dist.route_id)
    non_dup_route_list = sorted(list(set(route_list)))
    if num_route is None:
        select_routes = non_dup_route_list
    else:
        select_routes = non_dup_route_list[:num_route]
    selected_trips_var = []
    for route in select_routes:
        selected_trips_var += list(trips[(trips.route_id == route) & (trips.direction_id == direction_id)].trip_id)
    return selected_trips_var


def filter_history_data(date_start, date_end, selected_trips_var):
    # type: (object, object, object) -> object
    """
    Filtering the historical data to remove the unselected trips
    :rtype: object
    :param date_start: start date for historical date, int, yyyymmdd, ex: 20160109
    :param date_end: end date for historical date. Similar to date_start. The date_start and the date_end are included.
    :param selected_trips_var: the list of the trip_id for the selected routes
    :return: dataframe for the filtered historical data
    """
    # List the historical file
    file_list_var = os.listdir(path + 'data/history/')
    history_list = []
    print "filtering historical data"
    for filename in file_list_var:
        if not filename.endswith('.csv'):
            continue
        if filename[9:17] < str(date_start) or filename[9:17] > str(date_end):
            continue
        print filename
        ptr_history = pd.read_csv(path + 'data/history/' + filename)
        tmp_history = ptr_history[
            (ptr_history.trip_id.isin(selected_trips_var)) & (ptr_history.dist_along_route != '\N') & (
                ptr_history.dist_along_route != 0) & (ptr_history.progress == 0)]
        history_list.append(tmp_history)
    result = pd.concat(history_list)
    return result


def add_weather_info(weather, date_var):
    """
    add the weather information from the file: weather.csv
    The weather are expressed as below:
    0: sunny
    1: rainy
    2: snowy
    :param weather: the dataframe for weather.csv file
    :param date_var: the date for querying the weather
    :return: return the weather value today.
    """
    ptr_weather = weather[weather.date == date_var]
    if ptr_weather.iloc[0].snow == 1:
        weather_today = 2
    elif ptr_weather.iloc[0].rain == 1:
        weather_today = 1
    else:
        weather_today = 0
    return weather_today


def generate_original_segment(full_history_var, weather, stop_times_var):
    """
    Generate the original segment data
    Algorithm:
    Split the full historical data according to the service date, trip_id with groupby function
    For name, item in splitted historical dataset:
        service date, trip_id = name
        Split the item according to the vehicle_id, keep the data with the larget length of list for the vehicle_id
        calcualte the travel duration within the segement of this segment df and save the result into list
    concatenate the list
    :param full_history_var: the historical data after filtering
    :param weather: the dataframe for the weather information
    :param stop_times_var: the dataframe from stop_times.txt
    :return: dataframe for the original segment
    """
    full_history_var = full_history_var[full_history_var.service_date == 20160104]
    grouped = list(full_history_var.groupby(['service_date', 'trip_id']))
    print len(grouped)
    result_list = []
    for index in range(len(grouped)):
        name, single_history = grouped[index]
        if index % 150 == 0:
            print index
        service_date, trip_id = name
        if service_date <= 20160103:
            continue
        grouped_vehicle_id = list(single_history.groupby(['vehicle_id']))
        majority_length = -1
        majority_vehicle = -1
        majority_history = single_history
        for vehicle_id, item in grouped_vehicle_id:
            if len(item) > majority_length:
                majority_length = len(item)
                majority_history = item
                majority_vehicle = vehicle_id
        stop_sequence = [str(item) for item in list(stop_times_var[stop_times_var.trip_id == trip_id].stop_id)]
        current_segment_df = generate_original_segment_single_history(majority_history, stop_sequence)
        if current_segment_df is None:
            continue
        current_weather = add_weather_info(weather, service_date)
        current_segment_df['weather'] = current_weather
        day_of_week = datetime.strptime(str(service_date), '%Y%m%d').weekday()
        current_segment_df['service_date'] = service_date
        current_segment_df['day_of_week'] = day_of_week
        current_segment_df['trip_id'] = trip_id
        current_segment_df['vehicle_id'] = majority_vehicle
        result_list.append(current_segment_df)
    if result_list != []:
        result = pd.concat(result_list)
    else:
        return None
    return result


"""
Algorithm:
Filter the historical data with the stop sequence here
for i = 1, len(history) - 1:
    use prev and the next to mark the record:
        prev = history[i - 1]
        next = history[i]
    calculate the distance for prev and next respectively:
        prev_distance = prev.dist_along_route - prev.dist_from_stop
        next_distance = next.dist_along_route - next.dist_from_stop
    if prev_distance == next_distance or prev_distance = 0, continue to next row
    calcualte the time duration between the two spot:
        prev_time = datetime.strptime(prev.timestamp, '%Y-%m-%dT%H:%M:%SZ')
        next_time = ...
        travel_duration = next_time - prev_time

"""


def generate_original_segment_single_history(history, stop_sequence):
    """
    Calculate the travel duration for a single historical data
    Algorithm:
    Filter the historical data with the stop sequence here
    arrival_time_list = []
    for i = 1, len(history) - 1:
        use prev and the next to mark the record:
            prev = history[i - 1]
            next = history[i]
        calculate the distance for prev and next respectively:
            prev_distance = prev.dist_along_route - prev.dist_from_stop
            next_distance = next.dist_along_route - next.dist_from_stop
        if prev_distance == next_distance or prev_distance = 0, continue to next row
        distance_ratio = prev.dist_from_stop / (next_distance - prev_distance)
        calcualte the time duration between the two spot:
            prev_time = datetime.strptime(prev.timestamp, '%Y-%m-%dT%H:%M:%SZ')
            next_time = ...
            travel_duration = next_time - prev_time
        current_arrival_duration = travel_duration * distance_ratio
        current_arrival_time = current_arrival_duration + prev_time
        arrival_time_list.append((prev.next_stop_id, current_arrival_time))
    result = pd.Dataframe
    for i in range(1, len(arrival_time_list)):
        prev = arrival_time_list[i - 1]
        next = arrival_time_list[i]
        segment_start, segment_end obtained
        travel_duration = next[1] - prev[1]
        timestamp = prev[1]
        service_date = history[0].service_date
        ...
        save the record to result

    :param history: single historical data
    :param stop_sequence: stop sequence for the corresponding trip id
    :return: the dataframe of the origianl segment dataset
    """
    history = history[history.next_stop_id.isin(stop_sequence)]
    if len(history) < 3:
        return None
    arrival_time_list = []
    i = 1
    while i < len(history):
        prev_record = history.iloc[i - 1]
        next_record = history.iloc[i]
        while i < len(history) and stop_sequence.index(prev_record.next_stop_id) >= stop_sequence.index(next_record.next_stop_id):
            i += 1
            if i == len(history):
                break
            if stop_sequence.index(prev_record.next_stop_id) == stop_sequence.index(next_record.next_stop_id):
                prev_record = next_record
            next_record = history.iloc[i]
        if i == len(history):
            break
        # calculate the distance for prev and next respectively
        prev_distance = float(prev_record.dist_along_route) - float(prev_record.dist_from_stop)
        next_distance = float(next_record.dist_along_route) - float(next_record.dist_from_stop)
        if prev_distance == next_distance or prev_distance == 0:
            i += 1
            continue
        else:
            distance_ratio = float(prev_record.dist_from_stop) / (next_distance - prev_distance)
        # calcualte the time duration between the two spot
        prev_time = datetime.strptime(prev_record.timestamp, '%Y-%m-%dT%H:%M:%SZ')
        next_time = datetime.strptime(next_record.timestamp, '%Y-%m-%dT%H:%M:%SZ')
        travel_duration = next_time - prev_time
        travel_duration = travel_duration.total_seconds()
        # add it into the arrival time list
        current_arrival_duration = travel_duration * distance_ratio
        current_arrival_time = timedelta(0, current_arrival_duration) + prev_time
        arrival_time_list.append((prev_record.next_stop_id, current_arrival_time))
        i += 1
    result = pd.DataFrame(columns=['segment_start', 'segment_end', 'timestamp', 'travel_duration'])
    for i in range(1, len(arrival_time_list)):
        prev_record = arrival_time_list[i - 1]
        next_record = arrival_time_list[i]
        segment_start, segment_end = prev_record[0], next_record[0]
        timestamp = prev_record[1]
        travel_duration = next_record[1] - prev_record[1]
        travel_duration = travel_duration.total_seconds()
        result.loc[len(result)] = [segment_start, segment_end, timestamp, travel_duration]
    return result


file_list = os.listdir('./')
# export the segment data
print "export original_segment.csv file"
selected_trips = select_trip_list()
weather_df = pd.read_csv('weather.csv')
full_history = filter_history_data(20160104, 20160123, selected_trips)
stop_times = pd.read_csv(path + 'data/GTFS/gtfs/stop_times.txt')
segment_df = generate_original_segment(full_history, weather_df, stop_times)
segment_df.to_csv('original_segment.csv')
print "complete exporting the original_segement.csv file"
