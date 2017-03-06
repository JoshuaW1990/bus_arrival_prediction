"""
This file is used to calcualte the distance of each stops for a specific route from the initial stop.

It will read three different files: trips.txt, stop_times.txt and history file.
Use the stop_times.txt and trips.txt file to obtain the stop sequence for each route and use the historical data to calculate the actual distance for each stop.
If the specific stop has no records for the distance, we will use the average value as the result like calculating the travel duration.

Since the dist_along_route in the history data is actually the distance between the next_stop and the intial stop, which decrease the difficulty a lot.
"""

import pandas as pd
import numpy as np
import os

def read_data(route_num, direction_id):
    """
    Read all the corresponding data according to the requirements: number of the routes we need to calcualte.
    Input: route_num
    Output: Three different dataframe:
    trips, stop_times, history. All of these three data should have been filtered according to the trip_id and route_id
    """
    trips = pd.read_csv('../data/GTFS/gtfs/trips.txt')
    stop_times = pd.read_csv('../data/GTFS/gtfs/stop_times.txt')
    # Obtain the filterd trips dataframe
    route_list = list(trips.route_id)
    non_dup_route_list = [route_list[0]]
    for i in xrange(1, len(route_list)):
        if route_list[i] == non_dup_route_list[-1]:
            continue
        else:
            non_dup_route_list.append(route_list[i])
    selected_routes = non_dup_route_list[:route_num]
    result_trips = trips[(trips.route_id.isin(selected_routes)) & (trips.direction_id == direction_id)]
    # Obtain the filtered stop_times dataframe
    selected_trips = set(list(result_trips.trip_id))
    result_stop_times = stop_times[stop_times.trip_id.isin(selected_trips)]
    # Obtain the filtered history dataframe
    file_list = os.listdir('../data/history/')
    history_list = []
    for single_file in file_list:
        if not single_file.endswith('.csv'):
            continue
        else:
            current_history = pd.read_csv('../data/history/' + single_file)
            tmp_history = current_history[current_history.trip_id.isin(selected_trips)]
            if len(tmp_history) == 0:
                continue
            else:
                print single_file
                history_list.append(tmp_history)
    result_history = pd.concat(history_list)
    return result_trips, result_stop_times, result_history

def calculate_stop_distance(trips, stop_times, history, direction_id):
    """
    Calculate the distance of each stop with its inital stop. Notice that the dist_along_route is the distance between the next_stop and the initial stop
    Input: three filtered dataframe, trips, stop_times, history
    Output: One dataframe, route_stop_dist
    The format of the route_stop_dist:
    route_id    direction_id    stop_id    dist_along_route
    str         int             str        float
    """
    result = pd.DataFrame(columns=['route_id', 'direction_id', 'stop_id', 'dist_along_route'])
    selected_routes = set(trips.route_id)
    # Looping from each route to obtain the distance of each stops
    for single_route in selected_routes:
        selected_trips = set(trips[trips.route_id == single_route].trip_id)
        stop_sequence = list(stop_times[stop_times.trip_id == list(selected_trips)[0]].stop_id)
        result.loc[len(result)] = [single_route, direction_id, stop_sequence[0], 0.0]
        selected_history = history[history.trip_id.isin(selected_trips)]
        for i in range(1, len(stop_sequence)):
            stop_id = stop_sequence[i]
            current_history = selected_history[selected_history.next_stop_id == stop_id]
            if stop_id == str(result.iloc[-1].stop_id):
                continue
            elif len(current_history) == 0:
                dist_along_route = None
            else:
                current_dist = []
                for i in range(len(current_history)):
                    current_dist.append(current_history.iloc[i].dist_along_route)
                dist_along_route = sum(current_dist) / float(len(current_dist))
            result.loc[len(result)] = [single_route, direction_id, stop_id, dist_along_route]
    result.to_csv('original_route_dist.csv')
    # Since some of the stops might not record, it is necessary to check the dataframe again.
    count = 1
    for i in range(1, len(result) - 1):
        if result.iloc[i].dist_along_route == None:
            if result.iloc[i - 1].dist_along_route != None:
                prev = result.iloc[i - 1].dist_along_route
            count += 1
        else:
            if count != 1:
                distance = (float(result.iloc[i].dist_along_route) - float(prev)) / float(count)
                while count > 1:
                    result.iloc[i - count + 1, result.columns.get_loc('dist_along_route')] = result.iloc[i - count].dist_along_route + float(distance)
                    count -= 1
            else:
                continue
    return result

if __name__ == "__main__":
    route_num = 2
    direction_id = 0
    trips, stop_times, history = read_data(route_num, direction_id)
    route_stop_dist = calculate_stop_distance(trips, stop_times, history, direction_id)
    route_stop_dist.to_csv('route_stop_dist.csv')
