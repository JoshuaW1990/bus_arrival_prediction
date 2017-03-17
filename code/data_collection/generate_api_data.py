"""
According to the given route list, date list and the time list, generate the corresponding api_data.

Four stops will be selected according to the route_stop_dist randomly

The final return dataframe will be
trip_id    vehicle_id    route_id    stop_id    time_of_day    date    dist_along_route    stop_num_from_call
str         float          int         int          str        str              float              float
"""

import pandas as pd
import random
from datetime import datetime, timedelta


def obtain_api_data(history, trips, route_id, current_time, stop_times, direction, stop_id):
    """
    Obtain the api data for a history file of a specific date, time and stop_id from the MTA.

    For the first step, we use the GTFS scheduled data and the historical data:
    1. read the GTFS schedule data to obtain the stop sequence of the single trip.
    2. read the historical data to obtain the current location, distance, etc.

    This function return a dataframe of storing the api data from the schedule data and the historical data. The format of the dataframe is below:
    trip_id    vehicle_id    route_id    time_of_day    date    dist_along_route    stop_num_from_call
    str         float          int         str          str         float              float
    """
    # Obtain the date
    date = int(float(history.iloc[-1].service_date))
    # Generate the selected trip set
    # Extract the scheduled GTFS data
    selected_trips = set(stop_times[stop_times.stop_id == stop_id].trip_id) & set(trips[trips.route_id == route_id].trip_id)
    # Filtering the trips according to the direction
    removed_trips = set()
    print "length of the selected_trips: ", len(selected_trips)
    for single_trip in selected_trips:
        if trips[trips.trip_id == single_trip].iloc[0].direction_id == direction:
            continue
        else:
            removed_trips.add(single_trip)
    selected_trips = selected_trips - removed_trips

    # Filtering the trips according to the operation hour
    removed_trips = set()
    print "length of the selected_trips: ", len(selected_trips)
    for i, single_trip in enumerate(selected_trips):
        if len(history[history.trip_id == single_trip]) != 0:
            time_start = history[history.trip_id == single_trip].iloc[0].timestamp[11:19]
            time_end = history[history.trip_id == single_trip].iloc[-1].timestamp[11:19]
            if time_start < current_time and time_end > current_time:
                continue
            else:
                removed_trips.add(single_trip)
        else:
            removed_trips.add(single_trip)
    selected_trips = selected_trips - removed_trips

    # According to the selected trips set, for each trip, generate the api data from GTFS and historicald data
    """
    The api data we need for prediction for a specific bus at a specific time and stop:
    - dist_along_route(at the given time point)
    - dist_from_next_stop(at the given time point)
    - stop_num_from_call(...)

    algorithm to obtain these api data
    for each single_trip in selected_trips:
        Get the single_trip_stop_sequence
        Get the single_trip_history
        Looping in the single_trip_history to obtain the time segment containing the specific time point
        Calculate the distance between the location of the two time point
        Calcualte the time duration between the current time point and the previous time point
        Use the ratio of the time duration and the travel duration to estimate the distance, then we can get the dist_along_route, and the dist_from_next_stop
        According to the stop_sequence, obtain the stop_num_from_call
    """
    result = pd.DataFrame(columns=['trip_id', 'vehicle_id', 'route_id', 'stop_id', 'time_of_day', 'date', 'dist_along_route', 'stop_num_from_call'])
    print "length of the selected_trips: ", len(selected_trips)
    for single_trip in selected_trips:
        # generate stop sequence
        stop_sequence = list(stop_times[stop_times.trip_id == single_trip].stop_id)
        stop_sequence = [str(int(item)) for item in stop_sequence]
        # obtain the time segment containing the specific time point
        single_history = history[(history.trip_id == single_trip) & (history.next_stop_id.isin(stop_sequence))]
        tmp_history = None
        for i in xrange(1, len(single_history)):
            time_prev = single_history.iloc[i - 1].timestamp[11:19] # str
            time_next = single_history.iloc[i].timestamp[11:19] # str
            if time_prev <= current_time and time_next >= current_time:
                # Some trips are operated in the midnight
                if single_history.iloc[i - 1].service_date != single_history.iloc[i].service_date:
                    break
                else:
                    tmp_history = single_history[i - 1: i + 1]
                    break
            else:
                continue
        if tmp_history is None:
            continue
        """
        Calculate the distance between the two points:
        distance_stop1 = dist_along_route - dist_from_next_stop
        distance_stop2 = ...
        distance_between_points = distance_stop2 - distance_stop1
        time1, time2, time3
        ratio = (time2 - time1) / (time3 - time2)
        dist_along_route = distance_stop1 + (distance_between_points * ratio)
        """
        print single_trip
        distance_stop1 = float(tmp_history.iloc[0].dist_along_route) - float(tmp_history.iloc[0].dist_from_stop)
        distance_stop2 = float(tmp_history.iloc[1].dist_along_route) - float(tmp_history.iloc[1].dist_from_stop)
        distance_between_points =  distance_stop2 - distance_stop1
        time1 = datetime.strptime(time_prev, '%H:%M:%S')
        time2 = datetime.strptime(current_time, '%H:%M:%S')
        time3 = datetime.strptime(time_next, '%H:%M:%S')
        segment_duration = time3 - time1
        current_duration = time2 - time1
        ratio = current_duration.total_seconds() / segment_duration.total_seconds()
        current_dist = distance_between_points * ratio
        dist_along_route = float(distance_stop1) + current_dist
        """
        The dist_from_next_stop is harder to be calculated. Considering that this value might not be used in the baseline model, thus we ignore this parameter and continue.
        """
        # Obtain the stop_num_from_call by loop
        current_stop_index = stop_sequence.index(tmp_history.iloc[1].next_stop_id) - 1
        target_stop_index = stop_sequence.index(str(int(stop_id)))
        stop_num_from_call = target_stop_index - current_stop_index
        if stop_num_from_call < 0:
            continue
        route_id = trips[trips.trip_id == single_trip].iloc[0].route_id
        result.loc[len(result)] = [single_trip, history[history.trip_id == single_trip].iloc[10].vehicle_id, route_id, int(stop_id), current_time, str(date), dist_along_route, int(stop_num_from_call)]
        print "within single data, length of the api data: ", len(result)
    return result

def generate_complete_api_input(date_list, route_list):
    """
    Generate the complete api input data according to a specific date list, route list
    """
    pass


# test
date_list = range(20160129, 20160130)
new_segment_df = pd.read_csv('average_segment_travel_duration.csv')
stop_times = pd.read_csv('../data/GTFS/gtfs/stop_times.txt')
trips = pd.read_csv('../data/GTFS/gtfs/trips.txt')
route_stop_dist = pd.read_csv('route_stop_dist.csv')
route_set = set(route_stop_dist.route_id)
#stop_id = 201495
direction_id = 0
time_init1 = "12:00:00"
time_init2 = "18:00:00"

#####################################################################
# test: debug start
#####################################################################
# stop_id =  203613.0
# route_id =  'X11'
# filename = 'bus_time_20160128.csv'
# history = pd.read_csv('../data/history/' + filename)
# current_time = '12:00:00'
# api_data = obtain_api_data(history, trips, route_id, current_time, stop_times, direction_id, stop_id)
# print "complete!"

#####################################################################
# test: debug end
#####################################################################


"""
Algorithm:
for date in date_list:
    read the history file for the date
    for each route in route_set
        randomly pick four stops from the stop_list for the route
        for each stop in stops:
            for i in range():
                generate current time
                obtain the api_data
                result_list.append(api_data)
result = pd.concat(result_list)
"""

single_route_stop_dist = route_stop_dist[route_stop_dist.route_id.isin(route_set)]
time_start1 = datetime.strptime(time_init1, '%H:%M:%S')
time_start2 = datetime.strptime(time_init2, '%H:%M:%S')
result_list = []
for date in date_list:
    # Extract the historical data
    filename = 'bus_time_' + str(date) +'.csv'
    print filename
    history = pd.read_csv('../data/history/' + filename)
    for route_id in route_set:
        stop_sequence = list(single_route_stop_dist[single_route_stop_dist.route_id == route_id].stop_id)
        stop_set = set()
        while len(stop_set) < 4:
            current_stop = stop_sequence[random.randint(2, len(stop_sequence) - 3)]
            stop_set.add(current_stop)
        for stop_id in stop_set:
            print "route_id = ",  route_id, "stop_id = ", stop_id
            # for i in range(0, 7):
            #     delta = timedelta(0, i * 300)
            #     current_time = time_start1 + delta
            #     current_time = str(current_time)[11:19]
            #     print "current time: ", current_time
            #     api_data = obtain_api_data(history, trips, route_id, current_time, stop_times, direction_id, stop_id)
            #     result_list.append(api_data)
            for i in range(0, 7):
                delta = timedelta(0, i * 300)
                current_time = time_start2 + delta
                current_time = str(current_time)[11:19]
                print "current time: ", current_time
                api_data = obtain_api_data(history, trips, route_id, current_time, stop_times, direction_id, stop_id)
                result_list.append(api_data)
result = pd.concat(result_list)
result.to_csv('new_api_data.csv')
