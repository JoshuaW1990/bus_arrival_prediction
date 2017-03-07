# This file is used for the baseline 1 algorithm

"""
This file is composed of two different parts. The first part is used to read the historical and shedule data and extract the api information (distance, current time, stop_num_from_call). The second part is used to read the baseline1.csv file and predict according to the average segment pair travel duration.
"""


# import the module
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

##############################################################################################
# First part: read the data: historical file and the schedule file, extract the api information
##############################################################################################


def read_dataset():
    """
    read the data.csv file to obtain the segment dataset
    """
    segment_df = pd.read_csv('data.csv')
    return segment_df

# Obtain the set of the segment_pair
def obtain_segment_set(segment_df):
    segment_set = set(segment_df.segment_pair)
    # Loopin through all the segment in the set and calculate the average travel duration
    # Since during calculating the travel duration, according to the first algorithm, many unecessary information will be ignored, we will buid a much simpler dataframe for storing the result
    # The format of the new dataframe:
    #    segment_start    segment_end    segment_pair    travel_duration
    # 		str              str           (str, str)      float(second)
    new_segment_df = pd.DataFrame(columns = ['segment_start', 'segment_end', 'segment_pair', 'travel_duration'])
    for segment_pair in segment_set:
        tmp_segment_df = segment_df[segment_df.segment_pair == segment_pair]
        num = float(len(tmp_segment_df))
        average_travel_duration = sum(list(tmp_segment_df.travel_duration)) / num
        segment_start = segment_pair.split(',')[0][1:]
        segment_end = segment_pair.split(',')[1][:-1]
        new_segment_df.loc[len(new_segment_df)] = [segment_start, segment_end, segment_pair, average_travel_duration]
    return new_segment_df


def obtain_api_data(date, current_time, stop_times, direction, stop_id):
    """
    This file is used to obtain the api data from the MTA.

    For the first step, we use the GTFS scheduled data and the historical data:
    1. read the GTFS schedule data to obtain the stop sequence of the single trip.
    2. read the historical data to obtain the current location, distance, etc.

    This function return a dataframe of storing the api data from the schedule data and the historical data. The format of the dataframe is below:
    trip_id    vehicle_id    dist_along_route    stop_num_from_call
    str         float              float              float
    """
    # Generate the selected trip set
    # Extract the scheduled GTFS data
    trips = pd.read_csv('../data/GTFS/gtfs/trips.txt')
    selected_trips = set(list(stop_times[stop_times.stop_id == stop_id].trip_id))
    # Filtering the trips according to the direction
    removed_trips = set()
    for single_trip in selected_trips:
        if trips[trips.trip_id == single_trip].iloc[0].direction_id == direction:
            continue
        else:
            removed_trips.add(single_trip)
    selected_trips = selected_trips - removed_trips

    # Extract the historical data
    filename = 'bus_time_' + str(date) +'.csv'
    history = pd.read_csv('../data/history/' + filename)
    # Filtering the trips according to the operation hour
    removed_trips = set()
    for i, single_trip in enumerate(selected_trips):
        if i % 50 == 0:
            print i
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
    result = pd.DataFrame(columns=['trip_id', 'vehicle_id', 'dist_along_route', 'stop_num_from_call'])
    for single_trip in selected_trips:
        print single_trip
        # generate stop sequence
        stop_sequence = list(stop_times[stop_times.trip_id == single_trip].stop_id)
        # obtain the time segment containing the specific time point
        single_history = history[history.trip_id == single_trip]
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
        print tmp_history.iloc[0].next_stop_id, tmp_history.iloc[1].next_stop_id, current_dist
        dist_along_route = float(distance_stop1) + current_dist
        """
        The dist_from_next_stop is harder to be calculated. Considering that this value might not be used in the baseline model, thus we ignore this parameter and continue.
        """
        # Obtain the stop_num_from_call by loop
        current_stop_index = stop_sequence.index(int(tmp_history.iloc[1].next_stop_id)) - 1
        target_stop_index = stop_sequence.index(stop_id)
        stop_num_from_call = target_stop_index - current_stop_index
        if stop_num_from_call < 0:
            print "stop_num_from_call < 0: ", single_trip
            continue
        result.loc[len(result)] = [single_trip, history[history.trip_id == single_trip].iloc[10].vehicle_id, dist_along_route, stop_num_from_call]
    return result, trips


def generate_arrival_time(date, stop_times, trips, route_stop_dist, current_time, direction_id, stop_id):
    """
    This file is used to generate the arrival time according to the specific direction, stop_id, and the given local time.
    Input:
    Output: dataframe
    trip_id    vehicle_id    time_of_day    dist_along_route    stop_num_from_call    estimated_arrival_time
     int         str           str             float                int/float             float
    """
    # TODO change the api_df here or in the obtain_api_data function
    api_df = pd.read_csv('api_data.csv')
    selected_trips = set(api_df.trip_id)
    result = pd.DataFrame(columns=['trip_id', 'vehicle_id', 'time_of_day', 'dist_along_route', 'stop_num_from_call', 'estimated_arrival_time'])
    for single_trip in selected_trips:
        # obtain the dist_along_route and the stop_num_from_call according to the api data
        dist_along_route = float(api_df[api_df.trip_id == single_trip].dist_along_route)
        stop_num_from_call = int(api_df[api_df.trip_id == single_trip].stop_num_from_call)
        # obtain the stop_sequence
        stop_sequence = list(stop_times[stop_times.trip_id == single_trip].stop_id)
        # obtain all the related segment pairs
        route_id = trips[trips.trip_id == single_trip].iloc[0].route_id
        stop_dist_list = route_stop_dist[route_stop_dist.route_id == route_id]
        end_index = stop_sequence.index(stop_id)
        start_index = end_index - int(stop_num_from_call)
        segment_pair_list = []
        for index in range(start_index, end_index - 1):
            current_segment = (stop_sequence[index], stop_sequence[index + 1])
            segment_pair_list.append(current_segment)
        # calculate the estimated time based on the segment pair list
        estimated_arrival_time = 0.0
        # if segment_pair_list == []:
        #     print start_index, end_index
        #     print "estimated time is", estimated_arrival_time
        #     continue
        for current_segment in segment_pair_list[1:]:
            estimated_arrival_time += float(new_segment_df[new_segment_df.segment_pair == str(current_segment)].iloc[0].travel_duration)
        # calculate the travel duration for the first segment pair
        distance_stop1 = stop_dist_list[stop_dist_list.stop_id == stop_sequence[start_index]].iloc[0].dist_along_route
        distance_stop2 = stop_dist_list[stop_dist_list.stop_id == stop_sequence[start_index + 1]].iloc[0].dist_along_route
        distance_stops = float(distance_stop2) - float(distance_stop1)
        distance_locations = dist_along_route - distance_stop1
        if distance_stop1 < dist_along_route and dist_along_route < distance_stop2:
            ratio = distance_locations / float(distance_stops)
            if ratio > 1:
                print "error: ", single_trip, segment_pair_list[0]
            current_segment = segment_pair_list[0]
            travel_duration = new_segment_df[new_segment_df.segment_pair == str(current_segment)].iloc[0].travel_duration
            estimated_arrival_time += float(travel_duration) * float(ratio)
        else:
            print "error: ",  single_trip, segment_pair_list[0]
            print distance_stop1, dist_along_route, distance_stop2
            return
        print single_trip, dist_along_route, stop_id, estimated_arrival_time
        trip_id = single_trip
        time_of_day = current_time
        vehicle_id = str(api_df[api_df.trip_id == single_trip].iloc[0].vehicle_id)
        result.loc[len(result)] = [trip_id, vehicle_id, time_of_day, dist_along_route, stop_num_from_call, estimated_arrival_time]
    return result

def prediction_baseline1(date, stop_times, time, direction_id, stop_id):
    """
    This function is used to predict the arrival time for a specific time, stop
    Read the data from the `new_segment_df` dataframe, which stores the average travel duration for all the segment pairs we can find the dataframe.

    Algorithm:
    selected_trip = set(trip_id)
    for single_trip in selected_trip:
        1. obtain the stop sequence for the single_trip
        2. according to the stop_num_from_call and stop_sequence, obtain the current segment and the other related segment pair.
        3. Calculate the estimated arrival time based on the distance and the average segment pair travel duration
    """
    pass

# # prepare data
# segment_df = read_dataset()
# new_segment_df = obtain_segment_set(segment_df)
# stop_times = pd.read_csv('../data/GTFS/gtfs/stop_times.txt')
# trips = pd.read_csv('../data/GTFS/gtfs/trips.txt')
# route_stop_dist = pd.read_csv('route_stop_dist.csv')
# stop_id = 201495
# direction_id = 0
# date = 20160128
# current_time = "12:20:19"
#
# estimated_arrival_time = generate_arrival_time(date, stop_times, trips, route_stop_dist, current_time, direction_id, stop_id)


# code for function: generate_arrival_time
# selected_trips = set(api_df.trip_id)
# result = pd.DataFrame(columns=['trip_id', 'vehicle_id', 'time_of_day', 'dist_along_route', 'stop_num_from_call', 'estimated_arrival_time'])
# for single_trip in selected_trips:
#     # obtain the dist_along_route and the stop_num_from_call according to the api data
#     dist_along_route = float(api_df[api_df.trip_id == single_trip].dist_along_route)
#     stop_num_from_call = int(api_df[api_df.trip_id == single_trip].stop_num_from_call)
#     # obtain the stop_sequence
#     stop_sequence = list(stop_times[stop_times.trip_id == single_trip].stop_id)
#     # obtain all the related segment pairs
#     route_id = trips[trips.trip_id == single_trip].iloc[0].route_id
#     stop_dist_list = route_stop_dist[route_stop_dist.route_id == route_id]
#     end_index = stop_sequence.index(stop_id)
#     start_index = end_index - int(stop_num_from_call)
#     segment_pair_list = []
#     for index in range(start_index, end_index - 1):
#         current_segment = (stop_sequence[index], stop_sequence[index + 1])
#         segment_pair_list.append(current_segment)
#     # calculate the estimated time based on the segment pair list
#     estimated_arrival_time = 0.0
#     # if segment_pair_list == []:
#     #     print start_index, end_index
#     #     print "estimated time is", estimated_arrival_time
#     #     continue
#     for current_segment in segment_pair_list[1:]:
#         estimated_arrival_time += float(new_segment_df[new_segment_df.segment_pair == str(current_segment)].iloc[0].travel_duration)
#     # calculate the travel duration for the first segment pair
#     distance_stop1 = stop_dist_list[stop_dist_list.stop_id == stop_sequence[start_index]].iloc[0].dist_along_route
#     distance_stop2 = stop_dist_list[stop_dist_list.stop_id == stop_sequence[start_index + 1]].iloc[0].dist_along_route
#     distance_stops = float(distance_stop2) - float(distance_stop1)
#     distance_locations = dist_along_route - distance_stop1
#     if distance_stop1 < dist_along_route and dist_along_route < distance_stop2:
#         ratio = distance_locations / float(distance_stops)
#         if ratio > 1:
#             print "error: ", single_trip, segment_pair_list[0]
#         current_segment = segment_pair_list[0]
#         travel_duration = new_segment_df[new_segment_df.segment_pair == str(current_segment)].iloc[0].travel_duration
#         estimated_arrival_time += float(travel_duration) * float(ratio)
#     else:
#         print "error: ",  single_trip, segment_pair_list[0]
#         print distance_stop1, dist_along_route, distance_stop2
#         return
#     print single_trip, dist_along_route, stop_id, estimated_arrival_time
#     trip_id = single_trip
#     time_of_day = current_time
#     vehicle_id = str(api_df[api_df.trip_id == single_trip].iloc[0].vehicle_id)
#     result.loc[len(result)] = [trip_id, vehicle_id, time_of_day, dist_along_route, stop_num_from_call, estimated_arrival_time]




if __name__ == "__main__":
   segment_df = read_dataset()
   new_segment_df = obtain_segment_set(segment_df)
   # # export the file
   # new_segment_df.to_csv('baseline1.csv')
   stop_times = pd.read_csv('../data/GTFS/gtfs/stop_times.txt')
   trips = pd.read_csv('../data/GTFS/gtfs/trips.txt')
   route_stop_dist = pd.read_csv('route_stop_dist.csv')
   stop_id = 201495
   direction_id = 0
   date = 20160128
   current_time = "12:20:19"
   # api_df, trips = obtain_api_data(date, current_time, stop_times, direction_id, stop_id)
   # api_df.to_csv("api_data.csv")
   estimated_arrival_time = generate_arrival_time(date, stop_times, trips, route_stop_dist, current_time, direction_id, stop_id)
   print estimated_arrival_time
