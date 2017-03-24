"""
This file is used to calculate the advanced baseline by adding the weather and the rush hour consideration.
Steps:
1. Generate the average segment pair data according to the rush hour and the weather
2. Read the api data and calculate the estimated arrival time
3. estimate the actual arrival time
4. Export the result

Algorithm:
read the api_df from api_data.csv file

extract the date list from api_df

for each date in date_list:

    read the historical data for the corresponding date

    obtain the single_api_df according to the date

    obtain the estimated_arrival_time according to the single_api_df

    obtain the actual_arrival_time according to the single_api_df

Concatenate the records
"""

# import the packages
import pandas as pd
import numpy as np
from datetime import datetime, timedelta





# 1. Generate the average segment pair data according to the rush hour and the weather
# add the code in the calculate_travel_duration.py function

# TODO 2. Read the api data and calculate the estimated arrival time

def generate_estimated_arrival_time(new_segment_df, api_df, stop_times, route_stop_dist):
    """
    Calculate the estimated arrival time based on the average travel duration under different condition
    :param api_df: api dataframe obtained from the historical data and the gtfs data
    :param stop_times: dataframe of the stop_times.txt file
    :param route_stop_dist: the distance between each stop and the corresponding initial stop in each route
    :return: a dataframe storing the the estimated arrival time
    Format for the returned dataframe:
    trip_id    route_id    stop_id    vehicle_id    time_of_day    date    dist_along_route    stop_num_from_call    estimated_arrival_time
     int         int        int         str           str          int         float                int/float             float
    """
    result = pd.DataFrame(
        columns=['trip_id', 'route_id', 'stop_id', 'vehicle_id', 'time_of_day', 'date', 'dist_along_route',
                 'stop_num_from_call', 'estimated_arrival_time'])
    for i in xrange(len(api_df)):
        date = str(api_df.iloc[i].date)
        single_trip = str(api_df.iloc[i].trip_id)
        trip_id = single_trip
        current_time = str(api_df.iloc[i].time_of_day)
        time_of_day = current_time
        stop_id = int(api_df.iloc[i].stop_id)
        dist_along_route = float(api_df.iloc[i].dist_along_route)
        route_id = str(api_df.iloc[i].route_id)
        vehicle_id = str(api_df.iloc[i].vehicle_id)
        # obtain the stop_sequence
        stop_sequence = list(stop_times[stop_times.trip_id == single_trip].stop_id)
        # obtain all the related segment pairs
        stop_dist_list = route_stop_dist[route_stop_dist.route_id == route_id]
        end_index = stop_sequence.index(stop_id)
        estimated_arrival_time = 0.0
        stop_num_from_call = 0
        if stop_dist_list[stop_dist_list.stop_id == stop_id].iloc[0].dist_along_route == dist_along_route:
            result.loc[len(result)] = [trip_id, route_id, stop_id, vehicle_id, time_of_day, date, dist_along_route,
                                       stop_num_from_call, estimated_arrival_time]
            continue
        elif stop_dist_list[stop_dist_list.stop_id == stop_id].iloc[0].dist_along_route < dist_along_route:
            print "already pass the stop"
            continue
        else:
            start_index = end_index - 1
            prev_stop = stop_sequence[start_index]
            while stop_dist_list[stop_dist_list.stop_id == prev_stop].iloc[0].dist_along_route > dist_along_route:
                start_index -= 1
                prev_stop = stop_sequence[start_index]
        stop_num_from_call = end_index - start_index
        segment_pair_list = []
        for index in range(start_index, end_index):
            current_segment = (stop_sequence[index], stop_sequence[index + 1])
            segment_pair_list.append(current_segment)
        for current_segment in segment_pair_list[1:]:
            estimated_arrival_time += float(
                new_segment_df[new_segment_df.segment_pair == str(current_segment)].iloc[0].travel_duration)
        # calculate the travel duration for the first segment pair
        distance_stop1 = stop_dist_list[stop_dist_list.stop_id == stop_sequence[start_index]].iloc[0].dist_along_route
        distance_stop2 = stop_dist_list[stop_dist_list.stop_id == stop_sequence[start_index + 1]].iloc[
            0].dist_along_route
        distance_stops = float(distance_stop2) - float(distance_stop1)
        distance_locations = dist_along_route - distance_stop1
        if distance_stop1 < dist_along_route and dist_along_route < distance_stop2:
            ratio = distance_locations / float(distance_stops)
            if ratio > 1:
                print "error: ratio > 1"
                print single_trip, segment_pair_list[0]
                print distance_stop1, dist_along_route, distance_stop2
                continue
            current_segment = segment_pair_list[0]
            travel_duration = new_segment_df[new_segment_df.segment_pair == str(current_segment)].iloc[
                0].travel_duration
            estimated_arrival_time += float(travel_duration) * float(ratio)
        else:
            print "error: distance_along_route incorrect"
            print single_trip, segment_pair_list[0]
            print distance_stop1, dist_along_route, distance_stop2
            continue
        result.loc[len(result)] = [trip_id, route_id, stop_id, vehicle_id, time_of_day, date, dist_along_route,
                                   stop_num_from_call, estimated_arrival_time]
    return result

"""
Algorithm:
for each record in single_api_df:

    Obtain the necessary information like trip_id, date, time_of_day, route_id, stop_id, dist_along_route, etc from that record of single_api_df

    Obtain the stop sequence according to the trip_id from the stop_times.txt file

    Obtain the stop_dist_list according to the route_id from the route_stop_dist.csv file

    Find the index of the stop_id from the stop_sequence

    If the dist_along_route from api_df is equal to the dist_along_route of the target stop from the stop_dist_list:

        This means that the bus is at that stop at that time point, so save the data into the result directly and continue to next records of the single_api_df

    If the dist_along_route from api_df is larger than the dist_along_route of the target stop from the stop_dist_list:

        This means that the bus has already passed this stop and continue to the next records of the single_api_df

    Start from the target stop and loop to the inital stop in the stop_dist_list until the dist_along_route from api_df is larger than the dist_along_route of a stop in the stop_dist_list. Claim that stop as prev_stop. All the segment_pairs between the target stop and the prev_stop is needed to be considered when calculating the estimated_arrival_time.

    Generate the segment_pair list for the segment_pairs we obtained from the last step.

    Loop in the segment_pair list and accumulate the travel duration for theses segment pairs except for the first segment_pair.

    Since the current location of the bus at the time point is within the first segment_pair of the segment_pair list, we calculate the duration for the first segment_pair by ratio of the distance as well.

    Add all these travel duration including calculated result for the first segment_pair up and the result is the estimated_arrival_time.

Concatenate the all the records
"""

# Preparation of the data
original_segment_df = pd.read_csv('segment.csv')
segment_df = pd.read_csv('baseline2.csv')
stop_times = pd.read_csv('../data/GTFS/gtfs/stop_times.txt')
route_stop_dist = pd.read_csv('route_stop_dist.csv')
direction_id = 0
date = 20160128
current_time = "12:20:19"
weather = original_segment_df[original_segment_df.date == date].iloc[0].weather
new_segment_df = segment_df[segment_df.weather == weather]

api_df = pd.read_csv('api_data.csv')
result = pd.DataFrame(columns=['trip_id', 'route_id', 'stop_id', 'vehicle_id', 'time_of_day', 'date', 'dist_along_route', 'stop_num_from_call', 'estimated_arrival_time'])
for i in xrange(len(api_df)):
    print i
    date = str(api_df.iloc[i].date)
    single_trip = str(api_df.iloc[i].trip_id)
    trip_id = single_trip
    current_time = str(api_df.iloc[i].time_of_day)
    time_of_day = current_time
    stop_id = int(api_df.iloc[i].stop_id)
    dist_along_route = float(api_df.iloc[i].dist_along_route)
    route_id = str(api_df.iloc[i].route_id)
    vehicle_id = str(api_df.iloc[i].vehicle_id)
    # obtain the stop_sequence
    stop_sequence = list(stop_times[stop_times.trip_id == single_trip].stop_id)
    # obtain all the related segment pairs
    stop_dist_list = route_stop_dist[route_stop_dist.route_id == route_id]
    end_index = stop_sequence.index(stop_id)
    estimated_arrival_time = 0.0
    stop_num_from_call = 0
    if stop_dist_list[stop_dist_list.stop_id == stop_id].iloc[0].dist_along_route == dist_along_route:
        result.loc[len(result)] = [trip_id, route_id, stop_id, vehicle_id, time_of_day, date, dist_along_route,
                                   stop_num_from_call, estimated_arrival_time]
        continue
    elif stop_dist_list[stop_dist_list.stop_id == stop_id].iloc[0].dist_along_route < dist_along_route:
        print "already pass the stop"
        continue
    else:
        start_index = end_index - 1
        prev_stop = stop_sequence[start_index]
        while stop_dist_list[stop_dist_list.stop_id == prev_stop].iloc[0].dist_along_route > dist_along_route:
            start_index -= 1
            prev_stop = stop_sequence[start_index]
    stop_num_from_call = end_index - start_index
    segment_pair_list = []
    for index in range(start_index, end_index):
        current_segment = (stop_sequence[index], stop_sequence[index + 1])
        segment_pair_list.append(current_segment)
    for current_segment in segment_pair_list[1:]:
        estimated_arrival_time += float(new_segment_df[new_segment_df.segment_pair == str(current_segment)].iloc[0].travel_duration)
    # calculate the travel duration for the first segment pair
    distance_stop1 = stop_dist_list[stop_dist_list.stop_id == stop_sequence[start_index]].iloc[0].dist_along_route
    distance_stop2 = stop_dist_list[stop_dist_list.stop_id == stop_sequence[start_index + 1]].iloc[
        0].dist_along_route
    distance_stops = float(distance_stop2) - float(distance_stop1)
    distance_locations = dist_along_route - distance_stop1
    if distance_stop1 < dist_along_route and dist_along_route < distance_stop2:
        ratio = distance_locations / float(distance_stops)
        if ratio > 1:
            print "error: ratio > 1"
            print single_trip, segment_pair_list[0]
            print distance_stop1, dist_along_route, distance_stop2
            continue
        current_segment = segment_pair_list[0]
        travel_duration = new_segment_df[new_segment_df.segment_pair == str(current_segment)].iloc[
            0].travel_duration
        estimated_arrival_time += float(travel_duration) * float(ratio)
    else:
        print "error: distance_along_route incorrect"
        print single_trip, segment_pair_list[0]
        print distance_stop1, dist_along_route, distance_stop2
        continue
    result.loc[len(result)] = [trip_id, route_id, stop_id, vehicle_id, time_of_day, date, dist_along_route,
                               stop_num_from_call, estimated_arrival_time]






# TODO 3. Estimate the actual arrival time
# TODO 4. Export the result




