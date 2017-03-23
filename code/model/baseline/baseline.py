"""
Baseline algorithms:
1. simplest baseline
2. simple baseline
3. advanced baseline
"""


import pandas as pd
from datetime import datetime

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
        if i % 50 == 0:
            print "api index: ", i
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


def generate_actual_arrival_time(history, estimated_arrival_time, route_stop_dist):
    """
    Read the estimated_arrival_time dataframe and add the actual_arrival_time for each record in the dataframe for a specific given time and stop_id

    algorithm:
    Build the dataframe including the column of the actual_arrival_time
    For each record in estimated_arrival_time:
        Obtain the stop_id, time_of_day, trip_id
        Filter the history data into single_history data by trip_id
        Calculate the actual_arrival_time:
            If stop_id found in single_history.next_stop_id:
                Find the last one with the stop_id and the next one.
                Form a segment_pair with the previous step
            If stop_id not found in single_history.next_stop_id:
                Obtain the stop_sequence from the stop_times with trip_id
                Find the previous stop and the next stop according to the stop_sequence
                Use the nearest stops to form a segment_pair
            Calcualte the actual_arrival_time based on the ratio of the distance from the segment_pair
        add the actual_arrival_time into the new dataframe
    """
    # Build the dataframe including the column of the actual_arrival_time
    result = pd.DataFrame(columns=['trip_id', 'route_id', 'stop_id', 'vehicle_id', 'time_of_day', 'date', 'dist_along_route', 'stop_num_from_call', 'estimated_arrival_time', 'actual_arrival_time'])
    for i in xrange(len(estimated_arrival_time)):
        # Obtain the stop_id, time_of_day, trip_id
        stop_id = str(int(estimated_arrival_time.iloc[i].stop_id))
        time_of_day = str(estimated_arrival_time.iloc[i].time_of_day)
        trip_id = str(estimated_arrival_time.iloc[i].trip_id)
        date = int(estimated_arrival_time.iloc[i].date)
        vehicle_id = str(estimated_arrival_time.iloc[i].vehicle_id)
        dist_along_route = float(estimated_arrival_time.iloc[i].dist_along_route)
        stop_num_from_call = int(estimated_arrival_time.iloc[i].stop_num_from_call)
        route_id = str(estimated_arrival_time.iloc[i].route_id)
        # Filter the history data into single_history data by trip_id
        single_history = history[history.trip_id == trip_id]
        stop_set = set(single_history.next_stop_id)
        if stop_id in stop_set:
            index = list(single_history.next_stop_id).index(stop_id)
            while index < len(single_history) and single_history.iloc[index].next_stop_id == stop_id:
                index += 1
            start_index = index - 1
            if index == len(single_history):
                continue
            end_index = index
            if single_history.iloc[start_index].dist_from_stop == '0':
                time2 = datetime.strptime(single_history.iloc[start_index].timestamp[11:19], '%H:%M:%S')
                time1 = datetime.strptime(time_of_day, '%H:%M:%S')
                actual_arrival_time = time2 - time1
                actual_arrival_time = actual_arrival_time.total_seconds()
            else:
                time1 = datetime.strptime(single_history.iloc[start_index].timestamp[11:19], '%H:%M:%S')
                time2 = datetime.strptime(single_history.iloc[end_index].timestamp[11:19], '%H:%M:%S')
                distance_location1 = float(single_history.iloc[start_index].dist_along_route) - float(single_history.iloc[start_index].dist_from_stop)
                distance_location2 = float(single_history.iloc[end_index].dist_along_route) - float(single_history.iloc[end_index].dist_from_stop)
                travel_duration = time2 - time1
                distance_stop_location = float(single_history.iloc[start_index].dist_from_stop)
                ratio = distance_stop_location / (distance_location2 - distance_location1)
                travel_duration2 = travel_duration.total_seconds() * ratio
                time0 = datetime.strptime(time_of_day, '%H:%M:%S')
                travel_duration1 = time1 - time0
                actual_arrival_time = travel_duration2 + travel_duration1.total_seconds()
        else:
            stop_sequence = list(stop_times[stop_times.trip_id == trip_id].stop_id)
            index = stop_sequence.index(int(float(stop_id)))
            start_index = index - 1
            end_index = index + 1
            while start_index >= 0 and str(stop_sequence[start_index]) not in stop_set:
                start_index -= 1
            while end_index < len(stop_sequence) and str(stop_sequence[end_index]) not in stop_set:
                end_index += 1
            if start_index < 0 or end_index >= len(stop_sequence):
                continue
            start_stop = stop_sequence[start_index]
            end_stop = stop_sequence[end_index]
            if start_stop not in stop_set or end_stop not in stop_set:
                continue
            start_index = list(single_history.next_stop_id).index(start_stop)
            while single_history.iloc[start_index].next_stop_id == start_stop:
                start_index += 1
            start_index -= 1
            end_index = list(single_history.next_stop_id).index(end_stop)
            time0 = datetime.strptime(time_of_day, '%H:%M:%S')
            time1 = datetime.strptime(single_history.iloc[start_index].timestamp[11:19], '%H:%M:%S')
            time2 = datetime.strptime(single_history.iloc[end_index].timestamp[11:19], '%H:%M:%S')
            distance_location1 = float(single_history.iloc[start_index].dist_along_route) - float(single_history.iloc[start_index].dist_from_stop)
            distance_location2 = float(single_history.iloc[end_index].dist_along_route) - float(single_history.iloc[end_index].dist_from_stop)
            single_route_dist = route_stop_dist[route_stop_dist.route_id == route_id]
            travel_duration = time2 - time1
            distance_stop_location = float(single_route_dist[single_route_dist.stop_id == float(stop_id)].iloc[0].dist_along_route) - distance_location1
            ratio = distance_stop_location / (distance_location2 - distance_location1)
            travel_duration2 = travel_duration.total_seconds() * ratio
            travel_duration1 = time1 - time0
            actual_arrival_time = travel_duration2 + travel_duration1.total_seconds()
        result.loc[len(result)] = [trip_id, route_id, stop_id, vehicle_id, time_of_day, date, dist_along_route, stop_num_from_call, estimated_arrival_time.iloc[i].estimated_arrival_time, actual_arrival_time]
    return result


# Preparation of the data
original_segment_df = pd.read_csv('segment.csv')
segment_df = pd.read_csv('baseline2.csv')
stop_times = pd.read_csv('../../../data/GTFS/gtfs/stop_times.txt')
route_stop_dist = pd.read_csv('route_stop_dist.csv')
trips = pd.read_csv('../../../data/GTFS/gtfs/trips.txt')
api_df = pd.read_csv('api_data.csv')
direction_id = 0
# date = 20160128
# current_time = "12:20:19"


date_set = set(api_df.date)
date_list = list(date_set)
date_list.sort()
result_list = []
for date in date_list:
    # Extract the historical data
    print date
    weather = original_segment_df[original_segment_df.date == date].iloc[0].weather
    new_segment_df = segment_df[segment_df.weather == weather]
    filename = 'bus_time_' + str(date) +'.csv'
    history = pd.read_csv('../../../data/history/' + filename)
    single_api_df = api_df[api_df.date == date]
    estimated_arrival_time = generate_estimated_arrival_time(new_segment_df, api_df, stop_times, route_stop_dist)
    print "complete estimating"
    actual_arrival_time = generate_actual_arrival_time(history, estimated_arrival_time, route_stop_dist)
    print "complete actual computation"
    result_list.append(actual_arrival_time)
result = pd.concat(result_list)
print result.info()

