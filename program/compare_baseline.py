"""
Calculate the data for the baseline algorithm with the unit functions
"""

# import modules
import pandas as pd
import os
from datetime import timedelta, datetime

path = '../'

#################################################################################################################
#                                data preproces                                                                 #
#################################################################################################################
"""
Calculate the average travel duration for the segment data
1. simplest baseline
2. simple baseline
3. advanced baseline
Use the groupby function for the segment dataframe
"""


def preprocess_baseline1(segment_df):
    """
    preprocession for the simplest baseline: not considering the weather and the time
    Algorithm:
    Read the database
    Group the dataframe according to the segment start and the segment end
    For each item in the grouped list:
        obtain the name and the sub dataframe
        check whether the segment_start and the segment_end is the same (we need to fix this bug later when retrieving the segment data)
        Calculate the average travel duration
        save the record into the new dataframe
    :param segment_df:
    :return: the preprocessed segment dataframe
    """
    grouped_list = list(segment_df.groupby(['segment_start', 'segment_end']))
    print "length of the grouped list: ", len(grouped_list)
    result = pd.DataFrame(columns=['segment_start', 'segment_end', 'travel_duration'])
    for i in xrange(len(grouped_list)):
        if i % 100 == 0:
            print i
        name, item = grouped_list[i]
        segment_start, segment_end = name
        if segment_start == segment_end:
            continue
        travel_duration_list = list(item['travel_duration'])
        average_travel_duration = sum(travel_duration_list) / float(len(travel_duration_list))
        result.loc[len(result)] = [segment_start, segment_end, average_travel_duration]
    return result


def preprocess_baseline2(segment_df, rush_hour):
    """
    Preprocess the segment data considering the weather and the rush hour

    Algorithm:
    Preprocess segment_df to add a new column of rush hour
    split the dataframe with groupby(segment_start, segment_end, weather, rush_hour)
    Define the new dataframe
    For name, item in grouped:
        calcualte the average travel duration
        save the record into the new dataframe

    :param segment_df: dataframe after adding the rush hour from final_segment.csv file
    :param rush_hour: tuple to express which is the rush hour, example: ('17:00:00', '20:00:00')
    :return: dataframe for the baseline2
    """
    # Preprocess segment_df to add a new column of rush hour
    rush_hour_column = segment_df['timestamp'].apply(lambda x: x[11:19] < rush_hour[1] and x[11:19] > rush_hour[0])
    new_segment_df = segment_df
    new_segment_df['rush_hour'] = rush_hour_column
    grouped_list = list(new_segment_df.groupby(['segment_start', 'segment_end', 'weather', 'rush_hour']))
    print "length of the grouped_list is: ", len(grouped_list)
    result = pd.DataFrame(columns=['segment_start', 'segment_end', 'travel_duration', 'weather', 'rush_hour'])
    for i in xrange(len(grouped_list)):
        if i % 1000 == 0:
            print i
        name, item = grouped_list[i]
        segment_start, segment_end, weather, rush_hour_var = name
        travel_duration_list = list(item['travel_duration'])
        average_travel_duration = sum(travel_duration_list) / float(len(travel_duration_list))
        result.loc[len(result)] = [segment_start, segment_end, average_travel_duration, weather, rush_hour_var]
    return result


#################################################################################################################
#                   helper function for api data, segment data, and other calcualtion                           #
#################################################################################################################
"""
Helper functions for generating api data, segment data and even the arrival time

calculate_arrival_time are used in:
generate_segment_data
generate_actual_arrival_time


calculate_arrival_distance are used in:
generate_api_data
"""


def calculate_arrival_time(stop_dist, prev_dist, next_dist, prev_timestamp, next_timestamp):
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

    :param stop_dist: the distance of the target stop between the prev and next tuple
    :param prev_dist: the distance of the location of the bus at the previous record
    :param next_dist: the distance of the location of the bus at the next record
    :param prev_timestamp: the timestamp of the bus at the previous record
    :param next_timestamp: the timestamp of the bus at the next record
    :return result: the timestamp of the bus arrival the target stop
    """
    distance_prev_next = next_dist - prev_dist
    distance_prev_stop = stop_dist - prev_dist
    ratio = float(distance_prev_stop) / float(distance_prev_next)
    duration_prev_next = next_timestamp - prev_timestamp
    duration_prev_stop = ratio * duration_prev_next.total_seconds()
    duration_prev_stop = timedelta(0, duration_prev_stop)
    stop_timestamp = prev_timestamp + duration_prev_stop
    return stop_timestamp


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

    :param time_of_day: the given time for calculating the dist_along_route
    :param prev_dist: the distance of the location of the bus for the previous record in historical data
    :param next_dist: the distance of the location of the bus for the next record in historical data
    :param prev_timestamp: the timestamp of the bus for the previous record in historical data
    :param next_timestamp: the timestamp of the bus for the next record in historical data
    :return result: dist_along_route for the bus at the given time_of_day
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


def filter_single_history(single_history, stop_sequence):
    current_history = single_history[
        (single_history.next_stop_id.isin(stop_sequence)) & (single_history.dist_along_route > 0)]
    if len(current_history) < 3:
        return None
    tmp_history = pd.DataFrame(columns=current_history.columns)
    i = 1
    prev_record = current_history.iloc[0]
    while i < len(current_history):
        next_record = current_history.iloc[i]
        prev_distance = float(prev_record.total_distance)
        next_distance = float(next_record.total_distance)
        while prev_distance >= next_distance:
            i += 1
            if i == len(current_history):
                break
            next_record = current_history.iloc[i]
            next_distance = float(next_record.total_distance)
        tmp_history.loc[len(tmp_history)] = prev_record
        prev_record = next_record
        i += 1
    if float(prev_record.total_distance) > float(tmp_history.iloc[-1].total_distance):
        tmp_history.loc[len(tmp_history)] = prev_record
    return tmp_history


#################################################################################################################
#                                baseline 3                                                                     #
#################################################################################################################


def generate_estimated_arrival_time_baseline3(api_data, segment_data, route_stop_dist, trips):
    """
    Calculate the estimated arrival time based on the baseline 3. Use the segment data according to the specific trip and the date to predict the result

    Algorithm

    For row in api data:
        extract the trip id and the service date of that row
        extract the single segment data according to the trip id and the service date
        divide the single segment data through groupby(segment start, segment end)
        calculate the average value for each one in the groupby function
        calculate the estimated arrival time for this row according to the result
        save the result
    concatenate the result

    :param api_data: dataframe for the api_data.csv
    :param segment_data: dataframe for the preprocessed final_segment.csv file according to different baseline algorithm
    :param route_stop_dist: dataframe of the route_stop_dist.csv file
    :param trips: dataframe for the trips.txt file
    :return: dataframe to store the result including the esitmated arrival time
    """
    result = pd.DataFrame(
        columns=['trip_id', 'route_id', 'stop_id', 'vehicle_id', 'time_of_day', 'service_date', 'dist_along_route',
                 'stop_num_from_call', 'estimated_arrival_time'])
    print "length of the api data is: ", len(api_data)
    for i in xrange(len(api_data)):
        if i % 1000 == 0:
            print i
        # get the variables
        item = api_data.iloc[i]
        trip_id = item.get('trip_id')
        route_id = trips[trips.trip_id == trip_id].iloc[0].route_id
        single_route_stop_dist = route_stop_dist[route_stop_dist.route_id == route_id]
        stop_sequence = list(single_route_stop_dist.stop_id)
        target_stop = item.get('stop_id')
        target_index = stop_sequence.index(target_stop)
        dist_along_route = item.get('dist_along_route')
        vehicle_id = item.get('vehicle_id')
        time_of_day = item.get('time_of_day')
        service_date = item.get('date')
        # preprocess the segment data according to the trip id and the service date
        single_segment_data = segment_data[(segment_data['trip_id'] == trip_id)]
        grouped = single_segment_data.groupby(['segment_start', 'segment_end'])
        preprocessed_segment_data = grouped['travel_duration'].mean().reset_index()
        average_travel_duration = preprocessed_segment_data['travel_duration'].mean()
        # find the segment containing the current location of the api data
        if dist_along_route >= single_route_stop_dist.iloc[-1].dist_along_route:
            continue
        for j in range(1, len(stop_sequence)):
            if single_route_stop_dist.iloc[j - 1].dist_along_route < dist_along_route < single_route_stop_dist.iloc[
                j].dist_along_route:
                prev_record = single_route_stop_dist.iloc[j - 1]
                next_record = single_route_stop_dist.iloc[j]
                break
            elif single_route_stop_dist.iloc[j - 1].dist_along_route == dist_along_route:
                prev_record = single_route_stop_dist.iloc[j - 1]
                next_record = prev_record
                break
            else:
                continue
        next_index = stop_sequence.index(next_record.get('stop_id'))
        count = target_index - next_index
        # check how many stops between the current location and the target stop
        if count < 0:
            continue
        elif count == 0:
            total_travel_duration = calculate_time_from_stop(preprocessed_segment_data, dist_along_route, prev_record,
                                                             next_record)
        else:
            total_travel_duration = 0.0
            for j in xrange(next_index, target_index):
                segment_start = stop_sequence[j]
                segment_end = stop_sequence[j + 1]
                segment_record = preprocessed_segment_data[
                    (preprocessed_segment_data.segment_start == segment_start) & (
                        preprocessed_segment_data.segment_end == segment_end)]
                if len(segment_record) == 0:
                    single_travel_duration = average_travel_duration
                else:
                    single_travel_duration = segment_record.iloc[0]['travel_duration']
                total_travel_duration += single_travel_duration
            time_from_stop = calculate_time_from_stop(preprocessed_segment_data, dist_along_route, prev_record,
                                                      next_record)
            total_travel_duration += time_from_stop
        result.loc[len(result)] = [trip_id, route_id, target_stop, vehicle_id, time_of_day, service_date,
                                   dist_along_route, count + 1, total_travel_duration]
    return result


#################################################################################################################
#                                predict section                                                                #
#################################################################################################################
"""
Predict the arrival time for each record in the api data
"""


def calculate_time_from_stop(segment_df, dist_along_route, prev_record, next_record):
    """
    Calculate the time from stop within the tuple (prev_record, next_record)

    Algorithm:
    if prev_record = next_record:
        the bus is parking at the stop, return 0
    Calcualte the distance within the tuple
    Calculate the distance between the current location and the prev record
    Calcualte the ratio of these two distances
    Use the ratio to calcualte the time_from_stop

    :param segment_df: dataframe for the preprocessed segment data
    :param dist_along_route: distance between the intial stop and the current location of the bus
    :param prev_record: single record of the route_stop_dist.csv file
    :param next_record: single record of the route_stop_dist.csv file
    :return: total seconds of the time_from_stop
    """
    if prev_record.get('stop_id') == next_record.get('stop_id'):
        return 0.0
    distance_stop_stop = next_record.get('dist_along_route') - prev_record.get('dist_along_route')
    distance_bus_stop = next_record.get('dist_along_route') - dist_along_route
    ratio = float(distance_bus_stop) / float(distance_stop_stop)
    assert ratio < 1
    try:
        travel_duration = segment_df[(segment_df.segment_start == prev_record.get('stop_id')) & (
            segment_df.segment_end == next_record.get('stop_id'))].iloc[0]['travel_duration']
    except:
        travel_duration = segment_df['travel_duration'].mean()
    time_from_stop = travel_duration * ratio
    return time_from_stop


def generate_estimated_arrival_time(api_data, preprocessed_segment_data, route_stop_dist, trips):
    """
    Predict the estimated arrival time according to the api data

    Algorithm:
    Build the empty dataframe
    for row in api_data:
        get the route_id according to the trip id and the trips.txt file
        get the stop sequence and the dist_along_route according to the route id
        get the end_index according to the stop id
        get the (prev, next) stop tuple according to the dist_along_route in the record
        get the count = end_index - next_index
        if count < 0:
            the bus has passed
            continue to next row
        if count = 0, the next stop is the target stop:
            calcualte the time_from_stop for (prev, next) tuple
            save the result as the estimated time
        if count > 0:
            get the stop list from next_stop to target_stop
            sum the travel duration for all of them
            calculate the time_from_stop for (prev, next) tuple
            add the total travel duration with the time_from_stop(prev, next)
            save the result as the estimated time
        save the result into the dataframe

    :param api_data: dataframe for the api_data.csv
    :param preprocessed_segment_data: dataframe for the preprocessed final_segment.csv file according to different baseline algorithm
    :param route_stop_dist: dataframe of the route_stop_dist.csv file
    :param trips: dataframe for the trips.txt file
    :return: dataframe to store the result including the esitmated arrival time
    """
    result = pd.DataFrame(
        columns=['trip_id', 'route_id', 'stop_id', 'vehicle_id', 'time_of_day', 'service_date', 'dist_along_route',
                 'stop_num_from_call', 'estimated_arrival_time'])
    print "length of the api data is: ", len(api_data)
    average_travel_duration = preprocessed_segment_data['travel_duration'].mean()
    for i in xrange(len(api_data)):
        if i % 1000 == 0:
            print i
        item = api_data.iloc[i]
        trip_id = item.get('trip_id')
        route_id = trips[trips.trip_id == trip_id].iloc[0].route_id
        single_route_stop_dist = route_stop_dist[route_stop_dist.route_id == route_id]
        stop_sequence = list(single_route_stop_dist.stop_id)
        target_stop = item.get('stop_id')
        target_index = stop_sequence.index(target_stop)
        dist_along_route = item.get('dist_along_route')
        vehicle_id = item.get('vehicle_id')
        time_of_day = item.get('time_of_day')
        service_date = item.get('date')
        if dist_along_route >= single_route_stop_dist.iloc[-1].dist_along_route:
            continue
        for j in range(1, len(stop_sequence)):
            if single_route_stop_dist.iloc[j - 1].dist_along_route < dist_along_route < single_route_stop_dist.iloc[
                j].dist_along_route:
                prev_record = single_route_stop_dist.iloc[j - 1]
                next_record = single_route_stop_dist.iloc[j]
                break
            elif single_route_stop_dist.iloc[j - 1].dist_along_route == dist_along_route:
                prev_record = single_route_stop_dist.iloc[j - 1]
                next_record = prev_record
                break
            else:
                continue
        next_index = stop_sequence.index(next_record.get('stop_id'))
        count = target_index - next_index
        if count < 0:
            continue
        elif count == 0:
            total_travel_duration = calculate_time_from_stop(preprocessed_segment_data, dist_along_route, prev_record,
                                                             next_record)
        else:
            total_travel_duration = 0.0
            for j in xrange(next_index, target_index):
                segment_start = stop_sequence[j]
                segment_end = stop_sequence[j + 1]
                segment_record = preprocessed_segment_data[
                    (preprocessed_segment_data.segment_start == segment_start) & (
                        preprocessed_segment_data.segment_end == segment_end)]
                if len(segment_record) == 0:
                    single_travel_duration = average_travel_duration
                else:
                    single_travel_duration = segment_record.iloc[0]['travel_duration']
                total_travel_duration += single_travel_duration
            time_from_stop = calculate_time_from_stop(preprocessed_segment_data, dist_along_route, prev_record,
                                                      next_record)
            total_travel_duration += time_from_stop
        result.loc[len(result)] = [trip_id, route_id, target_stop, vehicle_id, time_of_day, service_date,
                                   dist_along_route, count + 1, total_travel_duration]
    return result


def generate_actual_arrival_time(full_history, segment_df, route_stop_dist):
    """
    Calculate the actual arrival time from the dataset

    Algorithm:
    Build the empty dataframe
    for row in segment_df:
        get trip_id, route_id, target_stop, service_date, etc
        get single_history data according to the trip id and the service date
        get single_route_stop_dist according to the route_id
        get stop_sequence from single_route_stop_dist
        get the dist_along_route for the target_stop from the single_route_stop_dist
        set prev_index, next_index = stop_sequence.index(target_stop)
        while stop_sequence(prev_index) not in set(single_history.next_stop_id):
            prev_index -= 1
            if prev_index == -1:
                break
        if prev_index == -1:
            continue
        prev_stop = stop_sequence(prev_index)
        next_index += 1
        while stop_sequence(next_index) not in set(single_history.next_stop_id):
            next_index += 1
            if next_index == len(stop_sequence):
                break
        if next_index == len(stop_sequence):
            continue
        next_stop = stop_sequence(next_itndex)
        prev_record = single_history[single_history.next_stop_id == prev_stop].iloc[-1]
        prev_time = prev_record.get('timestamp')
        if prev_record.dist_from_stop == 0:
            actual_arrival_time = prev_time - time_of_day
            save the record
            continue to next row
        next_record = single_history[single_history.next_stop_id == next_stop].iloc[-1]
        next_time = next_record.get('timestamp')
        travel_duration = next_time - prev_time
        prev_distance = prev_record.get('dist_along_route') - prev_record.get('dist_from_stop')
        next_distance = prev_record.get('dist_along_route') - prev_record.get('dist_from_stop')
        distance_prev_next = next_distance - prev_distance
        distance_prev_stop = single_route_stop_dist[single_route_stop_dist.stop_id == target_stop].iloc[0]['dist_along_route'] - prev_distance
        ratio = distance_prev_stop / distance_prev_next
        time_from_stop = ratio * travel_duration
        arrival_time = time_from_stop + prev_time
        actual_arrival_time = arrival_time - time_of_day
        save the record

    :param full_history: dataframe for the historical data
    :param segment_df: dataframe for the preprocessed average travel duration for the segmet data
    :param route_stop_dist: dataframe for the route_stop_dist.csv file
    :return: dataframe including both of the estimated arrival time and actual arrival time
    """
    result = pd.DataFrame(
        columns=['trip_id', 'route_id', 'stop_id', 'vehicle_id', 'time_of_day', 'service_date', 'dist_along_route',
                 'stop_num_from_call', 'estimated_arrival_time', 'actual_arrival_time'])
    grouped_list = list(segment_df.groupby(['service_date', 'trip_id', 'stop_id']))
    print 'length of the segment_df is: ', len(grouped_list)
    for i in xrange(len(grouped_list)):
        if i % 100 == 0:
            print i
        name, item = grouped_list[i]
        service_date, trip_id, target_stop = name
        route_id = item.iloc[0]['route_id']
        single_route_stop_dist = route_stop_dist[route_stop_dist.route_id == route_id]
        stop_sequence = list(single_route_stop_dist.stop_id)
        # stop_sequence_str = [str(int(stop_id)) for stop_id in stop_sequence]
        target_index = stop_sequence.index(target_stop)
        dist_along_route = single_route_stop_dist[single_route_stop_dist.stop_id == target_stop].iloc[0][
            'dist_along_route']
        vehicle_id = item.iloc[0]['vehicle_id']
        single_history = full_history[(full_history.service_date == service_date) & (full_history.trip_id == trip_id)]
        single_history = filter_single_history(single_history, stop_sequence)
        if single_history is None:
            continue
        prev_index, next_index = target_index, target_index + 1
        while stop_sequence[prev_index] not in set(single_history.next_stop_id):
            prev_index -= 1
            if prev_index == -1:
                break
        if prev_index == -1:
            continue
        prev_stop = stop_sequence[prev_index]
        while stop_sequence[next_index] not in set(single_history.next_stop_id):
            next_index += 1
            if next_index == len(stop_sequence):
                break
        if next_index == len(stop_sequence):
            continue
        next_stop = stop_sequence[next_index]
        prev_record = single_history[single_history.next_stop_id == prev_stop].iloc[-1]
        prev_time = prev_record.get('timestamp')
        prev_time = datetime.strptime(prev_time, '%Y-%m-%dT%H:%M:%SZ')
        if prev_record.dist_from_stop == 0 and prev_record.next_stop_id == target_stop:
            timestamp = prev_time
        else:
            next_record = single_history[single_history.next_stop_id == next_stop].iloc[-1]
            next_time = next_record.get('timestamp')
            next_time = datetime.strptime(next_time, '%Y-%m-%dT%H:%M:%SZ')
            prev_distance = float(prev_record.get('total_distance'))
            next_distance = float(next_record.get('total_distance'))
            timestamp = calculate_arrival_time(dist_along_route, prev_distance, next_distance, prev_time, next_time)
        for j in xrange(len(item)):
            single_record = item.iloc[j]
            time_of_day = single_record.get('time_of_day')
            stop_num_from_call = single_record.get('stop_num_from_call')
            estimated_arrival_time = single_record.get('estimated_arrival_time')
            time_of_day = datetime.strptime(time_of_day, '%Y-%m-%d %H:%M:%S')
            actual_arrival_time = timestamp - time_of_day
            actual_arrival_time = actual_arrival_time.total_seconds()
            result.loc[len(result)] = [trip_id, route_id, target_stop, vehicle_id, str(time_of_day), service_date,
                                       dist_along_route, stop_num_from_call, estimated_arrival_time,
                                       actual_arrival_time]
    return result


#################################################################################################################
#                                debug section                                                                  #
#################################################################################################################
# print "export the result for baseline1"
# route_stop_dist = pd.read_csv('route_stop_dist.csv')
# file_list = os.listdir('./')
# if 'estimated_segment_baseline3.csv' not in file_list:
#     api_data = pd.read_csv('api_data.csv')
#     preprocessed_segment_data = pd.read_csv('segment.csv')
#     trips = pd.read_csv(path + 'data/GTFS/gtfs/trips.txt')
#     segment_df = generate_estimated_arrival_time_baseline3(api_data, preprocessed_segment_data, route_stop_dist, trips)
#     segment_df.to_csv('estimated_segment_baseline3.csv')
# else:
#     segment_df = pd.read_csv('estimated_segment_baseline3.csv')
# full_history = pd.read_csv('test_history.csv')
# baseline_result = generate_actual_arrival_time(full_history, segment_df, route_stop_dist)
# baseline_result.to_csv('baseline3_result.csv')
# print "complete exporting the result for baseline1"


#################################################################################################################
#                                    main function                                                              #
#################################################################################################################

file_list = os.listdir('./')

if __name__ == "__main__":
    print "prepare the segment dataset for different baseline algorithm"
    segment_df = pd.read_csv('full_segment.csv')
    if "result/compare_baseline/baseline1_result.csv" not in file_list:
        print "export the result for baseline1"
        route_stop_dist = pd.read_csv('route_stop_dist.csv')
        if 'estimated_segment_baseline1.csv' not in file_list:
            api_data = pd.read_csv('api_data.csv')
            preprocessed_segment_data = pd.read_csv('segment_baseline1.csv')
            route_stop_dist = pd.read_csv('route_stop_dist.csv')
            trips = pd.read_csv(path + 'data/GTFS/gtfs/trips.txt')
            segment_df = generate_estimated_arrival_time(api_data, preprocessed_segment_data, route_stop_dist, trips)
            segment_df.to_csv('estimated_segment_baseline1.csv')
        else:
            segment_df = pd.read_csv('estimated_segment_baseline1.csv')
        full_history = pd.read_csv('test_history.csv')
        baseline_result = generate_actual_arrival_time(full_history, segment_df, route_stop_dist)
        baseline_result.to_csv('baseline1_result.csv')
        print "complete exporting the result for baseline1"
    if "baseline2_result.csv" not in file_list:
        print "export the result for baseline2"
        # estimated result
        route_stop_dist = pd.read_csv('route_stop_dist.csv')
        if 'estimated_segment_baseline2.csv' not in file_list:
            api_data = pd.read_csv('api_data.csv')
            rush_hour = api_data['time_of_day'].apply(lambda x: x[11:19] < '20:00:00' and x[11:19] > '17:00:00')
            api_data['rush_hour'] = rush_hour
            preprocessed_segment_data = pd.read_csv('segment_baseline2.csv')
            trips = pd.read_csv(path + 'data/GTFS/gtfs/trips.txt')
            grouped_segment_df = preprocessed_segment_data.groupby(['weather', 'rush_hour'])
            grouped_api_data = api_data.groupby(['date', 'rush_hour'])
            weather_df = pd.read_csv('weather.csv')
            estimated_result_list = []
            for current_date in range(20160125, 20160130):
                print current_date
                weather = weather_df[weather_df.date == current_date].iloc[0]['weather']
                current_result = generate_estimated_arrival_time(grouped_api_data.get_group((current_date, True)),
                                                                 grouped_segment_df.get_group((weather, True)),
                                                                 route_stop_dist, trips)
                estimated_result_list.append(current_result)
                current_result = generate_estimated_arrival_time(grouped_api_data.get_group((current_date, False)),
                                                                 grouped_segment_df.get_group((weather, False)),
                                                                 route_stop_dist, trips)
                estimated_result_list.append(current_result)
            segment_df = pd.concat(estimated_result_list, ignore_index=True)
            segment_df.to_csv('estimated_segment_baseline2.csv')
        else:
            segment_df = pd.read_csv('estimated_segment_baseline2.csv')

        # actual arrival time
        full_history = pd.read_csv('test_history.csv')
        baseline_result = generate_actual_arrival_time(full_history, segment_df, route_stop_dist)
        baseline_result.to_csv('baseline2_result.csv')
        print "complete exporting the result baseline2"
    if "sanity_baseline1_result.csv" not in file_list:
        print "export the result for sanity check of baseline1"
        route_stop_dist = pd.read_csv('route_stop_dist.csv')
        if 'sanity_estimated_segment_baseline1.csv' not in file_list:
            api_data = pd.read_csv('sanity_api_data.csv')
            preprocessed_segment_data = pd.read_csv('segment_baseline1.csv')
            route_stop_dist = pd.read_csv('route_stop_dist.csv')
            trips = pd.read_csv(path + 'data/GTFS/gtfs/trips.txt')
            segment_df = generate_estimated_arrival_time(api_data, preprocessed_segment_data, route_stop_dist, trips)
            segment_df.to_csv('sanity_estimated_segment_baseline1.csv')
        else:
            segment_df = pd.read_csv('sanity_estimated_segment_baseline1.csv')
        full_history = pd.read_csv('train_history.csv')
        full_history = full_history[full_history.service_date.isin(range(20160118, 20160124))]
        baseline_result = generate_actual_arrival_time(full_history, segment_df, route_stop_dist)
        baseline_result.to_csv('sanity_baseline1_result.csv')
        print "complete exporting the sanity check of result for baseline1"
    if "sanity_baseline2_result.csv" not in file_list:
        print "export the result for sanity check of baseline2"
        # estimated result
        route_stop_dist = pd.read_csv('route_stop_dist.csv')
        if 'sanity_estimated_segment_baseline2.csv' not in file_list:
            api_data = pd.read_csv('sanity_api_data.csv')
            rush_hour = api_data['time_of_day'].apply(lambda x: x[11:19] < '20:00:00' and x[11:19] > '17:00:00')
            api_data['rush_hour'] = rush_hour
            preprocessed_segment_data = pd.read_csv('segment_baseline2.csv')
            trips = pd.read_csv(path + 'data/GTFS/gtfs/trips.txt')
            grouped_segment_df = preprocessed_segment_data.groupby(['weather', 'rush_hour'])
            grouped_api_data = api_data.groupby(['date', 'rush_hour'])
            weather_df = pd.read_csv('weather.csv')
            estimated_result_list = []
            for current_date in range(20160118, 20160124):
                print current_date
                weather = weather_df[weather_df.date == current_date].iloc[0]['weather']
                current_result = generate_estimated_arrival_time(grouped_api_data.get_group((current_date, True)),
                                                                 grouped_segment_df.get_group((weather, True)),
                                                                 route_stop_dist, trips)
                estimated_result_list.append(current_result)
                current_result = generate_estimated_arrival_time(grouped_api_data.get_group((current_date, False)),
                                                                 grouped_segment_df.get_group((weather, False)),
                                                                 route_stop_dist, trips)
                estimated_result_list.append(current_result)
            segment_df = pd.concat(estimated_result_list, ignore_index=True)
            segment_df.to_csv('sanity_estimated_segment_baseline2.csv')
        else:
            segment_df = pd.read_csv('sanity_estimated_segment_baseline2.csv')

        # actual arrival time
        full_history = pd.read_csv('train_history.csv')
        full_history = full_history[full_history.service_date.isin(range(20160118, 20160124))]
        baseline_result = generate_actual_arrival_time(full_history, segment_df, route_stop_dist)
        baseline_result.to_csv('sanity_baseline2_result.csv')
        print "complete exporting the result of the sanity check for baseline2"
