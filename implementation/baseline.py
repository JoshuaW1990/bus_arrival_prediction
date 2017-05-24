"""
Calculate and assess the estimated arrival time with different baseline algorithm

"""

import pandas as pd
from datetime import datetime


def filter_single_history(single_history, stop_sequence):
    """
    Filter the history file with only one day and one stop sequence to remove abnormal record

    :param single_history: dataframe for history table with only one day
    :param stop_sequence: list of stop id
    :return: dataframe for filtered history table
    """
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



def generate_estimated_arrival_time_baseline3(api_data, full_segment_data, route_stop_dist, trips):
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
    def helper(preprocessed_segment_data, average_travel_duration, dist_along_route, prev_record, next_record):
        segment_start = prev_record.get('stop_id')
        segment_end = next_record.get('stop_id')
        if segment_start == segment_end:
            return 0.0
        distance_stop_stop = next_record.get('dist_along_route') - prev_record.get('dist_along_route')
        distance_bus_stop = next_record.get('dist_along_route') - dist_along_route
        ratio = float(distance_bus_stop) / float(distance_stop_stop)
        assert ratio < 1
        travel_duration = preprocessed_segment_data.get((segment_start, segment_end), average_travel_duration)
        time_from_stop = travel_duration * ratio
        return time_from_stop


    trip_shape_dict = trips.set_index('trip_id').to_dict(orient='index')
    result = pd.DataFrame(
        columns=['trip_id', 'route_id', 'stop_id', 'vehicle_id', 'time_of_day', 'service_date', 'dist_along_route',
                 'stop_num_from_call', 'estimated_arrival_time', 'shape_id'])
    # print "baseline3 length of api data: ", len(api_data)
    for i in xrange(len(api_data)):
        # if i % 1000 == 0:
        #     print i
        # get the variables
        item = api_data.iloc[i]
        trip_id = item.get('trip_id')
        shape_id = trip_shape_dict[trip_id]['shape_id']
        route_id = trips[trips.trip_id == trip_id].iloc[0].route_id
        single_route_stop_dist = route_stop_dist[route_stop_dist.shape_id == shape_id]
        stop_sequence = list(single_route_stop_dist.stop_id)
        target_stop = item.get('stop_id')
        target_index = stop_sequence.index(target_stop)
        dist_along_route = item.get('dist_along_route')
        vehicle_id = item.get('vehicle_id')
        time_of_day = item.get('time_of_day')
        service_date = item.get('date')
        # preprocess the segment data according to the trip id and the service date
        segment_data = full_segment_data[(full_segment_data.service_date != service_date) | (full_segment_data.trip_id != trip_id)]
        trip_list = set(trips[trips.shape_id == shape_id].trip_id)
        single_segment_data = segment_data[(segment_data.trip_id.isin(trip_list))]
        grouped = single_segment_data.groupby(['segment_start', 'segment_end'])
        preprocessed_segment_data = grouped['travel_duration'].mean()
        average_travel_duration = single_segment_data['travel_duration'].mean()
        # find the segment containing the current location of the api data
        prev_route_stop_dist = single_route_stop_dist[single_route_stop_dist.dist_along_route < dist_along_route]
        next_route_stop_dist = single_route_stop_dist[single_route_stop_dist.dist_along_route >= dist_along_route]
        if len(prev_route_stop_dist) == 0 or len(next_route_stop_dist) == 0:
            continue
        next_index = len(prev_route_stop_dist)
        count = target_index - next_index
        # check how many stops between the current location and the target stop
        prev_record = prev_route_stop_dist.iloc[-1]
        next_record = next_route_stop_dist.iloc[0]
        if count < 0:
            continue
        elif count == 0:
            total_travel_duration = helper(preprocessed_segment_data, average_travel_duration, dist_along_route, prev_record, next_record)
        else:
            total_travel_duration = 0.0
            for j in xrange(next_index, target_index):
                segment_start = stop_sequence[j]
                segment_end = stop_sequence[j + 1]
                single_travel_duration = preprocessed_segment_data.get((segment_start, segment_end), average_travel_duration)
                total_travel_duration += single_travel_duration
            time_from_stop = helper(preprocessed_segment_data, average_travel_duration, dist_along_route, prev_record, next_record)
            total_travel_duration += time_from_stop
        result.loc[len(result)] = [trip_id, route_id, target_stop, vehicle_id, time_of_day, service_date,
                                   dist_along_route, count + 1, total_travel_duration, shape_id]
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
        columns=['trip_id', 'route_id', 'stop_id', 'vehicle_id', 'time_of_day', 'service_date', 'dist_along_route', 'stop_num_from_call', 'estimated_arrival_time', 'actual_arrival_time', 'shape_id'])
    grouped_list = list(segment_df.groupby(['service_date', 'trip_id', 'stop_id']))
    print 'length of the segment_df is: ', len(grouped_list)
    for i in xrange(len(grouped_list)):
        if i % 1000 == 0:
            print i
        name, item = grouped_list[i]
        service_date, trip_id, target_stop = name
        route_id = item.iloc[0]['route_id']
        shape_id = item.iloc[0]['shape_id']
        single_route_stop_dist = route_stop_dist[route_stop_dist.shape_id == shape_id]
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
        if prev_stop == next_stop:
            print "error"
            continue
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
            dist_along_route = single_record.get('dist_along_route')
            result.loc[len(result)] = [trip_id, route_id, target_stop, vehicle_id, str(time_of_day), service_date, dist_along_route, stop_num_from_call, estimated_arrival_time, actual_arrival_time, shape_id]
    return result
