"""
predict the estimated arrival time based on the 
"""

# import module
import pandas as pd
import os
from datetime import datetime, timedelta

path = '../'


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
    for i in xrange(len(api_data)):
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
            if single_route_stop_dist.iloc[j - 1].dist_along_route < dist_along_route < single_route_stop_dist.iloc[j].dist_along_route:
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
            dist_along_route = single_record.get('dist_along_route')
            result.loc[len(result)] = [trip_id, route_id, target_stop, vehicle_id, str(time_of_day), service_date,
                                       dist_along_route, stop_num_from_call, estimated_arrival_time,
                                       actual_arrival_time]
    return result


#################################################################################################################
#                                    build dataset                                                              #
#################################################################################################################
"""
The dataset is composed of two parts: input feature and the output
input feature:
    - weather (0, 1, or 2)
    - rush hour (0 or 1)
    - estimated arrival time with third baseline algorithm
    - average speed(prev_seg_speed) for the previous neighboring segment in the that specific trip
    - average speed(current_seg_speed) for the same segment of the previous trip in the same route
Output:
    - True value: actual arrival time
    - Predicted value: predicted arrival time with regression model
"""


def generate_complete_dateset(api_data, segment_df, route_stop_dist, trips, full_history, weather_df, trip_list=None):
    """
    Calculate the complete result of baseline3

    Algorithm:
    1) generate the estimated arrival time from api data
    2) generate the actual arrival time
    3) Add weather to result:
    4) Add the binary result for the rush hour

    :param api_data: 
    :param segment_df: 
    :param route_stop_idst: 
    :param trips:
    :param full_history:
    :return: 
    """
    print "start to export the result of the dataset"
    weather_df['date'] = pd.to_numeric(weather_df['date'])

    if trip_list is not None:
        api_data = api_data[api_data.trip_id.isin(trip_list)]
    file_list = os.listdir('./')
    if 'estimated_segment_data.csv' not in file_list:
        estimated_segment_df = generate_estimated_arrival_time_baseline3(api_data, segment_df, route_stop_dist, trips)
        estimated_segment_df.to_csv('estimated_segment_data.csv')
    else:
        estimated_segment_df = pd.read_csv('estimated_segment_data.csv')

    result = generate_actual_arrival_time(full_history, estimated_segment_df, route_stop_dist)
    result['service_date'] = pd.to_numeric(result['service_date'])

    result['weather'] = result['service_date'].apply(lambda x: weather_df[weather_df.date == x].iloc[0]['weather'])
    result['rush_hour'] = result['time_of_day'].apply(lambda x: 1 if '20:00:00' >= x[11:19] >= '17:00:00' else 0)

    result.to_csv('baseline_result.csv')
    print "complete exporting the result of the dataset"
    return result


def generate_feature_api(single_segment, dist_along_route, target_dist, time_of_day, single_route_stop_dist):
    """
    Generate the api list for calculating the average delay
    
    Algorithm:
    (1) get the single segment smaller or equal than the target_dist
    (2) generate the feature api list by each row in filtered single segment data
    (3) use the timestamp in segment data for the actual arrival time
    
    :param single_segment: 
    :param dist_along_route:
    :param target_dist: 
    :param time_of_day: 
    :param single_route_stop_dist:
    :return: 
    """
    feature_api = pd.DataFrame(columns=['actual_arrival_time'])
    # get the single segment smaller or equal than the target_dist
    current_segment = single_segment[single_segment['segment_end'].apply(
        lambda x: single_route_stop_dist[single_route_stop_dist['stop_id'] == x].iloc[0][
                      'dist_along_route'] <= target_dist)]
    index = 0
    while index < len(current_segment):
        if single_route_stop_dist[single_route_stop_dist['stop_id'] == current_segment.iloc[index]['segment_start']].iloc[0]['dist_along_route'] < dist_along_route:
            index += 1
        else:
            break
    if index == len(current_segment):
        return None
    current_segment = current_segment[index:]
    current_segment.reset_index(inplace=True)

    # generate the actual arrival time from the travel duration of the single segment date
    feature_api.loc[0] = [current_segment.iloc[0]['travel_duration']]
    for i in xrange(1, len(current_segment)):
        feature_api.loc[i] = [feature_api.loc[i - 1]['actual_arrival_time'] + current_segment.loc[i]['travel_duration']]

    # generate the feature api list by each row in filtered single segment data
    route_id = single_route_stop_dist.iloc[0]['route_id']
    feature_api['trip_id'] = current_segment['trip_id']
    feature_api['vehicle_id'] = current_segment['vehicle_id']
    feature_api['route_id'] = route_id
    feature_api['stop_id'] = current_segment['segment_end']
    feature_api['time_of_day'] = time_of_day
    feature_api['date'] = current_segment['service_date']
    feature_api['dist_along_route'] = dist_along_route

    return feature_api


def obtain_prev_trip(single_segment, stop_id, time_of_day):
    """
    obtain the trip id of the previous trip
    
    Algorithm:
    1) divide the single_segment through groupby([trip id, service date])
    2) for name, item in grouped single history:
        (1) obtain trip id and service date from name
        (2) item is the single segment for trip id and the service date
        (3) find arrival time that bus arrived to the target stop from the single segment
        (4) compare the arrival time and the time_of_day:
            a) if arrival_time < time_of_day, compare the bucket tuple (arrival_time, trip id, service date) to obtain the maximum arrival_time
            b) else, continue
    3) return maximum bucket (arrival_time, trip id, service date)
    
    :param single_segment:
    :param stop_id:
    :param time_of_day: 
    :return: 
    """
    # divide the single_history through groupby([trip id, service date])
    grouped = single_segment.groupby(['trip_id', 'service_date'])
    time_of_day = datetime.strptime(time_of_day, '%Y-%m-%d %H:%M:%S')
    max_tuple = None

    for name, item in grouped:
        # obtain trip id and service date from name
        trip_id, service_date = name

        # find arrival time that bus arrived to the target stop from the single segment
        single_record = item[item['segment_start'] == stop_id]
        if len(single_record) == 0:
            continue
        arrival_time = single_record.iloc[0]['timestamp']
        arrival_time = datetime.strptime(arrival_time[:19], '%Y-%m-%d %H:%M:%S')

        # compare the arrival time and the time_of_day to obtain the maximum available arrival time
        if arrival_time <= time_of_day:
            if max_tuple is None:
                max_tuple = (arrival_time, trip_id, service_date)
            elif arrival_time > max_tuple[0]:
                max_tuple = (arrival_time, trip_id, service_date)
            else:
                continue
        else:
            continue

    return max_tuple


def obtain_time_of_day(single_segment, dist_along_route, single_route_stop_dist):
    """
    obtain the time of day when the bus located at the dist_along_route in baseline3result record
    
    Algorithm:
    1) use the dist_along_route find the corresponding segment pair
    2) calculate the time_of_day according to the dist_along_route within that segment pair
    
    :param single_segment: 
    :param dist_along_route: 
    :param single_route_stop_dist:
    :return: 
    """
    # use the dist_along_route find the corresponding segment pair
    flag = 0
    for i in xrange(len(single_segment)):
        prev_stop = single_segment.iloc[i]['segment_start']
        next_stop = single_segment.iloc[i]['segment_end']
        prev_timestamp = single_segment.iloc[i]['timestamp']
        travl_duration = single_segment.iloc[i]['travel_duration']
        prev_distance = single_route_stop_dist[single_route_stop_dist['stop_id'] == prev_stop].iloc[0]['dist_along_route']
        next_distance = single_route_stop_dist[single_route_stop_dist['stop_id'] == next_stop].iloc[0]['dist_along_route']
        if prev_distance <= dist_along_route < next_distance:
            flag = 1
            break
        else:
            continue
    if flag == 0:
        return None

    # calculate the time_of_day according to the dist_along_route within that segment pair
    prev_timestamp = datetime.strptime(prev_timestamp[:19], '%Y-%m-%d %H:%M:%S')
    travl_duration = timedelta(0, travl_duration)
    next_timestamp = prev_timestamp + travl_duration
    time_of_day = calculate_arrival_time(dist_along_route, prev_distance, next_distance, prev_timestamp, next_timestamp)

    return str(time_of_day)

    # # add the dist_along_route for the segment start and the segment end for the segment df
    # single_segment['dist_segment_start'] = single_segment['segment_start'].apply(
    #     lambda x: single_route_stop_dist[single_route_stop_dist['stop_id'] == x].iloc[0]['dist_along_route'])
    # single_segment['dist_segment_end'] = single_segment['segment_end'].apply(
    #     lambda x: single_route_stop_dist[single_route_stop_dist['stop_id'] == x].iloc[0]['dist_along_route'])
    #
    # # use the dist_along_route find the corresponding segment pair
    # single_segment_pair = single_segment[(single_segment['dist_segment_start'] <= dist_along_route) & (
    #     single_segment['dist_segment_end'] > dist_along_route)]
    #
    # # calculate the time_of_day according to the dist_along_route within that segment pair
    # prev_stop = single_segment_pair.iloc[0]['segment_start']
    # next_stop = single_segment_pair.iloc[0]['segment_end']
    # prev_distance = single_route_stop_dist[single_route_stop_dist['stop_id'] == prev_stop].iloc[0]['dist_along_route']
    # next_distance = single_route_stop_dist[single_route_stop_dist['stop_id'] == next_stop].iloc[0]['dist_along_route']
    # prev_timestamp = single_segment_pair.iloc[0]['timestamp']
    # travel_duration = single_segment_pair.iloc[0]['travel_duration']
    # prev_timestamp = datetime.strptime(prev_timestamp[:19], '%Y-%m-%d %H:%M:%S')
    # travel_duration = timedelta(0, travel_duration)
    # next_timestamp = prev_timestamp + travel_duration
    # time_of_day = calculate_arrival_time(dist_along_route, prev_distance, next_distance, prev_timestamp, next_timestamp)
    #
    # return str(time_of_day)


def calculate_average_delay(feature_api, segment_df, route_stop_dist, trips):
    """
    calculate the average arrival time
    
    Algorithm:
    1) run baseline3 algorithm for the estimated arrival time
    2) calculate the delay between the actual arrival time and the estimated arrival time
    3) calculate the average delay
    
    :param feature_api: 
    :param segment_df: 
    :param route_stop_dist:
    :param trips:
    :return: 
    
    'trip_id', 'vehicle_id', 'route_id', 'stop_id', 'time_of_day', 'date', 'dist_along_route'
    """
    # run baseline3 algorithm for the estimated arrival time
    result = generate_estimated_arrival_time_baseline3(feature_api, segment_df, route_stop_dist, trips)
    feature_api.reset_index(inplace=True)
    result['actual_arrival_time'] = feature_api['actual_arrival_time']

    # calculate the delay between the actual arrival time and the estimated arrival time
    result['delay'] = result['actual_arrival_time'] - result['estimated_arrival_time']

    # calculate the average delay
    return result['delay'].mean()


def preprocess_dataset(baseline_result, segment_df, route_stop_dist, trips):
    """
    Build the dataset for regression algorithms

    Algorithm:
    1) define the result for the dataset
    2) For row in baseline_result.csv file:
        (1) obtain single record, trip id, service date, route id, dist_along_route
        (2) generate delay of current trip:
            a) obtain the single segment and single history according to the trip id and the service date
            b) obtain the time_of_day from the first record of the single segment
            c) use the dist_along_route as the target_dist
            d) generate the feature_api list
        (3) calculate the average delay of the current trip
        (4) generate delay of previous trip:
            a) obtain the trip list according to the route id
            b) filter the segment and history by the trip list and the service date list
            c) obtain id of the previous trip
            d) obtain the single segment and single history by the previous trip id
            e) obtain the time_of_day such that the bus located at the dist_along_route in baseline3result record
            f) use the target stop dist as the target_dist
            g) generate the feature_api list
        (5) calculate the average delay of the previous trip
        (6) save the record in the result
    
    :param baseline_result: 
    :param segment_df: 
    :param route_stop_dist:
    :param trips: 
    :return: 
    """
    result = pd.DataFrame(columns=['weather', 'rush_hour', 'baseline_result', 'delay_current_trip', 'delay_prev_trip'])
    print "length of the baseline_result.csv file: ", len(baseline_result)
    for i in xrange(len(baseline_result)):
        print "index is ", i
        # obtain single record, trip id, service date, route id, dist_along_route
        single_record = baseline_result.iloc[i]
        trip_id = single_record.get('trip_id')
        service_date = single_record.get('service_date')
        route_id = single_record.get('route_id')
        stop_id = single_record.get('stop_id')
        dist_along_route = single_record.get('dist_along_route')
        time_of_day = single_record.get('time_of_day')
        single_route_stop_dist = route_stop_dist[route_stop_dist['route_id'] == route_id]

        # generate delay of the current trip
        single_segment = segment_df[(segment_df.trip_id == trip_id) & (segment_df.service_date.isin([service_date]))]
        current_time_of_day = single_segment.iloc[0]['timestamp']
        target_dist = dist_along_route
        initial_dist = single_route_stop_dist[single_route_stop_dist['stop_id'] == single_segment.iloc[0]['segment_start']].iloc[0]['dist_along_route']
        feature_api = generate_feature_api(single_segment, initial_dist, target_dist, current_time_of_day, single_route_stop_dist)
        if feature_api is None:
            continue
        delay_current_trip = calculate_average_delay(feature_api, segment_df, route_stop_dist, trips)
        print 'delay_current_trip', delay_current_trip

        # generate the delay of the previous trip
        trip_list = list(trips[trips['route_id'] == route_id]['trip_id'])
        single_segment = segment_df[(segment_df.trip_id.isin(trip_list)) & (segment_df.service_date.isin([service_date - 1, service_date]))]
        prev_trip_tuple = obtain_prev_trip(single_segment, stop_id, time_of_day)
        if prev_trip_tuple is None:
            continue
        single_segment = single_segment[(single_segment['trip_id'] == prev_trip_tuple[1]) & (single_segment['service_date'] == prev_trip_tuple[2])]
        current_time_of_day = obtain_time_of_day(single_segment, dist_along_route, single_route_stop_dist)
        target_dist = single_route_stop_dist[single_route_stop_dist['stop_id'] == stop_id].iloc[0]['dist_along_route']
        feature_api = generate_feature_api(single_segment, dist_along_route, target_dist, current_time_of_day, single_route_stop_dist)
        if feature_api is None:
            continue
        delay_prev_trip = calculate_average_delay(feature_api, segment_df, route_stop_dist, trips)
        print 'delay_prev_trip', delay_prev_trip

        # 'weather', 'rush_hour', 'baseline_result', 'delay_current_trip', 'delay_prev_trip'
        result.loc[len(result)] = [single_record.get('weather'), single_record.get('rush_hour'), single_record.get('estimated_arrival_time'), delay_current_trip, delay_prev_trip]

    return result






#################################################################################################################
#                                    predict dataset                                                            #
#################################################################################################################


#################################################################################################################
#                                    debug section                                                              #
#################################################################################################################
route_stop_dist = pd.read_csv('route_stop_dist.csv')
trips = pd.read_csv(path + 'data/GTFS/gtfs/trips.txt')
full_history = pd.read_csv('full_history.csv')
segment_df = pd.read_csv('full_segment.csv')
api_data = pd.read_csv('full_api_data.csv')
weather_df = pd.read_csv('weather.csv')

single_trip = api_data.iloc[0].trip_id
print single_trip

file_list = os.listdir('./')
if 'baseline_result.csv' not in file_list:
    baseline_result = generate_complete_dateset(api_data, segment_df, route_stop_dist, trips, full_history, weather_df, [single_trip])
else:
    baseline_result = pd.read_csv('baseline_result.csv')
dataset = preprocess_dataset(baseline_result, segment_df, route_stop_dist, trips)
