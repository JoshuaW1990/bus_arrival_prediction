"""
predict the estimated arrival time based on the
"""

# import module
import pandas as pd
from geopy.distance import great_circle
import os
from datetime import datetime, timedelta
from sqlalchemy import create_engine

path = '../'

engine = create_engine('postgresql://joshuaw:Wj2080989@localhost:5432/bus_prediction', echo=False)


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
    # print "calcualte the estimated result"
    # estimated_segment_df = generate_estimated_arrival_time_baseline3(api_data, segment_df, route_stop_dist, trips)
    # estimated_segment_df.to_csv('full_api_baseline.csv')
    # estimated_segment_df = pd.read_sql("SELECT * FROM full_api_baseline", con=engine)
    estimated_segment_df = pd.read_csv('full_api_baseline.csv')
    print "calcualte the actual result"
    result = generate_actual_arrival_time(full_history, estimated_segment_df, route_stop_dist)
    result['service_date'] = pd.to_numeric(result['service_date'])

    result['weather'] = result['service_date'].apply(lambda x: weather_df[weather_df.date == x].iloc[0]['weather'])
    result['rush_hour'] = result['time_of_day'].apply(lambda x: 1 if '20:00:00' >= x[11:19] >= '17:00:00' else 0)

    print "complete exporting the result of the dataset"
    return result


# def generate_feature_api(single_segment, dist_along_route, time_of_day, route_id):
#     """
#     caculcate the dictionary of feature api
#
#     :param single_segment:
#     :param dist_along_route:
#     :param time_of_day:
#     :param route_id:
#     :return:
#     """
#     feature_api = dict()
#     feature_api['actual_arrival_time'] = single_segment['travel_duration'].sum()
#     feature_api['trip_id'] = single_segment.iloc[0]['trip_id']
#     feature_api['vehicle_id'] = single_segment.iloc[0]['vehicle_id']
#     feature_api['route_id'] = route_id
#     feature_api['stop_id'] = single_segment.iloc[-1]['segment_end']
#     feature_api['time_of_day'] = time_of_day
#     feature_api['date'] = single_segment.iloc[0]['service_date']
#     feature_api['dist_along_route'] = dist_along_route
#     return feature_api


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
    try:
        current_segment = single_segment[single_segment['segment_end'].apply(lambda x: single_route_stop_dist[single_route_stop_dist['stop_id'] == x].iloc[0]['dist_along_route'] <= target_dist)]
    except:
        print "error"
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
    result = []
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
            result.append((arrival_time, trip_id, service_date))
        else:
            continue
    result.sort(key=lambda x: x[0])
    result.reverse()
    return result


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
        travel_duration = single_segment.iloc[i]['travel_duration']
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
    travel_duration = timedelta(0, travel_duration)
    next_timestamp = prev_timestamp + travel_duration
    time_of_day = calculate_arrival_time(dist_along_route, prev_distance, next_distance, prev_timestamp, next_timestamp)

    return str(time_of_day)



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
    result['delay'] = (result['actual_arrival_time'] - result['estimated_arrival_time']) / result['estimated_arrival_time']
    result['ratio'] = result['actual_arrival_time'] / result['estimated_arrival_time']

    # calculate the average delay
    return result['delay'].mean(), result['ratio'].mean()


def calcualte_prev_arrival_time(filtered_segment, segment_list, dist_along_route, single_route_stop_dist):
    """
    calculate the prev arrival time

    Algorithm:
    1) divide the filtered segment by segment start and the segment end
    2) obtain the arrival time in the first segment list:
        (1) calculate the distance between the segment start and the segment end
        (2) calculate the distance between the current distance and the segment end
        (3) use the last record in the grouped segment data with the specific segment start and the segment end
        (4) get the travel duration of the last record
        (5) calculate the arrival time for this segment according to the ratio
    3) add the travel duration of the other segments in the segment list in for loop



    :param filtered_segment:
    :param segment_list:
    :return:
    """
    prev_arrival_time = 0.0
    filtered_segment['segment_start_dist'] = filtered_segment['segment_start'].apply(lambda x: single_route_stop_dist[single_route_stop_dist.stop_id == x].iloc[0]['dist_along_route'])
    filtered_segment['segment_end_dist'] = filtered_segment['segment_end'].apply(lambda x: single_route_stop_dist[single_route_stop_dist.stop_id == x].iloc[0]['dist_along_route'])
    grouped = filtered_segment.groupby(['segment_start', 'segment_end'])
    # obtain the arrival time in the first segment list
    current_segment = segment_list[0]
    single_record = grouped.get_group(current_segment).iloc[-1]
    distance_stop_stop = single_record['segment_end_dist'] - single_record['segment_start_dist']
    distance_loc_stop = single_record['segment_end_dist'] - dist_along_route
    ratio = distance_loc_stop / distance_stop_stop
    travel_duration = single_record['travel_duration']
    prev_arrival_time += travel_duration * ratio
    # add the travel duration of the other segments in the segment list in for loop
    for i in xrange(1, len(segment_list)):
        current_segment = segment_list[i]
        if current_segment in grouped.groups.keys():
            prev_arrival_time += grouped.get_group(current_segment).iloc[-1]['travel_duration']
        else:
            prev_arrival_time += filtered_segment['travel_duration'].mean()
    return prev_arrival_time


def obtain_segment_list(current_route_stop_dist, single_route_stop_dist, stop_id, dist_along_route, stops):
    """
    calcualte the distance according to the gps location

    Algorithm:
    1) obtain the gps location of the target stop
    2) add the gps location for all the records of the current_route_stop_dist
    3) define the empty segment list, index= 0, start_index, end_index is None
    4) while index < len(current_route_stop_dist):
        (1) calcualte the distance for the current record of the current_route_stop_dist
        (2) if distance <= dist_loc_stop:
            a) if start_index is None:
                start_index = index
            b) else:
                end_index = index
        (3) else:
            a) if start_index is not None and end_index is not None:
                segment_list.append((current_segment[start_index], current_segment[end_index]))
            b) start_index, end_index = None, None
        (4) index += 1
    5) return segment list

    :param current_route_stop_dist:
    :param stop_id:
    :param dist_along_route:
    :param stops:
    :return:
    """
    current_record = stops[stops.stop_id == stop_id].iloc[0]
    target_location = (current_record.stop_lat, current_record.stop_lon)
    dist_loc_stop = single_route_stop_dist[single_route_stop_dist.stop_id == stop_id].iloc[0]['dist_along_route'] - dist_along_route
    tmp_route_stop_dist = pd.DataFrame(current_route_stop_dist)
    tmp_route_stop_dist['lat_lon'] = tmp_route_stop_dist['stop_id'].apply(lambda x: tuple(stops[stops.stop_id == x].iloc[0][['stop_lat', 'stop_lon']]))
    index = 0
    segment_list = []
    start_index, end_index = None, None
    while index < len(tmp_route_stop_dist):
        current_record = tmp_route_stop_dist.iloc[index]
        current_distance = great_circle(current_record.get('lat_lon'), target_location).meters
        if current_distance <= dist_loc_stop:
            if start_index is None:
                start_index = index
            else:
                end_index = index
        else:
            if start_index is not None and end_index is not None:
                segment_list.append((start_index, end_index))
            start_index, end_index = None, None
        index += 1
    if start_index is not None and end_index is not None:
        segment_list.append((start_index, end_index))
    return segment_list


def preprocess_dataset(baseline_result, segment_df, route_stop_dist, trips, stops):
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
        (6) generate the prev_arrival_time
            a) filter the segment data by service date
            b) generate the segment list between the dist_along_route and the target stop
                (a) index = 0
                (b) use the while loop to obtain the start index which is just the stop before the dist_along_route
                (c) use the same while loop to obtain the end index which is the target stop
                (d) obtain the filtered stop sequence from the start index and the end index
                (e) generate the segment list from the filtered stop sequence
            c) calculate the prev_arrival_time
        (7) generate the delay_neighbor_stops
            a) filter the segment by half an hour
            b) divide the filtered segment by trip id
            c) for trip_id, single_segment in grouped:
                (a) obtain the specific route id for the single_segment
                (b) obtain the route stop dist with the corresponding route id and add the new column indicating the distance between the current stop and the target stop
                (c) generate the segment list according to the distance by comparing with the (target_stop_location - current_location)
                (d) for each segment in stop list:
                    i) generate the feature api list for the corresponding segment
                    ii) use the baseline3 to calculate the estimated arrival time
                    iii) calculate the delay
            d) obtain the average delay
        (6) save the record in the result

    :param baseline_result:
    :param segment_df:
    :param route_stop_dist:
    :param trips:
    :return:
    """
    result = pd.DataFrame(columns=['trip_id', 'service_date', 'weather', 'rush_hour', 'baseline_result', 'delay_current_trip', 'ratio_current_trip', 'delay_prev_trip', 'ratio_prev_trip', 'prev_arrival_time',  'actual_arrival_time', 'shape_id', 'stop_id', 'time_of_day', 'dist_along_route'])
    print "length of the baseline_result.csv file: ", len(baseline_result)
    for i in xrange(len(baseline_result)):
        if i % 1000 == 0:
            print "index is ", i
        # obtain single record, trip id, service date, route id, dist_along_route
        single_record = baseline_result.iloc[i]
        trip_id = single_record.get('trip_id')
        service_date = single_record.get('service_date')
        if service_date == 20160118:
            print "come to debug place"
        stop_id = single_record.get('stop_id')
        shape_id = single_record.get('shape_id')
        dist_along_route = single_record.get('dist_along_route')
        time_of_day = single_record.get('time_of_day')
        single_route_stop_dist = route_stop_dist[route_stop_dist['shape_id'] == shape_id]

        # generate delay of the current trip
        single_segment = segment_df[(segment_df.trip_id == trip_id) & (segment_df.service_date.isin([service_date]))]
        single_segment = single_segment[single_segment['timestamp'] <= time_of_day]
        if len(single_segment) == 0:
            continue
        current_time_of_day = single_segment.iloc[0]['timestamp']
        target_dist = dist_along_route
        initial_dist = single_route_stop_dist[single_route_stop_dist['stop_id'] == single_segment.iloc[0]['segment_start']].iloc[0]['dist_along_route']
        feature_api = generate_feature_api(single_segment, initial_dist, target_dist, current_time_of_day, single_route_stop_dist)
        if feature_api is None:
            continue
        delay_current_trip, ratio_current_trip = calculate_average_delay(feature_api, segment_df, route_stop_dist, trips)
        # print delay_current_trip

        # generate the delay of the previous trip
        trip_list = list(trips[trips['shape_id'] == shape_id]['trip_id'])
        single_segment = segment_df[(segment_df.trip_id.isin(trip_list)) & (segment_df.service_date.isin([service_date - 1, service_date]))]
        prev_trip_list = obtain_prev_trip(single_segment, stop_id, time_of_day)
        if prev_trip_list == []:
            continue
        current_segment = single_segment
        for prev_trip_tuple in prev_trip_list:
            single_segment = current_segment[(current_segment['trip_id'] == prev_trip_tuple[1]) & (current_segment['service_date'] == prev_trip_tuple[2])]
            current_time_of_day = obtain_time_of_day(single_segment, dist_along_route, single_route_stop_dist)
            if current_time_of_day is not None:
                break
        if current_time_of_day is None:
            continue
        target_dist = single_route_stop_dist[single_route_stop_dist['stop_id'] == stop_id].iloc[0]['dist_along_route']
        feature_api = generate_feature_api(single_segment, dist_along_route, target_dist, current_time_of_day, single_route_stop_dist)
        if feature_api is None:
            continue
        delay_prev_trip, ratio_prev_trip = calculate_average_delay(feature_api, segment_df, route_stop_dist, trips)
        # print delay_prev_trip

        # generate the prev_arrival_time
        # generate the segment list between the dist_along_route and the target stop
        index = 0
        while index < len(single_route_stop_dist):
            if single_route_stop_dist.iloc[index]['dist_along_route'] >= dist_along_route:
                break
            index += 1
        if index == len(single_route_stop_dist):
            continue
        start_index = index - 1
        while index < len(single_route_stop_dist):
            if single_route_stop_dist.iloc[index]['stop_id'] == stop_id:
                break
            index += 1
        if index == len(single_route_stop_dist):
            continue
        end_index = index + 1
        filtered_stop_sequence = list(single_route_stop_dist.stop_id)[start_index:end_index]
        segment_list = [(filtered_stop_sequence[_ - 1], filtered_stop_sequence[_]) for _ in range(1, len(filtered_stop_sequence))]
        # calculate the prev_arrival_time
        filtered_segment = segment_df[segment_df.service_date.isin([service_date - 1, service_date])]
        filtered_segment = filtered_segment[(filtered_segment.segment_start.isin(filtered_stop_sequence[:-1])) & (filtered_segment.segment_end.isin(filtered_stop_sequence[1:]))]
        filtered_segment = filtered_segment[filtered_segment['timestamp'].apply(lambda x: datetime.strptime(x[:19], '%Y-%m-%d %H:%M:%S') <= datetime.strptime(time_of_day, '%Y-%m-%d %H:%M:%S'))]
        prev_arrival_time = calcualte_prev_arrival_time(filtered_segment, segment_list, dist_along_route, single_route_stop_dist)
        # print "previous arrival time: ", prev_arrival_time
        # print prev_arrival_time

        # # generate the delay_neighbor_stops
        # tmp_time_of_day = str(datetime.strptime(time_of_day, '%Y-%m-%d %H:%M:%S') - timedelta(0, 1800))
        # filtered_segment = segment_df[(segment_df.timestamp <= time_of_day) & (segment_df.timestamp >= tmp_time_of_day)]
        # if len(filtered_segment) == 0:
        #     continue
        # grouped = filtered_segment.groupby(['trip_id'])
        # delay_neighbor_stops_list = []
        # for name, item in grouped:
        #     current_trip_id = name
        #     current_route_id = trips[trips.trip_id == current_trip_id].iloc[0]['route_id']
        #     current_route_stop_dist = route_stop_dist[route_stop_dist.route_id == current_route_id]
        #     segment_list = obtain_segment_list(current_route_stop_dist, single_route_stop_dist, stop_id, dist_along_route, stops)
        #     if not segment_list:
        #         continue
        #     for segment in segment_list:
        #         start_index, end_index = segment
        #         current_stop_list = list(current_route_stop_dist[start_index:end_index + 1]['stop_id'])
        #         single_segment = item[(item.segment_start.isin(current_stop_list[:-1])) & (item.segment_end.isin(current_stop_list[1:]))]
        #         if len(single_segment) == 0:
        #             continue
        #         current_time_of_day = single_segment.iloc[0]['timestamp']
        #         target_dist = current_route_stop_dist[current_route_stop_dist['stop_id'] == single_segment.iloc[-1]['segment_end']].iloc[0]['dist_along_route']
        #         initial_dist = current_route_stop_dist[current_route_stop_dist['stop_id'] == single_segment.iloc[0]['segment_start']].iloc[0]['dist_along_route']
        #         feature_api = generate_feature_api(single_segment, initial_dist, target_dist, current_time_of_day, current_route_stop_dist)
        #         if feature_api is None:
        #             continue
        #         delay_neighbor_stops_list.append(calculate_average_delay(feature_api[-1:], segment_df, route_stop_dist, trips))
        # if len(delay_neighbor_stops_list) == 0:
        #     delay_neighbor_stops = 0.0
        # else:
        #     delay_neighbor_stops = sum(delay_neighbor_stops_list) / float(len(delay_neighbor_stops_list))
        # # print delay_neighbor_stops


        # 'weather', 'rush_hour', 'baseline_result', 'delay_current_trip', 'delay_prev_trip', 'prev_arrival_time', 'delay_neighbor_stops',
        result.loc[len(result)] = [trip_id, service_date, single_record.get('weather'), single_record.get('rush_hour'), single_record.get('estimated_arrival_time'), delay_current_trip, ratio_current_trip, delay_prev_trip, ratio_current_trip, prev_arrival_time,   single_record.get('actual_arrival_time'), shape_id, stop_id, time_of_day, dist_along_route]

    return result






#################################################################################################################
#                                    predict dataset                                                            #
#################################################################################################################


#################################################################################################################
#                                    debug section                                                              #
#################################################################################################################
# route_stop_dist = pd.read_csv('route_stop_dist.csv')
# trips = pd.read_csv(path + 'data/GTFS/gtfs/trips.txt')
segment_df = pd.read_csv('full_segment.csv')
# stops = pd.read_csv(path + 'data/GTFS/gtfs/stops.txt')
route_stop_dist = pd.read_sql("SELECT * FROM route_stop_dist", con=engine)
trips = pd.read_sql("SELECT * FROM trips", con=engine)
# segment_df = pd.read_sql("SELECT * FROM full_segment", con=engine)
stops = pd.read_sql("SELECT * FROM stops", con=engine)

# single_trip = api_data.iloc[0].trip_id
# print single_trip
# route_id = api_data.iloc[0].route_id
# trip_list = list(api_data[api_data.route_id == route_id]['trip_id'])

# single_trip = 'CA_H6-Weekday-044000_MISC_488'
# baseline_result = generate_complete_dateset(api_data, segment_df, route_stop_dist, trips, full_history, weather_df, [single_trip])


# api_data = pd.read_csv('full_api_data.csv')
# weather_df = pd.read_csv('weather.csv')
# api_data = pd.read_sql("SELECT * FROM full_api_data", con=engine)
# weather_df = pd.read_sql("SELECT * FROM weather", con=engine)
# full_history = pd.read_csv('preprocessed_full_history.csv')
# baseline_result = generate_complete_dateset(api_data, segment_df, route_stop_dist, trips, full_history, weather_df)
# baseline_result.to_csv('full_baseline_result.csv')
# baseline_result.to_sql(name='full_baseline_result', con=engine, if_exists='replace', index_label='id')

baseline_result = pd.read_csv("full_baseline_result.csv")


# # Preprocess the full_baseline_result to obtain part of the route ids to test
# route_set = set(baseline_result.route_id)
# route_list = sorted(list(route_set))
# ROUTE_NUM = 1
# route_filter_list = route_list[:ROUTE_NUM]
# baseline_result = baseline_result[baseline_result.route_id.isin(route_filter_list)]

# Preprocess the full_baseline_result to remove the first day of the baseline_result data to avoid error
baseline_result = baseline_result[baseline_result.service_date > 20160104]

dataset = preprocess_dataset(baseline_result, segment_df, route_stop_dist, trips, stops)
dataset.to_csv('full_dataset.csv')
