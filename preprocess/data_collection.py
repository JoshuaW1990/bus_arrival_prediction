"""
Preprocess the dataset
"""

# import module
import pandas as pd
import os
from datetime import datetime, timedelta
from dateutil.rrule import rrule, DAILY
import requests
import random
import urllib


#################################################################################################################
#                   helper function for api data, segment data, and other calcualtion                           #
#################################################################################################################
"""
Helper functions for generating api data, segment data and even the arrival time

list of helper functions:

* calculate_arrival_time
* calculate_arrival_distance
* extract_time
* calculate_time_span
* calculate_time_from_stop
* filter_single_history

"""


def calculate_arrival_time(stop_dist, prev_dist, next_dist, prev_timestamp, next_timestamp):
    """
     Calculate the arrival time according to the given tuple (prev_dist, next_dist), the current location, the timestamp of the prev location, and the timestamp of the next location

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


def extract_time(time):
    """
    Convert the string into datetime format.
    
    :param time: string of time need to be converted. Example: '2017-01-16T15:09:28Z'
    :return: the time in datetime format
    """
    result = datetime.strptime(time[11: 19], '%H:%M:%S')
    return result


def calculate_time_span(time1, time2):
    """
    Calculate the duration of two timepoints
    
    :param time1: previous time point in string format, ex: '2017-01-16T15:09:28Z'
    :param time2: next time point in string format, ex: '2017-01-16T15:09:28Z'
    :return: float number of seconds
    """
    timespan = extract_time(time2) - extract_time(time1)
    return timespan.total_seconds()


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


#################################################################################################################
#                                weather.csv                                                                    #
#################################################################################################################


def get_precip(gooddate, api_token):
    """
    Download the weather information for a specific date
    :param gooddate: date for downloading
    :param api_token: the token for api interface
    :return: list of the data
    """
    urlstart = 'http://api.wunderground.com/api/' + api_token + '/history_'
    urlend = '/q/NY/New_York.json'

    url = urlstart + str(gooddate) + urlend
    data = requests.get(url).json()
    result = None
    for summary in data['history']['dailysummary']:
        rain = summary['rain']
        snow = summary['snow']
        if snow == '1':
            weather = '2'
        elif rain == '1':
            weather = '1'
        else:
            weather = '0'
        result = [gooddate, rain, snow, weather]
    return result


def download_weather(date_start, date_end, api_token):
    """
    download the weather information for a date range
    
    :param date_start: start date, string, ex: '20160101'
    :param date_end: similar to date_start
    :param api_token: the token for api interface
    :return: dataframe for weather table
    weather = 2: snow
    weather = 1: rain
    weather = 0: sunny
    """

    a = datetime.strptime(date_start, '%Y%m%d')
    b = datetime.strptime(date_end, '%Y%m%d')

    result = pd.DataFrame(columns=['date', 'rain', 'snow', 'weather'])
    for dt in rrule(DAILY, dtstart=a, until=b):
        current_data = get_precip(dt.strftime("%Y%m%d"), api_token)
        if current_data is None:
            continue
        else:
            result.loc[len(result)] = current_data
    return result


#################################################################################################################
#                                route_stop_dist.csv                                                            #
#################################################################################################################
"""
Calculate the distance of each stops for a specific route from the initial stop.

It will read three different files: trips.txt, stop_times.txt and history file.
Use the stop_times.txt and trips.txt file to obtain the stop sequence for each route and use the historical data to calculate the actual distance for each stop.
"""


def calculate_stop_distance(stop_times, history, direction_id=0):
    """
    Calculate the distance of each stop with its initial stop. Notice that the dist_along_route is the distance between the next_stop and the initial stop
    
    Algorithm:
    split the history and stop_times table according to the route id and shape id
    for each subset of the divided history table:
        get the route id and shape id for the subset
        get the corresponding subset of the stop_times table
        get the stop sequence from this subset
        define a new dataframe based on the stop sequence for that shape id
        find the distance from history data for the corresponding stop and shape id
        save the result for this subset
    concatenate all the results
        
    :param stop_times: the stop_times table read from stop_times.txt file in GTFS
    :param history: the history table from preprocessed history.csv file
    :param direction_id: the direction id which can be 0 or 1
    :return: the route_stop_dist table in dataframe
    """
    route_grouped_history = history.groupby(['route_id', 'shape_id'])
    route_grouped_stop_times = stop_times.groupby(['route_id', 'shape_id'])
    result_list = []
    for name, single_route_history in route_grouped_history:
        route_id, shape_id = name
        flag = 0
        current_result = pd.DataFrame()
        single_stop_times = route_grouped_stop_times.get_group((route_id, shape_id))
        trip_id = single_stop_times.iloc[0]['trip_id']
        single_stop_times = single_stop_times[single_stop_times.trip_id == trip_id]
        single_stop_times.reset_index(inplace=True)
        current_result['stop_id'] = single_stop_times['stop_id']
        current_result['route_id'] = route_id
        current_result['shape_id'] = shape_id
        current_result['direction_id'] = direction_id
        stop_grouped = single_route_history.groupby(['next_stop_id']).mean()
        stop_grouped.reset_index(inplace=True)
        stop_grouped['next_stop_id'] = pd.to_numeric(stop_grouped['next_stop_id'])
        stop_set = set(stop_grouped['next_stop_id'])
        for i in xrange(len(current_result)):
            next_stop_id = current_result.iloc[i]['stop_id']
            if next_stop_id not in stop_set:
                print route_id, shape_id
                flag = 1
                break
            else:
                dist_along_route = stop_grouped[stop_grouped.next_stop_id == next_stop_id].iloc[0]['dist_along_route']
                current_result.set_value(i, 'dist_along_route', dist_along_route)
        if flag == 1:
            continue
        else:
            result_list.append(current_result)
    result = pd.concat(result_list, ignore_index=True)
    return result


#################################################################################################################
#                                    segment.csv                                                                #
#################################################################################################################
"""
generate the segment table
"""


def generate_original_segment(full_history_var, weather, stop_times_var):
    """
    Generate the original segment data
    
    Algorithm:
    Split the full historical data according to the service date, trip_id with groupby function
    For name, item in splitted historical dataset:
        service date, trip_id = name[0], name[1]
        Find the vehicle id which is the majority elements in this column (For removing the abnormal value in historical data)
        calcualte the travel duration within the segement of this splitted historical data and save the result into list
    concatenate the list
    
    :param full_history_var: the historical data after filtering
    :param weather: the dataframe for the weather information
    :param stop_times_var: the dataframe from stop_times.txt
    :return: dataframe for the original segment
    format:
    segment_start, segment_end, timestamp, travel_duration, weather, service date, day_of_week, trip_id, vehicle_id
    """
    full_history_var = full_history_var[full_history_var.total_distance > 0]
    grouped = list(full_history_var.groupby(['service_date', 'trip_id']))
    result_list = []
    step_count = range(0, len(grouped), len(grouped) / 10)
    for index in range(len(grouped)):
        name, single_history = grouped[index]
        if index in step_count:
            print "process: ", str(step_count.index(index)) + '/' + str(10)
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
        stop_sequence = [item for item in list(stop_times_var[stop_times_var.trip_id == trip_id].stop_id)]
        current_segment_df = generate_original_segment_single_history(majority_history, stop_sequence)
        if current_segment_df is None:
            continue
        current_weather = weather[weather.date == service_date].iloc[0]['weather']
        current_segment_df['weather'] = current_weather
        day_of_week = datetime.strptime(str(service_date), '%Y%m%d').weekday()
        current_segment_df['service_date'] = service_date
        current_segment_df['day_of_week'] = day_of_week
        current_segment_df['trip_id'] = trip_id
        current_segment_df['vehicle_id'] = majority_vehicle
        result_list.append(current_segment_df)
    if result_list != []:
        result = pd.concat(result_list, ignore_index=True)
    else:
        return None
    return result


def generate_original_segment_single_history(history, stop_sequence):
    """
    Calculate the travel duration for a single historical data
    
    Algorithm:
    Filter the historical data with the stop sequence here
    arrival_time_list = []
    i = 0
    while i < len(history):
        use prev and the next to mark the record:
            prev = history[i - 1]
            next = history[i]
        check whether the prev stop is the same as the next stop:
            if yes, skip this row and continue to next row
        prev_distance = prev.dist_along_route - prev.dist_from_stop
        next_distance = next.dist_along_route - next.dist_from_stop
        if prev_distance == next_distance:
            continue to next row
        elif prev.dist_from_stop = 0:
            current_arrival_time = prev.timestamp
        else:
            current_arrival_duration = calcualte_arrival_time(prev.dist_along_route, prev_distance, next_distance, prev_timestamp, next_timestamp)
        arrival_time_list.append((prev.next_stop_id, current_arrival_time))
    result = pd.Dataframe
    for i in range(1, len(arrival_time_list)):
        prev = arrival_time_list[i - 1]
        next = arrival_time_list[i]
        segment_start, segment_end obtained
        travel_duration = next[1] - prev[1]
        timestamp = prev[1]
        save the record to result

    :param history: single historical data
    :param stop_sequence: stop sequence for the corresponding trip id
    :return: the dataframe of the origianl segment dataset
    format:
    segment_start, segment_end, timestamp, travel_duration
    """
    single_history = filter_single_history(history, stop_sequence)
    if single_history is None or len(single_history) < 3:
        return None
    arrival_time_list = []
    grouped_list = list(single_history.groupby('next_stop_id'))
    if len(grouped_list) < 3:
        return None
    history = pd.DataFrame(columns=single_history.columns)
    for i in xrange(len(grouped_list)):
        history.loc[len(history)] = grouped_list[i][1].iloc[-1]
    history.sort_values(by='timestamp', inplace=True)
    if history.iloc[0]['total_distance'] < 1:
        prev_record = history.iloc[1]
        i = 2
    else:
        prev_record = history.iloc[0]
        i = 1
    while i < len(history):
        next_record = history.iloc[i]
        while stop_sequence.index(prev_record.next_stop_id) >= stop_sequence.index(next_record.next_stop_id):
            i += 1
            if i == len(history):
                break
            next_record = history.iloc[i]
        if i == len(history):
            break
        prev_distance = float(prev_record.total_distance)
        next_distance = float(next_record.total_distance)
        prev_timestamp = datetime.strptime(prev_record.timestamp, '%Y-%m-%dT%H:%M:%SZ')
        next_timestamp = datetime.strptime(next_record.timestamp, '%Y-%m-%dT%H:%M:%SZ')
        if prev_distance == next_distance:
            # the bus didn't move yet
            i += 1
            continue
        if prev_record.dist_from_stop == 0:
            # if prev.dist_from_stop is 0, the bus is 0, then save result into timestamp
            current_arrival_time = prev_timestamp
        else:
            stop_dist = prev_record.dist_along_route
            current_arrival_time = calculate_arrival_time(stop_dist, prev_distance, next_distance, prev_timestamp,
                                                          next_timestamp)
        arrival_time_list.append((prev_record.next_stop_id, current_arrival_time))
        prev_record = next_record
        i += 1
    result = pd.DataFrame(columns=['segment_start', 'segment_end', 'timestamp', 'travel_duration'])
    for i in range(1, len(arrival_time_list)):
        prev_record = arrival_time_list[i - 1]
        next_record = arrival_time_list[i]
        segment_start, segment_end = prev_record[0], next_record[0]
        timestamp = prev_record[1]
        travel_duration = next_record[1] - prev_record[1]
        travel_duration = travel_duration.total_seconds()
        result.loc[len(result)] = [segment_start, segment_end, str(timestamp), travel_duration]
    return result


def improve_dataset_unit(segment_df, stop_sequence):
    """
    improve the dataset for a specific trip_id at a specific date: add the skipped segments back into the dataframe
    
    Algorithm:
    define result_df
    For each row in segment_df:
        obtain segment_start, segment_end, timestamp, travel_duration from the current row
        start_index: index of segment_start in stop_sequence
        end_index: ...
        count = end_index - start_index
        if count is 1, save the current row and continue to next row
        average_travel_duration = travel_duration / count
        For index in range(start_index, end_index):
            current_segment_start = stop_sequence[index]
            current_segment_end = stop_sequence[index + 1]
            save the new row with the timestamp, average_travel_duration, current_segment_start, and current_segment_end into result_df
            timestamp = timestamp + average_travel_duration
    return result_df

    :param segment_df: a subset of segment table with one trip id and service date
    :param stop_sequence: stop sequence for the corresponding trip id
    :return: dataframe for improved segment table

    return format:
    segment_start, segment_end, timestamp, travel_duration
    """
    result = pd.DataFrame(columns=['segment_start', 'segment_end', 'timestamp', 'travel_duration'])
    for i in xrange(len(segment_df)):
        segment_start = segment_df.iloc[i]['segment_start']
        segment_end = segment_df.iloc[i]['segment_end']
        timestamp = segment_df.iloc[i]['timestamp']
        travel_duration = segment_df.iloc[i]['travel_duration']
        start_index = stop_sequence.index(segment_start)
        end_index = stop_sequence.index(segment_end)
        count = end_index - start_index
        if count <= 0:
            print "error"
            continue
        average_travel_duration = float(travel_duration) / float(count)
        for j in range(start_index, end_index):
            current_segment_start = stop_sequence[j]
            current_segment_end = stop_sequence[j + 1]
            result.loc[len(result)] = [current_segment_start, current_segment_end, timestamp, average_travel_duration]
            timestamp = datetime.strptime(timestamp[:19], '%Y-%m-%d %H:%M:%S') + timedelta(0, average_travel_duration)
            timestamp = str(timestamp)
    return result


def improve_dataset(segment_df, stop_times, weather_df):
    """
    Improve the segment table by adding the skipped stops and other extra columns like weather, day_of_week, etc.
    
    algorithm:
    split the segment dataframe by groupby(service_date, trip_id)
    result_list = []
    for name, item in grouped_segment:
        obtained the improved segment data for the item
        add the columns:  weather, service date, day_of_week, trip_id, vehicle_id
        save the result into result_list
    concatenate the dataframe in the result_list

    :param segment_df: the dataframe of segment table
    :param stop_times: the dataframe of the stop_times.txt file in GTFS dataset
    :param weather_df: the dataframe of the weather information
    :return: the dataframe of the improved segment table
    """
    grouped_list = list(segment_df.groupby(['service_date', 'trip_id']))
    result_list = []
    for i in xrange(len(grouped_list)):
        name, item = grouped_list[i]
        service_date, trip_id = name
        stop_sequence = list(stop_times[stop_times.trip_id == trip_id].stop_id)
        current_segment = improve_dataset_unit(item, stop_sequence)
        if current_segment is None:
            continue
        # add the other columns
        current_segment['weather'] = weather_df[weather_df.date == service_date].iloc[0].weather
        current_segment['service_date'] = service_date
        current_segment['day_of_week'] = datetime.strptime(str(service_date), '%Y%m%d').weekday()
        current_segment['trip_id'] = trip_id
        current_segment['vehicle_id'] = item.iloc[0].vehicle_id
        result_list.append(current_segment)
    if result_list == []:
        result = None
    else:
        result = pd.concat(result_list, ignore_index=True)
    return result


#################################################################################################################
#                                    api data section                                                              #
#################################################################################################################
"""
Generate the api data from the GTFS data and the historical data
"""

def generate_api_data(date_list, time_list, stop_num, route_stop_dist, full_history):
    """
    Generate the api data for the test_route_set and given time list

    Algorithm:
    Generate the set of trip id for test routes
    Generate the random test stop id for each test routes
    Filtering the historical data with trip id
    Generate the list of historical data Groupby(date, trip id)
    for each item in the list of the historical data:
        obtain the trip id and the date
        obtain the corresponding route
        obtain the corresponding stop set
        for stop in stop set:
            for each time point in the time list:
                check whether the bus has passed the stop at the time point
                if yes, continue to next stop
                otherwise, save the record into result
                
    :param time_list: the date list for testing [20160101, 20160102, ...]
    :param time_list: the time list for testing, ['12:00:00', '12:05:00', ...]
    :param stop_num: the number of the target stop for test
    :param route_stop_dist: the dataframe for the route_stop_dist table
    :param full_history: the dataframe for the history table
    :return: the dataframe for the api data
    """
    trip_route_dict = {}
    route_stop_dict = {}
    grouped = route_stop_dist.groupby(['shape_id'])
    for shape_id, single_route_stop_dist in grouped:
        stop_sequence = list(single_route_stop_dist.stop_id)
        if len(stop_sequence) < 5:
            continue
        trip_set = set(full_history[full_history.shape_id == shape_id].trip_id)
        current_dict = dict.fromkeys(trip_set, shape_id)
        trip_route_dict.update(current_dict)
        stop_set = set()
        for i in range(stop_num):
            stop_set.add(stop_sequence[random.randint(2, len(stop_sequence) - 2)])
        route_stop_dict[shape_id] = stop_set
    history = full_history[
        (full_history.trip_id.isin(trip_route_dict.keys())) & (full_history.service_date.isin(date_list))]
    history_grouped = history.groupby(['service_date', 'trip_id'])
    result = pd.DataFrame(
        columns=['trip_id', 'vehicle_id', 'route_id', 'stop_id', 'time_of_day', 'date', 'dist_along_route', 'shape_id'])
    print_dict = dict.fromkeys(date_list, True)
    for name, single_history in history_grouped:
        service_date, trip_id = name
        if service_date not in date_list:
            continue
        if print_dict[service_date]:
            print service_date
            print_dict[service_date] = False
        shape_id = trip_route_dict[trip_id]
        stop_set = [str(int(item)) for item in route_stop_dict[shape_id]]
        stop_sequence = list(route_stop_dist[route_stop_dist.shape_id == shape_id].stop_id)
        # filtering the history data: remove the abnormal value
        single_history = filter_single_history(single_history, stop_sequence)

        if single_history is None or len(single_history) < 2:
            continue
        for target_stop in stop_set:
            target_index = stop_sequence.index(float(target_stop))
            for current_time in time_list:
                prev_history = single_history[single_history['timestamp'].apply(lambda x: x[11:19] <= current_time)]
                next_history = single_history[single_history['timestamp'].apply(lambda x: x[11:19] > current_time)]
                if len(prev_history) == 0:
                    continue
                if len(next_history) == 0:
                    break
                tmp_stop = str(prev_history.iloc[-1].next_stop_id)
                tmp_index = stop_sequence.index(float(tmp_stop))
                if tmp_index > target_index:
                    break
                # If the bus does not pass the target stop, save the remained stops into the stop sequence and calculate the result
                route_id = single_history.iloc[0].route_id
                current_list = generate_single_api(current_time, route_id, prev_history, next_history, target_stop, shape_id)
                if current_list is not None:
                    result.loc[len(result)] = current_list
    return result


def generate_single_api(current_time, route_id, prev_history, next_history, stop_id, shape_id):
    """
    Calculate the single record for the api data

    Algorithm for calculate the single record:
    According to the time point, find the closest time duration (prev, next)
    Calculate the dist_along_route for the bus at current timepoint:
        calculate the space distance between the time duration (prev, next)
        calculate the time distance of two parts: (prev, current), (prev, next)
        use the ratio of the time distance to multiply with the space distance to obtain the dist_along_route for current
    According to the dista_along_route and the stop sequence confirm the remained stops including the target stop
    Count the number of the remained stops
    
    :param current_time: The current time for generating the api data
    :param route_id: the id of the route for the specific record
    :param prev_history: the dataframe of the history table before the timestamp on the record of api data with the same trip id
    :param next_history: the dataframe of the history table after the timestamp on the record of api data with the same trip id
    :param stop_id: The id of the target stop
    :param shape_id: The id of the shape (stop sequence)
    :return: the list for the result
    """
    single_trip = prev_history.iloc[0].trip_id
    prev_record = prev_history.iloc[-1]
    next_record = next_history.iloc[0]
    # calculate the dist_along_route for current
    prev_distance = float(prev_record.total_distance)
    next_distance = float(next_record.total_distance)
    prev_timestamp = datetime.strptime(prev_record['timestamp'], '%Y-%m-%dT%H:%M:%SZ')
    next_timestamp = datetime.strptime(next_record['timestamp'], '%Y-%m-%dT%H:%M:%SZ')
    # determine the current time
    if prev_record['timestamp'][11:19] <= current_time <= next_record['timestamp'][11:19]:
        time_of_day = prev_record['timestamp'][:11] + current_time + 'Z'
    else:
        # case: this trip is crossing between two days
        if current_time > next_record['timestamp'][11:19]:
            time_of_day = prev_record['timestamp'][11:19] + current_time + 'Z'
        else:
            time_of_day = next_record['timestamp'][11:19] + current_time + 'Z'
    time_of_day = datetime.strptime(time_of_day, '%Y-%m-%dT%H:%M:%SZ')
    dist_along_route = calculate_arrival_distance(time_of_day, prev_distance, next_distance, prev_timestamp, next_timestamp)
    # Generate the return list
    # trip_id    vehicle_id    route_id    stop_id    time_of_day    date    dist_along_route
    result = [single_trip, prev_record['vehicle_id'], route_id, stop_id, str(time_of_day), prev_record['service_date'], dist_along_route, shape_id]
    return result


#################################################################################################################
#                                    main function section                                                      #
#################################################################################################################
"""
Functions for users
"""

# weather data
def obtain_weather(start_date, end_date, api_token, save_path=None, engine=None):
    """
    Download the weather.csv file into save_path
    
    :param start_date: start date, string, example: '20160101'
    :param end_date: similar to start_date
    :param api_token: api_token for wunderground api interface. Anyone can apply for it in free.
    :param save_path: path of a csv file for storing the weather table.
    :param engine: database connect engine
    :return: return the weather table in dataframe
    """
    weather = download_weather(start_date, end_date, api_token)
    if save_path is not None:
        weather.to_csv(save_path)
    if engine is not None:
        weather.to_sql(name='weather', con=engine, if_exists='replace', index_label='id')
    return weather


# history data
def download_history_file(year, month, date_list, save_path):
    """
    Download the history data from nyc database. User still needs to uncompress the data into csv file
    
    :param year: integer to represent the year, example: 2016
    :param month: integer to represent the month, example: 1
    :param date_list: list of integer to represent the dates of the required data
    :param save_path: path for downloading the compressed data
    :return: None
    """
    year = str(year)
    if month < 10:
        month = '0' + str(month)
    else:
        month = str(month)
    base_url = 'http://data.mytransit.nyc/bus_time/'
    url = base_url + year + '/' + year + '-' + month + '/'
    download_file = urllib.URLopener()
    for date in date_list:
        if date < 10:
            date = '0' + str(date)
        else:
            date = str(date)
        filename = 'bus_time_' + year + month + date + '.csv.xz'
        file_url = url + filename
        download_file.retrieve(file_url, save_path + filename)


def obtain_history(start_date, end_date, trips, history_path, save_path=None, engine=None):
    """
    Generate the csv file for history data
    
    :param start_date: integer to represent the start date, example: 20160105
    :param end_date: integer to represent the end date, format is the same as start date
    :param trips: the dataframe storing the table from trips.txt file in GTFS dataset
    :param history_path: path of all the historical data. User should place all the historical data under the same directory and use this directory as the history_path. Please notice that the change of the filename might cause error.
    :param save_path: path of a csv file to store the history table
    :param engine: database connect engine
    :return: the history table in dataframe
    """
    trip_set = set(trips.trip_id)
    # generate the history data
    file_list = os.listdir(history_path)
    history_list = []
    for filename in file_list:
        if not filename.endswith('.csv'):
            continue
        if int(start_date) <= int(filename[9:17]) <= int(end_date):
            print filename
            ptr_history = pd.read_csv(history_path + filename)
            tmp_history = ptr_history[(ptr_history.dist_along_route != '\N') & (ptr_history.dist_along_route != 0) & (ptr_history.progress == 0) & (ptr_history.block_assigned == 1) & (ptr_history.dist_along_route > 1) & (ptr_history.trip_id.isin(trip_set))]
            history_list.append(tmp_history)
    result = pd.concat(history_list, ignore_index=True)
    # add some other information: total_distance, route_id, shape_id
    result['dist_along_route'] = pd.to_numeric(result['dist_along_route'])
    result['dist_from_stop'] = pd.to_numeric(result['dist_from_stop'])
    result['total_distance'] = result['dist_along_route'] - result['dist_from_stop']
    trip_route_dict = trips.set_index('trip_id').to_dict(orient='index')
    result['route_id'] = result['trip_id'].apply(lambda x: trip_route_dict[x]['route_id'])
    result['shape_id'] = result['trip_id'].apply(lambda x: trip_route_dict[x]['shape_id'])
    # export csv file
    if save_path is not None:
        result.to_csv(save_path)
    if engine is not None:
        result.to_sql(name='history', con=engine, if_exists='replace', index_label='id')
    return result


# route_stop_dist data
def obtain_route_stop_dist(trips, stop_times, history_file, save_path=None, engine=None):
    """
    Generate the csv file for route_stop_dist data. In order to obtain a more complete data for route_stop_dist, the size of the history file should be as large as possible.
    
    :param trips: the dataframe storing the table from trips.txt file in GTFS dataset
    :param stop_times: the dataframe storing the table from stop_times.txt file in GTFS dataset
    :param history_file: path of the preprocessed history file
    :param save_path: path of a csv file to store the route_stop_dist table
    :param engine: database connect engine
    :return: the route_stop_dist table in dataframe
    """
    trip_route_dict = trips.set_index('trip_id').to_dict(orient='index')
    stop_times['route_id'] = stop_times['trip_id'].apply(lambda x: trip_route_dict[x]['route_id'])
    stop_times['shape_id'] = stop_times['trip_id'].apply(lambda x: trip_route_dict[x]['shape_id'])
    history = pd.read_csv(history_file)
    route_stop_dist = calculate_stop_distance(stop_times, history)
    if save_path is not None:
        route_stop_dist.to_csv(save_path)
    if engine is not None:
        route_stop_dist.to_sql(name='route_stop_dist', con=engine, if_exists='replace', index_label='id')
    return route_stop_dist


# segment data
def obtain_segment(weather_df, trips, stop_times, route_stop_dist, full_history, training_date_list, save_path=None, engine=None):
    """
    Generate the csv file for segment table
    
    :param weather_df: the dataframe storing the weather data
    :param trips: the dataframe storing the table from trips.txt file in GTFS dataset
    :param stop_times: the dataframe storing the table from stop_times.txt file in GTFS dataset
    :param full_history: the dataframe storing the history table
    :param training_date_list: the list of dates to generate the segments from history table
    :param save_path: path of a csv file to store the segment table
    :param engine: database connect engine
    :return: the segment table in dataframe
    """
    full_history = full_history[full_history.service_date.isin(training_date_list)]
    shape_list = set(route_stop_dist.shape_id)
    full_history = full_history[full_history.shape_id.isin(shape_list)]
    segment_df = generate_original_segment(full_history, weather_df, stop_times)
    segment_df = improve_dataset(segment_df, stop_times, weather_df)
    trip_route_dict = trips.set_index('trip_id').to_dict(orient='index')
    segment_df['route_id'] = segment_df['trip_id'].apply(lambda x: trip_route_dict[x]['route_id'])
    segment_df['shape_id'] = segment_df['trip_id'].apply(lambda x: trip_route_dict[x]['shape_id'])

    if save_path is not None:
        segment_df.to_csv(save_path)
    if engine is not None:
        segment_df.to_sql(name='segment', con=engine, if_exists='replace', index_label='id')
    return segment_df


# api_data table
def obtain_api_data(route_stop_dist, full_history, date_list, time_list, stop_num, save_path=None, engine=None):
    """
    Generate the csv file for api_data table
    
    :param route_stop_dist: the dataframe storing route_stop_dist table
    :param full_history: the dataframe storing historical data
    :param date_list: the list of integers to represent the dates for generating api data. Example: [20160101, 20160102, 20160103]
    :param time_list: the list of strings to represent the time for generating api data. Example: ['12:00:00', '12:05:00', '12:10:00', '12:15:00', '12:20:00', '12:25:00', '12:30:00']. Please follow the same format.
    :param stop_num: the number of target stop for each shape id
    :param save_path: path of a csv file to store the api_data table
    :param engine: database connect engine
    :return: the dataframe storing api_data table
    """
    full_history = full_history[full_history.service_date.isin(date_list)]
    result = generate_api_data(date_list, time_list, stop_num, route_stop_dist, full_history)
    if save_path is not None:
        result.to_csv(save_path)
    if engine is not None:
        result.to_sql(name='api_data', con=engine, if_exists='replace', index_label='id')
    return result

