"""
This datacollection try to reuse the method in different places such that all the process are similar, and we can avoid the error
"""

# import module

import pandas as pd
import os
from datetime import datetime, timedelta
from dateutil.rrule import rrule, DAILY
import requests
import random

path = '../'


#################################################################################################################
#                                weather.csv                                                                    #
#################################################################################################################


def get_precip(gooddate):
    """
    Download the weather information for a specific date
    :param gooddate: date for downloading
    :return: list of the data
    """
    urlstart = 'http://api.wunderground.com/api/d083880ff5428216/history_'
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


def download_weather(date_start, date_end):
    """
    download the weather information for a date range
    :param date_start: start date, string, ex: '20160101'
    :param date_end: similar to date_start
    :return: list of the table record
    weather = 2: snow
    weather = 1: rain
    weather = 0: sunny
    """

    a = datetime.strptime(date_start, '%Y%m%d')
    b = datetime.strptime(date_end, '%Y%m%d')

    result = pd.DataFrame(columns=['date', 'rain', 'snow', 'weather'])
    for dt in rrule(DAILY, dtstart=a, until=b):
        current_data = get_precip(dt.strftime("%Y%m%d"))
        if current_data is None:
            continue
        else:
            result.loc[len(result)] = current_data
    return result


#################################################################################################################
#                                route_stop_dist.csv                                                            #
#################################################################################################################
"""
Calcualte the distance of each stops for a specific route from the initial stop.

It will read three different files: trips.txt, stop_times.txt and history file.
Use the stop_times.txt and trips.txt file to obtain the stop sequence for each route and use the historical data to calculate the actual distance for each stop.
If the specific stop has no records for the distance, we will use the average value as the result like calculating the travel duration.

Since the dist_along_route in the history data is actually the distance between the next_stop and the intial stop, which decrease the difficulty a lot.
"""


def read_data(route_num=None, direction_id=0):
    # type: (object, object) -> object
    """
    Read all the corresponding data according to the requirements: number of the routes we need to calcualte.
    Input: route_num
    Output: Three different dataframe:
    trips, stop_times, history. All of these three data should have been filtered according to the trip_id and route_id
    """
    trips = pd.read_csv(path + 'data/GTFS/gtfs/trips.txt')
    stop_times = pd.read_csv(path + 'data/GTFS/gtfs/stop_times.txt')
    # Obtain the filterd trips dataframe
    route_list = list(trips.route_id)
    non_dup_route_list = sorted(list(set(route_list)))
    if route_num is None:
        selected_routes = non_dup_route_list
    else:
        selected_routes = non_dup_route_list[:route_num]
    result_trips = trips[(trips.route_id.isin(selected_routes)) & (trips.direction_id == direction_id)]
    # Obtain the filtered stop_times dataframe
    selected_trips_var = set(list(result_trips.trip_id))
    result_stop_times = stop_times[stop_times.trip_id.isin(selected_trips_var)]
    # Obtain the filtered history dataframe
    file_list = os.listdir(path + 'data/history/')
    file_list.sort()
    history_list = []
    for single_file in file_list:
        if not single_file.endswith('.csv'):
            continue
        else:
            current_history = pd.read_csv(path + 'data/history/' + single_file)
            tmp_history = current_history[current_history.trip_id.isin(selected_trips_var)]
            if len(tmp_history) == 0:
                continue
            else:
                print "historical file name: ", single_file
                history_list.append(tmp_history)
    result_history = pd.concat(history_list, ignore_index=True)
    print "complete reading data"
    return result_trips, result_stop_times, result_history


def calculate_stop_distance(trips, stop_times, history, direction_id=0):
    """
    Calculate the distance of each stop with its initial stop. Notice that the dist_along_route is the distance between the next_stop and the initial stop
    Input: three filtered dataframe, trips, stop_times, history
    Output: One dataframe, route_stop_dist
    The format of the route_stop_dist:
    route_id    direction_id    stop_id    dist_along_route
    str         int             int        float
    """
    result = pd.DataFrame(columns=['route_id', 'direction_id', 'stop_id', 'dist_along_route'])
    selected_routes = set(trips.route_id)
    # Looping from each route to obtain the distance of each stops
    for single_route in selected_routes:
        print "route name: ", single_route
        selected_trips_var = set(trips[trips.route_id == single_route].trip_id)
        stop_sequence = list(stop_times[stop_times.trip_id == list(selected_trips_var)[0]].stop_id)
        result.loc[len(result)] = [single_route, int(direction_id), int(stop_sequence[0]), 0.0]
        selected_history = history[history.trip_id.isin(selected_trips_var)]
        for i in range(1, len(stop_sequence)):
            stop_id = stop_sequence[i]
            current_history = selected_history[selected_history.next_stop_id == stop_id]
            if float(stop_id) == float(result.iloc[-1].stop_id):
                continue
            elif len(current_history) == 0:
                dist_along_route = -1.0
            else:
                current_dist = []
                for j in range(len(current_history)):
                    current_dist.append(current_history.iloc[j].dist_along_route)
                dist_along_route = sum(current_dist) / float(len(current_dist))
            result.loc[len(result)] = [single_route, int(direction_id), int(stop_id), dist_along_route]
    result.to_csv('original_route_stop_dist.csv')
    # Since some of the stops might not record, it is necessary to check the dataframe again.
    # Because of the bug or other reasons, some of the routes have a long jump in the stop list, we should remove the corresponding stop list
    count = 1
    prev = 0
    remove_route_list = set()
    for i in range(1, len(result) - 1):
        if result.iloc[i].dist_along_route == -1:
            if result.iloc[i - 1].dist_along_route != -1:
                prev = result.iloc[i - 1].dist_along_route
            count += 1
        else:
            if count != 1:
                if count >= 4:
                    remove_route_list.add(result.iloc[i - 1].route_id)
                distance = (float(result.iloc[i].dist_along_route) - float(prev)) / float(count)
                while count > 1:
                    result.iloc[i - count + 1, result.columns.get_loc('dist_along_route')] = result.iloc[
                                                                                                 i - count].dist_along_route + float(
                        distance)
                    count -= 1
            else:
                continue
    result.to_csv('original_improve_route_stop_dist.csv')
    result = result[~result.route_id.isin(remove_route_list)]
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


def extract_time(time):
    """
    example of time(str): '2017-01-16T15:09:28Z'
    """
    result = datetime.strptime(time[11: 19], '%H:%M:%S')
    return result


def calculate_time_span(time1, time2):
    """
    Calculate the duration of two timepoints
    :param time1: previous time point, ex: '2017-01-16T15:09:28Z'
    :param time2: next time point, ex: '2017-01-16T15:09:28Z'
    :return: float number of seconds
    """
    timespan = extract_time(time2) - extract_time(time1)
    return timespan.total_seconds()


#################################################################################################################
#                                    segment.csv                                                                #
#################################################################################################################


"""
generate the segment data
"""


def select_trip_list(num_route=None, direction_id=0):
    """
    Generate the list of the trip id for the selected routes
    :param num_route: the number of the selected routes. If the num_route is None, then all the route id will be selected
    :param direction_id: the direction id can be 0 or 1
    :return: the list of the trip_id
    """
    # Read the GTFS data
    # data source: MTA, state island, Jan, 4, 2016
    trips = pd.read_csv(path + 'data/GTFS/gtfs/trips.txt')
    route_stop_dist = pd.read_csv('route_stop_dist.csv')

    # select a specific route and the corresponding trips
    route_list = list(route_stop_dist.route_id)
    non_dup_route_list = sorted(list(set(route_list)))
    if num_route is None:
        select_routes = non_dup_route_list
    else:
        select_routes = non_dup_route_list[:num_route]
    selected_trips_var = []
    for route in select_routes:
        selected_trips_var += list(trips[(trips.route_id == route) & (trips.direction_id == direction_id)].trip_id)
    return selected_trips_var


def filter_history_data(date_start, date_end, selected_trips_var):
    # type: (object, object, object) -> object
    """
    Filtering the historical data to remove the unselected trips
    :rtype: object
    :param date_start: start date for historical date, int, yyyymmdd, ex: 20160109
    :param date_end: end date for historical date. Similar to date_start. The date_start and the date_end are included.
    :param selected_trips_var: the list of the trip_id for the selected routes
    :return: dataframe for the filtered historical data
    """
    # List the historical file
    file_list_var = os.listdir(path + 'data/history/')
    file_list_var.sort()
    history_list = []
    print "filtering historical data"
    for filename in file_list_var:
        if not filename.endswith('.csv'):
            continue
        if filename[9:17] < str(date_start) or filename[9:17] > str(date_end):
            continue
        print filename
        ptr_history = pd.read_csv(path + 'data/history/' + filename)
        tmp_history = ptr_history[
            (ptr_history.trip_id.isin(selected_trips_var)) & (ptr_history.dist_along_route != '\N') & (
                ptr_history.dist_along_route != 0) & (ptr_history.progress == 0) & (ptr_history.block_assigned == 1) & (ptr_history.dist_along_route < 1)]
        history_list.append(tmp_history)
    result = pd.concat(history_list, ignore_index=True)
    return result


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
    grouped = list(full_history_var.groupby(['service_date', 'trip_id']))
    print len(grouped)
    result_list = []
    for index in range(len(grouped)):
        name, single_history = grouped[index]
        if index % 1000 == 0:
            print index
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
    history = history[history.next_stop_id.isin(stop_sequence)]
    if len(history) < 3:
        return None
    arrival_time_list = []
    i = 1
    while i < len(history):
        prev_record = history.iloc[i - 1]
        next_record = history.iloc[i]
        while i < len(history) and stop_sequence.index(prev_record.next_stop_id) >= stop_sequence.index(
                next_record.next_stop_id):
            i += 1
            if i == len(history):
                break
            if stop_sequence.index(prev_record.next_stop_id) == stop_sequence.index(next_record.next_stop_id):
                prev_record = next_record
            next_record = history.iloc[i]
        if i == len(history):
            break
        prev_distance = float(prev_record.dist_along_route) - float(prev_record.dist_from_stop)
        next_distance = float(next_record.dist_along_route) - float(next_record.dist_from_stop)
        prev_timestamp = datetime.strptime(prev_record.timestamp, '%Y-%m-%dT%H:%M:%SZ')
        next_timestamp = datetime.strptime(next_record.timestamp, '%Y-%m-%dT%H:%M:%SZ')
        # if prev.dist_from_stop is 0, the bus is 0, then save result into timestamp
        if prev_distance == next_distance:
            # if the prev_distance and the next_distance is the same, continue to the next row
            i += 1
            continue
        elif prev_record.dist_from_stop == 0:
            # if prev.dist_from_stop is 0, the bus is 0, then save result into timestamp
            current_arrival_time = prev_timestamp
        else:
            stop_dist = prev_record.dist_along_route
            current_arrival_time = calculate_arrival_time(stop_dist, prev_distance, next_distance, prev_timestamp,
                                                          next_timestamp)
        arrival_time_list.append((prev_record.next_stop_id, current_arrival_time))
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
    This funciton is used to improve the dataset for a specific trip_id at a spacific date.
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
        if count < 0:
            print "error"
            continue
        if count == 1:
            result.loc[len(result)] = [segment_start, segment_end, timestamp, travel_duration]
        average_travel_duration = float(travel_duration) / float(count)
        for j in range(start_index, end_index):
            current_segment_start = stop_sequence[j]
            current_segment_end = stop_sequence[j + 1]
            result.loc[len(result)] = [current_segment_start, current_segment_end, timestamp, average_travel_duration]
            timestamp = datetime.strptime(timestamp[:19], '%Y-%m-%d %H:%M:%S') + timedelta(0, average_travel_duration)
            timestamp = str(timestamp)
    return result


def improve_dataset():
    """
    algorithm:
    split the segment dataframe by groupby(service_date, trip_id)
    result_list = []
    for name, item in grouped_segment:
        obtained the improved segment data for the item
        add the columns:  weather, service date, day_of_week, trip_id, vehicle_id
        save the result into result_list
    concatenate the dataframe in the result_list

    segment_start, segment_end, timestamp, travel_duration, weather, service date, day_of_week, trip_id, vehicle_id
    """
    segment_df = pd.read_csv('original_segment.csv')
    stop_times = pd.read_csv(path + 'data/GTFS/gtfs/stop_times.txt')
    grouped_list = list(segment_df.groupby(['service_date', 'trip_id']))
    print "length of the total grouped list: ", len(grouped_list)

    result_list = []
    for i in xrange(len(grouped_list)):
        if i % 1000 == 0:
            print i
        name, item = grouped_list[i]
        service_date, trip_id = name
        stop_sequence = list(stop_times[stop_times.trip_id == trip_id].stop_id)
        current_segment = improve_dataset_unit(item, stop_sequence)
        if current_segment is None:
            continue
        # add the other columns
        current_segment['weather'] = item.iloc[0].weather
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


def fix_bug_segment(segment_df):
    """
    Fix the bugs for the segment after improvement:
    1. some of the rows are repeated, we need to remove the duplicated one
    2. Som travel duration is too large, we need to remove the travel duration which is larger than 10 minutes

    Algorithm:
    split the segment with groupby(service_date, trip_id)
    result_list = []
    for name, item in splitted segment:
        do improve for the item
        append the result into result_list
    concatenate the result_list
    :param segment_df: dataframe for segment.csv
    :return: the dataframe after fixing the bugs in the segment.csv
    """
    grouped_list = list(segment_df.groupby(['service_date', 'trip_id']))
    result_list = []
    print 'length of the grouped list: ', len(grouped_list)
    for i in xrange(len(grouped_list)):
        if i % 1000 == 0:
            print i
        name, item = grouped_list[i]
        if len(item) < 3:
            continue
        current_segment = fix_bug_single_segment(item)
        result_list.append(current_segment)
    result = pd.concat(result_list, ignore_index=True)
    columns = []
    for item in result.columns:
        if item.startswith('Unnamed'):
            columns.append(item)
    result.drop(columns, axis=1, inplace=True)
    # check the travel duration
    return result


def fix_bug_single_segment(segment_df):
    """
    Fix the bug for a segment dataframe with specific service date and the trip id

    Algorithm:
    Define the dataframe for the result
    For i in range(1, len(segment_df):
        prev_record = segment_df.iloc[i - 1]
        next_record = segment_df.iloc[i]
        if prev_record.segment_start = next_record.segment_start and prev_record.segment_end == next_record.segment_end:
            This is a duplicated record, continue to next row
        if the prev_record.travel_duration > 600 (10 minutes), continue to next row
        save prev_record into result

    :param segment_df: dataframe of the single segment data
    :return: dataframe for the segment after fixing the bug
    """
    result = pd.DataFrame(columns=segment_df.columns)
    for i in xrange(1, len(segment_df)):
        prev_record = segment_df.iloc[i - 1]
        next_record = segment_df.iloc[i]
        # check whether the row is duplicated
        if prev_record.segment_start == next_record.segment_start and prev_record.segment_end == next_record.segment_end:
            continue
        result.loc[len(result)] = prev_record
    if len(result) == 0:
        result.loc[len(result)] = segment_df.iloc[-1]
    elif result.iloc[-1].segment_start != segment_df.iloc[-1].segment_start and result.iloc[-1].segment_end != \
            segment_df.iloc[-1].segment_end:
        result.loc[len(result)] = segment_df.iloc[-1]
    mean_travel_duration = result['travel_duration'].mean()
    # some of the ditance of the segment pair is large, so it is possible that some of the segment pair is larger than 600 hundred seconds
    # result['travel_duration'] = result['travel_duration'].apply(lambda x: mean_travel_duration if x > 600 else x)
    return result


#################################################################################################################
#                                    api data section                                                              #
#################################################################################################################


"""
Generate the api data from the GTFS data and the historical data
"""


def generate_api_data(date_list, time_list, route_list, stop_num, route_stop_dist, full_history):
    """
    Generate the api data for the test_route_set and given time list
    :param time_list: the time list for testing, ['12:00:00', '12:05:00', ...]
    :param route_list: the list for the test route id
    :param stop_num: the number of the stop id for test
    :param route_stop_dist: the dataframe for the route_stop_dist.csv file
    :return: the dataframe for the api data
    trip_id    vehicle_id    route_id    stop_id    time_of_day    date    dist_along_route

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
    """
    trips = pd.read_csv(path + 'data/GTFS/gtfs/trips.txt')
    trip_route_dict = {}
    route_stop_dict = {}
    for route in route_list:
        print route
        stop_sequence = list(route_stop_dist[route_stop_dist.route_id == route].stop_id)
        if len(stop_sequence) < 5:
            continue
        trip_set = set(trips[trips.route_id == route].trip_id)
        current_dict = dict.fromkeys(trip_set, route)
        trip_route_dict.update(current_dict)
        stop_set = set()
        for i in range(stop_num):
            stop_set.add(stop_sequence[random.randint(2, len(stop_sequence) - 2)])
        route_stop_dict[route] = stop_set
    history = full_history[(full_history.trip_id.isin(trip_route_dict.keys())) & (full_history.service_date.isin(date_list))]
    history_grouped = history.groupby(['service_date', 'trip_id'])
    result = pd.DataFrame(
        columns=['trip_id', 'vehicle_id', 'route_id', 'stop_id', 'time_of_day', 'date', 'dist_along_route'])
    print_dict = dict.fromkeys(date_list, True)
    for name, single_history in list(history_grouped):
        service_date, trip_id = name
        if service_date not in date_list:
            continue
        if print_dict[service_date]:
            print service_date
            print_dict[service_date] = False
        route_id = trip_route_dict[trip_id]
        stop_set = [str(int(item)) for item in route_stop_dict[route_id]]
        stop_sequence = [str(int(item)) for item in list(route_stop_dist[route_stop_dist.route_id == route_id].stop_id)]
        tmp_history = single_history[
            (single_history.next_stop_id.isin(stop_sequence)) & (single_history.dist_along_route > '0')]
        if len(tmp_history) < 3:
            continue
        else:
            single_history = pd.DataFrame(columns=tmp_history.columns)
            for i in range(1, len(tmp_history)):
                if float(tmp_history.iloc[i - 1].dist_along_route) < float(tmp_history.iloc[i].dist_along_route):
                    single_history.loc[len(single_history)] = tmp_history.iloc[i - 1]
            if len(single_history) < 3:
                continue
            if tmp_history.iloc[-1].dist_along_route >= single_history.iloc[-1].dist_along_route:
                single_history.loc[len(single_history)] = tmp_history.iloc[-1]
        for target_stop in stop_set:
            target_index = stop_sequence.index(target_stop)
            for current_time in time_list:
                # If the bus has not started from the initial stop yet, continue to next time point in the time list
                if single_history.iloc[0].timestamp[11:19] > current_time:
                    continue
                # check whether the bus has passed the target stop, if yes, break and continue to the next target_stop
                index = 1
                while index < len(single_history) and single_history.iloc[index].timestamp[11:19] <= current_time:
                    index += 1
                if index == len(single_history):
                    break
                index -= 1
                tmp_stop = str(single_history.iloc[index].next_stop_id)
                tmp_index = stop_sequence.index(tmp_stop)
                if tmp_index > target_index:
                    break
                # If the bus does not pass the target stop, save the remained stops into the stop sequence and calculate the result
                current_list = generate_single_api(current_time, route_stop_dist, route_id, single_history[index:],
                                                   target_stop, target_index)
                if current_list is not None:
                    result.loc[len(result)] = current_list
    return result


def generate_single_api(current_time, route_stop_dist, route_id, single_history, stop_id, end_index):
    """
    Calculate the single record for the api data
    :param current_time: The current time for generating the api data
    :param single_history: The historical data for the specific date and the trip id
    :param stop_id: The target stop id
    :return: the list for the result
    [trip_id    vehicle_id    route_id    time_of_day    date    dist_along_route]

    Algorithm for calculate the single record:
    According to the time point, find the closest time duration (prev, next)
    Calculate the dist_along_route for the bus at current timepoint:
        calculate the space distance between the time duration (prev, next)
        calculate the time distance of two parts: (prev, current), (prev, next)
        use the ratio of the time distance to multiply with the space distance to obtain the dist_along_route for current
    According to the dista_along_route and the stop sequence confirm the remained stops including the target stop
    Count the number of the remained stops
    """
    single_trip = single_history.iloc[0].trip_id
    prev_record = single_history.iloc[0]
    next_record = single_history.iloc[1]
    # If the time duration between the prev and the next time point is larger than 5 minutes, ignore it for precision
    if calculate_time_span(prev_record['timestamp'], next_record['timestamp']) > 300:
        return None
    # calculate the dist_along_route for current
    prev_distance = float(prev_record['dist_from_stop']) - float(prev_record['dist_along_route'])
    next_distance = float(next_record['dist_from_stop']) - float(next_record['dist_along_route'])
    prev_timestamp = datetime.strptime(prev_record['timestamp'], '%Y-%m-%dT%H:%M:%SZ')
    next_timestamp = datetime.strptime(next_record['timestamp'], '%Y-%m-%dT%H:%M:%SZ')
    # determine the current time
    if prev_record['timestamp'][11:19] <= current_time <= next_record['timestamp'][11:19]:
        time_of_day = prev_record['timestamp'][:11] + current_time + 'Z'
    else:
        if current_time > next_record['timestamp'][11:19]:
            time_of_day = prev_record['timestamp'][11:19] + current_time + 'Z'
        else:
            time_of_day = next_record['timestamp'][11:19] + current_time + 'Z'
    time_of_day = datetime.strptime(time_of_day, '%Y-%m-%dT%H:%M:%SZ')
    dist_along_route = calculate_arrival_distance(time_of_day, prev_distance, next_distance, prev_timestamp,
                                                  next_timestamp)
    # Generate the return list
    # trip_id    vehicle_id    route_id    stop_id    time_of_day    date    dist_along_route
    result = [single_trip, prev_record['vehicle_id'], route_id, stop_id, str(time_of_day), prev_record['service_date'],
              dist_along_route]
    return result


#################################################################################################################
#                                    debug section                                                              #
#################################################################################################################


#################################################################################################################
#                                    main function                                                              #
#################################################################################################################


if __name__ == '__main__':
    file_list = os.listdir('./')
    # download weather information
    if 'weather.csv' not in file_list:
        print "download weather.csv file"
        weather_df = download_weather('20160101', '20160131')
        weather_df.to_csv('weather.csv')
        print "complete downloading weather information"
        # export the route dist data
    if 'route_stop_dist.csv' not in file_list:
        print "export route_stop_dist.csv file"
        trips, stop_times, history = read_data()
        route_stop_dist = calculate_stop_distance(trips, stop_times, history)
        route_stop_dist.to_csv('route_stop_dist.csv')
        print "complete exporting the route_stop_dist.csv file"
        # export the train_history.csv file
    if 'train_history.csv' not in file_list:
        print "export train_history.csv file"
        selected_trips = select_trip_list()
        full_history = filter_history_data(20160104, 20160123, selected_trips)
        full_history.to_csv('train_history.csv')
        print "complete exporting train_history.csv file"
    # # export the segment data
    # if 'original_segment.csv' not in file_list:
    #     print "export original_segment.csv file"
    #     weather_df = pd.read_csv('weather.csv')
    #     full_history = pd.read_csv('train_history.csv')
    #     stop_times = pd.read_csv(path + 'data/GTFS/gtfs/stop_times.txt')
    #     segment_df = generate_original_segment(full_history, weather_df, stop_times)
    #     segment_df.to_csv('original_segment.csv')
    #     print "complete exporting the original_segement.csv file"
    # if 'segment.csv' not in file_list:
    #     print "export segment.csv file"
    #     segment_df = improve_dataset()
    #     segment_df.to_csv('segment.csv')
    #     print "complete exporting the segment.csv file"
    # if "final_segment.csv" not in file_list:
    #     print "export final segment.csv file"
    #     segment_df = pd.read_csv('segment.csv')
    #     final_segment = fix_bug_segment(segment_df)
    #     final_segment.to_csv('final_segment.csv')
    #     print "complete exporting the final_segment.csv file"
    # # export the api data
    # if "test_history.csv" not in file_list:
    #     print "export test_history.csv file"
    #     selected_trips = select_trip_list()
    #     full_history = filter_history_data(201601025, 20160130, selected_trips)
    #     full_history.to_csv('test_history.csv')
    #     print "complete exporting test_history.csv file"
    # if 'api_data.csv' not in file_list:
    #     print "export api_data.csv file"
    #     date_list = range(20160125, 20160130)
    #     route_stop_dist = pd.read_csv('route_stop_dist.csv')
    #     stop_num = 4
    #     route_list = list(set(route_stop_dist.route_id))
    #     full_history = pd.read_csv('test_history.csv')
    #     api_data_list = []
    #     time_list = ['12:00:00', '12:05:00', '12:10:00', '12:15:00', '12:20:00', '12:25:00', '12:30:00']
    #     current_api_data = generate_api_data(date_list, time_list, route_list, stop_num, route_stop_dist, full_history)
    #     api_data_list.append(current_api_data)
    #     time_list = ['18:00:00', '18:05:00', '18:10:00', '18:15:00', '18:20:00', '18:25:00', '18:30:00']
    #     current_api_data = generate_api_data(date_list, time_list, route_list, stop_num, route_stop_dist, full_history)
    #     api_data_list.append(current_api_data)
    #     api_data = pd.concat(api_data_list)
    #     api_data.to_csv('api_data.csv')
    #     print "complete exporting the api_data.csv file"
    # print "complete data collection"
