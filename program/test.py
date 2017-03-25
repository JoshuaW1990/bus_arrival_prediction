"""
Test new algorithms for different section
"""

import pandas as pd
import os
from datetime import datetime, timedelta

path = '../'


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


def extractTime(time):
    """
    example of time(str): '2017-01-16T15:09:28Z'
    """
    result = datetime.strptime(time[11: 19], '%H:%M:%S')
    return result


def calculateTimeSpan(time1, time2):
    timespan = extractTime(time2) - extractTime(time1)
    return timespan.total_seconds()


def add_weather_info(date):
    """
    add the weather information from the file: weather.csv
    The weather are expressed as below:
    0: sunny
    1: rainy
    2: snowy
    :param date: the date for querying the weather
    :return: return the weather value today.
    """
    filename = 'weather.csv'
    weather = pd.read_csv(filename)
    ptr_weather = weather[weather.date == date]
    if ptr_weather.iloc[0].snow == 1:
        weather_today = 2
    elif ptr_weather.iloc[0].rain == 1:
        weather_today = 1
    else:
        weather_today = 0
    return weather_today


def calculate_travel_duration_single_date(history):
    """
    Calculate the travel duration of every segments for a specific trip at a specific date
    The format of the return value(dataframe):
     segment_start  segment_end  segment_pair   time_of_day  travel_duration
       str             str         (str, str)      str         float(seconds)
    """

    # Some of the stops might not be recored in the historical data, and it is necessary to be considered to avoid the mismatch of the schedule data and the historical data.
    # One of the method is to build a simple filter for the historical data at first. This filter will remove the unecessary records like repeated next_stop_id record. Then compared the result with the scheduled data.

    # filter the historical data
    # When filtering the last one, we need to notice that sometimes the bus has been stopped but the GPS device is still recording the location of the bus. Thus we need to check the last stop specificaly.
    trip_id = history.iloc[0].trip_id
    date = history.iloc[0].timestamp[:10].translate(None, '-')
    date_time = datetime.strptime(date, '%Y%m%d')
    filtered_history = pd.DataFrame(columns=history.columns)
    for i in xrange(1, len(history)):
        if history.iloc[i - 1].next_stop_id == history.iloc[i].next_stop_id:
            continue
        else:
            filtered_history.loc[len(filtered_history)] = list(history.iloc[i])
    if len(filtered_history) == 0:
        return None
    last_stop_id = filtered_history.iloc[-1].next_stop_id
    tmp_history = history[history.next_stop_id == last_stop_id]
    filtered_history.iloc[-1] = tmp_history.iloc[0]

    # analyze the result with the filtered historical data
    # Problems:
    # 1. Some of the stops might be skipped in the historical data, thus the historical data should be taken as the standard for the segment pair
    # 2. Some of the distance ratio is abnormal: ratio < 0, ratio >= 1, we should skip them. When the ratio == 0, it means it is actually stay at the stop
    # 3. The actual runtime in the historical data is totally different with the scheduled data, we should mainly focused on the historical data.
    # 4. One thing which is easy to be confused is that: in the historical data, when calcuating the arrival time, we don't care about the the second stop in a distance pair. All we need to remember is that the next stop is acutally the first one in the pair.

    # define a tuple list to store the stops and the corresponding arrival time
    stop_arrival_time = []
    for i in xrange(len(filtered_history) - 1):
        if filtered_history.iloc[i + 1].dist_along_route == '\N':
            continue
        next_stop = filtered_history.iloc[i].next_stop_id
        distance_location = float(filtered_history.iloc[i + 1].dist_along_route) - float(
            filtered_history.iloc[i].dist_along_route)
        distance_station = float(filtered_history.iloc[i].dist_from_stop)
        if distance_station >= distance_location or distance_location < 0:
            continue
        ratio = distance_station / distance_location
        time1 = filtered_history.iloc[i].timestamp
        time2 = filtered_history.iloc[i + 1].timestamp
        time_span = calculateTimeSpan(time1, time2)
        estimated_travel_time = time_span * ratio
        estimated_travel_time = timedelta(0, estimated_travel_time)
        estimated_arrival_time = extractTime(time1) + estimated_travel_time
        stop_arrival_time.append((next_stop, estimated_arrival_time))

    # Calculate the travel_duration according to the stop_arrival_time list
    # form a pair of segments and the corresponding travel_duration
    # the format of the dataframe:
    # segment_start  segment_end  time_of_day  travel_duration
    #   str             str         str           float(seconds)
    result = pd.DataFrame(
        columns=['segment_start', 'segment_end', 'segment_pair', 'time_of_day', 'day_of_week', 'date', 'weather',
                 'trip_id', 'travel_duration'])
    for i in xrange(len(stop_arrival_time) - 1):
        segment_start = stop_arrival_time[i][0]
        segment_end = stop_arrival_time[i + 1][0]
        travel_duration = stop_arrival_time[i + 1][1] - stop_arrival_time[i][1]
        time_of_day = stop_arrival_time[i][1]
        result.loc[len(result)] = [int(segment_start), int(segment_end),
                                   (int(segment_start), int(segment_end)), str(time_of_day)[11:19],
                                   date_time.weekday(), date, add_weather_info(int(date)), trip_id,
                                   travel_duration.total_seconds()]
    return result


def calculate_travel_duration(single_trip, full_history):
    """
    Calculate the travel duration between a specific segment pair for a specific trip
    :param single_trip: trip id for a specific trip
    :param full_history: historical data of several dates including this trip
    :return: dataframe for the segment and the travel duration, date, trip_id, etc. Format is below:
    segment_start    segment_end    segment_pair   time_of_day    day_of_week    date    weather    trip_id    travel_duration
    """
    history = full_history[full_history.trip_id == single_trip]
    print "trip id is ", single_trip
    print "historical data length: ", len(history)
    if len(history) == 0:
        return None
    date_set = set(list(history.service_date))
    segment_df_list = []
    # weather information and the day of week should be filled in each loop of the date
    for date in date_set:
        tmp_history = history[history.service_date == date]
        segment_pair_df = calculate_travel_duration_single_date(tmp_history)
        if segment_pair_df is None:
            continue
        segment_df_list.append(segment_pair_df)
    result = pd.concat(segment_df_list)
    return result


def generate_original_dataframe(selected_trips, full_history):
    """
    This function will read the list of the selected trips and read the them one by one and concatenate all their dataframe together.
    """
    result_list = []
    for i, single_trip in enumerate(selected_trips):
        if i % 10 == 0:
            print "index of the current trip id in the selected trips: ", i
        tmp_segment_df = calculate_travel_duration(single_trip, full_history)
        if tmp_segment_df is None:
            continue
        result_list.append(tmp_segment_df)
    result = pd.concat(result_list)
    return result


selected_trips = select_trip_list()
print "length of the selected trips: ", len(selected_trips)
# full_history = filter_history_data(20160104, 20160123, selected_trips)
full_history = pd.read_csv('full_history.csv')
segment_df = generate_original_dataframe(selected_trips[150:200] + selected_trips[300:350], full_history)
