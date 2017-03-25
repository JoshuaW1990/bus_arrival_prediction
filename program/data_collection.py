"""
Data collection process file. Try to improve the efficiency.
Exported files: `weather.csv`, `segment.csv`, `api_data.csv` and `route_stop_dist.csv`
Input files: gtfs file, historical data
"""

# import modules
import pandas as pd
import numpy as np
import os
import requests
import csv
from datetime import datetime, timedelta, date
from dateutil.rrule import rrule, DAILY

# set the path
path = '../'

print "working path"
print os.getcwd()
print os.listdir(path)


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
        result = [gooddate, summary['date']['year'], summary['date']['mon'], summary['date']['mday'], summary['fog'],
                  summary['rain'], summary['snow']]
    return result


def download_weather(date_start, date_end):
    """
    download the weather information for a date range
    :param date_start: start date, string, ex: '20160101'
    :param date_end: similar to date_start
    :return: list of the table record
    """

    a = datetime.strptime(date_start, '%Y%m%d')
    b = datetime.strptime(date_end, '%Y%m%d')

    result = [['date', 'year', 'month', 'day', 'fog', 'rain', 'snow']]
    for dt in rrule(DAILY, dtstart=a, until=b):
        current_data = get_precip(dt.strftime("%Y%m%d"))
        if current_data is None:
            continue
        else:
            result.append(current_data)

    # export to the csv file
    with open('weather.csv', 'wb') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        for item in result:
            spamwriter.writerow(item)

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
    result_history = pd.concat(history_list)
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
#                                        segment.csv                                                            #
#################################################################################################################
"""
Generate the orgininal segment data including the travel duration. Improve the segment data by adding back the skipped
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
    file_list = os.listdir(path + 'data/history/')
    history_list = []
    print "filtering historical data"
    for filename in file_list:
        if not filename.endswith('.csv'):
            continue
        if filename[9:17] < str(date_start) or filename[9:17] > str(date_end):
            continue
        print filename
        ptr_history = pd.read_csv(path + 'data/history/' + filename)
        tmp_history = ptr_history[ptr_history.trip_id.isin(selected_trips_var)]
        history_list.append(tmp_history)
    result = pd.concat(history_list)
    return result[result.next_stop_id != '\N']


def extract_time(time):
    """
    example of time(str): '2017-01-16T15:09:28Z'
    """
    result = datetime.strptime(time[11: 19], '%H:%M:%S')
    return result


def calculate_time_span(time1, time2):
    timespan = extract_time(time2) - extract_time(time1)
    return timespan.total_seconds()


def add_weather_info(date_var):
    """
    add the weather information from the file: weather.csv
    The weather are expressed as below:
    0: sunny
    1: rainy
    2: snowy
    :param date_var: the date for querying the weather
    :return: return the weather value today.
    """
    filename = 'weather.csv'
    weather = pd.read_csv(filename)
    ptr_weather = weather[weather.date == date_var]
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

    # in the historical data, sometimes there is a bug which the trip id was assigned to the wrong vehicle
    # we need to skipp this case by checking the vehicle id
    vehicle_id = history.iloc[0].vehicle_id
    count = 1
    for i in xrange(1, len(history)):
        if history.iloc[i].vehicle_id == vehicle_id:
            count += 1
        else:
            if count == 0:
                vehicle_id = history.iloc[i].vehicle_id
                count = 1
            else:
                count -= 1

    # filter the historical data
    history = history[history.vehicle_id == vehicle_id]
    # When filtering the last one, we need to notice that sometimes the bus has been stopped but the GPS device is still recording the location of the bus. Thus we need to check the last stop specificaly.
    trip_id = history.iloc[0].trip_id
    date_var = history.iloc[0].timestamp[:10].translate(None, '-')
    date_time = datetime.strptime(date_var, '%Y%m%d')
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
        time_span = calculate_time_span(time1, time2)
        estimated_travel_time = time_span * ratio
        estimated_travel_time = timedelta(0, estimated_travel_time)
        estimated_arrival_time = extract_time(time1) + estimated_travel_time
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
                                   date_time.weekday(), date_var, add_weather_info(int(date_var)), trip_id,
                                   travel_duration.total_seconds()]
    # print "complete obtaining the segement data for the date: ", date
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
    for date_var in date_set:
        tmp_history = history[history.service_date == date_var]
        segment_pair_df = calculate_travel_duration_single_date(tmp_history)
        if segment_pair_df is None:
            continue
        segment_df_list.append(segment_pair_df)
    if segment_df_list == []:
        result = None
    else:
        result = pd.concat(segment_df_list)
    return result


def generate_original_dataframe(selected_trips, date_start, date_end, full_history=None):
    """
    This function will read the list of the selected trips and read the them one by one and concatenate all their dataframe together.
    """
    if full_history is None:
        full_history = filter_history_data(date_start, date_end, selected_trips)
    result_list = []
    for i, single_trip in enumerate(selected_trips):
        if i % 100 == 0:
            print "index of the current trip id in the selected trips: ", i
        tmp_segment_df = calculate_travel_duration(single_trip, full_history)
        if tmp_segment_df is None:
            continue
        result_list.append(tmp_segment_df)
    if result_list == []:
        result = None
    else:
        result = pd.concat(result_list)
    return result


def improve_dataset_unit(single_trip, date_var, stop_sequence, segment_df):
    """
    This funciton is used to improve the dataset for a specific trip_id at a spacific date.
    """
    df = pd.DataFrame(
        columns=['segment_start', 'segment_end', 'segment_pair', 'time_of_day', 'day_of_week', 'date', 'weather',
                 'trip_id', 'travel_duration'])
    current_segmen_pair = segment_df[(segment_df.trip_id == single_trip) & (segment_df.date == date_var)]
    for i in xrange(1, len(current_segmen_pair)):
        segment_start = int(current_segmen_pair.iloc[i - 1].segment_start)
        segment_end = int(current_segmen_pair.iloc[i].segment_start)
        start_idx = stop_sequence.index(segment_start)
        end_idx = stop_sequence.index(segment_end)
        if end_idx - start_idx == 1:
            df.loc[len(df)] = current_segmen_pair.iloc[i - 1]
        else:
            skipped_stops = stop_sequence[start_idx + 1:end_idx]
            number_travel_duration = len(skipped_stops) + 1
            arrival_time1 = datetime.strptime(current_segmen_pair.iloc[i - 1].time_of_day, '%H:%M:%S')
            arrival_time2 = datetime.strptime(current_segmen_pair.iloc[i].time_of_day, '%H:%M:%S')
            timespan = arrival_time2 - arrival_time1
            total_duration = timespan.total_seconds()
            average_duration = total_duration / float(number_travel_duration)
            estimated_travel_time = timedelta(0, average_duration)
            tmp_total_stops = [segment_start] + skipped_stops + [segment_end]
            for j in xrange(len(tmp_total_stops) - 1):
                segment_start = tmp_total_stops[j]
                segment_end = tmp_total_stops[j + 1]
                segment_pair = (segment_start, segment_end)
                previous_arrival_time = current_segmen_pair.iloc[i - 1].time_of_day
                estimated_arrival_time = datetime.strptime(previous_arrival_time, '%H:%M:%S')
                for count in range(j):
                    estimated_arrival_time += estimated_travel_time
                time_of_day = str(estimated_arrival_time)[11:19]
                day_of_week = current_segmen_pair.iloc[0].day_of_week
                weather = current_segmen_pair.iloc[0].weather
                trip_id = single_trip
                travel_duration = average_duration
                df.loc[len(df)] = [segment_start, segment_end, segment_pair, time_of_day, day_of_week, date_var,
                                   weather,
                                   trip_id, travel_duration]
    return df


def improve_dataset():
    """
    algorithm:
    for each specific trip_id:
        obtain the date_list
        obtain the stop_sequence
        for each date in date_list:
            build the dataframe
            obtain the current_segment_pair for the specific trip_id and date
            obtain the segment_start sequence
            for each segment_start in the segment_start sequence:
                find the corresponding index in the stop_sequence
                find the index of the segment_end in the corresponding segment_pair
                if the indices of these two are i and i + 1:
                    add the segment_pair into the new dataframe as the result
                else:
                    use the index to find all the skipped stops from the stop_sequence
                    calculate the number of skipped travel duration within this segment
                    use the average value as the travel duration and add the stop arrival time for each skipped stops
                    add the segment_pair into the new dataframe as the result

    """
    segment_df = pd.read_csv('original_segment.csv')
    stop_times = pd.read_csv(path + 'data/GTFS/gtfs/stop_times.txt')

    trips = set(segment_df.trip_id)
    print "length of the trips: ", len(trips)
    df_list = []
    for i, single_trip in enumerate(trips):
        if i % 50 == 0:
            print "index = ", i, single_trip
        date_list = list(set(segment_df[segment_df.trip_id == single_trip].date))
        stop_sequence = list(stop_times[stop_times.trip_id == single_trip].stop_id)
        for date_var in date_list:
            df = improve_dataset_unit(single_trip, date_var, stop_sequence, segment_df)
            df_list.append(df)
    result = pd.concat(df_list)
    return result


#################################################################################################################
#                                    API data                                                                   #
#################################################################################################################
"""
Rearrange the order in the loop to save the time, utilize the group by
Algorithm:
Prepare the time list
Read the full historical data for testing
Determine the test routes (2 ~ 4), and the corresponding stop sequence
Generate the random stop id from the test routes
Generate the set of trip id for the testing routes
Filtering the historical data with trip id, NAN
generate date list
for date in date list:
    generate single historical data for the date
    for stop in stop set:
        Filtering the trip set according to the stop
        Filtering the single historical data according to the trip set
        for single_trip in trip set:
            obtain the single historical data for that trip
            for each time point in the time list:
                check whether the bus has passed the stop at the time point
                if yes, continue to next trip
                otherwise, save the record into result
"""



"""
algorithm for calculate the single record:

"""



#################################################################################################################
#                                    debug section                                                              #
#################################################################################################################
segment_df = improve_dataset()
segment_df.to_csv('segment.csv')


#################################################################################################################
#                                    main function                                                              #
#################################################################################################################


# if __name__ == '__main__':
#     file_list = os.listdir('./')
#     # download weather information
#     if 'weather.csv' not in file_list:
#         print "download weather.csv file"
#         download_weather('20160101', '20160131')
#         print "complete downloading weather information"
#     # export the route dist data
#     if 'route_stop_dist.csv' not in file_list:
#         print "export route_stop_dist.csv file"
#         trips, stop_times, history = read_data()
#         route_stop_dist = calculate_stop_distance(trips, stop_times, history)
#         route_stop_dist.to_csv('route_stop_dist.csv')
#         print "complete exporting the route_stop_dist.csv file"
#     # export the segment data
#     if 'segment.csv' not in file_list:
#         print "export segment.csv file"
#
#         print "complete exporting the segment.csv file"
#     print "complete data collection"
