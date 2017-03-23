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

    print "complete downloading weather information"

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
    selected_trips = set(list(result_trips.trip_id))
    result_stop_times = stop_times[stop_times.trip_id.isin(selected_trips)]
    # Obtain the filtered history dataframe
    file_list = os.listdir(path + 'data/history/')
    file_list.sort()
    history_list = []
    for single_file in file_list:
        if not single_file.endswith('.csv'):
            continue
        else:
            current_history = pd.read_csv(path + 'data/history/' + single_file)
            tmp_history = current_history[current_history.trip_id.isin(selected_trips)]
            if len(tmp_history) == 0:
                continue
            else:
                print "historical file name: ", single_file
                history_list.append(tmp_history)
    result_history = pd.concat(history_list)
    print "complete reading data"
    return result_trips, result_stop_times, result_history


def calculate_stop_distance(trips, stop_times, history, direction_id = 0):
    """
    Calculate the distance of each stop with its inital stop. Notice that the dist_along_route is the distance between the next_stop and the initial stop
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
        selected_trips = set(trips[trips.route_id == single_route].trip_id)
        stop_sequence = list(stop_times[stop_times.trip_id == list(selected_trips)[0]].stop_id)
        result.loc[len(result)] = [single_route, int(direction_id), int(stop_sequence[0]), 0.0]
        selected_history = history[history.trip_id.isin(selected_trips)]
        for i in range(1, len(stop_sequence)):
            stop_id = stop_sequence[i]
            current_history = selected_history[selected_history.next_stop_id == stop_id]
            if float(stop_id) == float(result.iloc[-1].stop_id):
                continue
            elif len(current_history) == 0:
                dist_along_route = -1.0
            else:
                current_dist = []
                for i in range(len(current_history)):
                    current_dist.append(current_history.iloc[i].dist_along_route)
                dist_along_route = sum(current_dist) / float(len(current_dist))
            result.loc[len(result)] = [single_route, int(direction_id), int(stop_id), dist_along_route]
    result.to_csv('original_route_stop_dist.csv')
    # Since some of the stops might not record, it is necessary to check the dataframe again.
    count = 1
    prev = 0
    for i in range(1, len(result) - 1):
        if result.iloc[i].dist_along_route == -1:
            if result.iloc[i - 1].dist_along_route != -1:
                prev = result.iloc[i - 1].dist_along_route
            count += 1
        else:
            if count != 1:
                distance = (float(result.iloc[i].dist_along_route) - float(prev)) / float(count)
                while count > 1:
                    result.iloc[i - count + 1, result.columns.get_loc('dist_along_route')] = result.iloc[i - count].dist_along_route + float(distance)
                    count -= 1
            else:
                continue
    return result




#################################################################################################################
#                                route_stop_dist.csv                                                            #
#################################################################################################################


if __name__ == '__main__':
    file_list = os.listdir('./')
    # download weather information
    if 'weather.csv' not in file_list:
        print "download weather.csv file"
        download_weather('20160101', '20160131')
    # export the route dist data
    if 'route_stop_dist.csv' not in file_list:
        print "export route_stop_dist.csv file"
        trips, stop_times, history = read_data()
        route_stop_dist = calculate_stop_distance(trips, stop_times, history)
