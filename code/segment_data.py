# This file is used to export the segment data for future analysis

# import module
import pandas as pd
import numpy as np
import os
import calendar as cal
from datetime import datetime, timedelta


path = '../data/GTFS/gtfs/'
def select_trip_list(path, num_route):
	# Read the GTFS data
	# data source: MTA, state island, Jan, 4, 2016
	trips = pd.read_csv(path + 'trips.txt')

	# select a specific route and the corresponding trips
	select_routes = trips.iloc[:num_route].route_id
	select_trips = []
	for route in select_routes:
		select_trips += list(trips[(trips.route_id == route) & (trips.direction_id == 0)].trip_id)
		print "number of the selected trips:"
		print len(select_trips)
	return select_trips

def filter_history_data(date_start, date_end, selected_trips):
	"""
	format for the input:
	date_start, date_end:    int, yyyymmdd, ex: 20160109
	"""
	# List the historical file
	path = '/Users/junwang/Documents/Github/bus_arrival_prediction/data/history/'
	dirs = os.listdir(path)
	file_list = []
	for dir in dirs:
	    if dir.endswith('.csv'):
	        file_list.append(dir)
	history_list = []
	for file in file_list:
		if file[9:17] < str(date_start) or file[9:17] > str(date_end):
			continue
		print file
		ptr_history = pd.read_csv('../data/history/' + file)
		tmp_history = ptr_history[ptr_history.trip_id.isin(selected_trips)]
		history_list.append(tmp_history)
	full_history = pd.concat(history_list)
	return full_history

def generate_segment_table(selected_trips):
	"""
	return table format:
	segment_start, segment_end, scheduled_time_of_day, trip_id
	"""
	stop_times = pd.read_csv(path + 'stop_times.txt')
	df = pd.DataFrame(columns = ['segment_start', 'segment_end', 'time_of_day', 'trip_id', 'travel_duration'])
	tmp = []
	for trip in selected_trips:
		selected_stop_times = stop_times[stop_times.trip_id == trip]
		for i in xrange(len(selected_stop_times) - 1):
			segment_start = selected_stop_times.iloc[i].stop_id
			segment_end = selected_stop_times.iloc[i + 1].stop_id
			scheduled_time_of_day = selected_stop_times.iloc[i].departure_time
			trip_id = selected_stop_times.iloc[i].trip_id
			df.loc[len(df)] = [segment_start, segment_end, 'time_of_day', trip_id, 0.0]
	return stop_times, df


def extractTime(time):
    """
    example of time(str): '2017-01-16T15:09:28Z'
    """
    result = datetime.strptime(time[11: 19], '%H:%M:%S')
    return result

def calculateTimeSpan(time1, time2):
	timespan = extractTime(time2) - extractTime(time1)
	return timespan.total_seconds()


def calculate_travel_duration_single_date(history):
	"""
	Calculate the travel duration of every segments for a specific trip at a specific date
	The format of the return value(dataframe):
	 segment_start  segment_end  time_of_day  travel_duration
	   str             str         str         float(seconds)
	"""

	# Some of the stops might not be recored in the historical data, and it is necessary to be considered to avoid the mismatch of the schedule data and the historical data.
	# One of the method is to build a simple filter for the historical data at first. This filter will remove the unecessary records like repeated next_stop_id record. Then compared the result with the scheduled data.

	# filter the historical data
	# When filtering the last one, we need to notice that sometimes the bus has been stopped but the GPS device is still recording the location of the bus. Thus we need to check the last stop specificaly.
	trip_id = history.iloc[0].trip_id
	filtered_history = pd.DataFrame(columns = history.columns)
	for i in xrange(1, len(history)):
		if history.iloc[i - 1].next_stop_id == history.iloc[i].next_stop_id:
			continue
		else:
			filtered_history.loc[len(filtered_history)] = list(history.iloc[i])
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
		next_stop = filtered_history.iloc[i].next_stop_id
		distance_location = float(filtered_history.iloc[i + 1].dist_along_route) - float(filtered_history.iloc[i].dist_along_route)
		distance_station = float(filtered_history.iloc[i].dist_from_stop)
		ratio = distance_station / distance_location
		if ratio >= 1 or ratio < 0:
			continue
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
	segment_pair_df = pd.DataFrame(columns=['segment_start', 'segment_end', 'segment_pair', 'time_of_day', 'trip_id', 'travel_duration'])
	for i in xrange(len(stop_arrival_time) - 1):
		segment_start = stop_arrival_time[i][0]
		segment_end = stop_arrival_time[i + 1][0]
		travel_duration = stop_arrival_time[i + 1][1] - stop_arrival_time[i][1]
		time_of_day = stop_arrival_time[i][1]
		segment_pair_df.loc[len(segment_pair_df)] = [segment_start, segment_end,  (segment_start, segment_end), str(time_of_day)[11:19], trip_id, travel_duration.total_seconds()]

	return segment_pair_df


# selected_trips = select_trip_list(path, 1)
# full_history = filter_history_data(20160101, 20160106, selected_trips)
# stop_times, segment_df = generate_segment_table(selected_trips)
# # temporary use this
# single_trip = full_history.iloc[0].trip_id

# history = full_history[full_history.trip_id == single_trip]
# date_set = list(set(list(history.service_date)))
# date = 20160106
# history = history[history.service_date == date]
# segment_pair_df = calculate_travel_duration_single_date(history)




def calculate_travel_duration(single_trip, full_history, segment_df):
	"""
	Calculate the travel duration between a specific segment pair for a specific trip
	Input:
	single_trip: trip id of a specfic trip
	stop_times: data of the stop_times.txt from GTFS
	history: historical data contains this single trip (might be composed by several day)
	Output:
	result: dataframe for the segment and the travel duration, date, trip_id
	format of the dataframe:
		segment_start  segment_end  date, trip_id, travel_duration
			str         strptime    int      str         float
	"""
	pass

# code for the function calculate_travel_duration
# data preparation
selected_trips = select_trip_list(path, 1)
full_history = filter_history_data(20160101, 20160106, selected_trips)
stop_times, segment_df = generate_segment_table(selected_trips)
# temporary use this
single_trip = full_history.iloc[0].trip_id

history = full_history[full_history.trip_id == single_trip]
date_set = set(list(history.service_date))
segment_df_list = []
# weather information and the day of week should be filled in each loop of the date
for date in date_set:
	print "date: ", date
	tmp_history = history[history.service_date == date]
	segment_pair_df = calculate_travel_duration_single_date(tmp_history)
	segment_df_list.append(segment_pair_df)
tmp_segment_df = pd.concat(segment_df_list)

	





def update_segment_table(selected_trips, full_history, stop_times, segment_df):
	"""
	Read the input and update the segment table: add the day_of_week, weather, and the travel_duration
	"""
	pass

# code for the function update_segment_table
# selected_trips = select_trip_list(path, 1)
# full_history = filter_history_data(20160101, 20160110, selected_trips)
# stop_times, segment_df = generate_segment_table(selected_trips)
# for single_trip in selected_trips:
# 	travel_duration_df = calculate_travel_duration(single_trip, stop_times, full_history)








# if __name__=="__main__":
# 	selected_trips = select_trip_list(path, 1)
# 	full_history = filter_history_data(20160101, 20160110, selected_trips)
# 	segment_df = generate_segment_table(selected_trips)








