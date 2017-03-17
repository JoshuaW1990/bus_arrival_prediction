"""
This file is used to export the segment data for future analysis
It will export the data.csv file which records the travel duration for all the segment pairs.
The format of the returned value: dataframe
segment_start    segment_end    segment_pair    time_of_day    day_of_week    date    weather    trip_id    travel_duration
  int             int            (int, int)      str            int            int      int        str        float(seconds)
"""

# import module
import pandas as pd
import os
from datetime import datetime, timedelta

#######################################################################################################################
# generate raw data
#######################################################################################################################




def select_trip_list(path, direction_id = 0, num_route = None):
	# Read the GTFS data
	# data source: MTA, state island, Jan, 4, 2016
	trips = pd.read_csv(path + 'trips.txt')
	# select a specific route and the corresponding trips
	route_list = list(trips.route_id)
	non_dup_route_list = sorted(list(set(route_list)))
	if num_route is None:
		select_routes = non_dup_route_list
	else:
		select_routes = non_dup_route_list[:num_route]
	select_trips = []
	for route in select_routes:
		select_trips += list(trips[(trips.route_id == route) & (trips.direction_id == direction_id)].trip_id)
	print "completing extract the select trips"
	return select_trips

def filter_history_data(date_start, date_end, selected_trips):
	"""

	:param date_start, date_end: int, yyyymmdd, ex: 20160109
	:param selected_trips: list of trip_id
	:return:
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
		ptr_history = pd.read_csv(path + file)
		tmp_history = ptr_history[ptr_history.trip_id.isin(selected_trips)]
		history_list.append(tmp_history)
	full_history = pd.concat(history_list)
	return full_history


def extractTime(time):
	"""
	:param time: example of time(str): '2017-01-16T15:09:28Z'
	:return: result: datetime format
	"""
	result = datetime.strptime(time[11: 19], '%H:%M:%S')
	return result

def calculateTimeSpan(time1, time2):
	"""
	Calculate the duration between two different time point
	:param time1, time2: datetime format
	:return: float(unit: second)
	"""
	timespan = extractTime(time2) - extractTime(time1)
	return timespan.total_seconds()

# add the weather information from the file: weather.csv
# The weather are expressed as below:
# 0: sunny
# 1: rainy
# 2: snowy
def add_weather_info(date):
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
	:param history: the historical data for a specific trip at a specific date
	:return: dataframe for the segment and the travel duration, date, trip_id, etc. The format is below:

	segment_start    segment_end    segment_pair   time_of_day    day_of_week    date    weather    trip_id    travel_duration
   	    int             int         (int, int)      str             int          int      int        str        float(seconds)
	"""

	# Some of the stops might not be recored in the historical data, and it is necessary to be considered to avoid the mismatch of the schedule data and the historical data.
	# One of the method is to build a simple filter for the historical data at first. This filter will remove the unecessary records like repeated next_stop_id record. Then compared the result with the scheduled data.

	# filter the historical data
	# When filtering the last one, we need to notice that sometimes the bus has been stopped but the GPS device is still recording the location of the bus. Thus we need to check the last stop specificaly.
	trip_id = history.iloc[0].trip_id
	date = int(history.iloc[0].timestamp[:10].translate(None, '-'))
	date_time = datetime.strptime(date, '%Y%m%d')
	filtered_history = pd.DataFrame(columns = history.columns)
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
	# if trip_id == 'CA_A6-Weekday-SDon-037800_X14_305':
	# 	print history.info()
	for i in xrange(len(filtered_history) - 1):
		next_stop = filtered_history.iloc[i].next_stop_id
		distance_location = float(filtered_history.iloc[i + 1].dist_along_route) - float(filtered_history.iloc[i].dist_along_route)
		distance_station = float(filtered_history.iloc[i].dist_from_stop)
		# if distance_location == 0:
		# 	print filtered_history.iloc[i - 2: i + 2]
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
	segment_pair_df = pd.DataFrame(columns=['segment_start', 'segment_end', 'segment_pair', 'time_of_day', 'day_of_week', 'date', 'weather', 'trip_id', 'travel_duration'])
	for i in xrange(len(stop_arrival_time) - 1):
		segment_start = stop_arrival_time[i][0]
		segment_end = stop_arrival_time[i + 1][0]
		travel_duration = stop_arrival_time[i + 1][1] - stop_arrival_time[i][1]
		time_of_day = stop_arrival_time[i][1]
		segment_pair_df.loc[len(segment_pair_df)] = [int(segment_start), int(segment_end),  (int(segment_start), int(segment_end)), str(time_of_day)[11:19], date_time.weekday(), date, add_weather_info(int(date)), trip_id, travel_duration.total_seconds()]

	return segment_pair_df


def calculate_travel_duration(single_trip, full_history):
	"""
	Calculate the travel duration between a specific segment pair for a specific trip
	:param single_trip: trip id of a specfic trip
	:param full_history: historical data
	:return: dataframe for the segment and the travel duration, date, trip_id, etc. The format is followed as the calculate_travel_duration_single_date function
	segment_start  segment_end  date, trip_id, travel_duration
		str         strptime    int      str         float
	"""
	history = full_history[full_history.trip_id == single_trip]
	print "trip id: ", single_trip
	print "history length: ", len(history)
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
	segment_df = pd.concat(segment_df_list)
	return segment_df



def generate_dataframe(selected_trips, date_start, date_end):
	"""
	This function will read the list of the selected trips and read the them one by one and concatenate all their dataframe together.
	"""
	full_history = filter_history_data(date_start, date_end, selected_trips)
	result_list = []
	for single_trip in selected_trips:
		tmp_segment_df = calculate_travel_duration(single_trip, full_history)
		if tmp_segment_df is None:
			continue
		result_list.append(tmp_segment_df)
	segment_df = pd.concat(result_list)
	return segment_df


#######################################################################################################################
# data improvement
#######################################################################################################################



# Use the first method to improve the data
"""
algorithm:
1. Read the dataset: data.csv, stop_times.txt
2. run the function
3. Export the file.
"""
def improve_dataset(segment_df, stop_times):
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
    print "complete reading the segment data and the stop_times.txt file"

    trips = set(segment_df.trip_id)
    df_list = []
    for single_trip in trips:
        print single_trip
        date_list = list(set(segment_df[segment_df.trip_id == single_trip].date))
        stop_sequence = list(stop_times[stop_times.trip_id == single_trip].stop_id)
        for date in date_list:
            df = improve_dataset_single_trip(single_trip, date, stop_sequence, segment_df)
            df_list.append(df)
    result = pd.concat(df_list)
    return result







def improve_dataset_single_trip(single_trip, date, stop_sequence, segment_df):
    """
    This funciton is used to improve the dataset for a specific trip_id at a spacific date.

    """
    df = pd.DataFrame(columns=['segment_start', 'segment_end', 'segment_pair', 'time_of_day', 'day_of_week', 'date', 'weather', 'trip_id', 'travel_duration'])
    current_segmen_pair = segment_df[(segment_df.trip_id == single_trip) & (segment_df.date == date)]
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
            arrival_time1 = datetime.strptime(current_segmen_pair.iloc[i-1].time_of_day, '%H:%M:%S')
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
                df.loc[len(df)] = [segment_start, segment_end, segment_pair, time_of_day, day_of_week, date, weather, trip_id, travel_duration]
    return df



#######################################################################################################################
# development debug
#######################################################################################################################
path = '../../data/GTFS/gtfs/'
selected_trips = select_trip_list(path)
full_history =filter_history_data(20160104, 20160123, selected_trips)
# segment_df = generate_dataframe(selected_trips, 20160104, 20160123)




# if __name__=="__main__":
# 	selected_trips = select_trip_list(path, 2)
# 	segment_df = generate_dataframe(selected_trips, 20160104, 20160123)
# 	stop_times = pd.read_csv('../data/GTFS/gtfs/stop_times.txt')
# 	improved_dataset = improve_dataset(segment_df, stop_times)
#     improved_dataset.to_csv('segment.csv')
