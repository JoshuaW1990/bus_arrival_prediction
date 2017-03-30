# import modules
import os
import numpy as np
import requests
import csv
import random
import pandas as pd
from datetime import datetime, timedelta, date
from dateutil.rrule import rrule, DAILY

# set the path
path = '../'


#################################################################################################################
#                                        segment.csv                                                            #
#################################################################################################################
"""
Improve the original segment such that the skipped stops will be added in the middle.
"""



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
    result_list = [
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
        result = pd.concat(result_list)
    return result


segment_df = improve_dataset()
segment_df.to_csv('segment.csv')
