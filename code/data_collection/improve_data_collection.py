# This file is used to improve the data obtained from the 'segment_data.py' code.
#
# In the 'segment_data.py' code, the travel duration for all the segment pairs in all the selected trips has been recorded. However, because of the precision issue, some of the stops for some trips were skipped in the record. It is not correct for data analysis.
#
# Thus, it is necessary to add the skipped stops back and give a reasonable travel duration for these new segment pairs.
#
# For the travel duration of these segment pairs, there are two ways for consider.
#
# 1. Add the skipped stations according to the schedule data and give the travel duration directly (like 20 seconds).
# 2. Add the skipped stations according to the other records for the same routes, add the travel duration by average ratio.

# import modules
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


# read the data.csv file at first
def read_data():
    segment_df = pd.read_csv('original_segment.csv')
    stop_times = pd.read_csv('../data/GTFS/gtfs/stop_times.txt')
    return segment_df, stop_times


# Use the first method to improve the data
"""
algorithm:
1. Read the dataset: data.csv, stop_times.txt
2. run the function
3. Export the file.
"""


def improve_dataset_method1():
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
    segment_df, stop_times = read_data()
    print "complete reading the segment data and the stop_times.txt file"

    trips = set(segment_df.trip_id)
    df_list = []
    for single_trip in trips:
        print single_trip
        date_list = list(set(segment_df[segment_df.trip_id == single_trip].date))
        stop_sequence = list(stop_times[stop_times.trip_id == single_trip].stop_id)
        for date in date_list:
            df = improve_dataset_method1_unit(single_trip, date, stop_sequence, segment_df)
            df_list.append(df)
    result = pd.concat(df_list)
    return result


def improve_dataset_method1_unit(single_trip, date, stop_sequence, segment_df):
    """
    This funciton is used to improve the dataset for a specific trip_id at a spacific date.
    """
    df = pd.DataFrame(
        columns=['segment_start', 'segment_end', 'segment_pair', 'time_of_day', 'day_of_week', 'date', 'weather',
                 'trip_id', 'travel_duration'])
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
                df.loc[len(df)] = [segment_start, segment_end, segment_pair, time_of_day, day_of_week, date, weather,
                                   trip_id, travel_duration]
    return df


if __name__ == "__main__":
    improved_dataset = improve_dataset_method1()
    improved_dataset.to_csv('improved_segment.csv')
