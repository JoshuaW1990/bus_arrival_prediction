import pandas as pd
import numpy as np
import os


data = pd.read_csv('data1.csv')
single_trip = data.iloc[0].trip_id
date = data.iloc[0].date
current_segment_pair = data[(data.trip_id == single_trip) & (data.date == date)]
stop_times = pd.read_csv('../data/GTFS/gtfs/stop_times.txt')
stop_sequence = list(stop_times[stop_times.trip_id == single_trip].stop_id) 

for i in xrange(1, len(current_segment_pair)):
    segment_start = current_segment_pair.iloc[i - 1].segment_start
    segment_end = current_segment_pair.iloc[i].segment_start
    start_idx = stop_sequence.index(segment_start)
    end_idx = stop_sequence.index(segment_end)
    if end_idx - start_idx == 1:
        print start_idx, end_idx


