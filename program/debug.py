"""

Used for debug

Currently check the abnormal route in route_stop_dist
"""


import pandas as pd
import os


file_list = os.listdir('./')

if 'preprocessed_full_baseline_result.csv' not in file_list:
    baseline_result = pd.read_csv('full_baseline_result.csv')
    trips = pd.read_csv('../data/GTFS/gtfs/trips.txt')
    trips_dict = trips.set_index('trip_id').to_dict(orient='index')
    baseline_result['shape_id'] = baseline_result['trip_id'].apply(lambda x: trips_dict[x]['shape_id'])
    baseline_result.to_csv('preprocessed_full_baseline_result.csv')
else:
    baseline_result = pd.read_csv('preprocessed_full_baseline_result.csv')


original_route_stop_dist = pd.read_csv('origin_route_stop_dist.csv')
route_stop_dist = pd.read_csv('route_stop_dist.csv')

original_grouped = original_route_stop_dist.groupby(['route_id'])
grouped = route_stop_dist.groupby(['route_id'])

shape_id_set = set()

for route_id, item in grouped:
    sub_grouped = item.groupby(['shape_id'])
    original_single_route_stop_dist = original_grouped.get_group(route_id)
    original_stop_list = list(original_single_route_stop_dist['stop_id'])
    for shape_id, single_route_stop_dist in sub_grouped:
        stop_list = list(single_route_stop_dist['stop_id'])
        if original_stop_list == stop_list:
            shape_id_set.add(shape_id)

final_baseline_result = baseline_result[baseline_result.shape_id.isin(shape_id_set)]
