# Read the different GTFS file and export them into csv file

import pandas as pd
import numpy as np
import csv
import os


file_path = '../data/GTFS/'

file_list = os.listdir(file_path)


"""
the file we need:
trip_id, route_id, stop_list(stop_times)
"""
def preprocess_data(folder_path):
    
    






# Main code
for file_name in file_list:
    folder_path = file_path + file_name + '/'
    # route_dict: key is the route_id, value is the stop_sequence
    route_dict = []
    # trip_dict: key is the trip_id, value is the route_id
    trip_dict = []
    preprocess_data(folder_path)



