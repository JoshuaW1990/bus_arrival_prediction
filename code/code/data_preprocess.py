# Preprocess of the data collection for the project

# module import
import pandas as pd
import numpy as np


# read gtfs data
trips = pd.read_csv('../../data/GTFS/google_transit_staten_island/trips.txt')
route = list(set(list(trips.route_id)))


