"""
Build the database from the csv file
"""

import pandas as pd
import psycopg2
from sqlalchemy import create_engine

# db = pymysql.connect(host='localhost', user='root', password='Wj15029054380', db='bus_prediction')


# read the full history data into database



engine = create_engine('postgresql://joshuaw:Wj15029054380@localhost:5432/bus_prediction', echo=False)


# trips = pd.read_csv('../data/GTFS/gtfs/trips.txt')
# stops = pd.read_csv('../data/GTFS/gtfs/stops.txt')
# stop_times = pd.read_csv('../data/GTFS/gtfs/stop_times.txt')

# trips.to_sql(name='trips', con=engine, if_exists='replace', index=False)
# print "complete trips table"
# stops.to_sql(name='stops', con=engine, if_exists='replace', index=False)
# print "complete stops table"
# stop_times.to_sql(name='stop_times', con=engine, if_exists='replace', index=False)
# print "complete stop_times table"


route_stop_dist = pd.read_csv('route_stop_dist.csv')
route_stop_dist.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1, inplace=True)
route_stop_dist.reset_index(inplace=True)
weather = pd.read_csv('weather.csv')
weather.drop(['Unnamed: 0'], axis=1, inplace=True)
weather.reset_index(inplace=True)

route_stop_dist.to_sql(name='route_stop_dist', con=engine, if_exists='replace')
print "complete route_Stop_dist"
weather.to_sql(name='weather', con=engine, if_exists='replace')
print "complete weather"





