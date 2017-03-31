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


weather_df = pd.read_csv('weather.csv')
new_weather_df = pd.DataFrame(columns=['date', 'year', 'month', 'day', 'fog', 'rain', 'snow', 'result'])
for i in xrange(len(weather_df)):
    item = weather_df.iloc[i]
    if item.get('snow') == 1:
        result = 2
    elif item.get('rain') == 1:
        result = 1
    else:
        result = 0
    new_weather_df.loc[len(new_weather_df)] = [item['date'], item['year'], item['month'], item['day'], item['fog'], item['rain'], item['snow'], result]

