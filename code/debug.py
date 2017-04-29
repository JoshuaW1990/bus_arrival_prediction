"""

Used for debug

Currently check the abnormal route in route_stop_dist
"""


import pandas as pd
from sqlalchemy import create_engine

# db = pymysql.connect(host='localhost', user='root', password='Wj15029054380', db='bus_prediction')


# read the full history data into database



engine = create_engine('postgresql://joshuaw:Wj15029054380@localhost:5432/bus_prediction', echo=False)
full_api_data = pd.read_csv('full_api_data.csv')
full_api_data.drop(['Unnamed: 0'], axis=1, inplace=True)
full_api_data.to_sql(name='full_api_data', con=engine, if_exists='replace', index_label='id')
print "complete save full_api_data into database"

full_api_data_baseline3 = pd.read_csv('tmp.csv')
full_api_data_baseline3.drop(['Unnamed: 0'], axis=1, inplace=True)
full_api_data_baseline3.to_sql(name='full_api_baseline', con=engine, if_exists='replace', index_label='id')

