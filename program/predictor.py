"""
predict the estimated arrival time based on the 
"""

# import module
import pandas as pd
import os

path = '../'


#################################################################################################################
#                                    build dataset                                                              #
#################################################################################################################
"""
The dataset is composed of two parts: input feature and the output
input feature:
    - weather (0, 1, 2)
    - rush hour (0, 1)
    - average predict hour 
    - average speed(ave_speed1) for the previous segment of the current specific trip
    - average speed(ave_speed2) for the current segment of the current specific route in the same date
output feature:
    - true result: actual arrival time
    - predicted result: estimated arrival time
"""


"""
algorithm:
generate the set of tuple(route_id, stop_id, time_of_day) from the test_input
generate date list from the segment_df
for each date in date_list:
    extract single segment data according to the service_date
    get the weather from the weather.csv file
    get the 
"""
def build_training_dataset(test_input, segment_df, weather_df, trips):
    pass







#################################################################################################################
#                                    predict dataset                                                            #
#################################################################################################################


#################################################################################################################
#                                    debug section                                                              #
#################################################################################################################
