"""
Calculate the travel duration for the each segment pair according to the data1.csv file
"""


import pandas as pd
import numpy as np
import os


def read_dataset():
    """
    read the data.csv file to obtain the segment dataset
    """
    segment_df = pd.read_csv('segment.csv')
    return segment_df

# Obtain the set of the segment_pair
def obtain_segment_set_baseline1(segment_df):
    # Loopin through all the segment in the set and calculate the average travel duration
    # Since during calculating the travel duration, according to the first algorithm, many unecessary information will be ignored, we will buid a much simpler dataframe for storing the result
    # The format of the new dataframe:
    #    segment_start    segment_end    segment_pair    travel_duration
    # 		str              str           (str, str)      float(second)
    segment_set = set(segment_df.segment_pair)
    new_segment_df = pd.DataFrame(columns = ['segment_start', 'segment_end', 'segment_pair', 'travel_duration'])
    for i, segment_pair in enumerate(segment_set):
        if i % 100 == 0:
            print i
        tmp_segment_df = segment_df[segment_df.segment_pair == segment_pair]
        num = float(len(tmp_segment_df))
        average_travel_duration = sum(list(tmp_segment_df.travel_duration)) / num
        segment_start = segment_pair.split(',')[0][1:]
        segment_end = segment_pair.split(',')[1][:-1]
        new_segment_df.loc[len(new_segment_df)] = [segment_start, segment_end, segment_pair, average_travel_duration]
    return new_segment_df

def SplitSegmentRushHour(segment_df, rush_hour):
    """
    Split the segment_df into two child group according to wether the time_of_day is in rush hour
    :param segment_df: segment dataframe, columns: 'segment_start,segment_end,segment_pair,time_of_day,day_of_week,date,weather,trip_id,travel_duration'
    :param rush_hour: tuple composed by the start and the end of the rush hour, ex: ('17:00:00', '20:00:00')
    :return: a list formed by two child segment dataframe with the exactly same columns like the input
    """
    rush_weather_segment_df = pd.DataFrame(columns = segment_df.columns)
    nonrush_weather_segment_df = pd.DataFrame(columns = segment_df.columns)
    print "splitting the segment dataframe"
    for i in xrange(len(segment_df)):
        if i % 100 == 0:
            print i
        if segment_df.iloc[i].time_of_day <= rush_duration[1] and segment_df.iloc[i].time_of_day >= rush_duration[0]:
            rush_weather_segment_df.loc[len(rush_weather_segment_df)] = segment_df.iloc[i]
        else:
            nonrush_weather_segment_df.loc[len(nonrush_weather_segment_df)] = segment_df.iloc[i]
    return (nonrush_weather_segment_df, rush_weather_segment_df)

# Obtain the set of the segment_pair with consideration of the weather and the rush hour
def obtain_segment_set_baseline2(segment_df, rush_duration):
    """
    This function can calculate the average travel duration based on the weather and the rush hours
    :param segment_df: the dataframe for storing the average travel duration according to the requirement.
    :param rush_duration: the tuple for determining the rush hour.
    :return: the dataframe for the calculated average travel duration
    """
    weather_set = set(segment_df.weather)
    print "weather set: ", weather_set
    result_list = []
    for weather in weather_set:
        print "weather: ", weather
        single_weather_segment_df = segment_df[segment_df.weather == weather]
        print "single_weather_segment_df: ", single_weather_segment_df.shape
        tmp_weather_segment_df = SplitSegmentRushHour(single_weather_segment_df, rush_duration)
        for i, tmp_single_weather_segment_df in enumerate(tmp_weather_segment_df):
            print i
            print "tmp_single_weather_segment_df: ", tmp_single_weather_segment_df.shape
            tmp_segment_df = obtain_segment_set_baseline1(tmp_single_weather_segment_df)
            if len(tmp_segment_df) != 0:
                tmp_segment_df['weather'] = weather
                tmp_segment_df['rush_hour'] = str(i)
            else:
                continue
            result_list.append(tmp_segment_df)
    result = pd.concat(result_list)
    return result

"""
Algorithm:
read the segment dataframe
Define the rush hour and the unrush hour
read the set of the weather
for each weather in the weather set:
    obtain the segment dataframe with the corresponding weather from the whole segment dataframe
    divide the segment dataframe into two different sub dataframe according to the time_of_day and the rush hour rule
    for each sub dataframe:
        calculate the average travel duration for each specific segment with the same method like obtain_segment_set_baseline1 function
concatenate the result
"""

# segment_df = read_dataset()
# # Define the rush hour
# rush_duration = ('17:00:00', '20:00:00')
# new_segment_df =obtain_segment_set_baseline2(segment_df, rush_duration)









if __name__ == "__main__":
    segment_df = read_dataset()
    # # baseline1
    # new_segment_df = obtain_segment_set_baseline1(segment_df)
    # new_segment_df.to_csv('average_segment_travel_duration.csv')
    # baseline 2
    # Define the rush hour
    rush_duration = ('17:00:00', '20:00:00')
    new_segment_df = obtain_segment_set_baseline2(segment_df, rush_duration)
    new_segment_df.to_csv('baseline2.csv')
