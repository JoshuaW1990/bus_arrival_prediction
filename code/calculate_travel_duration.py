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
    segment_df = pd.read_csv('improved_segment.csv')
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
    for segment_pair in segment_set:
        tmp_segment_df = segment_df[segment_df.segment_pair == segment_pair]
        num = float(len(tmp_segment_df))
        average_travel_duration = sum(list(tmp_segment_df.travel_duration)) / num
        segment_start = segment_pair.split(',')[0][1:]
        segment_end = segment_pair.split(',')[1][:-1]
        new_segment_df.loc[len(new_segment_df)] = [segment_start, segment_end, segment_pair, average_travel_duration]
    return new_segment_df


# Obtain the set of the segment_pair with consideration of the weather and the rush hour
def obtain_segment_set_baseline2(segment_df):
    """

    :param segment_df: the dataframe for storing the average travel duration according to the requirement.
            
    :return:
    """

if __name__ == "__main__":
    segment_df = read_dataset()
    new_segment_df = obtain_segment_set_baseline1(segment_df)
    # export the file
    new_segment_df.to_csv('average_segment_travel_duration.csv')
