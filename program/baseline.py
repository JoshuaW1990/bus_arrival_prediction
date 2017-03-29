"""
Implement the model for the baseline algorithm
1. simplest baseline
2. simple baseline
3. advanced baseline
The difference lies in the data preprocess section. The prediction phase should be the same.
"""

import pandas as pd
from datetime import datetime


#################################################################################################################
#                                data preproces                                                                 #
#################################################################################################################
"""
Calculate the average travel duration for the segment data
1. simplest baseline
2. simple baseline
3. advanced baseline
Use the groupby function for the segment dataframe
"""

def read_dataset():
    """
    read the segment.csv file to obtain the segment dataset
    :return: dataframe of the segemnt.csv file
    """
    segment_df = pd.read_csv('segment.csv')
    return segment_df


def preprocess_baseline1(segment_df):
    """
    preprocession for the simplest baseline: not considering the weather and the time
    Algorithm:
    Read the database
    Group the dataframe according to the segment start and the segment end
    For each item in the grouped list:
        obtain the name and the sub dataframe
        check whether the segment_start and the segment_end is the same (we need to fix this bug later when retrieving the segment data)
        Calculate the average travel duration
        save the record into the new dataframe
    :param segment_df: 
    :return: the preprocessed segment dataframe
    """
    grouped = segment_df.groupby(['segment_start', 'segment_end'])
    result = pd.DataFrame(columns=['segment_start', 'segment_end', 'travel_duration'])
    for name, item in grouped:
        segment_start, segment_end = name
        if segment_start == segment_end:
            continue
        travel_duration_list = list(item['travel_duration'])
        average_travel_duration = sum(travel_duration_list) / float(len(travel_duration_list))
        if average_travel_duration < 0:
            print name
        if average_travel_duration > 10:
            print name
        result.loc[len(result)] = [segment_start, segment_end, average_travel_duration]
    return result



#################################################################################################################
#                                debug section                                                                  #
#################################################################################################################
segment_df = read_dataset()
new_segment_df = preprocess_baseline1(segment_df)

