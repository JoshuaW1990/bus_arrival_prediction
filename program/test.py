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


#################################################################################################################
#                                        segment.csv                                                            #
#################################################################################################################
"""
Fix the bugs in the finals segment.csv file:
1. some of the rows are repeated, we need to remove the duplicated one
2. Som travel duration is too large, we need to remove the travel duration which is larger than 10 minutes

Algorithm:
split the segment with groupby(service_date, trip_id)
result_list = []
for name, item in splitted segment:
    do improve for the item
    append the result into result_list
concatenate the result_list
"""

def fix_bug_segment(segment_df):
    """
    Fix the bugs for the segment after improvement:
    1. some of the rows are repeated, we need to remove the duplicated one
    2. Som travel duration is too large, we need to remove the travel duration which is larger than 10 minutes
    
    Algorithm:
    split the segment with groupby(service_date, trip_id)
    result_list = []
    for name, item in splitted segment:
        do improve for the item
        append the result into result_list
    concatenate the result_list
    :param segment_df: dataframe for segment.csv
    :return: the dataframe after fixing the bugs in the segment.csv
    """
    grouped_list = list(segment_df.groupby(['service_date', 'trip_id']))
    result_list = []
    print 'length of the grouped list: ', len(grouped_list)
    for i in xrange(len(grouped_list[:10])):
        if i % 1000 == 0:
            print i
        name, item = grouped_list[i]
        current_segment = fix_bug_single_segment(item)
        result_list.append(current_segment)
    result = pd.concat(result_list)
    return result



def fix_bug_single_segment(segment_df):
    """
    Fix the bug for a segment dataframe with specific service date and the trip id
    
    Algorithm:
    Define the dataframe for the result
    For i in range(1, len(segment_df):
        prev_record = segment_df.iloc[i - 1]
        next_record = segment_df.iloc[i]
        if prev_record.segment_start = next_record.segment_start and prev_record.segment_end == next_record.segment_end:
            This is a duplicated record, continue to next row
        if the prev_record.travel_duration > 600 (10 minutes), continue to next row
        save prev_record into result
    
    :param segment_df: dataframe of the single segment data
    :return: dataframe for the segment after fixing the bug
    """
    result = pd.DataFrame(columns=segment_df.columns)
    for i in xrange(1, len(segment_df)):
        prev_record = segment_df.iloc[i - 1]
        next_record = segment_df.iloc[i]
        # check whether the row is duplicated
        if prev_record.segment_start == next_record.segment_start and prev_record.segment_end == next_record.segment_end:
            continue
        # check the travel duration
        if prev_record.travel_duration > 600:
            continue
        result.loc[len(result)] = prev_record
    if result.iloc[-1].segment_start != segment_df.iloc[-1].segment_start and result.iloc[-1].segment_end != segment_df.iloc[-1].segment_end:
        result.loc[len(result)] = segment_df.iloc[-1]
    return result


segment = pd.read_csv('segment.csv')
final_segment = fix_bug_segment(segment)
