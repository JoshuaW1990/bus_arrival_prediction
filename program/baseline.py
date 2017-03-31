"""
Implement the model for the baseline algorithm
1. simplest baseline
2. simple baseline
3. advanced baseline
The difference lies in the data preprocess section. The prediction phase should be the same.
"""

import pandas as pd
import os
from datetime import datetime, timedelta

path = '../'

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


def preprocess_baseline1(segment_df):
    # TODO prepare the dataset for the baseline algorithm
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
    grouped_list = list(segment_df.groupby(['segment_start', 'segment_end']))
    print "length of the grouped list: ", len(grouped_list)
    result = pd.DataFrame(columns=['segment_start', 'segment_end', 'travel_duration'])
    for i in xrange(len(grouped_list)):
        if i % 100 == 0:
            print i
        name, item = grouped_list[i]
        segment_start, segment_end = name
        if segment_start == segment_end:
            continue
        travel_duration_list = list(item['travel_duration'])
        average_travel_duration = sum(travel_duration_list) / float(len(travel_duration_list))
        result.loc[len(result)] = [segment_start, segment_end, average_travel_duration]
    return result


def preprocess_baseline2(segment_df, rush_hour):
    """
    Preprocess the segment data considering the weather and the rush hour
    
    Algorithm:
    Preprocess segment_df to add a new column of rush hour
    split the dataframe with groupby(segment_start, segment_end, weather, rush_hour)
    Define the new dataframe
    For name, item in grouped:
        calcualte the average travel duration
        save the record into the new dataframe
    
    :param segment_df: dataframe after adding the rush hour from final_segment.csv file
    :param rush_hour: tuple to express which is the rush hour, example: ('17:00:00', '20:00:00')
    :return: dataframe for the baseline2
    """
    # Preprocess segment_df to add a new column of rush hour
    rush_hour_column = segment_df['timestamp'].apply(lambda x: x[11:19] < rush_hour[1] and x[11:19] > rush_hour[0])
    new_segment_df = segment_df
    new_segment_df['rush_hour'] = rush_hour_column
    grouped_list = list(new_segment_df.groupby(['segment_start', 'segment_end', 'weather', 'rush_hour']))
    print "length of the grouped_list is: ", len(grouped_list)
    result = pd.DataFrame(columns=['segment_start', 'segment_end', 'travel_duration', 'weather', 'rush_hour'])
    for i in xrange(len(grouped_list)):
        if i % 1000 == 0:
            print i
        name, item = grouped_list[i]
        segment_start, segment_end, weather, rush_hour_var = name
        travel_duration_list = list(item['travel_duration'])
        average_travel_duration = sum(travel_duration_list) / float(len(travel_duration_list))
        result.loc[len(result)] = [segment_start, segment_end, average_travel_duration, weather, rush_hour_var]
    return result


#################################################################################################################
#                                predict section                                                                #
#################################################################################################################
"""
Predict the arrival time for each record in the api data
"""

def calculate_time_from_stop(segment_df, dist_along_route, prev_record, next_record):
    """
    Calculate the time from stop within the tuple (prev_record, next_record)
    
    Algorithm:
    if prev_record = next_record:
        the bus is parking at the stop, return 0
    Calcualte the distance within the tuple
    Calculate the distance between the current location and the prev record
    Calcualte the ratio of these two distances
    Use the ratio to calcualte the time_from_stop
    
    :param segment_df: dataframe for the preprocessed segment data
    :param dist_along_route: distance between the intial stop and the current location of the bus
    :param prev_record: single record of the route_stop_dist.csv file
    :param next_record: single record of the route_stop_dist.csv file
    :return: total seconds of the time_from_stop
    """
    if prev_record.get('stop_id') == next_record.get('stop_id'):
        return 0.0
    distance_stop_stop = next_record.get('dist_along_route') - prev_record.get('dist_along_route')
    distance_stop_bus = next_record.get('dist_along_route') - dist_along_route
    ratio = float(distance_stop_bus) / float(distance_stop_stop)
    assert ratio < 1
    try:
        travel_duration = segment_df[(segment_df.segment_start == prev_record.get('stop_id')) & (segment_df.segment_end == next_record.get('stop_id'))].iloc[0]['travel_duration']
    except:
        travel_duration = segment_df['travel_duration'].mean()
    time_from_stop = travel_duration * ratio
    return time_from_stop



def generate_estimated_arrival_time(api_data, preprocessed_segment_data, route_stop_dist, trips):
    """
    Predict the estimated arrival time according to the api data
    
    Algorithm:
Build the empty dataframe
for row in api_data:
    get the route_id according to the trip id and the trips.txt file
    get the stop sequence and the dist_along_route according to the route id
    get the end_index according to the stop id
    get the (prev, next) stop tuple according to the dist_along_route in the record
    get the count = end_index - next_index
    if count < 0:
        the bus has passed
        continue to next row
    if count = 0, the next stop is the target stop:
        calcualte the time_from_stop for (prev, next) tuple
        save the result as the estimated time
    if count > 0:
        get the stop list from next_stop to target_stop
        sum the travel duration for all of them
        calculate the time_from_stop for (prev, next) tuple
        add the total travel duration with the time_from_stop(prev, next)
        save the result as the estimated time
    save the result into the dataframe
    
    :param api_data: dataframe for the api_data.csv
    :param preprocessed_segment_data: dataframe for the preprocessed final_segment.csv file according to different baseline algorithm
    :param route_stop_dist: dataframe of the route_stop_dist.csv file
    :param trips: dataframe for the trips.txt file
    :return: dataframe to store the result including the esitmated arrival time
    """
    result = pd.DataFrame(columns=['trip_id', 'route_id', 'stop_id', 'vehicle_id', 'time_of_day', 'service_date', 'dist_along_route', 'stop_num_from_call', 'estimated_arrival_time'])
    print "length of the api data is: ", len(api_data)
    average_travel_duration = preprocessed_segment_data['travel_duration'].mean()
    for i in xrange(len(api_data)):
        if i % 1000 == 0:
            print i
        item = api_data.iloc[i]
        trip_id = item.get('trip_id')
        route_id = trips[trips.trip_id == trip_id].iloc[0].route_id
        single_route_stop_dist = route_stop_dist[route_stop_dist.route_id == route_id]
        stop_sequence = list(single_route_stop_dist.stop_id)
        target_stop = item.get('stop_id')
        target_index = stop_sequence.index(target_stop)
        dist_along_route = item.get('dist_along_route')
        vehicle_id = item.get('vehicle_id')
        time_of_day = item.get('time_of_day')
        service_date = item.get('date')
        if dist_along_route >= single_route_stop_dist.iloc[-1].dist_along_route:
            continue
        for j in range(1, len(stop_sequence)):
            if single_route_stop_dist.iloc[j - 1].dist_along_route < dist_along_route < single_route_stop_dist.iloc[j].dist_along_route:
                prev_record = single_route_stop_dist.iloc[j - 1]
                next_record = single_route_stop_dist.iloc[j]
                break
            elif single_route_stop_dist.iloc[j - 1].dist_along_route == dist_along_route:
                prev_record = single_route_stop_dist.iloc[j - 1]
                next_record = prev_record
                break
            else:
                continue
        next_index = stop_sequence.index(next_record.get('stop_id'))
        count = target_index - next_index
        if count < 0:
            continue
        elif count == 0:
            total_travel_duration = calculate_time_from_stop(preprocessed_segment_data, dist_along_route, prev_record, next_record)
        else:
            total_travel_duration = 0.0
            for j in xrange(next_index, target_index):
                segment_start = stop_sequence[j]
                segment_end = stop_sequence[j + 1]
                segment_record = preprocessed_segment_data[(preprocessed_segment_data.segment_start == segment_start) & (preprocessed_segment_data.segment_end == segment_end)]
                if len(segment_record) == 0:
                    single_travel_duration = average_travel_duration
                else:
                    single_travel_duration = segment_record.iloc[0]['travel_duration']
                total_travel_duration += single_travel_duration
            time_from_stop = calculate_time_from_stop(preprocessed_segment_data, dist_along_route, prev_record, next_record)
            total_travel_duration += time_from_stop
        result.loc[len(result)] = [trip_id, route_id, target_stop, vehicle_id, time_of_day, service_date, dist_along_route, count + 1, total_travel_duration]
    return result


def generate_actual_arrival_time(full_history, segment_df, route_stop_dist):
    """
    Calculate the actual arrival time from the dataset
    
    Algorithm:
    Build the empty dataframe
    for row in segment_df:
        get trip_id, route_id, target_stop, service_date, etc
        get single_history data according to the trip id and the service date
        get single_route_stop_dist according to the route_id
        get stop_sequence from single_route_stop_dist
        get the dist_along_route for the target_stop from the single_route_stop_dist
        set prev_index, next_index = stop_sequence.index(target_stop)
        while stop_sequence(prev_index) not in set(single_history.next_stop_id):
            prev_index -= 1
            if prev_index == -1:
                break
        if prev_index == -1:
            continue
        prev_stop = stop_sequence(prev_index)
        next_index += 1
        while stop_sequence(next_index) not in set(single_history.next_stop_id):
            next_index += 1
            if next_index == len(stop_sequence):
                break
        if next_index == len(stop_sequence):
            continue
        next_stop = stop_sequence(next_index)
        prev_record = single_history[single_history.next_stop_id == prev_stop].iloc[-1]
        prev_time = prev_record.get('timestamp')
        if prev_record.dist_from_stop == 0:
            actual_arrival_time = prev_time - time_of_day
            save the record
            continue to next row
        next_record = single_history[single_history.next_stop_id == next_stop].iloc[-1]
        next_time = next_record.get('timestamp')
        travel_duration = next_time - prev_time
        prev_distance = prev_record.get('dist_along_route') - prev_record.get('dist_from_stop')
        next_distance = prev_record.get('dist_along_route') - prev_record.get('dist_from_stop')
        distance_prev_next = next_distance - prev_distance
        distance_prev_stop = single_route_stop_dist[single_route_stop_dist.stop_id == target_stop].iloc[0]['dist_along_route'] - prev_distance
        ratio = distance_prev_stop / distance_prev_next
        time_from_stop = ratio * travel_duration
        arrival_time = time_from_stop + prev_time
        actual_arrival_time = arrival_time - time_of_day
        save the record
    
    :param full_history: 
    :param segment_df: 
    :param route_stop_dist: 
    :return: 
    """
    columns = segment_df.columns + ['actual_arrival_time']
    result = pd.DataFrame(columns=columns)
    print 'length of the segment_df is: ', len(segment_df)
    for i in xrange(len(segment_df)):
        if i % 1000 == 0:
            print i
        item = segment_df.iloc[i]
        trip_id = item.get('trip_id')
        route_id = item.get('route_id')
        single_route_stop_dist = route_stop_dist[route_stop_dist.route_id == route_id]
        stop_sequence = list(single_route_stop_dist.stop_id)
        target_stop = item.get('stop_id')
        target_index = stop_sequence.index(target_stop)
        dist_along_route = single_route_stop_dist[single_route_stop_dist.stop_id == target_stop].iloc[0]['dist_along_route']
        vehicle_id = item.get('vehicle_id')
        time_of_day = item.get('time_of_day')
        service_date = item.get('service_date')
        stop_num_from_call = item.get('stop_num_from_call')
        estimated_arrival_time = item.get('estimated_arrival_time')
        single_history = full_history[full_history.service_date == service_date]
        prev_index, next_index = target_index, target_index + 1
        while stop_sequence[prev_index] not in set(single_history.next_stop_id):
            prev_index -= 1
            if prev_index == -1:
                break
        if prev_index == -1:
            continue
        prev_stop = stop_sequence[prev_index]
        while stop_sequence[next_index] not in set(single_history.next_stop_id):
            next_index += 1
            if next_index == len(stop_sequence):
                break
        if next_index == len(stop_sequence):
            continue
        next_stop = stop_sequence[next_index]
        prev_record = single_history[single_history.next_stop_id == prev_stop].iloc[-1]
        prev_time = prev_record.get('timestamp')
        prev_time = datetime.strptime(prev_time[11:19], '%H:%M:%S')
        time_of_day = datetime.strptime(time_of_day, '%H:%M:%S')
        if prev_record.dist_from_stop == 0:
            actual_arrival_time = prev_time - time_of_day
            actual_arrival_time = actual_arrival_time.total_seconds()
            result.loc[len(result)] = [trip_id, route_id, target_stop, vehicle_id, str(time_of_day.time()), service_date, dist_along_route, stop_num_from_call, estimated_arrival_time, actual_arrival_time]
        next_record = single_history[single_history.next_stop_id == next_stop].iloc[-1]
        next_time = next_record.get('timestamp')
        next_time = datetime.strptime(next_time[11:19], '%H:%M:%S')
        travel_duration = next_time - prev_time
        travel_duration = travel_duration.total_seconds()
        prev_distance = prev_record.get('dist_along_route') - prev_record.get('dist_from_stop')
        next_distance = next_record.get('dist_along_route') - next_record.get('dist_from_stop')
        distance_prev_next = next_distance - prev_distance
        distance_prev_stop = single_route_stop_dist[single_route_stop_dist.stop_id == target_stop].iloc[0]['dist_along_route'] - prev_distance
        ratio = distance_prev_stop / distance_prev_next
        time_from_stop = ratio * travel_duration
        time_prev_bus = prev_time - time_of_day
        time_prev_bus = time_prev_bus.total_seconds()
        actual_arrival_time = time_from_stop + time_prev_bus
        result.loc[len(result)] = [trip_id, route_id, target_stop, vehicle_id, str(time_of_day.time()), service_date, dist_along_route, stop_num_from_call, estimated_arrival_time, actual_arrival_time]
    return result



#################################################################################################################
#                                debug section                                                                  #
#################################################################################################################
# api_data = pd.read_csv('api_data.csv')
# preprocessed_segment_data = pd.read_csv('segment_baseline1.csv')
# route_stop_dist = pd.read_csv('route_stop_dist.csv')
# trips = pd.read_csv(path + 'data/GTFS/gtfs/trips.txt')
# estimated_result = generate_estimated_arrival_time(api_data, preprocessed_segment_data, route_stop_dist, trips)
# estimated_result.to_csv('estimated_segment.csv')
date_list = range(20160125, 20160130)
history_list = []
for current_date in date_list:
    filename = 'bus_time_' + str(current_date) + '.csv'
    history_list.append(pd.read_csv(path + 'data/history/' + filename))
full_history = pd.concat(history_list, ignore_index=True)
segment_df = pd.read_csv('estimated_segment.csv')
route_stop_dist = pd.read_csv('route_stop_dist.csv')
baseline_result = generate_actual_arrival_time(full_history, segment_df, route_stop_dist)


#################################################################################################################
#                                    main function                                                              #
#################################################################################################################

#
# if __name__ == "__main__":
#     file_list = os.listdir('./')
#     print "prepare the segment dataset for different baseline algorithm"
#     segment_df = pd.read_csv('final_segment.csv')
#     if 'segment_baseline1.csv' not in file_list:
#         print "export the segment data for baseline1"
#         new_segment_df = preprocess_baseline1(segment_df)
#         new_segment_df.to_csv('segment_baseline1.csv')
#         print "complete exporting the segment data for baseline1"
#     if "segment_baseline2.csv" not in file_list:
#         print "export the segment data for baseline2"
#         rush_hour = ('17:00:00', '20:00:00')
#         new_segment_df = preprocess_baseline2(segment_df, rush_hour)
#         new_segment_df.to_csv('segment_baseline2.csv')
#         print "complete exporting the segment data for baseline2"
