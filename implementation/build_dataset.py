"""
predict the estimated arrival time based on the
"""

# import module
import baseline
import os
import pandas as pd
from datetime import datetime, timedelta

#################################################################################################################
#                                    build dataset                                                              #
#################################################################################################################
"""
The dataset is composed of two parts: input feature and the output
input feature:
    - weather (0, 1, or 2)
    - rush hour (0 or 1)
    - estimated arrival time with third baseline algorithm
    - average speed(prev_seg_speed) for the previous neighboring segment in the that specific trip
    - average speed(current_seg_speed) for the same segment of the previous trip in the same route
Output:
    - True value: actual arrival time
    - Predicted value: predicted arrival time with regression model
"""


def generate_complete_dateset(api_data, segment_df, route_stop_dist, trips, full_history, weather_df, rush_hour):
    """
    Calculate the complete result of baseline3

    :param api_data:
    :param segment_df:
    :param route_stop_idst:
    :param trips:
    :param full_history:
    :return:
    """
    print "start to export the result of the dataset"
    weather_df['date'] = pd.to_numeric(weather_df['date'])
    result = baseline.obtain_baseline3(segment_df, api_data, route_stop_dist, trips, full_history)
    result['service_date'] = pd.to_numeric(result['service_date'])
    result['weather'] = result['service_date'].apply(lambda x: weather_df[weather_df.date == x].iloc[0]['weather'])
    result['rush_hour'] = result['time_of_day'].apply(lambda x: 1 if rush_hour[1] >= x[11:19] >= rush_hour[0] else 0)
    return result


def generate_feature_api(single_segment, dist_along_route, target_dist, time_of_day, single_route_stop_dist):
    """
    Generate the api list for calculating the average delay

    Algorithm:
    (1) get the single segment smaller or equal than the target_dist
    (2) generate the feature api list by each row in filtered single segment data
    (3) use the timestamp in segment data for the actual arrival time

    :param single_segment:
    :param dist_along_route:
    :param target_dist:
    :param time_of_day:
    :param single_route_stop_dist:
    :return:
    """
    feature_api = pd.DataFrame(columns=['actual_arrival_time'])
    # get the single segment smaller or equal than the target_dist
    try:
        current_segment = single_segment[single_segment['segment_end'].apply(lambda x: single_route_stop_dist[single_route_stop_dist['stop_id'] == x].iloc[0]['dist_along_route'] <= target_dist)]
    except:
        print "error"
    index = 0
    while index < len(current_segment):
        if single_route_stop_dist[single_route_stop_dist['stop_id'] == current_segment.iloc[index]['segment_start']].iloc[0]['dist_along_route'] < dist_along_route:
            index += 1
        else:
            break
    if index == len(current_segment):
        return None
    current_segment = current_segment[index:]
    current_segment.reset_index(inplace=True)

    # generate the actual arrival time from the travel duration of the single segment date
    feature_api.loc[0] = [current_segment.iloc[0]['travel_duration']]
    for i in xrange(1, len(current_segment)):
        feature_api.loc[i] = [feature_api.loc[i - 1]['actual_arrival_time'] + current_segment.loc[i]['travel_duration']]

    # generate the feature api list by each row in filtered single segment data
    route_id = single_route_stop_dist.iloc[0]['route_id']
    feature_api['trip_id'] = current_segment['trip_id']
    feature_api['vehicle_id'] = current_segment['vehicle_id']
    feature_api['route_id'] = route_id
    feature_api['stop_id'] = current_segment['segment_end']
    feature_api['time_of_day'] = time_of_day
    feature_api['date'] = current_segment['service_date']
    feature_api['dist_along_route'] = dist_along_route
    feature_api['shape_id'] = single_route_stop_dist.iloc[0]['shape_id']

    return feature_api


def obtain_prev_trip(single_segment, stop_id, time_of_day):
    """
    obtain the trip id of the previous trip

    Algorithm:
    1) divide the single_segment through groupby([trip id, service date])
    2) for name, item in grouped single history:
        (1) obtain trip id and service date from name
        (2) item is the single segment for trip id and the service date
        (3) find arrival time that bus arrived to the target stop from the single segment
        (4) compare the arrival time and the time_of_day:
            a) if arrival_time < time_of_day, compare the bucket tuple (arrival_time, trip id, service date) to obtain the maximum arrival_time
            b) else, continue
    3) return maximum bucket (arrival_time, trip id, service date)

    :param single_segment:
    :param stop_id:
    :param time_of_day:
    :return:
    """
    # divide the single_history through groupby([trip id, service date])
    grouped = single_segment.groupby(['trip_id', 'service_date'])
    time_of_day = datetime.strptime(time_of_day, '%Y-%m-%d %H:%M:%S')
    result = []
    for name, item in grouped:
        # obtain trip id and service date from name
        trip_id, service_date = name

        # find arrival time that bus arrived to the target stop from the single segment
        single_record = item[item['segment_start'] == stop_id]
        if len(single_record) == 0:
            continue
        arrival_time = single_record.iloc[0]['timestamp']
        arrival_time = datetime.strptime(arrival_time[:19], '%Y-%m-%d %H:%M:%S')

        # compare the arrival time and the time_of_day to obtain the maximum available arrival time
        if arrival_time <= time_of_day:
            result.append((arrival_time, trip_id, service_date))
        else:
            continue
    result.sort(key=lambda x: x[0])
    result.reverse()
    return result


def obtain_time_of_day(single_segment, dist_along_route, single_route_stop_dist):
    """
    obtain the time of day when the bus located at the dist_along_route in baseline3result record

    Algorithm:
    1) use the dist_along_route find the corresponding segment pair
    2) calculate the time_of_day according to the dist_along_route within that segment pair

    :param single_segment:
    :param dist_along_route:
    :param single_route_stop_dist:
    :return:
    """
    # use the dist_along_route find the corresponding segment pair
    flag = 0
    for i in xrange(len(single_segment)):
        prev_stop = single_segment.iloc[i]['segment_start']
        next_stop = single_segment.iloc[i]['segment_end']
        prev_timestamp = single_segment.iloc[i]['timestamp']
        travel_duration = single_segment.iloc[i]['travel_duration']
        prev_distance = single_route_stop_dist[single_route_stop_dist['stop_id'] == prev_stop].iloc[0]['dist_along_route']
        next_distance = single_route_stop_dist[single_route_stop_dist['stop_id'] == next_stop].iloc[0]['dist_along_route']
        if prev_distance <= dist_along_route < next_distance:
            flag = 1
            break
        else:
            continue
    if flag == 0:
        return None

    # calculate the time_of_day according to the dist_along_route within that segment pair
    prev_timestamp = datetime.strptime(prev_timestamp[:19], '%Y-%m-%d %H:%M:%S')
    travel_duration = timedelta(0, travel_duration)
    next_timestamp = prev_timestamp + travel_duration
    time_of_day = baseline.calculate_arrival_time(dist_along_route, prev_distance, next_distance, prev_timestamp, next_timestamp)

    return str(time_of_day)



def calculate_average_delay(feature_api, segment_df, route_stop_dist, trips):
    """
    calculate the average arrival time

    Algorithm:
    1) run baseline3 algorithm for the estimated arrival time
    2) calculate the delay between the actual arrival time and the estimated arrival time
    3) calculate the average delay

    :param feature_api:
    :param segment_df:
    :param route_stop_dist:
    :param trips:
    :return:

    'trip_id', 'vehicle_id', 'route_id', 'stop_id', 'time_of_day', 'date', 'dist_along_route'
    """
    # run baseline3 algorithm for the estimated arrival time
    result = baseline.generate_estimated_arrival_time_baseline3(feature_api, segment_df, route_stop_dist, trips)
    feature_api.reset_index(inplace=True)
    result['actual_arrival_time'] = feature_api['actual_arrival_time']

    # calculate the delay between the actual arrival time and the estimated arrival time
    result['delay'] = (result['actual_arrival_time'] - result['estimated_arrival_time']) / result['estimated_arrival_time']
    result['ratio'] = result['actual_arrival_time'] / result['estimated_arrival_time']

    # calculate the average delay
    return result['delay'].mean(), result['ratio'].mean()


def calculate_prev_arrival_time(filtered_segment, segment_list, dist_along_route, single_route_stop_dist):
    """
    calculate the prev arrival time

    Algorithm:
    1) divide the filtered segment by segment start and the segment end
    2) obtain the arrival time in the first segment list:
        (1) calculate the distance between the segment start and the segment end
        (2) calculate the distance between the current distance and the segment end
        (3) use the last record in the grouped segment data with the specific segment start and the segment end
        (4) get the travel duration of the last record
        (5) calculate the arrival time for this segment according to the ratio
    3) add the travel duration of the other segments in the segment list in for loop



    :param filtered_segment:
    :param segment_list:
    :return:
    """
    prev_arrival_time = 0.0
    filtered_segment['segment_start_dist'] = filtered_segment['segment_start'].apply(lambda x: single_route_stop_dist[single_route_stop_dist.stop_id == x].iloc[0]['dist_along_route'])
    filtered_segment['segment_end_dist'] = filtered_segment['segment_end'].apply(lambda x: single_route_stop_dist[single_route_stop_dist.stop_id == x].iloc[0]['dist_along_route'])
    grouped = filtered_segment.groupby(['segment_start', 'segment_end'])
    # obtain the arrival time in the first segment list
    current_segment = segment_list[0]
    single_record = grouped.get_group(current_segment).iloc[-1]
    distance_stop_stop = single_record['segment_end_dist'] - single_record['segment_start_dist']
    distance_loc_stop = single_record['segment_end_dist'] - dist_along_route
    ratio = distance_loc_stop / distance_stop_stop
    travel_duration = single_record['travel_duration']
    prev_arrival_time += travel_duration * ratio
    # add the travel duration of the other segments in the segment list in for loop
    for i in xrange(1, len(segment_list)):
        current_segment = segment_list[i]
        if current_segment in grouped.groups.keys():
            prev_arrival_time += grouped.get_group(current_segment).iloc[-1]['travel_duration']
        else:
            prev_arrival_time += filtered_segment['travel_duration'].mean()
    return prev_arrival_time


def preprocess_dataset(total_baseline_result, segment_df, route_stop_dist, trips):
    """
    Build the dataset for regression algorithms

    Algorithm:
    1) define the result for the dataset
    2) For row in baseline_result.csv file:
        (1) obtain single record, trip id, service date, route id, dist_along_route
        (2) generate delay of current trip:
            a) obtain the single segment and single history according to the trip id and the service date
            b) obtain the time_of_day from the first record of the single segment
            c) use the dist_along_route as the target_dist
            d) generate the feature_api list
        (3) calculate the average delay of the current trip
        (4) generate delay of previous trip:
            a) obtain the trip list according to the route id
            b) filter the segment and history by the trip list and the service date list
            c) obtain id of the previous trip
            d) obtain the single segment and single history by the previous trip id
            e) obtain the time_of_day such that the bus located at the dist_along_route in baseline3result record
            f) use the target stop dist as the target_dist
            g) generate the feature_api list
        (5) calculate the average delay of the previous trip
        (6) generate the prev_arrival_time
            a) filter the segment data by service date
            b) generate the segment list between the dist_along_route and the target stop
                (a) index = 0
                (b) use the while loop to obtain the start index which is just the stop before the dist_along_route
                (c) use the same while loop to obtain the end index which is the target stop
                (d) obtain the filtered stop sequence from the start index and the end index
                (e) generate the segment list from the filtered stop sequence
            c) calculate the prev_arrival_time
        (6) save the record in the result

    :param total_baseline_result:
    :param segment_df:
    :param route_stop_dist:
    :param trips:
    :return:
    """
    grouped = total_baseline_result.groupby(['shape_id'])
    result_list = []
    for shape_id, baseline_result in grouped:
        print "length of the baseline_result table: ", len(baseline_result)
        result = pd.DataFrame(
            columns=['trip_id', 'service_date', 'weather', 'rush_hour', 'baseline_result', 'delay_current_trip',
                     'ratio_current_trip', 'delay_prev_trip', 'ratio_prev_trip', 'prev_arrival_time',
                     'actual_arrival_time', 'shape_id', 'stop_id', 'time_of_day', 'dist_along_route'])
        for i in xrange(len(baseline_result)):
            if i % 100 == 0:
                print "index is ", i
            # obtain single record, trip id, service date, route id, dist_along_route
            single_record = baseline_result.iloc[i]
            trip_id = single_record.get('trip_id')
            service_date = single_record.get('service_date')
            stop_id = single_record.get('stop_id')
            dist_along_route = single_record.get('dist_along_route')
            time_of_day = single_record.get('time_of_day')
            single_route_stop_dist = route_stop_dist[route_stop_dist['shape_id'] == shape_id]

            # generate delay of the current trip
            single_segment = segment_df[(segment_df.trip_id == trip_id) & (segment_df.service_date.isin([service_date]))]
            single_segment = single_segment[single_segment['timestamp'] <= time_of_day]
            if len(single_segment) == 0:
                continue
            current_time_of_day = single_segment.iloc[0]['timestamp']
            target_dist = dist_along_route
            initial_dist = single_route_stop_dist[single_route_stop_dist['stop_id'] == single_segment.iloc[1]['segment_start']].iloc[0]['dist_along_route']
            if initial_dist >= target_dist:
                continue
            feature_api = generate_feature_api(single_segment, initial_dist, target_dist, current_time_of_day, single_route_stop_dist)
            if feature_api is None:
                continue
            delay_current_trip, ratio_current_trip = calculate_average_delay(feature_api[-1:], segment_df, route_stop_dist, trips)
            # print delay_current_trip

            # generate the delay of the previous trip
            trip_list = list(trips[trips['shape_id'] == shape_id]['trip_id'])
            single_segment = segment_df[(segment_df.trip_id.isin(trip_list)) & (segment_df.service_date.isin([service_date - 1, service_date]))]
            prev_trip_list = obtain_prev_trip(single_segment, stop_id, time_of_day)
            if prev_trip_list == []:
                continue
            current_segment = single_segment
            for prev_trip_tuple in prev_trip_list:
                single_segment = current_segment[(current_segment['trip_id'] == prev_trip_tuple[1]) & (current_segment['service_date'] == prev_trip_tuple[2])]
                current_time_of_day = obtain_time_of_day(single_segment, dist_along_route, single_route_stop_dist)
                if current_time_of_day is not None:
                    break
            if current_time_of_day is None:
                continue
            target_dist = single_route_stop_dist[single_route_stop_dist['stop_id'] == stop_id].iloc[0]['dist_along_route']
            feature_api = generate_feature_api(single_segment, dist_along_route, target_dist, current_time_of_day, single_route_stop_dist)
            if feature_api is None:
                continue
            delay_prev_trip, ratio_prev_trip = calculate_average_delay(feature_api[-1:], segment_df, route_stop_dist, trips)
            # print delay_prev_trip

            # generate the prev_arrival_time
            # generate the segment list between the dist_along_route and the target stop
            index = 0
            while index < len(single_route_stop_dist):
                if single_route_stop_dist.iloc[index]['dist_along_route'] >= dist_along_route:
                    break
                index += 1
            if index == len(single_route_stop_dist):
                continue
            start_index = index - 1
            while index < len(single_route_stop_dist):
                if single_route_stop_dist.iloc[index]['stop_id'] == stop_id:
                    break
                index += 1
            if index == len(single_route_stop_dist):
                continue
            end_index = index + 1
            filtered_stop_sequence = list(single_route_stop_dist.stop_id)[start_index:end_index]
            segment_list = [(filtered_stop_sequence[_ - 1], filtered_stop_sequence[_]) for _ in range(1, len(filtered_stop_sequence))]
            # calculate the prev_arrival_time
            filtered_segment = segment_df[segment_df.service_date.isin([service_date - 1, service_date])]
            filtered_segment = filtered_segment[(filtered_segment.segment_start.isin(filtered_stop_sequence[:-1])) & (filtered_segment.segment_end.isin(filtered_stop_sequence[1:]))]
            filtered_segment = filtered_segment[filtered_segment['timestamp'].apply(lambda x: datetime.strptime(x[:19], '%Y-%m-%d %H:%M:%S') <= datetime.strptime(time_of_day, '%Y-%m-%d %H:%M:%S'))]
            prev_arrival_time = calculate_prev_arrival_time(filtered_segment, segment_list, dist_along_route, single_route_stop_dist)

            # 'weather', 'rush_hour', 'baseline_result', 'delay_current_trip', 'delay_prev_trip', 'prev_arrival_time', 'delay_neighbor_stops',
            single_result = [trip_id, service_date, single_record.get('weather'), single_record.get('rush_hour'), single_record.get('estimated_arrival_time'), delay_current_trip, ratio_current_trip, delay_prev_trip, ratio_current_trip, prev_arrival_time,   single_record.get('actual_arrival_time'), shape_id, stop_id, time_of_day, dist_along_route]
            result.loc[len(result)] = single_result
        result_list.append(result)
    return pd.concat(result_list, ignore_index=True)


#################################################################################################################
#                                    main function                                                             #
#################################################################################################################


def obtain_dataset(startdate, api_data, segment_df, route_stop_dist, trips, full_history, weather_df, rush_hour, tablename=None, save_path=None, engine=None):
    """
    Obtain the dataset for training and testing
    
    :param startdate: the start date for obtaining the dataset. Usually, the first date in the schedule dataset should be skipped because there is no historical data for it.
    :param api_data: the dataframe for the api_data table
    :param segment_df: the dataframe for the segment table
    :param route_stop_dist: the dataframe for the route_stop_dist table
    :param trips: the dataframe for the trips.txt file in the GTFS dataset
    :param full_history: the dataframe for the history table
    :param weather_df: the dataframe for the weather table
    :param rush_hour: the tuple of string to represent the rush hour, example: ('17:00:00', '20:00:00')
    :param tablename: the table name for exporting the file
    :param save_path: path of a csv file to store the baseline1 result
    :param engine: database connector
    :return: the dataframe for the dataset table
    """
    # generate the complete dataset
    print "generate the complete dataset"
    # api_data, segment_df, route_stop_dist, trips, full_history, weather_df, rush_hour
    baseline_result = generate_complete_dateset(api_data, segment_df, route_stop_dist, trips, full_history, weather_df, rush_hour)

    # preprocess the complete dataset for obtaining the input features
    print "preprocess the complete dataset for obtaining the input features"
    baseline_result = baseline_result[baseline_result.service_date > startdate]
    dataset = preprocess_dataset(baseline_result, segment_df, route_stop_dist, trips)
    if save_path is not None and tablename is not None:
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        dataset.to_csv(save_path+tablename + '.csv')
    if engine is not None and tablename is not None:
        dataset.to_sql(name=tablename, con=engine, if_exists='replace', index_label='id')

    return dataset




