# Baseline

## Simple Baseline

In the simple baseline algorithm, I used the average travel duration for each segment pair as the input and estimated the arrival time by simply adding up the travel duration for all the segment pairs between the current location and the target stop.

Since there is no way to obtain the exact arrival time for testing, I estimate the actual arrival time by calculating the distance of a specific segment pair which containing the target stop and the distance between the current location from api_data and the previous stop from the historical data and using the ratio of those two distance to obtain the actual arrival time.

The generative process for the simple baseline is below:

```
read the api_df from api_data.csv file

extract the date list from api_df

for each date in date_list:

    read the historical data for the corresponding date

    obtain the single_api_df according to the date

    obtain the estimated_arrival_time according to the single_api_df

    obtain the actual_arrival_time according to the single_api_df

Concatenate the records
```

Thus, the most important processes are the functions for calculating the estimated_arrival_time and the actual_arrival_time

### estimated_arrival_time

The process for calculating the estimated_arrival_time is below:

```
for each record in single_api_df:

    Obtain the necessary information like trip_id, date, time_of_day, route_id, stop_id, dist_along_route, etc from that record of single_api_df

    Obtain the stop sequence according to the trip_id from the stop_times.txt file

    Obtain the stop_dist_list according to the route_id from the route_stop_dist.csv file

    Find the index of the stop_id from the stop_sequence

    If the dist_along_route from api_df is equal to the dist_along_route of the target stop from the stop_dist_list:

        This means that the bus is at that stop at that time point, so save the data into the result directly and continue to next records of the single_api_df

    If the dist_along_route from api_df is larger than the dist_along_route of the target stop from the stop_dist_list:

        This means that the bus has already passed this stop and continue to the next records of the single_api_df

    Start from the target stop and loop to the inital stop in the stop_dist_list until the dist_along_route from api_df is larger than the dist_along_route of a stop in the stop_dist_list. Claim that stop as prev_stop. All the segment_pairs between the target stop and the prev_stop is needed to be considered when calculating the estimated_arrival_time.

    Generate the segment_pair list for the segment_pairs we obtained from the last step.

    Loop in the segment_pair list and accumulate the travel duration for theses segment pairs except for the first segment_pair.

    Since the current location of the bus at the time point is within the first segment_pair of the segment_pair list, we calculate the duration for the first segment_pair by ratio of the distance as well.

    Add all these travel duration including calculated result for the first segment_pair up and the result is the estimated_arrival_time.

Concatenate the all the records
```

### actual_arrival_time

The process for calculating the actual_arrival_time is below:

```
Build the new dataframe as the estimated_arrival_time and add a new column named 'actual_arrival_time'.

For each record in estimated_arrival_time:

    Obtain the stop_id, time_of_day, trip_id, time_of_day, dist_along_route, etc from that record of the estimated_arrival_time.

    Filter the history data into single_history data by trip_id

    Obtain the stop_set from the single_history

    If the stop_id from the record of the estimated_arrival_time is in the stop_set:

        Loop in the single_history to find the segment_pair which containing the current location at the time point.

        If the dist_from_next_stop of the segment_start in the single_history data is 0:

            It means that the bus is actually at that stop. Then the actual_arrival_time should be the duration between this timestamp and the time_of_day from the record of the estimated_arrival_time.

        Else:

            Calculate the actual arrival time by the ratio of the distance and the travel duration of the segment pair.

    If the stop_id from the record of the estimated_arrival_time is not in the stop_set:

        Obtain the stop_sequence from the stop_times with trip_id

        Find the previous stop and the next stop in the single_history data according to the stop_sequence

        Use the nearest stops to form a segment_pair containing the current location given by the record of the estimated_arrival_time

        Calculate the actual_arrival_time based on the ratio of the distance from the segment_pair

    Add the actual_arrival_time and other information into the new dataframe

Concatenate all the records from the dataframe
```
