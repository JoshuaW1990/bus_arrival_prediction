# Data Preparation

The preparation of the data is composed of four different parts for different files: `weather.csv`, `segment.csv`, `api_data.csv` and `route_stop_dist.csv`.


## Weather

Weather data is downloaded from the website [wunderground](http://api.wunderground.com/api). By giving the date list as the input, the corresponding weather data will be downloaded. The `weather.csv` file can indicates the existence of the fog, rain, snow on each date.

## Segment Pair

The segment_pair data is composed by two different parts:
1. Obtain the generative travel duration for all existed segment_pair in the historical data.
2. According to the stop sequence for each route, add the missed data for the missed segment_pair in each route.

### First Step

The process for this step is below:

```
Generate the nonduplicated list for route_id and select several routes according to the requirement.

Obtain all the corresponding trips for the selected trips and filter this trip list by direction_id (direction_id can only be 0 or 1, here we use 0).

Read all the historical data according to the given date list.

for loop: single_trip in trip list

    Filter the historical data according to the trip_id

    Obtain date list from the historical data

    for loop: date in date list

        Filter the historical data according to the date and obtain the single_history

        Sometimes the bus is operated within the same segments at several time point. Remove the previous ones and only keep the last record for each segment.

        Generate the segment pair for each neighboring records in the filtered historical data and calculate the corresponding travel duration.

        Add the additional data like date, weather, trip_id, etc for each records

Concatenate all the calculated records
```

### Second Step

The process for the second step is below:

```
Obtain the set of trip_id from the segment_pair data from previous step

for each specific trip_id:

    filter the segment_pair data according to the trip_id

    obtain the date_list from the segment_pair data for the specific trip id

    obtain the stop_sequence from the stop_times.txt file according to the trip id

    for each date in date_list:

        build the dataframe for storing the result

        obtain the current_segment_pair for the specific trip_id and date from the segment_pair data

        obtain the segment_start sequence from the current_segment_pair data

        for each segment_start in the segment_start sequence:

            find the corresponding index in the stop_sequence

            find the index of the segment_end in the corresponding segment_pair

            if the indices of these two are i and i + 1:

                add the segment_pair into the new dataframe as the result

            else:

                use the indices to find all the skipped stops from the stop_sequence

                calculate the number of skipped travel duration within this segment

                use the average value as the travel duration and add the stop arrival time for each skipped stops

                add the segment_pair into the new dataframe as the result

Concatenate all the records as the result
```

## API data

For this step, we use the GTFS scheduled data and the historical data to obtain the necessary data for testing:
1. read the GTFS schedule data to obtain the stop sequence of the single trip.
2. read the historical data to obtain the current location, distance, etc.

The process for generating the api data for multiple dates and time point is below:

```
Generate the date list for api data

Generate the route set

for date in date_list:

    read the history file for the date

    for each route in route_set

        randomly pick four stops from the stop_list for the route (I didn't pick the first two and the last two stops in any routes to avoid the lack of data)

        for each stop in stops:

            Generate time list: 12:00:00, 12:05:00, 12:10:00, ..., 12:30:00 and 18:00:00, 18:05:00, 18:10:00, ..., 18:30:00.

            for each time point in the time list:

                generate current time according to the time point

                obtain the api_data as the records

Concatenate the records
```

The process to obtain the api data for the specific stop, time point, and the date is below:

```
Filtering the trip list according to the stop_id, direction_id, route_id, and the operation hour and obtain the selected_trips list

for each single_trip in selected_trips:

    Get the single_trip_stop_sequence according to the trip_id

    Get the single_trip_history according to the trip_id

    Loop in the single_trip_history to obtain the time segment containing the input time point

    Calculate the distance between the location of the two time point

    Calculate the travel_duration1 between the current time point and the previous time point

    Calculate the travel_duration2 between the containing segment_pair

    Use the ratio of the travel_duration1 and the travel_duration2 to estimate the dist_along_route(the distance from the initial stop to the current location)

    Store the data including the trip_id, date, time_of_day, dist_along_route, stop_num_from_call, etc.

Concatenate all the records
```

## Stop Distance Along Route

This step is to calculate the distance from the initial stop to each stops along the route.

Since the dist_along_route in the historical data is actually the corresponding distance from the initial stop to that stop, we only need to record these distances within each route.

Here, I used all the historical data from Jan 4th, 2016 to Apr 2nd, 2016 to calculate the data for two routes `X11` and `X14`.
