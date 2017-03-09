# ReadMe

https://joshuaw1990.github.io/2017/03/08/Project-Overview-bus-arrival-time-prediction/#more
This is the advanced project for the master program in Tufts University.


## Project Introduction

With the widely application of the GPS and the machine learning, many public transportation systems provide the real-time prediction feature for users to know when the bus will arrive a specific stop. For example, **next bus** can listen to the current location and the current time of the user from cellphone or laptop, then it will provide the predicted bus arrival time for all the stops nearby.

The aim of this project is to build a real-time bus prediction system based on historical data and compare the performance with the available prediction system like **next bus**.

## Project Review

The project has been divided into several phases for now:

1. Data collection
2. Demo analysis with simple baseline algorithm


Current step: Demo analysis with simple baseline algorithm

## Data Collection

The file is composed by two different types of data:

- historical data
- scheduled data

The scheduled data described the relation between the routes, trips and the stops. The historical data records the location of all the operated bus every 1.5 minutes for every day. Through the `trip_id`, all the records in the historical data can be found the relational information like the route, stop_sequence, etc from the scheduled data.

### Download file

The format of the scheduled data is [GTFS](https://developers.google.com/transit/gtfs/). They can be obtained from the [transitfeeds](http://transitfeeds.com/). The historical data is obtained from the [NYC Transit Data Archive](http://data.mytransit.nyc.s3.amazonaws.com/README.HTML).

In order to consider the weather effects, we also need retrieve the weather information for the corresponding date. The weather information can be extracted from [wunderground](http://api.wunderground.com/api/).

### Preprocess data

To obtain the necessary data we needed for the data analysis, we need to clean and extract the data from the original ones. The data quality for the GTFS and the weather data is good, but considering the precision and the bugs in the historical data, we need to pay more attention when extracting the data from the historical data.

In order to make our analysis easier, several different types of data listed here:

1. `weather.csv` This file includes the weather(rainy, snowy, or sunny) for a specific date.
2. `data.csv` This file includes the travel duration for each segment pair in each trip at each day.
3. `route_stop_dist.csv` This file records the distance between the specific stop and the initial stop for all stops in different routes according to the route_id.
4. `average_segment_travel_duration.csv` This file calculates the average travel duration for each specific segment from `data.csv` without considering the date, trip and time.
5. `api_data.csv` This file obtains the api information like the `dist_along_route`, `stop_num_from_call`, etc from the historical data and the scheduled data. In this way, historical data can be used as the test for assessing the performance of the model

## Demo analysis with simple algorithm

For predicting the arrival time, the travel duration for the segment pairs are considered, such that the prediction process for a specific route will be able to consider the input from other routes. In this way, it is actually a multi-task learning process, which might improve the performance of the model.

Two demo baseline will be implemented:
1. without considering the weather, rush hours, etc.
2. Considering the weather, rush hours, etc.

## Test of demo analysis

MSE will be used to describe the performance of the algorithms.

## Traditional algorithm

Traditional algorithms for predicting the bus arrival time including the SVM, ANN, or Kalman filter algorithm. It might be necessary to compare the performance with their methods and ours.
