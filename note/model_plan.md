# model plan

The ML models for bus arrival time prediction can be divided into two different pars:
1. Input feature selection
2. Training model selection
3. Generate the weather information based on the hours.

## Input

For now, we only used the average travel duration for each segment pair in all the bus system in NYC as the input, which is quite simple and unreliable considering the effects like the weather, rush hour, etc.

Here are following different approaches which might improve the reliability of the input feature:
1. Use the median instead of the average value of the travel duration
2. Consider the weather, rush hour, etc when calculating the travel duration for the segment pair
3. Use the specific travel duration instead of the median or the mean of the travel duration. For example, when calculating the estimated arrival time, consider the travel duration for the segment of the 

## Model
- Linear regression model
- KNN model
