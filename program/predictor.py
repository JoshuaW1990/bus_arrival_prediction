"""
Predict the dataset with the dataset
"""


# import the modules
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt



# Read and divide the dataset
dataset = pd.read_csv('dataset.csv')
training_set = dataset[dataset.service_date < 20160125].reset_index()
test_set = dataset[dataset.service_date >= 20160125].reset_index()

train_X = training_set.as_matrix(columns=['weather', 'rush_hour', 'baseline_result', 'delay_current_trip', 'delay_prev_trip'])
train_Y = training_set.as_matrix(columns=['actual_arrival_time'])



test_X = test_set.as_matrix(columns=['weather', 'rush_hour', 'baseline_result', 'delay_current_trip', 'delay_prev_trip'])
test_Y = test_set.as_matrix(columns=['actual_arrival_time'])



# Use the linear model for regression

reg = linear_model.LinearRegression()
reg.fit(train_X, train_Y)
predict_Y = reg.predict(test_X)

result = pd.DataFrame(predict_Y, columns=['predict_arrival_time'])
result['actual_arrival_time'] = test_set['actual_arrival_time']

plt.style.use('ggplot')

result.plot(kind='scatter', x='actual_arrival_time', y='predict_arrival_time')
