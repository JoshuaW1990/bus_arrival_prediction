# import modules
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error as MSE
import json


plt.style.use('ggplot')

dataset = pd.read_csv('learning_curve.csv')
dataset = dataset[dataset.single_linear_regression < 1]
figure = dataset.plot(x='sample_size', marker='x', title='learning curve of ratio with increase of dataset size')
figure.set_xlabel('Dataset Size')
figure.set_ylabel('MSE of ratio')
fig = figure.get_figure()
plt.show(fig)
