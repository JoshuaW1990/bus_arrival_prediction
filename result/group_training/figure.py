# import modules
import pandas as pd
import matplotlib.pyplot as plt



plt.style.use('ggplot')

dataset = pd.read_csv('bin_chart.csv')
new_dataset = dataset.drop(['baseline'], axis=1)
figure = new_dataset.plot.bar(x='actual_arrival_time')
figure.set_xlabel('actual arrival time/sec')
figure.set_ylabel('MSE of ratio')
# fig = figure.get_figure()
plt.show()
