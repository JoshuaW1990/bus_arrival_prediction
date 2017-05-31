import baseline
import build_dataset
import feature_selection
import model_selection
import cross_validation
import learning_curve
import group_learning
from sqlalchemy import create_engine
import pandas as pd
import os

# setting for path
# path for exporting data
save_path = '/Users/junwang/PycharmProjects/bus_arrival_prediction/implementation/example/'
# path for storing the prepared example dataset
example_path = '/Users/junwang/PycharmProjects/bus_arrival_prediction/example_output/preprocessed_data/'
# path for storing raw historical data
history_path = '/Users/junwang/PycharmProjects/bus_arrival_prediction/data/history/'
# path for storing GTFS data
gtfs_path = '/Users/junwang/PycharmProjects/bus_arrival_prediction/data/GTFS/gtfs/'


# read files
segment_df = pd.read_csv(example_path+'segment.csv')
api_data = pd.read_csv(example_path+'api_data.csv')
route_stop_dist = pd.read_csv(example_path+'route_stop_dist.csv')
trips = pd.read_csv(gtfs_path+'trips.txt')
full_history = pd.read_csv(example_path+'history.csv')
rush_hour = ('17:00:00', '20:00:00')
# database_engine = create_engine('postgresql://[username]:[userpassword]@localhost:5432/[databasename]', echo=False)

database_engine = None

dataset = pd.read_csv(example_path + 'dataset.csv')
weather_df = pd.read_csv(example_path + 'weather.csv')

# dataset = dataset[:100]

print "end"

"""
Example for baseline algorithms
"""
# baseline1
baseline1 = baseline.obtain_baseline1(segment_df, api_data, route_stop_dist, trips, full_history, 'baseline1_result', save_path + 'baseline/', database_engine)

# baseline2
baseline2 = baseline.obtain_baseline2(segment_df, api_data, route_stop_dist, trips, full_history, rush_hour, weather_df, 'baseline2_result', save_path + 'baseline/', database_engine)

# baseline3
baseline3 = baseline.obtain_baseline3(segment_df, api_data, route_stop_dist, trips, full_history, 'baseline3_result', save_path + 'baseline/', database_engine)

"""
Example for generating the dataset for models
"""
complete_dataset = build_dataset.obtain_dataset(20160104, api_data, segment_df, route_stop_dist, trips, full_history, weather_df, rush_hour, 'dataset', save_path + 'dataset/', database_engine)

"""
Example for feature selection
"""
feature_selection_result = feature_selection.run_feature_selection(dataset, 'feature_mse_compare', save_path + 'feature_selection/', database_engine)


"""
Example for model selection
"""
if not os.path.exists(save_path + 'model_selection/'):
    os.mkdir(save_path + 'model_selection/')
# solver function for neural network
time_result, mse_time, ratio_result, mse_ratio = model_selection.compare_models(dataset, model_selection.generate_nn_solver_ratio_result, model_selection.check_nn_solver_performance, save_path + 'model_selection/nn_solver/')

# activation function for neural network
time_result, mse_time, ratio_result, mse_ratio = model_selection.compare_models(dataset, model_selection.generate_nn_activation_ratio_result, model_selection.check_nn_activation_performance, save_path + 'model_selection/nn_activation/')

# kernel function for gaussian process
time_result, mse_time, ratio_result, mse_ratio = model_selection.compare_models(dataset, model_selection.generate_gaussian_ratio_result, model_selection.check_gaussian_performance, save_path + 'model_selection/gp_kernel/')

"""
Example for 5-fold cross validation
"""
mse_time_result, mse_ratio_result = cross_validation.cross_validation(dataset, 5, save_path + 'cross_validation/')

"""
Example of obtaining learning curve
"""
mse_time_result, mse_ratio_result = learning_curve.obtain_learning_curve(dataset, save_path + 'learning_curve/')

"""
Example of obtaining group learning
"""
mse_time_result, mse_ratio_result = group_learning.obtain_group_learning(dataset, save_path + 'group_learning/')
