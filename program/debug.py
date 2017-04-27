"""

Used for debug

Currently check the abnormal route in route_stop_dist
"""


import pandas as pd

history = pd.read_csv('preprocessed_complete_history.csv')
history = history[history.next_stop_id != '\N']
history.to_csv('preprocessed_complete_history.csv')

