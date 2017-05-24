# ReadMe

## What is this repo about

This is a final project in Tufts University and the aim of the project is to set up a reasonable approach to obtain, preprocess the dataset from the available online database. Then assess the performance with different models and compare their result. Anyone who wants to know more detail can download the [complete report](https://github.com/JoshuaW1990/bus_arrival_prediction/blob/master/final_report.pdf) here.

This repo can be divided into two different parts:

* preprocess
* model implementation

For preprocess part, all the code can be found in the directory named `preprocess` and a simple example script is also provided. Users can feel free to use that code and example if need.





## Who would be interested in this repo

As a final project in the university, this project might not be enought for detailed application. However, it still might be useful for following users:

1. Users who are currently working on bus arrival time prediction related project
2. Users who want to find a more realistic example to understand the application of machine learning

If you are a user in the first group, based on the different strategies in feature choices and output choices, the implementation part might vary a lot. Thus, you might be more interested in the preprocess part. However, if you want to know some detail about my feature and output choices, feel free to read or test my code in implementation part.

If you are a user in the second group, I suggest to understand and try the code in both of the preprocess and implementation part, because the preprocess part represents the data cleaning step and the implementation part represents the model implementation and selection step. Both of these two steps are critical for any machine learning learners.

## How to use repo

### Prepare Development Environment

1. Download the repo

```
git clone https://github.com/JoshuaW1990/bus_arrival_prediction
```

2. Prepare python2.7

Recommend to use `virtualenv` for python.

3. Install the required libraries

```
pip install -r requirements.txt
```


### Preprocess

In preprocess part, it provides following functions: `obtain_weather`, `download_history_file`, `obtain_history`, `obtain_route_stop_dist`, `obtain_segment`, `obtain_api_data`.

Among these functions, `download_history_file` will download the compressed historical data into required path. Users need to decompress these historical data before using `obtain_history` function.

All the other functions will generate the corresponding data table. Considering users might choose different way to store the dataset, these function also provide two strategies for storing the dataset: In all the functions with prefix `obtain_`, users can choose to provide a `save_path` or `engine`. 

* If the user provides the `save_path` to indicate the directory of the file to store data, these functions will export the data into the file according to the `save_path`.
* If the user provides the `engine` to connect the database, these functions will write the data into the database.
* If the user doesn't provide any of them, these functions will neither export the file by directory nor write the data into dabase by `engine`.

More examples can be found in `example.py` file. 

**Note**:
For large data(>100 MB), the efficiency of the functions `pandas.to_sql` and `pandas.read_sql` is very slow. Since in all of these functions, `pandas.to_sql` is called, users need to pay attention to the size of the data before exporting them into database.

### Implementation


## Dependencies

**Python**
Python==2.7.12

**Library**
```
pandas==0.19.1
numpy==1.11.2
scikit-learn==0.18.1
scipy==0.18.1
GPy==1.6.1
matplotlib==1.5.3
```

These libraries and versions are also listed in the file `requirements.txt`.