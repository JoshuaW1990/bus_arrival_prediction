"""
Split the dataset according to the actual arrival time(which can indicate the distance between the current location of the vehicle and the targe stop). Then learn the subsets one by one.
"""

# import package
import toolbox
import os
import pandas as pd


def generate_dataset_list(full_dataset):
    """
    Generate the list of dataframe for testing
    :param full_dataset: 
    :return: 
    """
    dataset_list = []
    for time in range(0, 6000, 1000):
        sm_time = time
        bg_time = time + 500
        current_dataset = full_dataset[(full_dataset['actual_arrival_time'] >= sm_time) & (full_dataset['actual_arrival_time'] < bg_time)]
        dataset_list.append(current_dataset)
    return dataset_list


#################################################################################################################
#                                    main function                                                              #
#################################################################################################################


def obtain_group_learning(origin_dataset, save_path=None):
    """
    Split the dataset according to the actual arrival time(which can indicate the distance between the current location of the vehicle and the targe stop). Then learn the subsets one by one.
    
    :param origin_dataset: the dataframe for dataset table
    :param save_path: the path to export result
    :return: 
    """
    dataset = toolbox.preprocess_dataset(origin_dataset)

    # total_fold = 5
    dataset = dataset.sample(frac=1).reset_index(drop=True)

    mse_time_result = pd.DataFrame(
        columns=['baseline', 'single_linear_regression', 'single_SVM', 'single_NN', 'single_GP', 'MTL_GP',
                 'multiple_linear_regression', 'multiple_SVM', 'multiple_NN', 'multiple_GP', 'bin_number'])
    mse_ratio_result = pd.DataFrame(
        columns=['baseline', 'single_linear_regression', 'single_SVM', 'single_NN', 'single_GP', 'MTL_GP',
                 'multiple_linear_regression', 'multiple_SVM', 'multiple_NN', 'multiple_GP', 'bin_number'])

    dataset_list = generate_dataset_list(dataset)

    for bin_number, item in enumerate(dataset_list):
        print bin_number
        ratio_result, X_train_list, X_test_list, output_train_list, output_test_list, y_train_list, y_test_list = toolbox.single_shape_learning(item)
        if ratio_result is None:
            continue
        try:
            ratio_result = toolbox.multiple_shape_learning(ratio_result, X_train_list, y_train_list, X_test_list)
        except:
            continue
        time_result, mse_time, ratio_result, mse_ratio = toolbox.check_performance(output_test_list, ratio_result)

        mse_time_result.loc[len(mse_time_result)] = [mse_time['baseline'], mse_time['single_linear_regression'],
                                                     mse_time['single_SVM'], mse_time['single_NN'],
                                                     mse_time['single_GP'], mse_time['MTL_GP'],
                                                     mse_time['multiple_linear_regression'], mse_time['multiple_SVM'],
                                                     mse_time['multiple_NN'], mse_time['multiple_GP'], bin_number]

        mse_ratio_result.loc[len(mse_ratio_result)] = [mse_ratio['ratio_baseline'],
                                                       mse_ratio['single_linear_regression'], mse_ratio['single_SVM'],
                                                       mse_ratio['single_NN'], mse_ratio['single_GP'],
                                                       mse_ratio['MTL_GP'], mse_ratio['multiple_linear_regression'],
                                                       mse_ratio['multiple_SVM'], mse_ratio['multiple_NN'],
                                                       mse_ratio['multiple_GP'], bin_number]

    if save_path is not None:
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        mse_time_result.to_csv(save_path + 'mse_time.csv')
        mse_ratio_result.to_csv(save_path + 'mse_ratio.csv')

    return mse_time_result, mse_ratio_result
