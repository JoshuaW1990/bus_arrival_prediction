"""
Generate the learning curve of different models
"""

# import package
import toolbox
import os
import pandas as pd


def obtain_learning_curve(origin_dataset, save_path=None):
    """
    obtain the dataframe for the learning curve
    
    :param origin_dataset: the dataframe for dataset table
    :param save_path: the path to export result
    :return: the dataframe for the ratio and the time after prediction in the learning curve
    """

    dataset = toolbox.preprocess_dataset(origin_dataset)

    # dataset_list = generate_dataset_list(dataset)

    mse_time_result = pd.DataFrame(
        columns=['baseline', 'single_linear_regression', 'single_SVM', 'single_NN', 'single_GP', 'MTL_GP',
                 'multiple_linear_regression', 'multiple_SVM', 'multiple_NN', 'multiple_GP', 'sample_size',
                 'num_shapes'])
    mse_ratio_result = pd.DataFrame(
        columns=['baseline', 'single_linear_regression', 'single_SVM', 'single_NN', 'single_GP', 'MTL_GP',
                 'multiple_linear_regression', 'multiple_SVM', 'multiple_NN', 'multiple_GP', 'sample_size',
                 'num_shapes'])

    dataset = dataset.sample(frac=1).reset_index(drop=True)
    for count in range(0, len(dataset), len(dataset) / 11):
        if count <= 0:
            continue
        print count
        item = dataset[:count]
        shape_set = set(item.shape_id)
        ratio_result, X_train_list, X_test_list, output_train_list, output_test_list, y_train_list, y_test_list = toolbox.single_shape_learning(item)
        if ratio_result is None:
            continue
        ratio_result = toolbox.multiple_shape_learning(ratio_result, X_train_list, y_train_list, X_test_list)
        time_result, mse_time, ratio_result, mse_ratio = toolbox.check_performance(output_test_list, ratio_result)
        mse_time_result.loc[len(mse_time_result)] = [mse_time['baseline'], mse_time['single_linear_regression'],
                                                     mse_time['single_SVM'], mse_time['single_NN'],
                                                     mse_time['single_GP'], mse_time['MTL_GP'],
                                                     mse_time['multiple_linear_regression'], mse_time['multiple_SVM'],
                                                     mse_time['multiple_NN'], mse_time['multiple_GP'], len(item),
                                                     len(shape_set)]
        mse_ratio_result.loc[len(mse_ratio_result)] = [mse_ratio['ratio_baseline'],
                                                       mse_ratio['single_linear_regression'], mse_ratio['single_SVM'],
                                                       mse_ratio['single_NN'], mse_ratio['single_GP'],
                                                       mse_ratio['MTL_GP'], mse_ratio['multiple_linear_regression'],
                                                       mse_ratio['multiple_SVM'], mse_ratio['multiple_NN'],
                                                       mse_ratio['multiple_GP'], len(item), len(shape_set)]

    if save_path is not None:
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        mse_time_result.to_csv(save_path + 'mse_time_result.csv')
        mse_ratio_result.to_csv(save_path + 'mse_ratio_result.csv')

    return mse_time_result, mse_ratio_result





