import pandas as pd
import os
import time
import numpy as np


class Data:
    def __init__(self, csv):
        if isinstance(csv, str):
            if os.path.isfile(csv):
                df = pd.read_csv(csv)
            else:
                df = pd.DataFrame(csv)
        else:
            df = csv
        df = df[df.Speed > 0]
        df = df[df.Accuracy < 25]
        bearing_stats = get_column_stats(df, 'Bearing')
        acceleration_x_stats = get_column_stats(df, 'acceleration_x')
        acceleration_y_stats = get_column_stats(df, 'acceleration_y')
        acceleration_z_stats = get_column_stats(df, 'acceleration_z')
        gyro_x_stats = get_column_stats(df, 'gyro_x')
        gyro_y_stats = get_column_stats(df, 'gyro_y')
        gyro_z_stats = get_column_stats(df, 'gyro_z')
        speed_stats = get_column_stats(df, 'Speed')
        all_stats = {'acceleration_x': acceleration_x_stats,
                     'acceleration_y': acceleration_y_stats,
                     'acceleration_z': acceleration_z_stats,
                     'gyro_x': gyro_x_stats,
                     'gyro_y': gyro_y_stats,
                     'gyro_z': gyro_z_stats,
                     'Bearing': bearing_stats,
                     'Speed': speed_stats}

        additional_stats = {'max_total_acceleration_1': [],
                            'max_total_gyro_1': [],
                            'mean_total_acceleration_1': [],
                            'mean_max_diff_total_acceleration_1': []}

        for i, bookingID in enumerate(df.bookingID.unique()):
            temp_df = df[df.bookingID == bookingID]
            squared_accelerations = np.array(temp_df[['acceleration_x', 'acceleration_y', 'acceleration_z']]) ** 2
            squared_accelerations = squared_accelerations.sum(axis=1) ** 0.5
            max_total_acceleration = max(squared_accelerations)
            mean_total_acceleration = np.mean(squared_accelerations)
            additional_stats['max_total_acceleration_1'].append(max_total_acceleration)
            additional_stats['mean_total_acceleration_1'].append(mean_total_acceleration)
            additional_stats['mean_max_diff_total_acceleration_1'].append(max_total_acceleration - mean_total_acceleration)
            squared_gyros = np.array(temp_df[['gyro_x', 'gyro_y', 'gyro_z']]) ** 2
            additional_stats['max_total_gyro_1'].append(max(squared_gyros.sum(axis=1)) ** 0.5)
            print(i, 'complete.', end='\r')

        more_additional_stats = {'has_negative_acceleration_x_1': [],
                                 'has_negative_acceleration_y_1': [],
                                 'has_negative_acceleration_z_1': []}
        for i, bookingID in enumerate(df.bookingID.unique()):
            temp_df = df[df.bookingID == bookingID]
            more_additional_stats['has_negative_acceleration_x_1'].append(1 if min(temp_df.acceleration_x) < 0 else 0)
            more_additional_stats['has_negative_acceleration_y_1'].append(1 if min(temp_df.acceleration_y) < 0 else 0)
            more_additional_stats['has_negative_acceleration_z_1'].append(1 if min(temp_df.acceleration_z) < 0 else 0)
            print(i, 'complete.', end='\r')

        all_stats['Additional_stats'] = {**additional_stats, **more_additional_stats}
        df_features = pd.DataFrame()
        for stat in all_stats:
            for feature in all_stats[stat]:
                df_features[feature] = all_stats[stat][feature]
        df_features = df_features.drop(columns=['mean_max_diff_proportion_gyro_y_1', 'mean_max_diff_proportion_Bearing_1'])
        self.processed_data = df_features

    def get_processed_data(self):
        return self.processed_data


def max_change_per_second(dataframe, column):
    dataframe = dataframe.sort_values(by='second')
    previous_second = 0
    previous_column = 0
    max_change = 0
    for second, column in zip(dataframe.second, dataframe[column]):
        column_change = column - previous_column
        second_change = second - previous_second if second > previous_second else 1
        change = column_change / second_change
        max_change = change if abs(change) > abs(max_change) else max_change
        previous_second = second
        previous_column = column
    return max_change


def get_column_stats(df, column_name, ignore=None):
    start = time.time()
    unique_ids = df.bookingID.unique()
    column_stats = {}
    for i, bookingID in enumerate(unique_ids):
        column = np.array(df[df.bookingID == bookingID][column_name])
        if ignore and 'max' in ignore:
            pass
        else:
            if f'max_{column_name}_1' in column_stats:
                column_stats[f'max_{column_name}_1'].append(max(column))
            else:
                column_stats[f'max_{column_name}_1'] = [max(column)]
        if ignore and 'mean' in ignore:
            pass
        else:
            if f'mean_{column_name}_1' in column_stats:
                column_stats[f'mean_{column_name}_1'].append(np.mean(column))
            else:
                column_stats[f'mean_{column_name}_1'] = [np.mean(column)]
        if ignore and 'mean-max' in ignore:
            pass
        else:
            if f'mean_max_diff_{column_name}_1' in column_stats:
                column_stats[f'mean_max_diff_{column_name}_1'].append(np.array(max(column)) - np.array(np.mean(column)))
            else:
                column_stats[f'mean_max_diff_{column_name}_1'] = [np.array(max(column)) - np.array(np.mean(column))]
        if ignore and 'mean-max-proportion' in ignore:
            pass
        else:
            if f'mean_max_diff_proportion_{column_name}_1' in column_stats:
                column_stats[f'mean_max_diff_proportion_{column_name}_1'].append(np.array(max(column)) / np.array(np.mean(column)))
            else:
                column_stats[f'mean_max_diff_proportion_{column_name}_1'] = [np.array(max(column)) / np.array(np.mean(column))]
        if ignore and 'var' in ignore:
            pass
        else:
            if f'var_{column_name}_1' in column_stats:
                column_stats[f'var_{column_name}_1'].append(np.var(column))
            else:
                column_stats[f'var_{column_name}_1'] = [np.var(column)]
        if ignore and 'max_change' in ignore:
            pass
        else:
            if f'max_{column_name}_change_1' in column_stats:
                column_stats[f'max_{column_name}_change_1'].append(max_change_per_second(df, column_name))
            else:
                column_stats[f'max_{column_name}_change_1'] = [max_change_per_second(df, column_name)]
        print(i, 'done', end='\r')
    print(f'Time taken: {time.time()-start} seconds')
    return column_stats


if __name__ == '__main__':
    processed_data = Data('test_data.csv').get_processed_data()
    print(processed_data.shape)
