import os
import click
import pandas as pd

from features import FEATURES_NO_TIME, RAW_FORECAST_FEATURES
from features.add_power_features import add_power_features
from features.add_time_features import add_time_features
from features.remove_step_zero import remove_step_zero


@click.command()
@click.argument('local_interim_data_path', type=click.Path(exists=True))
@click.argument('local_processed_data_path', type=click.Path(exists=False))
def build_features(local_interim_data_path, local_processed_data_path):
    """
    Read the interim data, build features (float down casting, removes NaNs and the step zero data, calculates and adds to the processed data the power and time features), saves the processed data.

    Args:
        local_interim_data_path (str): Interim data directory
        local_processed_data_path (str): Processed data directory
    """
    interim_df = pd.read_csv(os.path.join(local_interim_data_path,
                                          'interim_data.csv'),
                             usecols=RAW_FORECAST_FEATURES,
                             header=0,
                             index_col=False,
                             low_memory=False)
    interim_df[FEATURES_NO_TIME] = interim_df[FEATURES_NO_TIME].apply(
        pd.to_numeric, errors='coerce', downcast='float')
    interim_df.dropna(axis=0, inplace=True)
    interim_df = remove_step_zero(interim_df)
    interim_df = add_power_features(interim_df)
    interim_df = add_time_features(interim_df)
    interim_df.columns = interim_df.columns.str.lstrip()
    interim_df.columns = interim_df.columns.str.replace(' ', '_')
    interim_df.to_csv(os.path.join(local_processed_data_path,
                                   'processed_data.csv'),
                      index=False)


if __name__ == '__main__':
    build_features()
