import numpy as np
import pandas as pd


def add_time_features(df):
    """Adds duration, running second and running hours features to the dataframe, returns the modified dataframe.

    Args:
        df (pd.DataFrame): The dataframe to modify

    Returns:
        pd.DataFrame: The modified dataframe
    """
    df['DURATION'] = pd.to_timedelta(range(len(df)), unit='s')
    df['RUNNING_SECONDS'] = (pd.to_timedelta(range(
        len(df)), unit='s').total_seconds()).astype(np.uint64)
    df['RUNNING_HOURS'] = (df['RUNNING_SECONDS'] / 3600).astype(np.float64)
    return df


if __name__ == '__main__':
    add_time_features()
