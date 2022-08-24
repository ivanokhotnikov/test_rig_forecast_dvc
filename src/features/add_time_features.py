import numpy as np
import pandas as pd


def add_time_features(df):
    """
    Add time features to dataframe.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe to add time features to.

    Returns
    -------
    pandas.DataFrame
        Dataframe with time features added.
    """
    # df['TIME'] = pd.to_datetime(df['TIME'], errors='coerce').dt.time
    # df[' DATE'] = pd.to_datetime(df[' DATE'], errors='coerce')
    # # df['TIME'] = pd.to_datetime(range(len(df)),
    # #                             unit='s',
    # #                             origin=f'{df[" DATE"].min()} 00:00:00')
    df['DURATION'] = pd.to_timedelta(range(len(df)), unit='s')
    df['RUNNING SECONDS'] = (pd.to_timedelta(range(
        len(df)), unit='s').total_seconds()).astype(np.uint64)
    df['RUNNING HOURS'] = (df['RUNNING SECONDS'] / 3600).astype(np.float64)
    return df


if __name__ == '__main__':
    add_time_features()
