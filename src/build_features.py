import click
import pandas as pd
from dvc.api import params_show

from features import FEATURES_NO_TIME, RAW_FORECAST_FEATURES
from features.add_power_features import add_power_features
from features.add_time_features import add_time_features
from features.remove_step_zero import remove_step_zero


@click.command()
def build_features() -> None:
    """
    Build features from interim data.

    Returns
    -------
    None
    """
    params = params_show()
    local_interim_data_path = params['interim_data']
    local_processed_data_path = params['processed_data']
    interim_df = pd.read_csv(local_interim_data_path,
                             usecols=RAW_FORECAST_FEATURES,
                             header=0,
                             index_col=False)
    interim_df[FEATURES_NO_TIME] = interim_df[FEATURES_NO_TIME].apply(
        pd.to_numeric, errors='coerce', downcast='float')
    interim_df.dropna(axis=0, inplace=True)
    interim_df = remove_step_zero(interim_df)
    interim_df = add_power_features(interim_df)
    interim_df = add_time_features(interim_df)
    interim_df.to_csv(local_processed_data_path, index=False)


if __name__ == '__main__':
    build_features()
