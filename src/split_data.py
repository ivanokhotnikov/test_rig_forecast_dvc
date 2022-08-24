import click
import pandas as pd
from dvc.api import params_show


@click.command()
def split_data() -> None:
    """
    Split processed data into train and test data.

    Returns
    -------
    None
    """
    params = params_show()
    local_processed_data_path = params['processed_data']
    local_train_data_path = params['train_data']
    local_test_data_path = params['test_data']
    train_data_size = params['train_split']
    processed_df = pd.read_csv(local_processed_data_path, index_col=False)
    train_df = processed_df.loc[:int(len(processed_df) * train_data_size)]
    test_df = processed_df.loc[int(len(processed_df) * train_data_size):]
    train_df.to_csv(local_train_data_path, index=False)
    test_df.to_csv(local_test_data_path, index=False)


if __name__ == '__main__':
    split_data()
