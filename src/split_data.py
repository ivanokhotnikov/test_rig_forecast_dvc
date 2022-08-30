import os

import click
import pandas as pd
from dvc.api import params_show


@click.command()
@click.argument('local_processed_data_path', type=click.Path(exists=True))
@click.argument('local_train_data_path', type=click.Path(exists=False))
@click.argument('local_test_data_path', type=click.Path(exists=False))
def split_data(local_processed_data_path, local_train_data_path,
               local_test_data_path):
    """
    Split processed data into train and test data.

    Returns
    -------
    None
    """
    params = params_show()
    train_data_size = params['split_data']['train_data_size']
    processed_df = pd.read_csv(os.path.join(local_processed_data_path,
                                            'processed_data.csv'),
                               index_col=False)
    train_df = processed_df.loc[:int(len(processed_df) * train_data_size)]
    test_df = processed_df.loc[int(len(processed_df) * train_data_size):]
    train_df.to_csv(os.path.join(local_train_data_path, 'train_data.csv'),
                    index=False)
    test_df.to_csv(os.path.join(local_test_data_path, 'test_data.csv'),
                   index=False)


if __name__ == '__main__':
    split_data()
