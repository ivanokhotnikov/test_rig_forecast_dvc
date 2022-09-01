import gc
import logging
import os

import click
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)


@click.command()
@click.argument('local_raw_data_path', type=click.Path(exists=True))
@click.argument('local_interim_data_path', type=click.Path(exists=False))
def read_raw_local(local_raw_data_path, local_interim_data_path):
    """
    Reads raw data from the local raw data directory and saves the interim_data.csv to the interim data directory.

    Args:
    
        local_raw_data_path (str): Raw data directory
        local_interim_data_path (str): Interim data directory
    """
    final_df = pd.DataFrame()
    units = []
    for file in os.listdir(os.path.join(local_raw_data_path)):
        if 'RAW' in file:
            try:
                if file.endswith('.csv'):
                    current_df = pd.read_csv(os.path.join(
                        local_raw_data_path, file),
                                             header=0,
                                             index_col=False)
                elif file.endswith('.xlsx') or file.endswith('.xls'):
                    current_df = pd.read_excel(os.path.join(
                        local_raw_data_path, file),
                                               header=0,
                                               index_col=False)
            except:
                logging.info(f'Can\'t read {file}')
                continue
            if current_df is not None:
                logging.info(f'{file} was read!')
                name_list = file.split('-')
                try:
                    unit = np.uint8(name_list[0][-3:].lstrip('0D'))
                except ValueError:
                    unit = np.uint8(
                        name_list[0].split('_')[0][-3:].lstrip('0D'))
                units.append(unit)
                current_df['UNIT'] = unit
                current_df['TEST'] = np.uint8(units.count(unit))
                final_df = pd.concat((final_df, current_df), ignore_index=True)
                del current_df
                gc.collect()
    if not os.path.exists(local_interim_data_path):
        os.makedirs(local_interim_data_path)
    final_df.to_csv(os.path.join(local_interim_data_path, 'interim_data.csv'),
                    index=False)


if __name__ == '__main__':
    read_raw_local()
