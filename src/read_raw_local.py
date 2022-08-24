import gc
import logging
import os

import click
import numpy as np
import pandas as pd
from dvc.api import params_show


@click.command()
def read_raw_local() -> None:
    """
    Read raw data from local disk.

    
    Returns
    -------
    None
    """
    final_df = pd.DataFrame()
    units = []
    params = params_show()
    local_raw_data_path = params['raw_data']
    local_interim_data = params['interim_data']
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
                name_list = file.split('-')
                try:
                    unit = np.uint8(name_list[0][-3:].lstrip('0D'))
                except ValueError:
                    unit = np.uint8(
                        name_list[0].split('_')[0][-3:].lstrip('0D'))
                units.append(unit)
                current_df['ARMANI'] = 1 if name_list[0][3] == '2' else 0
                current_df['UNIT'] = unit
                current_df['TEST'] = np.uint8(units.count(unit))
                final_df = pd.concat((final_df, current_df), ignore_index=True)
                del current_df
                gc.collect()
    if not os.path.exists(local_interim_data[:-17]):
        os.makedirs(local_interim_data[:-17])
    final_df.to_csv(local_interim_data, index=False)


if __name__ == '__main__':
    read_raw_local()
