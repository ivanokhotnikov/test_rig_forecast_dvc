import gc
import io

import click
import numpy as np
import pandas as pd

from google.cloud import storage

STORAGE_CLIENT = storage.Client()


@click.command()
@click.argument('bucket_name',
                type=str,
                default='test_rig_data',
                help='Name of bucket')
@click.option('-v',
              '--verbose',
              is_flag=True,
              default=True,
              help='Print verbose output')
def read_raw_gcs(bucket_name, verbose):
    """
    Read raw data from Google Cloud Storage.

    Parameters
    ----------
    bucket_name : str
        Name of bucket
    verbose : bool
        Print verbose output

    Returns
    -------
    None
    """
    bucket = STORAGE_CLIENT.get_bucket(bucket_name)
    if verbose: print(f'Reading raw data from gs:\\{bucket.name}\\')
    final_df = pd.DataFrame()
    units = []
    for blob in bucket.list_blobs(prefix='raw'):
        if 'RAW' in blob.name:
            if verbose: print(f'Reading {blob.name}')
            data_bytes = blob.download_as_bytes()
            try:
                if blob.name.endswith('.csv'):
                    current_df = pd.read_csv(io.BytesIO(data_bytes),
                                             header=0,
                                             infer_datetime_format=True,
                                             index_col=False)
                elif blob.name.endswith('.xlsx') or blob.name.endswith('.xls'):
                    current_df = pd.read_excel(io.BytesIO(data_bytes),
                                               header=0,
                                               index_col=False)
            except:
                print(f'Can\'t read {blob.name}')
                continue
            if current_df is not None:
                name_list = blob.name.split('-')
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
    if verbose: print('Reading done!')
    blob = bucket.blob('interim/interim_data.csv')
    blob.upload_from_string(final_df.to_csv(index=False),
                            content_type='text/csv')
    if verbose: print('Interim data saved!')


if __name__ == '__main__':
    read_raw_gcs()
