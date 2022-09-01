import gc
import io
import logging

import click
import numpy as np
import pandas as pd
from google.cloud import storage

logging.basicConfig(level=logging.INFO)
STORAGE_CLIENT = storage.Client()


@click.command()
@click.argument('bucket_name', type=click.STRING)
def read_raw_gcs(bucket_name):
    bucket = STORAGE_CLIENT.get_bucket(bucket_name)
    final_df = pd.DataFrame()
    units = []
    for blob in bucket.list_blobs(prefix='raw'):
        if 'RAW' in blob.name:
            data_bytes = blob.download_as_bytes()
            try:
                if blob.name.endswith('.csv'):
                    current_df = pd.read_csv(io.BytesIO(data_bytes),
                                             header=0,
                                             index_col=False)
                elif blob.name.endswith('.xlsx') or blob.name.endswith('.xls'):
                    current_df = pd.read_excel(io.BytesIO(data_bytes),
                                               header=0,
                                               index_col=False)
            except:
                logging.info(f'Can\'t read {blob.name}')
                continue
            if current_df is not None:
                logging.info(f'{blob.name} was read!')
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
    blob = bucket.blob('interim/interim_data.csv')
    blob.upload_from_string(final_df.to_csv(index=False),
                            content_type='text/csv')


if __name__ == '__main__':
    read_raw_gcs()
