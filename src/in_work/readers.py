import gc
import io
import os
from typing import Optional, List

import gcsfs
import h5py
import numpy as np
import pandas as pd
import streamlit as st
from joblib import load
from tensorflow import keras

from features import (FEATURES_NO_TIME, FORECAST_FEATURES, LOCAL_DATA_PATH,
                      MODELS_PATH, PREDICTIONS_PATH, RAW_FORECAST_FEATURES,
                      TIME_STEPS, STORAGE_CLIENT)


class DataReader:

    @classmethod
    def get_processed_data(cls,
                           raw=False,
                           local=True,
                           features_to_read=RAW_FORECAST_FEATURES,
                           verbose=False):
        if local:
            if verbose:
                print(f'Reading processed local data from {LOCAL_DATA_PATH}\\')
            df = pd.read_csv(os.path.join(LOCAL_DATA_PATH,
                                          'forecast_data.csv'))
            df[FORECAST_FEATURES] = df[FORECAST_FEATURES].apply(
                pd.to_numeric, errors='coerce', downcast='float')
            if verbose:
                print('Reading done!')
            return df
        if not raw:
            if verbose:
                print(
                    f'Reading processed data from {LOCAL_DATA_PATH}\\processed\\'
                )
            df = pd.read_csv(os.path.join(LOCAL_DATA_PATH, 'processed',
                                          'combined_timed_data.csv'),
                             parse_dates=True,
                             infer_datetime_format=True,
                             dtype=dict(
                                 zip(FEATURES_NO_TIME,
                                     [np.float32] * len(FEATURES_NO_TIME))))
            df[['STEP', 'UNIT', 'TEST',
                'ARMANI']] = df[['STEP', 'UNIT', 'TEST',
                                 'ARMANI']].astype(np.uint8)
            df['TIME'] = pd.to_datetime(df['TIME'])
            df[' DATE'] = pd.to_datetime(df[' DATE'])
            df[FORECAST_FEATURES] = df[FORECAST_FEATURES].apply(
                pd.to_numeric, errors='coerce', downcast='float')
            if verbose:
                print('Reading done!')
            return df
        if verbose:
            print(f'Reading raw data from {LOCAL_DATA_PATH}\\raw\\')
        df = cls.read_all_raw_data(features_to_read=features_to_read)
        df = Preprocessor.remove_step_zero(df)
        df = Preprocessor.add_power_features(df)
        if verbose:
            print('Reading done!')
        return df

    @staticmethod
    def read_all_raw_data_from_gcs(raw=True):
        bucket = STORAGE_CLIENT.get_bucket('test_rig_data')
        final_df = pd.DataFrame()
        units = []
        loading_bar = st.progress(0)
        for idx, blob in enumerate(list(bucket.list_blobs(prefix='raw')), 1):
            loading_bar.progress(idx /
                                 len(list(bucket.list_blobs(prefix='raw'))))
            data_bytes = blob.download_as_bytes()
            current_df = None
            try:
                if blob.name.endswith('.csv'):
                    current_df = pd.read_csv(io.BytesIO(data_bytes),
                                             usecols=RAW_FORECAST_FEATURES,
                                             infer_datetime_format=True,
                                             index_col=False)
                elif blob.name.endswith('.xlsx') or blob.name.endswith('.xls'):
                    current_df = pd.read_excel(io.BytesIO(data_bytes),
                                               usecols=RAW_FORECAST_FEATURES,
                                               index_col=False)
            except:
                print(f'Can\'t read {blob.name}')
                continue
            print(f'{blob.name} has been read')
            if current_df is not None:
                final_df = pd.concat((final_df, current_df), ignore_index=True)
                del current_df
                gc.collect()
        try:
            final_df.sort_values(by=[' DATE', 'TIME'],
                                 inplace=True,
                                 ignore_index=True)
        except:
            print('Can\'t sort dataframe')
        return final_df, units

    @staticmethod
    def read_processed_data_from_gcs():
        bucket = STORAGE_CLIENT.get_bucket('test_rig_data')
        bucket.blob('processed/forecast_data.csv')
        blob = bucket.get_blob('processed/forecast_data.csv')
        forecast_data_bytes = blob.download_as_bytes()
        df = pd.read_csv(io.BytesIO(forecast_data_bytes))
        df[FORECAST_FEATURES] = df[FORECAST_FEATURES].apply(pd.to_numeric,
                                                            errors='coerce',
                                                            downcast='float')
        return df

    @staticmethod
    def check_if_in_bucket(csv_file):
        try:
            bucket = STORAGE_CLIENT.get_bucket('test_rig_data')
        except:
            bucket = STORAGE_CLIENT.get_bucket('rig_data')
        raw_folder_content = {
            blob.name[4:]
            for blob in list(bucket.list_blobs(prefix='raw'))
        }
        return bucket, csv_file.name in raw_folder_content

    @classmethod
    def read_newcoming_data(cls, csv_file):
        bucket, in_bucket = cls.check_if_in_bucket(csv_file)
        if in_bucket:
            st.write(f'{csv_file.name} in the GCS bucket {bucket.name}')
        else:
            st.write(f'{csv_file.name} not in the GCS bucket {bucket.name}')
            blob = bucket.blob(f'raw/{csv_file.name}')
            blob.upload_from_file(csv_file, content_type='text/csv')
            st.write(
                f'{csv_file.name} uploaded to the GCS bucket {bucket.name}')
            updated_df = cls.get_processed_data_from_gcs(raw=True)
            blob = bucket.blob('processed/forecast_data.csv')
            blob.upload_from_string(updated_df.to_csv(index=False),
                                    content_type='text/csv')
            st.write(
                f'{csv_file.name} uploaded to the GCS bucket {bucket.name}')
        blob = bucket.get_blob(f'raw/{csv_file.name}')
        df = pd.read_csv(io.BytesIO(blob.download_as_bytes()),
                         usecols=RAW_FORECAST_FEATURES,
                         index_col=False)
        df[FEATURES_NO_TIME] = df[FEATURES_NO_TIME].apply(pd.to_numeric,
                                                          errors='coerce',
                                                          downcast='float')
        df = df.dropna(axis=0)
        name_list = csv_file.name.split('-')
        try:
            unit = np.uint8(name_list[0][-3:].lstrip('0D'))
        except ValueError:
            unit = np.uint8(name_list[0].split('_')[0][-3:].lstrip('0D'))
        df['UNIT'] = unit
        df['STEP'] = df['STEP'].astype(np.uint8)
        #TODO retesting case needs to be addressed
        df['TEST'] = np.uint8(1)
        return df

    @staticmethod
    def read_predictions(file):
        return pd.read_csv(os.path.join(PREDICTIONS_PATH, file),
                           index_col=False)


class Preprocessor:

    @staticmethod
    def preprocess(final_df):
        final_df = Preprocessor.remove_step_zero(final_df)
        final_df['TIME'] = pd.to_datetime(
            range(len(final_df)),
            unit='s',
            origin=f'{final_df[" DATE"].min()} 00:00:00')
        final_df['DURATION'] = pd.to_timedelta(range(len(final_df)), unit='s')
        final_df['RUNNING SECONDS'] = (pd.to_timedelta(
            range(len(final_df)), unit='s').total_seconds()).astype(np.uint64)
        final_df['RUNNING HOURS'] = (final_df['RUNNING SECONDS'] /
                                     3600).astype(np.float64)
        final_df = Preprocessor.add_power_features(final_df)
        bucket = STORAGE_CLIENT.get_bucket('test_rig_data')
        bucket.blob('processed/forecast_data.csv').upload_from_string(
            final_df.to_csv(index=False), content_type='text/csv')
        return final_df

    @staticmethod
    def create_sequences(values, lookback=TIME_STEPS, inference=False):
        X, Y = [], []
        for i in range(lookback, len(values)):
            X.append(values[i - lookback:i])
            Y.append(values[i])
        if inference:
            return np.stack(X)
        return np.stack(X), np.stack(Y)

    @staticmethod
    def clean_down_data(df):
        df[FEATURES_NO_TIME] = df[FEATURES_NO_TIME].apply(pd.to_numeric,
                                                          errors='coerce',
                                                          downcast='float')
        df.dropna(inplace=True)

    def add_unit_features(units, file, df):
        name_list = file.split('-')
        try:
            unit = np.uint8(name_list[0][-3:].lstrip('0D'))
        except ValueError:
            unit = np.uint8(name_list[0].split('_')[0][-3:].lstrip('0D'))
        units.append(unit)
        df['ARMANI'] = 1 if name_list[0][3] == '2' else 0
        df['ARMANI'] = df['ARMANI'].astype(np.uint8)
        df['UNIT'] = unit
        df['TEST'] = np.uint8(units.count(unit))

    @staticmethod
    def add_time_features(df):
        df['TIME'] = pd.to_datetime(df['TIME'], errors='coerce').dt.time
        df[' DATE'] = pd.to_datetime(df[' DATE'], errors='coerce')
        # df['TIME'] = pd.to_datetime(range(len(df)),
        #                             unit='s',
        #                             origin=f'{df[" DATE"].min()} 00:00:00')
        df['DURATION'] = pd.to_timedelta(range(len(df)), unit='s')
        df['RUNNING SECONDS'] = (pd.to_timedelta(range(
            len(df)), unit='s').total_seconds()).astype(np.uint64)
        df['RUNNING HOURS'] = (df['RUNNING SECONDS'] / 3600).astype(np.float64)

    @staticmethod
    def add_power_features(df):
        df['DRIVE POWER'] = (df['M1 SPEED'] * df['M1 TORQUE'] * np.pi / 30 /
                             1e3).astype(np.float32)
        df['LOAD POWER'] = abs(df['D1 RPM'] * df['D1 TORQUE'] * np.pi / 30 /
                               1e3).astype(np.float32)
        df['CHARGE MECH POWER'] = (df['M2 RPM'] * df['M2 Torque'] * np.pi /
                                   30 / 1e3).astype(np.float32)
        df['CHARGE HYD POWER'] = (df['CHARGE PT'] * 1e5 * df['CHARGE FLOW'] *
                                  1e-3 / 60 / 1e3).astype(np.float32)
        df['SERVO MECH POWER'] = (df['M3 RPM'] * df['M3 Torque'] * np.pi / 30 /
                                  1e3).astype(np.float32)
        df['SERVO HYD POWER'] = (df['Servo PT'] * 1e5 * df['SERVO FLOW'] *
                                 1e-3 / 60 / 1e3).astype(np.float32)
        df['SCAVENGE POWER'] = (df['M5 RPM'] * df['M5 Torque'] * np.pi / 30 /
                                1e3).astype(np.float32)
        df['MAIN COOLER POWER'] = (df['M6 RPM'] * df['M6 Torque'] * np.pi /
                                   30 / 1e3).astype(np.float32)
        df['GEARBOX COOLER POWER'] = (df['M7 RPM'] * df['M7 Torque'] * np.pi /
                                      30 / 1e3).astype(np.float32)
        return df

    @staticmethod
    def remove_step_zero(df):
        return df.drop(df[df['STEP'] == 0].index,
                       axis=0).reset_index(drop=True)

    @staticmethod
    def get_warm_up_steps(df):
        return df[(df['STEP'] >= 1) & (df['STEP'] <= 11)]

    @staticmethod
    def get_break_in_steps(df):
        return df[(df['STEP'] >= 12) & (df['STEP'] <= 22)]

    @staticmethod
    def get_performance_check_steps(df):
        return df[(df['STEP'] >= 23) & (df['STEP'] <= 33)]


class ModelReader:

    @staticmethod
    def read_model(model):
        if 'scaler' in model:
            return load(os.path.join(MODELS_PATH, model + '.joblib'))
        return keras.models.load_model(os.path.join(MODELS_PATH,
                                                    model + '.h5'))

    @staticmethod
    def read_model_from_gcs(model):
        bucket = STORAGE_CLIENT.get_bucket('models_forecasting')
        if 'scaler' in model:
            blob = bucket.get_blob(model + '.joblib')
            data_bytes = blob.download_as_bytes()
            return load(io.BytesIO(data_bytes))
        fs = gcsfs.GCSFileSystem()
        with fs.open(f'gs://models_forecasting/{model}.h5',
                     'rb') as model_file:
            model_gcs = h5py.File(model_file, 'r')
            return keras.models.load_model(model_gcs)


if __name__ == '__main__':
    raw_data_df = DataReader.get_processed_data(
        raw=True, features_to_read=RAW_FORECAST_FEATURES)
    combined_data_df = DataReader.get_processed_data(
        raw=False, features_to_read=RAW_FORECAST_FEATURES)
