import json
import os

import click
import keras
import pandas as pd
from dvc.api import params_show
from joblib import dump
from sklearn.preprocessing import MinMaxScaler

from features import FORECAST_FEATURES
from features.create_sequences import create_sequences


@click.command()
@click.argument('local_train_data_path', type=click.Path(exists=True))
@click.argument('local_models_path', type=click.Path(exists=False))
@click.argument('local_metrics_path', type=click.Path(exists=False))
def train(local_train_data_path, local_models_path, local_metrics_path):
    """
    Train the model.

    Returns
    -------
    None
    """
    params = params_show()
    train_df = pd.read_csv(os.path.join(local_train_data_path,
                                        'train_data.csv'),
                           index_col=False)
    if not os.path.exists(local_models_path):
        os.makedirs(local_models_path)
    for feature in FORECAST_FEATURES:
        model = f'RNN_{feature}'
        train_data = train_df[feature].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_train = scaler.fit_transform(train_data)
        dump(scaler, os.path.join(local_models_path, f'{model}_scaler.joblib'))
        x_train, y_train = create_sequences(
            scaled_train,
            lookback=params['model']['look_back'],
            inference=False)
        forecaster = keras.models.Sequential()
        forecaster.add(
            keras.layers.LSTM(params['model']['lstm_units'],
                              input_shape=(x_train.shape[1], x_train.shape[2]),
                              return_sequences=False))
        forecaster.add(keras.layers.Dense(1))
        forecaster.compile(loss=keras.losses.mean_squared_error,
                           metrics=keras.metrics.RootMeanSquaredError(),
                           optimizer=keras.optimizers.RMSprop(
                               learning_rate=params['train']['learning_rate']))
        history = forecaster.fit(
            x_train,
            y_train,
            shuffle=False,
            epochs=params['train']['epochs'],
            batch_size=params['train']['batch_size'],
            validation_split=0.2,
            verbose=1,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    patience=params['train']['patience'],
                    monitor='val_loss',
                    mode='min',
                    verbose=1,
                    restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.75,
                    patience=params['train']['patience'] // 2,
                    verbose=1,
                    mode='min')
            ])
        with open(os.path.join(local_metrics_path, f'train_{model}.json'), 'w') as f:
            for k, v in history.history.items():
                history.history[k] = [float(vi) for vi in v]
            f.write(json.dumps(history.history))
        forecaster.save(os.path.join(local_models_path, f'{model}.h5'))
        if params['one_feature']: break


if __name__ == '__main__':
    train()
