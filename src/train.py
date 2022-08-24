import os

import click
import keras
import pandas as pd
from dvc.api import params_show
from dvclive import Live
from joblib import dump
from sklearn.preprocessing import MinMaxScaler

from features import FORECAST_FEATURES
from features.create_sequences import create_sequences


@click.command()
def train():
    """
    Train the model.

    Returns
    -------
    None
    """
    params = params_show()
    local_train_data_path = params['train_data']
    local_models_path = params['models']
    local_metrics_path = params['metrics']
    learning_rate = params['train']['learning_rate']
    patience = params['train']['patience']
    verbosity = params['train']['verbosity']
    epochs = params['train']['epochs']
    batch_size = params['train']['batch_size']
    look_back = params['model']['look_back']
    lstm_units = params['model']['lstm_units']
    one_feature = params['one_feature']
    train_df = pd.read_csv(local_train_data_path, index_col=False)
    if not os.path.exists(local_metrics_path):
        os.makedirs(local_metrics_path)
    if not os.path.exists(local_models_path):
        os.makedirs(local_models_path)
    for feature in FORECAST_FEATURES:
        live = Live(path=os.path.join(local_metrics_path, 'train'))
        model = f'RNN_{feature}'
        train_data = train_df[feature].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_train = scaler.fit_transform(train_data)
        dump(scaler, os.path.join(local_models_path, f'{model}_scaler.joblib'))
        x_train, y_train = create_sequences(scaled_train,
                                            lookback=look_back,
                                            inference=False)
        forecaster = keras.models.Sequential()
        forecaster.add(
            keras.layers.LSTM(lstm_units,
                              input_shape=(x_train.shape[1], x_train.shape[2]),
                              return_sequences=False))
        forecaster.add(keras.layers.Dense(1))
        forecaster.compile(
            loss=keras.losses.mean_squared_error,
            metrics=keras.metrics.RootMeanSquaredError(),
            optimizer=keras.optimizers.RMSprop(learning_rate=learning_rate))
        history = forecaster.fit(
            x_train,
            y_train,
            shuffle=False,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=verbosity,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=patience,
                                              monitor='val_loss',
                                              mode='min',
                                              verbose=verbosity,
                                              restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                  factor=0.75,
                                                  patience=patience // 2,
                                                  verbose=verbosity,
                                                  mode='min')
            ])
        for metric, values in history.history.items():
            live.log(f'train_{model}_{metric}', values[-1])
        forecaster.save(os.path.join(local_models_path, f'{model}.h5'))
        if one_feature: break


if __name__ == '__main__':
    train()
