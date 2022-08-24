import json
import os

import click
import keras
import pandas as pd
from dvc.api import params_show
from dvclive import Live
from joblib import load

from features import FORECAST_FEATURES
from features.create_sequences import create_sequences


def read_model(model_path, model):
    if 'scaler' in model:
        return load(os.path.join(model_path, model))
    return keras.models.load_model(os.path.join(model_path, model))


@click.command()
def evaluate():
    """
    Evaluate the model.

    Returns
    -------
    None
    
    """
    params = params_show()
    local_test_data_path = params['test_data']
    local_models_path = params['models']
    local_metrics_path = params['metrics']
    verbosity = params['train']['verbosity']
    batch_size = params['train']['batch_size']
    look_back = params['model']['look_back']
    one_feature = params['one_feature']
    test_df = pd.read_csv(local_test_data_path, index_col=False)
    for feature in FORECAST_FEATURES:
        model = f'RNN_{feature}'
        live = Live(path=os.path.join(local_metrics_path, 'eval'))
        test_data = test_df[feature].values.reshape(-1, 1)
        scaler = read_model(local_models_path, model + '_scaler.joblib')
        scaled_test = scaler.transform(test_data)
        x_test, y_test = create_sequences(scaled_test,
                                          lookback=look_back,
                                          inference=False)
        forecaster = read_model(local_models_path, model + '.h5')
        results = forecaster.evaluate(x_test,
                                      y_test,
                                      verbose=verbosity,
                                      batch_size=batch_size,
                                      return_dict=True)
        for metric, values in results.items():
            live.log(f'eval_{model}_{metric}', values)
        if one_feature: break


if __name__ == '__main__':
    evaluate()
