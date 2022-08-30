import json
import os

import click
import keras
import pandas as pd
from dvc.api import params_show
from joblib import load
from dvclive.keras import DvcLiveCallback
from features import FORECAST_FEATURES
from features.create_sequences import create_sequences


def read_model(model_path, model):
    if 'scaler' in model:
        return load(os.path.join(model_path, model))
    return keras.models.load_model(os.path.join(model_path, model))


@click.command()
@click.argument('local_test_data_path', type=click.Path(exists=True))
@click.argument('local_models_path', type=click.Path(exists=True))
@click.argument('local_metrics_path', type=click.Path(exists=False))
def evaluate(local_test_data_path, local_models_path, local_metrics_path):
    """
    Evaluate the model.

    Returns
    -------
    None
    
    """
    params = params_show()
    test_df = pd.read_csv(os.path.join(local_test_data_path, 'test_data.csv'),
                          index_col=False)
    if not os.path.exists(local_metrics_path):
        os.makedirs(local_metrics_path)
    for feature in FORECAST_FEATURES:
        model = f'RNN_{feature}'
        test_data = test_df[feature].values.reshape(-1, 1)
        scaler = read_model(local_models_path, model + '_scaler.joblib')
        scaled_test = scaler.transform(test_data)
        x_test, y_test = create_sequences(
            scaled_test,
            lookback=params['model']['look_back'],
            inference=False)
        forecaster = read_model(local_models_path, model + '.h5')
        results = forecaster.evaluate(x_test,
                                      y_test,
                                      verbose=1,
                                      batch_size=params['train']['batch_size'],
                                      return_dict=True)
        with open(os.path.join(local_metrics_path, f'eval_{model}.json'),
                  'w') as f:
            f.write(json.dumps(results))
        if params['one_feature']: break


if __name__ == '__main__':
    evaluate()
