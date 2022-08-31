import json
import os

import click
import keras
import pandas as pd
from dvc.api import params_show
from joblib import load

from features.create_sequences import create_sequences


@click.command()
@click.argument('local_test_data_path', type=click.Path(exists=True))
@click.argument('local_models_path', type=click.Path(exists=True))
@click.argument('local_metrics_path', type=click.Path(exists=False))
@click.argument('feature', type=click.STRING)
def evaluate(local_test_data_path, local_models_path, local_metrics_path,
             feature):
    """Takes the model from the local model directory, evaluates it on the test data set, saves the resultant metrics to the local metrics directory

    Args:
        local_test_data_path (str): Test data directory
        local_models_path (str): Models directory
        local_metrics_path (str): Metrics directory
        feature (str): Feature
    """
    params = params_show()
    test_df = pd.read_csv(os.path.join(local_test_data_path, 'test_data.csv'),
                          index_col=False)
    test_data = test_df[feature].values.reshape(-1, 1)
    scaler = load(os.path.join(local_models_path, f'{feature}.joblib'))
    scaled_test = scaler.transform(test_data)
    x_test, y_test = create_sequences(scaled_test,
                                      lookback=params['model']['look_back'],
                                      inference=False)
    forecaster = keras.models.load_model(
        os.path.join(local_models_path, f'{feature}.h5'))
    results = forecaster.evaluate(x_test,
                                  y_test,
                                  verbose=1,
                                  batch_size=params['train']['batch_size'],
                                  return_dict=True)
    with open(os.path.join(local_metrics_path, f'eval_{feature}.json'),
              'w') as f:
        f.write(json.dumps(results))


if __name__ == '__main__':
    evaluate()
