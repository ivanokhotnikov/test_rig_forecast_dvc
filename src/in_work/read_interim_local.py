import gc
import os

import click
import pandas as pd


@click.command()
@click.argument('local_interim_data_path',
                type=click.Path(exists=True),
                default=r'..\data\interim')
@click.argument('local_processed_data_path',
                type=click.Path(),
                default=r'..\data\processed')
@click.option('-v',
              '--verbose',
              is_flag=True,
              default=True,
              help='Print verbose output')
def read_interim_local(local_interim_data_path, local_processed_data_path,
                       verbose):
    """
    Read interim data from local directory.

    Parameters
    ----------
    local_interim_data_path : str
        Path to local interim data directory
    local_processed_data_path : str
        Path to output directory
    verbose : bool 
        Print verbose output

    Returns
    -------
    None
    """
    if verbose: print(f'Reading interim data from {local_interim_data_path}')
    pass


if __name__ == '__main__':
    read_interim_local()
