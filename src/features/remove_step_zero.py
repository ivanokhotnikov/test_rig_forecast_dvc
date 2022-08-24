import pandas as pd


def remove_step_zero(df):
    """
    Remove step zero from dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe to remove step zero from.

    Returns
    -------
    pandas.DataFrame
        Dataframe with step zero removed.
    """
    return df.drop(df[df['STEP'] == 0].index, axis=0).reset_index(drop=True)


if __name__ == '__main__':
    remove_step_zero()