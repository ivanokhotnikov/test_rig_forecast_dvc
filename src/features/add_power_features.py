import numpy as np
import pandas as pd


def add_power_features(df):
    """
    Add power features to dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe to add power features to.

    Returns
    -------
    pandas.DataFrame
        Dataframe with power features added.    
    """
    df['DRIVE POWER'] = (df['M1 SPEED'] * df['M1 TORQUE'] * np.pi / 30 /
                         1e3).astype(np.float32)
    df['LOAD POWER'] = abs(df['D1 RPM'] * df['D1 TORQUE'] * np.pi / 30 /
                           1e3).astype(np.float32)
    df['CHARGE MECH POWER'] = (df['M2 RPM'] * df['M2 Torque'] * np.pi / 30 /
                               1e3).astype(np.float32)
    df['CHARGE HYD POWER'] = (df['CHARGE PT'] * 1e5 * df['CHARGE FLOW'] *
                              1e-3 / 60 / 1e3).astype(np.float32)
    df['SERVO MECH POWER'] = (df['M3 RPM'] * df['M3 Torque'] * np.pi / 30 /
                              1e3).astype(np.float32)
    df['SERVO HYD POWER'] = (df['Servo PT'] * 1e5 * df['SERVO FLOW'] * 1e-3 /
                             60 / 1e3).astype(np.float32)
    df['SCAVENGE POWER'] = (df['M5 RPM'] * df['M5 Torque'] * np.pi / 30 /
                            1e3).astype(np.float32)
    df['MAIN COOLER POWER'] = (df['M6 RPM'] * df['M6 Torque'] * np.pi / 30 /
                               1e3).astype(np.float32)
    df['GEARBOX COOLER POWER'] = (df['M7 RPM'] * df['M7 Torque'] * np.pi / 30 /
                                  1e3).astype(np.float32)
    return df


if __name__ == '__main__':
    add_power_features()
