import numpy as np


def add_power_features(df):
    """Creates power features, saves the DataFrame with the added feature.

    Args:
        df (DataFrame): The DataFrame to add the new features to

    Returns:
        DataFrame: The resultant dataframe
    """
    df['DRIVE_POWER'] = (df['M1 SPEED'] * df['M1 TORQUE'] * np.pi / 30 /
                         1e3).astype(np.float32)
    df['LOAD_POWER'] = abs(df['D1 RPM'] * df['D1 TORQUE'] * np.pi / 30 /
                           1e3).astype(np.float32)
    df['CHARGE_MECH_POWER'] = (df['M2 RPM'] * df['M2 Torque'] * np.pi / 30 /
                               1e3).astype(np.float32)
    df['CHARGE_HYD_POWER'] = (df['CHARGE PT'] * 1e5 * df['CHARGE FLOW'] *
                              1e-3 / 60 / 1e3).astype(np.float32)
    df['SERVO_MECH_POWER'] = (df['M3 RPM'] * df['M3 Torque'] * np.pi / 30 /
                              1e3).astype(np.float32)
    df['SERVO_HYD_POWER'] = (df['Servo PT'] * 1e5 * df['SERVO FLOW'] * 1e-3 /
                             60 / 1e3).astype(np.float32)
    df['SCAVENGE_POWER'] = (df['M5 RPM'] * df['M5 Torque'] * np.pi / 30 /
                            1e3).astype(np.float32)
    df['MAIN_COOLER_POWER'] = (df['M6 RPM'] * df['M6 Torque'] * np.pi / 30 /
                               1e3).astype(np.float32)
    df['GEARBOX_COOLER_POWER'] = (df['M7 RPM'] * df['M7 Torque'] * np.pi / 30 /
                                  1e3).astype(np.float32)
    return df


if __name__ == '__main__':
    add_power_features()
