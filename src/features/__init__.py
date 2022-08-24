from . import add_power_features
from . import add_time_features
from . import remove_step_zero
from . import create_sequences

RAW_FORECAST_FEATURES = [
    'TIME', 'STEP', 'HSU DEMAND', 'M1 SPEED', 'M1 CURRENT', 'M1 TORQUE',
    'PT4 SETPOINT', 'PT4', 'D1 RPM', 'D1 CURRENT', 'D1 TORQUE', 'M2 RPM',
    'M2 Amp', 'M2 Torque', 'CHARGE PT', 'CHARGE FLOW', 'M3 RPM', 'M3 Amp',
    'M3 Torque', 'Servo PT', 'SERVO FLOW', 'M4 ANGLE', 'HSU IN', 'TT2',
    'HSU OUT', 'M5 RPM', 'M5 Amp', 'M5 Torque', 'M6 RPM', 'M6 Amp',
    'M6 Torque', 'M7 RPM', 'M7 Amp', 'M7 Torque', 'Vibration 1',
    ' Vibration 2', ' DATE'
]

FEATURES_FOR_ANOMALY_DETECTION = [
    'TIME', 'STEP', 'HSU DEMAND', 'M1 SPEED', 'M1 CURRENT', 'M1 TORQUE',
    'PT4 SETPOINT', 'PT4', 'D1 RPM', 'D1 CURRENT', 'D1 TORQUE', 'M2 RPM',
    'M2 Amp', 'M2 Torque', 'CHARGE PT', 'CHARGE FLOW', 'M3 RPM', 'M3 Amp',
    'M3 Torque', 'Servo PT', 'SERVO FLOW', 'M4 ANGLE', 'HSU IN', 'TT2',
    'HSU OUT', 'M5 RPM', 'M5 Amp', 'M5 Torque', 'M6 RPM', 'M6 Amp',
    'M6 Torque', 'M7 RPM', 'M7 Amp', 'M7 Torque'
]

ENGINEERED_FEATURES = [
    'DRIVE POWER', 'LOAD POWER', 'CHARGE MECH POWER', 'CHARGE HYD POWER',
    'SERVO MECH POWER', 'SERVO HYD POWER', 'SCAVENGE POWER',
    'MAIN COOLER POWER', 'GEARBOX COOLER POWER'
]
PRESSURE_TEMPERATURE = ['PT4', 'HSU IN', 'TT2', 'HSU OUT']
VIBRATIONS = ['Vibration 1', ' Vibration 2']
COMMANDS = ['TIME', ' DATE', 'STEP', 'HSU DEMAND', 'PT4 SETPOINT']
TIME_FEATURES = ['DURATION', 'RUNNING SECONDS', 'RUNNING HOURS']
UNIT_FEATURES = ['UNIT', 'TEST', 'ARMANI']
FEATURES_NO_TIME = [
    f for f in RAW_FORECAST_FEATURES if f not in ('TIME', ' DATE')
]
FEATURES_NO_TIME_AND_COMMANDS = [
    f for f in FEATURES_NO_TIME if f not in COMMANDS
]

FORECAST_FEATURES = ENGINEERED_FEATURES + PRESSURE_TEMPERATURE + VIBRATIONS