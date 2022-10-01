import pandas as pd, numpy as np

df = pd.read_csv('data.csv')

_move_amplitude_threshold = 0.05

for dt_minutes in [10, 30]:
    diff_column = 'diff_future_{m}m'.format(m=dt_minutes)
    df[diff_column]=-df.groupby(['market', 'symbol'])['priceAtAnalysis'].diff(-dt_minutes)
    return_column = 'return_future_{m}m'.format(m=dt_minutes)
    df[return_column]=df[diff_column] / df.priceAtAnalysis
    df.drop(diff_column, axis=1, inplace=True)
    df['return_jump_{m}m'.format(m=dt_minutes)] = df[return_column] > _move_amplitude_threshold
    df['return_drop_{m}m'.format(m=dt_minutes)] = df[return_column] < -_move_amplitude_threshold

df.to_csv('data_ml.csv', index=False)
