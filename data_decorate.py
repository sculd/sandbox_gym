import pandas as pd, numpy as np

df = pd.read_csv('data.csv')

for dt_minutes in [10, 30]:
    diff_column = 'diff_future_{m}m'.format(m=dt_minutes)
    df[diff_column]=-df.groupby(['market', 'symbol'])['priceAtAnalysis'].diff(-dt_minutes)
    df['return_future_{m}m'.format(m=dt_minutes)]=df[diff_column] / df.priceAtAnalysis
    df.drop(diff_column, axis=1, inplace=True)
    df['return_jump_{m}m'.format(m=dt_minutes)] = df.return_future_10m > 0.05

df.to_csv('data_ml.csv')



for dt_minutes in [10]:
    diff_column = 'diff_future_{m}m'.format(m=dt_minutes)
    df[diff_column]=-df.groupby(['market', 'symbol'])['priceAtAnalysis'].diff(-dt_minutes)
    df['return_future_{m}m'.format(m=dt_minutes)]=df[diff_column] / df.priceAtAnalysis
    df.drop(diff_column, axis=1, inplace=True)



