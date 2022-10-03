import pandas as pd, numpy as np

df = pd.read_csv('data.csv')

_move_amplitude_threshold = 0.05

dff = df.set_index(['epochSeconds', 'market', 'symbol'])
dfp = pd.pivot_table(df, index='epochSeconds', columns=['market', 'symbol'], values='priceAtAnalysis')

for dt_minutes in [10, 30]:
    lookforward_indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=dt_minutes)
    max_column = 'max_{m}m'.format(m=dt_minutes)
    min_column = 'min_{m}m'.format(m=dt_minutes)
    dff[max_column] = dfp.rolling(window=lookforward_indexer).max().stack(['market', 'symbol'])
    df[max_column] = dff.reset_index()[max_column]
    dff[min_column] = dfp.rolling(window=lookforward_indexer).min().stack(['market', 'symbol'])
    df[min_column] = dff.reset_index()[min_column]
    jump_return_column = 'jump_future_{m}m'.format(m=dt_minutes)
    drop_return_column = 'drop_future_{m}m'.format(m=dt_minutes)
    df[jump_return_column] = (df[max_column] - df['priceAtAnalysis']) / df['priceAtAnalysis']
    df[drop_return_column] = (df[min_column] - df['priceAtAnalysis']) / df['priceAtAnalysis']
    df['return_jump_{m}m'.format(m=dt_minutes)] = df[jump_return_column] > _move_amplitude_threshold
    df['return_drop_{m}m'.format(m=dt_minutes)] = df[drop_return_column] < -_move_amplitude_threshold
    df.drop([jump_return_column, drop_return_column, max_column, min_column], axis=1, inplace=True)


df.to_csv('data_ml.csv', index=False)
