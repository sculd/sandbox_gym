'''
epochSeconds,market,symbol,stepChange,priceAtAnalysis,
minDrop,maxJump,changeSinceMinDrop,changeSinceMaxJump,
normalizedPricePT3H,changePT6H,rsiPT3H,bandwidthPT2H2,bandPercentPT2H2,moneyFlowPT40M,
return_future_10m,return_jump_10m,return_drop_10m,return_future_30m,return_jump_30m,return_drop_30m
'''


import tensorflow as tf
import numpy as np
import pandas as pd
import keras
from keras import layers

dataframe = pd.read_csv('data_ml.csv')

dataframe = dataframe[['stepChange','priceAtAnalysis','minDrop','maxJump','changeSinceMinDrop','changeSinceMaxJump','changePT6H','bandwidthPT2H2','bandPercentPT2H2','moneyFlowPT40M','return_jump_30m']]

dataframe['target'] = dataframe['return_jump_30m']
dataframe = dataframe.drop('return_jump_30m', axis=1)


val_dataframe = dataframe.sample(frac=0.2, random_state=1337)
train_dataframe = dataframe.drop(val_dataframe.index)

print(
    "Using %d samples for training and %d for validation"
    % (len(train_dataframe), len(val_dataframe))
)


def dataframe_to_dataset(dataframe):
    dataframe = dataframe.copy()
    labels = dataframe.pop("target")
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    ds = ds.shuffle(buffer_size=len(dataframe))
    return ds


train_ds = dataframe_to_dataset(train_dataframe)
val_ds = dataframe_to_dataset(val_dataframe)


for x, y in train_ds.take(1):
    print("Input:", x)
    print("Target:", y)


train_ds = train_ds.batch(32)
val_ds = val_ds.batch(32)

