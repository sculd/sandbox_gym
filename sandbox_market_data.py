from trading.envs.trading_env import MarketData, TrainTestDataType
md = MarketData('data.csv')
for entry in md:
    print(entry)

print('shuffle and re iterate')
md.reset()
for entry in md:
    print(entry)

md.set_train_test(TrainTestDataType.TEST)
for entry in md:
    print(entry)

md.reset()
for entry in md:
    print(entry)
