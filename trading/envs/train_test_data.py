import csv, random
from collections import defaultdict
from enum import Enum, auto

class TrainTestDataType(Enum):
    TRAIN = auto()
    TEST = auto()

class TrainingData:
    def __init__(self, filename, test_split=0.4):
        filename = filename
        csvreader = csv.reader(open(filename, newline=''), delimiter=',', quotechar='|')
        self.market_symbol_to_entries = defaultdict(list)
        for entry in csvreader:
            market, symbol = entry[1], entry[2]
            if market == 'market':
                continue
            self.market_symbol_to_entries[market + symbol].append(entry)
        market_symbols = list(self.market_symbol_to_entries.keys())
        self.market_symbols_train, self.market_symbols_test = [], []
        for market_symbol in market_symbols:
            r = random.random()
            if r < test_split:
                self.market_symbols_test.append(market_symbol)
            else:
                self.market_symbols_train.append(market_symbol)
        self.set_train_test(TrainTestDataType.TRAIN, if_reset=False)
        self.reset(shuffle=False)

    def set_train_test(self, train_test_data_type, if_reset=True):
        self.train_test_data_type = train_test_data_type
        if train_test_data_type == TrainTestDataType.TRAIN:
            self.market_symbols = self.market_symbols_train
        else:
            self.market_symbols = self.market_symbols_test
        if if_reset:
            self.reset()

    def reset(self, shuffle=True):
        if shuffle:
            random.shuffle(self.market_symbols)
        self.market_symbol_i = 0
        self.market_symbol_iter = iter(self.market_symbol_to_entries[self.market_symbols[self.market_symbol_i]])

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            try:
                return next(self.market_symbol_iter)
            except StopIteration:
                self.market_symbol_i += 1
                if self.market_symbol_i >= len(self.market_symbols):
                    self.reset()
                    raise StopIteration
                self.market_symbol_iter = iter(self.market_symbol_to_entries[self.market_symbols[self.market_symbol_i]])