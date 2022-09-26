import csv, random
from collections import defaultdict
from enum import Enum, auto

DEFAULT_SEED = 100

class TrainTestDataType(Enum):
    TRAIN = auto()
    TEST = auto()

def _hash_with_seed(value: str, seed: int):
    hash=0
    for ch in value:
        hash = (hash * (seed * 17) ^ ord(ch) * seed) & 0xFFFFFFFF
    return hash

class MarketData:
    def __init__(self, filename: str, test_split: float=0.4, seed: int=DEFAULT_SEED):
        filename = filename
        csvreader = csv.reader(open(filename, newline=''), delimiter=',', quotechar='|')
        self.column_name_to_idx = {}
        self.market_symbol_to_entries = defaultdict(list)
        for entry in csvreader:
            market, symbol = entry[1], entry[2]
            if market == 'market':
                for i, column in enumerate(entry):
                    self.column_name_to_idx[column] = i
                continue
            self.market_symbol_to_entries[market + symbol].append(entry)
        market_symbols = list(self.market_symbol_to_entries.keys())
        self.market_symbols_train, self.market_symbols_test = [], []

        self._split_train_test(market_symbols, test_split, seed)
        self.set_train_test(TrainTestDataType.TRAIN, if_reset=False)
        self.reset(shuffle=False)

    def _split_train_test(self, market_symbols, test_split, seed):
        symbols_with_hash = [(_hash_with_seed(symbol, seed), symbol,) for symbol in market_symbols]
        symbols_with_hash.sort()

        train_size = int(len(symbols_with_hash) * (1.0 - test_split))
        self.market_symbols_train = [s[1] for s in symbols_with_hash[:train_size]]
        self.market_symbols_test = [s[1] for s in symbols_with_hash[train_size:]]

    def set_train_test(self, train_test_data_type: TrainTestDataType, if_reset: bool=True):
        self.train_test_data_type = train_test_data_type
        if train_test_data_type == TrainTestDataType.TRAIN:
            self.market_symbols = self.market_symbols_train
        else:
            self.market_symbols = self.market_symbols_test
        if if_reset:
            self.reset()

    def reset(self, shuffle: bool=True):
        if shuffle:
            random.shuffle(self.market_symbols)
        self.market_symbol_i = 0
        symbol = self.market_symbols[self.market_symbol_i]
        self.market_symbol_iter = iter(self.market_symbol_to_entries[symbol])

    def __iter__(self):
        return self

    def _seek_next_symbol(self):
        self.market_symbol_i += 1
        if self.market_symbol_i >= len(self.market_symbols):
            self.reset()
        symbol = self.market_symbols[self.market_symbol_i]
        print('the next symbol is {symbol}'.format(symbol=symbol))
        self.market_symbol_iter = iter(self.market_symbol_to_entries[symbol])

    def __next__(self):
        while True:
            try:
                return next(self.market_symbol_iter)
            except StopIteration:
                # end of the current epoch. The next epoch reads the next symbol.
                self._seek_next_symbol()
                raise StopIteration
