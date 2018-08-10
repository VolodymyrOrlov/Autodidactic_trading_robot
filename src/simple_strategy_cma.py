import pandas as pd
import matplotlib.pyplot as plt
import random

from deap import creator, base, tools, algorithms, cma
import numpy as np
import multiprocessing
from microssembly2 import Microssembly
from price_generator import random_price


def plot_trades(price, signal, plt):
    l = ['price']

    p = price
    ax = p.plot()
    ax.xaxis.grid(True, which='minor', linestyle='-', linewidth=0.25)

    indices = [['g^', signal[signal == 1].index, 'long entry'],
               ['ko', signal[signal == 0].index, 'exit'],
               ['rv', signal[signal == 2].index, 'short entry']]

    for style, idx, label in indices:
        if len(idx) > 0:
            p[idx].plot(style=style)
            l.append(label)

    plt.xlim([p.index[0], p.index[-1]])

    plt.legend(l, loc='best')
    plt.title('trades')


def get_signal(price, code):

    price = normalize_price(price)

    in_mem_size = 3

    mssembly = Microssembly(architecture=4)

    def pad_left(series, n):
        head = series.index[0]
        pad_value = series[0]
        pad = pd.Series(np.full(n, pad_value), index=pd.date_range(start = head - pd.DateOffset(days=n-1),
                                                                   end = head) - pd.DateOffset(days=1))
        return pad.append(series)

    def apply_code(values):
        mssembly.load_data(values.tolist())
        mssembly.run(code, cycles=100)
        return mssembly.memory[15] if mssembly.memory[14] != 0 else np.nan

    return pad_left(price, 2**in_mem_size - 1).rolling(2**in_mem_size).apply(apply_code)[(2**in_mem_size - 1):]


def loss_function(actual, observed):

    def recall(signum):
        return np.sum(actual[observed[observed == signum].index] == signum) / len(observed[observed == signum])

    return 3 / (1 / recall(1) + 1 / recall(0) + 1 / recall(2))


def normalize_price(price):
    return (price - price.shift(1)).fillna(0)


def eval_individual(ind, get_strategy_signal):
    ind = np.round(ind).astype(np.int)
    price = random_price()
    observed_signal = get_strategy_signal(price)
    while len(np.unique(observed_signal.dropna())) < 1 and len(observed_signal.dropna()) > 5:
        price = random_price()
        observed_signal = get_strategy_signal(price)
    signal = get_signal(price, ''.join(map(str, ind)))
    return loss_function(signal.fillna(2, inplace=False), observed_signal.fillna(2, inplace=False)),


def find_best_model(strategy, ngen=100, pop_size=300, ind_size = 1000, sigma=0.001):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)


    toolbox = base.Toolbox()
    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)
    toolbox.register("evaluate", eval_individual, get_strategy_signal=strategy)

    strategy = cma.Strategy(centroid=[0.5]*ind_size, sigma=sigma, lambda_=pop_size)
    toolbox.register("generate", strategy.generate, creator.Individual)
    toolbox.register("update", strategy.update)

    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("sig/noise", lambda x: np.mean(x) / np.std(x))
    stats.register("std", np.std)
    stats.register("max", np.max)

    algorithms.eaGenerateUpdate(toolbox, ngen=ngen, stats=stats, halloffame=hof)

    return np.round(hof[0]).astype(np.int)

