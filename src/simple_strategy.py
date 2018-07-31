import pandas as pd
import matplotlib.pyplot as plt
import random

from deap import base
from deap import creator
from deap import tools
from deap import algorithms
import numpy as np
import multiprocessing
from microssembly import Microssembly
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

    in_mem_size = 3

    mssembly = Microssembly(in_memory_length=in_mem_size, out_memory_length=1, registers_length=5)

    def pad_left(series, n):
        head = price.index[0]
        pad_value = price[0]
        pad = pd.Series(np.full(n, pad_value), index=pd.date_range(start = head - pd.DateOffset(days=n-1),
                                                                   end = head) - pd.DateOffset(days=1))
        return pad.append(series)

    def apply_code(values):
        mssembly.load_data(values.tolist())
        mssembly.run(code)
        return mssembly.out_memory[0] if mssembly.out_memory[1] != 0 else np.nan

    return pad_left(price, 2**in_mem_size - 1).rolling(2**in_mem_size).apply(apply_code)[(2**in_mem_size - 1):]


def loss_function(actual, observed):

    def recall(signum):
        return np.sum(actual[observed[observed == signum].index] == signum) / len(observed[observed == signum]) + 1e-10

    # def precision():
    #     return 1 - (np.abs(len(actual.dropna()) - len(observed.dropna()))
    #                 / max(len(actual.dropna()), len(observed.dropna()))) + 1e-10

    def precision():
        return (np.sum(np.isnan(actual[observed[observed.isna()].index]))
         / len(observed[observed.isna()]) + 1e-9)

    return 3 / (1 / recall(1) + 1 / recall(0) + 1 / precision())


def eval_individual(ind, get_strategy_signal):
    price = random_price()
    observed_signal = get_strategy_signal(price)
    while len(np.unique(observed_signal.dropna())) < 1 and len(observed_signal.dropna()) > 5:
        price = random_price()
        observed_signal = get_strategy_signal(price)
    signal = get_signal(price, ''.join(map(str, ind)))
    return loss_function(signal, observed_signal),


def cx_random_one_point(ind1, ind2, mutpb):

    def random_roll(i):
        return np.roll(i, shift=np.random.randint(1, len(i))).tolist()

    if np.random.uniform() < mutpb:
        size = min(len(ind1), len(ind2))
        cxpoint = random.randint(1, size - 1)
        ind1[cxpoint:], ind2[cxpoint:] = random_roll(ind2)[cxpoint:], random_roll(ind1)[cxpoint:]
    else:
        size = min(len(ind1), len(ind2))
        cxpoint = random.randint(1, size - 1)
        ind1[cxpoint:], ind2[cxpoint:] = ind2[cxpoint:], ind1[cxpoint:]

    return ind1, ind2


def find_best_model(strategy, ngen=100, cxpb=0.5, mutpb=0.2, indpb = 0.05, pop_size=300, ind_size = 1000):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     toolbox.attr_bool, ind_size)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)

    toolbox.register("evaluate", eval_individual, get_strategy_signal=strategy)
    toolbox.register("mate", cx_random_one_point, mutpb=mutpb)
    toolbox.register("mutate", tools.mutFlipBit, indpb=indpb)
    toolbox.register("select", tools.selRoulette)
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, log = algorithms.eaMuPlusLambda(pop, toolbox, pop_size, int(pop_size*2), cxpb=cxpb, mutpb=mutpb, ngen=ngen,
                                         stats=stats, halloffame=hof, verbose=True)

    return pop, hof, log

