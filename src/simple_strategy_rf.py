import pandas as pd
import matplotlib.pyplot as plt
import random

from deap import base
from deap import creator
from deap import tools
from deap import algorithms

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

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


def get_features(price, code):

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
        return mssembly.memory

    p = pad_left(price, 2**in_mem_size - 1)
    df = pd.DataFrame([apply_code(p.shift(-x).values[:2**in_mem_size]) for x in range(len(price))[::]])
    df[~np.isfinite(df)] = 0
    return df.values


def get_signal(price, signal, code):
    X = get_features(price, code)
    X_train, X_test, y_train, y_test = train_test_split(X, signal, test_size=0.5)
    y_train.fillna(2, inplace=True)
    y_test.fillna(2, inplace=True)
    estimator = DecisionTreeClassifier(class_weight='balanced')
    estimator.fit(X_train, y_train)
    return pd.Series(estimator.predict(X_test), index=y_test.index)


def loss_function(actual, observed):

    def recall(signum):
        return np.sum(actual[observed[observed == signum].index] == signum) / len(observed[observed == signum]) + 1e-10

    return 3 / (1 / recall(1) + 1 / recall(0) + 1 / recall(2))


def eval_individual(ind, get_strategy_signal):

    def get_good_price():
        price = random_price(127 * 2)
        signal = get_strategy_signal(price)
        while len(np.unique(signal.dropna())) < 1 and len(signal.dropna()) > 5:
            price = random_price(127 * 2)
            signal = get_strategy_signal(price)
        return price, signal

    price, signal = get_good_price()

    X = get_features(price, ''.join(map(str, ind)))
    X_train, X_test, y_train, y_test = train_test_split(X, signal, test_size=0.5)
    y_train.fillna(2, inplace=True)
    y_test.fillna(2, inplace=True)

    try:
        estimator = DecisionTreeClassifier(class_weight='balanced')
        estimator.fit(X_train, y_train)
        signal = estimator.predict(X_test)
        return loss_function(pd.Series(signal, index=y_test.index), y_test),
    except:
        return 0,


def cx_random_one_point(ind1, ind2):

    size = min(len(ind1), len(ind2))
    cxpoint = random.randint(1, size - 1)
    cxpoint = cxpoint // (4 + 4 * 2) * (4 + 4 * 2)
    ind1[cxpoint:], ind2[cxpoint:] = ind2[cxpoint:], ind1[cxpoint:]

    return ind1, ind2


def population(start_pop, f, n):
    if start_pop:
        pop = tools.initRepeat(list, f, n)
        for (i, p) in enumerate(start_pop):
            pop[i][:] = start_pop[i]
        return pop
    return tools.initRepeat(list, f, n)


def find_best_model(strategy, ngen=100, cxpb=0.5, mutpb=0.2, indpb = 0.05, pop_size=300, ind_size = 1000, start_pop = []):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     toolbox.attr_bool, ind_size)
    toolbox.register("population", population, start_pop, toolbox.individual)
    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)

    toolbox.register("evaluate", eval_individual, get_strategy_signal=strategy)
    toolbox.register("mate", cx_random_one_point)
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

