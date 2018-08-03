import random

from deap import base
from deap import creator
from deap import tools
from deap import algorithms
import numpy as np
import multiprocessing
from microssembly2 import Microssembly


MAX_SCORE = 10000


def eval_individual(ind):
    scores = []
    mssembly = Microssembly(trace=False, architecture=4)
    for i in range(100):
        data = np.random.randint(low=0, high=MAX_SCORE, size=4)
        mssembly.reset()
        mssembly.load_data(data.tolist())
        mssembly.run(''.join(map(str, ind)), cycles=50)
        scores.append(1 if np.abs(np.sum(data) - mssembly.memory[15]) < 1e-2 else 0)
    score = np.average(scores) + np.random.rand() * 1e-10
    return score,


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


def find_best_model(ngen=100, cxpb=0.5, mutpb=0.2, indpb=0.05, pop_size=300, ind_size=200, start_pop = []):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     toolbox.attr_bool, ind_size)
    toolbox.register("population", population, start_pop, toolbox.individual)
    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)

    toolbox.register("evaluate", eval_individual)
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


