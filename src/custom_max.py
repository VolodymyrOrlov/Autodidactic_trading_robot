import random

from deap import base
from deap import creator
from deap import tools
import numpy as np
import multiprocessing
from microssembly import Microssembly

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

MAX_SCORE = 1000

def eval_individual(ind):
    scores = []
    mssembly = Microssembly(trace=False)
    try:
        for i in range(10):
            data = np.random.randint(low=0, high=MAX_SCORE, size=4)
            mssembly.reset()
            mssembly.load_data(data.tolist())
            mssembly.run(''.join(map(str, ind)))
            scores.append(1 if np.abs(np.max(data) - mssembly.out_memory[15]) < 0.05 else 0)
        score = np.average(scores)
        return score,
    except:
        return 0,


def varOr(population, toolbox, lambda_, cxpb, mutpb):
    assert (cxpb + mutpb) <= 1.0, (
        "The sum of the crossover and mutation probabilities must be smaller "
        "or equal to 1.0.")

    offspring = []
    for _ in range(lambda_):
        op_choice = random.random()
        if op_choice < cxpb:            # Apply crossover
            ind1, ind2 = map(toolbox.clone, random.sample(population, 2))
            ind1, ind2 = toolbox.mate(ind1, ind2)
            del ind1.fitness.values
            offspring.append(ind1)
        elif op_choice < cxpb + mutpb:  # Apply mutation
            ind = toolbox.clone(random.choice(population))
            ind, = toolbox.mutate(ind)
            del ind.fitness.values
            offspring.append(ind)
        else:                           # Apply reproduction
            offspring.append(random.choice(population))

    return offspring


def find_best_model(ngen=100, cxpb=0.5, mutpb=0.2, indpb = 0.05, pop_size=300):

    toolbox.register("attr_bool", random.randint, 0, 1)

    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     toolbox.attr_bool, 100)

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", eval_individual)

    toolbox.register("mate", tools.cxTwoPoint)

    toolbox.register("mutate", tools.mutFlipBit, indpb=indpb)

    toolbox.register("select", tools.selRoulette)

    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)

    population = toolbox.population(n=pop_size)

    CAT_PB = 0.0001
    AFTER_CAT_PB = 0.05

    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    print('gen\teval\tmin\tmax\tmean\tstd')

    for gen in range(1, ngen + 1):

        offspring = varOr(population, toolbox, pop_size, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        population[:] = toolbox.select(offspring, pop_size)

        if np.random.uniform() < CAT_PB:
            print('Catastrophe!!!')
            np.random.shuffle(population)
            before_len = len(population)
            population = population[:int(before_len*AFTER_CAT_PB)]
            new_pop = toolbox.population(n=int(before_len*(1.0 - AFTER_CAT_PB)))
            fitnesses = toolbox.map(toolbox.evaluate, new_pop)
            for ind, fit in zip(new_pop, fitnesses):
                ind.fitness.values = fit
            population = population + new_pop

        if fitnesses and gen % 100 == 0:
            print('{}\t{}\t{}\t{}\t{}\t{}'.format(gen,
                                                  len(invalid_ind),
                                                  round(np.min(fitnesses), 3),
                                                  round(np.max(fitnesses), 3),
                                                  round(np.mean(fitnesses), 3),
                                                  round(np.std(fitnesses), 3)))

    best_ind = tools.selBest(population, 1)
    return population, best_ind, None