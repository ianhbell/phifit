import random

import numpy as np

# Imports from deap
from deap import algorithms, base, creator, tools
from deap.algorithms import eaSimple

def minimize_deap(f, bounds, **kwargs):
    N = len(bounds)

    # weight is -1 because we want to minimize the error
    creator.create("FitnessMax", base.Fitness, weights=(-1.0,)) 
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    # See: http://deap.readthedocs.org/en/master/tutorials/basic/part1.html#a-funky-one    
    toolbox = base.Toolbox()

    def generate_individual(ind_class, bounds):
        return ind_class([random.uniform(l,u) for l,u in bounds])

    toolbox.register("individual", generate_individual,  creator.Individual, bounds)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # If args are passed in as a keyword argument, remove them from keyword arguments and unpack into objective function
    args = kwargs.pop('args', ())
    def myfunc(c, **kwarwgs):
        """ Wrap the objective function so that it returns a tuple for compatibility with deap """
        return (f(c, *args, **kwargs), )
    toolbox.register("evaluate", myfunc, **kwargs)
    if 'DeltaPenality' in dir(tools):
        toolbox.decorate("evaluate", tools.DeltaPenality(feasible, 100000))
    # If two individuals mate, interpolate between them, allow for a bit of extrapolation
    toolbox.register("mate", tools.cxBlend, alpha = 0.3)
    sigma = np.array([0.01]*N).tolist()

    toolbox.register("mutate", tools.mutGaussian, mu = 0, sigma = sigma, indpb=1.0)
    toolbox.register("select", tools.selTournament, tournsize=5) # The greater the tournament size, the greater the selection pressure 
    
    hof = tools.HallOfFame(50)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    
    pop = toolbox.population(n=5000)
    pop, log = eaSimple(pop, 
                        toolbox, 
                        cxpb=0.5, # Crossover probability
                        mutpb=0.3, # Mutation probability
                        ngen=20, 
                        stats=stats, 
                        halloffame=hof, 
                        verbose=True)
    best = max(pop, key=attrgetter("fitness"))
    print(best, best.fitness)
    return list(best)