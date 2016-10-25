# Standard library imports
import random
from operator import attrgetter
from collections import Sequence
from itertools import repeat

# Conda-installable packages
import numpy as np

# Imports from deap
from deap import algorithms, base, creator, tools
from deap.algorithms import eaSimple

def myBlend(ind1, ind2, alpha, normalizing_functions = None):
    """Executes a blend crossover that modify in-place the input individuals.
    The blend crossover expects :term:`sequence` individuals of floating point
    numbers.
    
    :param ind1: The first individual participating in the crossover.
    :param ind2: The second individual participating in the crossover.
    :param alpha: Extent of the interval in which the new values can be drawn
                  for each attribute on both side of the parents' attributes.
    :param normalizing_functions: A list of functions the same length as the 
        number of individuals that can be used to 
    :returns: A tuple of two individuals.
    
    This function uses the :func:`~random.random` function from the python base
    :mod:`random` module.

    Originally from DEAP v 1.1.0, but modified by IB to allow for normalizing the 
    output variables.  Originally this was used to force values to stay at integer
    values

    """

    # Don't normalize if not provided
    if normalizing_functions is None:
        normalizing_functions = [lambda x: x]*len(ind1)

    for i, (x1, x2, normalizer) in enumerate(zip(ind1, ind2, normalizing_functions)):
        gamma = (1. + 2. * alpha) * random.random() - alpha
        ind1[i] = normalizer((1. - gamma) * x1 + gamma * x2)
        ind2[i] = normalizer(gamma * x1 + (1. - gamma) * x2)

    return ind1, ind2

def myMutateGaussian(individual, mu, sigma, indpb, normalizing_functions = None):
    """This function applies a gaussian mutation of mean *mu* and standard
    deviation *sigma* on the input individual. This mutation expects a
    :term:`sequence` individual composed of real valued attributes.
    The *indpb* argument is the probability of each attribute to be mutated.
    
    :param individual: Individual to be mutated.
    :param mu: Mean or :term:`python:sequence` of means for the
               gaussian addition mutation.
    :param sigma: Standard deviation or :term:`python:sequence` of 
                  standard deviations for the gaussian addition mutation.
    :param indpb: Independent probability for each attribute to be mutated.
    :param normalizing_functions: A list of functions, each of which operates
        on the perturbed value in the individual (for instance to snap it back to an integer)
    :returns: A tuple of one individual.
    
    This function uses the :func:`~random.random` and :func:`~random.gauss`
    functions from the python base :mod:`random` module.

    Originally from DEAP v. 1.1.0, modified by IB to allow for normalzing function
    """

    # Don't normalize if not provided
    if normalizing_functions is None:
        normalizing_functions = [lambda x: x]*len(ind1)

    size = len(individual)
    if not isinstance(mu, Sequence):
        mu = repeat(mu, size)
    elif len(mu) < size:
        raise IndexError("mu must be at least the size of individual: %d < %d" % (len(mu), size))
    if not isinstance(sigma, Sequence):
        sigma = repeat(sigma, size)
    elif len(sigma) < size:
        raise IndexError("sigma must be at least the size of individual: %d < %d" % (len(sigma), size))
    
    for i, m, s, fcn in zip(xrange(size), mu, sigma, normalizing_functions):
        if random.random() < indpb:
            individual[i] += fcn(random.gauss(m, s))
    
    return individual,    

def minimize_deap(f, bounds, Nindividuals = 5000, Ngenerations = 20, Nhof = 50, 
                  generator_functions = None, normalizing_functions = None,
                  **kwargs):
    """
    Parameters
    ----------
    Nindividuals: int
        Number of individuals in a generation
    Ngenerations: int
        Number of generations
    Nhof : int
        Number of individuals in the hall of fame
    generator_functions : list of functions    
        A list of functions, each of which take the arguments (lower bound, 
        higher bound) and generate a value (e.g., random.uniform()).  Defaults
        to random.uniform
    normalizing_functions : list of functions
        A list of functions, each of which operate on an element in the individual,
        and can modify its value (used to keep integer values integers, but allow
        for some blending)
    """
    N = len(bounds)

    # weight is -1 because we want to minimize the error
    creator.create("FitnessMax", base.Fitness, weights=(-1.0,)) 
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    # See: http://deap.readthedocs.org/en/master/tutorials/basic/part1.html#a-funky-one    
    toolbox = base.Toolbox()

    def generate_individual(ind_class, bounds, generator_functions):
        """ Generate a custom individual based on the generator functions provided """
        o = [] #output
        #assert(len(bounds) == len(generator_functions))
        for (l,u), generator_function in zip(bounds, generator_functions):
            o.append(generator_function(l, u))
        return ind_class(o)

    if generator_functions is None:
        generator_functions = [random.uniform]*len(bounds)

    toolbox.register("individual", generate_individual,  creator.Individual, bounds, generator_functions)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # If args are passed in as a keyword argument, remove them from keyword 
    # arguments and unpack into objective function
    args = kwargs.pop('args', ())
    def myfunc(c, **kwarwgs):
        """ 
        Wrap the objective function so that it returns a tuple for 
        compatibility with deap 
        """
        return (f(c, *args, **kwargs), )
    toolbox.register("evaluate", myfunc, **kwargs)
    if 'DeltaPenality' in dir(tools):
        toolbox.decorate("evaluate", tools.DeltaPenality(feasible, 100000))
    # If two individuals mate, interpolate between them, allow for a bit of extrapolation
    toolbox.register("mate", myBlend, alpha = 0.0, normalizing_functions = normalizing_functions)
    sigma = np.array([0.01]*N).tolist()

    toolbox.register("mutate", myMutateGaussian, mu = 0, sigma = sigma, indpb=1.0, normalizing_functions = normalizing_functions)
    toolbox.register("select", tools.selTournament, tournsize=3) # The greater the tournament size, the greater the selection pressure 
    
    hof = tools.HallOfFame(Nhof)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    
    pop = toolbox.population(n=Nindividuals)
    pop, log = eaSimple(pop, 
                        toolbox, 
                        cxpb=0.5, # Crossover probability
                        mutpb=0.3, # Mutation probability
                        ngen=Ngenerations, 
                        stats=stats, 
                        halloffame=hof, 
                        verbose=True)
    best = max(pop, key=attrgetter("fitness"))
    return dict(best = best, pop = pop, hof = hof)