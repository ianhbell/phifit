# Standard library imports
import time
import random
from operator import attrgetter
from collections import Sequence
from itertools import repeat

# Conda-installable packages
import numpy as np

# Imports from deap
from deap import algorithms, base, creator, tools
from deap.algorithms import varAnd

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

    size = len(individual)

    # Don't normalize if not provided
    if normalizing_functions is None:
        normalizing_functions = [lambda x: x]*size # this dummy lambda function just returns its argument

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

def myEaSimple(population, toolbox, cxpb, mutpb, ngen, stats=None,
             halloffame=None, verbose=__debug__):
    """This algorithm reproduce the simplest evolutionary algorithm as
    presented in chapter 7 of [Back2000]_.
    :param population: A list of individuals.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :param ngen: The number of generation.
    :param stats: A :class:`~deap.tools.Statistics` object that is updated
                  inplace, optional.
    :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                       contain the best individuals, optional.
    :param verbose: Whether or not to log the statistics.
    :returns: The final population
    :returns: A class:`~deap.tools.Logbook` with the statistics of the
              evolution
    The algorithm takes in a population and evolves it in place using the
    :meth:`varAnd` method. It returns the optimized population and a
    :class:`~deap.tools.Logbook` with the statistics of the evolution. The
    logbook will contain the generation number, the number of evalutions for
    each generation and the statistics if a :class:`~deap.tools.Statistics` is
    given as argument. The *cxpb* and *mutpb* arguments are passed to the
    :func:`varAnd` function. The pseudocode goes as follow ::
        evaluate(population)
        for g in range(ngen):
            population = select(population, len(population))
            offspring = varAnd(population, toolbox, cxpb, mutpb)
            evaluate(offspring)
            population = offspring
    As stated in the pseudocode above, the algorithm goes as follow. First, it
    evaluates the individuals with an invalid fitness. Second, it enters the
    generational loop where the selection procedure is applied to entirely
    replace the parental population. The 1:1 replacement ratio of this
    algorithm **requires** the selection procedure to be stochastic and to
    select multiple times the same individual, for example,
    :func:`~deap.tools.selTournament` and :func:`~deap.tools.selRoulette`.
    Third, it applies the :func:`varAnd` function to produce the next
    generation population. Fourth, it evaluates the new individuals and
    compute the statistics on this population. Finally, when *ngen*
    generations are done, the algorithm returns a tuple with the final
    population and a :class:`~deap.tools.Logbook` of the evolution.
    .. note::
        Using a non-stochastic selection method will result in no selection as
        the operator selects *n* individuals from a pool of *n*.
    This function expects the :meth:`toolbox.mate`, :meth:`toolbox.mutate`,
    :meth:`toolbox.select` and :meth:`toolbox.evaluate` aliases to be
    registered in the toolbox.
    .. [Back2000] Back, Fogel and Michalewicz, "Evolutionary Computation 1 :
       Basic Algorithms and Operators", 2000.

    Originally from DEAP v. 1.1.0, modified by Ian Bell, NIST
    """
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals', 'elapsed'] + (stats.fields if stats else [])

    # Start timing this generation
    start_time = time.clock()
        
    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    # Add the elapsed time for the generation to the record dictionary
    record['elapsed'] = time.clock() - start_time
    if verbose:
        print logbook.stream

    # Begin the generational process
    for gen in range(1, ngen + 1):

        # Start timing this generation
        start_time = time.clock()

        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))

        # Vary the pool of individuals
        offspring = varAnd(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        # Add the elapsed time for the generation to the record dictionary
        record['elapsed'] = time.clock() - start_time
        # Store the statistics in the logbook
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print logbook.stream

    return population, logbook     

def minimize_deap(f, bounds, Nindividuals = 5000, Ngenerations = 20, Nhof = 50, 
                  generator_functions = None, normalizing_functions = None,
                  sigma = None, **kwargs):
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
    sigma : Constant standard deviation for all coefficients in individual or 
        :term:`python:sequence` of standard deviations for the gaussian addition mutation.
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

    toolbox.register("mutate", myMutateGaussian, mu = 0, sigma = sigma, indpb=1.0, normalizing_functions = normalizing_functions)
    toolbox.register("select", tools.selTournament, tournsize=3) # The greater the tournament size, the greater the selection pressure 
    
    hof = tools.HallOfFame(Nhof)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min fitness", np.min)
    stats.register("stddev fitness", np.std)
    
    pop = toolbox.population(n=Nindividuals)
    pop, log = myEaSimple(pop, 
                          toolbox, 
                          cxpb=0.5, # Crossover probability
                          mutpb=0.3, # Mutation probability
                          ngen=Ngenerations, 
                          stats=stats, 
                          halloffame=hof, 
                          verbose=True)
    best = max(pop, key=attrgetter("fitness"))
    return dict(best = best, pop = pop, hof = hof)