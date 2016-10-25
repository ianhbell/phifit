"""
Simple script to do the optimization
"""
from __future__ import print_function
import time, json, random


# Munge the python path to make it load our module we compiled using pybind11
import sys
sys.path.append('build/pybind11/Release')
import MixtureCoefficientFitter as MCF

# Other imports (conda installable packages)
import numpy as np, matplotlib.pyplot as plt, pandas, scipy.optimize

# Our deap optimization routine
from deap_optimize import minimize_deap

def random_dist(dims, minval, maxval):
    return (maxval-minval)*np.random.random(dims) + minval

#--------------------------------------
#             Instantiation
#--------------------------------------

with open('fit0.json','r') as fp:
    departure0 = json.load(fp)

def get_data():

    Smolen_JSON = open('data/ammonia-water/PTXY-Smolen.json','r').read()
    Smolen_data = json.loads(Smolen_JSON)
    Harms_JSON = open('data/ammonia-water/Harms-Watzenberg.json','r').read()
    Harms_data = json.loads(Harms_JSON)

    all_JSON_data = {'about': Harms_data['about'], 'data': []}
    all_JSON_data['data'] += Smolen_data['data']
    all_JSON_data['data'] += Harms_data['data']

    #all_JSON_data['data'] = all_JSON_data['data'][0::4] # every fourth data point

    # Use the new EOS from Kehui and Eric
    all_JSON_data['about']['names'] = ['Ammonia(Hui)','Water']

    print ('# data points:', len(all_JSON_data['data']))
    return all_JSON_data

cfc = MCF.CoeffFitClass(json.dumps(get_data()))

Nterms = 7
Npoly = Nterms
departure0 = departure0.copy()

departure0['departure[ij]']['n'] = [0]*Nterms
departure0['departure[ij]']['d'] = [0]*Nterms
departure0['departure[ij]']['t'] = [0]*Nterms
departure0['departure[ij]']['ctau'] = [[0]]*Nterms
departure0['departure[ij]']['ltau'] = [[0]]*Nterms
departure0['departure[ij]']['cdelta'] = [[-1]]*Nterms
departure0['departure[ij]']['ldelta'] = [[0]]*Nterms

departure = departure0.copy()
cfc.setup(json.dumps(departure))

cfc.evaluate_serial([1,1,1,1])
with open('baseline.json', 'w') as fp:
    fp.write(json.dumps(json.loads(cfc.dump_outputs_to_JSON()), indent=2))

coeffs = MCF.Coefficients()

def chunkify(l, lengths):
    ranges = []
    istart = 0
    for length in lengths:
        ranges.append((istart, istart+length))
        istart += length
    if ranges[-1][1] != len(l):
        raise AssertionError('length of list [{0:d}] not equal to last range el [{1:d}'.format(len(l), ranges[-1][1]))

    return [l[range_[0]:range_[1]] for range_ in ranges]

def chunks(l, n):
    n = max(1, n)
    return [l[i:i+n] for i in xrange(0, len(l), n)]

def objective(x, departure, cfc, Nterms, Npoly, fit_delta = True, x0 = None, write_JSON = False):
    if isinstance(x, np.ndarray):
        x = x.tolist()
    Nexp_delta = 1
    Nexp_tau = 1
    betagamma,coeffs.n,coeffs.t,d, ldelta_els, cdelta_els = chunkify(x, [4, Nterms, Nterms, Nterms, Nexp_delta*Nterms, Nexp_delta*Nterms])
    if fit_delta:
        coeffs.d = d
        coeffs.ldelta = chunks(ldelta_els,1)
        coeffs.cdelta = chunks(cdelta_els,1)
    else:
        junk, d0, ldelta_els0, cdelta_els0 = chunkify(x0, [4 + 2*Nterms, Nterms, Nexp_delta*Nterms, Nexp_delta*Nterms])
        coeffs.d = d0
        coeffs.ldelta = chunks(ldelta_els0,1)
        coeffs.cdelta = chunks(cdelta_els0,1)
    cfc.setup(coeffs)

    try:
        cfc.evaluate_parallel(betagamma, 4)
        err = cfc.sum_of_squares()**0.5
        if err < 0.1 or write_JSON:
            jj = json.loads(cfc.dump_outputs_to_JSON())
            with open(str(err).replace('.','_') + '.json', 'w') as fp:
                fp.write(json.dumps(jj, indent = 2))
            print(err, 'GOOD')
        else:
            print(err)
        return err
    except BaseException as BE:
        print('XX', BE)
        return 1e10

start_time = time.clock()

# Generate a set of inputs that can be passed to objective function
x0 = [0.911640, 0.9111660, 1.0541730, 1.3223907] + departure['departure[ij]']['n'] + departure['departure[ij]']['t'] + departure['departure[ij]']['d']
x0 += [_[0] for _ in departure['departure[ij]']['ldelta'][0:Npoly]]
for els in departure['departure[ij]']['cdelta'][Npoly::]:
    x0 += els

# N = 20
# tic = time.clock()
# for i in range(N):
#     objective(x0, departure, cfc, Nterms, Npoly)
# toc = time.clock()
# print((toc-tic)/N, 's/eval')
# sys.exit(-1)
    
coeffs_bounds = [(0.1, 1.5)]*4
n_bounds = [(-5, 5)]*Nterms
t_bounds = [(0.25, 5)]*Nterms
d_bounds = [(0, 5)]*Nterms
ldelta_bounds = [(0,3)]*Npoly
cdelta_bounds = [(-5,0)]*Npoly
bounds = coeffs_bounds + n_bounds + t_bounds + d_bounds + ldelta_bounds + cdelta_bounds
args=(departure, cfc, Nterms, Npoly)

generator_functions = [random.uniform]*(4 + 2*Nterms) + [random.randint]*(Nterms*2) + [random.uniform]*Nterms
normalizing_functions = [lambda x: x]*(4 + 2*Nterms) + [lambda x: int(round(x))]*(Nterms*2) + [lambda x: x]*Nterms

print('About to fit ', len(bounds), 'coefficients')

# Minimize using deap (global optimization using evolutionary optimization)
results = minimize_deap(objective, bounds, Nindividuals=400, Ngenerations=20, Nhof = 20, args=args, 
                        generator_functions = generator_functions, normalizing_functions = normalizing_functions)
print(time.clock()-start_time, 's for deap optimization')

# Serialize the hall of fame
with open('hof.json', 'w') as fp:
   fp.write(json.dumps([{'c': list(ind), 'fitness': ind.fitness.values} for ind in results['hof']], indent = 2))

# Load the HOF back in from file (this serves as rough serialization)
hof = json.load(open('hof.json','r'))

# Refine the solutions for each individual
for iind, soln in enumerate(hof):
    # Get the individual (the coefficients)
    ind = soln[u'c']
    # Then try to run Nelder-Mead minimization from each promising point in the hall of fame
    # Many options for algorithms here, but Nelder-Mead is well-regarded for its stability (though
    # perhaps not its speed)
    fit_delta = False # Do not fit d, ldelta, or cdelta in this phase
    args=(departure, cfc, Nterms, Npoly, fit_delta, ind)
    r = scipy.optimize.minimize(objective, ind, method='Nelder-Mead', args=args, options=dict(maxiter = 50, maxfev = 20))
    print ('xfinal:', r.x)
    with open('soln{i:04d}.json'.format(i=iind), 'w') as fp:
        fp.write(json.dumps(json.loads(cfc.dump_outputs_to_JSON()), indent=2))
print(time.clock()-start_time, 's in total')
