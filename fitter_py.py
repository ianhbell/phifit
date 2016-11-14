"""
Simple script to do the optimization
"""
from __future__ import print_function
import time, json, random


# Munge the python path to make it load our module we compiled using pybind11
import sys
sys.path.append('build/pybind11/Release')
import MixtureCoefficientFitter as MCF
from coeffs import ArrayDeconstructor

# Other imports (conda installable packages)
import numpy as np, matplotlib.pyplot as plt, pandas, scipy.optimize

# Our deap optimization routine
from deap_optimize import minimize_deap

#--------------------------------------
#             Instantiation
#--------------------------------------

def get_data():

    Smolen_JSON = open('data/ammonia-water/PTXY-Smolen.json','r').read()
    Smolen_data = json.loads(Smolen_JSON)
    Harms_JSON = open('data/ammonia-water/Harms-Watzenberg.json','r').read()
    Harms_data = json.loads(Harms_JSON)

    all_JSON_data = {'about': Harms_data['about'], 'data': []}
    all_JSON_data['data'] += Smolen_data['data']
    all_JSON_data['data'] += Harms_data['data']

    random.shuffle(all_JSON_data['data'])

    #all_JSON_data['data'] = all_JSON_data['data'][0::4] # every fourth data point

    # Use the new EOS from Kehui and Eric
    all_JSON_data['about']['names'] = ['Ammonia(Hui)','Water']

    print ('# "data" points (+ constraints):', len(all_JSON_data['data']))
    return all_JSON_data

Nterms = 6 # Total number of terms in the summation of terms
Npoly = 6 # Number of "polynomial" terms of the form n_i*tau^t_u*delta^d_i*exp(-cdelta_i*delta^ldelta_i-ctau_i*tau^ltau_i)
Nexp_terms = 5 # Number of terms in each of the inner summations over j for i-th term
Nexp = Nterms - Npoly # Number of "exponential" terms with the full polynomial term in the exponential
Nu = (Nexp*Nexp_terms + Npoly) # Number of total terms associated with one of the things in the exp(u) function

# Bounds for the random generation for each family of parameter
coeffs_bounds = [(0.1, 1.5)]*4
n_bounds = [(-5, 5)]*Nterms
t_bounds = [(0.25, 20)]*Nterms
d_bounds = [(1, 5)]*Nterms
ldelta_bounds = [(1,5)]*Nu
cdelta_bounds = [(-5,5)]*Nu
ltau_bounds = [(0.1,5)]*Nu
ctau_bounds = [(-5,5)]*Nu
bounds = coeffs_bounds + n_bounds + t_bounds + d_bounds + ldelta_bounds + cdelta_bounds + ltau_bounds + ctau_bounds

# Definition of how the array of coefficients should be partitioned into chunks 
# for each variable.  Each entry in the xdims list defines how many 
xu = [1]*Npoly + [Nexp_terms]*Nexp # The term consumption definition for one of the things in the exp(u) function
xdims = [4, Nterms, Nterms, Nterms, xu, xu, xu, xu]

generator_functions = [random.uniform]*(4 + 2*Nterms) + [random.randint]*(Nterms + Nu) + [random.uniform]*Nu*3
normalizing_functions = [lambda x: x]*(4 + 2*Nterms) + [lambda x: min(max(int(round(x)), 1), 5) ]*(Nterms + Nu) + [lambda x: x]*Nu*3
sigma = [0.01]*(4 + 2*Nterms) + [1]*(Nterms + Nu) + [0.01]*Nu*3

# generator_functions = [random.uniform]*(4 + 2*Nterms) + [random.uniform]*(Nterms + Nu) + [random.uniform]*Nu*3
# normalizing_functions = [lambda x: x]*(4 + 2*Nterms) + [lambda x: x]*(Nterms + Nu) + [lambda x: x]*Nu*3
# sigma = [0.01]*(4 + 2*Nterms) + [0.01]*(Nterms + Nu) + [0.01]*Nu*3

# A couple of sanity checks
assert(len(bounds) == len(generator_functions))
assert(len(bounds) == len(normalizing_functions))
assert(len(bounds) == len(sigma))

with open('EWL_NH3_H2O.json','r') as fp:
    departure0 = json.load(fp)

# Instantiate the fitter class with the experimental data stored in JSON format
cfc = MCF.CoeffFitClass(json.dumps(get_data()))
cfc.setup(json.dumps(departure0.copy()))

cfc.evaluate_serial([0.911640, 0.9111660, 1.0541730, 1.3223907])
jj = json.loads(cfc.dump_outputs_to_JSON())
with open('baseline.json', 'w') as fp:
    fp.write(json.dumps(jj, indent = 2))
print('baseline:', cfc.sum_of_squares()**0.5)
cfc.run(True, 4, [0.911640, 0.9111660, 1.0541730, 1.3223907])
jj = json.loads(cfc.dump_outputs_to_JSON())
with open('w_fitting_betasgammas.json', 'w') as fp:
    fp.write(json.dumps(jj, indent = 2))
print('w/ fitting betas, gammas:', cfc.sum_of_squares()**0.5)


def objective(x, cfc, Nterms, fit_delta = True, x0 = None, write_JSON = False):
    if isinstance(x, np.ndarray):
        x = x.tolist()
    # Instantiate the struct holding coefficients for the departure function
    coeffs = MCF.Coefficients()
    
    # Deconstruct the array into the data structures passed to the C++ class
    betagamma, coeffs.n, coeffs.t, coeffs.d, coeffs.ldelta, coeffs.cdelta, coeffs.ltau, coeffs.ctau = ArrayDeconstructor(xdims, x)

    if not fit_delta:
        # The variables d and ldelta need to stay as integers, so we don't let them be modified
        # because the fitter will make them be non-integer values, so we over-write
        # them with the values obtained from the initial values for d and ldelta
        _bg, _n, _t, coeffs.d, coeffs.ldelta, _cdelta, _ltau_els, _ctau_els = ArrayDeconstructor(xdims, x0)

    # Pass the coefficients into the C++ class, set up the departure function
    cfc.setup(coeffs)

    try:
        # cfc.evaluate_serial(betagamma)
        cfc.evaluate_parallel(betagamma, 4)
        err = cfc.sum_of_squares()**0.5
        if np.isnan(err):
            err = 1e8
        if err < 0.1 or write_JSON:
            jj = json.loads(cfc.dump_outputs_to_JSON())
            with open(str(err).replace('.','_') + '.json', 'w') as fp:
                fp.write(json.dumps(jj, indent = 2))
            print(err, 'GOOD')
        else:
            pass
            #print(err)
        return err
    except BaseException as BE:
        print('XX', BE)
        return 1e10

# # Speed testing code (uncomment to run)
# # ------------------
# random.seed(0) # Always have the same "random" values since the seed is set
# N = 20 # This many runs will be made
# # Generate a set of inputs that can be passed to objective function
# x0 = [fcn(b[0], b[1]) for fcn, b in zip(generator_functions, bounds)]
# tic = time.clock()
# for i in range(N):
#     objective(x0, cfc, Nterms, False, x0=x0)
# toc = time.clock()
# print((toc-tic)/N, 's/ind;', (toc-tic)/N/len(get_data()['data'])*1e6, 'us/row')
# sys.exit(-1) # EXIT

# *****************
# *****************
# DEAP OPTIMIZATION
# *****************
# *****************

start_time = time.clock()

print('About to fit', len(bounds), 'coefficients')

# Minimize using deap (global optimization using evolutionary optimization)
results = minimize_deap(objective, bounds, Nindividuals= 250*len(bounds), Ngenerations=40, Nhof = 50, args=(cfc, Nterms), 
                        generator_functions=generator_functions, normalizing_functions=normalizing_functions, sigma=sigma)

# Print elapsed time for this first global optimization
print(time.clock()-start_time, 's for deap optimization')

# Serialize the hall of fame
with open('hof.json', 'w') as fp:
   fp.write(json.dumps([{'c': list(ind), 'fitness': ind.fitness.values} for ind in results['hof']], indent = 2))

# Load the HOF back in from file (this write/load serves as rough serialization and checkpointing)
hof = json.load(open('hof.json','r'))

# ****************
# ****************
# REFINE SOLUTIONS
# ****************
# ****************

# Refine the solutions for each individual
for iind, soln in enumerate(hof):

    # Get the individual (the coefficients)
    ind = soln[u'c']
    
    # Then try to run Nelder-Mead minimization from each promising point in the hall of fame
    # Many options for algorithms here, but Nelder-Mead is well-regarded for its stability (though
    # perhaps not its speed)
    # ----------------------
    fit_delta = False # Do not fit d or ldelta in this phase
    args = (cfc, Nterms, fit_delta, ind[:])
    r = scipy.optimize.minimize(objective, ind[:], method='Nelder-Mead', args=args, options=dict(maxiter = 5000, maxfev = 20000))
    print ('xfinal:', r.x)
    
    # Write out each solution to file in JSON format
    with open('soln{i:04d}.json'.format(i=iind), 'w') as fp:
        fp.write(json.dumps(json.loads(cfc.dump_outputs_to_JSON()), indent=2))

# Print out total elapsed time
print(time.clock()-start_time, 's in total')
