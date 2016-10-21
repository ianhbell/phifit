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

with open('EWL_NH3_H2O.json','r') as fp:
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

for i in range(1):
    departure = departure0.copy()
    cfc.setup(json.dumps(departure))

    # if i == 0:
    #     cfc.setup(json.dumps(departure))
    #     cfc.evaluate_serial([0.911640, 0.9111660, 1.0541730, 1.3223907])
    #     with open('baseline.json', 'w') as fp:
    #         fp.write(json.dumps(json.loads(cfc.dump_outputs_to_JSON()), indent = 2, sort_keys = True))
    #     print('BASELINE:', cfc.sum_of_squares()**0.5)
    
    Nterms = 13
    Npoly = 9

    coeffs = MCF.Coefficients()

    def chunks(l, n):
        n = max(1, n)
        return [l[i:i+n] for i in xrange(0, len(l), n)]

    def objective(x, departure, cfc, Nterms, Npoly, write_JSON = False):
        if isinstance(x, np.ndarray):
            x = x.tolist()
        N = len(x)
        betagamma = x[0:4]
        coeffs.n = x[4:4+Nterms]
        coeffs.t = x[4+Nterms:4+2*Nterms]
        coeffs.d = x[4+2*Nterms:4+3*Nterms]
        coeffs.ldelta = [[_] for _ in x[4+3*Nterms:4+3*Nterms+Npoly]] + departure['departure[ij]']['ldelta'][Npoly::]
        coeffs.cdelta = departure['departure[ij]']['cdelta'][0:Npoly] + chunks(x[4+3*Nterms+Npoly:4+3*Nterms+Npoly+3*(Nterms-Npoly)+1], 3)
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
        
    coeffs_bounds = [(0.1, 1.5) for _ in range(4)]
    n_bounds = [(-5, 5)]*Nterms
    t_bounds = [(0.25, 5)]*Nterms
    d_bounds = [(0.25, 6)]*Nterms
    ldelta_bounds = [(0,3)]*Npoly
    cdelta_bounds = [(-5,5)]*(Nterms-Npoly)*3
    bounds = coeffs_bounds + n_bounds + t_bounds + d_bounds + ldelta_bounds + cdelta_bounds
    args=(departure, cfc, Nterms, Npoly)

    # Minimize using deap (global optimization using evolutionary optimization)
    results = minimize_deap(objective, bounds, Nindividuals=7000, Ngenerations=20, args=args)
    # Serialize the hall of fame
    with open('hof.json', 'w') as fp:
        fp.write(json.dumps([{'c': list(ind), 'fitness': ind.fitness.values} for ind in results['hof']], indent = 2))
    # Refine the solutions for each individual
    for iind, ind in enumerate(results['hof']):
        print ('ind:', ind)
        # Then try to run Nelder-Mead minimization from each promising point in the hall of fame
        # Many options for algorithms here, but Nelder-Mead is well-regarded for its stability (though
        # perhaps not its speed)
        r = scipy.optimize.minimize(objective, ind, method='Nelder-Mead', args=args, options=dict(maxiter = 5000, maxfev = 20000))
        print ('xfinal:', r.x)
        with open('soln{i:04d}.json'.format(i=iind), 'w') as fp:
            fp.write(json.dumps(json.loads(cfc.dump_outputs_to_JSON()), indent=2))
    
    # Differential evolution (global optimization)
    # result = scipy.optimize.differential_evolution(objective, bounds, args=(departure, cfc, Nterms, Npoly), disp = True)

    # Nelder-Mead (local optimization)
    # result = scipy.optimize.minimize(objective, x0, method='Nelder-Mead', args=(departure, cfc, Nterms, Npoly), options=dict(maxiter = 12000))
    # print ('xfinal:', result.x)

