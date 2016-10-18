"""
Simple script to do the optimization
"""
from __future__ import print_function
import time, json

# Munge the python path to make it load our module we compiled using pybind11
import sys
sys.path.append('build/pybind11/Release')
import MixtureCoefficientFitter as MCF

# Other imports (conda installable packages)
import numpy as np, matplotlib.pyplot as plt, pandas, scipy.optimize

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

    # Use the new EOS from Kehui and Eric
    all_JSON_data['about']['names'] = ['Ammonia(Hui)','Water']

    print ('# data points:', len(all_JSON_data['data']))
    return all_JSON_data

cfc = MCF.CoeffFitClass(json.dumps(get_data()))
coeffs = [1,1,1,1]

for i in range(1):
    departure = departure0.copy()
    cfc.setup(json.dumps(departure))

    if i == 0:
        cfc.setup(json.dumps(departure))
        cfc.evaluate_serial([0.911640, 0.9111660, 1.0541730, 1.3223907])
        with open('baseline.json', 'w') as fp:
            fp.write(json.dumps(json.loads(cfc.dump_outputs_to_JSON()), indent = 2, sort_keys = True))
        print('BASELINE:', cfc.sum_of_squares()**0.5)
    
    Nterms = 13
    Npoly = 9
    Nexp_tau = 0
    Nexp_delta = 3

    if i > 0:
        departure['departure[ij]']['n'] = (random_dist((1, Nterms), -5, 5)).tolist()[0]
        departure['departure[ij]']['t'] = (random_dist((1, Nterms), 0.2, 2.5)).tolist()[0]
        departure['departure[ij]']['d'] = (np.floor(random_dist((1, Nterms), 0, 4))).tolist()[0]

        departure['departure[ij]']['ldelta'] = [[_] for _ in random_dist((1, Npoly), 0,3).tolist()[0]] + [range(Nexp_delta-1,-1,-1) for j in range(Nterms-Npoly)]
        departure['departure[ij]']['cdelta'] = [[-1] for _ in range(Npoly)] + [(random_dist((1, Nexp_delta), -2,0)).tolist()[0] for j in range(Nterms-Npoly)]
        departure['departure[ij]']['ltau'] = [[0] for _ in range(Npoly)] + [(np.floor(random_dist((1, Nexp_tau), 0, 4))).tolist()[0] for j in range(Nterms-Npoly)]
        departure['departure[ij]']['ctau'] = [[0] for _ in range(Npoly)] + [(random_dist((1, Nexp_tau), -2,0)).tolist()[0] for j in range(Nterms-Npoly)]

    def objective(x, departure, cfc, Nterms, Npoly, write_JSON = False):
        
        x = x.tolist()
        N = len(x)
        coeffs = x[0:4]
        departure['departure[ij]']['n'] = x[4:4+Nterms]
        departure['departure[ij]']['t'] = x[4+Nterms:4+2*Nterms]
        departure['departure[ij]']['d'] = x[4+2*Nterms:4+3*Nterms]
        departure['departure[ij]']['ldelta'][0:Npoly] = [[_] for _ in x[4+3*Nterms:4+3*Nterms+Npoly]]
        cfc.setup(json.dumps(departure))
        try:
            #cfc.evaluate_serial(coeffs)
            cfc.evaluate_parallel(coeffs, 4)
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

    # N = 200
    # n = np.array(departure['departure[ij]']['n'])
    # tic = time.clock()
    # for i in range(N):
    #     objective(n, departure, cfc)
    # toc = time.clock()
    # print((toc-tic)/N, 's/eval')
    # sys.exit(-1)
    # print(objective(n, departure, cfc))
    # sys.exit(-1)
    

    # # Differential evolution (global optimization)
    # coeffs_bounds = [(0.1, 1.5) for _ in range(4)]
    # n_bounds = [(-5, 5)]*Nterms
    # t_bounds = [(0.25, 5)]*Nterms
    # d_bounds = [(0.25, 4)]*Nterms
    # ldelta_bounds = [(0,3)]*Npoly
    # bounds = coeffs_bounds + n_bounds + t_bounds + d_bounds + ldelta_bounds
    # result = scipy.optimize.differential_evolution(objective, bounds, args=(departure, cfc, Nterms, Npoly), disp = True)

    # Nelder-Mead (local optimization)
    x0 = [0.911640, 0.9111660, 1.0541730, 1.3223907] + departure['departure[ij]']['n'] + departure['departure[ij]']['t'] + departure['departure[ij]']['d'] + [_[0] for _ in departure['departure[ij]']['ldelta']]
    result = scipy.optimize.minimize(objective, x0, method='Nelder-Mead', args=(departure, cfc, Nterms, Npoly), options=dict(maxiter = 12000))
    print ('xfinal:', result.x)
    objective(result.x, departure, cfc, Nterms, Npoly, write_JSON = True)