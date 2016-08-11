import CoolProp.CoolProp as CP
import json, numpy as np, matplotlib.pyplot as plt

fluids = ['Ethane','n-Propane']
AS = CP.AbstractState('HEOS', '&'.join(fluids))
for k in ['betaT','gammaT','betaV','gammaV','Fij']:
    print AS.get_binary_interaction_double(0,1,k)

json_data = {}
json_data['data'] = []
for T in np.linspace(200,250,20):
    for x0 in np.arange(0.1, 0.91, 0.2):
        x = [x0, 1-x0]
        AS.set_mole_fractions(x)
        AS.update(CP.QT_INPUTS, 0, T)
        y = AS.mole_fractions_vapor()
        pt = {'p (Pa)': AS.p(),
              'T (K)': T,
              'x (molar)': x,
              'y (molar)': y,
              'rho\' (guess,mol/m3)': AS.saturated_liquid_keyed_output(CP.iDmolar),
              'rho\'\' (guess,mol/m3)': AS.saturated_vapor_keyed_output(CP.iDmolar),
              'type': 'PTXY'
              }
        json_data['data'].append(pt)
        mu0 = AS.chemical_potential(0)

        AS.set_mole_fractions(y)
        AS.update(CP.QT_INPUTS, 1, T)
        mu1 = AS.chemical_potential(0)
        print T, x0, mu0,mu1

# Add some metadata
json_data['about'] = {}
json_data['about']['names'] = AS.fluid_names()

# Write to file
with open('_'.join(AS.fluid_names()) + '.json','w') as fp:
    json.dump(json_data, fp, indent =2)