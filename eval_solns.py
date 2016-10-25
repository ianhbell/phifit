""" Just a little script to spit out some diagnostics for the solutions """
from __future__ import print_function
import matplotlib.pyplot as plt, pandas, json, glob, sys, numpy as np

print('name\t Npoints \t residual\t MARE(rho)[%]\tr(PTXY)')
for soln in glob.glob('soln*.json'):
	data = json.loads(open(soln,'r').read())['data']
	df = pandas.DataFrame(data)
	PRhoT = df[df['type'] == 'PRhoT']
	PTXY = df[df['type'] == 'PTXY']
	print(soln, len(df), ((df['residue']**2).sum())**0.25, np.mean(np.abs(PRhoT['residue']))*100, np.mean(np.abs(PTXY['residue']))*100)