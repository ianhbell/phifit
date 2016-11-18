""" Just a little script to spit out some diagnostics for the solutions """
from __future__ import print_function
import matplotlib.pyplot as plt, pandas, json, glob, sys, numpy as np

#for soln in glob.glob('soln*.json') + ['baseline.json', 'EthanolWaterFit.json']:
results = []
for soln in ['baseline.json', 'GeneralizedAirWaterFit.json']:
	data = json.loads(open(soln,'r').read())['data']
	df = pandas.DataFrame(data)
	PRhoT = df[df['type'] == 'PRhoT']
	PTXY = df[df['type'] == 'PTXY']
	df.to_excel(soln.rsplit('.')[0]+'.xlsx')
	R = ((df['residue']**2).sum())**0.25
	results.append(dict(fname=soln, Np=len(df), residue=R, 
		                 mean_abs_PRhoT_res=np.mean(np.abs(PRhoT['residue']))*100, mean_abs_PTXY_res=np.mean(np.abs(PTXY['residue'])),
	                 	 max_abs_PRhoT_res=np.max(np.abs(PRhoT['residue']))*100, max_abs_PTXY_res=np.max(np.abs(PTXY['residue']))
	                 	))
print(pandas.DataFrame(results))