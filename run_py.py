"""
Simple script to test the pybind11 interface
"""
import sys
sys.path.append('build/pybind11/Release')
import MixtureCoefficientFitter as MCF
data_JSON_string = open('ammonia_water.json','r').read()
fit0_JSON_string = open('fit0.json','r').read()
cfc = MCF.CoeffFitClass(data_JSON_string)
cfc.setup(fit0_JSON_string)
cfc.run(True, 4, [1.0, 1.0, 1.0, 1.0])
print cfc.elapsed_sec()
print cfc.cfinal()