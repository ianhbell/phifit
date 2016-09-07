#ifndef PHIFIT_DEPARTURE_FUNCTION_H
#define PHIFIT_DEPARTURE_FUNCTION_H

#include "Backends/Helmholtz/HelmholtzEOSMixtureBackend.h"
#include "Backends/Helmholtz/ExcessHEFunction.h"

class PhiFitDepartureFunction : public CoolProp::DepartureFunction
{
private:
    std::vector<double> n, t, d;
    std::vector<std::vector<double> > c, l, omega, m;
public:
    PhiFitDepartureFunction(rapidjson::Value &JSON_data) ;
    void update(double tau, double delta);
};

#endif
