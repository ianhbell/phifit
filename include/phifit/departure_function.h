#ifndef PHIFIT_DEPARTURE_FUNCTION_H
#define PHIFIT_DEPARTURE_FUNCTION_H

#include "Backends/Helmholtz/HelmholtzEOSMixtureBackend.h"
#include "Backends/Helmholtz/ExcessHEFunction.h"

#include "phifit/data_structures.h"

class PhiFitDepartureFunction : public CoolProp::DepartureFunction
{
private:
    std::vector<double> n, t, d;
    std::vector<std::vector<double> > cdelta, ldelta, ctau, ltau;
public:
    PhiFitDepartureFunction(rapidjson::Value &JSON_data) ;
    void update(double tau, double delta);
    rapidjson::Value to_JSON(rapidjson::Document &doc);
    void update_coeffs(const Coefficients &coeffs){ this->n = coeffs.n; this->t = coeffs.t; this->d = coeffs.d; this->ldelta = coeffs.ldelta; this->cdelta = coeffs.cdelta; }
};

#endif
