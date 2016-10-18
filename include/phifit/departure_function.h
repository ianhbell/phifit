#ifndef PHIFIT_DEPARTURE_FUNCTION_H
#define PHIFIT_DEPARTURE_FUNCTION_H

#include "Backends/Helmholtz/HelmholtzEOSMixtureBackend.h"
#include "Backends/Helmholtz/ExcessHEFunction.h"

class PhiFitDepartureFunction : public CoolProp::DepartureFunction
{
private:
    std::vector<double> n, t, d;
    std::vector<std::vector<double> > cdelta, ldelta, ctau, ltau;
public:
    PhiFitDepartureFunction(rapidjson::Value &JSON_data) ;
    void update(double tau, double delta);
    rapidjson::Value to_JSON(rapidjson::Document &doc);
    void set_n(const std::vector<double> &n) { this->n = n ; }
    void set_nt(const std::vector<double> &n, const std::vector<double> &t) { this->n = n; this->t = t; }
};

#endif
