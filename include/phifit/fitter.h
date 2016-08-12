#ifndef PHIFIT_FITTER_H
#define PHIFIT_FITTER_H

#include <vector>
#include <string>

/// The function that actually does the fitting
double simplefit(const std::string &JSON_data_string, const std::string &JSON_fit0_string, bool threading, short Nthreads, std::vector<double> &c0, std::vector<double> &cfinal);

#endif
