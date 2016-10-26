#ifndef PHIFIT_FITTER_H
#define PHIFIT_FITTER_H

#include <vector>
#include <string>

// Includes from NISTfit
#include "NISTfit/abc.h"

// Includes from phifit
#include "phifit/data_structures.h"

/// The function that actually does the fitting
double simplefit(const std::string &JSON_data_string, const std::string &JSON_fit0_string, bool threading, short Nthreads, std::vector<double> &c0, std::vector<double> &cfinal);

class CoeffFitClass
{
public:
    std::shared_ptr<NISTfit::AbstractEvaluator> m_eval;
    std::vector<double> m_cfinal;
    double m_elap_sec;

    /// Instantiator
    CoeffFitClass(const std::string &JSON_data_string);
    /// Setup the departure function
    void setup(const std::string &JSON_fit0_string);
    /// Setup the departure function using coefficients passed as a Coefficients class instance
    void setup(const Coefficients &coeffs);
    /// Run the optimizer
    void run(bool threading, short Nthreads, const std::vector<double> &c0);
    /// Just evaluate the residual vector (serially), and cache values internally
    void evaluate_serial(const std::vector<double> &c0);
    /// Just evaluate the residual vector (in parallel), and cache values internally
    void evaluate_parallel(const std::vector<double> &c0, short Nthreads);
    /// Accessor for final values
    std::vector<double> cfinal() { return m_cfinal; }
    /// Accessor for elapsed time
    double elapsed_sec() { return m_elap_sec; }
    /// The sum of squares (residual) that is the current best value
    double sum_of_squares();
    /// Return the error vector from the evaluator
    std::vector<double> errorvec();
    /// Return all the outputs in JSON form, in a form similar to the input JSON structure, plus any additional metadata desired
    std::string dump_outputs_to_JSON();
    /// Dump the departure function to JSON
    std::string departure_function_to_JSON();
};

#endif
