#ifndef DATA_GENERATION_H
#define DATA_GENERATION_H

#include <string>

/// Options for the data generation script
struct gen_JSON_data_options {
    double Tmin, Tmax, x0min, x0max;
    gen_JSON_data_options() : Tmin(200), Tmax(250), x0min(0.1), x0max(0.91) {};
};

/// Generate some data for fitting purposes
std::string gen_JSON_data(const std::string &backend, const std::string &names, gen_JSON_data_options options = gen_JSON_data_options());

#endif