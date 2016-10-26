// Includes from Catch
#include "externals/CoolProp/externals/Catch/single_include/catch.hpp"

// Includes from PhiFit
#include "phifit/data_generation.h"
#include "phifit/fitter.h"

// Includes from CoolProp
#include "AbstractState.h"

// Includes from standard library
#include<memory>

TEST_CASE("Test fitting betas,gammas", "[simple]") {
    std::string backend = "HEOS", names="Ethane&n-Propane";
    std::string data = gen_JSON_data(backend, names);
    std::shared_ptr<CoolProp::AbstractState> AS(CoolProp::AbstractState::factory(backend, names));
    double betaT_old = AS->get_binary_interaction_double(0,1,"betaT");
    double gammaT_old = AS->get_binary_interaction_double(0, 1, "gammaT");
    double betaV_old = AS->get_binary_interaction_double(0, 1, "betaV");
    double gammaV_old = AS->get_binary_interaction_double(0, 1, "gammaV");
    std::vector<double> c0 = {1,1,1,1}, cfinal;

    CoeffFitClass CFC(data);
    bool threading = false; int Nthreads = 1;
    REQUIRE_NOTHROW(CFC.run(threading, Nthreads, c0));
    cfinal = CFC.cfinal();
    //for (int i = 0; i < cfinal.size(); i += 1) { std::cout << cfinal[i] << std::endl; }

    REQUIRE(betaT_old - cfinal[0] < 1e-3);
    REQUIRE(gammaT_old - cfinal[1] < 1e-3);
    REQUIRE(betaV_old - cfinal[2] < 1e-3);
    REQUIRE(gammaV_old - cfinal[3] < 1e-3);
}

TEST_CASE("Test applying new departure function", "[set_departure_function]") {

    /// Converted version of departure function from GERG - see conversion script
    std::string new_dep = R"(
        {
            "departure[ij]": {"BibTeX": "Kunz-JCED-2012", "Name": "Methane-Propane", "Npower": 5, "aliases": ["KW2"], 
            "cdelta": [[-0.0, 0.0, 0.0], [-0.0, 0.0, 0.0], [-0.0, 0.0, 0.0], [-0.0, 0.0, 0.0], [-0.0, 0.0, 0.0], [-0.25, -0.5, 0.3125], [-0.25, -0.75, 0.4375], [-0.0, -2.0, 1.0], [-0.0, -3.0, 1.5]], 
            "ctau": [[0], [0], [0], [0], [0], [0], [0], [0], [0]], 
            "d": [3, 3, 4, 4, 4, 1, 1, 1, 2], 
            "ldelta": [[2, 1, 0], [2, 1, 0], [2, 1, 0], [2, 1, 0], [2, 1, 0], [2, 1, 0], [2, 1, 0], [2, 1, 0], [2, 1, 0]], 
            "ltau": [[0], [0], [0], [0], [0], [0], [0], [0], [0]], 
            "n": [0.013746429958576, -0.0074425012129552, -0.0045516600213685, -0.0054546603350237, 0.0023682016824471, 0.18007763721438, -0.44773942932486, 0.0193273748882, -0.30632197804624], 
            "t": [1.85, 3.95, 0.0, 1.85, 3.85, 5.25, 3.85, 0.2, 6.5], 
            "type": "GERG-2008"}
        }
    )";

    std::string backend = "HEOS", names = "Methane&n-Propane";

    // Generate sample data in JSON format
    gen_JSON_data_options o;
    o.Tmax = 200; o.Tmin = 100; 
    std::string data = gen_JSON_data(backend, names, o);
    
    std::shared_ptr<CoolProp::AbstractState> AS(CoolProp::AbstractState::factory(backend, names));
    std::vector<double> c0 = { 1,1,1,1 }, cfinal0, cfinal1;

    CoeffFitClass CFC(data);
    bool threading = false; int Nthreads = 1;
    
    // With the default departure term - the baseline evaluation
    REQUIRE_NOTHROW(CFC.run(threading, Nthreads, c0));
    cfinal0 = CFC.cfinal();

    // With a 1-to-1 translation of the departure term from GERG to new form - should get the same result!!
    CFC = CoeffFitClass(data);
    CFC.setup(new_dep);
    REQUIRE_NOTHROW(CFC.run(threading, Nthreads, c0));
    cfinal1 = CFC.cfinal();

    // Check that all are the same
    REQUIRE(cfinal0.size() == cfinal1.size());
    for (std::size_t i = 0; i < cfinal0.size(); ++i) {
        CHECK(std::abs(cfinal1[i] - cfinal0[i]) < 1e-6);
    }
}