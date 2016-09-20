// Includes from Catch
#include "externals/Catch/single_include/catch.hpp"

// Includes from PhiFit
#include "phifit/data_generation.h"
#include "phifit/fitter.h"

// Includes from CoolProp
#include "AbstractState.h"

// Includes from standard library
#include<memory>

TEST_CASE("Test fitting betas,gammas", "[simple]") {
    std::string backend = "HEOS", names="Ethane&n-Propane";
    std::string JSON_fit0_string = get_file_contents("../../fit0.json");
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

    REQUIRE(betaT_old - cfinal[0] < 1e-2);
    REQUIRE(gammaT_old - cfinal[1] < 1e-2);
    REQUIRE(betaV_old - cfinal[2] < 1e-2);
    REQUIRE(gammaV_old - cfinal[3] < 1e-2);
}