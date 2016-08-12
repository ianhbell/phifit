#include "externals/Catch/single_include/catch.hpp"
#include "phifit/data_generation.h"
#include "phifit/fitter.h"
#include "crossplatform_shared_ptr.h"
#include "AbstractState.h"

TEST_CASE("Test fitting betas,gammas", "[simple]") {
    std::string backend = "HEOS", names="Methane&Ethane";
    std::string data = gen_JSON_data(backend, names);
    shared_ptr<CoolProp::AbstractState> AS(CoolProp::AbstractState::factory(backend, names));
    double betaT_old = AS->get_binary_interaction_double(0,1,"betaT");
    double gammaT_old = AS->get_binary_interaction_double(0, 1, "gammaT");
    double betaV_old = AS->get_binary_interaction_double(0, 1, "betaV");
    double gammaV_old = AS->get_binary_interaction_double(0, 1, "gammaV");
    std::vector<double> c0 = {1,1,1,1}, cfinal;
    CHECK_NOTHROW(simplefit(data, "", false, 1, c0, cfinal));
    REQUIRE(betaT_old - cfinal[0] < 1e-3);
    REQUIRE(gammaT_old - cfinal[1] < 1e-3);
    REQUIRE(betaV_old - cfinal[2] < 1e-3);
    REQUIRE(gammaV_old - cfinal[3] < 1e-3);
}