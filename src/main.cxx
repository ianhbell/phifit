
#include "phifit/data_generation.h"
#include "phifit/fitter.h"

#include "AbstractState.h"

int main() {

    std::string JSON_data_string = gen_JSON_data("HEOS", "R32&n-Propane");

    std::vector<double> c0 = {1,1,1,1}, cfinal;
    fmt::printf("%g\n", simplefit(JSON_data_string, "", false, 1, c0, cfinal));
    for (auto &Nthreads : { 1,2,3,4,5,6,7,8 }) {
        fmt::printf("%d %g\n", Nthreads, simplefit(JSON_data_string, "", true, Nthreads, c0, cfinal));
    }
}