
#include "phifit/data_generation.h"
#include "phifit/fitter.h"

#include "AbstractState.h"

int main() {
    std::string JSON_data_string = get_file_contents("../../ammonia_water.json");
    std::string JSON_fit0_string = get_file_contents("../../fit0.json");
    std::vector<double> c0 = { 0.911640,0.9111660,1.0541730, 1.3223907 }, cfinal;
    fmt::printf("%g\n", simplefit(JSON_data_string, JSON_fit0_string, false, 4, c0, cfinal));
    for (int i = 0; i < cfinal.size(); i += 1) { std::cout << cfinal[i] << "," << std::endl; }
    for (auto &Nthreads : { 1,2,3,4,5,6,7,8 }) {
        fmt::printf("%d %g\n", Nthreads, simplefit(JSON_data_string, JSON_fit0_string, true, Nthreads, c0, cfinal));
    }
}