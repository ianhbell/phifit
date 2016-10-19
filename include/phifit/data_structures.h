#ifndef DATASTRUCTURES_H
#define DATASTRUCTURES_H

class Coefficients {
public:
    std::vector<double> n, t, d;
    std::vector<std::vector<double> > ldelta, cdelta, ltau, ctau;
};

#endif
