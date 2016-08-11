//
//  Inspired by http://www.drdobbs.com/cpp/c11s-async-template/240001196
//  See also http://www2.imm.dtu.dk/pubdb/views/edoc_download.php/3215/pdf/imm3215.pdf
//

// Includes from NISTfit
#include "NISTfit/abc.h"
#include "NISTfit/optimizers.h"
#include "NISTfit/numeric_evaluators.h"

// Includes from CoolProp
#include "crossplatform_shared_ptr.h"
#include "AbstractState.h"
#include "rapidjson_include.h"
#include "CoolProp.h"
#include "Configuration.h"
#include "Backends/Helmholtz/MixtureParameters.h"
#include "Backends/Helmholtz/HelmholtzEOSMixtureBackend.h"

// Includes from c++
#include <iostream>
#include <chrono>

using namespace NISTfit;

/// The data structure used to hold an input to Levenberg-Marquadt fitter for parallel evaluation
/// Does not have any of its own routines
class PTXYInput : public NumericInput
{
protected:
    shared_ptr<CoolProp::AbstractState> AS;
    double m_T, //< Temperature (K)
           m_p; //< Pressure (Pa)
    std::vector<double> m_x, //< Molar composition of liquid
                        m_y; //< Molar composition of vapor
    double m_rhoL, m_rhoV;
public:
    /*
     @param AS AbstractState to be used for fitting
     @param T Temperature in K
     @param p Pressure in Pa
     */
    PTXYInput(shared_ptr<CoolProp::AbstractState> &AS, double T, double p, const std::vector<double>&x, const std::vector<double> &y, double rhoL, double rhoV)
        : AS(AS), NumericInput(T, p), m_T(T), m_p(p), m_x(x), m_y(y), m_rhoL(rhoL), m_rhoV(rhoV)  {};
    /// Return a reference to the AbstractState being modified
    shared_ptr<CoolProp::AbstractState> &get_AS() { return AS; }
    /// Get the temperature (K)
    double T(){ return m_T; }
    /// Get the pressure (Pa)
    double p() { return m_p; }
    /// Get the liquid mole fractions
    const std::vector<double> &x() { return m_x; }
    /// Get the vapor mole fractions
    const std::vector<double> &y() { return m_y; }
    /// Get the guess value for the liquid density (mol/m^3)
    double rhoL() { return m_rhoL; }
    /// Set the guess value for the liquid density (mol/m^3)
    void set_rhoL(double rhoL) { this->m_rhoL = rhoL; }
    /// Get the guess value for the vapor density (mol/m^3)
    double rhoV() { return m_rhoV; }
    /// Set the guess value for the vapor density (mol/m^3)
    void set_rhoV(double rhoV){ this->m_rhoV = rhoV; }
};

class PTXYOutput : public NumericOutput {
protected:
    AbstractNumericEvaluator *m_evaluator; // The evaluator connected with this output
public:
    PTXYOutput(const std::shared_ptr<NumericInput> &in, AbstractNumericEvaluator *eval)
        : NumericOutput(in), m_evaluator(eval) { };

    /// Return the error
    double get_error() { return m_y_calc; };

    // Do the calculation
    void evaluate_one() {
        const std::vector<double> &c = m_evaluator->get_const_coefficients();
        if (Jacobian_row.size() != c.size()) {
            resize(c.size());
        }
        m_y_calc = evaluate(c, false); 
        for (std::size_t i = 0; i < c.size(); ++i) {
            // Numerical derivatives :(
            Jacobian_row[i] = der(i, 0.00001);
        }
    }
    double evaluate(const std::vector<double> &c, bool update_densities = false) {
        
        // Cast abstract input to the derived type so we can access its attributes
        PTXYInput *in = static_cast<PTXYInput*>(m_in.get());
        CoolProp::HelmholtzEOSMixtureBackend *HEOS = static_cast<CoolProp::HelmholtzEOSMixtureBackend*>(in->get_AS().get());

        // Set the BIP in main instance and its children
        HEOS->set_binary_interaction_double(0,1,"betaT",c[0]);
        HEOS->set_binary_interaction_double(0,1,"gammaT",c[1]);
        HEOS->set_binary_interaction_double(0,1,"betaV",c[2]);
        HEOS->set_binary_interaction_double(0,1,"gammaV",c[3]);

        HEOS->SatL->set_mole_fractions(in->x());
        double RT = HEOS->SatL->gas_constant()*in->T();
        HEOS->SatL->update_TP_guessrho(in->T(), in->p(), in->rhoL());
        double mu0L = HEOS->SatL->chemical_potential(0)/RT;
        if (update_densities) { in->set_rhoL(HEOS->SatL->rhomolar()); }
        double diff = in->rhoL()/HEOS->SatL->rhomolar()-1;
        
        HEOS->SatV->set_mole_fractions(in->y());
        HEOS->SatV->update_TP_guessrho(in->T(), in->p(), in->rhoV());
        double mu0V = HEOS->SatV->chemical_potential(0)/RT;
        if (update_densities) { in->set_rhoV(HEOS->SatV->rhomolar()); }
        double rhodiffV = in->rhoV()/HEOS->SatV->rhomolar()-1;
        
        return mu0V - mu0L;
    }
    /// Numerical derivative of the residual term
    double der(std::size_t i, double dc) {
        const std::vector<double> &c0 = m_evaluator->get_const_coefficients();
        std::vector<double> cp = c0, cm = c0;
        cp[i] += dc; cm[i] -= dc;
        return (evaluate(cp) - evaluate(cm))/(2*dc);
    }
    static std::shared_ptr<NumericOutput> factory(rapidjson::Value &v, const std::string &backend, const std::string &fluids, AbstractNumericEvaluator *eval){
        std::shared_ptr<NumericOutput> out;

        // Extract parameters from JSON data
        double T = cpjson::get_double(v, "T (K)");
        double p = cpjson::get_double(v, "p (Pa)");
        std::vector<double> x = cpjson::get_double_array(v, "x (molar)");
        std::vector<double> y = cpjson::get_double_array(v, "y (molar)");
        double rhoL = cpjson::get_double(v, "rho' (guess,mol/m3)"); // First guess
        double rhoV = cpjson::get_double(v, "rho'' (guess,mol/m3)"); // First guess

        // Sum up the mole fractions of the liquid and vapor phase - if not 1, it's not a PTxy point, perhaps PTx or PTy
        double sumx = std::accumulate(x.begin(), x.end(), 0.0);
        double sumy = std::accumulate(y.begin(), y.end(), 0.0);

        // Only keep points where both x and y are given
        if (sumx > 0.0 && sumy > 0) {
            // Generate the AbstractState instance owned by this data point
            std::shared_ptr<CoolProp::AbstractState> AS(CoolProp::AbstractState::factory(backend, fluids));
            // Generate the input which stores the PTxy data that is to be fit
            std::shared_ptr<NumericInput> in(new PTXYInput(AS, T, p, x, y, rhoL, rhoV));
            // Generate and add the output value
            out.reset(new PTXYOutput(std::move(in), eval));
        }
        return out;
    }
};

/// The class for the evaluation of a single output value for a single input value
class MixtureEvaluator : public AbstractNumericEvaluator {
public:
    void add_terms(const std::string &backend, const std::string &fluids, rapidjson::Value& terms)
    {
        // Iterate over the terms in the input
        for (rapidjson::Value::ValueIterator itr = terms.Begin(); itr != terms.End(); ++itr)
        {
            // Get the type of the data point
            if (!(*itr).HasMember("type")){ throw CoolProp::ValueError("Missing type"); }
            std::string type = cpjson::get_string(*itr, "type");

            if (type == "PTXY"){
                auto out = PTXYOutput::factory(*itr, backend, fluids, static_cast<AbstractNumericEvaluator*>(this));
                m_outputs.push_back(std::move(out));
            }
            else {
                throw CoolProp::ValueError(fmt::format("I don't understand this data type: %s", type));
            }
        }
    };
};

/// Convert a JSON-formatted string to a rapidjson::Document object
rapidjson::Document JSON_string_to_rapidjson(const std::string &JSON_string)
{
    rapidjson::Document doc;
    doc.Parse<0>(JSON_string.c_str());
    if (doc.HasParseError()) {
        throw CoolProp::ValueError("Unable to load JSON string");
    }
    return doc;
}

/// The function that actually does the fitting
double fit(const std::string &JSON_data_string, const std::string &JSON_fit0_string, bool threading = false, short Nthreads = 4)
{
    // TODO: Validate the JSON against schema
    rapidjson::Document datadoc = JSON_string_to_rapidjson(JSON_data_string);
    //rapidjson::Document fit0doc = JSON_string_to_rapidjson(JSON_fit0_string);
    std::vector<std::string> component_names = cpjson::get_string_array(datadoc["about"], std::string("names"));

    auto CAS1 = CoolProp::get_fluid_param_string(component_names[0], "CAS");
    auto CAS2 = CoolProp::get_fluid_param_string(component_names[1], "CAS");
    /*
    CoolProp::apply_simple_mixing_rule(CAS1, CAS2, "linear");
    */

    // Instantiate the evaluator
    std::shared_ptr<AbstractEvaluator> eval(new MixtureEvaluator());
    MixtureEvaluator* mixeval = static_cast<MixtureEvaluator*>(eval.get());
    mixeval->add_terms("HEOS", strjoin(component_names, "&"), datadoc["data"]);

    //std::vector<double> c0 = { 0.996199694, 1.01473019, 0.997607277, 1.00303472};
    //std::vector<double> c0 = {0.984068272,1.268636194,1.007469726,1.071917985}; 
    std::vector<double> c0 = { 1,1,1,1 };
    
    auto startTime = std::chrono::system_clock::now();
    auto cc = LevenbergMarquadt(eval, c0, threading, Nthreads);
    for (int i = 0; i < cc.size(); i += 1) { std::cout << cc[i] << std::endl; }
    return std::chrono::duration<double>(std::chrono::system_clock::now() - startTime).count();
}

int main() {
    std::string JSON_filename = "C:\\Users\\ihb\\Documents\\Code\\THERMfit\\Ethane_n-Propane.json";
    std::string JSON_data_string = get_file_contents(JSON_filename.c_str());

    fmt::printf("%g\n", fit(JSON_data_string, "", false, 1));
    for (auto &Nthreads : {1,2,3,4,5,6,7,8}){
        fmt::printf("%d %g\n", Nthreads, fit(JSON_data_string, "", true, Nthreads));
    }
}