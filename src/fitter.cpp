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
#include "Backends/Helmholtz/ReducingFunctions.h"
#include "Backends/Helmholtz/MixtureDerivatives.h"

// Includes from c++
#include <iostream>
#include <chrono>

// Includes from phifit
#include "phifit/fitter.h"
#include "phifit/departure_function.h"

using namespace NISTfit;

/// This class holds common terms for inputs
class PhiFitInput : public NumericInput{    
protected:
    shared_ptr<CoolProp::AbstractState> AS;
public:
    PhiFitInput(double x, double y): NumericInput(x, y) {};
    /// Return a reference to the AbstractState being modified
    shared_ptr<CoolProp::AbstractState> &get_AS() { return AS; }
};

/// This class holds common terms for outputs
class PhiFitOutput : public NumericOutput {
public:
    PhiFitOutput(const std::shared_ptr<NumericInput> &in) : NumericOutput(in) {};
    virtual void to_JSON(rapidjson::Value &, rapidjson::Document &) = 0;
};

/// The data structure used to hold an input to Levenberg-Marquadt fitter for parallel evaluation
/// Does not have any of its own routines
class PTXYInput : public PhiFitInput
{
protected:
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
        : PhiFitInput(T, p), m_T(T), m_p(p), m_x(x), m_y(y), m_rhoL(rhoL), m_rhoV(rhoV)  { this->AS = AS; };
    
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

class PTXYOutput : public PhiFitOutput {
protected:
    AbstractNumericEvaluator *m_evaluator; // The evaluator connected with this output (DO NOT FREE!)
    std::vector<double> JtempL, ///< A temporary buffer for holding the liquid evaluation of derivatives w.r.t. coefficients
                        JtempV; ///< A temporary buffer for holding the vapor evaluation of derivatives w.r.t. coefficients
public:
    PTXYOutput(const std::shared_ptr<NumericInput> &in, AbstractNumericEvaluator *eval)
        : PhiFitOutput(in), m_evaluator(eval) { };

    /// Return the error
    double get_error() { return m_y_calc; };

    // Do the calculation
    void evaluate_one() {
        const std::vector<double> &c = m_evaluator->get_const_coefficients();
        // Resize the row in the Jacobian matrix if needed
        std::size_t N = c.size();
        if (Jacobian_row.size() != N) {
            resize(N);
        }
        if (JtempL.size() != N){ JtempL.resize(N); }
        if (JtempV.size() != N) { JtempV.resize(N); }

        // Evaluate the residual at given coefficients
        m_y_calc = evaluate(c, false); 

        // Cast abstract input to the derived type so we can access its attributes
        PTXYInput *in = static_cast<PTXYInput*>(m_in.get());
        CoolProp::HelmholtzEOSMixtureBackend *HEOS = static_cast<CoolProp::HelmholtzEOSMixtureBackend*>(in->get_AS().get());

        std::size_t i = 0;
        evaluate_mu0_over_RT_derivatives(HEOS->SatL.get(), in->x(), i, JtempL);
        evaluate_mur_over_RT_derivatives(HEOS->SatL.get(), in->x(), i, JtempL);
        evaluate_mu0_over_RT_derivatives(HEOS->SatV.get(), in->y(), i, JtempV);
        evaluate_mur_over_RT_derivatives(HEOS->SatV.get(), in->y(), i, JtempV);
        
        for (std::size_t i = 0; i < c.size(); ++i) {
            // Numerical derivatives for checking purposes
            // ------------
            //Jacobian_row[i] = der_num(i, 0.00001);

            Jacobian_row[i] = JtempV[i] - JtempL[i];
        }

    }
    double evaluate(const std::vector<double> &c, bool update_densities = false) {
        
        // Cast abstract input to the derived type so we can access its attributes
        PTXYInput *in = static_cast<PTXYInput*>(m_in.get());
        CoolProp::HelmholtzEOSMixtureBackend *HEOS = static_cast<CoolProp::HelmholtzEOSMixtureBackend*>(in->get_AS().get());
        
        CoolProp::GERG2008ReducingFunction *GERGL = static_cast<CoolProp::GERG2008ReducingFunction*>(HEOS->SatL->Reducing.get());
        GERGL->set_binary_interaction_double(0,1,c[0],c[1],c[2],c[3]);
        CoolProp::GERG2008ReducingFunction *GERGV = static_cast<CoolProp::GERG2008ReducingFunction*>(HEOS->SatV->Reducing.get());
        GERGV->set_binary_interaction_double(0,1,c[0],c[1],c[2],c[3]);

        // Calculate the chemical potentials for liquid and vapor
        std::size_t i = 0;
        double muL = mu_over_RT(HEOS->SatL.get(), in->x(), i);
        double muV = mu_over_RT(HEOS->SatV.get(), in->y(), i);
        
        // Update the densities if requested
        if (update_densities) { 
            in->set_rhoV(HEOS->SatV->rhomolar());
            in->set_rhoL(HEOS->SatL->rhomolar());
        }
        return muV - muL;
    }
    double mu_over_RT(CoolProp::HelmholtzEOSMixtureBackend *HEOS, const std::vector<double> &z, std::size_t i) {
        // Cast abstract input to the derived type so we can access its attributes
        PTXYInput *in = static_cast<PTXYInput*>(m_in.get());

        HEOS->set_mole_fractions(z);
        if (in->rhoV() < 0) {
            HEOS->update(CoolProp::PT_INPUTS, in->p(), in->T());
        }
        else {
            HEOS->update_TP_guessrho(in->T(), in->p(), in->rhoV());
        }

        return HEOS->chemical_potential(i)/(HEOS->gas_constant()*HEOS->T());
    }
    void evaluate_mu0_over_RT_derivatives(CoolProp::HelmholtzEOSMixtureBackend *HEOS, const std::vector<double> &z, std::size_t i, std::vector<double> & buffer) {
        // Cast abstract input to the derived type so we can access its attributes
        PTXYInput *in = static_cast<PTXYInput*>(m_in.get());
        CoolProp::GERG2008ReducingFunction *GERG = static_cast<CoolProp::GERG2008ReducingFunction*>(HEOS->Reducing.get());

        // Zero out the buffer
        buffer[0] = 0; buffer[1] = 0; buffer[2] = 0; buffer[3] = 0;

        double dtau_dbetaT__constTP = GERG->dTr_dbetaT(z) / HEOS->T();
        double dtau_dgammaT__constTP = GERG->dTr_dgammaT(z) / HEOS->T();

        double ddelta_dbetaT__constTP = -POW2(HEOS->delta())*HEOS->d2alphar_dDelta_dTau()*dtau_dbetaT__constTP / (1 + 2 * HEOS->delta()*HEOS->dalphar_dDelta() + POW2(HEOS->delta())*HEOS->d2alphar_dDelta2());
        double ddelta_dgammaT__constTP = -POW2(HEOS->delta())*HEOS->d2alphar_dDelta_dTau()*dtau_dgammaT__constTP / (1 + 2 * HEOS->delta()*HEOS->dalphar_dDelta() + POW2(HEOS->delta())*HEOS->d2alphar_dDelta2());
        double ddelta_dbetaV__constTP = -HEOS->delta()*(1 + HEOS->delta()*HEOS->dalphar_dDelta())*GERG->drhormolar_dbetaV(z) / (HEOS->rhomolar_reducing()*(1 + 2 * HEOS->delta()*HEOS->dalphar_dDelta() + POW2(HEOS->delta())*HEOS->d2alphar_dDelta2()));
        double ddelta_dgammaV__constTP = -HEOS->delta()*(1 + HEOS->delta()*HEOS->dalphar_dDelta())*GERG->drhormolar_dgammaV(z) / (HEOS->rhomolar_reducing()*(1 + 2 * HEOS->delta()*HEOS->dalphar_dDelta() + POW2(HEOS->delta())*HEOS->d2alphar_dDelta2()));

        double rhor = HEOS->rhomolar_reducing();
        double rhoci = HEOS->get_fluid_constant(i, CoolProp::irhomolar_critical);
        double delta_oi = HEOS->delta()*rhor/rhoci;
        double ddelta_oi_ddelta = rhor/rhoci;
        double ddeltaoi_dbetaV__consttaudelta = HEOS->delta()/rhoci*GERG->drhormolar_dbetaV(z);
        double ddeltaoi_dgammaV__consttaudelta = HEOS->delta()/rhoci*GERG->drhormolar_dgammaV(z);
        
        double Tr = HEOS->T_reducing();
        double Tci = HEOS->get_fluid_constant(i, CoolProp::iT_critical);
        double tau_oi = HEOS->tau()*Tci/Tr;
        double dtau_oi_dtau = Tci/Tr; 
        double dtauoi_dbetaT__consttaudelta = -HEOS->tau()*Tci/POW2(Tr)*GERG->dTr_dbetaT(z);
        double dtauoi_dgammaT__consttaudelta = -HEOS->tau()*Tci/POW2(Tr)*GERG->dTr_dgammaT(z);
        
        double dalpha0oi_ddeltaoi = HEOS->get_components()[i].EOS().alpha0.dDelta(tau_oi, delta_oi);
        double dY0_ddelta__consttau = dalpha0oi_ddeltaoi*ddelta_oi_ddelta;
        double dY0_dbetaV_constdeltatau = dalpha0oi_ddeltaoi*ddeltaoi_dbetaV__consttaudelta;
        double dY0_dgammaV_constdeltatau = dalpha0oi_ddeltaoi*ddeltaoi_dgammaV__consttaudelta;

        double dalpha0oi_dtauoi = HEOS->get_components()[i].EOS().alpha0.dTau(tau_oi, delta_oi);
        double dY0_dtau__constdelta = dalpha0oi_dtauoi*dtau_oi_dtau;
        double dY0_dbetaT_constdeltatau = dalpha0oi_dtauoi*dtauoi_dbetaT__consttaudelta;
        double dY0_dgammaT_constdeltatau = dalpha0oi_dtauoi*dtauoi_dgammaT__consttaudelta;

        // buffer is derivatives of mu_i/RT w.r.t. betaT, gammaT, betaV, gammaV in order, starting with the zero index
        // First we calculate just the ideal-gas part
        buffer[0] = dY0_ddelta__consttau*ddelta_dbetaT__constTP  + dY0_dtau__constdelta*dtau_dbetaT__constTP  + dY0_dbetaT_constdeltatau;
        buffer[1] = dY0_ddelta__consttau*ddelta_dgammaT__constTP + dY0_dtau__constdelta*dtau_dgammaT__constTP + dY0_dgammaT_constdeltatau;
        buffer[2] = dY0_ddelta__consttau*ddelta_dbetaV__constTP                                               + dY0_dbetaV_constdeltatau;
        buffer[3] = dY0_ddelta__consttau*ddelta_dgammaV__constTP                                              + dY0_dgammaV_constdeltatau;
    }
    void evaluate_mur_over_RT_derivatives(CoolProp::HelmholtzEOSMixtureBackend *HEOS, const std::vector<double> &z, std::size_t i, std::vector<double> & buffer){

        // Buffer already partially filled from ideal-gas contribution

        // Cast abstract input to the derived type so we can access its attributes
        PTXYInput *in = static_cast<PTXYInput*>(m_in.get());
        CoolProp::GERG2008ReducingFunction *GERG = static_cast<CoolProp::GERG2008ReducingFunction*>(HEOS->Reducing.get());

        double rhor = HEOS->rhomolar_reducing();
        double Tr = HEOS->T_reducing();

        double dY_ddelta__consttau = HEOS->dalphar_dDelta() + CoolProp::MixtureDerivatives::d_ndalphardni_dDelta(*HEOS, i, CoolProp::XN_INDEPENDENT);
        double dY_dtau__constdelta = HEOS->dalphar_dTau() + CoolProp::MixtureDerivatives::d_ndalphardni_dTau(*HEOS, i, CoolProp::XN_INDEPENDENT);

        double dtau_dbetaT__constTP = GERG->dTr_dbetaT(z) / HEOS->T();
        double dtau_dgammaT__constTP = GERG->dTr_dgammaT(z) / HEOS->T();

        double ddelta_dbetaT__constTP = -POW2(HEOS->delta())*HEOS->d2alphar_dDelta_dTau()*dtau_dbetaT__constTP/(1 + 2 * HEOS->delta()*HEOS->dalphar_dDelta() + POW2(HEOS->delta())*HEOS->d2alphar_dDelta2());
        double ddelta_dgammaT__constTP = -POW2(HEOS->delta())*HEOS->d2alphar_dDelta_dTau()*dtau_dgammaT__constTP/(1 + 2 * HEOS->delta()*HEOS->dalphar_dDelta() + POW2(HEOS->delta())*HEOS->d2alphar_dDelta2());
        double ddelta_dbetaV__constTP = -HEOS->delta()*(1 + HEOS->delta()*HEOS->dalphar_dDelta())*GERG->drhormolar_dbetaV(z) / (HEOS->rhomolar_reducing()*(1 + 2 * HEOS->delta()*HEOS->dalphar_dDelta() + POW2(HEOS->delta())*HEOS->d2alphar_dDelta2()));
        double ddelta_dgammaV__constTP = -HEOS->delta()*(1 + HEOS->delta()*HEOS->dalphar_dDelta())*GERG->drhormolar_dgammaV(z) / (HEOS->rhomolar_reducing()*(1 + 2 * HEOS->delta()*HEOS->dalphar_dDelta() + POW2(HEOS->delta())*HEOS->d2alphar_dDelta2()));

        double d_ndrhordni_dbetaV = GERG->d2rhormolar_dxidbetaV(z, i, CoolProp::XN_INDEPENDENT);
        for (std::size_t k = 0; k < z.size(); ++k) {
            d_ndrhordni_dbetaV -= z[k] * GERG->d2rhormolar_dxidbetaV(z, k, CoolProp::XN_INDEPENDENT);
        }
        double d_ndrhordni_dgammaV = GERG->d2rhormolar_dxidgammaV(z, i, CoolProp::XN_INDEPENDENT);
        for (std::size_t k = 0; k < z.size(); ++k) {
            d_ndrhordni_dgammaV -= z[k] * GERG->d2rhormolar_dxidgammaV(z, k, CoolProp::XN_INDEPENDENT);
        }
        double d_ndTrdni_dbetaT = GERG->d2Tr_dxidbetaT(z, i, CoolProp::XN_INDEPENDENT);
        for (std::size_t k = 0; k < z.size(); ++k) {
            d_ndTrdni_dbetaT -= z[k] * GERG->d2Tr_dxidbetaT(z, k, CoolProp::XN_INDEPENDENT);
        }
        double d_ndTrdni_dgammaT = GERG->d2Tr_dxidgammaT(z, i, CoolProp::XN_INDEPENDENT);
        for (std::size_t k = 0; k < z.size(); ++k) {
            d_ndTrdni_dgammaT -= z[k] * GERG->d2Tr_dxidgammaT(z, k, CoolProp::XN_INDEPENDENT);
        }

        double dY_dbetaV_constdeltatau = -HEOS->delta()*HEOS->dalphar_dDelta()/rhor*(d_ndrhordni_dbetaV - GERG->ndrhorbardni__constnj(z, i, CoolProp::XN_INDEPENDENT) / rhor*GERG->drhormolar_dbetaV(z));
        double dY_dgammaV_constdeltatau = -HEOS->delta()*HEOS->dalphar_dDelta()/rhor*(d_ndrhordni_dgammaV - GERG->ndrhorbardni__constnj(z, i, CoolProp::XN_INDEPENDENT) / rhor*GERG->drhormolar_dgammaV(z));
        double dY_dbetaT_constdeltatau = HEOS->tau()*HEOS->dalphar_dTau()/Tr*(d_ndTrdni_dbetaT - GERG->ndTrdni__constnj(z, i, CoolProp::XN_INDEPENDENT)/Tr*GERG->dTr_dbetaT(z));
        double dY_dgammaT_constdeltatau = HEOS->tau()*HEOS->dalphar_dTau()/Tr*(d_ndTrdni_dgammaT - GERG->ndTrdni__constnj(z, i, CoolProp::XN_INDEPENDENT)/Tr*GERG->dTr_dgammaT(z));

        // Add contributions from the residual part
        buffer[0] += dY_ddelta__consttau*ddelta_dbetaT__constTP  + dY_dtau__constdelta*dtau_dbetaT__constTP  + dY_dbetaT_constdeltatau;
        buffer[1] += dY_ddelta__consttau*ddelta_dgammaT__constTP + dY_dtau__constdelta*dtau_dgammaT__constTP + dY_dgammaT_constdeltatau;
        buffer[2] += dY_ddelta__consttau*ddelta_dbetaV__constTP                                              + dY_dbetaV_constdeltatau;
        buffer[3] += dY_ddelta__consttau*ddelta_dgammaV__constTP                                             + dY_dgammaV_constdeltatau;
    }

    /// Numerical derivative of the residual term
    double der_num(std::size_t i, double dc) {
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
    /// Dump this data structure to JSON
    void to_JSON(rapidjson::Value &list, rapidjson::Document &doc) {
        
        PTXYInput *in = static_cast<PTXYInput*>(m_in.get());

        // Populate the JSON structure
        rapidjson::Value val;
        val.SetObject(); 
        val.AddMember("type", "PTXY", doc.GetAllocator());
        val.AddMember("T (K)", in->T(), doc.GetAllocator());
        val.AddMember("p (Pa)", in->p(), doc.GetAllocator());
        val.AddMember("residue ", m_y_calc, doc.GetAllocator());
        cpjson::set_double_array("x", in->x(), val, doc);
        cpjson::set_double_array("y", in->y(), val, doc);

        // Add it to the list
        list.PushBack(val, doc.GetAllocator());
    }
};

/// The data structure used to hold an input to Levenberg-Marquadt fitter for parallel evaluation
/// Does not have any of its own routines
class PRhoTInput : public PhiFitInput
{
protected:
    double m_p, //< Pressure (Pa)
           m_rhomolar, //< Molar density (mol/m^3)
           m_T; //< Temperature (K)
    std::vector<double> m_z; //< Molar composition of mixture
public:
    /*
    @param AS AbstractState to be used for fitting
    @param p Pressure in Pa
    @param rho Molar density in mol/m^3
    @param T Temperature in K
    @param z Molar composition vector
    */
    PRhoTInput(shared_ptr<CoolProp::AbstractState> &AS, double p, double rhomolar, double T, const std::vector<double>&z)
        : PhiFitInput(T, p), m_p(p), m_rhomolar(rhomolar), m_T(T), m_z(z) {this->AS = AS;};
    /// Get the temperature (K)
    double T() { return m_T; }
    /// Get the molar density (mol/m^3)
    double rhomolar() { return m_rhomolar; }
    /// Get the pressure (Pa)
    double p() { return m_p; }
    /// Get the mole fractions
    const std::vector<double> &z() { return m_z; }
};

class PRhoTOutput : public PhiFitOutput {
protected:
    AbstractNumericEvaluator *m_evaluator; // The evaluator connected with this output (DO NOT FREE!)
public:
    PRhoTOutput(const std::shared_ptr<NumericInput> &in, AbstractNumericEvaluator *eval)
        : PhiFitOutput(in), m_evaluator(eval) { };

    /// Return the error
    double get_error() { return m_y_calc; };

    // Do the calculation
    void evaluate_one() {
        try{
            const std::vector<double> &c = m_evaluator->get_const_coefficients();
            // Resize the row in the Jacobian matrix if needed
            if (Jacobian_row.size() != c.size()) {
                resize(c.size());
            }
        
            // Evaluate the residual at given coefficients
            m_y_calc = evaluate(c, false);
            // Evaluate the analytic derivatives of the residuals with respect to the coefficients
            analyt_derivs(Jacobian_row);
        }
        catch (...) {
            // Cast abstract input to the derived type so we can access its attributes
            PTXYInput *in = static_cast<PTXYInput*>(m_in.get());
            CoolProp::HelmholtzEOSMixtureBackend *HEOS = static_cast<CoolProp::HelmholtzEOSMixtureBackend*>(in->get_AS().get());
            std::size_t i = 0, j = 1;
            PhiFitDepartureFunction* dep = static_cast<PhiFitDepartureFunction*>(HEOS->residual_helmholtz->Excess.DepartureFunctionMatrix[i][j].get());
            m_y_calc = evaluate(m_evaluator->get_const_coefficients(), false);
            throw;
        }

        //// Numerical derivatives for testing purposes if necessary
        //for (std::size_t i = 0; i < c.size(); ++i) {
        //    // Numerical derivatives :(
        //    Jacobian_row[i] = der(i, 0.00001);
        //}
    }
    double evaluate(const std::vector<double> &c, bool update_densities = false) {

        // Cast abstract input to the derived type so we can access its attributes
        PRhoTInput *in = static_cast<PRhoTInput*>(m_in.get());
        CoolProp::HelmholtzEOSMixtureBackend *HEOS = static_cast<CoolProp::HelmholtzEOSMixtureBackend*>(in->get_AS().get());

        // Set the BIP in main instance
        CoolProp::GERG2008ReducingFunction *GERG = static_cast<CoolProp::GERG2008ReducingFunction*>(HEOS->Reducing.get());
        GERG->set_binary_interaction_double(0,1,c[0],c[1],c[2],c[3]);

        // Set the mole fractions
        HEOS->set_mole_fractions(in->z());
        // Calculate p = f(T,rho)
        HEOS->update_DmolarT_direct(in->rhomolar(), in->T());
        // The derivative dpdrho__T (needs to be positive always for homogenous states!)
        double dpdrho__T = HEOS->first_partial_deriv(CoolProp::iP, CoolProp::iDmolar, CoolProp::iT);
        // This penalty function is added to avoid negative derivatives
        double penalty = (dpdrho__T > 0) ? 0 : std::abs(dpdrho__T);
        // Pressures should be positive, penalize negative pressures
        penalty += (HEOS->p() > 0) ? 0 : -HEOS->p();
        // Return residual as (p_calc - p_exp)/rho_exp*drhodP_exp|T 
        return (HEOS->p() - in->p())/in->rhomolar()/dpdrho__T;// + penalty;
    }
    void analyt_derivs(std::vector<double> &J) {
        
        // Cast abstract input to the derived type so we can access its attributes
        PRhoTInput *in = static_cast<PRhoTInput*>(m_in.get());
        CoolProp::HelmholtzEOSMixtureBackend *HEOS = static_cast<CoolProp::HelmholtzEOSMixtureBackend*>(in->get_AS().get());
        CoolProp::GERG2008ReducingFunction *GERG = static_cast<CoolProp::GERG2008ReducingFunction*>(HEOS->Reducing.get());

        // ---
        // Evaluate already called, constants set
        // ---

        double DELTAp = HEOS->p() - in->p();
        double rho_exp = in->rhomolar();
        double drho_dp__constT_c = HEOS->first_partial_deriv(CoolProp::iDmolar, CoolProp::iP, CoolProp::iT);

        // Some intermediate terms that show up in a couple of places
        double delta = HEOS->delta();
        double RT = HEOS->gas_constant()*HEOS->T();
        double dtau_dbetaT = 1/HEOS->T()*GERG->dTr_dbetaT(in->z());
        double dtau_dgammaT = 1/HEOS->T()*GERG->dTr_dgammaT(in->z());
        double rhor = GERG->rhormolar(in->z());
        double ddelta_dbetaV = -delta*GERG->drhormolar_dbetaV(in->z())/rhor;
        double ddelta_dgammaV = -delta*GERG->drhormolar_dgammaV(in->z())/rhor;

        // First derivatives of pressure with respect to each of the coefficients at constant T,rho
        double dp_dbetaT = HEOS->rhomolar()*RT*delta*HEOS->d2alphar_dDelta_dTau()*dtau_dbetaT;
        double dp_dgammaT = HEOS->rhomolar()*RT*delta*HEOS->d2alphar_dDelta_dTau()*dtau_dgammaT;
        double dp_dbetaV = HEOS->rhomolar()*RT*(HEOS->dalphar_dDelta() + delta*HEOS->d2alphar_dDelta2())*ddelta_dbetaV;
        double dp_dgammaV = HEOS->rhomolar()*RT*(HEOS->dalphar_dDelta() + delta*HEOS->d2alphar_dDelta2())*ddelta_dgammaV;

        // First derivatives of d(rho)/dp|T with respect to each of the coefficients
        // ----
        // common term for temperature coefficients
        double bracket_T = -POW2(drho_dp__constT_c)*RT*(2*delta*HEOS->d2alphar_dDelta_dTau() + POW2(delta)*HEOS->d3alphar_dDelta2_dTau());
        double d_drhodp_dbetaT = bracket_T*dtau_dbetaT;
        double d_drhodp_dgammaT = bracket_T*dtau_dgammaT;
        // common term for density coefficients
        double bracket_rho = -POW2(drho_dp__constT_c)*RT*(2*HEOS->dalphar_dDelta() + 4*delta*HEOS->d2alphar_dDelta2() + POW2(delta)*HEOS->d3alphar_dDelta3());
        double d_drhodp_dbetaV = bracket_rho*ddelta_dbetaV;
        double d_drhodp_dgammaV = bracket_rho*ddelta_dgammaV;

        J[0] = 1/rho_exp*(DELTAp*d_drhodp_dbetaT + dp_dbetaT*drho_dp__constT_c);
        J[1] = 1/rho_exp*(DELTAp*d_drhodp_dgammaT + dp_dgammaT*drho_dp__constT_c);
        J[2] = 1/rho_exp*(DELTAp*d_drhodp_dbetaV + dp_dbetaV*drho_dp__constT_c);
        J[3] = 1/rho_exp*(DELTAp*d_drhodp_dgammaV + dp_dgammaV*drho_dp__constT_c);
        
    }

    /// Numerical derivative of the residual term
    double der(std::size_t i, double dc) {
        const std::vector<double> &c0 = m_evaluator->get_const_coefficients();
        std::vector<double> cp = c0, cm = c0;
        cp[i] += dc; cm[i] -= dc;
        return (evaluate(cp) - evaluate(cm)) / (2 * dc);
    }
    static std::shared_ptr<NumericOutput> factory(rapidjson::Value &v, const std::string &backend, const std::string &fluids, AbstractNumericEvaluator *eval) {
        std::shared_ptr<NumericOutput> out;

        // Extract parameters from JSON data
        double T = cpjson::get_double(v, "T (K)");
        double p = cpjson::get_double(v, "p (Pa)");
        double rhomolar = cpjson::get_double(v, "rho (mol/m3)");
        std::vector<double> z = cpjson::get_double_array(v, "z (molar)");
        
        // Generate the AbstractState instance owned by this data point
        std::shared_ptr<CoolProp::AbstractState> AS(CoolProp::AbstractState::factory(backend, fluids));
        // Generate the input which stores the PTxy data that is to be fit
        std::shared_ptr<NumericInput> in(new PRhoTInput(AS, p, rhomolar, T, z));
        // Generate and add the output value
        out.reset(new PRhoTOutput(std::move(in), eval));
        return out;
    }
    /// Dump this data structure to JSON
    void to_JSON(rapidjson::Value &list, rapidjson::Document &doc) {

        PRhoTInput *in = static_cast<PRhoTInput*>(m_in.get());
        CoolProp::HelmholtzEOSMixtureBackend *HEOS = static_cast<CoolProp::HelmholtzEOSMixtureBackend*>(in->get_AS().get());

        // Populate the JSON structure
        rapidjson::Value val;
        val.SetObject();
        val.AddMember("type", "PRhoT", doc.GetAllocator());
        val.AddMember("T (K)", in->T(), doc.GetAllocator());
        val.AddMember("p[exp] (Pa)", in->p(), doc.GetAllocator());
        val.AddMember("p[calc] (Pa)", HEOS->p(), doc.GetAllocator());
        val.AddMember("dp/drho|T (Pa/(mol/m3))", HEOS->first_partial_deriv(CoolProp::iP, CoolProp::iDmolar, CoolProp::iT), doc.GetAllocator());
        val.AddMember("rhomolar (mol/m3)", in->rhomolar(), doc.GetAllocator());
        val.AddMember("residue", m_y_calc, doc.GetAllocator());
        cpjson::set_double_array("z", in->z(), val, doc);

        // Add it to the list
        list.PushBack(val, doc.GetAllocator());
    }
};

/// The evaluator class that is used to evaluate the output values from the input values
class MixtureEvaluator : public AbstractNumericEvaluator {
public:
    void add_terms(const std::string &backend, const std::string &fluids, rapidjson::Value& terms)
    {
        // Iterate over the terms in the input
        for (rapidjson::Value::ValueIterator itr = terms.Begin(); itr != terms.End(); ++itr)
        {
            // Get the type of the data point (make sure it has one)
            if (!(*itr).HasMember("type")){ throw CoolProp::ValueError("Missing type"); }
            std::string type = cpjson::get_string(*itr, "type");

            if (type == "PTXY"){
                auto out = PTXYOutput::factory(*itr, backend, fluids, static_cast<AbstractNumericEvaluator*>(this));
                if (out){
                    m_outputs.push_back(std::move(out));
                }
            }
            else if (type == "PRhoT") {
                auto out = PRhoTOutput::factory(*itr, backend, fluids, static_cast<AbstractNumericEvaluator*>(this));
                if (out) {
                    m_outputs.push_back(std::move(out));
                }
            }
            else {
                throw CoolProp::ValueError(fmt::format("I don't understand this data type: %s", type));
            }
        }
    };
    std::string dump_outputs_to_JSON() {
        // Construct the output document
        rapidjson::Document doc;
        doc.SetObject();

        // Get the list of outputs, store as "data"
        rapidjson::Value list(rapidjson::kArrayType);
        for (auto &o : m_outputs) {
            static_cast<PhiFitOutput*>(o.get())->to_JSON(list, doc);
        }
        doc.AddMember("data", list, doc.GetAllocator());

        // Get the departure function
        NumericOutput *_out = static_cast<NumericOutput *>(m_outputs[0].get());
        PhiFitInput * in = static_cast<PhiFitInput *>(_out->get_input().get());
        CoolProp::HelmholtzEOSMixtureBackend *HEOS = static_cast<CoolProp::HelmholtzEOSMixtureBackend*>(in->get_AS().get());
        PhiFitDepartureFunction* pdep = static_cast<PhiFitDepartureFunction*>(HEOS->residual_helmholtz->Excess.DepartureFunctionMatrix[0][1].get());
        rapidjson::Value dep = pdep->to_JSON(doc);
        doc.AddMember("departure[i][j]", dep, doc.GetAllocator());

        return cpjson::json2string(doc);
    }
    void update_departure_function(rapidjson::Value& fit0data) {
        for (auto &out : m_outputs) {
            NumericOutput *_out = static_cast<NumericOutput *>(out.get());
            PhiFitInput * in = static_cast<PhiFitInput *>(_out->get_input().get());
            CoolProp::HelmholtzEOSMixtureBackend *HEOS = static_cast<CoolProp::HelmholtzEOSMixtureBackend*>(in->get_AS().get());
            for (std::size_t i = 0; i <= 1; ++i){
                std::size_t j = 1 - i;
                HEOS->residual_helmholtz->Excess.DepartureFunctionMatrix[i][j].reset(new PhiFitDepartureFunction(fit0data["departure[ij]"]));
                HEOS->SatL->residual_helmholtz->Excess.DepartureFunctionMatrix[i][j].reset(new PhiFitDepartureFunction(fit0data["departure[ij]"]));
                HEOS->SatV->residual_helmholtz->Excess.DepartureFunctionMatrix[i][j].reset(new PhiFitDepartureFunction(fit0data["departure[ij]"]));
                HEOS->set_binary_interaction_double(i, j, "Fij", 1.0); // Turn on departure term
            }
        }
    }
    void set_n(const std::vector<double> &n) {
        for (auto &out : m_outputs) {
            NumericOutput *_out = static_cast<NumericOutput *>(out.get());
            PhiFitInput * in = static_cast<PhiFitInput *>(_out->get_input().get());
            CoolProp::HelmholtzEOSMixtureBackend *HEOS = static_cast<CoolProp::HelmholtzEOSMixtureBackend*>(in->get_AS().get());
            std::size_t i = 0, j = 1;
            PhiFitDepartureFunction* p;
            p = static_cast<PhiFitDepartureFunction*>(HEOS->residual_helmholtz->Excess.DepartureFunctionMatrix[i][j].get()); p->set_n(n);
            p = static_cast<PhiFitDepartureFunction*>(HEOS->SatL->residual_helmholtz->Excess.DepartureFunctionMatrix[i][j].get()); p->set_n(n);
            p = static_cast<PhiFitDepartureFunction*>(HEOS->SatV->residual_helmholtz->Excess.DepartureFunctionMatrix[i][j].get()); p->set_n(n);
            i = 1, j = 0;
            p = static_cast<PhiFitDepartureFunction*>(HEOS->residual_helmholtz->Excess.DepartureFunctionMatrix[i][j].get()); p->set_n(n);
            p = static_cast<PhiFitDepartureFunction*>(HEOS->SatL->residual_helmholtz->Excess.DepartureFunctionMatrix[i][j].get()); p->set_n(n);
            p = static_cast<PhiFitDepartureFunction*>(HEOS->SatV->residual_helmholtz->Excess.DepartureFunctionMatrix[i][j].get()); p->set_n(n);
        }
    }
    void set_nt(const std::vector<double> &n, const std::vector<double> &t) {
        for (auto &out : m_outputs) {
            NumericOutput *_out = static_cast<NumericOutput *>(out.get());
            PhiFitInput * in = static_cast<PhiFitInput *>(_out->get_input().get());
            CoolProp::HelmholtzEOSMixtureBackend *HEOS = static_cast<CoolProp::HelmholtzEOSMixtureBackend*>(in->get_AS().get());
            std::size_t i = 0, j = 1;
            PhiFitDepartureFunction* p;
            p = static_cast<PhiFitDepartureFunction*>(HEOS->residual_helmholtz->Excess.DepartureFunctionMatrix[i][j].get()); p->set_nt(n,t);
            p = static_cast<PhiFitDepartureFunction*>(HEOS->SatL->residual_helmholtz->Excess.DepartureFunctionMatrix[i][j].get()); p->set_nt(n,t);
            p = static_cast<PhiFitDepartureFunction*>(HEOS->SatV->residual_helmholtz->Excess.DepartureFunctionMatrix[i][j].get()); p->set_nt(n,t);
            i = 1, j = 0;
            p = static_cast<PhiFitDepartureFunction*>(HEOS->residual_helmholtz->Excess.DepartureFunctionMatrix[i][j].get()); p->set_nt(n,t);
            p = static_cast<PhiFitDepartureFunction*>(HEOS->SatL->residual_helmholtz->Excess.DepartureFunctionMatrix[i][j].get()); p->set_nt(n,t);
            p = static_cast<PhiFitDepartureFunction*>(HEOS->SatV->residual_helmholtz->Excess.DepartureFunctionMatrix[i][j].get()); p->set_nt(n,t);
        }
    }
    std::string departure_function_to_JSON() {
        for (auto &out : m_outputs) {
            NumericOutput *_out = static_cast<NumericOutput *>(out.get());
            PhiFitInput * in = static_cast<PhiFitInput *>(_out->get_input().get());
            CoolProp::HelmholtzEOSMixtureBackend *HEOS = static_cast<CoolProp::HelmholtzEOSMixtureBackend*>(in->get_AS().get());
            std::size_t i =0, j=1;
            PhiFitDepartureFunction* dep = static_cast<PhiFitDepartureFunction*>(HEOS->residual_helmholtz->Excess.DepartureFunctionMatrix[i][j].get());
            rapidjson::Document doc;
            return cpjson::json2string(dep->to_JSON(doc));
        }
    }
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

CoeffFitClass::CoeffFitClass(const std::string &JSON_data_string){
    // TODO: Validate the JSON against schema
    rapidjson::Document datadoc = JSON_string_to_rapidjson(JSON_data_string);
    std::vector<std::string> component_names = cpjson::get_string_array(datadoc["about"], std::string("names"));

    try {
        std::shared_ptr<CoolProp::AbstractState> AS(CoolProp::AbstractState::factory("HEOS", strjoin(component_names,"&")));
    }
    catch(...){
        auto CAS1 = CoolProp::get_fluid_param_string(component_names[0], "CAS");
        auto CAS2 = CoolProp::get_fluid_param_string(component_names[1], "CAS");
        CoolProp::apply_simple_mixing_rule(CAS1, CAS2, "linear");
    }

    // Instantiate the evaluator
    m_eval.reset(new MixtureEvaluator());
    MixtureEvaluator* mixeval = static_cast<MixtureEvaluator*>(m_eval.get());
    mixeval->add_terms("HEOS", strjoin(component_names, "&"), datadoc["data"]);

}
void CoeffFitClass::setup(const std::string &JSON_fit0_string)
{
    // Make sure string is not empty
    if (JSON_fit0_string.empty()) { throw CoolProp::ValueError("fit0 string is empty"); }

    // TODO: Validate the JSON against schema
    rapidjson::Document fit0doc = JSON_string_to_rapidjson(JSON_fit0_string);
        
    // Inject the desired departure function
    MixtureEvaluator* mixeval = static_cast<MixtureEvaluator*>(m_eval.get()); // Type-cast
    mixeval->update_departure_function(fit0doc);
}
void CoeffFitClass::set_n(const std::vector<double> &n)
{
    // Inject the desired n coefficients
    MixtureEvaluator* mixeval = static_cast<MixtureEvaluator*>(m_eval.get()); // Type-cast
    mixeval->set_n(n);
}
void CoeffFitClass::set_nt(const std::vector<double> &n, const std::vector<double> &t)
{
    // Inject the desired coefficients for n and t
    MixtureEvaluator* mixeval = static_cast<MixtureEvaluator*>(m_eval.get()); // Type-cast
    mixeval->set_nt(n,t);
}
void CoeffFitClass::run(bool threading, short Nthreads, const std::vector<double> &c0){
    auto startTime = std::chrono::system_clock::now();
    LevenbergMarquadtOptions opts;
    opts.c0 = c0; 
    opts.threading = threading; 
    opts.Nthreads = Nthreads; 
    opts.omega = 0.35;
    m_cfinal = LevenbergMarquadt(m_eval, opts);
    //for (int i = 0; i < cc.size(); i += 1) { std::cout << cc[i] << std::endl; }
    m_elap_sec = std::chrono::duration<double>(std::chrono::system_clock::now() - startTime).count();
}
/// Just evaluate the residual vector, and cache values internally
void CoeffFitClass::evaluate_serial(const std::vector<double> &c0) {
    m_eval->set_coefficients(c0);
    m_eval->evaluate_serial(0, m_eval->get_outputs_size(), 0);
}
/// Just evaluate the residual vector, and cache values internally
void CoeffFitClass::evaluate_parallel(const std::vector<double> &c0, short Nthreads) {
    m_eval->set_coefficients(c0);
    m_eval->evaluate_parallel(Nthreads);
}
double CoeffFitClass::sum_of_squares() { return m_eval->get_error_vector().norm(); }
std::vector<double> CoeffFitClass::errorvec(){
    const Eigen::VectorXd &vec = m_eval->get_error_vector(); 
    return std::vector<double>(vec.data(), vec.data() + vec.size());
}
std::string CoeffFitClass::dump_outputs_to_JSON() {
    MixtureEvaluator* mixeval = static_cast<MixtureEvaluator*>(m_eval.get());
    return mixeval->dump_outputs_to_JSON();
}

/// The function that actually does the fitting - a thin wrapper around the CoeffFitClass
double simplefit(const std::string &JSON_data_string, const std::string &JSON_fit0_string, bool threading, short Nthreads, std::vector<double> &c0, std::vector<double> &cfinal)
{
    CoeffFitClass CFC(JSON_data_string);
    CFC.setup(JSON_fit0_string);
    CFC.run(threading, Nthreads, c0);
    cfinal = CFC.cfinal();
    //for (int i = 0; i < cfinal.size(); i += 1) { std::cout << cfinal[i] << std::endl; }
    return CFC.elapsed_sec();
}

#ifdef PYBIND11

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

PYBIND11_PLUGIN(MixtureCoefficientFitter) {
    py::module m("MixtureCoefficientFitter", "MixtureCoefficientFitter module");

    // For later(?)
    //m.def("set_dry_air_defaults", &set_dry_air_defaults, "Set the default air components/composition");

    py::class_<CoeffFitClass>(m, "CoeffFitClass")
        .def(py::init<const std::string &>())
        .def("setup", &CoeffFitClass::setup)
        .def("set_n", &CoeffFitClass::set_n)
        .def("set_nt", &CoeffFitClass::set_nt)
        .def("run", &CoeffFitClass::run)
        .def("evaluate_parallel", &CoeffFitClass::evaluate_parallel)
        .def("evaluate_serial", &CoeffFitClass::evaluate_serial)
        .def("cfinal", &CoeffFitClass::cfinal)
        .def("errorvec", &CoeffFitClass::errorvec)
        .def("dump_outputs_to_JSON", &CoeffFitClass::dump_outputs_to_JSON)
        .def("sum_of_squares", &CoeffFitClass::sum_of_squares)
        .def("elapsed_sec", &CoeffFitClass::elapsed_sec);

    return m.ptr();
}

#endif