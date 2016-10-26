#include "phifit/departure_function.h"
#include "rapidjson_include.h"
PhiFitDepartureFunction::PhiFitDepartureFunction(rapidjson::Value &JSON_data) {
    n = cpjson::get_double_array(JSON_data, "n");
    t = cpjson::get_double_array(JSON_data, "t");
    d = cpjson::get_double_array(JSON_data, "d");
    cdelta = cpjson::get_double_array2D(JSON_data["cdelta"]);
    ldelta = cpjson::get_double_array2D(JSON_data["ldelta"]);
    ctau = cpjson::get_double_array2D(JSON_data["ctau"]);
    ltau = cpjson::get_double_array2D(JSON_data["ltau"]);

    // Check the values for the coefficients in the exponential
    double max_cdelta = -1e20, max_ctau = -1e20;
    // Check the coefficients for delta
    for (std::size_t i = 0; i < cdelta.size(); ++i) {
        for (std::size_t j = 0; j < cdelta[i].size(); ++j) {
            if (ldelta[i][j] > 0){ // Only check non-constant terms
                max_cdelta = std::max(cdelta[i][j], max_cdelta);
            }
        }
    }
    // Check the coefficients for tau
    for (std::size_t i = 0; i < ctau.size(); ++i) {
        for (std::size_t j = 0; j < ctau[i].size(); ++j) {
            if (ltau[i][j] != 0) { // Only check non-constant terms
                max_ctau = std::max(ctau[i][j], max_ctau);
            }
        }
    }
    //if (max_cdelta > 0.0) { throw CoolProp::ValueError("All coefficients of cdelta with non-zero power MUST be non-positive"); }
    //if (max_ctau > 0.0) { throw CoolProp::ValueError("All coefficients of ctau with non-zero power MUST be non-positive"); }
}
void PhiFitDepartureFunction::update_coeffs(const Coefficients &coeffs){
    this->n = coeffs.n; this->t = coeffs.t; this->d = coeffs.d;
    this->ldelta = coeffs.ldelta; this->cdelta = coeffs.cdelta;
    this->ltau = coeffs.ltau; this->ctau = coeffs.ctau;
}

rapidjson::Value PhiFitDepartureFunction::to_JSON(rapidjson::Document &doc) {
    rapidjson::Value val; val.SetObject();
    cpjson::set_double_array("n", n, val, doc);
    cpjson::set_double_array("t", t, val, doc);
    cpjson::set_double_array("d", d, val, doc);
    cpjson::set_double_array2D("cdelta", cdelta, val, doc);
    cpjson::set_double_array2D("ldelta", ldelta, val, doc);
    cpjson::set_double_array2D("ctau", ctau, val, doc);
    cpjson::set_double_array2D("ltau", ltau, val, doc);
    return val;
}

void PhiFitDepartureFunction::update(double tau, double delta)
{
    derivs.reset(0.0);

    CoolPropDbl log_tau = log(tau), log_delta = log(delta), ndteu,
                one_over_delta = 1 / delta, one_over_tau = 1 / tau; // division is much slower than multiplication, so do one division here

    for (std::size_t i = 0; i < n.size(); ++i)
    {
        CoolPropDbl ni = n[i], di = d[i], ti = t[i];
        const std::vector<double> &cdeltai = cdelta[i], &ldeltai = ldelta[i], &ctaui = ctau[i], &ltaui = ltau[i];

        // Set the u part of exp(u) to zero
        CoolPropDbl u = 0;
        CoolPropDbl du_ddelta = 0;
        CoolPropDbl du_dtau = 0;
        CoolPropDbl d2u_ddelta2 = 0;
        CoolPropDbl d2u_dtau2 = 0;
        CoolPropDbl d3u_ddelta3 = 0;
        CoolPropDbl d3u_dtau3 = 0;
        CoolPropDbl d4u_ddelta4 = 0;
        CoolPropDbl d4u_dtau4 = 0;

        for (std::size_t j = 0; j < ldeltai.size(); ++j)
        {
            const double ldeltaij = ldeltai[j], cdeltaij = cdeltai[j], delta_to_lij = pow(delta, ldeltaij);
            u += cdeltaij*delta_to_lij;
            du_ddelta += cdeltaij*ldeltaij*delta_to_lij;
            d2u_ddelta2 += cdeltaij*ldeltaij*(ldeltaij -1)*delta_to_lij;
            d3u_ddelta3 += cdeltaij*ldeltaij*(ldeltaij -1)*(ldeltaij -2)*delta_to_lij;
            d4u_ddelta4 += cdeltaij*ldeltaij*(ldeltaij -1)*(ldeltaij -2)*(ldeltaij -3)*delta_to_lij;
        }
        for (std::size_t j = 0; j < ltaui.size(); ++j){
            const double ltauij = ltaui[j], ctauij = ctaui[j], tau_to_mij = pow(tau, ltauij); 
            u += ctauij*tau_to_mij;
            du_dtau += ctauij*ltauij*tau_to_mij;
            d2u_dtau2 += ctauij*ltauij*(ltauij -1)*tau_to_mij;
            d3u_dtau3 += ctauij*ltauij*(ltauij -1)*(ltauij -2)*tau_to_mij;
            d4u_dtau4 += ctauij*ltauij*(ltauij -1)*(ltauij -2)*(ltauij -3)*tau_to_mij;
        }
        u *= 1;
        du_ddelta *= one_over_delta;
        d2u_ddelta2 *= POW2(one_over_delta);
        d3u_ddelta3 *= POW3(one_over_delta);
        d4u_ddelta4 *= POW4(one_over_delta);
        du_dtau *= one_over_tau;
        d2u_dtau2 *= POW2(one_over_tau);
        d3u_dtau3 *= POW3(one_over_tau);
        d4u_dtau4 *= POW4(one_over_tau);

        ndteu = ni*exp(ti*log_tau + di*log_delta + u);

        const CoolPropDbl dB_delta_ddelta = delta*d2u_ddelta2 + du_ddelta;
        const CoolPropDbl d2B_delta_ddelta2 = delta*d3u_ddelta3 + 2 * d2u_ddelta2;
        const CoolPropDbl d3B_delta_ddelta3 = delta*d4u_ddelta4 + 3 * d3u_ddelta3;

        const CoolPropDbl B_delta = (delta*du_ddelta + di);
        const CoolPropDbl B_delta2 = delta*dB_delta_ddelta + (B_delta - 1)*B_delta;
        const CoolPropDbl dB_delta2_ddelta = delta*d2B_delta_ddelta2 + 2 * B_delta*dB_delta_ddelta;
        const CoolPropDbl B_delta3 = delta*dB_delta2_ddelta + (B_delta - 2)*B_delta2;
        const CoolPropDbl dB_delta3_ddelta = delta*delta*d3B_delta_ddelta3 + 3 * delta*B_delta*d2B_delta_ddelta2 + 3 * delta*POW2(dB_delta_ddelta) + 3 * B_delta*(B_delta - 1)*dB_delta_ddelta;
        const CoolPropDbl B_delta4 = delta*dB_delta3_ddelta + (B_delta - 3)*B_delta3;

        const CoolPropDbl dB_tau_dtau = tau*d2u_dtau2 + du_dtau;
        const CoolPropDbl d2B_tau_dtau2 = tau*d3u_dtau3 + 2 * d2u_dtau2;
        const CoolPropDbl d3B_tau_dtau3 = tau*d4u_dtau4 + 3 * d3u_dtau3;

        const CoolPropDbl B_tau = (tau*du_dtau + ti);
        const CoolPropDbl B_tau2 = tau*dB_tau_dtau + (B_tau - 1)*B_tau;
        const CoolPropDbl dB_tau2_dtau = tau*d2B_tau_dtau2 + 2 * B_tau*dB_tau_dtau;
        const CoolPropDbl B_tau3 = tau*dB_tau2_dtau + (B_tau - 2)*B_tau2;
        const CoolPropDbl dB_tau3_dtau = tau*tau*d3B_tau_dtau3 + 3 * tau*B_tau*d2B_tau_dtau2 + 3 * tau*POW2(dB_tau_dtau) + 3 * B_tau*(B_tau - 1)*dB_tau_dtau;
        const CoolPropDbl B_tau4 = tau*dB_tau3_dtau + (B_tau - 3)*B_tau3;

        derivs.alphar += ndteu;

        derivs.dalphar_ddelta += ndteu*B_delta;
        derivs.dalphar_dtau += ndteu*B_tau;

        derivs.d2alphar_ddelta2 += ndteu*B_delta2;
        derivs.d2alphar_ddelta_dtau += ndteu*B_delta*B_tau;
        derivs.d2alphar_dtau2 += ndteu*B_tau2;

        derivs.d3alphar_ddelta3 += ndteu*B_delta3;
        derivs.d3alphar_ddelta2_dtau += ndteu*B_delta2*B_tau;
        derivs.d3alphar_ddelta_dtau2 += ndteu*B_delta*B_tau2;
        derivs.d3alphar_dtau3 += ndteu*B_tau3;

        derivs.d4alphar_ddelta4 += ndteu*B_delta4;
        derivs.d4alphar_ddelta3_dtau += ndteu*B_delta3*B_tau;
        derivs.d4alphar_ddelta2_dtau2 += ndteu*B_delta2*B_tau2;
        derivs.d4alphar_ddelta_dtau3 += ndteu*B_delta*B_tau3;
        derivs.d4alphar_dtau4 += ndteu*B_tau4;
    }
    derivs.dalphar_ddelta *= one_over_delta;
    derivs.dalphar_dtau *= one_over_tau;
    derivs.d2alphar_ddelta2 *= POW2(one_over_delta);
    derivs.d2alphar_dtau2 *= POW2(one_over_tau);
    derivs.d2alphar_ddelta_dtau *= one_over_delta*one_over_tau;

    derivs.d3alphar_ddelta3 *= POW3(one_over_delta);
    derivs.d3alphar_dtau3 *= POW3(one_over_tau);
    derivs.d3alphar_ddelta2_dtau *= POW2(one_over_delta)*one_over_tau;
    derivs.d3alphar_ddelta_dtau2 *= one_over_delta*POW2(one_over_tau);

    derivs.d4alphar_ddelta4 *= POW4(one_over_delta);
    derivs.d4alphar_dtau4 *= POW4(one_over_tau);
    derivs.d4alphar_ddelta3_dtau *= POW3(one_over_delta)*one_over_tau;
    derivs.d4alphar_ddelta2_dtau2 *= POW2(one_over_delta)*POW2(one_over_tau);
    derivs.d4alphar_ddelta_dtau3 *= one_over_delta*POW3(one_over_tau);
};