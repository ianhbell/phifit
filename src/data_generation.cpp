#include "crossplatform_shared_ptr.h"
#include "AbstractState.h"
#include "rapidjson_include.h"

////   *************************************************************************
////   *************************** THE TESTS  **********************************
////   *************************************************************************

inline void set_string_array(const std::string &key, const std::vector<std::string> &vec, rapidjson::Value &value, rapidjson::Document &doc)
{
    rapidjson::Value _v(rapidjson::kArrayType);
    for (unsigned int i = 0; i < vec.size(); ++i)
    {
        _v.PushBack(rapidjson::Value(vec[i].c_str(), doc.GetAllocator()).Move(),
            doc.GetAllocator()
        );
    }
    value.AddMember(rapidjson::Value(key.c_str(), doc.GetAllocator()).Move(),
        _v,
        doc.GetAllocator());
};
// Generate data for the given binary pair, for purposes of fitting betas and gammas
std::string gen_JSON_data(const std::string &backend, const std::string &names) {

    rapidjson::Document doc;
    doc.SetObject();

    rapidjson::Value about;
    about.SetObject();
    doc.AddMember("about", about, doc.GetAllocator());
    rapidjson::Value &rabout = doc["about"];
    set_string_array("names", strsplit(names, '&'), rabout, doc);

    shared_ptr<CoolProp::AbstractState> AS(CoolProp::AbstractState::factory(backend, names));

    rapidjson::Value v_data(rapidjson::kArrayType);
    for (double T = 200; T < 250; T += 20) {
        for (double x0 = 0.1; x0 < 0.91; x0 += 0.4) {
            std::vector<double> x(2, x0); x[1] = 1 - x[0];
            AS->set_mole_fractions(x);
            AS->update(CoolProp::QT_INPUTS, 0, T);
            auto y = AS->mole_fractions_vapor();

            rapidjson::Value point;
            point.SetObject();
            point.AddMember("type", "PTXY", doc.GetAllocator());
            point.AddMember("p (Pa)", AS->p(), doc.GetAllocator());
            point.AddMember("T (K)", AS->T(), doc.GetAllocator());
            point.AddMember("rho' (guess,mol/m3)", -1, doc.GetAllocator());//AS->saturated_liquid_keyed_output(CoolProp::iDmolar), doc.GetAllocator());
            point.AddMember("rho'' (guess,mol/m3)", -1, doc.GetAllocator());// AS->saturated_vapor_keyed_output(CoolProp::iDmolar), doc.GetAllocator());
            cpjson::set_double_array("x (molar)", x, point, doc);
            cpjson::set_double_array("y (molar)", y, point, doc);

            v_data.PushBack(point, doc.GetAllocator());
        }
    }
    doc.AddMember("data", v_data, doc.GetAllocator());

    return cpjson::to_string(doc);
}