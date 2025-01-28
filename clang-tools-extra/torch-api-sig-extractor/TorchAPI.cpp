#include "TorchAPI.h"

using json = nlohmann::json;

TorchAPI::TorchAPI(TorchAPIKind kind_, std::string api_name_,
                   std::unique_ptr<TorchType> return_type_)
    : kind(kind_), api_name(api_name_), return_type(std::move(return_type_)) {}
json TorchAPI::to_json() const {
  json j;
  j["API_kind"] = tak_to_string();
  j["API_name"] = api_name;
  if (return_type != nullptr) {
    j["return_type"] = return_type->to_json();
  } else {
    j["return_type"] = nullptr;
  }
  return j;
}
std::string TorchAPI::tak_to_string() const {
  switch (kind) {
    case TAK_Function:
      return "Function";
    case TAK_TensorMethod:
      return "TensorMethod";
    case TAK_Module:
      return "Module";
    default: {
      assert(false);
      return "";
    }
  }
}

TorchFunction::TorchFunction(std::string func_name,
                             std::vector<TorchParam> params_,
                             std::unique_ptr<TorchType> return_type_)
    : TorchAPI(TAK_Function, func_name, std::move(return_type_)) {
  params = std::move(params_);
}
json TorchFunction::to_json() const {
  json j = TorchAPI::to_json();
  std::vector<json> j_params;
  for (auto &param : params) {
    json j_param;
    j_param["param_name"] = param.first;
    j_param["param_type"] = param.second->to_json();
    j_params.push_back(j_param);
  }
  j["params"] = j_params;
  return j;
}

TorchTensorMethod::TorchTensorMethod(std::string method_name,
                                     std::vector<TorchParam> params_,
                                     std::unique_ptr<TorchType> return_type_)
    : TorchAPI(TAK_TensorMethod, method_name, std::move(return_type_)) {
  params = std::move(params_);
}
json TorchTensorMethod::to_json() const {
  json j = TorchAPI::to_json();
  std::vector<json> j_params;
  for (auto &param : params) {
    json j_param;
    j_param["param_name"] = param.first;
    j_param["param_type"] = param.second->to_json();
    j_params.push_back(j_param);
  }
  j["params"] = j_params;
  return j;
}

TorchModule::TorchModule(std::string module_name,
                         std::vector<TorchParam> ctor_params_,
                         std::vector<TorchParam> forward_params_,
                         std::unique_ptr<TorchType> return_type_)
    : TorchAPI(TAK_Module, module_name, std::move(return_type_)) {
  ctor_params = std::move(ctor_params_);
  forward_params = std::move(forward_params_);
}
json TorchModule::to_json() const {
  json j = TorchAPI::to_json();

  std::vector<json> j_ctor_params;
  for (auto &param : ctor_params) {
    json j_param;
    j_param["param_name"] = param.first;
    j_param["param_type"] = param.second->to_json();
    j_ctor_params.push_back(j_param);
  }
  j["ctor_params"] = j_ctor_params;

  std::vector<json> j_forward_params;
  for (auto &param : forward_params) {
    json j_param;
    j_param["param_name"] = param.first;
    j_param["param_type"] = param.second->to_json();
    j_forward_params.push_back(j_param);
  }
  j["forward_params"] = j_forward_params;

  return j;
}
