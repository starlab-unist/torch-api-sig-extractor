#ifndef TORCH_API_SIGNATURE_EXTRACTOR_TORCH_API
#define TORCH_API_SIGNATURE_EXTRACTOR_TORCH_API

#include "TorchTypes.h"

using namespace llvm;
using namespace clang;
using json = nlohmann::json;

class TorchAPI {
 public:
  enum TorchAPIKind {
    TAK_Function,
    TAK_TensorMethod,
    TAK_Module,
  };

  TorchAPI(TorchAPIKind kind_, std::string api_name_,
           std::unique_ptr<TorchType> return_type_);
  virtual json to_json() const;

 private:
  std::string tak_to_string() const;

 protected:
  const TorchAPIKind kind;
  std::string api_name;
  std::unique_ptr<TorchType> return_type;
};

class TorchFunction : public TorchAPI {
 public:
  TorchFunction(std::string func_name, std::vector<TorchParam> params_,
                std::unique_ptr<TorchType> return_type_);
  virtual json to_json() const override;

 private:
  std::vector<TorchParam> params;
};

class TorchTensorMethod : public TorchAPI {
 public:
  TorchTensorMethod(std::string method_name, std::vector<TorchParam> params_,
                    std::unique_ptr<TorchType> return_type_);
  virtual json to_json() const override;

 private:
  std::vector<TorchParam> params;
};

class TorchModule : public TorchAPI {
 public:
  TorchModule(std::string module_name, std::vector<TorchParam> ctor_params_,
              std::vector<TorchParam> forward_params_,
              std::unique_ptr<TorchType> return_type_);
  virtual json to_json() const override;

 private:
  std::vector<TorchParam> ctor_params;
  std::vector<TorchParam> forward_params;
};

#endif
