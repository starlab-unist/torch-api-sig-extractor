#ifndef TORCH_API_SIGNATURE_EXTRACTOR_TORCH_TYPES
#define TORCH_API_SIGNATURE_EXTRACTOR_TORCH_TYPES

#include <iostream>
#include <map>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <vector>

#include "Utils.h"
#include "clang/AST/AST.h"
#include "llvm/Support/Casting.h"

using namespace llvm;
using namespace clang;
using json = nlohmann::json;

class TorchType;
typedef std::pair<std::string, std::unique_ptr<TorchType>> TorchParam;

class TorchType {
 public:
  enum TorchTypeKind {
    TTK_Unknown,

    TTK_Void,

    TTK_Enum,

    TTK_Int,
    TTK_SymInt,
    TTK_UnsignedInt,
    TTK_Dtype,

    TTK_Bool,
    TTK_String,
    TTK_Float,
    TTK_Double,
    TTK_MemoryFormat,
    TTK_Layout,
    TTK_Device,
    TTK_Variant,

    TTK_Vector,
    TTK_ArrayRef,
    TTK_OptionalArrayRef,

    TTK_ExpandingArray,
    TTK_ExpandingArrayWithOptionalElem,
    TTK_Tuple,
    TTK_Pair,

    TTK_Tensor,
    TTK_Scalar,
    TTK_Optional,
    TTK_APIOptions,
  };

  TorchType(TorchTypeKind kind_);
  virtual ~TorchType() = default;
  virtual void set_default(Expr *default_expr);
  virtual bool precise() const;
  virtual json to_json() const;

  TorchTypeKind get_kind() const;

 private:
  std::string ttk_to_string() const;
  const TorchTypeKind kind;
};

class TorchUnknownType : public TorchType {
 public:
  TorchUnknownType(std::string type_str_);
  virtual bool precise() const override;
  virtual json to_json() const override;

 private:
  std::string type_str;
};

class TorchVoidType : public TorchType {
 public:
  TorchVoidType();
};

class TorchIntType : public TorchType {
 public:
  TorchIntType(std::string specifier_);
  virtual void set_default(Expr *default_expr) override;
  void set_default(int value);
  virtual json to_json() const override;

 private:
  std::string specifier;
  Optional<int> default_value = None;
};

class TorchSymIntType : public TorchType {
 public:
  TorchSymIntType();
  virtual void set_default(Expr *default_expr) override;
  void set_default(int value);
  virtual json to_json() const override;

 private:
  Optional<int> default_value = None;
};

class TorchUnsignedIntType : public TorchType {
 public:
  TorchUnsignedIntType();
  virtual void set_default(Expr *default_expr) override;
  void set_default(unsigned int value);
  virtual json to_json() const override;

 private:
  Optional<unsigned int> default_value = None;
};

class TorchBoolType : public TorchType {
 public:
  TorchBoolType();
};

class TorchStringType : public TorchType {
 public:
  TorchStringType(bool is_view_);
  virtual json to_json() const override;

 private:
  bool is_view;
};

class TorchFloatType : public TorchType {
 public:
  TorchFloatType();
};

class TorchDoubleType : public TorchType {
 public:
  TorchDoubleType();
};

class TorchMemoryFormatType : public TorchType {
 public:
  TorchMemoryFormatType();
};

class TorchLayoutType : public TorchType {
 public:
  TorchLayoutType();
};

class TorchDeviceType : public TorchType {
 public:
  TorchDeviceType();
};

class TorchDtypeType : public TorchType {
 public:
  TorchDtypeType();
};

class TorchVariantType : public TorchType {
 public:
  TorchVariantType(std::vector<std::unique_ptr<TorchType>> ttypes_);
  virtual void set_default(Expr *default_expr) override;
  virtual bool precise() const override;
  virtual json to_json() const override;

 private:
  std::vector<std::unique_ptr<TorchType>> ttypes;
};

class TorchEnumType : public TorchType {
 public:
  TorchEnumType(std::string enum_name_);
  virtual json to_json() const override;

 private:
  std::string enum_name;
};

class TorchVectorType : public TorchType {
 public:
  TorchVectorType(std::unique_ptr<TorchType> value_type_);
  virtual bool precise() const override;
  virtual json to_json() const override;

 private:
  std::unique_ptr<TorchType> value_type;
};

class TorchArrayRefType : public TorchType {
 public:
  TorchArrayRefType(std::unique_ptr<TorchType> value_type_);
  virtual bool precise() const override;
  virtual json to_json() const override;

 private:
  std::unique_ptr<TorchType> value_type;
};

class TorchOptionalArrayRefType : public TorchType {
 public:
  TorchOptionalArrayRefType(std::unique_ptr<TorchType> value_type_);
  virtual bool precise() const override;
  virtual json to_json() const override;

 private:
  std::unique_ptr<TorchType> value_type;
};

class TorchExpandingArrayType : public TorchType {
 public:
  TorchExpandingArrayType(size_t size_, std::unique_ptr<TorchType> value_type_);
  virtual void set_default(Expr *default_expr) override;
  virtual bool precise() const override;
  virtual json to_json() const override;

 private:
  size_t size;
  std::unique_ptr<TorchType> value_type;
};

class TorchExpandingArrayWithOptionalElemType : public TorchType {
 public:
  TorchExpandingArrayWithOptionalElemType(
      size_t size_, std::unique_ptr<TorchType> value_type_);
  virtual void set_default(Expr *default_expr) override;
  virtual bool precise() const override;
  virtual json to_json() const override;

 private:
  size_t size;
  std::unique_ptr<TorchType> value_type;
};

class TorchTupleType : public TorchType {
 public:
  TorchTupleType(std::vector<std::unique_ptr<TorchType>> ttypes_);
  virtual bool precise() const override;
  virtual json to_json() const override;

 private:
  std::vector<std::unique_ptr<TorchType>> ttypes;
};

class TorchPairType : public TorchType {
 public:
  TorchPairType(std::vector<std::unique_ptr<TorchType>> ttypes_);
  virtual bool precise() const override;
  virtual json to_json() const override;

 private:
  std::vector<std::unique_ptr<TorchType>> ttypes;
};

class TorchTensorType : public TorchType {
 public:
  TorchTensorType();
};

class TorchScalarType : public TorchType {
 public:
  TorchScalarType();
};

class TorchOptionalType : public TorchType {
 public:
  TorchOptionalType(std::unique_ptr<TorchType> value_type_);
  virtual void set_default(Expr *default_expr) override;
  virtual bool precise() const override;
  virtual json to_json() const override;

 private:
  std::unique_ptr<TorchType> value_type;
};

class TorchAPIOptionsType : public TorchType {
 public:
  TorchAPIOptionsType(std::string class_name_,
                      std::vector<TorchParam> ctor_params_,
                      std::vector<TorchParam> member_params_);
  virtual bool precise() const override;
  virtual json to_json() const override;

 private:
  std::string class_name;
  std::vector<TorchParam> ctor_params;
  std::vector<TorchParam> member_params;
};

#endif
