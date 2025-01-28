#include "TorchTypes.h"

using namespace llvm;
using namespace clang;
using json = nlohmann::json;

Optional<int> literal_to_int(Expr *literal) {
  if (literal == nullptr) return None;

  if (const auto *il = dyn_cast<IntegerLiteral>(literal)) {
    assert(il != nullptr);
    return il->getValue().getZExtValue();
  }

  return None;
}

TorchType::TorchType(TorchType::TorchTypeKind kind_) : kind(kind_) {}
void TorchType::set_default(Expr *default_expr) {}
bool TorchType::precise() const { return true; }
json TorchType::to_json() const {
  json j;
  j["type_kind"] = ttk_to_string();
  return j;
}
TorchType::TorchTypeKind TorchType::get_kind() const { return kind; }
std::string TorchType::ttk_to_string() const {
  switch (kind) {
    case TTK_Unknown:
      return "Unknown";
    case TTK_Void:
      return "Void";
    case TTK_Enum:
      return "Enum";
    case TTK_Int:
      return "Int";
    case TTK_SymInt:
      return "SymInt";
    case TTK_UnsignedInt:
      return "UnsignedInt";
    case TTK_Dtype:
      return "Dtype";
    case TTK_Bool:
      return "Bool";
    case TTK_String:
      return "String";
    case TTK_Float:
      return "Float";
    case TTK_Double:
      return "Double";
    case TTK_MemoryFormat:
      return "MemoryFormat";
    case TTK_Layout:
      return "Layout";
    case TTK_Device:
      return "Device";
    case TTK_Variant:
      return "Variant";
    case TTK_Vector:
      return "Vector";
    case TTK_ArrayRef:
      return "ArrayRef";
    case TTK_OptionalArrayRef:
      return "OptionalArrayRef";
    case TTK_ExpandingArray:
      return "ExpandingArray";
    case TTK_ExpandingArrayWithOptionalElem:
      return "ExpandingArrayWithOptionalElem";
    case TTK_Tuple:
      return "Tuple";
    case TTK_Pair:
      return "Pair";
    case TTK_Tensor:
      return "Tensor";
    case TTK_Scalar:
      return "Scalar";
    case TTK_Optional:
      return "Optional";
    case TTK_APIOptions:
      return "APIOptions";
    default: {
      assert(false);
      return "";
    }
  }
}

TorchUnknownType::TorchUnknownType(std::string type_str_)
    : TorchType(TTK_Unknown), type_str(type_str_) {}
bool TorchUnknownType::precise() const { return false; }
json TorchUnknownType::to_json() const {
  json j = TorchType::to_json();
  j["type_str"] = type_str;
  return j;
}

TorchVoidType::TorchVoidType() : TorchType(TTK_Void) {}

TorchIntType::TorchIntType(std::string specifier_)
    : TorchType(TTK_Int), specifier(specifier_) {}
void TorchIntType::set_default(Expr *default_expr) {
  default_value = literal_to_int(default_expr);
}
void TorchIntType::set_default(int value) { default_value = value; }
json TorchIntType::to_json() const {
  json j = TorchType::to_json();
  j["specifier"] = specifier;
  if (default_value.hasValue()) j["default_value"] = default_value.getValue();
  return j;
}

TorchSymIntType::TorchSymIntType() : TorchType(TTK_SymInt) {}
void TorchSymIntType::set_default(Expr *default_expr) {
  default_value = literal_to_int(default_expr);
}
void TorchSymIntType::set_default(int value) { default_value = value; }
json TorchSymIntType::to_json() const {
  json j = TorchType::to_json();
  if (default_value.hasValue()) j["default_value"] = default_value.getValue();
  return j;
}

TorchUnsignedIntType::TorchUnsignedIntType() : TorchType(TTK_UnsignedInt) {}
void TorchUnsignedIntType::set_default(Expr *default_expr) {
  Optional<int> n_opt = literal_to_int(default_expr);
  if (n_opt.hasValue()) default_value = n_opt.getValue();
}
void TorchUnsignedIntType::set_default(unsigned int value) {
  default_value = value;
}
json TorchUnsignedIntType::to_json() const {
  json j = TorchType::to_json();
  if (default_value.hasValue()) j["default_value"] = default_value.getValue();
  return j;
}

TorchBoolType::TorchBoolType() : TorchType(TTK_Bool) {}

TorchStringType::TorchStringType(bool is_view_)
    : TorchType(TTK_String), is_view(is_view_) {}
json TorchStringType::to_json() const {
  json j = TorchType::to_json();
  j["is_view"] = is_view;
  return j;
}

TorchFloatType::TorchFloatType() : TorchType(TTK_Float) {}

TorchDoubleType::TorchDoubleType() : TorchType(TTK_Double) {}

TorchMemoryFormatType::TorchMemoryFormatType() : TorchType(TTK_MemoryFormat) {}

TorchLayoutType::TorchLayoutType() : TorchType(TTK_Layout) {}

TorchDeviceType::TorchDeviceType() : TorchType(TTK_Device) {}

TorchDtypeType::TorchDtypeType() : TorchType(TTK_Dtype) {}

TorchVariantType::TorchVariantType(
    std::vector<std::unique_ptr<TorchType>> ttypes_)
    : TorchType(TTK_Variant) {
  ttypes = std::move(ttypes_);
}
void TorchVariantType::set_default(Expr *default_expr) {
  for (auto &ttype : ttypes) ttype->set_default(default_expr);
}
bool TorchVariantType::precise() const {
  for (auto &ttype : ttypes) {
    assert(ttype != nullptr);
    if (!ttype->precise()) return false;
  }
  return true;
}
json TorchVariantType::to_json() const {
  json j = TorchType::to_json();
  std::vector<json> j_ttypes;
  for (auto &ttype : ttypes) j_ttypes.push_back(ttype->to_json());
  j["types"] = j_ttypes;
  return j;
}

TorchEnumType::TorchEnumType(std::string enum_name_) : TorchType(TTK_Enum) {
  enum_name = enum_name_;
}
json TorchEnumType::to_json() const {
  json j = TorchType::to_json();
  j["enum_name"] = enum_name;
  return j;
}

TorchVectorType::TorchVectorType(std::unique_ptr<TorchType> value_type_)
    : TorchType(TTK_Vector) {
  value_type = std::move(value_type_);
}
bool TorchVectorType::precise() const {
  assert(value_type != nullptr);
  return value_type->precise();
}
json TorchVectorType::to_json() const {
  json j = TorchType::to_json();
  assert(value_type != nullptr);
  j["value_type"] = value_type->to_json();
  return j;
}

TorchArrayRefType::TorchArrayRefType(std::unique_ptr<TorchType> value_type_)
    : TorchType(TTK_ArrayRef) {
  value_type = std::move(value_type_);
}
bool TorchArrayRefType::precise() const {
  assert(value_type != nullptr);
  return value_type->precise();
}
json TorchArrayRefType::to_json() const {
  json j = TorchType::to_json();
  assert(value_type != nullptr);
  j["value_type"] = value_type->to_json();
  return j;
}

TorchOptionalArrayRefType::TorchOptionalArrayRefType(
    std::unique_ptr<TorchType> value_type_)
    : TorchType(TTK_ArrayRef) {
  value_type = std::move(value_type_);
}
bool TorchOptionalArrayRefType::precise() const {
  assert(value_type != nullptr);
  return value_type->precise();
}
json TorchOptionalArrayRefType::to_json() const {
  json j = TorchType::to_json();
  assert(value_type != nullptr);
  j["value_type"] = value_type->to_json();
  return j;
}

TorchExpandingArrayType::TorchExpandingArrayType(
    size_t size_, std::unique_ptr<TorchType> value_type_)
    : TorchType(TTK_ExpandingArray) {
  size = size_;
  value_type = std::move(value_type_);
}
void TorchExpandingArrayType::set_default(Expr *default_expr) {
  assert(value_type != nullptr);
  value_type->set_default(default_expr);
}
bool TorchExpandingArrayType::precise() const {
  assert(value_type != nullptr);
  return value_type->precise();
}
json TorchExpandingArrayType::to_json() const {
  json j = TorchType::to_json();
  j["size"] = size;
  assert(value_type != nullptr);
  j["value_type"] = value_type->to_json();
  return j;
}

TorchExpandingArrayWithOptionalElemType::
    TorchExpandingArrayWithOptionalElemType(
        size_t size_, std::unique_ptr<TorchType> value_type_)
    : TorchType(TTK_ExpandingArrayWithOptionalElem) {
  size = size_;
  value_type = std::move(value_type_);
}
void TorchExpandingArrayWithOptionalElemType::set_default(Expr *default_expr) {
  assert(value_type != nullptr);
  value_type->set_default(default_expr);
}
bool TorchExpandingArrayWithOptionalElemType::precise() const {
  assert(value_type != nullptr);
  return value_type->precise();
}
json TorchExpandingArrayWithOptionalElemType::to_json() const {
  json j = TorchType::to_json();
  j["size"] = size;
  assert(value_type != nullptr);
  j["value_type"] = value_type->to_json();
  return j;
}

TorchTupleType::TorchTupleType(std::vector<std::unique_ptr<TorchType>> ttypes_)
    : TorchType(TTK_Tuple) {
  ttypes = std::move(ttypes_);
}
bool TorchTupleType::precise() const {
  for (auto &ttype : ttypes) {
    assert(ttype != nullptr);
    if (!ttype->precise()) return false;
  }
  return true;
}
json TorchTupleType::to_json() const {
  json j = TorchType::to_json();
  std::vector<json> j_ttypes;
  for (auto &ttype : ttypes) j_ttypes.push_back(ttype->to_json());
  j["types"] = j_ttypes;
  return j;
}

TorchPairType::TorchPairType(std::vector<std::unique_ptr<TorchType>> ttypes_)
    : TorchType(TTK_Pair) {
  ttypes = std::move(ttypes_);
  assert(ttypes.size() == 2);
}
bool TorchPairType::precise() const {
  for (auto &ttype : ttypes) {
    assert(ttype != nullptr);
    if (!ttype->precise()) return false;
  }
  return true;
}
json TorchPairType::to_json() const {
  json j = TorchType::to_json();
  std::vector<json> j_ttypes;
  for (auto &ttype : ttypes) j_ttypes.push_back(ttype->to_json());
  j["types"] = j_ttypes;
  return j;
}

TorchTensorType::TorchTensorType() : TorchType(TTK_Tensor) {}

TorchScalarType::TorchScalarType() : TorchType(TTK_Scalar) {}

TorchOptionalType::TorchOptionalType(std::unique_ptr<TorchType> value_type_)
    : TorchType(TTK_Optional) {
  value_type = std::move(value_type_);
}
void TorchOptionalType::set_default(Expr *default_expr) {
  assert(value_type != nullptr);
  value_type->set_default(default_expr);
}
bool TorchOptionalType::precise() const {
  assert(value_type != nullptr);
  return value_type->precise();
}
json TorchOptionalType::to_json() const {
  json j = TorchType::to_json();
  assert(value_type != nullptr);
  j["value_type"] = value_type->to_json();
  return j;
}

TorchAPIOptionsType::TorchAPIOptionsType(std::string class_name_,
                                         std::vector<TorchParam> ctor_params_,
                                         std::vector<TorchParam> member_params_)
    : TorchType(TTK_APIOptions) {
  class_name = class_name_;
  ctor_params = std::move(ctor_params_);
  member_params = std::move(member_params_);
}
bool TorchAPIOptionsType::precise() const {
  for (auto &ctor_param_type : ctor_params) {
    auto &param_type = ctor_param_type.second;
    assert(param_type != nullptr);
    if (!param_type->precise()) return false;
  }
  for (auto &member_param_type : member_params) {
    auto &param_type = member_param_type.second;
    assert(param_type != nullptr);
    if (!param_type->precise()) return false;
  }
  return true;
}
json TorchAPIOptionsType::to_json() const {
  json j = TorchType::to_json();
  j["class_name"] = class_name;

  std::vector<json> j_ctor_params;
  for (auto &param : ctor_params) {
    json j_param;
    j_param["param_name"] = param.first;
    j_param["param_type"] = param.second->to_json();
    j_ctor_params.push_back(j_param);
  }
  j["ctor_params"] = j_ctor_params;

  std::vector<json> j_member_params;
  for (auto &param : member_params) {
    json j_param;
    j_param["param_name"] = param.first;
    j_param["param_type"] = param.second->to_json();
    j_member_params.push_back(j_param);
  }
  j["member_params"] = j_member_params;

  return j;
}
