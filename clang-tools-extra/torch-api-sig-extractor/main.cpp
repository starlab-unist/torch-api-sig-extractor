#include <fstream>
#include <iostream>
#include <regex>

#include "Extract.h"
#include "TorchAPI.h"
#include "Utils.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/CommandLine.h"

using namespace clang::tooling;
using namespace llvm;
using namespace clang;

json output;

class FindNamedClassVisitor
    : public RecursiveASTVisitor<FindNamedClassVisitor> {
 public:
  explicit FindNamedClassVisitor(ASTContext *Context) : Context(Context) {}

  bool VisitFunctionDecl(FunctionDecl *Declaration) {
    std::string function_name = Declaration->getNameAsString();
    std::string function_name_qualified =
        Declaration->getQualifiedNameAsString();

    std::string function_group;
    for (auto &entry : get_torch_function_list()) {
      if (entry.second.find(function_name_qualified) != entry.second.end()) {
        function_group = std::regex_replace(entry.first, std::regex("::"), "_");
        break;
      }
    }
    if (function_group == "") return true;

    // std::cout << "Generating fuzz target of API `" << function_name_qualified
    //           << "`..." << std::endl;

    option_class_done = false;
    auto param_decls = Declaration->parameters();
    std::vector<TorchParam> params;
    for (auto param_decl : param_decls) {
      std::string param_name = param_decl->getNameAsString();
      clang::QualType t = param_decl->getType();
      std::unique_ptr<TorchType> param_type = extractTorchType(t, *Context);
      if (param_type == nullptr) {
        std::cerr << "WARNING: Parsing fail on param `" << param_name << "`.\n"
                  << "         Type `" << t.getAsString()
                  << "` is not supported.\n"
                  << "FAILED: Generating fuzz target of API `"
                  << function_name_qualified << "` failed." << std::endl;
        return true;
      }
      params.push_back({param_name, std::move(param_type)});
    }
    std::unique_ptr<TorchType> return_type =
        extractTorchType(Declaration->getReturnType(), *Context);

    TorchFunction torch_function(function_name_qualified, std::move(params),
                                 std::move(return_type));
    output.push_back(torch_function.to_json());

    return true;
  }

  Optional<std::vector<TorchParam>> extractModuleCtor(
      CXXConstructorDecl *ctor) {
    if (ctor->isCopyOrMoveConstructor() ||
        ctor->isSpecializationCopyingObject() ||
        ctor->isInheritingConstructor())
      return None;

    option_class_done = false;
    std::vector<TorchParam> params;
    for (const auto *param : ctor->parameters()) {
      std::string param_name = param->getNameAsString();
      clang::QualType t = param->getType();
      std::unique_ptr<TorchType> param_type = extractTorchType(t, *Context);
      if (param_type == nullptr) {
        std::cerr << "WARNING: Parsing fail on param `" << param_name << "`.\n"
                  << "         Type `" << t.getAsString()
                  << "` is not supported." << std::endl;
        return None;
      }
      params.push_back({param_name, std::move(param_type)});
    }
    return params;
  }

  Optional<std::vector<TorchParam>> extractForward(CXXMethodDecl *forward) {
    std::vector<TorchParam> params;
    for (const auto *param : forward->parameters()) {
      std::string param_name = param->getNameAsString();
      clang::QualType t = param->getType();
      std::unique_ptr<TorchType> param_type = extractTorchType(t, *Context);
      if (param_type == nullptr) {
        std::cerr << "WARNING: Parsing fail on param `" << param_name << "`.\n"
                  << "         Type `" << t.getAsString()
                  << "` is not supported." << std::endl;
        return None;
      }
      params.push_back({param_name, std::move(param_type)});
    }
    return params;
  }

  std::vector<TorchParam> pickBest(
      std::vector<std::vector<TorchParam>> ctor_params_candidates) {
    assert(!ctor_params_candidates.empty());

    for (auto &&params : ctor_params_candidates)
      for (auto &&param : params)
        if (param.second->get_kind() == TorchType::TTK_APIOptions)
          return std::move(params);

    size_t num_params_best = 0;
    size_t best_idx = 0;
    for (size_t i = 0; i < ctor_params_candidates.size(); i++) {
      if (ctor_params_candidates[i].size() > num_params_best) {
        num_params_best = ctor_params_candidates[i].size();
        best_idx = i;
      }
    }

    return std::move(ctor_params_candidates[best_idx]);
  }

  void extractModule(CXXRecordDecl *Declaration) {
    std::string module_name = Declaration->getNameAsString();
    std::string module_name_qualified = Declaration->getQualifiedNameAsString();

    // std::cout << "Generating fuzz target of API `" << module_name_qualified
    //           << "`..." << std::endl;

    assert(!Declaration->bases().empty());
    const auto *elaborated =
        dyn_cast<ElaboratedType>(Declaration->bases_begin()->getType());
    assert(elaborated != nullptr);
    assert(elaborated->isSugared());
    const auto *tstype =
        dyn_cast<TemplateSpecializationType>(elaborated->desugar());
    assert(tstype->isSugared());
    if (!(tstype->desugar()
              ->getAs<RecordType>()
              ->getDecl()
              ->getNameAsString() == "ModuleHolder"))
      return;
    auto targs = tstype->template_arguments();
    assert(targs.size() == 1);
    auto *class_decl = dyn_cast<CXXRecordDecl>(
        targs[0].getAsType()->getAs<RecordType>()->getDecl());

    std::vector<std::vector<TorchParam>> ctor_params_candidates;
    std::vector<TorchParam> forward_params;
    std::unique_ptr<TorchType> return_type;

    for (auto ctor : class_decl->ctors())
      if (auto extracted = extractModuleCtor(ctor))
        ctor_params_candidates.push_back(std::move(extracted.getValue()));
    for (auto method : class_decl->methods()) {
      if (method->getNameAsString() == "forward") {
        if (auto extracted = extractForward(method)) {
          forward_params = std::move(extracted.getValue());
          return_type = extractTorchType(method->getReturnType(), *Context);
          break;
        }
      }
    }

    if (!class_decl->bases().empty()) {
      const TemplateSpecializationType *tstype2;
      if (const auto *etype =
              dyn_cast<ElaboratedType>(class_decl->bases_begin()->getType())) {
        tstype2 = dyn_cast<TemplateSpecializationType>(etype->desugar());
      } else {
        tstype2 = dyn_cast<TemplateSpecializationType>(
            class_decl->bases_begin()->getType());
      }

      if (tstype2 != nullptr) {
        assert(tstype2->isSugared());
        if (const auto *ctsdecl = dyn_cast<ClassTemplateSpecializationDecl>(
                tstype2->desugar()->getAs<RecordType>()->getDecl())) {
          if (!ctsdecl->bases().empty() &&
              ctsdecl->bases_begin()
                      ->getType()
                      ->getAs<RecordType>()
                      ->getDecl()
                      ->getNameAsString() == "NormImplBase") {
            auto targs = ctsdecl->bases_begin()
                             ->getType()
                             ->getAs<TemplateSpecializationType>()
                             ->template_arguments();
            assert(targs.size() == 3 &&
                   targs[2].getKind() == TemplateArgument::ArgKind::Type);
            option_class_done = false;
            std::vector<TorchParam> params;
            if (auto param_type =
                    extractTorchType(targs[2].getAsType(), *Context))
              params.push_back({"options", std::move(param_type)});
            ctor_params_candidates.push_back(std::move(params));
          } else {
            for (auto ctor : ctsdecl->ctors())
              if (auto extracted = extractModuleCtor(ctor))
                ctor_params_candidates.push_back(
                    std::move(extracted.getValue()));
          }
          if (forward_params.empty()) {
            for (auto method : ctsdecl->methods()) {
              if (method->getNameAsString() == "forward") {
                if (auto extracted = extractForward(method)) {
                  forward_params = std::move(extracted.getValue());
                  return_type =
                      extractTorchType(method->getReturnType(), *Context);
                  break;
                }
              }
            }
          }
        }
      }
    }

    if (ctor_params_candidates.empty() || forward_params.empty()) {
      std::cerr << "FAILED: Generating fuzz target of API `"
                << module_name_qualified << "` failed." << std::endl;
      return;
    }
    auto ctor_params = pickBest(std::move(ctor_params_candidates));

    TorchModule torch_module(module_name_qualified, std::move(ctor_params),
                             std::move(forward_params), std::move(return_type));

    output.push_back(torch_module.to_json());
  }

  bool extractTensorMethod(CXXMethodDecl *method) {
    std::string method_name = method->getNameAsString();
    // std::cout << "Generating fuzz target of Tensor method `" << method_name
    //           << "`..." << std::endl;

    std::vector<TorchParam> params;
    for (const auto *param : method->parameters()) {
      std::string param_name = param->getNameAsString();
      clang::QualType t = param->getType();
      std::unique_ptr<TorchType> param_type = extractTorchType(t, *Context);
      if (param_type == nullptr) {
        std::cerr << "WARNING: Parsing fail on param `" << param_name << "`.\n"
                  << "         Type `" << t.getAsString()
                  << "` is not supported.\n"
                  << "FAILED: Generating fuzz target of Tensor method `"
                  << method_name << "` failed." << std::endl;
        return false;
      }
      params.push_back({param_name, std::move(param_type)});
    }
    std::unique_ptr<TorchType> return_type =
        extractTorchType(method->getReturnType(), *Context);

    TorchTensorMethod torch_tensor_method(method_name, std::move(params),
                                          std::move(return_type));

    output.push_back(torch_tensor_method.to_json());

    return true;
  }

  bool VisitCXXRecordDecl(CXXRecordDecl *Declaration) {
    // We want to visit Tensor class only once
    static bool tensor_method_done = false;

    auto torch_module_list = get_torch_module_list();
    if (torch_module_list.find(Declaration->getQualifiedNameAsString()) !=
        torch_module_list.end())
      extractModule(Declaration);

    auto torch_tensor_method_list = get_torch_tensor_method_list();
    if (!tensor_method_done && Declaration->getNameAsString() == "Tensor")
      for (auto method : Declaration->methods())
        if (torch_tensor_method_list.find(method->getNameAsString()) !=
            torch_tensor_method_list.end())
          if (extractTensorMethod(method)) tensor_method_done = true;

    return true;
  }

 private:
  ASTContext *Context;
};

class FindNamedClassConsumer : public clang::ASTConsumer {
 public:
  explicit FindNamedClassConsumer(ASTContext *Context) : Visitor(Context) {}

  virtual void HandleTranslationUnit(clang::ASTContext &Context) override {
    Visitor.TraverseDecl(Context.getTranslationUnitDecl());
  }

 private:
  FindNamedClassVisitor Visitor;
};

class FindNamedClassAction : public clang::ASTFrontendAction {
 public:
  virtual std::unique_ptr<clang::ASTConsumer> CreateASTConsumer(
      clang::CompilerInstance &Compiler, llvm::StringRef InFile) override {
    return std::make_unique<FindNamedClassConsumer>(&Compiler.getASTContext());
  }
};

static llvm::cl::OptionCategory MyToolCategory("my-tool options");

int main(int argc, const char **argv) {
  init_torch_api_list();

  CommonOptionsParser OptionsParser(argc, argv, MyToolCategory);
  ClangTool Tool(OptionsParser.getCompilations(),
                 OptionsParser.getSourcePathList());

  Tool.run(newFrontendActionFactory<FindNamedClassAction>().get());

  std::ofstream f("torch-api-signatures.json");
  if (!f.is_open()) {
    std::cerr << "Failed to open output file" << std::endl;
    exit(1);
  }
  f << output.dump(4) << std::endl;

  return 0;
}
