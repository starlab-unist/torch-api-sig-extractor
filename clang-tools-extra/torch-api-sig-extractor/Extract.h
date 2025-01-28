#ifndef TORCH_API_SIGNATURE_EXTRACTOR_EXTRACT
#define TORCH_API_SIGNATURE_EXTRACTOR_EXTRACT

#include "TorchTypes.h"

extern bool option_class_done;

std::unique_ptr<TorchType> extractTorchType(clang::QualType t, ASTContext &Ctx);

#endif
