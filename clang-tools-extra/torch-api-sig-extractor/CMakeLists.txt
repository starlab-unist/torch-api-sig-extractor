set(LLVM_LINK_COMPONENTS support)

add_subdirectory(json)

add_clang_executable(extractor
  main.cpp
  Extract.cpp
  TorchTypes.cpp
  TorchAPI.cpp
  Utils.cpp
  )
target_link_libraries(extractor
  PRIVATE
  clangAST
  clangASTMatchers
  clangBasic
  clangFrontend
  clangSerialization
  clangTooling
  nlohmann_json::nlohmann_json
  )