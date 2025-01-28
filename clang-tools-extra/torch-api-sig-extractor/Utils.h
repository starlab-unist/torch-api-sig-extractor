#ifndef TORCH_API_SIGNATURE_EXTRACTOR_UTILS
#define TORCH_API_SIGNATURE_EXTRACTOR_UTILS

#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <fstream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

bool startswith(std::string base, std::string prefix);
bool endswith(std::string base, std::string suffix);
bool include(std::vector<std::string> &vec, std::string name);
std::string strip_ext(std::string filename);
std::string torch_tensor_method_list_file_name();
std::string torch_module_list_file_name();
void init_torch_api_list();
const std::map<std::string, std::set<std::string>> &get_torch_function_list();
const std::set<std::string> &get_torch_tensor_method_list();
const std::set<std::string> &get_torch_module_list();

#endif
