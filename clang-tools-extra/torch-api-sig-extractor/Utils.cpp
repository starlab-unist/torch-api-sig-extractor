#include "Utils.h"

#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <cassert>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

bool include(std::vector<std::string> &vec, std::string name) {
  for (auto n : vec) {
    if (n == name) return true;
  }

  return false;
}

std::string strip_ext(std::string filename) {
  return filename.substr(0, filename.find_last_of("."));
}

bool startswith(std::string base, std::string prefix) {
  return base.compare(0, prefix.size(), prefix) == 0;
}

bool endswith(std::string base, std::string suffix) {
  return base.length() >= suffix.length() &&
         base.compare(base.length() - suffix.length(), suffix.length(),
                      suffix) == 0;
}

std::string get_directory_path() {
  std::string file_path = __FILE__;
  std::string dir_path = file_path.substr(0, file_path.rfind("/"));

  return dir_path;
}

bool is_space(char c) {
  return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

std::string strip(std::string str) {
  if (str.size() == 0) return str;

  size_t pos_right = str.size() - 1;
  for (; pos_right > 0; pos_right--)
    if (!is_space(str[pos_right])) break;

  size_t pos_left = 0;
  for (; pos_left <= pos_right; pos_left++)
    if (!is_space(str[pos_left])) break;

  return str.substr(pos_left, pos_right - pos_left + 1);
}

std::vector<std::string> read_lines(std::string filepath) {
  std::vector<std::string> contents;
  std::string buffer;

  std::ifstream f(filepath);
  while (getline(f, buffer)) {
    std::string line = strip(buffer);
    if (line == "" || startswith(line, "#")) continue;
    contents.push_back(line);
  }
  f.close();
  return contents;
}

long GetEpoch(const std::string &Path) {
  struct stat St;
  if (stat(Path.c_str(), &St)) return 0;  // Can't stat, be conservative.
  return St.st_mtime;
}

char GetSeparator() { return '/'; }

std::string DirPlusFile(const std::string &DirPath,
                        const std::string &FileName) {
  return DirPath + GetSeparator() + FileName;
}

bool IsFile(const std::string &Path) {
  struct stat St;
  if (stat(Path.c_str(), &St)) return false;
  return S_ISREG(St.st_mode);
}

void ListFilesInDir(const std::string &Dir, std::vector<std::string> &V) {
  GetEpoch(Dir);

  DIR *D = opendir(Dir.c_str());
  if (!D) {
    printf("%s: %s; exiting\n", strerror(errno), Dir.c_str());
    exit(1);
  }
  while (auto E = readdir(D)) {
    std::string Path = DirPlusFile(Dir, E->d_name);
    if (E->d_type == DT_REG || E->d_type == DT_LNK ||
        (E->d_type == DT_UNKNOWN && IsFile(Path)))
      V.push_back(E->d_name);
  }
  closedir(D);
}

std::string torch_api_list_dir_name() { return "torch-api-list"; }
std::string torch_tensor_method_list_file_name() { return "torch::Tensor"; }
std::string torch_module_list_file_name() { return "torch::nn"; }

std::vector<std::string> get_torch_api_groups() {
  std::string torch_api_list_dir =
      DirPlusFile(get_directory_path(), torch_api_list_dir_name());
  std::vector<std::string> torch_api_groups;
  ListFilesInDir(torch_api_list_dir, torch_api_groups);
  return torch_api_groups;
}

std::map<std::string, std::set<std::string>> read_torch_function_list() {
  std::map<std::string, std::set<std::string>> torch_function_list;
  for (auto &torch_api_group : get_torch_api_groups()) {
    if (torch_api_group == torch_tensor_method_list_file_name() ||
        torch_api_group == torch_module_list_file_name() ||
        startswith(torch_api_group, "_"))
      continue;

    std::string torch_api_list_dir =
        DirPlusFile(get_directory_path(), torch_api_list_dir_name());
    std::string filepath = DirPlusFile(torch_api_list_dir, torch_api_group);
    std::vector<std::string> lines = read_lines(filepath);
    torch_function_list[torch_api_group] =
        std::set<std::string>(lines.begin(), lines.end());
  }
  return torch_function_list;
}

std::set<std::string> read_torch_tensor_method_list() {
  std::string torch_api_list_dir_path =
      DirPlusFile(get_directory_path(), torch_api_list_dir_name());
  std::string filepath = DirPlusFile(torch_api_list_dir_path,
                                     torch_tensor_method_list_file_name());
  std::vector<std::string> lines = read_lines(filepath);
  return std::set<std::string>(lines.begin(), lines.end());
}

std::set<std::string> read_torch_module_list() {
  std::string torch_api_list_dir_path =
      DirPlusFile(get_directory_path(), torch_api_list_dir_name());
  std::string filepath =
      DirPlusFile(torch_api_list_dir_path, torch_module_list_file_name());
  std::vector<std::string> lines = read_lines(filepath);
  return std::set<std::string>(lines.begin(), lines.end());
}

std::map<std::string, std::set<std::string>> torch_function_list;
std::set<std::string> torch_tensor_method_list;
std::set<std::string> torch_module_list;

void init_torch_api_list() {
  torch_function_list = read_torch_function_list();
  torch_tensor_method_list = read_torch_tensor_method_list();
  torch_module_list = read_torch_module_list();
}

const std::map<std::string, std::set<std::string>> &get_torch_function_list() {
  return torch_function_list;
}

const std::set<std::string> &get_torch_tensor_method_list() {
  return torch_tensor_method_list;
}

const std::set<std::string> &get_torch_module_list() {
  return torch_module_list;
}

void make_dir(std::string dir) {
  if (mkdir(dir.c_str(), 0775) != 0) {
    std::cout << "Failed to create directory " << dir << std::endl;
    exit(0);
  }
}

bool file_exist(std::string filename) {
  static std::set<std::string> generated_pathfinder;

  if (generated_pathfinder.find(filename) == generated_pathfinder.end()) {
    generated_pathfinder.insert(filename);
    return false;
  } else {
    return true;
  }
}
