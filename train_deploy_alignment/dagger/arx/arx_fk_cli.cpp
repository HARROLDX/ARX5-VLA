#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "arx_x5_src/interfaces/InterfaceTools.hpp"

int main(int argc, char** argv) {
  int end_type = 0;
  if (argc > 1) {
    end_type = std::stoi(argv[1]);
  }

  arx::x5::InterfacesTools tools(end_type);
  std::cout << std::setprecision(17);

  std::string line;
  while (std::getline(std::cin, line)) {
    if (line.empty()) {
      continue;
    }
    std::istringstream iss(line);
    std::vector<double> q(6, 0.0);
    for (int i = 0; i < 6; ++i) {
      if (!(iss >> q[i])) {
        std::cerr << "Expected 6 joint values per line, got: " << line << std::endl;
        return 2;
      }
    }
    std::vector<double> pose = tools.ForwardKinematicsRpy(q);
    if (pose.size() != 6) {
      std::cerr << "ARX ForwardKinematicsRpy returned " << pose.size() << " values" << std::endl;
      return 3;
    }
    for (size_t i = 0; i < pose.size(); ++i) {
      if (i) {
        std::cout << ' ';
      }
      std::cout << pose[i];
    }
    std::cout << '\n';
  }

  return 0;
}
