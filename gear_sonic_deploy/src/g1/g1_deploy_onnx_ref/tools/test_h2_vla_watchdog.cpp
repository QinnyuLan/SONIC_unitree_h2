#include "../include/h2_vla_watchdog.hpp"
#include <cassert>
#include <limits>
#include <iostream>

int main() {
  std::vector<double> token(64, 0.2);
  assert(h2_vla::ValidPacket(token, 10.8, 10.0));
  assert(!h2_vla::ValidPacket(token, 10.8, 10.8));
  assert(!h2_vla::ValidPacket(token, 11.0, 10.0));
  token[5] = std::numeric_limits<double>::quiet_NaN();
  assert(!h2_vla::ValidPacket(token, 10.8, 10.0));
  token[5] = 1.3;
  assert(!h2_vla::ValidPacket(token, 10.8, 10.0));
  token.resize(63);
  assert(!h2_vla::ValidPacket(token, 10.8, 10.0));
  assert(!h2_vla::Expired(10.2, 10.0, 10.8));
  assert(h2_vla::Expired(10.3, 10.0, 10.8));
  assert(h2_vla::Expired(10.8, 10.7, 10.8));
  assert(h2_vla::Expired(9.9, 10.0, 10.8));
  std::cout << "H2 VLA token shape/finite/bounds/deadline/watchdog checks passed\n";
}
