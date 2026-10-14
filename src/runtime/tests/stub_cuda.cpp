// Interpose only the first CUDA call; startup tests must never touch the GPU.
#include <cuda_runtime_api.h>
#include <cstdlib>
#include <unistd.h>

extern "C" cudaError_t CUDARTAPI cudaGetDeviceCount(int* count) {
  if (std::getenv("SONIC_TEST_NO_GPU")) {
    *count = 0;
    return cudaSuccess;
  }
  constexpr char marker[] = "TEST: CUDA initialization is blocked\n";
  if (::write(STDERR_FILENO, marker, sizeof(marker) - 1) != sizeof(marker) - 1)
    ::_exit(2);
  for (;;) ::pause();
}
