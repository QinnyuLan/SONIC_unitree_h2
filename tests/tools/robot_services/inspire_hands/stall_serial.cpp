// Fault injection for the isolated watchdog test: let main-thread probing
// finish, then block a serial worker inside its device check.
#include <dlfcn.h>
#include <sys/syscall.h>
#include <termios.h>
#include <unistd.h>

extern "C" int tcgetattr(int fd, termios* attributes)
{
  using Function = int (*)(int, termios*);
  static auto original = reinterpret_cast<Function>(dlsym(RTLD_NEXT, "tcgetattr"));
  if (syscall(SYS_gettid) != getpid())
    for (;;) pause();
  return original(fd, attributes);
}
