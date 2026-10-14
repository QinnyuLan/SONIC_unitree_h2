"""Small nonblocking terminal UI shared by the independent input clients."""
import curses
import signal
import sys


class Terminal:
    def __enter__(self):
        if not sys.stdin.isatty() or not sys.stdout.isatty():
            raise ValueError("Keyboard control requires an interactive terminal")
        self.previous_term = signal.getsignal(signal.SIGTERM)
        self.screen = curses.initscr()
        try:
            curses.noecho()
            curses.cbreak()
            self.screen.keypad(True)
            self.screen.nodelay(True)
            curses.set_escdelay(25)
            try:
                curses.curs_set(0)
            except curses.error:
                pass
            signal.signal(signal.SIGTERM, self.interrupt)
            return self
        except BaseException:
            self.__exit__()
            raise

    @staticmethod
    def interrupt(*_):
        raise KeyboardInterrupt

    def keys(self):
        keys = []
        names = {curses.KEY_UP: "up", curses.KEY_DOWN: "down", curses.KEY_LEFT: "left",
                 curses.KEY_RIGHT: "right", curses.KEY_ENTER: "\n"}
        for _ in range(32):
            try:
                key = self.screen.get_wch()
            except curses.error:
                break
            keys.append(names.get(key, key))
        return keys

    @property
    def rows(self):
        return self.screen.getmaxyx()[0]

    def draw(self, lines):
        self.screen.erase()
        height, width = self.screen.getmaxyx()
        for row, line in enumerate(lines[:height]):
            try:
                self.screen.addnstr(row, 0, line, max(0, width - 1))
            except curses.error:
                pass
        self.screen.refresh()

    def __exit__(self, *_):
        self.screen.keypad(False)
        curses.nocbreak()
        curses.echo()
        curses.endwin()
        signal.signal(signal.SIGTERM, self.previous_term)
