"""Terminal-identical output of a detached job in a Colab cell.

`launch` starts `python colab/terminal.py STEM COLUMNS -- CMD...` in its own session, so the job outlives the browser
and the kernel. That runner gives CMD a pseudo-terminal (Rich draws its live progress bar exactly as in a terminal) and
feeds the byte stream to a pyte terminal emulator. On the VM's local disk (LOCAL_DIR) the lines that scroll off the
emulated screen are appended to NAME.html and the screen is rewritten to NAME.screen.html at most once a second; the
Drive copies STEM.log (plain) and STEM.html get the scrollback every DRIVE_EVERY seconds and on exit, to spare the Drive
mount. `follow` shows the local files in one display handle, so it re-attaches to a running job after a kernel restart.
"""
import fcntl
import html
import os
import select
import signal
import struct
import subprocess
import sys
import termios
import time
from typing import Dict, List, Optional, Sequence, Tuple

import pyte

COLUMNS, LINES = 140, 24
LOCAL_DIR = "/content/run"  # survives a kernel restart, not a new VM (which also ends the job)
DRIVE_EVERY = 30.0
COLOR_NAMES = ("black", "red", "green", "brown", "blue", "magenta", "cyan", "white")  # pyte's names for SGR 30-37
# the 16 ANSI colours of the VS Code terminal, dark and light theme
PALETTES = {
    "dark": ("#000000", "#cd3131", "#0dbc79", "#e5e510", "#2472c8", "#bc3fbc", "#11a8cd", "#e5e5e5",
             "#666666", "#f14c4c", "#23d18b", "#f5f543", "#3b8eea", "#d670d6", "#29b8db", "#e5e5e5"),
    "light": ("#000000", "#cd3131", "#00bc00", "#949800", "#0451a5", "#bc05bc", "#0598bc", "#555555",
              "#666666", "#cd3131", "#14ce14", "#b5ba00", "#0451a5", "#bc05bc", "#0598bc", "#a5a5a5"),
}


class _Screen(pyte.Screen):
    """A pyte screen that keeps the lines scrolled off its top and renders SGR 2 (dim), which pyte ignores.

    Dim is stored in the `blink` flag of pyte's fixed `Char` tuple; Rich never emits blink.
    """

    def __init__(self, columns: int, lines: int):
        super().__init__(columns, lines)
        self.scrolled: List[Dict[int, pyte.screens.Char]] = []

    def index(self) -> None:
        top, bottom = self.margins or (0, self.lines - 1)
        if top == 0 and self.cursor.y == bottom:
            self.scrolled.append(self.buffer[0])
        super().index()

    def select_graphic_rendition(self, *attrs: int, private: bool = False) -> None:
        if private:
            return
        super().select_graphic_rendition(*attrs)
        dim = self.cursor.attrs.blink
        params = iter(attrs)
        for attr in params:
            if attr in (38, 48):  # skip the colour arguments: 5;n or 2;r;g;b
                for _ in range(1 if next(params, None) == 5 else 3):
                    next(params, None)
            elif attr in (2, 5):
                dim = True
            elif attr in (0, 22, 25):
                dim = False
        self.cursor.attrs = self.cursor.attrs._replace(blink=dim)


def _color(value: str) -> str:
    name = value.removeprefix("bright")
    if name in COLOR_NAMES:
        return f"var(--c{COLOR_NAMES.index(name) + 8 * (name != value)})"
    return f"#{value}"


def _css(char: pyte.screens.Char) -> str:
    fg, bg = (char.bg, char.fg) if char.reverse else (char.fg, char.bg)
    decoration = " ".join(d for d, on in (("underline", char.underscore), ("line-through", char.strikethrough)) if on)
    styles = {
        f"color:{_color(fg)}": fg != "default",
        f"background:{_color(bg)}": bg != "default",
        "font-weight:bold": char.bold,
        "font-style:italic": char.italics,
        f"text-decoration:{decoration}": decoration,
        "opacity:.6": char.blink,
    }
    return ";".join(css for css, on in styles.items() if on)


def _render(line: Dict[int, pyte.screens.Char], columns: int) -> Tuple[str, str]:
    """(plain text, HTML) of one screen line, trailing blanks dropped."""
    chars = [line[x] for x in range(columns)]
    end = max((x + 1 for x, c in enumerate(chars) if c.data.strip() or c.bg != "default" or c.reverse), default=0)
    runs: List[Tuple[str, str]] = []
    for char in chars[:end]:
        style = _css(char)
        if runs and runs[-1][0] == style:
            runs[-1] = (style, runs[-1][1] + char.data)
        else:
            runs.append((style, char.data))
    plain = "".join(text for _, text in runs)
    spans = "".join(f'<span style="{s}">{html.escape(t)}</span>' if s else html.escape(t) for s, t in runs)
    return plain, spans


def _local(stem: str) -> str:
    return os.path.join(LOCAL_DIR, os.path.basename(stem))


class Recorder:
    """Emulated terminal: scrollback to the local and Drive `.html` (and Drive `.log`), screen to the local disk.

    The local screen file starts with `<history lines> <runner pid> <exit code or ->`, so a reader can join it to the
    local scrollback without showing a line twice.
    """

    def __init__(self, stem: str, pid: int, columns: int = COLUMNS, lines: int = LINES):
        self.stem, self.local, self.pid = stem, _local(stem), pid
        os.makedirs(LOCAL_DIR, exist_ok=True)
        self.screen = _Screen(columns, lines)
        self.stream = pyte.ByteStream(self.screen)
        with open(f"{self.local}.html", "a+", encoding="utf-8") as f:
            f.seek(0)
            self.history = sum(1 for _ in f)
        self._shown = ""
        self._unsynced: List[Tuple[str, str]] = []
        self._synced = time.monotonic()

    def feed(self, data: bytes) -> None:
        self.stream.feed(data)

    def flush(self, rc: str = "-") -> None:
        columns = self.screen.columns
        if self.screen.scrolled:
            rendered = [_render(line, columns) for line in self.screen.scrolled]
            with open(f"{self.local}.html", "a", encoding="utf-8") as f:
                f.writelines(spans + "\n" for _, spans in rendered)
            self._unsynced += rendered
            self.history += len(rendered)
            self.screen.scrolled.clear()
        if self._unsynced and (rc != "-" or time.monotonic() - self._synced >= DRIVE_EVERY):
            try:
                with open(f"{self.stem}.log", "a", encoding="utf-8") as plain, \
                        open(f"{self.stem}.html", "a", encoding="utf-8") as rich:
                    plain.writelines(text + "\n" for text, _ in self._unsynced)
                    rich.writelines(spans + "\n" for _, spans in self._unsynced)
                self._unsynced.clear()
            except OSError:  # a Drive hiccup is retried next time: a runner crash would close the job's pty
                pass
            self._synced = time.monotonic()
        rows = [_render(self.screen.buffer[y], columns)[1] for y in range(self.screen.lines)]
        while rows and not rows[-1]:
            rows.pop()
        shown = "\n".join([f"{self.history} {self.pid} {rc}", *rows])
        if shown != self._shown:
            with open(f"{self.local}.screen.html", "w", encoding="utf-8") as f:
                f.write(shown + "\n")
            self._shown = shown

    def close(self, rc: int) -> None:
        """Moves the screen into the scrollback, as a terminal does when the next prompt pushes it up."""
        newline = "\r\n" if self.screen.cursor.x else ""
        self.feed(f"{newline}===== exit code {rc}\r\n".encode())
        used = [y for y in range(self.screen.lines) if _render(self.screen.buffer[y], self.screen.columns)[0]]
        self.screen.scrolled.extend(self.screen.buffer[y] for y in range(max(used, default=-1) + 1))
        self.screen.reset()
        self.flush(str(rc))


def run(stem: str, cmd: Sequence[str], columns: int = COLUMNS, lines: int = LINES) -> int:
    """Runs `cmd` on a pty of `columns` x `lines` and records it; SIGINT to this process reaches `cmd` as Ctrl-C."""
    recorder = Recorder(stem, os.getpid(), columns, lines)
    recorder.feed(f"===== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd)}\r\n".encode())
    master, slave = os.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", lines, columns, 0, 0))
    env = {k: v for k, v in os.environ.items() if k not in ("NO_COLOR", "TTY_COMPATIBLE", "TTY_INTERACTIVE")}
    env.update(TERM="xterm-256color", COLORTERM="truecolor", COLUMNS=str(columns), LINES=str(lines),
               PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=slave, stderr=slave, env=env)
    os.close(slave)
    signal.signal(signal.SIGINT, lambda *_: proc.send_signal(signal.SIGINT))
    last = 0.0
    while True:
        if select.select([master], [], [], 1.0)[0]:
            try:
                data = os.read(master, 1 << 16)
            except OSError:  # EIO once every process holding the pty has exited
                data = b""
            if not data:
                break
            recorder.feed(data)
        elif proc.poll() is not None:  # exited, and whatever still holds the pty stays silent
            break
        if time.monotonic() - last >= 1.0:
            recorder.flush()
            last = time.monotonic()
    rc = proc.wait()
    recorder.close(rc)
    return rc


def _state(stem: str) -> Optional[Tuple[int, int, str, List[str]]]:
    """(history lines, runner pid, exit code or -, screen rows) from the local screen file; None if missing."""
    try:
        with open(f"{_local(stem)}.screen.html", encoding="utf-8") as f:
            header, *rows = f.read().rstrip("\n").split("\n")
        history, pid, rc = header.split(" ")
        return int(history), int(pid), rc, rows
    except (OSError, ValueError):
        return None


def _alive(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return b"terminal.py" in f.read()
    except OSError:
        return False


def running(stem: str) -> bool:
    state = _state(stem)
    return state is not None and state[2] == "-" and _alive(state[1])


def launch(stem: str, cmd: Sequence[str], cwd: str, columns: int = COLUMNS) -> None:
    """Starts `cmd` detached under the recorder unless it already runs (resources: src/utils/resources.py)."""
    if running(stem):
        print(f"{stem} is already running")
        return
    if os.path.exists(f"{_local(stem)}.screen.html"):
        os.remove(f"{_local(stem)}.screen.html")
    with open(f"{stem}.log", "a") as log:  # a crash of the runner itself lands in the plain log
        runner = subprocess.Popen([sys.executable, os.path.abspath(__file__), stem, str(columns), "--", *cmd],
                                  cwd=cwd, stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
    while _state(stem) is None:
        if runner.poll() is not None:
            raise RuntimeError(f"the runner exited with code {runner.returncode}; see {stem}.log")
        time.sleep(0.2)
    print(f"{stem}: runner pid {runner.pid}")


def stop(stem: str) -> None:
    """Ctrl-C for the job; `follow` then shows its traceback and exit code."""
    if running(stem):
        os.kill(_state(stem)[1], signal.SIGINT)


def _light_theme() -> bool:
    # as Lightning's RichProgressBar (1.9) does to keep its colours readable on Colab's light theme
    try:
        from google.colab import output
        # eval_js waits for the browser without a limit by default, which would block follow() before any output
        return bool(output.eval_js('document.documentElement.matches("[theme=light]")', timeout_sec=3))
    except Exception:
        return False


def frame(history: List[str], rows: List[str], light: bool) -> str:
    colors = ";".join(f"--c{i}:{c}" for i, c in enumerate(PALETTES["light" if light else "dark"]))
    body = "\n".join([*history, *rows])
    return (f'<pre style="{colors};margin:0;line-height:1.25;font-family:monospace;white-space:pre;'
            f'overflow-x:auto">{body}</pre>')


def follow(stem: str, lines: int = 400, every: float = 1.0) -> Optional[str]:
    """Shows the job's last `lines` scrollback lines and its screen in one output, updated in place, until it exits.

    Interrupting this cell only stops following; `follow` again re-attaches. Returns the exit code. Without local
    files (a new VM) it shows the tail of the Drive scrollback once.
    """
    from IPython.display import HTML, display

    print(f"following {stem} (Ctrl-C / interrupt only stops watching)", flush=True)
    light = _light_theme()
    if _state(stem) is None:
        if not os.path.exists(f"{stem}.html"):
            print(f"{stem}: no job on this VM and no scrollback on Drive")
            return None
        with open(f"{stem}.html", encoding="utf-8") as f:
            display(HTML(frame(f.read().split("\n")[:-1][-lines:], [], light)))
        print(f"{stem}: no job on this VM; the Drive scrollback lags the job by up to {DRIVE_EVERY:.0f} s")
        return None
    handle = display(HTML(""), display_id=True)
    history: List[str] = []
    offset, shown, pending = 0, "", ""
    try:
        while True:
            state = _state(stem)
            if state is not None:
                count, pid, rc, rows = state
                with open(f"{_local(stem)}.html", "rb") as f:
                    f.seek(offset)
                    data = f.read()
                offset += len(data)
                *complete, pending = (pending + data.decode("utf-8", "replace")).split("\n")
                history.extend(complete)
                page = frame(history[max(0, count - lines):count], rows, light)
                if page != shown:
                    handle.update(HTML(page))
                    shown = page
                if rc != "-":
                    return rc
                if not _alive(pid):
                    print(f"{stem}: the runner (pid {pid}) is gone without an exit code")
                    return None
            time.sleep(every)
    except KeyboardInterrupt:
        print(f"stopped following; {stem} keeps running")
        return None


if __name__ == "__main__":
    if len(sys.argv) < 5 or sys.argv[3] != "--":
        raise SystemExit("usage: terminal.py STEM COLUMNS -- CMD...")
    sys.exit(run(sys.argv[1], sys.argv[4:], columns=int(sys.argv[2])))
