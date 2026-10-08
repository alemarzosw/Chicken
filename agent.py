#!/usr/bin/env python3
"""chicken — a local AI agent for the terminal, powered by Ollama.

Run `agent` in any folder, then type what you want. Type / to see the commands.
Config, preset tasks and saved sessions live in ~/.agent/
"""
import argparse
import atexit
import base64
import collections
import datetime
import difflib
import fnmatch
import gzip
import hashlib
import html
import json
import math
import os
import re
import shutil
import signal
import socket
import struct
import subprocess
import sys
import termios
import threading
import time
import tty
import urllib.parse
from contextlib import contextmanager
from html.parser import HTMLParser
from pathlib import Path

import requests

try:
    from prompt_toolkit import Application, PromptSession
    from prompt_toolkit.application import get_app
    from prompt_toolkit.patch_stdout import patch_stdout
    from prompt_toolkit.completion import Completer, Completion, PathCompleter
    from prompt_toolkit.document import Document
    from prompt_toolkit.filters import Condition, has_completions
    from prompt_toolkit.formatted_text import FormattedText
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import Layout, Window
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.lexers import Lexer
    from prompt_toolkit.styles import Style
    from prompt_toolkit.keys import Keys
    from prompt_toolkit.input.ansi_escape_sequences import ANSI_SEQUENCES
    # Shift+Enter: terminals that tell it apart from Enter send one of these codes (CSI u / xterm modifyOtherKeys).
    # They become F24, a key nothing else uses, so it can be bound to "new line".
    for _seq in ("\x1b[13;2u", "\x1b[27;2;13~"):
        ANSI_SEQUENCES[_seq] = Keys.F24
    HAVE_PT = True
except ImportError:
    HAVE_PT = False

HOME = Path.home()
CONF_DIR = HOME / ".agent"
CONF_FILE = CONF_DIR / "config.json"
TASKS_DIR = CONF_DIR / "tasks"
SESS_DIR = CONF_DIR / "sessions"
WIKI_DIR = CONF_DIR / "wiki"
TRASH_DIR = CONF_DIR / "trash"
HIST_FILE = CONF_DIR / "prompt_history"

# keyboard actions: name -> (description, default key)
ACTIONS = {
    "details":     ("Show/hide details: reasoning, full results and command logs", "c-b"),
    "plan":        ("Show the AI's current plan", "c-t"),
    "last_output": ("Show the full output of the last action", "c-o"),
    "info":        ("Show computer & AI status (CPU, GPU, RAM, context…)", "c-g"),
    "paste":       ("Paste from the clipboard", "c-v"),
    "newline":     ("Start a new line in your message (Enter sends it)", "escape enter"),
}
# keys only the newline action may use (they aren't Ctrl+letter shortcuts)
NEWLINE_KEYS = {"escape enter": "Alt+Enter (Esc then Enter also works)",
                "shift enter": "Shift+Enter · only if your terminal sends it separately (test with Push key)",
                "c-j": "Ctrl+J: a real line feed, works in every terminal"}
RESERVED_KEYS = {"c-c": "copy / interrupt / quit", "c-d": "quit", "c-m": "Enter", "c-j": "Enter",
                 "c-i": "Tab", "c-h": "Backspace", "c-z": "suspend", "c-s": "terminal freeze",
                 "c-q": "terminal resume", "c-l": "clear screen"}

DEFAULTS = {
    "model": "qwen3.5:4b",     # the main brain: the orchestrator (or, in classic routing, the agent that works)
    "host": "http://localhost:11434",
    "num_ctx": "auto",         # context size: auto = all the GPU memory the model leaves in the budget · a number = fixed
    "ctx_min": 8192,           # smallest context the main agent shrinks to so a helper fits beside it (else: eject)
    "compact_for_helpers": True,  # compact the conversation if that lets a helper fit beside the main agent
    "knowledge_auto": True,    # add the most relevant sections of the knowledge wikis to every worker's task
    "knowledge_chars": 6000,   # at most this much wiki text per task
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "think": True,             # let qwen3 reason before answering
    "verbose": False,          # details mode (ctrl+b): show reasoning and full outputs
    "root": str(HOME),         # the agent can never touch files outside this folder
    "protected": [".ssh", ".gnupg", ".agent/config.json"],
    "max_tool_rounds": 30,     # tool calls per normal request
    "loop_max_steps": 80,      # model calls per /loop run
    "command_timeout": 300,
    "max_answer_tokens": 8192, # cap per answer (reasoning included), so nothing runs away
    "wiki_max_files": 12,      # file pages written per /save -wiki
    "theme": "chicken",        # /theme: colour palette (terminal = your terminal's own colours)
    "prompt_color": "",        # /color: colour of the ❯ prompt arrow ("" = the theme's)
    "agents_mode": "ask",      # helpers: ask (you approve each) · auto · manual (only call -agent)
    "mode": "ask",             # /mode ask: confirm each worker and each action · auto: no confirmations
    "routing": "orchestrator",  # /mode orchestrator · bonsai-only · bonsai-workers · classic
    "heavy_model": "bonsai2:27b",  # planner, hard coding, review, recovery; the only model in bonsai-only
    "orchestrator_ctx": 32768,  # the orchestrator only needs a small memory: goal, plan, state, last result
    "vram_budget_gb": "90%",   # GPU limit: a share of the card ("90%") or GB; above it Chicken pauses, compresses, ejects
    # models that Ollama can't run, served by a llama.cpp server that Chicken starts and stops itself
    "servers": {
        "bonsai2:27b": {
            "bin": "~/.local/share/prism-llama/bin/llama-server",
            "model": "~/.local/share/prism-llama/models/Ternary-Bonsai-2-27B-PTQ1_0.gguf",
            "libs": "/usr/local/lib/ollama/cuda_v12",
            "port": 8081,
            "description": "Bonsai 2 27B (PrismML, ternary Qwen3.8-27B) · 7.4 GB",
        },
        # small models served by llama.cpp on the GPU
        "qwen3.5:2b": {
            "bin": "~/.local/share/prism-llama/bin/llama-server",
            "model": "~/.local/share/prism-llama/models/Qwen3.5-2B-Q4_K_M.gguf",
            "libs": "/usr/local/lib/ollama/cuda_v12", "port": 8082, "context": 4096,
            "description": "Qwen3.5 2B on the GPU · trivial tasks",
        },
        "qwen3.5:0.8b": {
            "bin": "~/.local/share/prism-llama/bin/llama-server",
            "model": "~/.local/share/prism-llama/models/Qwen3.5-0.8B-Q4_K_M.gguf",
            "libs": "/usr/local/lib/ollama/cuda_v12", "port": 8083, "context": 4096,
            "description": "Qwen3.5 0.8B on the GPU · /btw",
        },
    },
    "models": {},               # your own models for the orchestrator: {"name": ["good at", "intelligence", "speed"]}
    "btw_model": "qwen3.5:0.8b",  # /btw: small read-only GPU model for side questions ("" = the main model)
    "keys": {k: v[1] for k, v in ACTIONS.items()},
}

SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".cache", ".mozilla",
             ".local", ".npm", ".cargo", ".rustup", "snap", ".Trash", ".agent"}
MAX_READ_LINES = 400
MAX_TOOL_CHARS = 12000
UA = "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0"

# ---------------------------------------------------------------- terminal look

TTY = sys.stdout.isatty()


def c(code, s):
    return f"\033[{code}m{s}\033[0m" if TTY else str(s)


THEME = {}        # palette of the active /theme; empty = the terminal's own colours


def _tc(name, code, s):
    h = THEME.get(name)
    return c(f"38;2;{int(h[1:3], 16)};{int(h[3:5], 16)};{int(h[5:7], 16)}" if h else code, s)


def dim(s): return _tc("dim", "2", s)
def bold(s): return c("1", s)
def red(s): return _tc("red", "31", s)
def green(s): return _tc("green", "32", s)
def yellow(s): return _tc("yellow", "33", s)
def cyan(s): return _tc("cyan", "36", s)
def magenta(s): return _tc("magenta", "35", s)


# /theme: colour palettes. bg/fg are also sent to the terminal (OSC 10/11) while the agent runs.
THEMES = {
    "chicken":    {"desc": "Chicken's own dark palette, as in the promo (default)", "bg": "#0a0e14", "fg": "#eef2f7",
                   "panel": "#1c2430", "red": "#ff6a5f", "green": "#5cc98b", "yellow": "#f5c542", "cyan": "#5fcff5",
                   "magenta": "#e684d4", "accent": "#5fcff5", "muted": "#98a4b8", "dim": "#7f8aa0",
                   "prompt": "#ff6a5f"},
    "terminal":   {"desc": "Your terminal's own colours"},
    "dracula":    {"desc": "Dark purple", "bg": "#282a36", "fg": "#f8f8f2", "panel": "#44475a", "red": "#ff5555",
                   "green": "#50fa7b", "yellow": "#f1fa8c", "cyan": "#8be9fd", "magenta": "#ff79c6",
                   "accent": "#bd93f9", "muted": "#7a86b8"},
    "catppuccin": {"desc": "Soft pastel dark (Mocha)", "bg": "#1e1e2e", "fg": "#cdd6f4", "panel": "#313244",
                   "red": "#f38ba8", "green": "#a6e3a1", "yellow": "#f9e2af", "cyan": "#89dceb",
                   "magenta": "#f5c2e7", "accent": "#cba6f7", "muted": "#7f849c"},
    "nord":       {"desc": "Cool arctic blue", "bg": "#2e3440", "fg": "#d8dee9", "panel": "#3b4252", "red": "#bf616a",
                   "green": "#a3be8c", "yellow": "#ebcb8b", "cyan": "#88c0d0", "magenta": "#b48ead",
                   "accent": "#81a1c1", "muted": "#7b88a1"},
    "gruvbox":    {"desc": "Warm retro dark", "bg": "#282828", "fg": "#ebdbb2", "panel": "#3c3836", "red": "#fb4934",
                   "green": "#b8bb26", "yellow": "#fabd2f", "cyan": "#8ec07c", "magenta": "#d3869b",
                   "accent": "#83a598", "muted": "#928374"},
    "monokai":    {"desc": "Classic vivid dark", "bg": "#272822", "fg": "#f8f8f2", "panel": "#3e3d32", "red": "#f92672",
                   "green": "#a6e22e", "yellow": "#e6db74", "cyan": "#66d9ef", "magenta": "#ae81ff",
                   "accent": "#fd971f", "muted": "#8f8b77"},
    "solarized":  {"desc": "Solarized dark, low contrast", "bg": "#002b36", "fg": "#93a1a1", "panel": "#073642",
                   "red": "#dc322f", "green": "#859900", "yellow": "#b58900", "cyan": "#2aa198",
                   "magenta": "#d33682", "accent": "#268bd2", "muted": "#657b83"},
    "light":      {"desc": "White background, dark text", "bg": "#ffffff", "fg": "#24292f", "panel": "#eaeef2",
                   "red": "#cf222e", "green": "#1a7f37", "yellow": "#9a6700", "cyan": "#0969da",
                   "magenta": "#8250df", "accent": "#0969da", "muted": "#6e7781"},
}

# /color: colours for the ❯ prompt arrow ("" = the theme's own)
PROMPT_COLORS = {"theme": "", "coral": "#ff6a5f", "green": "#5fd75f", "cyan": "#5fd7ff", "blue": "#5f87ff", "purple": "#af87ff",
                 "pink": "#ff5fd7", "red": "#ff5f5f", "orange": "#ff8700", "yellow": "#ffd75f", "white": "#eeeeee"}
_TERM_COLORS_SET = False


def prompt_hex(cfg):
    col = cfg.get("prompt_color") or ""
    t = THEMES.get(cfg.get("theme"), {})
    return PROMPT_COLORS.get(col, col) or t.get("prompt") or t.get("green") or "#5fd75f"


def arrow(cfg):
    """The coloured ❯ used for the prompt and for echoed messages."""
    h = prompt_hex(cfg)
    return c(f"1;38;2;{int(h[1:3], 16)};{int(h[3:5], 16)};{int(h[5:7], 16)}", "❯ ")


def build_style(cfg):
    """The prompt_toolkit style for the current theme and prompt colour."""
    t = THEMES.get(cfg.get("theme"), {})
    if "bg" not in t:
        d = {"status": "#8a8a8a", "status.model": "#5fafd7 bold", "status.auto": "#ff5f5f bold",
             "spin": "#d75fd7 bold", "queued": "#87afd7", "btw": "#d7af5f", "ask": "#ffd75f bold",
             "bottom-toolbar": "noreverse #6c6c6c", "bottom-toolbar.flash": "noreverse #ffd75f bold",
             "completion-menu": "bg:#262626 #d0d0d0", "completion-menu.completion.current": "bg:#005f87 #ffffff bold",
             "completion-menu.meta.completion": "bg:#262626 #8a8a8a",
             "completion-menu.meta.completion.current": "bg:#005f87 #e4e4e4",
             "mhint": "#8a8a8a", "mfilter": "#ffd75f", "msel": "#5fd7ff bold", "mmeta": "#8a8a8a",
             "mmeta.sel": "#bcbcbc", "msep": "#4e4e4e", "mdetail": "#d0d0d0",
             "kb.border": "#5f5f87", "kb.head": "#87afd7 bold", "kb.box": "#8a8a8a",
             "kb.box.sel": "bg:#005f87 #ffffff bold", "kb.key": "#d0d0d0", "kb.changed": "#ffd75f bold",
             "kb.capture": "#ff5fd7 bold blink", "kb.ok": "#5fd75f", "kb.err": "#ff5f5f bold",
             "in.cmd": "#5fd7ff bold", "in.flag": "#d75fd7 bold", "in.shell": "#ffd75f",
             "in.agent": "#ff5fd7 bold underline"}
    else:
        bg, fg, pn, mu, ac = t["bg"], t["fg"], t["panel"], t["muted"], t["accent"]
        d = {"status": mu, "status.model": f"{ac} bold", "status.auto": f"{t['red']} bold",
             "spin": f"{t['magenta']} bold", "queued": t["cyan"], "btw": t["yellow"], "ask": f"{t['yellow']} bold",
             "bottom-toolbar": f"noreverse {mu}", "bottom-toolbar.flash": f"noreverse {t['yellow']} bold",
             "completion-menu": f"bg:{pn} {fg}", "completion-menu.completion.current": f"bg:{ac} {bg} bold",
             "completion-menu.meta.completion": f"bg:{pn} {mu}",
             "completion-menu.meta.completion.current": f"bg:{ac} {bg}",
             "mhint": mu, "mfilter": t["yellow"], "msel": f"{t['cyan']} bold", "mmeta": mu, "mmeta.sel": fg,
             "msep": mu, "mdetail": fg, "kb.border": mu, "kb.head": f"{ac} bold", "kb.box": mu,
             "kb.box.sel": f"bg:{ac} {bg} bold", "kb.key": fg, "kb.changed": f"{t['yellow']} bold",
             "kb.capture": f"{t['magenta']} bold blink", "kb.ok": t["green"], "kb.err": f"{t['red']} bold",
             "in.cmd": f"{t['cyan']} bold", "in.flag": f"{t['magenta']} bold", "in.shell": t["yellow"],
             "in.agent": f"{t['magenta']} bold underline"}
    d.update({"prompt": f"{prompt_hex(cfg)} bold", "spin.label": "bold", "mtitle": "bold", "mitem": ""})
    return Style.from_dict(d)


def apply_theme(cfg):
    """Make the chosen theme active: output colours, prompt style and terminal background."""
    global THEME, MENU_STYLE, _TERM_COLORS_SET
    THEME = {k: v for k, v in THEMES.get(cfg.get("theme"), {}).items() if k.startswith(("red", "green", "yellow", "cyan", "magenta", "dim"))}
    if HAVE_PT:
        MENU_STYLE = build_style(cfg)
    t = THEMES.get(cfg.get("theme"), {})
    if not TTY:
        return
    if "bg" in t:
        sys.stdout.write(f"\033]11;{t['bg']}\007\033]10;{t['fg']}\007")
        _TERM_COLORS_SET = True
    else:
        reset_terminal_colors()
    sys.stdout.flush()


def reset_terminal_colors():
    global _TERM_COLORS_SET
    if _TERM_COLORS_SET and TTY:
        sys.stdout.write("\033]111\007\033]110\007")
        sys.stdout.flush()
        _TERM_COLORS_SET = False


def gradient(text, start=(95, 215, 255), end=(215, 95, 215)):
    """Bold text fading from one 24-bit colour to another, character by character."""
    if not TTY:
        return text
    n = max(len(text) - 1, 1)
    parts = []
    for i, ch in enumerate(text):
        r, g, b = (int(a + (z - a) * i / n) for a, z in zip(start, end))
        parts.append(f"\033[1;38;2;{r};{g};{b}m{ch}")
    return "".join(parts) + "\033[0m"


# the pixel chicken shown at start-up: the promo's logo redrawn at 10×10
# R comb and wattle · W feathers · K eye · Y beak and legs · . empty
CHICKEN = [
    "....RR....",
    "...RRRR...",
    "...WWKWYY.",
    "...WWWWY..",
    "W..WWWWR..",
    "WW.WWWWR..",
    "WWWWWWWW..",
    ".WWWWWWW..",
    "...Y..Y...",
    "..YY.YY...",
]
CHICKEN_COLORS = {"R": (215, 38, 38), "W": (250, 250, 245), "K": (30, 30, 30), "Y": (250, 200, 30)}
WORDMARK = ((255, 236, 210), (255, 90, 74))      # the CHICKEN name: cream to coral


def pixel_lines(grid, colors):
    """Pixel art for the terminal: each character is two pixels stacked (▀ with its own colour and background)."""
    if not TTY:
        return []
    lines = []
    for y in range(0, len(grid), 2):
        top, bot = grid[y], grid[y + 1] if y + 1 < len(grid) else "." * len(grid[y])
        s = ""
        for a, b in zip(top, bot):
            ca, cb = colors.get(a), colors.get(b)
            if ca and cb:
                s += f"\033[38;2;{ca[0]};{ca[1]};{ca[2]};48;2;{cb[0]};{cb[1]};{cb[2]}m▀"
            elif ca:
                s += f"\033[0;38;2;{ca[0]};{ca[1]};{ca[2]}m▀"
            elif cb:
                s += f"\033[0;38;2;{cb[0]};{cb[1]};{cb[2]}m▄"
            else:
                s += "\033[0m "
        lines.append(s + "\033[0m")
    return lines


def progress_bar(pct, width=12):
    full = int(width * pct / 100)
    return "█" * full + "░" * (width - full)


def short_path(p):
    p = str(p)
    return "~" + p[len(str(HOME)):] if p.startswith(str(HOME)) else p


def clip(s, n):
    s = str(s)
    return s if len(s) <= n else s[:n] + f"\n… [truncated, {len(s) - n} more characters]"


def keyname(k):
    if k == "escape enter":
        return "alt+enter"
    if k == "shift enter":
        return "shift+enter"
    return k.replace("c-", "ctrl+") if k else "—"


def parse_key(s):
    s = s.strip().lower().replace(" ", "")
    if s in ("alt+enter", "alt+return", "meta+enter", "esc+enter", "escape+enter"):
        return "escape enter"
    if s in ("shift+enter", "shift+return", "s-enter"):
        return "shift enter"
    s = s.replace("ctrl+", "c-").replace("ctrl-", "c-").replace("control+", "c-")
    if s.startswith("^") and len(s) == 2:
        s = "c-" + s[1]
    return s if re.fullmatch(r"c-[a-z]", s) else None


def key_ok(action, key):
    """Can this action use this key? Returns None if yes, else the reason."""
    if action == "newline" and key in NEWLINE_KEYS:
        return None
    if key in NEWLINE_KEYS:
        return f"{keyname(key)} can only be used for a new line"
    if key in RESERVED_KEYS:
        return f"{keyname(key)} is reserved for {RESERVED_KEYS[key]}"
    return None if re.fullmatch(r"c-[a-z]", key or "") else f"{key} isn't Ctrl + a letter"


def term_width():
    return shutil.get_terminal_size((100, 24)).columns


# ---------------------------------------------------------------- preset tasks

BUILTIN_TASKS = {
    "review": ("Review files or a project for bugs, problems and improvements", False,
               "Review {args}. Read the relevant files carefully but do NOT modify anything. "
               "Report, ordered by importance: bugs and errors, security problems, confusing or "
               "badly written parts, and concrete improvement suggestions. Reference each point "
               "as file:line. End by offering to fix the most important ones."),
    "explain": ("Explain what a file, folder or project does", False,
                "Explain {args}: what it is for, how it is organized, and how the main parts work. "
                "Read the files first. Keep it clear for a non-expert."),
    "summarize": ("Summarize documents or a folder", False,
                  "Read {args} and write a concise summary of the content: key points, important "
                  "details, and anything that needs attention."),
    "fix": ("Find and fix a problem, then verify the fix (autonomous)", True,
            "Find and fix this problem: {args}. Investigate first and find the real cause. "
            "Make the smallest correct change, then verify it works (run the program or its tests "
            "when possible)."),
    "continue": ("Continue a project until it is finished (autonomous)", True,
                 "Continue the project in {args}. Look for notes that describe the goal and state "
                 "(README, TODO, PROGRESS.md, plans, comments), figure out what is left, and keep "
                 "working until it is finished. Keep a PROGRESS.md file in the project up to date "
                 "with what is done and what remains."),
    "research": ("Research a topic online and write a report", False,
                 "Research online: {args}. Run several searches and read multiple good sources "
                 "with fetch_url. Then write a clear, well-structured report with the source URLs. "
                 "If I asked for a file, save the report there."),
    "organize": ("Propose a tidier organization for a folder", False,
                 "Look at {args} and propose a cleaner folder organization (grouping, naming, "
                 "duplicates, junk). Show the plan as a list of moves first and ask for my OK "
                 "before moving anything."),
    "tests": ("Write and run tests for code (autonomous)", True,
              "Write tests for {args}. Use the testing tool the project already uses (or the "
              "standard one for the language), run them, and fix the tests or report real bugs "
              "you find."),
    "docs": ("Write or update a README / documentation", False,
             "Write or update the documentation for {args}: read the code or files first, then "
             "write a README.md explaining what it is, how to install/use it, and examples."),
}
FREE_TEXT_TASKS = {"fix", "research"}   # their argument is text, not a path

TASK_TEMPLATE = """---
description: {desc}
loop: {loop}
---
{body}
"""


def ensure_setup():
    for d in (CONF_DIR, TASKS_DIR, SESS_DIR, WIKI_DIR, TRASH_DIR):
        d.mkdir(parents=True, exist_ok=True)
    ensure_agents()
    if not SYSTEM_FILE.exists():
        SYSTEM_FILE.write_text(SYSTEM_DEFAULT)
    if not CONF_FILE.exists():
        CONF_FILE.write_text(json.dumps(DEFAULTS, indent=2) + "\n")
    for name, (desc, loop, body) in BUILTIN_TASKS.items():
        f = TASKS_DIR / f"{name}.md"
        if not f.exists():
            f.write_text(TASK_TEMPLATE.format(desc=desc, loop=str(loop).lower(), body=body))


def load_config():
    cfg = json.loads(json.dumps(DEFAULTS))
    try:
        data = json.loads(CONF_FILE.read_text())
        keys = data.pop("keys", {})
        data.pop("show_thinking", None)
        cfg.update(data)
        cfg["keys"].update({k: v for k, v in keys.items() if k in ACTIONS})
    except Exception as e:
        print(yellow(f"Could not read {CONF_FILE}: {e} — using defaults"))
    return cfg


def save_config_key(key, value):
    try:
        data = json.loads(CONF_FILE.read_text())
    except Exception:
        data = {}
    data[key] = value
    CONF_FILE.write_text(json.dumps(data, indent=2) + "\n")


def load_tasks():
    tasks = {}
    for f in sorted(TASKS_DIR.glob("*.md")):
        text = f.read_text()
        meta, body = {}, text
        m = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.S)
        if m:
            for line in m.group(1).splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    meta[k.strip()] = v.strip()
            body = m.group(2)
        tasks[f.stem] = {
            "description": meta.get("description", ""),
            "loop": meta.get("loop", "false").lower() == "true",
            "body": body.strip(),
        }
    return tasks


# ---------------------------------------------------------------- clipboard

class Clipboard:
    """Copy via the terminal (OSC 52, works over SSH) and native tools when present."""
    text = ""

    @staticmethod
    def _native(copy):
        if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            return []
        cmds = ([["wl-copy"], ["xclip", "-selection", "clipboard"], ["xsel", "-bi"]] if copy else
                [["wl-paste", "-n"], ["xclip", "-selection", "clipboard", "-o"], ["xsel", "-bo"]])
        return [c for c in cmds if shutil.which(c[0])]

    @classmethod
    def copy(cls, text, output=None):
        cls.text = text
        for cmd in cls._native(True):
            try:
                subprocess.run(cmd, input=text, text=True, timeout=2)
                break
            except Exception:
                pass
        seq = "\033]52;c;" + base64.b64encode(text.encode()).decode() + "\a"
        if output is not None:
            output.write_raw(seq)
            output.flush()
        elif TTY:
            sys.stdout.write(seq)
            sys.stdout.flush()

    @classmethod
    def paste(cls):
        for cmd in cls._native(False):
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=2)
                if r.returncode == 0:
                    return r.stdout
            except Exception:
                pass
        return cls.text


# ---------------------------------------------------------------- system stats

class SysStats:
    def __init__(self):
        self.ram = (0.0, 0.0)
        self.gpu = None
        self._started = False

    def start(self):
        if not self._started:
            self._started = True
            self.refresh()
            threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while True:
            time.sleep(3)
            self.refresh()

    def refresh(self):
        try:
            mem = dict(l.split(":", 1) for l in Path("/proc/meminfo").read_text().splitlines())
            kb = lambda k: int(mem[k].split()[0]) / 1048576
            self.ram = (kb("MemTotal") - kb("MemAvailable"), kb("MemTotal"))
        except Exception:
            pass
        if shutil.which("nvidia-smi"):
            try:
                line = subprocess.run(
                    ["nvidia-smi", "--query-gpu=temperature.gpu,memory.used,memory.total,utilization.gpu,name",
                     "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3).stdout.splitlines()[0]
                t, used, total, util, name = [x.strip() for x in line.split(",")]
                self.gpu = {"temp": int(t), "used": int(used) / 1024, "total": int(total) / 1024,
                            "util": int(util), "name": name}
            except Exception:
                self.gpu = None


STATS = SysStats()


def machine_bits(agent, full=False):
    bits = []
    if agent:
        n = agent.cfg["num_ctx"]
        used = agent.context_used()
        bits.append(f"ctx {100 * used / n:.0f}%" + (f" ({used / 1000:.1f}k/{n / 1000:.0f}k)" if full else ""))
    used, total = STATS.ram
    if total:
        bits.append(f"RAM {used:.1f}/{total:.0f}GB")
    g = STATS.gpu
    if g:
        bits.append(f"GPU {g['temp']}°C" + (f" VRAM {g['used']:.1f}/{g['total']:.0f}GB" if full else ""))
    if agent:
        bits.append(f"temp {agent.cfg['temperature']}")
    return bits


# ---------------------------------------------------------------- live status line & key listener

class UI:
    """What the AI is doing right now (the prompt area shows it live). All output goes through out()."""
    FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self):
        self.lock = threading.RLock()
        self.label = ""
        self.depth = 0
        self.tok = 0
        self.tok_t0 = None
        self.tok_last = None       # time of the last token: the rate doesn't sink while a model is silent
        self.agent = None
        self.keymap = {}           # "c-b" -> action
        self.indent = ""           # prefix for every line while a helper works

    def out(self, text=""):
        with self.lock:
            if self.indent:
                text = "\n".join(self.indent + l for l in str(text).split("\n"))
            print(text, flush=True)

    def set_label(self, label):
        self.label = label

    def tick(self, n=1):
        now = time.time()
        if self.tok_t0 is None:
            self.tok_t0 = now
        self.tok += n
        self.tok_last = now

    def reset_rate(self):
        self.tok, self.tok_t0, self.tok_last = 0, None, None

    def rate(self):
        """Tokens per second up to the last token (a pause, e.g. a tool call Ollama sends in one piece, isn't slowness)."""
        if not (self.tok > 1 and self.tok_t0):
            return 0.0
        return (self.tok - 1) / max(self.tok_last - self.tok_t0, 0.2)

    @contextmanager
    def busy(self, label):
        prev = self.label
        self.label = label
        self.depth += 1
        try:
            yield
        finally:
            self.depth -= 1
            self.label = prev


ui = UI()
out = ui.out


# ---------------------------------------------------------------- streaming text printer

EMOJI_RE = re.compile("[\U0001F000-\U0001FAFF\u2600-\u26FF\u2705\u274C\u274E\u2753-\u2757\u2B50\u2B55\uFE0F\u200D]")


def no_emoji(text):
    """Remove emoji (pictographs) from text shown on screen; keeps plain symbols like ✓ • ❯."""
    return re.sub(r"  +", " ", EMOJI_RE.sub("", text)) if EMOJI_RE.search(text) else text


def render_md(line, st):
    s = line.strip()
    if s.startswith("```"):
        st["code"] = not st["code"]
        return dim(line)
    if st["code"]:
        return cyan(line)
    m = re.match(r"^(#{1,6})\s+(.*)", line)
    if m:
        return c("1;4", m.group(2)) if len(m.group(1)) <= 2 else bold(m.group(2))
    if re.match(r"^\s*([-*_])(\s*\1){2,}\s*$", line):
        return dim("─" * 40)
    line = re.sub(r"^(\s*)[-*] ", r"\1• ", line)
    line = re.sub(r"\*\*(.+?)\*\*", lambda m: bold(m.group(1)), line)
    line = re.sub(r"`([^`]+)`", lambda m: cyan(m.group(1)), line)
    return line


class StreamPrinter:
    """Prints streamed text line by line (word-wrapped) above the live status line."""

    def __init__(self, kind="text", color=None):
        self.kind, self.buf, self.st, self.started = kind, "", {"code": False}, False
        self.color = color

    def feed(self, s):
        self.buf += s
        while True:
            if "\n" in self.buf:
                line, self.buf = self.buf.split("\n", 1)
                self.emit(line)
                continue
            w = term_width() - (5 if self.kind == "think" else 2)
            if len(self.buf) > w and not self.st["code"]:
                cut = self.buf.rfind(" ", 0, w)
                cut = cut if cut > w // 3 else w
                line, self.buf = self.buf[:cut], self.buf[cut:].lstrip(" ")
                self.emit(line)
                continue
            break

    def flush(self):
        if self.buf:
            self.emit(self.buf)
            self.buf = ""

    def emit(self, line):
        line = no_emoji(line)
        if not self.started and not line.strip():
            return
        self.started = True
        if self.kind == "think":
            out((hexc(self.color, "  ┊ ") if self.color else dim("  ┊ ")) + dim(line))
        else:
            out(render_md(line, self.st))


# ---------------------------------------------------------------- web helpers

class _TextExtractor(HTMLParser):
    SKIP = {"script", "style", "noscript", "svg", "head", "nav", "footer", "form"}
    BLOCK = {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "section",
             "article", "pre", "blockquote", "table"}

    def __init__(self):
        super().__init__()
        self.parts, self.skip, self.title, self._in_title = [], 0, "", False

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip += 1
        if tag == "title":
            self._in_title = True
        if tag in self.BLOCK:
            self.parts.append("\n")
        if tag in ("h1", "h2", "h3"):
            self.parts.append("## ")
        if tag == "li":
            self.parts.append("- ")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self.skip:
            self.skip -= 1
        if tag == "title":
            self._in_title = False
        if tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self.skip:
            self.parts.append(data)


def html_to_text(src):
    p = _TextExtractor()
    try:
        p.feed(src)
    except Exception:
        pass
    text = "".join(p.parts)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return p.title.strip(), text.strip()


def ddg_search(query, max_results=8):
    r = requests.post("https://html.duckduckgo.com/html/", data={"q": query},
                      headers={"User-Agent": UA}, timeout=20)
    r.raise_for_status()
    results = []
    blocks = re.findall(
        r'class="result__a" href="(.*?)">(.*?)</a>.*?class="result__snippet"[^>]*>(.*?)</a>',
        r.text, re.S)
    for href, title, snippet in blocks:
        if "duckduckgo.com/y.js" in href:      # ads
            continue
        if "uddg=" in href:
            href = urllib.parse.unquote(re.search(r"uddg=([^&]+)", href).group(1))
        strip = lambda s: html.unescape(re.sub(r"<.*?>", "", s)).strip()
        results.append((strip(title), href, strip(snippet)))
        if len(results) >= max_results:
            break
    return results


# ---------------------------------------------------------------- tool schemas

def _fn(name, desc, props, required=()):
    return {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": props, "required": list(required)}}}


S = lambda d: {"type": "string", "description": d}
I = lambda d: {"type": "integer", "description": d}
B = lambda d: {"type": "boolean", "description": d}

TOOLS = [
    _fn("list_dir", "List files and folders in a directory.",
        {"path": S("Directory (default: working directory)"), "depth": I("How many levels deep, 1-3 (default 1)")}),
    _fn("read_file", f"Read a text file (also PDF). Returns numbered lines, max {MAX_READ_LINES} lines per call.",
        {"path": S("File path"), "start_line": I("First line (default 1)"), "end_line": I("Last line")}, ["path"]),
    _fn("write_file", "Create a file or completely replace its content.",
        {"path": S("File path"), "content": S("Full new content")}, ["path", "content"]),
    _fn("edit_file", "Replace an exact piece of text in a file: old_text is REMOVED and new_text is put in its place. old_text must match exactly (including indentation) and be unique unless replace_all is true. To insert lines, put the anchor line in old_text and the anchor + new lines in new_text; never repeat lines that come after old_text.",
        {"path": S("File path"), "old_text": S("Exact text to find"), "new_text": S("Replacement text"),
         "replace_all": B("Replace every occurrence")}, ["path", "old_text", "new_text"]),
    _fn("find_files", "Find files by name pattern (glob, e.g. '*.py' or 'report*'), searching subfolders.",
        {"pattern": S("Glob pattern"), "path": S("Folder to search (default: working directory)")}, ["pattern"]),
    _fn("search_in_files", "Search text inside files (regular expression, case-insensitive). Returns file:line: text.",
        {"pattern": S("Regex or plain text"), "path": S("Folder or file (default: working directory)"),
         "file_glob": S("Only files matching this glob, e.g. '*.md'")}, ["pattern"]),
    _fn("make_dir", "Create a directory (and parents).", {"path": S("Directory path")}, ["path"]),
    _fn("move_path", "Move or rename a file or folder.", {"src": S("Source"), "dst": S("Destination")}, ["src", "dst"]),
    _fn("delete_path", "Delete a file or folder (it is moved to the agent's trash, recoverable).",
        {"path": S("Path to delete")}, ["path"]),
    _fn("run_command", "Run a bash command in the working directory and return its output. Non-interactive only.",
        {"command": S("The command"), "timeout": I("Seconds before it is killed")}, ["command"]),
    _fn("web_search", "Search the internet (DuckDuckGo). Returns titles, URLs and snippets.",
        {"query": S("Search query"), "max_results": I("Default 8")}, ["query"]),
    _fn("fetch_url", "Download a web page (or PDF) and return its readable text.",
        {"url": S("URL"), "offset": I("Character offset to continue a long page")}, ["url"]),
    _fn("search_knowledge", "Search Chicken's knowledge wikis (programming, design, math, energy, physics, security…): "
        "returns the most relevant sections with their page and sources. Reference material: the user's files and real "
        "command output take precedence.",
        {"query": S("What you need to know, in English, with key terms"),
         "wiki": S("Optional: limit to one wiki, e.g. 'frontend', 'energy'")}, ["query"]),
    _fn("update_plan", "Write or update your step-by-step plan (a checklist with [x] for done steps). It stays visible to you even when old messages are compacted.",
        {"plan": S("The full plan as a markdown checklist")}, ["plan"]),
]
TOOLS_CHARS = len(json.dumps(TOOLS))

TASK_COMPLETE = _fn("task_complete", "Call ONLY when the goal is fully finished and verified, or impossible to continue.",
                    {"summary": S("What was done and the final state"),
                     "status": {"type": "string", "enum": ["done", "blocked"]}}, ["summary"])

WRITE_TOOLS = {"write_file", "edit_file", "make_dir"}
RISKY_TOOLS = WRITE_TOOLS | {"move_path", "delete_path", "run_command"}


def describe(name, a):
    """Human labels for an action: (while doing, when done)."""
    p = _brief(a.get("path", "."), 60)
    if name == "read_file":
        rng = f" (lines {a.get('start_line')}-{a.get('end_line', '…')})" if a.get("start_line", 1) not in (1, None) else ""
        return f"Reading {p}{rng}", f"Read {p}{rng}"
    if name == "list_dir":
        return f"Listing {p}", f"Listed {p}"
    if name == "find_files":
        return f"Looking for files '{a.get('pattern', '')}'", f"Searched for files '{a.get('pattern', '')}' in {p}"
    if name == "search_in_files":
        q = _brief(a.get("pattern", ""), 40)
        return f"Searching for '{q}' in {p}", f"Searched for '{q}' in {p}"
    if name == "write_file":
        return f"Writing {p}", f"Write {p}"
    if name == "edit_file":
        return f"Editing {p}", f"Edit {p}"
    if name == "make_dir":
        return f"Creating folder {p}", f"Create folder {p}"
    if name == "move_path":
        s, d = _brief(a.get("src", ""), 40), _brief(a.get("dst", ""), 40)
        return f"Moving {s} → {d}", f"Move {s} → {d}"
    if name == "delete_path":
        return f"Deleting {p}", f"Delete {p}"
    if name == "run_command":
        cmd = _brief(a.get("command", ""), 70)
        return f"Running {cmd}", f"Run {cmd}"
    if name == "web_search":
        q = _brief(a.get("query", ""), 60)
        return f'Searching the web for "{q}"', f'Searched the web for "{q}"'
    if name == "fetch_url":
        u = a.get("url", "")
        return f"Reading {urllib.parse.urlparse(u).netloc or u}", f"Read {_brief(u, 70)}"
    if name == "update_plan":
        return "Updating the plan", "Plan"
    if name == "search_knowledge":
        q = _brief(a.get("query", ""), 50)
        return f"Searching the knowledge wikis for '{q}'", f"Searched the knowledge wikis for '{q}'"
    if name == "task_complete":
        return "Finishing", "Finished"
    if name == "call_agent":
        return f"Handing a step to {a.get('role', '?')}", f"Call {a.get('role', '?')}"
    return f"Running {name}", name


def summarize_result(name, result):
    if result.startswith("Error"):
        return None
    first = result.splitlines()[0] if result else ""
    n = result.count("\n")
    if name == "read_file":
        m = re.search(r"\((lines .*?)\)", first)
        return m.group(1) if m else ""
    if name == "list_dir":
        return f"{n} entries"
    if name in ("find_files", "search_in_files"):
        return first if first.startswith("No ") else f"{n + 1} result{'s' if n else ''}"
    if name == "web_search":
        hits = re.findall(r"(?m)^\d+\. ", result)
        return f"{len(hits)} results"
    if name == "fetch_url":
        return f"{len(result):,} characters"
    if name == "search_knowledge":
        found = re.findall(r"(?m)^### (\S+)", result)
        return ", ".join(found[:3]) if found else "nothing found"
    return ""


# ---------------------------------------------------------------- commands (for / menu and /help)

CMDS = [
    ("/help", "", "Browse all commands and shortcuts",
     "Opens this browser. ↑↓ to move, type to filter, Enter puts the command in your prompt, Esc closes.", "/help"),
    ("/file", "<path> [...]", "Load files or folders to work on",
     "Reads the files and sends their content to the AI with your next message, so it knows exactly what "
     "you mean. For folders it sends the structure. Tab completes paths.", "/file notes.md report.pdf"),
    ("/files", "", "Show the loaded files", "Lists the files and folders you loaded with /file.", "/files"),
    ("/drop", "[path]", "Stop working on a file (or all)",
     "Removes a file from the working list. Without a path, removes all of them.", "/drop notes.md"),
    ("/search", "<query>", "Search the web and answer with sources",
     "Searches DuckDuckGo, reads the best pages and answers with the links. You can also just ask "
     "normally, e.g. \"look online for…\".", "/search how to back up my home folder on linux"),
    ("/loop", "<goal> [--max N]", "Work autonomously until the goal is done",
     "The AI makes a plan, then works step by step (reading, editing, running commands, searching the web), "
     "checks its own work and keeps going until it's finished or blocked. It asks once whether to "
     "auto-approve file edits. '/loop continue' resumes the last goal. Ctrl+C stops it.",
     "/loop write a script that sorts ~/Downloads into folders by file type"),
    ("/mode", "[ask|auto|orchestrator|bonsai-only|bonsai-workers|classic]", "Confirmations and who does the work",
     "ask: confirm each worker and each action; auto: no confirmations. orchestrator: the small main model decides and "
     "workers do the work (default); bonsai-only: Bonsai everywhere; bonsai-workers: every worker on Bonsai; classic: "
     "the main model works itself with all tools (the old Chicken). Without an argument it shows the current mode.",
     "/mode auto"),
    ("/knowledge", "[question|auto on|off]", "Chicken's knowledge wikis",
     "Lists the wikis in ~/.agent/Built_skills_and_knowledge (programming, design, math, energy, physics, security…), "
     "or searches them. The most relevant sections are added to every worker's task automatically, with their "
     "sources; '/knowledge auto off' stops that. Tool workers can also search with search_knowledge.",
     "/knowledge css grid layout"),
    ("/allocation", "", "What uses the GPU and RAM right now",
     "Shows GPU memory by model (weights and context), other programs, the 90% limit and what's free, then RAM: "
     "total, available, file cache, and what Chicken, Ollama and the llama.cpp servers use.", "/allocation"),
    ("/btw", "<question>", "Ask a side question while the AI works",
     "Answered right away by a small read-only model on the GPU (qwen3.5:0.8b, setting btw_model), from the task state and the recent "
     "conversation, without interrupting or changing the main work (no tools, not added to the conversation). "
     "Set btw_model to \"\" in ~/.agent/config.json to use the main model instead.",
     "/btw what does the -r flag of cp do?"),
    ("/queue", "[clear]", "Show or clear queued messages",
     "Messages and commands you send while the AI is working wait in a queue and run one after another. "
     "↑ on an empty line takes the last queued one back to edit it. '/queue clear' removes them all.",
     "/queue clear"),
    ("/plan", "", "Show the AI's current plan", "Shows the checklist the AI is following ([x] = done).", "/plan"),
    ("/tasks", "", "List the preset tasks",
     "Preset tasks are ready-made jobs, and each one is also a command (e.g. /review). Add your own by "
     "creating a .md file in ~/.agent/tasks/.", "/tasks"),
    ("/cd", "<folder>", "Change the working folder",
     "The AI works relative to this folder. It can never go outside your home folder.", "/cd ~/Documents"),
    ("/model", "[name]", "Choose the main agent's model",
     "Without a name: a list of every model (Ollama models and local servers such as Bonsai 2), with its GPU "
     "size and what it can do (tools, thinking, vision). With a name: switches directly. Saved as default. "
     "Helpers set to 'boss' follow this choice. bonsai2:27b and qwen3:14b both passed the tool tests here; "
     "qwen2.5-coder:14b can't use tools, and the 27B Ollama models don't fit in the GPU.", "/model bonsai2:27b"),
    ("/think", "[on|off]", "Reasoning on/off",
     "On: the AI thinks before answering — smarter but slower. Off: much faster, good for simple jobs. "
     "Use the details shortcut (ctrl+b) to read its reasoning.", "/think off"),
    ("/temperature", "[0-2]", "Show or set creativity",
     "Low (0.2) = precise and repeatable, high (1.0) = creative and varied. 0.6 is recommended for qwen3. "
     "Saved as default.", "/temperature 0.6"),
    ("/details", "[on|off]", "Show/hide reasoning and full outputs",
     "Same as the details shortcut (ctrl+b). When on, you see the AI's reasoning as it thinks, full tool "
     "results and complete command output.", "/details on"),
    ("/infocomputer", "", "CPU, GPU, RAM, disk, model and memory status",
     "Shows the processor, RAM, GPU (VRAM, load, temperature), disk space, the loaded model and whether "
     "it runs fully on the GPU, and how full the AI's memory (context) is.", "/infocomputer"),
    ("/context", "", "How full the AI's memory is",
     "The context size follows the GPU memory: alone, the main agent gets everything its model leaves in the "
     "budget; when a helper sits beside it, it shrinks. At ~72% it automatically summarizes older messages.", "/context"),
    ("/compact", "[auto on|off]", "Summarize the conversation to free memory",
     "Replaces older messages with a summary, keeping recent ones. Useful before starting a big job. "
     "'/compact auto off' stops Chicken from compacting on its own to fit a helper beside the main agent "
     "(it ejects and injects instead); '/compact auto on' turns it back on (the default).", "/compact auto off"),
    ("/clear", "", "Start a fresh conversation", "Forgets the conversation, plan and loaded files.", "/clear"),
    ("/copy", "", "Copy the AI's last answer to the clipboard",
     "Copies through your terminal (works over SSH in most terminals: Windows Terminal, MobaXterm, iTerm2, "
     "kitty, WezTerm…). You can also paste it back here with ctrl+v.", "/copy"),
    ("/save", "[-wiki] [name]", "Save as a log, or as a wiki (-wiki)",
     "'/save name' saves the whole conversation as a compressed log (gzip, nothing lost). "
     "'/save -wiki name' turns it into a wiki in ~/.agent/wiki/name/: chat.md (goals, decisions, what's done and "
     "what's next), one .md page per file worked on, and index.md linking them. Saving again to the same wiki "
     "updates it, and unchanged files are skipped. Progress is shown live. Esc stops it.",
     "/save -wiki photos-project"),
    ("/load", "[-wiki] [name]", "Resume a log, or load a wiki's knowledge (-wiki)",
     "'/load name' resumes a saved log; outputs that became outdated are trimmed so it uses less memory. "
     "'/load -wiki name' starts fresh with only the wiki's index and chat summary in memory (a few hundred "
     "tokens); the AI opens the other pages when it needs them. Without a name, a menu lists what's saved. "
     "'/load last' restores the automatic save.", "/load -wiki photos-project"),
    ("/auto", "", "Toggle auto-approve for everything",
     "When on, the AI edits, deletes and runs commands WITHOUT asking. Use with care.", "/auto"),
    ("/setkeyboard", "[action key]", "Change keyboard shortcuts",
     "Opens a table of the shortcuts. ↑↓ picks a row; ←→ picks a box: the key (Enter shows the free "
     "combinations to choose from), 'Push key' (press the new Ctrl+letter) or 'Reset'. The bottom bar has "
     "presets, Reset all, Save and Cancel. Or directly: "
     "'/setkeyboard details ctrl+k'. '/setkeyboard reset' restores the defaults.", "/setkeyboard"),
    ("/color", "[name|#hex]", "Change the colour of the ❯ prompt arrow",
     "Without a name: a list of colours, each shown in its own colour (↑↓ + Enter). Or directly: '/color pink', "
     "'/color #ff8700'. '/color theme' goes back to the theme's colour. Saved as default.", "/color pink"),
    ("/theme", "[name]", "Change the colour theme",
     "Changes every colour of the agent (status line, menus, messages) and the terminal's background and text "
     "colour while the agent runs (restored when you quit). 'terminal' keeps your terminal's own colours. "
     "Themes: " + ", ".join(THEMES) + ". Saved as default.", "/theme dracula"),
    ("/agents", "[mode|resume|vram|list]", "Helpers: models, kinds and when they run",
     "Without arguments: a table of the helpers. ↑↓ picks one, ←→ picks its model or kind, Enter changes it; the "
     "bottom bar sets the mode. 'call -agent <helper> <task>' runs one yourself. Modes: ask (Chicken "
     "proposes each helper and you approve), auto (it calls them on its own), manual (only call -agent). "
     "'/agents vram' shows what's in the GPU, '/agents resume' continues a job stopped during a helper. "
     "Prompts are in ~/.agent/agents/<name>.md.", "/agents mode auto"),
    ("/system", "[edit|reset]", "Show Chicken's system prompt",
     "Shows the exact instructions sent to the main model: the editable part from ~/.agent/system.md, then the parts "
     "added automatically (files you loaded, helpers, plan, wiki, autonomous mode). '/system edit' opens the file in "
     "your editor ($EDITOR, else nano); changes apply to the next message. '/system reset' restores the default. "
     "{date}, {cwd} and {root} are filled in each time.", "/system edit"),
    ("/config", "", "Show all settings", "Settings are stored in ~/.agent/config.json (you can edit it).", "/config"),
    ("/exit", "", "Quit", "You can also press Ctrl+D, or Ctrl+C twice.", "/exit"),
]
CMD_ARGS = {name: args for name, args, *_ in CMDS}


class ToolError(Exception):
    pass


class _ThinkUnsupported(Exception):
    pass


class Interrupted(Exception):
    pass


# ---------------------------------------------------------------- interactive menu (arrows)

MENU_STYLE = None


def menu(title, items, detail=None, numbered=False, filterable=False, height=12, hint=None, styles=None, start=0):
    """items: list of (label, meta); styles: optional style per label. Returns the chosen index, or None on Esc."""
    if not (HAVE_PT and sys.stdin.isatty() and TTY):
        return None
    st = {"i": start, "f": ""}

    def visible():
        f = st["f"].lower()
        return [k for k, (l, m) in enumerate(items) if f in l.lower() or f in m.lower()] if f else list(range(len(items)))

    def render():
        vis = visible()
        st["i"] = min(st["i"], max(len(vis) - 1, 0))
        w = min(max((len(items[k][0]) for k in vis), default=10), 34)
        h = hint or ("↑↓ move · Enter select · Esc close" + (" · type to filter" if filterable else ""))
        lines = [("class:mtitle", f" {title}  "), ("class:mhint", h)]
        if st["f"]:
            lines.append(("class:mfilter", f"   filter: {st['f']}"))
        lines.append(("", "\n"))
        top = max(0, min(st["i"] - height // 2, len(vis) - height))
        for n, k in enumerate(vis[top:top + height], top):
            label, meta = items[k]
            sel = n == st["i"]
            num = f"{n + 1}. " if numbered else ""
            style = (styles[k] + (" bold underline" if sel else "")) if styles else "class:msel" if sel else "class:mitem"
            lines.append(("class:msel" if sel else "", f" {'❯' if sel else ' '} {num}"))
            lines.append((style, f"{label:<{w}}"))
            lines.append(("class:mmeta.sel" if sel else "class:mmeta", f"  {meta}" if meta else ""))
            lines.append(("", "\n"))
        if not vis:
            lines.append(("class:mmeta", "   (no matches)\n"))
        elif len(vis) > height:
            lines.append(("class:mmeta", f"   {st['i'] + 1}/{len(vis)}\n"))
        if detail and vis:
            text = detail(vis[st["i"]])
            if text:
                lines.append(("class:msep", " " + "─" * min(term_width() - 2, 70) + "\n"))
                lines.append(("class:mdetail", "\n".join(" " + l for l in text.splitlines()) + "\n"))
        return lines

    kb = KeyBindings()

    @kb.add("up")
    def _(e):
        n = len(visible())
        if n:
            st["i"] = (st["i"] - 1) % n

    @kb.add("down")
    def _(e):
        n = len(visible())
        if n:
            st["i"] = (st["i"] + 1) % n

    @kb.add("pageup")
    def _(e):
        st["i"] = max(0, st["i"] - height)

    @kb.add("pagedown")
    def _(e):
        st["i"] = min(len(visible()) - 1, st["i"] + height)

    @kb.add("enter")
    def _(e):
        vis = visible()
        e.app.exit(result=vis[st["i"]] if vis else None)

    @kb.add("escape")
    def _(e):
        e.app.exit(result=None)

    @kb.add("c-c")
    def _(e):
        e.app.exit(exception=KeyboardInterrupt())

    if numbered:
        for d in "123456789":
            @kb.add(d)
            def _(e, d=d):
                k = int(d) - 1
                if k < len(items):
                    e.app.exit(result=k)

    if filterable:
        @kb.add("<any>")
        def _(e):
            ch = e.data
            if ch and ch.isprintable():
                st["f"] += ch
                st["i"] = 0

        @kb.add("backspace")
        def _(e):
            st["f"] = st["f"][:-1]
            st["i"] = 0

    app = Application(layout=Layout(Window(FormattedTextControl(render), wrap_lines=True)),
                      key_bindings=kb, full_screen=False, style=MENU_STYLE, erase_when_done=True)
    app.ttimeoutlen = 0.05
    app.timeoutlen = 0.05
    return app.run()


# what the free Ctrl keys normally do while typing (overriding them loses that)
LINE_EDIT_KEYS = {"c-a": "go to line start", "c-b": "cursor left", "c-e": "go to line end", "c-f": "cursor right",
                  "c-k": "delete to line end", "c-u": "delete to line start", "c-w": "delete previous word",
                  "c-y": "paste deleted text", "c-p": "previous message", "c-n": "next message",
                  "c-r": "search history", "c-t": "swap two letters", "c-x": "ctrl+x ctrl+e: edit in editor"}
KEY_CHOICES = ["c-" + l for l in "abcdefghijklmnopqrstuvwxyz" if "c-" + l not in RESERVED_KEYS]
KEY_PRESETS = {
    "Default":          {"details": "c-b", "plan": "c-t", "last_output": "c-o", "info": "c-g", "paste": "c-v",
                         "newline": "escape enter"},
    "Claude Code-like": {"details": "c-o", "plan": "c-t", "last_output": "c-b", "info": "c-g", "paste": "c-v",
                         "newline": "c-j"},
    "Mnemonic":         {"details": "c-b", "plan": "c-p", "last_output": "c-o", "info": "c-g", "paste": "c-v",
                         "newline": "c-n"},
}


def keyboard_editor(keys, focus=None, msg=""):
    """Interactive table of shortcuts. Returns the new {action: key} dict on Save, None on Cancel.
    focus: an action to start on, with its list of key choices already open."""
    if not (HAVE_PT and sys.stdin.isatty() and TTY):
        return None
    acts = list(ACTIONS)
    saved = dict(keys)
    cur = {a: keys.get(a, ACTIONS[a][1]) for a in acts}
    buttons = ["Preset", "Reset all", "Save", "Cancel"]
    row_boxes = ["key", "Push key", "Reset"]
    st = {"row": acts.index(focus) if focus in acts else 0, "col": 0, "mode": "table", "list": [], "li": 0,
          "msg": ("err" if msg else "", msg)}
    # rows 0..n-1 = actions, row n = the button bar

    def preset_name():
        return next((n for n, p in KEY_PRESETS.items() if all(cur[a] == p.get(a) for a in acts)), "Custom")

    def note(key):
        if key in LINE_EDIT_KEYS:
            return f"replaces: {LINE_EDIT_KEYS[key]}"
        return "free"

    def set_key(act, key):
        why = key_ok(act, key)
        if why:
            st["msg"] = ("err", why + " — choose another")
            return
        other = next((x for x in acts if cur[x] == key and x != act), None)
        if other:
            # swap, unless the old key can't be used by the other action: then give it the first free key
            new = cur[act] if key_ok(other, cur[act]) is None else next(
                k for k in KEY_CHOICES if k not in cur.values() and k != key)
            cur[other] = new
            st["msg"] = ("ok", f"{act} → {keyname(key)} · {other} moved to {keyname(new)}")
        else:
            st["msg"] = ("ok", f"{act} → {keyname(key)}" + (f"  ({note(key)})" if key in LINE_EDIT_KEYS else ""))
        cur[act] = key

    def open_list(kind):
        if kind == "preset":
            names = list(KEY_PRESETS)
            items = [(n, " · ".join(f"{a} {keyname(k)}" for a, k in KEY_PRESETS[n].items())) for n in names]
            st.update(mode="list", kind="preset", list=names, items=items,
                      li=names.index(preset_name()) if preset_name() in names else 0)
        else:
            act = acts[st["row"]]
            choices = (list(NEWLINE_KEYS) if act == "newline" else []) + KEY_CHOICES
            items = []
            for k in choices:
                used = next((x for x in acts if cur[x] == k), None)
                meta = ("● current" if used == act else f"used by {used} (will swap)" if used
                        else NEWLINE_KEYS.get(k) or note(k))
                items.append((keyname(k), meta))
            st.update(mode="list", kind="key", list=choices, items=items, li=choices.index(cur[act])
                      if cur[act] in choices else 0)

    def box(text, sel, width=None):
        t = f" {text:<{width}} " if width else f" {text} "
        return ("class:kb.box.sel" if sel else "class:kb.box", f"[{t}]")

    def render():
        W = max(min(term_width() - 2, 110), 40)
        wa = max(len(a) for a in acts) + 2
        wk = 40
        wd = max(W - wa - wk - 4, 0)
        show_desc = wd >= 16
        B = "class:kb.border"
        P = [("class:mtitle", " Keyboard shortcuts"), ("class:mhint", f"   preset: {preset_name()}\n")]

        def hline(l, m, r):
            P.append((B, " " + l + "─" * wa + m + "─" * wk + ((m + "─" * wd) if show_desc else "") + r + "\n"))

        hline("╭", "┬", "╮")
        P += [(B, " │"), ("class:kb.head", f" {'Action':<{wa - 1}}"), (B, "│"), ("class:kb.head", f" {'Shortcut':<{wk - 1}}")]
        if show_desc:
            P += [(B, "│"), ("class:kb.head", f" {'What it does':<{wd - 1}}")]
        P.append((B, "│\n"))
        hline("├", "┼", "┤")
        for i, act in enumerate(acts):
            sel = st["row"] == i
            changed = cur[act] != saved.get(act)
            P += [(B, " │"), ("class:msel" if sel else "class:mitem",
                             f"{'❯' if sel else ' '}{act:<{wa - 2}}{'•' if changed else ' '}"), (B, "│ ")]
            used = 1
            if sel and st["mode"] == "capture":
                txt = ("⌨ press the new key now · Esc cancel" if acts[i] == "newline"
                       else "⌨ press Ctrl+letter now · Esc cancel")
                P.append(("class:kb.capture", txt))
                used += len(txt)
            elif sel:
                for j, name in enumerate(row_boxes):
                    label = f"{keyname(cur[act]):<7} ▾" if name == "key" else name
                    b = box(label, st["mode"] == "table" and st["col"] == j)
                    P += [b, ("", " ")]
                    used += len(b[1]) + 1
            else:
                k = keyname(cur[act])
                P.append(("class:kb.changed" if changed else "class:kb.key", f"  {k}"))
                used += len(k) + 2
            P.append(("", " " * max(wk - used, 0)))
            if show_desc:
                d = ACTIONS[act][0]
                d = d if len(d) <= wd - 2 else d[:wd - 3] + "…"
                P += [(B, "│"), ("class:mmeta", f" {d:<{wd - 1}}")]
            P.append((B, "│\n"))
        hline("╰", "┴", "╯")
        P.append(("", "  "))
        on_bar = st["mode"] == "table" and st["row"] == len(acts)
        for j, name in enumerate(buttons):
            label = f"Preset: {preset_name()} ▾" if name == "Preset" else name
            P += [box(label, on_bar and st["col"] == j), ("", "  ")]
        P.append(("", "\n"))
        if st["mode"] == "list":
            title = "Choose a preset" if st["kind"] == "preset" else f"Choose a shortcut for {acts[st['row']]}"
            P.append(("class:mtitle", f"  {title}"))
            P.append(("class:mhint", "   ↑↓ move · Enter choose · Esc back\n"))
            h = 8
            top = max(0, min(st["li"] - h // 2, len(st["items"]) - h))
            wl = max(len(l) for l, _ in st["items"])
            for n, (label, meta) in enumerate(st["items"][top:top + h], top):
                s = n == st["li"]
                P += [("class:msel" if s else "class:mitem", f"   {'❯' if s else ' '} {label:<{wl}}"),
                      ("class:mmeta.sel" if s else "class:mmeta", f"  {meta}\n")]
            if len(st["items"]) > h:
                P.append(("class:mmeta", f"     {st['li'] + 1}/{len(st['items'])}\n"))
        kind, msg = st["msg"]
        if msg:
            P.append(("class:kb.err" if kind == "err" else "class:kb.ok", f"  {msg}\n"))
        if st["mode"] == "table":
            P.append(("class:mhint", "  ↑↓ row · ←→ box · Enter use it · r reset row · s save · Esc cancel\n"))
        return P

    if focus in acts:
        open_list("key")

    kb = KeyBindings()
    table = Condition(lambda: st["mode"] == "table")
    lst = Condition(lambda: st["mode"] == "list")
    capture = Condition(lambda: st["mode"] == "capture")

    def ncols():
        return len(buttons) if st["row"] == len(acts) else len(row_boxes)

    def move(step):
        was_bar = st["row"] == len(acts)
        st["row"] = (st["row"] + step) % (len(acts) + 1)
        if was_bar != (st["row"] == len(acts)):
            st["col"] = 0          # crossing between the table and the button bar starts at the first box

    @kb.add("up", filter=table)
    def _(e):
        move(-1)

    @kb.add("down", filter=table)
    def _(e):
        move(1)

    @kb.add("left", filter=table)
    @kb.add("s-tab", filter=table)
    def _(e):
        st["col"] = (st["col"] - 1) % ncols()

    @kb.add("right", filter=table)
    @kb.add("tab", filter=table)
    def _(e):
        st["col"] = (st["col"] + 1) % ncols()

    @kb.add("r", filter=table)
    def _(e):
        if st["row"] < len(acts):
            set_key(acts[st["row"]], ACTIONS[acts[st["row"]]][1])

    @kb.add("s", filter=table)
    def _(e):
        e.app.exit(result=dict(cur))

    @kb.add("enter", filter=table)
    def _(e):
        st["msg"] = ("", "")
        if st["row"] == len(acts):
            name = buttons[st["col"]]
            if name == "Preset":
                open_list("preset")
            elif name == "Reset all":
                cur.update({a: ACTIONS[a][1] for a in acts})
                st["msg"] = ("ok", "All shortcuts back to the defaults (Save to keep them)")
            elif name == "Save":
                e.app.exit(result=dict(cur))
            else:
                e.app.exit(result=None)
            return
        name = row_boxes[st["col"]]
        if name == "key":
            open_list("key")
        elif name == "Push key":
            st["mode"] = "capture"
        else:
            act = acts[st["row"]]
            set_key(act, ACTIONS[act][1])

    @kb.add("up", filter=lst)
    def _(e):
        st["li"] = (st["li"] - 1) % len(st["list"])

    @kb.add("down", filter=lst)
    def _(e):
        st["li"] = (st["li"] + 1) % len(st["list"])

    @kb.add("enter", filter=lst)
    def _(e):
        choice = st["list"][st["li"]]
        if st["kind"] == "preset":
            cur.update(KEY_PRESETS[choice])
            st["msg"] = ("ok", f"Preset '{choice}' applied (Save to keep it)")
        else:
            set_key(acts[st["row"]], choice)
        st["mode"] = "table"

    @kb.add("escape", filter=lst)
    def _(e):
        st["mode"] = "table"

    @kb.add("escape", "enter", filter=capture)
    def _(e):
        st["mode"] = "table"
        set_key(acts[st["row"]], "escape enter")

    @kb.add(Keys.F24, filter=capture)
    def _(e):
        st["mode"] = "table"
        set_key(acts[st["row"]], "shift enter")

    @kb.add("enter", filter=capture)
    def _(e):
        st["mode"] = "table"
        st["msg"] = ("err", "Your terminal sends a plain Enter for Shift+Enter. Use alt+enter or ctrl+j "
                            "(README: Shift+Enter)")

    @kb.add("<any>", filter=capture)
    def _(e):
        k = getattr(e.key_sequence[0].key, "value", str(e.key_sequence[0].key))
        st["mode"] = "table"
        if k == "escape":
            st["msg"] = ("", "")
        elif re.fullmatch(r"c-[a-z]", k):
            set_key(acts[st["row"]], k)
        else:
            st["msg"] = ("err", "That's not Ctrl + a letter — press Enter on 'Push key' to try again")

    @kb.add("escape", filter=table)
    def _(e):
        e.app.exit(result=None)

    @kb.add("c-c", filter=~capture)
    def _(e):
        e.app.exit(result=None)

    app = Application(layout=Layout(Window(FormattedTextControl(render), wrap_lines=False)),
                      key_bindings=kb, full_screen=False, style=MENU_STYLE, erase_when_done=True)
    app.ttimeoutlen = 0.05
    app.timeoutlen = 0.05
    return app.run()


def agents_editor(roles, mode, models):
    """Interactive table of helpers. models: [(name, meta, capabilities)].
    Returns ({role: {"model": .., "kind": ..}}, mode) on Save, None on Cancel."""
    if not (HAVE_PT and sys.stdin.isatty() and TTY):
        return None
    names = list(roles)
    cur = {n: {"model": roles[n]["model"], "kind": roles[n]["kind"]} for n in names}
    st = {"row": 0, "col": 0, "mode": "table", "items": [], "values": [], "li": 0, "kind": "", "msg": ("", ""),
          "agents_mode": mode}
    buttons = ["Mode", "Save", "Cancel"]
    caps = {m: cp for m, _, cp in models}
    sizes = {m: float(re.sub(r"[^\d.]", "", meta.split()[0]) or 0) for m, meta, _ in models}
    boss_caps = set()

    def check(n):
        r = cur[n]
        m = r["model"]
        if sizes.get(m, 0) > 11.5:
            return f"{m} needs ~{sizes[m]:.0f} GB, more than the 12 GB GPU: it runs partly on the CPU, much slower"
        if r["kind"] != "tool" or m == "boss":
            return ""
        if TOOLS_TESTED.get(m) is False:
            return f"{m} failed the tool test here — set {n} to writer"
        if "tools" not in caps.get(m, {"tools"}):
            return f"{m} can't use tools — set {n} to writer"
        return ""

    def open_list(kind):
        n = names[st["row"]] if st["row"] < len(names) else None
        if kind == "model":
            items = [("boss", "same model as Chicken · no GPU swap")] + [(m, meta) for m, meta, _ in models]
            vals = [x[0] for x in items]
            sel = cur[n]["model"]
        elif kind == "kind":
            items = [("tool", "uses tools itself (read, search, edit…) · needs a model that can"),
                     ("writer", "the program gives it the files and applies its answer · any model")]
            vals = ["tool", "writer"]
            sel = cur[n]["kind"]
        else:
            items = list(AGENT_MODES.items())
            vals = list(AGENT_MODES)
            sel = st["agents_mode"]
        st.update(mode="list", kind=kind, items=items, values=vals, li=vals.index(sel) if sel in vals else 0)

    def box(text, sel):
        return ("class:kb.box.sel" if sel else "class:kb.box", f"[ {text} ]")

    def render():
        W = max(min(term_width() - 2, 120), 50)
        wr = max(len(n) for n in names) + 3
        wm, wk = 24, 13
        wd = max(W - wr - wm - wk - 5, 0)
        show_desc = wd >= 16
        B = "class:kb.border"
        P = [("class:mtitle", " Helpers"), ("class:mhint", f"   mode: {st['agents_mode']} — {AGENT_MODES[st['agents_mode']]}\n")]

        def hline(l, m, r):
            P.append((B, " " + l + "─" * wr + m + "─" * wm + m + "─" * wk + ((m + "─" * wd) if show_desc else "") + r + "\n"))

        hline("╭", "┬", "╮")
        P += [(B, " │"), ("class:kb.head", f" {'Helper':<{wr - 1}}"), (B, "│"), ("class:kb.head", f" {'Model':<{wm - 1}}"),
              (B, "│"), ("class:kb.head", f" {'Kind':<{wk - 1}}")]
        if show_desc:
            P += [(B, "│"), ("class:kb.head", f" {'What it does':<{wd - 1}}")]
        P.append((B, "│\n"))
        hline("├", "┼", "┤")
        for i, n in enumerate(names):
            sel = st["row"] == i
            changed = cur[n] != {"model": roles[n]["model"], "kind": roles[n]["kind"]}
            warn = check(n)
            P += [(B, " │"), ("class:msel" if sel else "class:mitem",
                             f"{'❯' if sel else ' '}{n:<{wr - 3}}{'•' if changed else ' '}{'!' if warn else ' '}"), (B, "│")]
            for j, (key, w) in enumerate((("model", wm), ("kind", wk))):
                val = cur[n][key]
                if sel:
                    t = f"{val if len(val) <= w - 7 else val[:w - 8] + '…'} ▾"
                    b = box(t, st["mode"] == "table" and st["col"] == j)
                    P += [b, ("", " " * max(w - len(b[1]), 0))]
                else:
                    P.append(("class:kb.changed" if changed else "class:kb.key", f" {_brief(val, w - 2):<{w - 1}}"))
                P.append((B, "│"))
            if show_desc:
                d = roles[n]["description"]
                d = d if len(d) <= wd - 2 else d[:wd - 3] + "…"
                P += [("class:mmeta", f" {d:<{wd - 1}}"), (B, "│")]
            P.append(("", "\n"))
        hline("╰", "┴", "╯")
        on_bar = st["mode"] == "table" and st["row"] == len(names)
        P.append(("", "  "))
        for j, b in enumerate(buttons):
            label = f"Mode: {st['agents_mode']} ▾" if b == "Mode" else b
            P += [box(label, on_bar and st["col"] == j), ("", "  ")]
        P.append(("", "\n"))
        if st["mode"] == "list":
            title = {"model": f"Model for {names[st['row']]}" if st["row"] < len(names) else "",
                     "kind": f"Kind for {names[st['row']]}" if st["row"] < len(names) else "",
                     "mode": "When helpers run"}[st["kind"]]
            P += [("class:mtitle", f"  {title}"), ("class:mhint", "   ↑↓ move · Enter choose · Esc back\n")]
            h = 9
            top = max(0, min(st["li"] - h // 2, len(st["items"]) - h))
            wl = max(len(l) for l, _ in st["items"])
            for k, (label, meta) in enumerate(st["items"][top:top + h], top):
                s = k == st["li"]
                P += [("class:msel" if s else "class:mitem", f"   {'❯' if s else ' '} {label:<{wl}}"),
                      ("class:mmeta.sel" if s else "class:mmeta", f"  {meta}\n")]
            if len(st["items"]) > h:
                P.append(("class:mmeta", f"     {st['li'] + 1}/{len(st['items'])}\n"))
        kind, msg = st["msg"]
        if not msg and st["row"] < len(names):
            w = check(names[st["row"]])
            kind, msg = ("err", "! " + w) if w else ("", "")
        if msg:
            P.append(("class:kb.err" if kind == "err" else "class:kb.ok", f"  {msg}\n"))
        if st["mode"] == "table":
            P.append(("class:mhint", "  ↑↓ row · ←→ box · Enter change · s save · Esc cancel · "
                                     "prompts: ~/.agent/agents/<name>.md\n"))
        return P

    kb = KeyBindings()
    table = Condition(lambda: st["mode"] == "table")
    lst = Condition(lambda: st["mode"] == "list")

    def ncols():
        return len(buttons) if st["row"] == len(names) else 2

    def move(step):
        was_bar = st["row"] == len(names)
        st["row"] = (st["row"] + step) % (len(names) + 1)
        st["msg"] = ("", "")
        if was_bar != (st["row"] == len(names)):
            st["col"] = 0

    @kb.add("up", filter=table)
    def _(e):
        move(-1)

    @kb.add("down", filter=table)
    def _(e):
        move(1)

    @kb.add("left", filter=table)
    @kb.add("s-tab", filter=table)
    def _(e):
        st["col"] = (st["col"] - 1) % ncols()

    @kb.add("right", filter=table)
    @kb.add("tab", filter=table)
    def _(e):
        st["col"] = (st["col"] + 1) % ncols()

    @kb.add("s", filter=table)
    def _(e):
        e.app.exit(result=(cur, st["agents_mode"]))

    @kb.add("enter", filter=table)
    def _(e):
        st["msg"] = ("", "")
        if st["row"] == len(names):
            b = buttons[st["col"]]
            if b == "Mode":
                open_list("mode")
            else:
                e.app.exit(result=(cur, st["agents_mode"]) if b == "Save" else None)
            return
        open_list("model" if st["col"] == 0 else "kind")

    @kb.add("up", filter=lst)
    def _(e):
        st["li"] = (st["li"] - 1) % len(st["values"])

    @kb.add("down", filter=lst)
    def _(e):
        st["li"] = (st["li"] + 1) % len(st["values"])

    @kb.add("enter", filter=lst)
    def _(e):
        v = st["values"][st["li"]]
        if st["kind"] == "mode":
            st["agents_mode"] = v
            st["msg"] = ("ok", f"Mode: {v} — {AGENT_MODES[v]} (Save to keep it)")
        else:
            n = names[st["row"]]
            cur[n][st["kind"]] = v
            if st["kind"] == "model" and v != "boss" and "tools" not in caps.get(v, set()):
                cur[n]["kind"] = "writer"
                st["msg"] = ("ok", f"{n} → {v} (set to writer: this model can't use tools)")
            else:
                st["msg"] = ("ok", f"{n} → {v}")
        st["mode"] = "table"

    @kb.add("escape", filter=lst)
    def _(e):
        st["mode"] = "table"

    @kb.add("escape", filter=table)
    @kb.add("c-c")
    def _(e):
        e.app.exit(result=None)

    app = Application(layout=Layout(Window(FormattedTextControl(render), wrap_lines=False)),
                      key_bindings=kb, full_screen=False, style=MENU_STYLE, erase_when_done=True)
    app.ttimeoutlen = 0.05
    app.timeoutlen = 0.05
    return app.run()


def ask_text(question):
    if HAVE_PT and sys.stdin.isatty():
        from prompt_toolkit import prompt as pt_prompt
        return pt_prompt(FormattedText([("class:mhint", question)]), style=MENU_STYLE)
    return input(question)


# ---------------------------------------------------------------- the agent

# ---------------------------------------------------------------- saved logs & wiki

ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
READ_TOOLS = {"read_file", "list_dir", "find_files", "search_in_files", "fetch_url", "web_search"}
CHANGE_TOOLS = {"write_file", "edit_file", "move_path", "delete_path"}
WIKI_FLAGS = {"-wiki", "--wiki", "-w"}


def session_files():
    """name -> path of saved conversation logs (gzipped .json.gz, or older plain .json)."""
    found = {}
    for p in SESS_DIR.iterdir():
        if p.name.endswith(".json.gz"):
            name = p.name[:-8]
        elif p.suffix == ".json":
            name = p.stem
        else:
            continue
        if name not in found or p.stat().st_mtime > found[name].stat().st_mtime:
            found[name] = p
    return found


def wiki_names():
    return sorted(p.name for p in WIKI_DIR.iterdir() if (p / "index.md").exists()) if WIKI_DIR.exists() else []


def write_log(path, data, progress=None):
    """Write a conversation as gzipped JSON (lossless). Returns (raw bytes, bytes on disk)."""
    raw = json.dumps(data, ensure_ascii=False).encode()
    tmp = path.with_name(path.name + ".tmp")
    step = max(len(raw) // 20, 1 << 15)
    with gzip.open(tmp, "wb", compresslevel=6) as f:
        for i in range(0, len(raw), step):
            f.write(raw[i:i + step])
            if progress:
                progress(100 * min(i + step, len(raw)) / len(raw))
    tmp.replace(path)
    return len(raw), path.stat().st_size


def read_log(path):
    data = path.read_bytes()
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    return json.loads(data)


def messages_chars(messages):
    return sum(len(m.get("content") or "") + len(json.dumps(m.get("tool_calls", ""))) for m in messages)


def _args(fn):
    args = fn.get("arguments") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            args = {}
    return args if isinstance(args, dict) else {}


def _tool_results(messages):
    """index of each tool result -> (tool name, arguments) of the call that produced it."""
    pairs, pending = {}, []
    for i, m in enumerate(messages):
        if m["role"] == "assistant":
            pending = [tc.get("function", {}) for tc in m.get("tool_calls") or []]
        elif m["role"] == "tool":
            fn = pending.pop(0) if pending else {"name": m.get("tool_name", "")}
            pairs[i] = (fn.get("name", ""), _args(fn))
    return pairs


def prune_messages(messages, keep_recent=8):
    """Context compression for a saved conversation, without losing anything the AI can't get back:
    - outputs superseded later (a file read again or changed afterwards, the same search/listing/page
      fetched again) are replaced by a one-line note;
    - long outputs of old actions keep their start and end;
    - the full content of files written long ago is dropped from the call (the file is on disk);
    - terminal colour codes and runs of blank lines are removed.
    Recent messages are left untouched. Returns a new list."""
    pairs = _tool_results(messages)
    msgs = [dict(m) for m in messages]
    old = len(msgs) - keep_recent
    later_calls, later_changed = set(), set()
    for i in range(len(msgs) - 1, -1, -1):
        m = msgs[i]
        if m.get("content"):
            m["content"] = re.sub(r"\n{3,}", "\n\n", ANSI_RE.sub("", m["content"]))
        if m["role"] == "assistant" and m.get("tool_calls") and i < old:
            calls = []
            for tc in m["tool_calls"]:
                fn = tc.get("function", {})
                a = _args(fn)
                if fn.get("name") == "write_file" and len(str(a.get("content", ""))) > 400:
                    a = {**a, "content": f"[{len(a['content'])} characters written — read the file to see them]"}
                    tc = {**tc, "function": {**fn, "arguments": a}}
                calls.append(tc)
            m["tool_calls"] = calls
        if i not in pairs:
            continue
        name, a = pairs[i]
        path = str(a.get("path") or a.get("src") or "")
        if name in CHANGE_TOOLS:
            later_changed.update({path, str(a.get("dst") or "")})
            continue
        if name not in READ_TOOLS:
            if i < old and len(m.get("content") or "") > 2500:
                m["content"] = _head_tail(m["content"])
            continue
        key = (name, json.dumps(a, sort_keys=True))
        stale = key in later_calls or (name == "read_file" and path in later_changed)
        later_calls.add(key)
        if stale:
            m["content"] = f"[{name} output removed on load: superseded later in the conversation — call it again if needed]"
        elif i < old and len(m.get("content") or "") > 2500:
            m["content"] = _head_tail(m["content"])
    return msgs


def _head_tail(text, head=1800, tail=500):
    return text[:head] + f"\n[… {len(text) - head - tail} characters removed on load — repeat the action to see them …]\n" + text[-tail:]


def transcript(messages, tool_clip=800):
    parts = []
    for m in messages:
        text = m.get("content") or ""
        if m["role"] == "tool":
            text = clip(text, tool_clip)
        if m.get("tool_calls"):
            text += "\n[called: " + ", ".join(
                f"{tc['function']['name']}({_brief(tc['function'].get('arguments'), 200)})" for tc in m["tool_calls"]) + "]"
        if text.strip():
            parts.append(f"{m['role'].upper()}: {text.strip()}")
    return parts


def front_matter(text):
    """Split '---\\nkey: value\\n---\\nbody' into (dict, body)."""
    m = re.match(r"^---\n(.*?)\n---\n?(.*)$", text, re.S)
    if not m:
        return {}, text
    meta = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    return meta, m.group(2)


def file_digest(path, limit=14000):
    """File content with line numbers; big files keep the start, an outline of definitions and the end."""
    lines = path.read_text(errors="replace").splitlines()
    numbered = [f"{n:>5}  {l}" for n, l in enumerate(lines, 1)]
    text = "\n".join(numbered)
    if len(text) <= limit:
        return text
    head, acc = [], 0
    for l in numbered:
        acc += len(l) + 1
        if acc > limit * 0.5:
            break
        head.append(l)
    outline = [l for l in numbered[len(head):] if re.match(
        r"\s*\d+\s+(\s{0,4}(def |class |async def |function |export |const \w+ = \(|#{1,3} |<h[1-3]|<section|\[)|[A-Z_]{3,} = )", l)]
    outline = outline[:int(limit * 0.35 / 60)]
    tail = numbered[-15:]
    return "\n".join(head + [f"      [… {len(lines)} lines in total — outline of the rest:]"] + outline +
                     ["      [… end of the file:]"] + tail)


def is_text_file(path):
    try:
        with open(path, "rb") as f:
            chunk = f.read(4096)
        return b"\0" not in chunk
    except OSError:
        return False


WIKI_CHAT_PROMPT = """You maintain a project's memory as a compact Markdown wiki. Write the page chat.md: everything an AI agent needs to continue this work later WITHOUT the original conversation.

Rules:
- Dense bullet points. No filler, no pleasantries, no repetition.
- Keep EXACT: file paths, function/variable names, commands, numbers, versions, URLs, error messages, the user's preferences and requirements.
- {merge}
- When you mention one of these files, link it exactly as shown: {links}
- Write in English. Output only the page, starting with "# Chat summary".

Sections (skip empty ones):
## Goal
## User preferences & constraints
## Decisions (and why)
## Done
## Open problems / next steps
## Key facts
"""

WIKI_NOTES_PROMPT = """This is part {n} of {total} of a conversation between a user and an AI agent. Extract dense bullet-point notes: goals, requests, decisions, files touched (exact paths), what was done, facts found, errors, what is still open. Keep exact names, paths, numbers, commands and URLs. No filler.

"""

WIKI_FILE_PROMPT = """Write the wiki page for the file {rel}. An AI agent will read this page INSTEAD of the file to know what it is and how it works, and will open the real file only to edit it.

First line exactly: DESCRIPTION: <what the file is, max 15 words>
Then these sections (skip empty ones):
## Purpose
## Structure
(only the meaningful parts — functions, classes, sections — one line each with its line number; skip boilerplate; max 15 lines)
## Key details
(exact names, settings, values, dependencies, entry points, how to run it)
## Changes in this conversation

Rules: dense bullets, exact identifiers, no filler, no code blocks longer than 3 lines. Output only the page.

File content (with line numbers):
{content}

What happened to this file in the conversation:
{history}
"""


# ---------------------------------------------------------------- multi-agent: roles, GPU memory, runs

AGENTS_DIR = CONF_DIR / "agents"
RUNS_DIR = CONF_DIR / "runs"
VRAM_FILE = CONF_DIR / "vram.json"
AGENT_MODES = {
    "ask": "Chicken proposes each helper, you approve it",
    "auto": "Chicken calls helpers without asking (file changes and commands still ask, unless /mode auto)",
    "manual": "helpers run only when you type call -agent",
}
TOOL_GROUPS = {
    "read": ["list_dir", "read_file", "find_files", "search_in_files"],
    "web": ["web_search", "fetch_url"],
    "write": ["write_file", "edit_file", "make_dir", "move_path", "delete_path"],
    "run": ["run_command"],
}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
# GB in the GPU, measured on this computer (model@context). Updated each time a model loads.
VRAM_SEED = {"bonsai2:27b@24576": 7.4, "qwen3:14b@24576": 11.1, "qwen2.5-coder:14b@16384": 10.4, "gemma3:12b@8192": 7.8,
             "qwen2.5:7b@8192": 4.9, "deepseek-r1:8b@16384": 6.4, "llama3.2:3b@8192": 2.7,
             "glm-ocr:latest@4096": 1.9}
# tested here: does the model really call tools (not just claim to)?
TOOLS_TESTED = {"qwen3:14b": True, "qwen2.5-coder:14b": False, "bonsai2:27b": True}

BUILTIN_AGENTS = {
    "explorer": dict(description="Map a folder or project and find where things are (read only)",
                     model="boss", kind="tool", tools="read", think="false", context="", output="text", color="#5fd7ff",
                     body="You are the explorer. Look through files and folders to answer the task: what is where, how it "
                          "is organised, which files and lines matter. Never change anything.\n\nEnd with a short report "
                          "(max 250 words): the answer, the key paths (with line numbers when useful) and anything "
                          "surprising."),
    "researcher": dict(description="Search the web and write a short report with sources",
                       model="boss", kind="tool", tools="read,web", think="false", context="", output="text",
                       color="#87d787",
                       body="You are the researcher. Search the web for the task, read the 2-4 best sources with "
                            "fetch_url and check facts against each other.\n\nEnd with a report (max 400 words): the "
                            "findings as bullet points, each with its source URL, then anything uncertain."),
    "reviewer": dict(description="Check work critically and list concrete problems (read only)",
                     model="bonsai2:27b", kind="tool", tools="read,run", think="true", context="", output="text",
                     color="#ffaf5f",
                     body="You are the reviewer. Check the work described in the task critically: does it do what was "
                          "asked, is it correct, what is missing or broken. Read the files. Run existing tests only if "
                          "there are any. Never change files.\n\nEnd with exactly this: a line 'VERDICT: ok' or "
                          "'VERDICT: needs fixes', then a numbered list of concrete problems (file:line, what is wrong, "
                          "how to fix it). Max 300 words."),
    "planner": dict(description="Split a multi-step goal into ordered steps (never executes them)",
                    model="bonsai2:27b", kind="writer", tools="", think="true", context="", output="text",
                    color="#9d8cff",
                    body="You are a planning worker. Your only job is to decompose the goal into an ordered sequence of "
                         "executable steps. Do not execute the steps, do not call agents, do not change files.\n\n"
                         "Each step has: id, description, required_capabilities, preferred_role (one of the roles you "
                         "are given), depends_on (ids), success_criteria. Prefer the smallest useful number of steps.\n\n"
                         "Return only this JSON in a ```json block and stop:\n"
                         '{"task_type": "multi_step", "summary": "...", "steps": [{"id": 1, "description": "...", '
                         '"required_capabilities": ["..."], "preferred_role": "explorer", "depends_on": [], '
                         '"success_criteria": "..."}]}'),
    "coder": dict(description="Write and change code, run it and fix it (uses tools)",
                  model="bonsai2:27b", kind="tool", tools="read,write,run", think="false", context="", output="text",
                  color="#ff5fd7",
                  body="You are a senior programmer. Do exactly the task, keeping the existing style of the files. "
                       "Write complete, working code: no placeholders, no '...', no 'rest unchanged'."),
    "vision": dict(description="Look at images and screenshots and describe or read them",
                   model="gemma3:12b", kind="writer", tools="", think="false", context="8192", output="text",
                   color="#af87ff",
                   body="You look at the images and files you are given and answer the task precisely. Copy any text "
                        "exactly as written. Say clearly when something is not readable."),
    "summarizer": dict(description="Shrink long documents or logs to the essentials (low stakes)",
                       model="qwen2.5:7b", kind="writer", tools="", think="false", context="8192", output="text",
                       color="#d7d787",
                       body="Summarize the given files or text for the task. Keep exact names, numbers, paths and "
                            "commands. Bullet points, max 300 words."),
}
AGENT_TEMPLATE = """---
description: {description}
# model: the default model (the orchestrator may pick another) · boss = the main agent's model (no GPU swap)
model: {model}
# kind: tool = uses tools itself · writer = the program gives it the files and applies its answer
kind: {kind}
# tools (tool kind only): read, web, write, run
tools: {tools}
think: {think}
# context: memory size in tokens (empty = default)
context: {context}
# output (writer kind): files = its answer is applied to files · text = its answer is the result
output: {output}
color: {color}
---
{body}
"""

WRITER_FILES_RULES = """

HOW TO ANSWER. A program applies your answer to the files, so follow this format exactly:
- To create a file or replace a whole file: a line `FILE: <path>`, then the COMPLETE new file in one fenced code block.
- To change only part of a big file: a line `FILE: <path>`, then one or more blocks like this:
<<<<<<< SEARCH
lines copied exactly from the current file
=======
the new lines
>>>>>>> REPLACE
- After the files, write 1-3 short lines saying what you changed. Nothing else."""

WRITER_TEXT_RULES = "\n\nAnswer directly with the result. It goes back to the agent that asked for it. Never use emoji."

REPORT_FORMAT = ('{"status": "done", "summary": "what you did or found", "artifacts": ["paths created or changed"], '
                 '"important_facts": ["..."], "blocked_by": null, "errors": [], "suggested_followup": null}')
WORKER_RULES = f"""

You are a WORKER in a sequential multi-agent system. You have exactly one task: do only that task. You do not call
other agents and you do not decide the next task; the orchestrator does.
When finished, end with your report as JSON in a ```json block (status is one of done, partial, blocked, failed,
needs_review; put your full answer or report text in "summary"):
{REPORT_FORMAT}
Then stop."""

# what the orchestrator knows about each model (all run fully on the GPU; a model that can't fit is never offered)
MODEL_REGISTRY = {
    "bonsai2:27b": ("reasoning, planning, hard coding, debugging, architecture, review, recovery", "very high", "slow"),
    "qwen3.5:4b": ("orchestration, routing, light planning, reading and mapping files, structured output", "medium-high", "fast"),
    "qwen3.5:2b": ("trivial tasks, short answers, light summaries", "low-medium", "very fast"),
    "qwen3:14b": ("general reasoning, tool use, writing", "high", "medium"),
    "qwen2.5-coder:14b": ("coding, code editing, tests (weak at tool calls: use it as a writer)", "high", "medium"),
    "qwen2.5:14b": ("general tasks, writing, analysis", "medium-high", "medium"),
    "qwen2.5:7b": ("summarization, rewriting, general tasks", "medium", "fast"),
    "gemma3:12b": ("vision, image and screenshot understanding", "medium", "medium"),
    "glm-ocr:latest": ("OCR, document parsing", "specialized", "fast"),
    "maternion/LightOnOCR-2:latest": ("OCR, document parsing (fallback)", "specialized", "fast"),
    "deepseek-r1:8b": ("math and step-by-step reasoning", "medium", "slow"),
    "llama3.2:3b": ("trivial tasks", "low", "very fast"),
}



def model_registry(cfg):
    """The built-in registry plus the user's own models from config.json:
    "models": {"name": ["what it is good at", "intelligence", "speed"]}. A user entry replaces a built-in one."""
    reg = dict(MODEL_REGISTRY)
    for name, info in (cfg.get("models") or {}).items():
        if isinstance(info, (list, tuple)) and len(info) == 3:
            reg[name] = tuple(str(x) for x in info)
    return reg


ORCHESTRATOR_PROMPT = """You are Chicken, the ORCHESTRATOR of a sequential multi-agent system running locally on the user's Linux computer.
Your job is to decide what happens next. You do NOT perform the user's task yourself: workers do it.

Date: {date}
Working directory: {cwd} (workers can only access files under {root})
Mode: {mode} · routing: {routing}

You may:
- understand the goal and classify its complexity: trivial, low, medium, high, frontier
- decide whether it is single-step or multi-step
- launch ONE worker with call_agent: a role, a model, a stand-alone task, the reason and the complexity
- ask the user a necessary clarification: just write the question as your answer
- answer directly only for greetings, questions about yourself, or when worker results already contain the answer
- finish: when the goal is done, call finish with a short summary for the user

You must never: do the work yourself, write code for the user, launch several workers at once, or let a worker choose the next worker.

Rules:
- For multi-step tasks call the planner first (role planner, model {heavy}), then run its steps one by one with call_agent, passing the step number.
- Use the cheapest model likely to succeed. Escalate to {heavy} for hard coding, debugging, architecture, review after big changes, a worker that failed or was blocked twice, contradictory results, or when the user asks for the best quality.
- Each worker starts with an empty memory: write each task so it stands alone (goal, exact paths, constraints, what to return). Write worker tasks in English.
- After every worker result: check its status, validation and errors, then decide exactly one next action.
- Reply in the user's language. Be brief. Never use emoji."""

BTW_PROMPT = """You are the read-only /btw assistant of Chicken, a local multi-agent system on the user's Linux computer.
You answer questions about: the current task status, the plan, the model running now, finished steps, worker results,
files changed, why the orchestrator chose a model, what comes next. You can also answer short general questions.
You never change anything: you only read the snapshot below. Answer briefly, in the user's language. Never use emoji."""

MODE_INFO = {
    "ask": "confirm each worker and each action (file changes, commands)",
    "auto": "no confirmations: workers and actions run on their own",
    "orchestrator": "the main model only decides, workers do the work (default)",
    "bonsai-only": "Bonsai is the orchestrator and every worker",
    "bonsai-workers": "the orchestrator stays, every worker runs on Bonsai",
    "classic": "the main model works itself with all tools and calls helpers (the old Chicken)",
}
LOGS_DIR = CONF_DIR / "logs"
PROC_GB = 0.45          # GPU memory each extra model process needs beyond its weights and context (CUDA context)


def parse_report(text):
    """A worker's JSON report (the last JSON object with a "status" or "steps" key); None if there is none."""
    blocks = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text or "", re.S)
    candidates = blocks[::-1] + [m.group(0) for m in re.finditer(r"\{.*\}", text or "", re.S)][::-1]
    for c in candidates:
        try:
            d = json.loads(c)
        except ValueError:
            continue
        if isinstance(d, dict) and ("status" in d or "steps" in d):
            return d
    return None


def fmt_tokens(n):
    return f"{n / 1000:.1f}k" if n >= 1000 else str(int(n))


def fmt_secs(s):
    s = int(round(s))
    return f"{s // 60}m {s % 60:02d}s" if s >= 60 else f"{s}s"


def parse_text_tool_calls(content, names):
    """Tool calls a model wrote as text (Qwen's <tool_call><function=x><parameter=y>…) instead of real calls.
    Returns (content without them, calls)."""
    calls = []
    for block in re.findall(r"<tool_call>(.*?)(?:</tool_call>|$)", content, re.S):
        m = re.search(r"<function=([\w.-]+)>", block)
        if not m or m.group(1) not in names:
            continue
        args = {}
        for k, v in re.findall(r"<parameter=([\w.-]+)>\n?(.*?)\n?</parameter>", block, re.S):
            try:
                args[k] = json.loads(v) if v.strip()[:1] in "[{0123456789-" or v.strip() in ("true", "false", "null") else v
            except ValueError:
                args[k] = v
        calls.append({"function": {"name": m.group(1), "arguments": args}})
    if calls:
        content = re.sub(r"<tool_call>.*?(?:</tool_call>|$)", "", content, flags=re.S)
    return content, calls


def validate_files(paths):
    """Deterministic checks on files a worker changed: Python compiles, JSON parses, shell scripts parse."""
    notes = []
    for f in sorted(set(paths)):
        f = Path(f)
        if not f.is_file():
            continue
        try:
            if f.suffix == ".py":
                compile(f.read_text(errors="replace"), str(f), "exec")
            elif f.suffix == ".json":
                json.loads(f.read_text(errors="replace"))
            elif f.suffix in (".sh", ".bash"):
                r = subprocess.run(["bash", "-n", str(f)], capture_output=True, text=True, timeout=10)
                if r.returncode:
                    raise ValueError(r.stderr.strip().splitlines()[-1] if r.stderr.strip() else "syntax error")
            else:
                continue
            notes.append((f, True, "ok"))
        except (SyntaxError, ValueError) as e:
            notes.append((f, False, str(e).splitlines()[0][:200]))
    return notes


def agent_color(helper):
    """The colour of an agent in the output: the helper's own, or cyan for Chicken."""
    return (helper or {}).get("color") or THEME.get("cyan") or "#5fcff5"


def result_line(result):
    """One line for a helper's result: 'wrote a.py, b.py  +86 −3' for file changes, else its first line."""
    files = re.findall(r"(?m)^- (.+?): \+(\d+) −(\d+) lines", result)
    if result.startswith("Changes by the") and files:
        names = ", ".join(Path(f).name for f, _, _ in files[:3]) + (f" +{len(files) - 3} more" if len(files) > 3 else "")
        plus, minus = sum(int(a) for _, a, _ in files), sum(int(b) for _, _, b in files)
        return dim(f"wrote {names}  ") + green(f"+{plus}") + " " + red(f"−{minus}")
    first = next((l for l in result.splitlines() if l.strip()), "")
    return _brief(first, 100)


def hexc(h, s, bold=False):
    """Text in a #rrggbb colour."""
    try:
        r, g, b = int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)
    except (ValueError, TypeError):
        return s
    return c(f"{'1;' if bold else ''}38;2;{r};{g};{b}", s)


def ensure_agents():
    AGENTS_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    for name, d in BUILTIN_AGENTS.items():
        f = AGENTS_DIR / f"{name}.md"
        if not f.exists():
            f.write_text(AGENT_TEMPLATE.format(**d))


def load_roles():
    roles = {}
    for f in sorted(AGENTS_DIR.glob("*.md")) if AGENTS_DIR.exists() else []:
        text = f.read_text()
        meta, body = {}, text
        m = re.match(r"^---\n(.*?)\n---\n?(.*)$", text, re.S)
        if m:
            for line in m.group(1).splitlines():
                if ":" in line and not line.lstrip().startswith("#"):
                    k, v = line.split(":", 1)
                    meta[k.strip()] = v.strip()
            body = m.group(2)
        try:
            ctx = int(meta.get("context") or 0)
        except ValueError:
            ctx = 0
        roles[f.stem] = {
            "name": f.stem, "description": meta.get("description", ""), "model": meta.get("model") or "boss",
            "kind": "writer" if meta.get("kind", "tool").lower() == "writer" else "tool",
            "tools": [t.strip() for t in meta.get("tools", "read").split(",") if t.strip() in TOOL_GROUPS],
            "think": meta.get("think", "false").lower() == "true", "context": ctx,
            "output": "files" if meta.get("output", "text").lower() == "files" else "text",
            "color": meta.get("color", "#d7af5f"), "prompt": body.strip(), "file": f,
        }
    return roles


def set_role_meta(role, key, value):
    """Change one setting in a role's .md file, keeping the rest (and its comments) as they are."""
    f = role["file"]
    text = f.read_text()
    new, n = re.subn(rf"^{re.escape(key)}:.*$", f"{key}: {value}", text, count=1, flags=re.M)
    if not n:
        new = text.replace("\n---\n", f"\n{key}: {value}\n---\n", 1)
    f.write_text(new)


def role_model(role, cfg):
    return cfg["model"] if role["model"] in ("boss", "") else role["model"]


def role_ctx(role, cfg):
    if role_model(role, cfg) == cfg["model"]:
        return cfg["num_ctx"]          # same model as the main agent: same memory size, so no reload
    return role["context"] or 8192


def mname(m):
    return m if ":" in m else m + ":latest"


SERVERS_DIR = CONF_DIR / "servers"
SYSTEM_FILE = CONF_DIR / "system.md"
SYSTEM_DEFAULT = """You are Chicken, an AI assistant running locally on the user's Linux computer, inside a terminal. You work directly on the user's files and folders through tools.

Date: {date}
Working directory: {cwd}
You can only access files under {root}.

How to work:
- Reply in the same language the user writes in.
- Before using tools, write ONE short sentence telling the user what you are about to do (e.g. "Reading the config to find the port setting."). Don't narrate every tiny step.
- Use tools to look at files and folders instead of guessing. Read a file before editing it.
- For small changes use edit_file (copy old_text exactly from read_file output, without the line-number prefix). Use write_file for new files or complete rewrites.
- Relative paths are relative to the working directory.
- For current events, facts you are unsure about, documentation or anything online: use web_search, then fetch_url on the best results. Mention the URLs you used.
- Actions that change things (write, edit, move, delete, run_command) are shown to the user for approval. If one is denied, do not retry it; follow the user's note or ask what they prefer.
- For multi-step work, call update_plan with a checklist and keep it updated.
- Be concise and practical. When you finish, say briefly what you did.
- Never use emoji. Plain text and Markdown only.
"""


def system_template():
    """Chicken's base instructions: ~/.agent/system.md if it exists (edit it with /system edit), else the default."""
    try:
        text = SYSTEM_FILE.read_text()
        return text if text.strip() else SYSTEM_DEFAULT
    except OSError:
        return SYSTEM_DEFAULT


KNOWLEDGE_DIR = CONF_DIR / "Built_skills_and_knowledge"
_STOP = set("a an and are as at be by can do does for from has have how i if in into is it its of on or that the "
            "this to use using what when which with you your not no all any each more most other some such than "
            "then there these they we will would should could also only just very".split())


def _terms(text):
    out_ = []
    for w in re.findall(r"[a-z0-9][a-z0-9_+#.-]*", text.lower()):
        w = w.strip(".-")
        if len(w) < 2 or w in _STOP:
            continue
        out_.append(w[:-1] if len(w) > 4 and w.endswith("s") and not w.endswith("ss") else w)
    return out_


class Knowledge:
    """The knowledge wikis in ~/.agent/Built_skills_and_knowledge: one folder per wiki, pages split into sections,
    searched with BM25 (keywords, CPU only). Each hit carries its page's sources."""

    def __init__(self, root=KNOWLEDGE_DIR):
        self.root, self.sig, self.sections, self.wikis = Path(root), None, [], {}

    def _load(self):
        files = sorted(self.root.glob("*/*.md")) if self.root.is_dir() else []
        sig = tuple((str(f), f.stat().st_mtime) for f in files)
        if sig == self.sig:
            return
        self.sig, self.sections, self.wikis = sig, [], {}
        for f in files:
            wiki, text = f.parent.name, f.read_text(errors="replace")
            if f.name == "index.md":
                lines = [l for l in text.splitlines() if l.strip()]
                self.wikis[wiki] = lines[1] if len(lines) > 1 else ""
                continue
            title = (re.search(r"^# (.+)$", text, re.M) or [None, f.stem])[1]
            checked = (re.search(r"^> Checked: (.+)$", text, re.M) or [None, "?"])[1]
            parts = re.split(r"(?m)^## ", text)
            sources = next((p.split("\n", 1)[1].strip() for p in parts[1:] if p.startswith("Sources")), "")
            for part in parts[1:]:
                head, _, body = part.partition("\n")
                if head.strip() == "Sources" or not body.strip():
                    continue
                self.sections.append({"wiki": wiki, "page": f.stem, "path": f, "title": title, "head": head.strip(),
                                      "text": body.strip(), "sources": sources, "checked": checked,
                                      "terms": _terms(f"{title} {head} {head} {body}")})
        self.df = collections.Counter(t for s_ in self.sections for t in set(s_["terms"]))
        self.avg = sum(len(s_["terms"]) for s_ in self.sections) / max(1, len(self.sections))

    def search(self, query, k=4, wiki=None):
        self._load()
        q = set(_terms(query))
        if not q or not self.sections:
            return []
        n, hits = len(self.sections), []
        for sec in self.sections:
            if wiki and sec["wiki"] != wiki:
                continue
            tf = collections.Counter(sec["terms"])
            score = 0.0
            for t in q:
                if t in tf:
                    idf = math.log(1 + (n - self.df[t] + 0.5) / (self.df[t] + 0.5))
                    score += idf * tf[t] * 2.2 / (tf[t] + 1.2 * (0.25 + 0.75 * len(sec["terms"]) / self.avg))
            if score > 0:
                hits.append((score, sec))
        hits.sort(key=lambda h: -h[0])
        return hits[:k]

    def listing(self):
        self._load()
        return self.wikis

    @staticmethod
    def render(sec, limit=None):
        body = sec["text"] if limit is None else clip(sec["text"], limit)
        src = "; ".join(l.lstrip("- ").split(": http")[0] for l in sec["sources"].splitlines() if l.strip())[:400]
        return (f"### {sec['wiki']}/{sec['page']}.md › {sec['head']}  (checked {sec['checked']})\n{body}\n"
                f"Sources: {src}")


KNOWLEDGE = Knowledge()
KNOWLEDGE_NOTE = ("Reference knowledge from Chicken's wikis. It is background, not the truth about this task: the "
                  "user's files, the project's own docs and real command output take precedence; anything that changes "
                  "over time (versions, prices, statistics, laws) must be checked again. If you rely on it, name the "
                  "page and its source in important_facts.")


KV_BYTES = {"f32": 4.0, "f16": 2.0, "bf16": 2.0, "q8_0": 1.0625, "q5_1": 0.75, "q5_0": 0.6875,
            "q4_1": 0.625, "q4_0": 0.5625}     # bytes per value of the context cache, by cache type


def gguf_meta(path):
    """The metadata of a GGUF model file (architecture, layers, heads, context length…); long arrays are skipped."""
    fmt = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
    meta = {}
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError(f"{path} is not a GGUF file")
        _, _, n = struct.unpack("<IQQ", f.read(20))

        def text():
            k, = struct.unpack("<Q", f.read(8))
            return f.read(k).decode("utf-8", "replace")

        def value(t):
            if t == 8:
                return text()
            if t == 9:
                at, cnt = struct.unpack("<IQ", f.read(12))
                if at == 8:
                    for _ in range(cnt):
                        k, = struct.unpack("<Q", f.read(8))
                        f.seek(k, 1)
                    return None
                size = struct.calcsize("<" + fmt[at])
                if cnt > 64:
                    f.seek(size * cnt, 1)
                    return None
                return list(struct.unpack(f"<{cnt}{fmt[at]}", f.read(size * cnt)))
            return struct.unpack("<" + fmt[t], f.read(struct.calcsize("<" + fmt[t])))[0]

        for _ in range(n):
            key = text()
            t, = struct.unpack("<I", f.read(4))
            meta[key] = value(t)
    return meta


class Servers:
    """llama.cpp servers for models Ollama can't run (e.g. Bonsai). Started and stopped on demand."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.started = set()        # servers this Chicken started (stopped when it quits)

    def spec(self, name):
        s = self.cfg.get("servers", {}).get(name)
        return s if s and Path(os.path.expanduser(s["model"])).exists() else None

    def names(self):
        return [n for n in self.cfg.get("servers", {}) if self.spec(n)]

    def url(self, name):
        return f"http://127.0.0.1:{self.spec(name)['port']}"

    def _state(self, name):
        try:
            return json.loads((SERVERS_DIR / f"{name.replace(':', '_')}.json").read_text())
        except (OSError, ValueError):
            return {}

    def running(self, name):
        try:
            return requests.get(self.url(name) + "/health", timeout=0.7).status_code == 200
        except Exception:
            return False

    def ctx(self, name):
        return self._state(name).get("ctx")

    def vram_gb(self, name):
        """GPU memory the server really uses (from nvidia-smi), or None."""
        pid = self._state(name).get("pid")
        try:
            rows = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                                  capture_output=True, text=True, timeout=5).stdout.splitlines()
        except Exception:
            return None
        for row in rows:
            p, _, mib = row.partition(",")
            if p.strip() == str(pid):
                return round(int(mib) * 1.048576 / 1000, 1)
        return None

    def start(self, name, ctx, cancel=None):
        spec = self.spec(name)
        if self.running(name):
            if self.ctx(name) == ctx:
                return
            self.stop(name)
        binary = Path(os.path.expanduser(spec["bin"]))
        SERVERS_DIR.mkdir(parents=True, exist_ok=True)
        log = open(SERVERS_DIR / f"{name.replace(':', '_')}.log", "w")
        gpu = spec.get("gpu", True)
        cmd = [str(binary), "-m", os.path.expanduser(spec["model"]), "-ngl", "99" if gpu else "0", "-c", str(ctx),
               "-np", "1", "--cache-ram", "8192", "--host", "127.0.0.1", "--port", str(spec["port"])]
        cmd += ["-fa", "on"] if gpu else ["-t", str(spec.get("threads", 8))]
        if spec.get("mmproj"):
            cmd += ["--mmproj", os.path.expanduser(spec["mmproj"])]
        env = {**os.environ, "LD_LIBRARY_PATH": ":".join(
            [str(binary.parent)] + [os.path.expanduser(p) for p in str(spec.get("libs", "")).split(":") if p])}
        if not gpu:
            env["CUDA_VISIBLE_DEVICES"] = ""          # CPU only: don't take any GPU memory
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                start_new_session=True, env=env)
        (SERVERS_DIR / f"{name.replace(':', '_')}.json").write_text(json.dumps({"pid": proc.pid, "ctx": ctx}))
        self.started.add(name)
        t0 = time.time()
        while not self.running(name):
            if proc.poll() is not None:
                tail = (SERVERS_DIR / f"{name.replace(':', '_')}.log").read_text(errors="replace").strip().splitlines()[-3:]
                raise RuntimeError(f"The {name} server stopped while starting: " + " | ".join(tail))
            if cancel is not None and cancel.is_set():
                self.stop(name)
                raise Interrupted()
            if time.time() - t0 > 240:
                self.stop(name)
                raise RuntimeError(f"The {name} server didn't start within 4 minutes")
            time.sleep(0.3)

    def stop(self, name):
        pid = self._state(name).get("pid")
        if pid:
            try:
                os.killpg(pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
            for _ in range(50):
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    break
                time.sleep(0.1)
        (SERVERS_DIR / f"{name.replace(':', '_')}.json").unlink(missing_ok=True)
        self.started.discard(name)

    def stop_started(self):
        for name in list(self.started):
            self.stop(name)


def to_openai(messages):
    """Chicken's (Ollama-style) messages → OpenAI chat format for llama.cpp servers."""
    out_, pending, system = [], [], []
    for k, m in enumerate(messages):
        role = m["role"]
        if role == "system":
            system.append(m.get("content") or "")      # the Bonsai template wants exactly one system message, first
        elif role == "assistant":
            msg = {"role": "assistant", "content": m.get("content") or ""}
            if m.get("tool_calls"):
                pending, calls = [], []
                for j, tc in enumerate(m["tool_calls"]):
                    fn = tc.get("function", {})
                    args = fn.get("arguments", {})
                    tid = tc.get("id") or f"call_{k}_{j}"
                    pending.append(tid)
                    calls.append({"id": tid, "type": "function", "function": {
                        "name": fn.get("name", ""), "arguments": args if isinstance(args, str) else json.dumps(args or {})}})
                msg["tool_calls"] = calls
            out_.append(msg)
        elif role == "tool":
            out_.append({"role": "tool", "tool_call_id": pending.pop(0) if pending else f"call_{k}",
                         "content": m.get("content") or ""})
        else:
            out_.append({"role": role, "content": m.get("content") or ""})
    if system:
        out_.insert(0, {"role": "system", "content": "\n\n".join(system)})
    return out_


class VRAM:
    """Keeps models in the GPU together while they fit in the budget; otherwise ejects the ones not needed."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.servers = Servers(cfg)
        self.used = {}              # model -> last time it was needed
        self.warned_limit = False
        self.cancel = None          # set by the agent: lets Esc stop a slow start
        try:
            self.known = {**VRAM_SEED, **json.loads(VRAM_FILE.read_text())}
        except (OSError, ValueError):
            self.known = dict(VRAM_SEED)
        self.profiles = {}          # model -> {"base": GB, "tok": GB per context token, "max": tokens}
        self.ctx_of = {}            # loaded model -> its context size
        self.partial = {}           # loaded model -> True if Ollama put part of it on the CPU
        self._env = None
        self._total = 0.0

    def gpu_mem(self):
        """(used, total) GPU memory in GB (decimal, like model sizes), from nvidia-smi."""
        try:
            row = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=5).stdout.splitlines()[0]
            used, total = (int(x) * 1.048576 / 1000 for x in row.split(","))
            return used, total
        except Exception:
            return 0.0, 0.0

    def budget(self):
        """The GPU limit in GB: vram_budget_gb is a share of the card ("90%") or a number of GB."""
        b = self.cfg["vram_budget_gb"]
        if isinstance(b, str) and b.strip().endswith("%"):
            if not self._total:
                self._total = self.gpu_mem()[1] or 12.0
            return round(self._total * float(b.strip()[:-1]) / 100, 2)
        return float(b)

    def eligible(self):
        """Models the orchestrator may choose: in the registry, installed, and fitting fully in the GPU limit alone."""
        try:
            tags = {t["name"] for t in requests.get(self.cfg["host"] + "/api/tags", timeout=5).json()["models"]}
        except Exception:
            tags = set()
        out_ = []
        for m, info in model_registry(self.cfg).items():
            if (self.servers.spec(m) or m in tags) and self.need(m, 8192) <= self.budget():
                out_.append((m, info))
        return out_

    def norm(self, model):
        return model if self.servers.spec(model) else mname(model)

    def on_gpu_only(self, model):
        spec = self.servers.spec(model)
        return not spec or spec.get("gpu", True)

    def ollama_env(self):
        """The Ollama service's settings (OLLAMA_*), read once."""
        if self._env is None:
            self._env = {}
            try:
                line = subprocess.run(["systemctl", "show", "ollama", "-p", "Environment"], capture_output=True,
                                      text=True, timeout=5).stdout.strip().partition("=")[2]
                self._env = dict(kv.split("=", 1) for kv in line.split() if "=" in kv)
            except Exception:
                pass
        return self._env

    def can_share(self, a, b):
        """Can the two models be in the GPU together? Servers are their own processes; Ollama may keep only one."""
        if self.servers.spec(a) or self.servers.spec(b):
            return True
        return self.ollama_env().get("OLLAMA_MAX_LOADED_MODELS", "0") != "1"

    def profile(self, model):
        """Memory of a model: weights and buffers ("base", GB) + context cache per token ("tok", GB), from the
        model's own metadata; a measured size corrects the base. "max" is the longest context it supports."""
        model = self.norm(model)
        if model in self.profiles:
            return self.profiles[model]
        spec = self.servers.spec(model)
        try:
            if spec:
                path = os.path.expanduser(spec["model"])
                meta, elem = gguf_meta(path), KV_BYTES.get(spec.get("cache_type", "f16"), 2.0)
                base = Path(path).stat().st_size / 1e9 + 0.25
            else:
                meta = requests.post(self.cfg["host"] + "/api/show", json={"model": model}, timeout=10).json()["model_info"]
                env = self.ollama_env()
                elem = (KV_BYTES.get(env.get("OLLAMA_KV_CACHE_TYPE", "f16"), 2.0)
                        if env.get("OLLAMA_FLASH_ATTENTION", "").lower() in ("1", "true") else 2.0)
                tags = requests.get(self.cfg["host"] + "/api/tags", timeout=5).json()["models"]
                base = next(t["size"] for t in tags if t["name"] == model) / 1e9 * 0.97
            arch = meta.get("general.architecture", "")

            def g(k):
                v = meta.get(f"{arch}.{k}")
                return max(v) if isinstance(v, list) and v else v
            layers, heads = g("block_count"), g("attention.head_count")
            kv_heads = g("attention.head_count_kv") or heads
            klen = g("attention.key_length") or g("embedding_length") // heads
            vlen = g("attention.value_length") or klen
            cache_layers = math.ceil(layers / (g("full_attention_interval") or 1))   # hybrid models: few attention layers
            prof = {"base": base, "tok": cache_layers * kv_heads * (klen + vlen) * elem / 1e9,
                    "max": int(g("context_length") or 32768)}
        except Exception:
            return {"base": 8.0, "tok": 1 / 16384, "max": 32768}      # unknown: a cautious guess, not remembered
        seen = [(int(k.rsplit("@", 1)[1]), gb) for k, gb in self.known.items() if k.rsplit("@", 1)[0] == model]
        if seen:
            ctx, gb = max(seen)
            prof["base"] = gb - prof["tok"] * ctx
        self.profiles[model] = prof
        return prof

    def fit_ctx(self, model, gb):
        """The longest context (a multiple of 1024 tokens) with which `model` fits in `gb` of GPU memory; 0 if none."""
        p = self.profile(model)
        if p["tok"] <= 0 or gb <= p["base"]:
            return 0
        return min(int((gb - p["base"]) / p["tok"]) // 1024 * 1024, p["max"])

    def loaded(self):
        """Models in memory now: name -> GB in the GPU."""
        found = {}
        try:
            for m in requests.get(self.cfg["host"] + "/api/ps", timeout=5).json().get("models", []):
                found[m["name"]] = m.get("size_vram", 0) / 1e9
                self.ctx_of[m["name"]] = m.get("context_length")
                self.partial[m["name"]] = m.get("size", 0) > m.get("size_vram", 0) * 1.02
        except Exception:
            pass
        for name in self.servers.names():
            if self.servers.spec(name).get("gpu", True) and self.servers.running(name):
                ctx = self.servers.ctx(name)
                found[name], self.ctx_of[name], self.partial[name] = self.need(name, ctx), ctx, False
        return found

    def fully_on_gpu(self, *models):
        """False if Ollama had to put part of one of these models on the CPU (never wanted)."""
        self.loaded()
        return not any(self.partial.get(self.norm(m)) for m in models)

    def need(self, model, ctx):
        spec = self.servers.spec(model)
        key = f"{model if spec else mname(model)}@{ctx}"
        if key in self.known:
            return self.known[key]
        p = self.profile(model)
        return round(p["base"] + p["tok"] * ctx, 1)     # weights + the context cache

    def _load(self, model, ctx):
        with ui.busy(f"Loading {model} into the GPU"):
            if self.servers.spec(model):
                self.servers.start(model, ctx, self.cancel)
                return
            try:
                r = requests.post(self.cfg["host"] + "/api/generate", timeout=600, json={
                    "model": model, "prompt": "", "keep_alive": "30m", "options": {"num_ctx": ctx}})
            except requests.ConnectionError:
                raise RuntimeError(f"Cannot reach Ollama at {self.cfg['host']}")
            if r.status_code != 200:
                raise RuntimeError(f"Could not load {model}: {r.text[:200]}")

    def eject(self, model):
        if self.servers.spec(model):
            self.servers.stop(model)
            return
        try:
            requests.post(self.cfg["host"] + "/api/generate", json={"model": model, "keep_alive": 0}, timeout=30)
        except Exception:
            pass

    def ensure(self, model, ctx):
        """Get `model` ready in the GPU. Prints what it did. Returns the seconds spent."""
        model = model if self.servers.spec(model) else mname(model)
        spec = self.servers.spec(model)
        if spec and not spec.get("gpu", True):
            t0 = time.time()
            self.servers.start(model, ctx, self.cancel)          # CPU model: no GPU budget, nothing to eject
            return time.time() - t0
        self.used[model] = time.time()
        loaded = self.loaded()
        if model in loaded and self.ctx_of.get(model) in (ctx, None):
            return 0.0
        if model in loaded:            # loaded with another context size: take it out, then load it at the new size
            self.eject(model)
            loaded.pop(model)
        budget, need = self.budget(), self.need(model, ctx)
        others = sorted(loaded, key=lambda m: self.used.get(m, 0))
        ejected = []
        while others and sum(loaded[m] for m in others) + need + PROC_GB * len(others) > budget:
            m = others.pop(0)
            self.eject(m)
            ejected.append(m)
        t0 = time.time()
        self._load(model, ctx)
        self.loaded()
        if self.partial.get(model):          # Ollama put part of it on the CPU: never allowed. Clear the GPU, retry once.
            for m in [m for m in others if m in self.loaded()]:
                self.eject(m)
                ejected.append(m)
            self.eject(model)
            self._load(model, ctx)
            self.loaded()
            if self.partial.get(model):
                self.eject(model)
                raise RuntimeError(f"{model} doesn't fit fully in the GPU at {ctx // 1024}k context, so it was not "
                                   f"run (Chicken never runs models on the CPU)")
        dt = time.time() - t0
        if self.servers.spec(model):
            gb = self.servers.vram_gb(model)
            if gb:
                self.known[f"{model}@{ctx}"] = gb
        now = self.loaded()
        if model in now and now[model] > 0 and not self.servers.spec(model):
            self.known[f"{model}@{ctx}"] = round(now[model], 1)
            try:
                VRAM_FILE.write_text(json.dumps(self.known, indent=1))
            except OSError:
                pass
        kept = [m for m in others if m in now]
        dropped = [m for m in others if m not in now]
        msg = "⇄ " + (f"ejected {', '.join(ejected)} · " if ejected else "") + \
            f"loaded {model} ({self.known.get(f'{model}@{ctx}', need):.1f} GB, {ctx // 1024}k context) in {dt:.1f}s"
        if kept:
            msg += f" · kept {', '.join(kept)} (fits in the {budget:.1f} GB limit)"
        out(dim("  " + msg))
        if dropped and not self.warned_limit:
            self.warned_limit = True
            out(dim(f"  ⇄ Ollama also unloaded {', '.join(dropped)}: it is set to keep only one model at a time "
                    f"(OLLAMA_MAX_LOADED_MODELS)"))
        return dt


def new_run():
    rid = time.strftime("%Y%m%d-%H%M%S")
    d = RUNS_DIR / rid
    d.mkdir(parents=True, exist_ok=True)
    return {"id": rid, "dir": d, "steps": []}


def parse_writer(text, paths, cwd):
    """Turn a writer's answer into edits: [(path, 'file', content) | (path, 'replace', old, new)], plus its notes."""
    edits, used = [], []
    marks = list(re.finditer(r"^\s*(?:#+\s*)?(?:\*\*)?(?:FILE|File|file)\s*:?\s*(?:\*\*)?\s*`?([^\s`*]+?)`?\s*(?:\*\*)?\s*$",
                             text, re.M))
    segments = [(m.group(1), text[m.end():marks[i + 1].start() if i + 1 < len(marks) else len(text)], m)
                for i, m in enumerate(marks)]
    if not marks and len(paths) == 1:
        segments = [(str(paths[0]), text, None)]
    sr = re.compile(r"<<<<<<< ?SEARCH\n(.*?)\n?=======\n(.*?)\n?>>>>>>> ?REPLACE", re.S)
    fence = re.compile(r"```[^\n]*\n(.*?)\n?```", re.S)
    for path, seg, m in segments:
        blocks = sr.findall(seg)
        if blocks:
            edits += [(path, "replace", old, new) for old, new in blocks]
            used.append(sr.sub("", seg))
            continue
        fm = fence.search(seg)
        if fm:
            edits.append((path, "file", fm.group(1) + ("\n" if not fm.group(1).endswith("\n") else "")))
            used.append(seg[:fm.start()] + seg[fm.end():])
        else:
            used.append(seg)
    notes = text
    if marks or edits:
        notes = "\n".join(u.strip() for u in used if u.strip())
        if marks:
            notes = text[:marks[0].start()].strip() + "\n" + notes
    notes = re.sub(r"```.*?```", "", notes, flags=re.S).strip()
    return edits, notes


def diff_stats(old, new):
    plus = minus = 0
    for line in difflib.ndiff((old or "").splitlines(), (new or "").splitlines()):
        if line.startswith("+ "):
            plus += 1
        elif line.startswith("- "):
            minus += 1
    return plus, minus


class Agent:
    def __init__(self, cfg, auto=False):
        self.cfg = cfg
        self.root = Path(os.path.realpath(os.path.expanduser(cfg["root"])))
        cwd = Path(os.path.realpath(os.getcwd()))
        self.cwd = cwd if (cwd == self.root or self.root in cwd.parents) else self.root
        os.chdir(self.cwd)
        self.messages = []
        self.focus = []            # files the user is working on
        self.pending = []          # context to attach to the next user message
        self.plan = ""
        self.wiki = None           # folder of the project wiki loaded with /load -wiki
        self.auto = auto
        self.always = set()        # tool names approved for the whole session
        self.last_tokens = 0
        self.last_tps = 0.0
        self.last_output = None    # (label, full result) of the last action
        self.loop_goal = None
        self.loop_steps = 0
        self.loop_max = cfg["loop_max_steps"]
        self.completed = None
        self.recent_calls = []
        self.think = cfg["think"]
        self._est = (None, 0)
        self.cancel = threading.Event()   # set by interrupt(): stops the current work
        self.response = None              # streaming HTTP response in progress
        self.proc = None                  # command in progress
        self.asker = None                 # fn(question, options, note_index) -> (index|None, note)
        self.helper = None                # role dict when this agent is a helper
        self.child = None                 # helper running right now
        self.run = None                   # current multi-agent run: {"id", "dir", "steps"}
        self.vram = VRAM(cfg)
        self.touched = set()              # files changed by the current step (for the validators)
        self.usage = {"in": 0, "out": 0, "think": 0.0}   # this request, all agents: tokens read, written; seconds thinking
        self.task = None                  # external task state of the current request (authoritative)
        self.base_model = cfg["model"]    # the main model chosen with /model (bonsai-only replaces it while on)
        self.ctx_auto = cfg.get("num_ctx") == "auto"
        self.ctx_fixed = None if self.ctx_auto else int(cfg["num_ctx"])
        if self.ctx_auto:
            cfg["num_ctx"] = self.full_ctx()

    def orchestrating(self):
        """True for the main agent unless routing is classic: it decides, workers do the work."""
        return not self.helper and self.cfg.get("routing", "orchestrator") != "classic"

    def apply_mode(self):
        """Make the saved mode and routing active: approvals, and the main model in bonsai-only."""
        c = self.cfg
        self.auto = self.auto or c.get("mode") == "auto"
        if c.get("mode") == "auto" and c.get("agents_mode") != "manual":
            c["agents_mode"] = "auto"
        c["model"] = c["heavy_model"] if c.get("routing") == "bonsai-only" else self.base_model
        c["num_ctx"] = self.full_ctx()

    def pick_model(self, role, asked=None):
        """The model a worker runs on: forced by bonsai-only / bonsai-workers, else the orchestrator's pick if it is
        eligible, else the role's default."""
        if self.cfg.get("routing") in ("bonsai-only", "bonsai-workers"):
            return self.cfg["heavy_model"]
        if asked and asked != "boss":
            names = {m for m, _ in self.vram.eligible()}
            if asked in names or mname(asked) in names:
                return asked if asked in names else mname(asked)
        return role_model(role, self.cfg)

    def new_task(self, goal):
        return {"task_id": time.strftime("%Y%m%d-%H%M%S"), "mode": self.cfg.get("mode"),
                "routing": self.cfg.get("routing"), "goal": goal, "status": "running", "plan": [], "artifacts": [],
                "last_result": None, "errors": [], "notes": [], "calls": [], "fails": {}}

    def save_task(self):
        if self.task and self.run:
            try:
                (self.run["dir"] / "state.json").write_text(json.dumps(self.task, indent=1, ensure_ascii=False))
            except OSError:
                pass

    def full_ctx(self):
        """The main agent's context when it's alone in the GPU: all the memory its weights leave in the budget
        (the orchestrator needs less: orchestrator_ctx caps it)."""
        if not self.ctx_auto:
            return self.ctx_fixed
        fit = self.vram.fit_ctx(self.cfg["model"], self.vram.budget())
        if self.orchestrating() and self.cfg.get("orchestrator_ctx"):
            fit = min(fit, self.cfg["orchestrator_ctx"])
        return max(fit, self.cfg["ctx_min"])

    def guard_vram(self, keep):
        """The GPU limit rule: over it, pause; compress (shrink the main agent's context when the conversation still
        fits); only then eject models the current step doesn't need. Never the CPU."""
        v = self.vram
        used, _ = v.gpu_mem()
        lim = v.budget()
        if not used or used <= lim:
            return
        out(yellow(f"  ! GPU {used:.1f} GB is over the {lim:.1f} GB limit: pausing to compress"))
        main = self.cfg["model"]
        if not self.helper and main not in keep and main in v.loaded():
            target = v.fit_ctx(main, v.need(main, self.cfg["num_ctx"]) - (used - lim) - 0.2)
            if target >= self.cfg["ctx_min"] and self.context_used() + 1024 <= target:
                out(dim(f"  ⇄ chicken context {self.cfg['num_ctx'] // 1024}k → {target // 1024}k (GPU limit)"))
                self.cfg["num_ctx"] = target
                v.ensure(main, target)
                used, _ = v.gpu_mem()
        for m in sorted(v.loaded(), key=lambda m: v.used.get(m, 0)):
            if used <= lim:
                break
            if m in keep:
                continue
            v.eject(m)
            out(dim(f"  ⇄ ejected {m} (GPU limit)"))
            time.sleep(0.5)
            used, _ = v.gpu_mem()
        if used > lim:
            out(yellow(f"  ! still {used:.1f} GB in use: other programs are using the GPU"))

    def make_room(self, model, ctx, name):
        """Keep the main agent in the GPU beside a helper by shrinking its context, compacting the conversation
        first if needed. True if they now fit together; False means the usual eject and inject."""
        main, v = self.cfg["model"], self.vram
        if not v.can_share(main, model):
            return False
        fit = v.fit_ctx(main, v.budget() - v.need(model, ctx) - PROC_GB - 0.15)
        if fit < self.cfg["ctx_min"]:
            return False
        reserve = 1024          # the main agent doesn't answer while the helper works, and grows back before it does
        if self.context_used() + reserve > fit:
            if not self.cfg["compact_for_helpers"]:
                return False
            self.maybe_compact(force=True, target=fit - reserve, reason=f"to make room for {name}")
            if self.context_used() + reserve > fit:
                return False
        old = self.cfg["num_ctx"]
        if fit < old:
            out(dim(f"  ⇄ chicken context {old // 1024}k → {fit // 1024}k to make room for {name}"))
            self.cfg["num_ctx"] = fit
            v.ensure(main, fit)
        return True

    def interrupt(self):
        self.cancel.set()
        if self.child is not None:
            self.child.interrupt()
        r, p = self.response, self.proc
        if r is not None:
            _hard_close(r)
        if p is not None:
            _killpg(p)

    def mark_interrupted(self):
        """Record that the user stopped the current request, so the AI doesn't pick it up again later."""
        last = self.messages[-1] if self.messages else None
        if last and not (last["role"] == "assistant" and not last.get("tool_calls")):
            self.messages.append({"role": "assistant", "content": "[Stopped: the user interrupted this request.]"})
        self.was_interrupted = True
        self.autosave()

    def check_cancel(self):
        if self.cancel.is_set():
            raise Interrupted()

    # ---------------- paths & permissions

    def resolve(self, p, write=False):
        p = os.path.expanduser(str(p or "."))
        full = Path(p) if os.path.isabs(p) else self.cwd / p
        full = Path(os.path.realpath(full))
        if full != self.root and self.root not in full.parents:
            raise ToolError(f"Access denied: {full} is outside {self.root}")
        if write:
            rel = str(full.relative_to(self.root))
            for pr in self.cfg["protected"]:
                if rel == pr or rel.startswith(pr.rstrip("/") + "/"):
                    raise ToolError(f"Access denied: {short_path(full)} is protected")
        return full

    def rel(self, p):
        try:
            return str(Path(p).relative_to(self.cwd)) or "."
        except ValueError:
            return short_path(p)

    def main_agent(self):
        """The main agent (a worker's parent chain ends there)."""
        a = self
        while getattr(a, "parent", None) is not None:
            a = a.parent
        return a

    def approve(self, name, question):
        """Returns True if allowed, otherwise the message to send back to the AI."""
        if self.auto or self.main_agent().auto or name in self.always:
            return True
        denied = "The user denied this action."
        if not self.asker:
            return denied
        what = {"write_file": "file writes", "edit_file": "file edits", "make_dir": "new folders",
                "move_path": "moves", "delete_path": "deletions", "run_command": "commands",
                "call_agent": "helpers"}[name]
        choice, note = self.asker(question, ["Yes", f"Yes, and don't ask again for {what} this session",
                                             "No, and tell the AI what to do instead"], 2)
        if choice == 0:
            out(dim("  └ allowed"))
            return True
        if choice == 1:
            self.always.add(name)
            out(dim(f"  └ allowed (won't ask again for {what})"))
            return True
        out(yellow("  └ denied") + (dim(f" — {note}") if note else ""))
        return denied + (f" The user says: {note}" if note else " Do not retry it; ask the user how to proceed.")

    @staticmethod
    def show_diff(old, new):
        a, b = old.splitlines(), new.splitlines()
        groups = list(difflib.SequenceMatcher(None, a, b, autojunk=False).get_grouped_opcodes(2))
        if not groups:
            out(dim("    (no changes)"))
            return
        added = sum(j2 - j1 for g in groups for t, i1, i2, j1, j2 in g if t in ("replace", "insert"))
        removed = sum(i2 - i1 for g in groups for t, i1, i2, j1, j2 in g if t in ("replace", "delete"))
        spans = [(i1 + 1, max(i2, j2 - j1 + i1)) for g in groups for t, i1, i2, j1, j2 in g if t != "equal"]
        where = ", ".join(f"{x}-{y}" if y > x else f"{x}" for x, y in spans)
        out(dim(f"    lines {where}:  ") + green(f"+{added}") + " " + red(f"-{removed}"))
        shown = 0
        for gi, g in enumerate(groups):
            if gi:
                out(dim("      ⋮"))
            for tag, i1, i2, j1, j2 in g:
                if tag == "equal":
                    rows = [(dim, f"{j1 + k - i1 + 1:>5}   {a[k]}") for k in range(i1, i2)]
                else:
                    rows = [(red, f"{k + 1:>5} - {a[k]}") for k in range(i1, i2)] + \
                           [(green, f"{k + 1:>5} + {b[k]}") for k in range(j1, j2)]
                for color, text in rows:
                    if shown >= 80:
                        out(dim("    … (more changes not shown)"))
                        return
                    out("  " + color(text[:220]))
                    shown += 1

    # ---------------- tools

    def t_list_dir(self, path=".", depth=1):
        base = self.resolve(path)
        if not base.is_dir():
            raise ToolError(f"Not a directory: {path}")
        depth = max(1, min(int(depth or 1), 3))
        lines, count = [], 0

        def walk(d, level, indent):
            nonlocal count
            try:
                entries = sorted(d.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
            except PermissionError:
                lines.append(indent + "(permission denied)")
                return
            for e in entries:
                if count >= 300:
                    return
                count += 1
                if e.is_dir():
                    lines.append(f"{indent}{e.name}/")
                    if level < depth and e.name not in SKIP_DIRS:
                        walk(e, level + 1, indent + "  ")
                else:
                    try:
                        size = e.stat().st_size
                    except OSError:
                        size = 0
                    lines.append(f"{indent}{e.name}  ({_size(size)})")

        walk(base, 1, "")
        if count >= 300:
            lines.append("… (more entries not shown)")
        return f"{short_path(base)}/\n" + ("\n".join(lines) or "(empty)")

    def t_read_file(self, path, start_line=1, end_line=None):
        f = self.resolve(path)
        if not f.is_file():
            raise ToolError(f"File not found: {path}")
        if f.suffix.lower() == ".pdf":
            if not shutil.which("pdftotext"):
                raise ToolError("pdftotext is not installed")
            text = subprocess.run(["pdftotext", "-layout", str(f), "-"], capture_output=True,
                                  text=True, timeout=60).stdout
        else:
            raw = f.read_bytes()
            if b"\0" in raw[:8192]:
                raise ToolError(f"{path} is a binary file ({_size(len(raw))})")
            text = raw.decode("utf-8", errors="replace")
        lines = text.splitlines()
        total = len(lines)
        start = max(1, int(start_line or 1))
        end = min(total, int(end_line) if end_line else start + MAX_READ_LINES - 1, start + MAX_READ_LINES - 1)
        body, chars, last = [], 0, start - 1
        for i in range(start - 1, end):
            line = lines[i] if len(lines[i]) <= 1000 else lines[i][:1000] + " …"
            chars += len(line)
            if chars > MAX_TOOL_CHARS:
                break
            body.append(f"{i + 1:>5}| {line}")
            last = i + 1
        head = f"{self.rel(f)} (lines {start}-{last} of {total})"
        tail = f"\n[{total - last} more lines — call read_file with start_line={last + 1}]" if last < total else ""
        return head + "\n" + "\n".join(body) + tail

    def t_write_file(self, path, content):
        f = self.resolve(path, write=True)
        if f.is_dir():
            raise ToolError(f"{path} is a directory")
        old = f.read_text(errors="replace") if f.exists() else None
        if old is None:
            n = len(content.splitlines())
            out(dim(f"    new file, {n} lines"))
            for k, line in enumerate(content.splitlines()[:20], 1):
                out("  " + green(f"{k:>5} + {line}"[:220]))
            if n > 20:
                out(dim(f"    … {n - 20} more lines"))
        else:
            self.show_diff(old, content)
        verb = "Create" if old is None else "Overwrite"
        ok = self.approve("write_file", f"{verb} {self.rel(f)}?")
        if ok is not True:
            return ok
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(content)
        return (f"{'Created' if old is None else 'Wrote'} {self.rel(f)} ({len(content.splitlines())} lines)"
                + _dup_warning(old or "", content))

    def t_edit_file(self, path, old_text, new_text, replace_all=False):
        f = self.resolve(path, write=True)
        if not f.is_file():
            raise ToolError(f"File not found: {path}")
        text = f.read_text(errors="replace")
        if re.match(r"^\s*\d+\| ", old_text):     # model copied read_file's line numbers
            strip_nums = lambda t: re.sub(r"(?m)^\s*\d+\| ?", "", t)
            old_text, new_text = strip_nums(old_text), strip_nums(new_text)
        n = text.count(old_text)
        if n == 0 and not replace_all:
            fixed = _loose_replace(text, old_text, new_text)
            if fixed is not None:
                n, old_text, new_text = 1, *fixed
        if n == 0:
            hint = ""
            first = old_text.strip().splitlines()[0].strip() if old_text.strip() else ""
            if first and first in text:
                ln = text[:text.index(first)].count("\n") + 1
                hint = f" A similar line exists at line {ln}; re-read the file and copy the text exactly (whitespace matters)."
            raise ToolError(f"old_text not found in {path}.{hint}")
        if n > 1 and not replace_all:
            raise ToolError(f"old_text appears {n} times in {path}; include more surrounding lines to make it unique, or set replace_all.")
        if not replace_all:
            new_text, fixed = _drop_repeated_tail(text, old_text, new_text)
            if fixed:
                out(yellow(f"  ! the AI re-typed {fixed} existing line{'s' if fixed > 1 else ''} after the edit — removed the duplicate"))
        new = text.replace(old_text, new_text) if replace_all else text.replace(old_text, new_text, 1)
        self.show_diff(text, new)
        lost = _lost_lines(old_text, new_text)
        if lost:
            out(yellow(f"  ! this edit DELETES {len(lost)} line{'s' if len(lost) > 1 else ''}: ")
                + " · ".join(_brief(l, 50) for l in lost[:4]))
        question = f"Apply edit to {self.rel(f)}?" + (f" (it deletes {len(lost)} line{'s' if len(lost) > 1 else ''})" if lost else "")
        ok = self.approve("edit_file", question)
        if ok is not True:
            return ok
        f.write_text(new)
        note = ("\nNOTE: this edit deleted these lines: " + " | ".join(lost[:6]) +
                ". If that was not intended, restore them now.") if lost else ""
        return (f"Edited {self.rel(f)} ({n if replace_all else 1} replacement{'s' if replace_all and n > 1 else ''})"
                + _dup_warning(text, new) + note)

    def _walk_files(self, base):
        if base.is_file():
            yield base
            return
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for fn in filenames:
                yield Path(dirpath) / fn

    def t_find_files(self, pattern, path="."):
        base = self.resolve(path)
        hits = []
        for f in self._walk_files(base):
            if fnmatch.fnmatch(f.name.lower(), pattern.lower()) or fnmatch.fnmatch(str(f.relative_to(base)), pattern):
                hits.append(self.rel(f))
                if len(hits) >= 200:
                    break
        if not hits:
            return f"No files matching '{pattern}' in {self.rel(base)}"
        return "\n".join(hits) + ("\n… (stopped at 200 results)" if len(hits) >= 200 else "")

    def t_search_in_files(self, pattern, path=".", file_glob=None):
        base = self.resolve(path)
        try:
            rx = re.compile(pattern, re.I)
        except re.error:
            rx = re.compile(re.escape(pattern), re.I)
        hits = []
        for f in self._walk_files(base):
            if file_glob and not fnmatch.fnmatch(f.name, file_glob):
                continue
            try:
                if f.stat().st_size > 2_000_000:
                    continue
                raw = f.read_bytes()
            except OSError:
                continue
            if b"\0" in raw[:4096]:
                continue
            for i, line in enumerate(raw.decode("utf-8", errors="replace").splitlines(), 1):
                if rx.search(line):
                    hits.append(f"{self.rel(f)}:{i}: {line.strip()[:200]}")
                    if len(hits) >= 100:
                        return "\n".join(hits) + "\n… (stopped at 100 matches)"
        return "\n".join(hits) if hits else f"No matches for '{pattern}'"

    def t_make_dir(self, path):
        d = self.resolve(path, write=True)
        if d.exists():
            return f"{self.rel(d)} already exists"
        ok = self.approve("make_dir", f"Create folder {self.rel(d)}?")
        if ok is not True:
            return ok
        d.mkdir(parents=True)
        return f"Created folder {self.rel(d)}"

    def t_move_path(self, src, dst):
        s, d = self.resolve(src, write=True), self.resolve(dst, write=True)
        if not s.exists():
            raise ToolError(f"Not found: {src}")
        ok = self.approve("move_path", f"Move {self.rel(s)} → {self.rel(d)}?")
        if ok is not True:
            return ok
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(s), str(d))
        return f"Moved {self.rel(s)} → {self.rel(d)}"

    def t_delete_path(self, path):
        p = self.resolve(path, write=True)
        if not p.exists():
            raise ToolError(f"Not found: {path}")
        if p == self.root or p == self.cwd:
            raise ToolError("Refusing to delete the root or working directory")
        ok = self.approve("delete_path", f"Delete {self.rel(p)}{'/ (folder)' if p.is_dir() else ''}? (goes to ~/.agent/trash)")
        if ok is not True:
            return ok
        dest = TRASH_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}_{p.name}"
        shutil.move(str(p), str(dest))
        return f"Deleted {self.rel(p)} (recoverable in {short_path(dest)})"

    def t_run_command(self, command, timeout=None):
        timeout = int(timeout or self.cfg["command_timeout"])
        out(magenta("• ") + bold("Run ") + cyan(command))
        ok = self.approve("run_command", "Run this command?")
        if ok is not True:
            return ok
        proc = subprocess.Popen(command, shell=True, cwd=self.cwd, executable="/bin/bash",
                                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, errors="replace",
                                start_new_session=True)
        self.proc = proc
        timed_out = []
        timer = threading.Timer(timeout, lambda: (timed_out.append(1), _killpg(proc)))
        timer.start()
        lines = []
        try:
            for line in proc.stdout:
                lines.append(line)
                limit = 10_000 if self.cfg["verbose"] else 8
                if len(lines) <= limit:
                    out("    " + dim(line.rstrip()[:220]))
                elif len(lines) == limit + 1:
                    out(dim("    … (more output — ctrl+o to see it all afterwards)"))
            proc.wait()
        except KeyboardInterrupt:
            _killpg(proc)
            raise
        finally:
            timer.cancel()
            self.proc = None
        self.check_cancel()
        text = "".join(lines)
        if len(text) > MAX_TOOL_CHARS:
            text = f"[… first {len(text) - MAX_TOOL_CHARS} chars cut]\n" + text[-MAX_TOOL_CHARS:]
        status = f"killed after {timeout}s timeout" if timed_out else f"exit code {proc.returncode}"
        return f"{status}\n{text}" if text else status

    def t_web_search(self, query, max_results=8):
        try:
            results = ddg_search(query, int(max_results or 8))
        except requests.RequestException as e:
            raise ToolError(f"Search failed: {e}")
        if not results:
            return "No results."
        return "\n\n".join(f"{i}. {t}\n   {u}\n   {s}" for i, (t, u, s) in enumerate(results, 1))

    def t_fetch_url(self, url, offset=0):
        if not re.match(r"^https?://", url):
            url = "https://" + url
        try:
            r = requests.get(url, headers={"User-Agent": UA}, timeout=30)
            r.raise_for_status()
        except requests.RequestException as e:
            raise ToolError(f"Could not fetch {url}: {e}")
        ctype = r.headers.get("content-type", "")
        title = ""
        if "pdf" in ctype or url.lower().endswith(".pdf"):
            tmp = CONF_DIR / "fetch.pdf"
            tmp.write_bytes(r.content)
            text = subprocess.run(["pdftotext", "-layout", str(tmp), "-"], capture_output=True, text=True).stdout
            tmp.unlink(missing_ok=True)
        elif "html" in ctype:
            title, text = html_to_text(r.text)
        else:
            text = r.text
        offset = int(offset or 0)
        chunk = text[offset:offset + MAX_TOOL_CHARS]
        more = len(text) - offset - len(chunk)
        head = f"{title}\n{url}\n" if title else f"{url}\n"
        tail = f"\n[{more} more characters — call fetch_url with offset={offset + len(chunk)}]" if more > 0 else ""
        return head + chunk + tail

    def t_search_knowledge(self, query, wiki=None):
        hits = KNOWLEDGE.search(query, k=4, wiki=wiki or None)
        if not hits:
            wikis = ", ".join(KNOWLEDGE.listing()) or "none installed"
            return f"No matching section. Wikis: {wikis}. Try other key terms (in English) or no wiki filter."
        return KNOWLEDGE_NOTE + "\n\n" + "\n\n".join(Knowledge.render(sec, 3000) for _, sec in hits)

    def t_update_plan(self, plan):
        self.plan = plan.strip()
        if not self.helper:
            out(hexc(agent_color(None), "● ") + hexc(agent_color(None), "chicken", True) + dim("  plan"))
        if self.task is not None and not self.helper:
            steps = []
            for line in self.plan.splitlines():
                m = re.match(r"^\s*(?:[-*]\s*)?(?:\[([ xX])\]\s*)?(?:(\d+)[.)]\s*)?(.+)$", line)
                if m and m.group(3).strip():
                    steps.append({"id": int(m.group(2)) if m.group(2) else len(steps) + 1, "task": m.group(3).strip(),
                                  "status": "done" if (m.group(1) or "").lower() == "x" else "pending"})
            old = {st["id"]: st for st in self.task["plan"]}
            merged = []
            for st in steps:                 # keep what the program already knows (model, running, needs_review…)
                m = {**old.get(st["id"], {}), "id": st["id"], "task": st["task"]}
                m["status"] = "done" if st["status"] == "done" else m.get("status", "pending")
                merged.append(m)
            self.task["plan"] = merged
            self.save_task()
        for line in self.plan.splitlines():
            m = re.match(r"^\s*(?:[-*]|\d+[.)])?\s*\[([ xX])\]\s*(.*)", line)
            if m and m.group(1).lower() == "x":
                out("    " + green("✓ ") + dim(m.group(2)))
            elif m:
                out("    " + dim("○ ") + m.group(2))
            elif line.strip():
                out("    " + line.strip())
        return "Plan saved."

    def t_task_complete(self, summary, status="done"):
        self.completed = (status or "done", summary)
        return "Task marked as complete."

    # ---------------- tool dispatch

    def exec_tool(self, name, args):
        fn = getattr(self, "t_" + name, None)
        if fn is None or name not in {t["function"]["name"] for t in self.tools()}:
            out(red(f"• unknown tool {name}"))
            return f"Error: unknown tool '{name}'"
        if isinstance(args, str):
            try:
                args = json.loads(args or "{}")
            except json.JSONDecodeError:
                return "Error: tool arguments must be a JSON object"
        args = {k: v for k, v in (args or {}).items() if v is not None}
        doing, done = describe(name, args)
        bullet = magenta("• ")
        if name in RISKY_TOOLS and name != "run_command":
            out(bullet + bold(done))
        sig = name + json.dumps(args, sort_keys=True)
        self.recent_calls = (self.recent_calls + [sig])[-3:]
        try:
            with ui.busy(doing):
                result = fn(**args)
        except ToolError as e:
            result = f"Error: {e}"
        except TypeError as e:
            result = f"Error: bad arguments for {name}: {e}"
        except (KeyboardInterrupt, Interrupted):
            raise
        except Exception as e:
            result = f"Error: {type(e).__name__}: {e}"
        self.last_output = (done, result)
        if name in WRITE_TOOLS | {"move_path"} and not result.startswith(("Error", "The user denied")):
            try:
                self.touched.add(self.resolve(args.get("path") or args.get("dst", "")))
            except (ToolError, TypeError):
                pass
        if name not in RISKY_TOOLS and name not in ("update_plan", "task_complete", "call_agent"):
            summary = summarize_result(name, result)
            out(bullet + bold(done) + (dim(" · " + summary) if summary else ""))
        if result.startswith("Error"):
            out(red("  └ " + result.splitlines()[0][:200]))
        elif name == "run_command":
            code = result.splitlines()[0]
            out(dim("  └ ") + (green(code) if code == "exit code 0" else red(code)))
        elif name in RISKY_TOOLS and not result.startswith("The user denied"):
            out(dim("  └ " + result.splitlines()[0][:200]))
        if self.cfg["verbose"] and name in ("read_file", "list_dir", "find_files", "search_in_files",
                                            "web_search", "fetch_url") and not result.startswith("Error"):
            body = result.splitlines()[1:31]
            for line in body:
                out(dim("  │ " + line[:220]))
            if result.count("\n") > 31:
                out(dim("  │ … (ctrl+o shows everything)"))
        if len(self.recent_calls) == 3 and len(set(self.recent_calls)) == 1:
            result += "\n\nNote: you have made this exact same call 3 times in a row. Stop repeating it and try a different approach."
        return clip(result, MAX_TOOL_CHARS + 500)

    # ---------------- model

    def system_prompt(self):
        now = datetime.datetime.now().strftime("%A %d %B %Y")   # no clock time: it would break the prompt cache every minute
        if self.helper:
            r = self.helper
            return f"""{r['prompt']}

You are the helper "{r['name']}", working for Chicken, the main agent on the user's Linux computer. Chicken cannot see your work, only your final message: make it the complete report.
Date: {now}
Working directory: {self.cwd}. You can only access files under {self.root}.
- Use tools to look at things instead of guessing. Relative paths are relative to the working directory.
- Actions that change things are shown to the user for approval. If one is denied, do not retry it.
- Do only this task, quickly. Reply in English. Never use emoji.""" + WORKER_RULES
        if self.orchestrating():
            return self.orchestrator_prompt(now)
        p = (system_template().replace("{date}", now).replace("{cwd}", str(self.cwd))
             .replace("{root}", str(self.root)).rstrip())
        if self.focus:
            p += "\n\nThe user is currently working on these files: " + ", ".join(self.rel(f) for f in self.focus)
        roles = load_roles() if self.cfg["agents_mode"] != "manual" else {}
        if roles:
            p += ("\n\nHelpers: you can hand a self-contained step to a helper agent with call_agent. A helper starts "
                  "with an empty memory (it cannot see this conversation). It gets your task, the files you list and "
                  "the previous helper's result, and returns a short result.\n"
                  + "".join(f"- {n}: {r['description']}\n" for n, r in roles.items()) +
                  "- When the user names a helper (\"use the coder\", \"have the reviewer check\"), you MUST call that "
                  "helper with call_agent for that part. Never do it yourself instead.\n"
                  "- Write each task so it stands alone: the goal, exact file paths, constraints, what to return.\n"
                  "- After each result, check it and decide: next step, the same helper again with what to fix, or finish.\n"
                  "- For writing or changing code of a feature or a whole file, prefer the coder. Do trivial things "
                  "yourself (reading one file, a one-line edit, answering from what you know).")
            if self.cfg["agents_mode"] == "ask":
                p += "\n- The user approves each helper call; if one is denied, do the step yourself or ask."
        if self.wiki:
            p += (f"\n\nProject wiki: {self.wiki} (index.md lists every page). When you need details about a file "
                  "or past decisions, read its wiki page with read_file first; open the real file only to edit it "
                  "or when the page is not enough.")
        if self.plan:
            p += f"\n\nYour current plan:\n{self.plan}"
        if self.loop_goal:
            p += f"""

AUTONOMOUS MODE. Goal: {self.loop_goal}
- Work on your own until the goal is fully achieved. Do not stop to ask the user questions: make reasonable decisions and note them.
- Keep the plan updated with update_plan, marking finished steps [x].
- Verify your work (re-read changed files, run the code or tests when it makes sense).
- When everything is done and verified, call task_complete with a summary. If you are truly blocked, call task_complete with status "blocked" and explain why."""
        return p

    def orchestrator_prompt(self, now):
        c = self.cfg
        p = ORCHESTRATOR_PROMPT.format(date=now, cwd=self.cwd, root=self.root, mode=c.get("mode"),
                                       routing=c.get("routing"), heavy=c["heavy_model"])
        if c.get("routing") in ("bonsai-only", "bonsai-workers"):
            p += f"\n- Routing {c['routing']}: every worker runs on {c['heavy_model']}, whatever model you name."
        p += "\n\nModels you can choose (each runs fully on the GPU, one at a time):\n" + "".join(
            f"- {m}: {caps} · intelligence {iq} · {speed} · {self.vram.need(m, 8192):.1f} GB\n"
            for m, (caps, iq, speed) in self.vram.eligible())
        p += "\nRoles (the tools and behaviour of a worker; you choose its model):\n"
        for n, r in load_roles().items():
            tools = ", ".join(r["tools"]) if r["kind"] == "tool" else "none (gets the files, answers)"
            p += f"- {n}: {r['description']} · tools: {tools or 'none'} · default model {role_model(r, c)}\n"
        wikis = KNOWLEDGE.listing()
        if wikis:
            p += ("\nKnowledge wikis (the relevant sections are added to each worker's task automatically, and tool "
                  "workers can search them):\n" + "".join(f"- {w}: {d}\n" for w, d in wikis.items()))
        if self.task:
            t = {k: v for k, v in self.task.items() if k not in ("calls", "fails")}
            p += "\nTask state (authoritative; your memory of this task):\n```json\n" + json.dumps(
                t, ensure_ascii=False)[:6000] + "\n```"
        if self.plan:
            p += f"\n\nCurrent plan:\n{self.plan}"
        if self.loop_goal:
            p += (f"\n\nAUTONOMOUS MODE. Goal: {self.loop_goal}\nKeep launching workers until the goal is achieved "
                  "and verified, then call task_complete. If truly blocked, call task_complete with status blocked.")
        return p

    def orchestrator_tools(self):
        roles = load_roles()
        models = [m for m, _ in self.vram.eligible()]
        call = _fn("call_agent", "Launch exactly ONE worker for one step. It starts with an empty memory, does the step, "
                   "returns a structured report and exits.",
                   {"role": {"type": "string", "enum": list(roles), "description": "Which worker role"},
                    "model": {"type": "string", "enum": models or [self.cfg["model"]],
                              "description": "The model the worker runs on (cheapest likely to succeed)"},
                    "task": S("Complete, stand-alone instructions: goal, exact paths, constraints, what to return"),
                    "reason": S("One short sentence: why this role and model"),
                    "complexity": {"type": "string", "enum": ["trivial", "low", "medium", "high", "frontier"]},
                    "step": {"type": "integer", "description": "The plan step this call does (if there is a plan)"},
                    "files": {"type": "array", "items": {"type": "string"},
                              "description": "Paths of the files the worker needs (also new files to create)"},
                    "use_previous": B("Give it the previous worker's result (default true)")},
                   ["role", "task", "reason"])
        plan = next(t for t in TOOLS if t["function"]["name"] == "update_plan")
        finish = _fn("finish", "The goal is done (or can't be done): end the task with a short summary for the user.",
                     {"summary": S("What was done, files changed, open issues. In the user's language.")}, ["summary"])
        return [call, plan, finish] + ([TASK_COMPLETE] if self.loop_goal else [])

    def t_finish(self, summary):
        st = {"code": False}
        for line in summary.strip().splitlines():
            out(render_md(no_emoji(line), st))
        if self.task:
            self.task["status"] = "complete"
            self.save_task()
        self.completed = ("done", summary)
        self.finished_by_tool = True
        return "Task finished."

    def tools(self):
        if self.orchestrating() and self.cfg["agents_mode"] != "manual":
            return self.orchestrator_tools()
        if self.helper:
            names = {n for g in self.helper["tools"] for n in TOOL_GROUPS[g]} | {"search_knowledge"}
            return [t for t in TOOLS if t["function"]["name"] in names]
        extra = [TASK_COMPLETE] if self.loop_goal else []
        if self.cfg["agents_mode"] != "manual":
            roles = load_roles()
            if roles:
                extra.append(_fn(
                    "call_agent", "Hand ONE self-contained step to a helper agent with an empty memory. It returns a "
                    "short result. Helpers: " + "; ".join(f"{n} = {r['description']}" for n, r in roles.items()),
                    {"role": {"type": "string", "enum": list(roles), "description": "Which helper"},
                     "task": S("Complete, stand-alone instructions: goal, exact paths, constraints, what to return"),
                     "files": {"type": "array", "items": {"type": "string"},
                               "description": "Paths of the files the helper needs (also new files to create)"},
                     "use_previous": B("Give it the previous helper's result (default true)")},
                    ["role", "task"]))
        return TOOLS + extra

    def chat(self, messages, tools=None, think=None, quiet=False, label=None, on_text=None):
        think = self.think if think is None else think
        if self.orchestrating():
            think = False                      # the orchestrator only decides: short structured answers
        model = self.cfg["model"]
        self.check_cancel()
        self.vram.cancel = self.cancel
        if not self.helper:
            self.cfg["num_ctx"] = self.full_ctx()          # the main agent alone: its full context
        self.vram.ensure(model, self.cfg["num_ctx"])      # load it, swapping others out if the GPU is full
        self.guard_vram({model, self.vram.norm(model)})
        server = self.vram.servers.spec(model)
        text_p, think_p = StreamPrinter("text"), StreamPrinter("think", agent_color(self.helper))
        content, calls = "", []
        think_span = [None, None]          # first and last reasoning chunk: how long this call spent thinking
        ui.reset_rate()
        with ui.busy(label or ("Summarizing" if quiet else "Thinking")):
            try:
                r, events = (self._open_server(model, messages, tools, think) if server
                             else self._open_ollama(messages, tools, think))
            except _ThinkUnsupported:
                self.think = False
                return self.chat(messages, tools, False, quiet, label, on_text)
            self.response = r
            try:
                for ev in events:
                    if self.cancel.is_set():
                        raise Interrupted()
                    if "think" in ev:
                        ui.tick()
                        now = time.time()
                        think_span[0] = think_span[0] or now
                        think_span[1] = now
                        if not quiet and self.cfg["verbose"]:
                            think_p.feed(ev["think"])
                    if "text" in ev:
                        ui.tick()
                        content += ev["text"]
                        if on_text:
                            on_text(len(content))
                        if not quiet:
                            if think_p.buf or think_p.started:
                                think_p.flush()
                                think_p.started = False
                            ui.set_label("Writing")
                            text_p.feed(ev["text"])
                    if "tool_part" in ev:
                        ui.tick()
                        tool, n = ev["tool_part"]
                        what = {"write_file": "the file", "edit_file": "the edit", "run_command": "the command",
                                "update_plan": "the plan", "call_agent": "the helper's task"}.get(tool, tool or "an action")
                        ui.set_label(f"Writing {what} · {n:,} characters")
                    if "calls" in ev:
                        ui.tick()
                        calls.extend(ev["calls"])
                    if "stats" in ev:
                        prompt_n, gen_n, tps = ev["stats"]
                        self.last_tokens = prompt_n + gen_n
                        use = self.main_agent().usage
                        use["in"] += prompt_n
                        use["out"] += gen_n
                        if tps:
                            self.last_tps = tps
                            self.turn_tokens = getattr(self, "turn_tokens", 0) + gen_n
                if think_span[0]:
                    self.main_agent().usage["think"] += think_span[1] - think_span[0]
                if not quiet:
                    think_p.flush()
                    text_p.flush()
            except Exception as e:
                if not (isinstance(e, (Interrupted, KeyboardInterrupt)) or self.cancel.is_set()):
                    raise
                r.close()
                text_p.flush()
                if not quiet and content:
                    self.messages.append({"role": "assistant", "content": content + "\n[interrupted by the user]"})
                raise Interrupted()
            finally:
                self.response = None
        if tools and not calls and "<tool_call>" in content:
            content, calls = parse_text_tool_calls(content, {t["function"]["name"] for t in tools})
        return content.strip(), calls

    def _open_ollama(self, messages, tools, think):
        payload = {
            "model": self.cfg["model"], "messages": messages, "stream": True, "keep_alive": "30m",
            "options": {**{k: self.cfg[k] for k in ("num_ctx", "temperature", "top_p", "top_k")},
                        "num_predict": self.cfg["max_answer_tokens"]},
        }
        if tools:
            payload["tools"] = tools
        if think:
            payload["think"] = True
        elif "qwen3" in self.cfg["model"] or "deepseek-r1" in self.cfg["model"]:
            payload["think"] = False
        try:
            r = requests.post(self.cfg["host"] + "/api/chat", json=payload, stream=True, timeout=(10, 900))
        except requests.ConnectionError:
            raise RuntimeError(f"Cannot reach Ollama at {self.cfg['host']}. Start it with: sudo systemctl start ollama")
        if r.status_code != 200:
            try:
                err = r.json().get("error", r.text)
            except ValueError:
                err = r.text
            if "does not support thinking" in err and think:
                raise _ThinkUnsupported()
            raise RuntimeError(f"Ollama error: {err}")

        def events():
            for line in r.iter_lines():
                if not line:
                    continue
                d = json.loads(line)
                if d.get("error"):
                    raise RuntimeError(f"Ollama error: {d['error']}")
                m = d.get("message", {})
                if m.get("thinking"):
                    yield {"think": m["thinking"]}
                if m.get("content"):
                    yield {"text": m["content"]}
                if m.get("tool_calls"):
                    yield {"calls": m["tool_calls"]}
                if d.get("done"):
                    tps = d["eval_count"] / (d["eval_duration"] / 1e9) if d.get("eval_duration") else 0
                    yield {"stats": (d.get("prompt_eval_count", 0), d.get("eval_count", 0), tps)}
        return r, events()

    def _open_server(self, model, messages, tools, think):
        """A llama.cpp server (OpenAI API). Sampling follows the model card: thinking vs non-thinking mode."""
        body = {"messages": to_openai(messages), "stream": True, "reasoning_effort": "medium" if think else "none",
                "max_tokens": max(self.cfg["max_answer_tokens"], 16384) if think else self.cfg["max_answer_tokens"],
                "top_k": 20}
        body.update({"temperature": 1.0, "top_p": 0.95, "min_p": 0.05} if think else
                    {"temperature": 0.7, "top_p": 0.8, "min_p": 0.0, "presence_penalty": 1.5})
        if tools:
            body["tools"] = tools
        url = self.vram.servers.url(model)
        try:
            r = requests.post(url + "/v1/chat/completions", json=body, stream=True, timeout=(10, 900))
        except requests.ConnectionError:
            raise RuntimeError(f"Cannot reach the {model} server at {url}")
        if r.status_code != 200:
            raise RuntimeError(f"{model} server error {r.status_code}: {r.text[:300]}")

        def events():
            acc = {}                                  # tool calls arrive in pieces: index -> name, arguments
            for raw in r.iter_lines():
                if not raw.startswith(b"data: "):
                    continue
                data = raw[6:]
                if data.strip() == b"[DONE]":
                    break
                d = json.loads(data)
                if d.get("error"):
                    raise RuntimeError(f"{model} error: {d['error']}")
                delta = ((d.get("choices") or [{}])[0].get("delta")) or {}
                if delta.get("reasoning_content"):
                    yield {"think": delta["reasoning_content"]}
                if delta.get("content"):
                    yield {"text": delta["content"]}
                for tc in delta.get("tool_calls") or []:
                    a = acc.setdefault(tc.get("index", 0), {"name": "", "args": "", "id": None})
                    fn = tc.get("function") or {}
                    a["id"] = a["id"] or tc.get("id")
                    a["name"] += fn.get("name") or ""
                    a["args"] += fn.get("arguments") or ""
                    yield {"tool_part": (a["name"], len(a["args"]))}     # the model is writing a tool call: a token
                t = d.get("timings")
                if t:
                    yield {"stats": (t.get("prompt_n", 0) + t.get("cache_n", 0), t.get("predicted_n", 0),
                                     t.get("predicted_per_second", 0))}
            calls = []
            for i in sorted(acc):
                try:
                    args = json.loads(acc[i]["args"] or "{}")
                except ValueError:
                    args = {"_malformed_arguments": acc[i]["args"][:300]}   # exec_tool reports it back to the model
                calls.append({"function": {"name": acc[i]["name"], "arguments": args}})
            if calls:
                yield {"calls": calls}
        return r, events()

    # ---------------- context management

    def estimate_tokens(self):
        key = (len(self.messages), len(self.plan), len(self.focus), bool(self.loop_goal))
        if self._est[0] == key:
            return self._est[1]
        try:
            chars = len(self.system_prompt()) + TOOLS_CHARS
            for m in list(self.messages):
                chars += len(m.get("content") or "") + len(json.dumps(m.get("tool_calls", "")))
        except Exception:
            return self._est[1]
        self._est = (key, int(chars / 3.3))
        return self._est[1]

    def context_used(self):
        return max(self.estimate_tokens(), self.last_tokens)

    def maybe_compact(self, force=False, target=None, reason=None):
        """Summarize older messages. With `target`, make the whole context fit in that many tokens."""
        used = self.context_used()
        if not force and used < 0.72 * self.cfg["num_ctx"]:
            return
        keep_chars = int(0.25 * self.cfg["num_ctx"] * 3.3)
        if target:
            fixed = int((len(self.system_prompt()) + TOOLS_CHARS) / 3.3) + 2500     # instructions, tools, summary
            keep_chars = min(keep_chars, max(0, int((target - fixed) * 3.3)))
        i, acc = len(self.messages), 0
        while i > 0:
            size = len(self.messages[i - 1].get("content") or "")
            if acc + size > keep_chars:
                break
            acc += size
            i -= 1
        while i < len(self.messages) and self.messages[i]["role"] == "tool":
            i += 1
        if i < 2:
            # a single huge exchange: shrink old tool outputs instead
            for m in self.messages[:-1]:
                if m["role"] == "tool" and len(m["content"]) > 2000:
                    m["content"] = m["content"][:2000] + "\n[… output shortened to save memory]"
            return
        old, self.messages = self.messages[:i], self.messages[i:]
        before = 100 * used / self.cfg["num_ctx"]
        transcript = []
        for m in old:
            text = m.get("content") or ""
            if m["role"] == "tool":
                text = clip(text, 1200)
            if m.get("tool_calls"):
                text += "\n[called: " + ", ".join(
                    f"{tc['function']['name']}({_brief(tc['function'].get('arguments'), 150)})" for tc in m["tool_calls"]) + "]"
            transcript.append(f"{m['role'].upper()}: {text}")
        body = "\n\n".join(transcript)
        body = body[-int(0.55 * self.cfg["num_ctx"] * 3.3):]
        prompt = ("Summarize this conversation between a user and an AI agent so the agent can continue the work "
                  "without it. Keep: the user's goals and requests, decisions made, files read/created/changed "
                  "(with paths), important facts and findings, what is done and what is still to do. Be concise "
                  "but complete; use bullet points.\n\n" + body)
        try:
            summary, _ = self.chat([{"role": "user", "content": prompt}], think=False, quiet=True,
                                   label="Compacting the context")
        except Interrupted:
            summary = "(summary interrupted)"
        self.messages.insert(0, {"role": "user", "content": "[Summary of our earlier conversation]\n" + summary})
        self.last_tokens = 0
        after = 100 * self.estimate_tokens() / self.cfg["num_ctx"]
        how = f"compacted {reason}" if reason else "compacted" if force else "auto-compacted"
        out(dim(f"  ↻ context {before:.0f}% → {after:.0f}% · {how}"))

    # ---------------- turns

    def run_turn(self, text, footer=True):
        if self.pending:
            text = "\n\n".join(self.pending) + "\n\n" + text
            self.pending = []
        if getattr(self, "was_interrupted", False):
            text = ("(I interrupted my previous request on purpose. Do not continue it; "
                    "only do what I ask now.)\n\n" + text)
            self.was_interrupted = False
        if not self.helper and self.cfg["agents_mode"] != "manual":
            named = [n for n in load_roles() if re.search(
                rf"\b(?:use|with|ask|have|let|call|via)\s+(?:the\s+|a\s+)?{re.escape(n)}\b|\b{re.escape(n)}\s+helper\b", text, re.I)]
            if named:
                text += "\n\n(Use call_agent with the " + " and the ".join(named) + " helper for this, as I asked.)"
        self.messages.append({"role": "user", "content": text})
        if not self.helper and not self.loop_goal:
            self.run = None            # each request that uses helpers gets its own run folder
            self.task = self.new_task(text) if self.orchestrating() else None
        t0, self.turn_tokens = time.time(), 0
        if not self.helper:
            self.usage = {"in": 0, "out": 0, "think": 0.0}
        rounds, finished = 0, False
        steps0 = len(self.run["steps"]) if self.run else 0
        try:
            while True:
                if self.loop_goal:
                    self.loop_steps += 1
                    if self.loop_steps > self.loop_max:
                        return False
                elif rounds >= self.cfg["max_tool_rounds"]:
                    out(yellow(f"  Stopped after {rounds} tool rounds. Say 'continue' to let it go on."))
                    return False
                rounds += 1
                self.check_cancel()
                self.maybe_compact()
                msgs = [{"role": "system", "content": self.system_prompt()}] + self.messages
                content, calls = self.chat(msgs, self.tools())
                msg = {"role": "assistant", "content": content}
                if calls:
                    msg["tool_calls"] = calls
                self.messages.append(msg)
                if not calls:
                    self.autosave()
                    finished = True
                    if self.task and not self.helper:
                        self.task["status"] = "complete"
                        self.save_task()
                    return rounds > 1
                for tc in calls:
                    self.check_cancel()
                    fn = tc.get("function", {})
                    result = self.exec_tool(fn.get("name", ""), fn.get("arguments", {}))
                    self.messages.append({"role": "tool", "tool_name": fn.get("name", ""), "content": result})
                self.autosave()
                if self.completed:
                    finished = True
                    if getattr(self, "finished_by_tool", False) and not self.loop_goal:
                        self.completed, self.finished_by_tool = None, False
                    return True
        finally:
            if footer and (self.turn_tokens or self.usage["out"]):
                n = self.cfg["num_ctx"]
                k = (len(self.run["steps"]) if self.run else 0) - steps0
                u = self.usage
                stats = (f"{'1 helper · ' if k == 1 else f'{k} helpers · one at a time · ' if k > 1 else ''}"
                         f"{fmt_secs(time.time() - t0)} · thought {fmt_secs(u['think'])} · "
                         f"{fmt_tokens(u['in'] + u['out'])} tokens ({fmt_tokens(u['in'])} in · {fmt_tokens(u['out'])} out) · "
                         f"{self.last_tps:.1f} tok/s · ctx {100 * self.context_used() / n:.0f}%")
                out((green("✓ ") + bold("Done") + "  " + dim(stats)) if finished else dim("✻ " + stats))

    def autosave(self):
        if self.helper:
            return
        try:
            write_log(SESS_DIR / "last.json.gz", self.snapshot())
            (SESS_DIR / "last.json").unlink(missing_ok=True)
        except OSError:
            pass

    def snapshot(self):
        return {"cwd": str(self.cwd), "plan": self.plan, "focus": [str(p) for p in self.focus],
                "wiki": str(self.wiki or ""), "messages": list(self.messages)}

    # ---------------- helpers (multi-agent)

    def t_call_agent(self, role, task, files=None, use_previous=True, model=None, reason="", complexity="", step=None):
        mode = self.cfg["agents_mode"]
        if mode == "manual":
            return "Error: helpers are off (manual mode). Do the step yourself."
        roles = load_roles()
        if role not in roles:
            return f"Error: there is no role '{role}'. Roles: {', '.join(roles)}"
        model = self.pick_model(roles[role], model)
        while True:
            col = roles[role]["color"]
            out(hexc(col, "▸ ") + bold(role) + "  " + _brief(task, 300))
            info = [model] + ([complexity] if complexity else []) + ([f"step {step}"] if step else [])
            out(dim("    " + " · ".join(info) + (f" · {_brief(reason, 160)}" if reason else "")))
            if files:
                out(dim("    files: " + ", ".join(str(f) for f in files)))
            if self.cfg["agents_mode"] != "ask" or self.main_agent().auto or "call_agent" in self.always:
                break
            if not self.asker:
                return "The user denied this action."
            choice, note = self.asker(f"Run the {role} on {model}?", [
                "Yes", "Yes, and don't ask again for workers this session",
                "No, or change it: type a note, /model <name> or /role <name>"], 2)
            if choice == 1:
                self.always.add("call_agent")
            if choice in (0, 1):
                break
            m = re.match(r"\s*/(model|role)\s+(\S+)\s*$", note or "")
            if m and m.group(1) == "role" and m.group(2) in roles:
                role, model = m.group(2), self.pick_model(roles[m.group(2)])
                self.task and self.task["notes"].append(f"user changed the role to {role}")
                continue
            if m and m.group(1) == "model":
                model = self.pick_model(roles[role], m.group(2))
                self.task and self.task["notes"].append(f"user changed the model to {model}")
                continue
            out(yellow("  └ denied") + (dim(f" — {note}") if note else ""))
            return "The user denied this worker." + (f" The user says: {note}" if note else " Ask how to proceed.")
        return self.delegate(role, task, files or [], use_previous, source="boss", announced=True, model=model,
                             reason=reason, complexity=complexity, step=step)

    def delegate(self, name, task, files=(), use_previous=True, source="boss", announced=False, model=None,
                 reason="", complexity="", step=None):
        """Run one step with a helper: step card → GPU swap if needed → helper → result card → back."""
        roles = load_roles()
        if name not in roles:
            raise ToolError(f"There is no helper called '{name}'. Helpers: {', '.join(roles)}")
        role = roles[name]
        model = model or self.pick_model(role)
        ctx = role_ctx(role, {**self.cfg, "model": self.cfg["model"]}) if model == self.cfg["model"] else (
            role["context"] or max(self.cfg["ctx_min"], min(self.vram.fit_ctx(model, self.vram.budget()), 65536)))
        if self.run is None:
            self.run = new_run()
        if self.task is None and not self.helper:
            self.task = self.new_task(task)
        if step and self.task:
            for st in self.task["plan"]:
                if st.get("id") == step:
                    st["status"] = "running"
                    st["assigned_model"] = model
        n = len(self.run["steps"]) + 1
        prev = self.run["steps"][-1] if use_previous and self.run["steps"] else None
        paths = []
        for fp in list(files) + re.findall(r"[\w.~/-]+\.[A-Za-z0-9]{1,5}\b", task):
            try:
                p = self.resolve(fp)
            except ToolError:
                continue
            if p not in paths and (p.is_file() or (fp in files and not p.is_dir())):
                paths.append(p)
        card = [f"# Step {n:02d} · {name}", f"- model: {model} · kind: {role['kind']} · asked by: {source}",
                f"- files: {', '.join(self.rel(p) for p in paths) or '—'}", "", "## Task", task.strip()]
        if prev:
            card += ["", f"## Result of the previous step ({prev['n']:02d} · {prev['role']})", prev["result"]]
        knowledge = self.knowledge_for(task) if not self.helper else ""
        if knowledge:
            card += ["", "## Reference knowledge", knowledge]
        card_text = "\n".join(card)
        stem = f"step-{n:02d}-{name}"
        (self.run["dir"] / f"{stem}.md").write_text(card_text + "\n")
        self.checkpoint("running", n)
        col = role["color"]
        if not announced:
            out(hexc(col, "▸ ") + bold(name) + "  " + _brief(task, 80))
        t0 = time.time()
        main, v = self.cfg["model"], self.vram
        beside = (not self.helper and v.norm(model) != v.norm(main) and v.on_gpu_only(model)
                  and self.make_room(model, ctx, name))
        v.ensure(model, ctx)
        self.guard_vram({model, v.norm(model)})
        if beside and not v.fully_on_gpu(model, main):     # the estimate was off: never run partly on the CPU
            out(dim(f"  ⇄ not enough room after all: ejecting {main}"))
            v.eject(model)
            v.eject(main)
            v.ensure(model, ctx)
        self.check_cancel()
        ui.indent = hexc(col, "│ ")
        self.touched = set()
        try:
            if role["kind"] == "tool":
                result = self._run_tool_helper(role, model, ctx, card_text)
            else:
                result = self._run_writer(role, model, ctx, task, paths, prev, knowledge)
        finally:
            ui.indent = ""
            self.child = None
        secs = time.time() - t0
        result = result.strip() or "(empty result)"
        (self.run["dir"] / f"{stem}.result.md").write_text(f"# Result of step {n:02d} · {name}\n\n{result}\n")
        self.run["steps"].append({"n": n, "role": name, "model": model, "task": task,
                                  "result": clip(result, 6000), "seconds": round(secs)})
        self.checkpoint("step-done", n)
        report = parse_report(result) or {}
        status = str(report.get("status") or ("done" if name != "planner" or report.get("steps") else "needs_review"))
        checks = validate_files(self.touched)
        bad = [f"{self.rel(f)}: {why}" for f, ok, why in checks if not ok]
        if bad and status == "done":
            status = "needs_review"
        icon = green("✓ ") if status == "done" else red("✗ ") if status in ("failed", "blocked") else yellow("! ")
        shown = (report.get("summary") or "") if report else ""
        out(hexc(col, "└ ") + icon + bold(name) + dim(f"  {secs:.0f}s · ") +
            (_brief(shown, 100) if shown and not result.startswith("Changes by the") else result_line(result)))
        for f, ok, why in checks:
            out(dim("    ") + (green("✓ ") if ok else red("✗ ")) + dim(f"{self.rel(f)} {why}"))
        answer = self.record_step(name, model, n, stem, secs, status, report, result, checks, reason, complexity, step)
        if not self.helper:            # grow back right away: the main agent alone with its full context
            self.cfg["num_ctx"] = self.full_ctx()
            self.vram.ensure(self.cfg["model"], self.cfg["num_ctx"])
        return answer

    def record_step(self, name, model, n, stem, secs, status, report, result, checks, reason, complexity, step):
        """Update the task state and the decision log; return the compact result the orchestrator reads."""
        where = f"{short_path(self.run['dir'])}/{stem}.result.md"
        lines = [f"Result of step {n} · {name} on {model} · {secs:.0f}s · status: {status}"]
        t = self.task
        if name == "planner" and report.get("steps"):
            steps = [{"id": s.get("id", i + 1), "task": s.get("description", ""), "role": s.get("preferred_role", ""),
                      "depends_on": s.get("depends_on", []), "status": "pending"} for i, s in enumerate(report["steps"])]
            if t is not None:
                t["plan"] = steps
            self.plan = "\n".join(f"- [ ] {s['id']}. {s['task']}{' (' + s['role'] + ')' if s.get('role') else ''}" for s in steps)
            out(hexc(agent_color(None), "● ") + hexc(agent_color(None), "chicken", True) + dim("  plan"))
            for s in steps:
                out("    " + dim("○ ") + f"{s['id']}. {s['task']}" + dim(f"  {s['role']}"))
            lines.append(f"Plan saved in the task state: {len(steps)} steps. " + report.get("summary", ""))
            lines += [f"{s['id']}. {s['task']} (role: {s['role']}; after: {s['depends_on'] or '-'})" for s in steps]
        elif report:
            lines.append("Summary: " + clip(str(report.get("summary", "")), 2500))
            for key, label in (("artifacts", "Artifacts"), ("important_facts", "Facts"), ("errors", "Errors")):
                if report.get(key):
                    lines.append(f"{label}: " + "; ".join(str(x) for x in report[key])[:800])
            if report.get("blocked_by"):
                lines.append(f"Blocked by: {report['blocked_by']}")
            if report.get("suggested_followup"):
                lines.append(f"Suggested follow-up (not binding): {report['suggested_followup']}")
        else:
            lines.append(clip(result, 4000))
        if checks:
            lines.append("Validation: " + "; ".join(f"{self.rel(f)} {'ok' if ok else 'FAILED: ' + why}"
                                                     for f, ok, why in checks))
        if t is not None:
            if status in ("failed", "blocked"):
                t["fails"][name] = t["fails"].get(name, 0) + 1
                if t["fails"][name] >= 2:
                    lines.append(f"The {name} failed or was blocked twice: escalate to {self.cfg['heavy_model']}.")
            if step:
                for st in t["plan"]:
                    if st.get("id") == step:
                        st["status"] = "done" if status == "done" else status
                self.plan = "\n".join(f"- [{'x' if s['status'] == 'done' else ' '}] {s['id']}. {s['task']}{' (' + s['role'] + ')' if s.get('role') else ''}"
                                      for s in t["plan"])
            arts = []
            for x in [str(a) for a in report.get("artifacts") or []] + [str(f) for f, _, _ in checks]:
                try:
                    arts.append(self.rel(self.resolve(x)))
                except ToolError:
                    arts.append(x)
            t["artifacts"] = sorted(set(t["artifacts"]) | set(arts))
            t["last_result"] = {"agent": name, "model": model, "status": status,
                                "summary": clip(str(report.get("summary") or result), 600)}
            t["errors"] += [str(e) for e in report.get("errors") or []] + [
                f"{self.rel(f)}: {why}" for f, ok, why in checks if not ok]
            t["calls"].append({"n": n, "role": name, "model": model, "status": status, "seconds": round(secs)})
            self.save_task()
        try:
            LOGS_DIR.mkdir(parents=True, exist_ok=True)
            with open(LOGS_DIR / "decisions.jsonl", "a") as f:
                f.write(json.dumps({"timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                                    "goal": (t or {}).get("goal", "")[:300], "mode": self.cfg.get("mode"),
                                    "routing": self.cfg.get("routing"),
                                    "decision": {"role": name, "model": model, "reason": reason,
                                                 "complexity": complexity, "step": step},
                                    "outcome": {"status": status, "elapsed_seconds": round(secs),
                                                "validation_failed": sum(1 for c in checks if not c[1])}},
                                   ensure_ascii=False) + "\n")
        except OSError:
            pass
        lines.append(f"Full report: {where}")
        return "\n".join(lines)

    def _helper(self, role, model, ctx):
        sub = Agent({**self.cfg, "model": model, "num_ctx": ctx, "think": role["think"]}, auto=self.auto)
        sub.helper, sub.always, sub.asker, sub.cancel, sub.vram = role, self.always, self.asker, self.cancel, self.vram
        sub.parent = self                  # approvals follow the main agent's mode, even if it changes mid-step
        sub.touched = self.touched         # files it changes are checked by the validators
        self.child = sub
        return sub

    def _run_tool_helper(self, role, model, ctx, card_text):
        sub = self._helper(role, model, ctx)
        sub.run_turn(card_text, footer=False)
        last = sub.messages[-1] if sub.messages else {}
        if last.get("role") != "assistant" or last.get("tool_calls") or not last.get("content"):
            # it stopped mid-work (step limit): ask for the report now, without tools
            sub.messages.append({"role": "user", "content": "Stop now and write your final report."})
            text, _ = sub.chat([{"role": "system", "content": sub.system_prompt()}] + sub.messages, think=False)
            return text
        return last["content"]

    def knowledge_for(self, task):
        """The wiki sections most relevant to a worker's task (empty if none is clearly relevant)."""
        if not self.cfg.get("knowledge_auto", True):
            return ""
        hits = KNOWLEDGE.search(task, k=3)
        if not hits:
            return ""
        top = hits[0][0]
        hits = [(sc, sec) for sc, sec in hits if sc >= max(10.0, 0.45 * top)]   # only clearly relevant sections (tuned)
        if not hits:
            return ""
        budget, blocks = self.cfg.get("knowledge_chars", 6000), []
        for sc, sec in hits:
            block = Knowledge.render(sec, max(800, budget // len(hits)))
            if sum(len(b) for b in blocks) + len(block) > budget:
                break
            blocks.append(block)
        if not blocks:
            return ""
        out(dim("    ⌕ knowledge: " + ", ".join(f"{sec['wiki']}/{sec['page']} › {sec['head']}"
                                                 for _, sec in hits[:len(blocks)])))
        return KNOWLEDGE_NOTE + "\n\n" + "\n\n".join(blocks)

    def _run_writer(self, role, model, ctx, task, paths, prev, knowledge=""):
        parts = [f"TASK:\n{task.strip()}"]
        if knowledge:
            parts.append("REFERENCE KNOWLEDGE:\n" + knowledge)
        if prev:
            parts.append(f"RESULT OF THE PREVIOUS STEP ({prev['role']}):\n{prev['result']}")
        images, per = [], int(ctx * 3.3 * 0.5) // max(1, len(paths))
        for p in paths:
            rel = self.rel(p)
            if not p.exists():
                parts.append(f"FILE: {rel}\n(does not exist yet: create it)")
            elif p.suffix.lower() in IMAGE_EXT:
                images.append(base64.b64encode(p.read_bytes()).decode())
                parts.append(f"IMAGE: {rel} (attached)")
            else:
                text = self.t_read_file(str(p)) if p.suffix.lower() == ".pdf" else p.read_text(errors="replace")
                if len(text) > per:
                    parts.append(f"FILE: {rel} (large, {len(text.splitlines())} lines: only the start is shown. Change "
                                 f"it with SEARCH/REPLACE blocks, never rewrite it whole)\n```\n{text[:per]}\n```")
                else:
                    parts.append(f"FILE: {rel}\n```\n{text}\n```")
        if role["name"] == "planner":
            parts.append("ROLES THE STEPS CAN USE:\n" + "\n".join(
                f"- {n}: {r['description']}" for n, r in load_roles().items() if n != "planner"))
        system = role["prompt"] + (WRITER_FILES_RULES if role["output"] == "files" else
                                   "" if role["name"] == "planner" else WRITER_TEXT_RULES + WORKER_RULES)
        user = {"role": "user", "content": "\n\n".join(parts)}
        if images:
            user["images"] = images
        sub = self._helper(role, model, ctx)
        name = role["name"]
        text, _ = sub.chat([{"role": "system", "content": system}, user], think=role["think"], quiet=True,
                           label=f"{name} is writing",
                           on_text=lambda n: ui.set_label(f"{name} is writing · {n:,} characters"))
        self.last_tps = sub.last_tps
        if role["output"] != "files":
            st = {"code": False}
            for line in text.splitlines()[:60]:
                out(render_md(line, st))
            return text or "(no answer)"
        return self._apply_writer(role, model, text, paths)

    def _apply_writer(self, role, model, text, paths):
        """Apply a writer's answer to the files (with the usual diff and approval) and describe what changed."""
        edits, notes = parse_writer(text, paths, self.cwd)
        name = role["name"]
        if not edits:
            return (f"The {name} answered without file changes the program could apply. Its answer:\n"
                    + clip(text, 3000))
        before, bad = {}, []
        for e in edits:
            try:
                f = self.resolve(e[0], write=True)
            except ToolError as err:
                bad.append(f"- {e[0]}: {err}")
                continue
            if f not in before:
                before[f] = f.read_text(errors="replace") if f.exists() else None
            if e[1] == "file":
                cur = f.read_text(errors="replace") if f.exists() else ""
                if cur and len(e[2].splitlines()) < 0.5 * len(cur.splitlines()):
                    bad.append(f"- {self.rel(f)}: not applied: the {name} returned {len(e[2].splitlines())} lines for "
                               f"a {len(cur.splitlines())}-line file, so it looks like a partial file. Ask again with "
                               f"SEARCH/REPLACE blocks or the complete file.")
                    out(yellow(f"  ! {self.rel(f)}: the answer looks like a partial file, not applied"))
                    continue
                res = self.exec_tool("write_file", {"path": self.rel(f), "content": e[2]})
            else:
                res = self.exec_tool("edit_file", {"path": self.rel(f), "old_text": e[2], "new_text": e[3]})
            if res.startswith(("Error", "The user denied")):
                bad.append(f"- {self.rel(f)}: {res.splitlines()[0][:240]}")
        done = []
        for f, old in before.items():
            new = f.read_text(errors="replace") if f.exists() else None
            if new != old:
                plus, minus = diff_stats(old, new)
                done.append(f"- {self.rel(f)}: +{plus} −{minus} lines" + (" (new file)" if old is None else ""))
                changed = [l for l in difflib.unified_diff((old or "").splitlines(), (new or "").splitlines(),
                                                           lineterm="", n=0)
                           if l[:1] in "+-" and not l.startswith(("+++", "---"))]
                done += ["  " + l[:160] for l in changed[:40]]
                if len(changed) > 40:
                    done.append(f"  … {len(changed) - 40} more changed lines (read the file for the rest)")
        card = [f"Changes by the {name} ({model}):"] + (done or ["- none"])
        if bad:
            card += ["Not applied:"] + bad
        if notes:
            card.append(f"The {name}'s note: {clip(notes, 600)}")
        return "\n".join(card)

    def checkpoint(self, status, step=None):
        """Save the main agent's state in the run folder, so a stopped job can be resumed."""
        if not self.run or self.helper:
            return
        try:
            write_log(self.run["dir"] / "boss.json.gz", {**self.snapshot(), "status": status, "step": step})
            (self.run["dir"] / "steps.json").write_text(json.dumps(self.run["steps"], indent=1, ensure_ascii=False))
        except OSError:
            pass

    def resume_run(self):
        """Restore the last job stopped while a helper was working. Returns the message to continue with."""
        for d in sorted((d for d in RUNS_DIR.iterdir() if (d / "boss.json.gz").exists()), reverse=True):
            st = read_log(d / "boss.json.gz")
            if st.get("status") == "running":
                break
        else:
            return None
        self.messages = st["messages"]
        self.plan = st.get("plan", "")
        self.focus = [Path(p) for p in st.get("focus", [])]
        try:
            self.cwd = self.resolve(st.get("cwd", str(self.root)))
        except ToolError:
            self.cwd = self.root
        os.chdir(self.cwd)
        steps = json.loads((d / "steps.json").read_text()) if (d / "steps.json").exists() else []
        self.run = {"id": d.name, "dir": d, "steps": steps}
        last = self.messages[-1] if self.messages else {}
        if last.get("tool_calls"):
            for tc in last["tool_calls"]:
                self.messages.append({"role": "tool", "tool_name": tc.get("function", {}).get("name", ""),
                                      "content": "[stopped: the helper did not finish]"})
        write_log(d / "boss.json.gz", {**st, "status": "resumed"})
        card = next(iter(sorted(d.glob(f"step-{st.get('step') or 0:02d}-*.md"))), None)
        what = card.read_text().split("## Task", 1)[-1].strip()[:400] if card else "?"
        return (f"(The job was stopped during step {st.get('step')} while a helper was working. That step's task "
                f"was: {what}\nIts result is missing. Continue the job from there: repeat that step if it is still "
                f"needed.)")

    # ---------------- save / load (log and wiki)

    def load_log(self, path):
        """Resume a saved conversation. Outputs that are stale or can be fetched again are compressed.
        Returns (context tokens before, after)."""
        d = read_log(path)
        before = messages_chars(d["messages"])
        self.messages = prune_messages(d["messages"])
        self.plan = d.get("plan", "")
        self.focus = [Path(p) for p in d.get("focus", [])]
        self.wiki = Path(d["wiki"]) if d.get("wiki") and Path(d["wiki"]).is_dir() else None
        self.cwd = self.resolve(d.get("cwd", str(self.root)))
        os.chdir(self.cwd)
        self.last_tokens = 0
        return int(before / 3.3), int(messages_chars(self.messages) / 3.3)

    def load_wiki(self, name):
        """Start a fresh conversation that knows the project through its wiki: only index.md and chat.md go
        into the context, other pages are read on demand. Returns (pages, context tokens)."""
        d = WIKI_DIR / name
        meta, index = front_matter((d / "index.md").read_text())
        chat = (d / "chat.md").read_text() if (d / "chat.md").exists() else ""
        plan = (d / "plan.md").read_text() if (d / "plan.md").exists() else ""
        text = (f"[Project wiki '{name}' — folder {d}]\n"
                f"This is what we know about the project from earlier sessions. Other pages are listed in the index: "
                f"read them with read_file (e.g. {d}/files/…) only when you need them.\n\n"
                f"--- index.md ---\n{index.strip()}\n\n--- chat.md ---\n{chat.strip()}")
        self.messages = [{"role": "user", "content": text},
                         {"role": "assistant", "content": f"I have the '{name}' project wiki loaded. What should I do next?"}]
        self.plan = "\n".join(l for l in plan.splitlines() if not l.startswith("# ")).strip()
        self.focus = [p for p in (Path(os.path.expanduser(f.strip())) for f in meta.get("focus", "").split(",") if f.strip())
                      if p.exists()]
        self.wiki = d
        self.pending = []
        try:
            self.cwd = self.resolve(os.path.expanduser(meta.get("cwd") or str(self.root)))
        except ToolError:
            self.cwd = self.root
        os.chdir(self.cwd)
        self.last_tokens = 0
        pages = len(list(d.rglob("*.md")))
        return pages, int(messages_chars(self.messages) / 3.3)

    def touched_files(self, messages, limit):
        """Files of this conversation, most important first: changed, then read, then loaded with /file."""
        changed, read = [], []
        for m in messages:
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function", {})
                name, a = fn.get("name", ""), _args(fn)
                for key in (("dst",) if name == "move_path" else ("path",)):
                    if not a.get(key):
                        continue
                    try:
                        p = self.resolve(str(a[key]))
                    except ToolError:
                        continue
                    (changed if name in CHANGE_TOOLS else read if name == "read_file" else []).append(p)
        seen, files = set(), []
        for p in changed + read + self.focus:
            if p in seen or not p.is_file() or CONF_DIR in p.parents or not is_text_file(p):
                continue
            seen.add(p)
            files.append(p)
        return files[:limit], set(changed)

    def file_history(self, messages, path):
        rel, base, notes = str(path), path.name, []
        for m in messages:
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function", {})
                a = _args(fn)
                target = str(a.get("path") or a.get("dst") or "")
                if not target or os.path.basename(target) != base:
                    continue
                if fn.get("name") == "edit_file":
                    notes.append(f"- edited: `{_brief(a.get('old_text'), 160)}` → `{_brief(a.get('new_text'), 160)}`")
                elif fn.get("name") == "write_file":
                    notes.append(f"- written ({len(str(a.get('content', '')))} characters)")
                elif fn.get("name") in ("move_path", "delete_path"):
                    notes.append(f"- {fn['name']}: {_brief(a, 160)}")
            if m["role"] == "user" and base in (m.get("content") or ""):
                notes.append("- user asked: " + _brief(m["content"], 300))
        text = "\n".join(notes) or "(only read, not changed)"
        return text[-3000:] if len(text) > 3000 else text

    def save_wiki(self, name, report):
        """Save the conversation as a wiki of linked Markdown pages in ~/.agent/wiki/<name>/.
        report(percent, what) shows live progress. Returns a dict of what was written."""
        d = WIKI_DIR / name
        (d / "files").mkdir(parents=True, exist_ok=True)
        now = time.strftime("%Y-%m-%d %H:%M")
        msgs = list(self.messages)
        if msgs and str(msgs[0].get("content", "")).startswith("[Project wiki '"):
            msgs = msgs[2:]        # the loaded wiki itself + the AI's acknowledgement
        msgs = prune_messages(msgs)
        files, changed = self.touched_files(msgs, self.cfg["wiki_max_files"])
        pages = {p: Path("files") / (str(p.relative_to(self.root)) + ".md") for p in files}
        prev_src = (self.wiki or d) / "chat.md"
        prev = prev_src.read_text() if prev_src.exists() else ""

        body, budget = transcript(msgs), int(0.45 * self.cfg["num_ctx"] * 3.3)
        chunks, cur = [], ""
        for part in body:
            part = clip(part, budget - 100)
            if cur and len(cur) + len(part) > budget:
                chunks.append(cur)
                cur = ""
            cur += part + "\n\n"
        if cur or not chunks:
            chunks.append(cur)
        if prev and (d / "chat.md").exists() and not any(m["role"] == "user" for m in msgs):
            chunks = []            # nothing new said since the wiki was loaded: keep chat.md as it is
        steps = max(len(chunks), 1) + (1 if len(chunks) > 1 else 0) + len(files) + 1
        state = {"done": 0, "written": [], "kept": []}

        def llm(prompt, what, expect):
            base = 100 * state["done"] / steps
            span = 100 / steps
            report(base, what)
            text, _ = self.chat([{"role": "user", "content": prompt}], think=False, quiet=True,
                                label=f"Saving wiki {base:.0f}%",
                                on_text=lambda n: report(base + span * min(n / expect, 0.95), what))
            return text

        def step_done(page, kept=False):
            state["done"] += 1
            (state["kept"] if kept else state["written"]).append(page)
            pct = 100 * state["done"] / steps
            report(pct, str(page))
            out(dim(f"  [{pct:3.0f}%] {'= kept' if kept else '✓'} {page}"))

        try:
            # 1. the conversation → chat.md (long conversations: notes per part, then merged)
            if not chunks:
                step_done("chat.md", kept=True)
            else:
                if len(chunks) > 1:
                    notes = []
                    for n, ch in enumerate(chunks, 1):
                        notes.append(llm(WIKI_NOTES_PROMPT.format(n=n, total=len(chunks)) + ch,
                                         f"reading part {n}/{len(chunks)}", 1500))
                        state["done"] += 1
                    source = "\n\n".join(f"Notes, part {n}:\n{t}" for n, t in enumerate(notes, 1))
                else:
                    source = chunks[0]
                links = ", ".join(f"[{self.rel(p)}]({pg.as_posix()})" for p, pg in pages.items()) or "(no files)"
                merge = ("Merge the PREVIOUS STATE with the new conversation: newer information wins, drop what is obsolete."
                         if prev else "Summarize the conversation.")
                prompt = WIKI_CHAT_PROMPT.format(merge=merge, links=links)
                if prev:
                    prompt += f"\nPREVIOUS STATE (chat.md):\n{front_matter(prev)[1].strip()}\n"
                if self.plan:
                    prompt += f"\nCURRENT PLAN:\n{self.plan}\n"
                chat = llm(prompt + f"\nNEW CONVERSATION:\n{source}", "chat.md", 2500)
                if not chat.lstrip().startswith("#"):
                    chat = "# Chat summary\n\n" + chat
                (d / "chat.md").write_text(f"---\nupdated: {now}\n---\n{chat.strip()}\n\n[index](index.md)\n")
                step_done("chat.md")
            if self.plan:
                (d / "plan.md").write_text(f"# Plan\n\n{self.plan.strip()}\n")

            # 2. one page per file (unchanged files whose page is up to date are kept)
            for p in files:
                page = pages[p]
                dest = d / page
                sha = hashlib.sha1(p.read_bytes()).hexdigest()[:12]
                if dest.exists() and p not in changed and front_matter(dest.read_text())[0].get("sha1") == sha:
                    step_done(page, kept=True)
                    continue
                text = llm(WIKI_FILE_PROMPT.format(rel=self.rel(p), content=file_digest(p),
                                                   history=self.file_history(msgs, p)), str(page), 1800)
                m = re.match(r"\s*DESCRIPTION:\s*(.+)", text)
                desc = m.group(1).strip() if m else _brief(text, 80)
                text = text[m.end():].strip() if m else text.strip()
                up = "../" * (len(page.parts) - 1)
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(f"---\npath: {short_path(p)}\ndescription: {desc}\nsha1: {sha}\nupdated: {now}\n---\n"
                                f"# {short_path(p)}\n\n{text}\n\n[index]({up}index.md) · [chat]({up}chat.md)\n")
                step_done(page)
        finally:
            self.write_wiki_index(d, name, now)
        step_done("index.md")
        self.wiki = d
        size = sum(f.stat().st_size for f in d.rglob("*.md"))
        return {"dir": d, "written": state["written"], "kept": state["kept"], "bytes": size,
                "tokens": int(((d / "index.md").stat().st_size + (d / "chat.md").stat().st_size) / 3.3)}

    def write_wiki_index(self, d, name, now):
        rows = []
        if (d / "chat.md").exists():
            rows.append("- [chat.md](chat.md) — conversation summary: goal, decisions, done, next steps")
        if (d / "plan.md").exists():
            rows.append("- [plan.md](plan.md) — the current checklist")
        for f in sorted((d / "files").rglob("*.md")):
            meta = front_matter(f.read_text())[0]
            rows.append(f"- [{f.relative_to(d).as_posix()}]({f.relative_to(d).as_posix()}) — "
                        f"{meta.get('path', '')}: {meta.get('description', '')}")
        focus = ", ".join(short_path(p) for p in self.focus)
        (d / "index.md").write_text(
            f"---\nwiki: {name}\ncwd: {short_path(self.cwd)}\nfocus: {focus}\nupdated: {now}\n---\n"
            f"# Wiki: {name}\n\nProject memory for {short_path(self.cwd)}. Start from chat.md; open a file page only "
            f"when you need it, and the real file only to edit it.\n\n## Pages\n" + "\n".join(rows) + "\n")

    def run_loop(self, goal, max_steps=None):
        self.loop_goal, self.completed, self.loop_steps = goal, None, 0
        self.run = None
        self.loop_max = max_steps or self.cfg["loop_max_steps"]
        if not self.auto and self.asker and not WRITE_TOOLS <= self.always:
            ch, _ = self.asker("Auto-approve file edits during this loop? (moves, deletions and commands still ask)",
                               ["Yes", "No, ask for every change"], None)
            if ch == 0:
                self.always |= WRITE_TOOLS
        out(cyan(f"Autonomous mode — up to {self.loop_max} steps. Ctrl+C to stop.\n"))
        text = (f"GOAL: {goal}\n\nWork on this autonomously until it is completely done. "
                "Start by exploring what is needed, then call update_plan with your plan.")
        idle = 0
        t0 = time.time()
        try:
            while not self.completed:
                used_tools = self.run_turn(text, footer=False)
                if self.completed:
                    break
                if self.loop_steps > self.loop_max:
                    out(yellow(f"\nReached the step limit ({self.loop_max}). Type /loop continue to keep going."))
                    break
                idle = 0 if used_tools else idle + 1
                if idle >= 3:
                    out(yellow("\nThe AI stopped making progress. Reply to it, or /loop continue."))
                    break
                text = ("Continue working toward the goal. Check your plan and do the next step. "
                        "If everything is done and verified, call task_complete. Do not ask me questions; decide yourself.")
        finally:
            self.loop_goal = None
        if self.completed:
            status, summary = self.completed
            icon = green("✓ Done") if status == "done" else yellow("! Blocked")
            mins = (time.time() - t0) / 60
            out(f"\n{icon} {dim(f'({self.loop_steps} steps, {mins:.1f} min)')}")
            for line in summary.splitlines():
                out(render_md(line, {"code": False}))
            self.completed = None

    def side_question(self, question):
        """/btw: a quick answer from the conversation so far, without tools and without touching history."""
        small = self.cfg.get("btw_model")
        spec = self.vram.servers.spec(small) if small else None
        if spec:
            ctx = spec.get("context", 4096)
            v = self.vram
            if not v.servers.running(small):
                used, _ = v.gpu_mem()
                if used and used + v.need(small, ctx) + PROC_GB > v.budget():
                    return ("(No GPU room for /btw right now: the current step is using the GPU up to the limit. "
                            "Ask again when it finishes.)")
            v.servers.start(small, ctx)                  # ~1 s the first time, then it stays up while there's room
            lines = transcript(list(self.messages), tool_clip=500)
            convo, size = [], 0
            for line in reversed(lines):                 # the most recent part that fits its small memory
                if size + len(line) > int(0.55 * ctx * 3.3):
                    break
                convo.insert(0, line)
                size += len(line)
            system = BTW_PROMPT + f"\n\nWorking directory: {self.cwd}. Main model: {self.cfg['model']}."
            child = self.child
            if child is not None:
                system += f"\nWorking now: the {child.helper['name']} worker on {child.cfg['model']}."
            if self.task:
                snap = {k: self.task.get(k) for k in ("goal", "mode", "routing", "status", "plan", "last_result",
                                                      "artifacts", "errors")}
                system += "\n\nRead-only task snapshot:\n" + json.dumps(snap, ensure_ascii=False)[:5000]
            if self.plan:
                system += f"\n\nChicken's current plan:\n{self.plan}"
            msgs = [{"role": "system", "content": system}]
            if convo:
                msgs.append({"role": "user", "content": "The conversation so far:\n\n" + "\n\n".join(convo)})
            msgs.append({"role": "user", "content": question})
            r = requests.post(self.vram.servers.url(small) + "/v1/chat/completions", timeout=(10, 300), json={
                "messages": msgs, "max_tokens": 600, "temperature": 0.7, "top_p": 0.8, "top_k": 20,
                "presence_penalty": 1.5, "chat_template_kwargs": {"enable_thinking": False}})
            r.raise_for_status()
            return (r.json()["choices"][0]["message"].get("content") or "").strip() or "(no answer)"
        budget = int(0.5 * self.cfg["num_ctx"] * 3.3)
        snap, size = [], 0
        for m in reversed(list(self.messages)):
            n = len(m.get("content") or "") + len(json.dumps(m.get("tool_calls", "")))
            if size + n > budget:
                break
            snap.insert(0, m)
            size += n
        while snap and snap[0]["role"] == "tool":
            snap.pop(0)
        system = (self.system_prompt() + "\n\nSIDE QUESTION (/btw): while you keep working on the main task, the user "
                  "asks something quick. Answer briefly and directly from the conversation and your knowledge. "
                  "Tools are not available for this answer; don't mention them.")
        payload = {"model": self.cfg["model"], "stream": False, "keep_alive": "30m",
                   "messages": [{"role": "system", "content": system}] + snap + [{"role": "user", "content": question}],
                   "options": {**{k: self.cfg[k] for k in ("num_ctx", "temperature", "top_p", "top_k")}, "num_predict": self.cfg["max_answer_tokens"]}}
        if self.vram.servers.spec(self.cfg["model"]):
            if not self.vram.servers.running(self.cfg["model"]):
                return "(the model isn't loaded right now — ask again in a moment)"
            r = requests.post(self.vram.servers.url(self.cfg["model"]) + "/v1/chat/completions", timeout=(10, 900), json={
                "messages": to_openai(payload["messages"]), "reasoning_effort": "none", "temperature": 0.7,
                "top_p": 0.8, "top_k": 20, "max_tokens": self.cfg["max_answer_tokens"]})
            r.raise_for_status()
            return (r.json()["choices"][0]["message"].get("content") or "").strip() or "(no answer)"
        if "qwen3" in self.cfg["model"] or "deepseek-r1" in self.cfg["model"]:
            payload["think"] = False
        r = requests.post(self.cfg["host"] + "/api/chat", json=payload, timeout=(10, 900))
        r.raise_for_status()
        return (r.json().get("message", {}).get("content") or "").strip() or "(no answer)"

    # ---------------- info panels (shared by commands and shortcuts)

    def plan_lines(self):
        if not self.plan:
            return [dim("  No plan yet.")]
        return [bold("  Plan")] + ["  " + (dim(l) if "[x]" in l.lower() else cyan(l)) for l in self.plan.splitlines()]

    def last_output_lines(self):
        if not self.last_output:
            return [dim("  No actions yet.")]
        label, result = self.last_output
        lines = result.splitlines()
        body = [dim("  │ ") + l[:300] for l in lines[:200]]
        if len(lines) > 200:
            body.append(dim(f"  │ … {len(lines) - 200} more lines"))
        return [bold(f"  Last action: {label}")] + body

    def info_lines(self):
        STATS.refresh()
        run = lambda cmd: subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()
        cpu = run("lscpu | sed -n 's/^Model name: *//p' | head -1")
        load = os.getloadavg()
        du = shutil.disk_usage(HOME)
        used, total = STATS.ram
        row = lambda k, v: f"  {bold(k)}{' ' * (18 - len(k))} {v}"
        L = [bold("  Computer"),
             row("CPU", f"{cpu} · {os.cpu_count()} threads · load {load[0]:.1f}"),
             row("RAM", f"{used:.1f} / {total:.1f} GB used")]
        g = STATS.gpu
        L.append(row("GPU", f"{g['name']} · VRAM {g['used']:.1f} / {g['total']:.1f} GB · {g['util']}% busy · {g['temp']}°C")
                 if g else row("GPU", "no NVIDIA GPU found"))
        L.append(row("Disk (home)", f"{(du.total - du.free) / 1e9:.0f} / {du.total / 1e9:.0f} GB used · {du.free / 1e9:.0f} GB free"))
        L += ["", bold("  AI"), row("Model", self.cfg["model"])]
        try:
            ps = requests.get(self.cfg["host"] + "/api/ps", timeout=5).json().get("models", [])
            ver = requests.get(self.cfg["host"] + "/api/version", timeout=5).json().get("version", "?")
        except Exception:
            ps, ver = None, "not running"
        L.append(row("Ollama", ver))
        for m in ps or []:
            size, vram = m.get("size", 0), m.get("size_vram", 0)
            where = "100% GPU" if size and vram >= size else f"{100 * vram / size:.0f}% GPU / {100 - 100 * vram / size:.0f}% CPU (slower)"
            L.append(row("Loaded", f"{m['name']} · {size / 1e9:.1f} GB · {where} · context {m.get('context_length', '?')}"))
        if ps == []:
            L.append(row("Loaded", "nothing (the model loads on your first message)"))
        u = self.context_used()
        L.append(row("Context (memory)", f"{u} / {self.cfg['num_ctx']} tokens ({100 * u / self.cfg['num_ctx']:.0f}%) · {len(self.messages)} messages"))
        L.append(row("Speed", f"{self.last_tps:.1f} tokens/s (last answer)" if self.last_tps else "—"))
        L.append(row("Temperature", self.cfg["temperature"]))
        L.append(row("Reasoning", ("on" if self.think else "off") + (" · details shown" if self.cfg["verbose"] else "")))
        L.append(row("Auto-approve", "ON (no questions)" if self.auto else ("off" + (f" · always allowed: {', '.join(sorted(self.always))}" if self.always else ""))))
        L.append(row("Working dir", short_path(self.cwd)))
        return L


def _loose_replace(text, old_text, new_text):
    """Find old_text ignoring indentation/trailing spaces (small models often lose indentation).
    Returns (real_old, reindented_new) if there is exactly one match, else None."""
    want = [l.strip() for l in old_text.strip("\n").splitlines()]
    if not want or not any(want):
        return None
    lines = text.splitlines(keepends=True)
    n = len(want)
    hits = [i for i in range(len(lines) - n + 1) if all(lines[i + k].strip() == want[k] for k in range(n))]
    if len(hits) != 1:
        return None
    i = hits[0]
    real_old = "".join(lines[i:i + n])
    first_old = old_text.strip("\n").splitlines()[0]
    have = re.match(r"\s*", lines[i]).group()
    had = re.match(r"\s*", first_old).group()
    new_lines = []
    for l in new_text.strip("\n").splitlines():
        if l.strip():
            l = have + l[len(had):] if l.startswith(had) else have + l.lstrip()
        new_lines.append(l)
    real_new = "\n".join(new_lines) + ("\n" if real_old.endswith("\n") else "")
    return real_old, real_new


def _drop_repeated_tail(text, old_text, new_text):
    """Fix a common small-model mistake: using old_text as an anchor and re-typing the lines that
    already follow it, which duplicates them. Returns (new_text, number_of_lines_removed)."""
    if not new_text.startswith(old_text.rstrip("\n")):
        return new_text, 0
    idx = text.index(old_text) + len(old_text)
    following = [l.strip() for l in text[idx:].lstrip("\n").splitlines()]
    added = new_text[len(old_text.rstrip("\n")):].strip("\n").splitlines()
    for k in range(len(added) - 1, 0, -1):          # keep at least one genuinely new line
        tail = [l.strip() for l in added[-k:]]
        if any(tail) and following[:k] == tail:
            keep = added[:-k]
            while keep and not keep[-1].strip():
                keep.pop()
            return old_text.rstrip("\n") + "\n" + "\n".join(keep) + ("\n" if old_text.endswith("\n") else ""), k
    return new_text, 0


def _lost_lines(old_text, new_text):
    """Lines an edit removes that don't come back in a similar form (changed lines are not reported)."""
    new_lines = [l.strip() for l in new_text.splitlines() if l.strip()]
    lost = []
    for l in old_text.splitlines():
        t = l.strip()
        if len(t) > 2 and t not in new_lines and not difflib.get_close_matches(t, new_lines, n=1, cutoff=0.6):
            lost.append(t)
    return lost


def _dup_warning(old, new):
    """Small models often repeat the line that follows their edit. Warn them so they check."""
    def dups(t):
        L = t.splitlines()
        return {(i + 1, L[i].strip()) for i in range(len(L) - 1) if len(L[i].strip()) > 3 and L[i] == L[i + 1]}
    before = {line for _, line in dups(old)}
    new_dups = [(n, line) for n, line in dups(new) if line not in before]
    if not new_dups:
        return ""
    where = ", ".join(f"lines {n}-{n + 1}" for n, _ in new_dups[:3])
    out(yellow(f"  ! possible duplicated line ({where}) — the AI will be asked to check"))
    return (f"\nWARNING: the file now has duplicated consecutive lines ({where}). This is probably a mistake "
            "in your edit: re-read the file and fix it.")


def _size(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def _brief(v, n=60):
    s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
    s = s.replace("\n", "⏎")
    return s if len(s) <= n else s[:n] + "…"


def _hard_close(response):
    """Drop an HTTP stream at once, even while another thread is reading it (so Ollama stops generating)."""
    try:
        sock = response.raw._connection.sock
        if sock is not None:
            sock.shutdown(socket.SHUT_RDWR)
    except Exception:
        pass
    try:
        response.close()
    except Exception:
        pass


def _killpg(proc):
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


# ---------------------------------------------------------------- REPL

if HAVE_PT:
    class InputLexer(Lexer):
        """Colours what you type: /commands, -flags, !shell commands and 'call -agent'."""
        AGENT = re.compile(r"\bcall\s+-agent\b")

        def lex_document(self, document):
            def line(i):
                text = document.lines[i]
                if i == 0 and text.startswith("!"):
                    return [("class:in.shell", text)]
                parts = []
                if i == 0 and text.startswith("/"):
                    cmd, rest = re.match(r"(\S*)(.*)", text).groups()
                    parts.append(("class:in.cmd", cmd))
                    for tok in re.split(r"(\s+)", rest):
                        parts.append(("class:in.flag" if re.match(r"-{1,2}[a-z]", tok) else "", tok))
                    return parts
                pos = 0
                for m in self.AGENT.finditer(text):
                    parts += [("", text[pos:m.start()]), ("class:in.agent", m.group())]
                    pos = m.end()
                return parts + [("", text[pos:])]
            return line

    class AgentCompleter(Completer):
        def __init__(self, repl):
            self.r = repl
            self.paths = PathCompleter(expanduser=True)
            self.dirs = PathCompleter(expanduser=True, only_directories=True)

        def _word(self, completer, word, ev):
            yield from completer.get_completions(Document(word, len(word)), ev)

        def get_completions(self, document, ev):
            text = document.text_before_cursor
            if text.startswith("/") and " " not in text:
                low = text.lower()
                for name, args, short in self.r.command_entries():
                    if name.startswith(low):
                        yield Completion(name, start_position=-len(text),
                                         display=f"{name} {args}".strip(), display_meta=short)
                return
            if text.startswith("/"):
                cmd, _, rest = text.partition(" ")
                word = rest.split(" ")[-1]
                opts = self.r.arg_options(cmd.lower(), rest)
                if opts == "path":
                    yield from self._word(self.paths, word, ev)
                elif opts == "dir":
                    yield from self._word(self.dirs, word, ev)
                elif isinstance(opts, list) and rest.count(" ") <= (1 if rest.split(" ")[0] in WIKI_FLAGS | {"mode"} else 0):
                    for o in opts:
                        if o.startswith(word):
                            yield Completion(o, start_position=-len(word), display_meta=self.r.arg_meta(cmd.lower(), o))
                return
            m = re.match(r"(call\s+-agent\s+)(\S*)$", text, re.I)
            if m:
                for name, r in load_roles().items():
                    if name.startswith(m.group(2)):
                        yield Completion(name + " ", start_position=-len(m.group(2)), display=name,
                                         display_meta=r["description"])
                return
            if re.fullmatch(r"call( -?a?g?e?n?t?)?", text, re.I):
                yield Completion("call -agent ", start_position=-len(text), display_meta="run a helper yourself")
                return
            word = text.split(" ")[-1]
            if word.startswith(("~/", "./", "../", "/")) and len(text) > len(word):
                yield from self._word(self.paths, word, ev)


class Worker:
    """Runs the AI's work in the background so the prompt stays usable. Jobs run one after another."""

    def __init__(self, repl):
        self.r = repl
        self.q = collections.deque()
        self.cv = threading.Condition()
        self.current = None
        self.t0 = 0.0
        threading.Thread(target=self._run, daemon=True).start()

    def busy(self):
        return self.current is not None

    def pending(self):
        return self.current is not None or bool(self.q)

    def put(self, job):
        with self.cv:
            self.q.append(job)
            self.cv.notify()

    def _run(self):
        while True:
            with self.cv:
                while not self.q:
                    self.cv.wait()
                job = self.q.popleft()
                self.current, self.t0 = job, time.time()
            self.r.invalidate()
            try:
                self.r.run_job(job)
            except Interrupted:
                self.r.a.mark_interrupted()
                out(red("  interrupted") + dim(" · Ctrl+C again to quit"))
                if self.q:
                    out(dim(f"  {len(self.q)} queued message(s) will run next — /queue clear to drop them"))
            except RuntimeError as e:
                out(red(f"  {e}"))
            except ToolError as e:
                out(red(f"  {e}"))
            except Exception as e:
                out(red(f"  Unexpected error: {type(e).__name__}: {e}"))
            finally:
                self.r.a.cancel.clear()
                self.current = None
                out()
                self.r.invalidate()


# commands that run at once even while the AI works (they don't change the conversation)
IMMEDIATE_CMDS = {"/help", "/?", "/plan", "/files", "/tasks", "/model", "/think", "/details", "/temperature", "/mode", "/allocation", "/knowledge",
                  "/infocomputer", "/context", "/auto", "/copy", "/setkeyboard", "/color", "/theme", "/agents", "/system", "/config", "/btw", "/queue",
                  "/exit", "/quit", "/q"}
# run at once when idle, queued while the AI works
CONVO_CMDS = {"/file", "/drop", "/cd", "/clear", "/load"}
# everything else (messages, /search, /loop, /compact, /save, tasks, !commands) runs in the background worker


class Repl:
    def __init__(self, agent):
        self.a = agent
        self.tasks = load_tasks()
        self.last_loop_goal = None
        self.last_cc = 0.0
        self.flash = ("", 0.0)
        self.prefill = ""
        self._models = None
        self.pending_ask = None
        self.ask_lock = threading.Lock()
        self.btw_n = 0
        apply_theme(agent.cfg)
        ui.agent = agent
        agent.asker = self.ask
        self.refresh_keymap()
        self.worker = Worker(self)
        self.session = PromptSession(
            history=FileHistory(str(HIST_FILE)), completer=AgentCompleter(self), lexer=InputLexer(),
            complete_while_typing=True, key_bindings=self.bindings(), bottom_toolbar=self.toolbar,
            style=MENU_STYLE, refresh_interval=0.2, multiline=True, reserve_space_for_menu=8,
            erase_when_done=True, prompt_continuation=lambda width, n, soft: "  ")

    # ---------------- prompt pieces

    def refresh_keymap(self):
        ui.keymap = {v: k for k, v in self.a.cfg["keys"].items()}

    def key_of(self, action):
        return keyname(self.a.cfg["keys"].get(action))

    def invalidate(self):
        try:
            app = self.session.app
            if app.is_running:
                app.invalidate()
        except Exception:
            pass

    def message(self):
        a, w = self.a, term_width() - 1
        P = []

        def line(style, text):
            P.append((style, text[:w] + "\n"))

        if self.worker.busy():
            now = time.time()
            label = ui.label or "Working"
            rate = ui.rate() if ui.depth else 0
            info = f" {now - self.worker.t0:.0f}s" + (f" · {rate:.1f} tok/s" if rate else "")
            info += " · " + " · ".join(machine_bits(a))
            P += [("class:spin", UI.FRAMES[int(now * 10) % len(UI.FRAMES)] + " "),
                  ("class:spin.label", label + "…"), ("class:status", info[:max(0, w - len(label) - 3)] + "\n")]
        if self.btw_n:
            line("class:btw", "  answering your /btw…")
        queued = list(self.worker.q)
        for job in queued[:4]:
            line("class:queued", f"  ↳ queued: {_brief(job['shown'], 90)}")
        if len(queued) > 4:
            line("class:queued", f"  ↳ … and {len(queued) - 4} more")
        ask = self.pending_ask
        if ask and ask["stage"] == "choose":
            line("class:ask", " " + ask["q"])
            for i, o in enumerate(ask["opts"]):
                sel = i == ask["sel"]
                line("class:msel" if sel else "class:mitem", f" {'❯' if sel else ' '} {i + 1}. {o}")
        elif ask:
            line("class:ask", " What should the AI do instead? (Enter sends · empty = just stop)")
        bits = [short_path(a.cwd)] + machine_bits(a, full=True)
        if a.last_tps:
            bits.append(f"{a.last_tps:.1f} tok/s")
        s = " · ".join(bits)
        room = w - len(a.cfg["model"]) - 4 - (6 if a.auto else 0)
        P += [("class:status.model", " " + a.cfg["model"]), ("class:status", " · " + s[:max(room, 0)])]
        if a.auto:
            P.append(("class:status.auto", "  AUTO"))
        P.append(("", "\n"))
        P.append(("class:prompt", "✎ " if ask and ask["stage"] == "note" else "❯ "))
        return FormattedText(P)

    def toolbar(self):
        msg, until = self.flash
        if msg and time.time() < until:
            return FormattedText([("class:bottom-toolbar.flash", " " + msg)])
        k = self.key_of
        ask = self.pending_ask
        if ask and ask["stage"] == "choose":
            return " press 1 2 3 (or ↑↓ + Enter) to answer · esc = no · ctrl+c = stop the AI"
        if ask:
            return " type what the AI should do instead · Enter sends · esc = just stop"
        if self.worker.busy():
            return (f" esc interrupt · Enter queues a message · /btw side question · {k('details')} details"
                    f"{' (on)' if self.a.cfg['verbose'] else ''} · {k('plan')} plan · {k('last_output')} last output")
        return (f" / commands · {k('details')} details{' (on)' if self.a.cfg['verbose'] else ''} · {k('plan')} plan · "
                f"{k('last_output')} last output · {k('info')} info · {k('newline')} new line · ctrl+c ×2 quit")

    def flash_msg(self, msg, app=None):
        self.flash = (msg, time.time() + 2.5)
        if app:
            app.invalidate()

    def command_entries(self):
        ent = [(n, a, s) for n, a, s, *_ in CMDS]
        ent += [("/" + t, "[path]" if t not in FREE_TEXT_TASKS else "<text>", v["description"])
                for t, v in self.tasks.items() if "/" + t not in CMD_ARGS]
        return ent

    def arg_options(self, cmd, rest=""):
        if cmd in ("/file", "/drop"):
            return "path"
        if cmd == "/cd":
            return "dir"
        if cmd == "/model":
            return self.model_names()
        if cmd == "/agents":
            return list(AGENT_MODES) if rest.startswith("mode ") else ["mode", "resume", "vram", "list"]
        if cmd == "/compact":
            return ["auto on", "auto off"]
        if cmd == "/mode":
            return list(MODE_INFO)
        if cmd == "/knowledge":
            return ["auto on", "auto off"] + list(KNOWLEDGE.listing())
        if cmd == "/system":
            return ["edit", "reset"]
        if cmd == "/color":
            return list(PROMPT_COLORS)
        if cmd == "/theme":
            return list(THEMES)
        if cmd in ("/think", "/details"):
            return ["on", "off"]
        if cmd == "/temperature":
            return ["0.2", "0.4", "0.6", "0.7", "0.8", "1.0"]
        if cmd in ("/save", "/load"):
            if rest.split(" ")[0] in WIKI_FLAGS and " " in rest:
                return wiki_names()
            return ["-wiki"] + (sorted(session_files()) if cmd == "/load" else [])
        if cmd == "/setkeyboard":
            return list(ACTIONS) + ["reset"]
        if cmd == "/loop":
            return ["continue"]
        if cmd == "/queue":
            return ["clear"]
        if cmd[1:] in self.tasks and cmd[1:] not in FREE_TEXT_TASKS:
            return "path"
        return None

    def arg_meta(self, cmd, option):
        if cmd == "/setkeyboard":
            if option == "reset":
                return "all shortcuts back to the defaults"
            return f"now {keyname(self.a.cfg['keys'].get(option))} · Enter opens the table to change it"
        return ""

    def model_names(self):
        if self._models is None:
            try:
                ollama = sorted(m["name"] for m in requests.get(self.a.cfg["host"] + "/api/tags", timeout=2).json()["models"])
            except Exception:
                ollama = []
            self._models = self.a.vram.servers.names() + ollama
        return self._models

    # ---------------- questions from the AI (permissions), answered in the prompt

    def ask(self, question, options, note_index=None):
        """Called from the worker thread. Shows the question above the input and waits for the answer."""
        req = {"q": question, "opts": options, "sel": 0, "stage": "choose", "note_index": note_index,
               "event": threading.Event(), "result": (None, ""), "saved": ""}
        with self.ask_lock, ui.busy("Waiting for your answer"):
            ui.reset_rate()
            self.pending_ask = req
            self.invalidate()
            try:
                while not req["event"].wait(0.1):
                    self.a.check_cancel()
            finally:
                self.pending_ask = None
                self.invalidate()
        return req["result"]

    def _answer(self, idx, buf):
        ask = self.pending_ask
        if not ask or idx >= len(ask["opts"]):
            return
        if idx == ask["note_index"]:
            ask["stage"], ask["saved"] = "note", buf.text
            buf.reset()
            return
        ask["result"] = (idx, "")
        ask["event"].set()

    def _finish_note(self, buf, note):
        ask = self.pending_ask
        if not ask:
            return
        saved = ask["saved"]
        buf.reset(Document(saved, len(saved)))
        ask["result"] = (ask["note_index"], note)
        ask["event"].set()

    # ---------------- key bindings

    def bindings(self):
        kb = KeyBindings()
        choosing = Condition(lambda: bool(self.pending_ask) and self.pending_ask["stage"] == "choose")
        noting = Condition(lambda: bool(self.pending_ask) and self.pending_ask["stage"] == "note")
        busy_empty = Condition(lambda: self.worker.busy() and not self.pending_ask
                               and not get_app().current_buffer.text)
        can_pull = Condition(lambda: not self.pending_ask and not get_app().current_buffer.text
                             and bool(self.worker.q))

        @kb.add("c-c")
        def _(e):
            b = e.current_buffer
            now = time.time()
            if self.pending_ask or (self.worker.busy() and not b.text and not b.selection_state):
                if self.pending_ask and self.pending_ask["stage"] == "note":
                    saved = self.pending_ask["saved"]
                    b.reset(Document(saved, len(saved)))
                self.a.interrupt()
                self.last_cc = now
                self.flash_msg("Interrupted · press Ctrl+C again to quit", e.app)
                return
            if b.selection_state:
                data = b.copy_selection()
                Clipboard.copy(data.text, e.app.output)
                self.flash_msg("Copied to clipboard", e.app)
                return
            if b.text:
                b.reset()
                return
            if now - self.last_cc < 2:
                e.app.exit(exception=EOFError())
                return
            self.last_cc = now
            self.flash_msg("Press Ctrl+C again to quit", e.app)

        @kb.add("escape", filter=busy_empty & ~has_completions)
        def _(e):
            self.a.interrupt()
            self.flash_msg("Interrupted", e.app)

        @kb.add("up", filter=choosing)
        def _(e):
            a = self.pending_ask
            a["sel"] = (a["sel"] - 1) % len(a["opts"])

        @kb.add("down", filter=choosing)
        def _(e):
            a = self.pending_ask
            a["sel"] = (a["sel"] + 1) % len(a["opts"])

        for d in "123456789":
            @kb.add(d, filter=choosing)
            def _(e, d=d):
                self._answer(int(d) - 1, e.current_buffer)

        @kb.add("escape", filter=choosing)
        def _(e):
            a = self.pending_ask
            a["result"] = (None, "")
            a["event"].set()

        @kb.add("escape", filter=noting)
        def _(e):
            self._finish_note(e.current_buffer, "")

        @kb.add("<any>", filter=choosing)
        def _(e):
            pass                       # while a question is shown, other keys do nothing

        @kb.add("up", filter=can_pull)
        def _(e):
            with self.worker.cv:
                job = self.worker.q.pop() if self.worker.q else None
            if job:
                e.current_buffer.text = job["shown"]
                e.current_buffer.cursor_position = len(job["shown"])
                self.flash_msg("Took the last queued message back — edit it and press Enter", e.app)

        @kb.add("enter")
        def _(e):
            b = e.current_buffer
            if choosing():
                self._answer(self.pending_ask["sel"], b)
                return
            if noting():
                self._finish_note(b, b.text.strip())
                return
            st = b.complete_state
            if st and st.current_completion:
                b.apply_completion(st.current_completion)
                t = b.text
                if t.startswith("/") and " " not in t:
                    if CMD_ARGS.get(t) or (t[1:] in self.tasks):
                        b.insert_text(" ")
                        return
                    b.validate_and_handle()
                return
            if b.text.endswith("\\"):
                b.delete_before_cursor(1)
                b.insert_text("\n")
                return
            b.validate_and_handle()

        newline_is = lambda k: Condition(lambda: self.a.cfg["keys"].get("newline", "escape enter") == k)

        @kb.add("escape", "enter", filter=newline_is("escape enter") | newline_is("shift enter"))
        def _(e):
            e.current_buffer.insert_text("\n")

        @kb.add(Keys.F24)            # Shift+Enter, from terminals that send it separately
        def _(e):
            if self.a.cfg["keys"].get("newline") == "shift enter":
                e.current_buffer.insert_text("\n")
            else:
                self.flash_msg("Shift+Enter works here: choose it for new line in /setkeyboard", e.app)

        @kb.add("c-j", filter=newline_is("c-j"))
        def _(e):
            e.current_buffer.insert_text("\n")

        for letter in "abcdefghijklmnopqrstuvwxyz":
            key = "c-" + letter
            if key in RESERVED_KEYS:
                continue

            @kb.add(key, filter=Condition(lambda key=key: key in ui.keymap))
            def _(e, key=key):
                self.on_key(e, ui.keymap[key])

        return kb

    def on_key(self, e, action):
        if action == "newline":
            e.current_buffer.insert_text("\n")
        elif action == "paste":
            e.current_buffer.insert_text(Clipboard.paste())
        elif action == "details":
            self.a.cfg["verbose"] = not self.a.cfg["verbose"]
            self.flash_msg(f"Details {'ON: reasoning and full outputs are shown' if self.a.cfg['verbose'] else 'OFF'}", e.app)
        elif action in ("plan", "last_output", "info"):
            lines = self.action_lines(action)
            threading.Thread(target=lambda: out("\n".join(lines)), daemon=True).start()

    def action_lines(self, action):
        if action == "plan":
            return self.a.plan_lines()
        if action == "last_output":
            return self.a.last_output_lines()
        if action == "info":
            return self.a.info_lines()
        return []

    # ---------------- main loop

    def banner(self):
        text = ["",
                gradient("CHICKEN", *WORDMARK) + dim("  local AI agent"),
                dim(f"{self.a.cfg['model']} · mode: {self.a.cfg.get('mode')} · {self.a.cfg.get('routing')} · "
                    f"{short_path(self.a.cwd)}"),
                dim("type / for commands · /mode to switch · /allocation for memory"),
                ""]
        art = pixel_lines(CHICKEN, CHICKEN_COLORS)
        print()
        if not art:
            print("  Chicken · " + text[3])
        for pic, t in zip(art, text):
            print("  " + pic + "   " + t)
        print()

    def run(self, initial=None):
        STATS.start()
        atexit.register(reset_terminal_colors)
        atexit.register(self.a.vram.servers.stop_started)
        apply_theme(self.a.cfg)
        self.banner()
        with patch_stdout(raw=True):
            if initial:
                self.dispatch(initial)
            while True:
                try:
                    default, self.prefill = self.prefill, ""
                    line = self.session.prompt(self.message, default=default).strip()
                except EOFError:
                    break
                except KeyboardInterrupt:
                    continue
                if not line:
                    continue
                try:
                    if not self.dispatch(line):
                        break
                except KeyboardInterrupt:
                    pass
                except (RuntimeError, ToolError) as e:
                    out(red(f"  {e}"))
            self.a.interrupt()
        reset_terminal_colors()

    def dispatch(self, line):
        """Decide what runs now and what waits for the AI to finish."""
        if re.match(r"call\s+-", line, re.I):
            return self.enqueue("cmd", line, line)
        if line.startswith("!"):
            return self.enqueue("shell", line[1:].strip(), line)
        if not line.startswith("/"):
            return self.enqueue("turn", line, line)
        cmd = self.resolve_cmd(line.partition(" ")[0].lower())
        if cmd in IMMEDIATE_CMDS or (cmd in CONVO_CMDS and not self.worker.pending()):
            out(arrow(self.a.cfg) + line)
            ok = self.handle(line)
            out()
            return ok
        return self.enqueue("cmd", line, line)

    def enqueue(self, kind, text, shown):
        if self.worker.pending():
            out(dim(f"❯ {shown}   · queued"))
            self.worker.put({"kind": kind, "text": text, "shown": shown, "echo": True})
        else:
            out(arrow(self.a.cfg) + shown)
            self.worker.put({"kind": kind, "text": text, "shown": shown, "echo": False})
        return True

    def run_job(self, job):
        if job["echo"]:
            out(arrow(self.a.cfg) + job["shown"])
        if job["kind"] == "turn":
            self.a.run_turn(job["text"])
        elif job["kind"] == "shell":
            self.shell(job["text"])
        else:
            self.handle(job["text"])

    def btw(self, question):
        if not question:
            print(yellow("  Usage: /btw <question>   e.g. /btw what does this error mean?"))
            return
        self.btw_n += 1
        small = self.a.cfg.get("btw_model")
        if not (small and self.a.vram.servers.spec(small)) and self.worker.busy():
            print(dim("  I'll answer as soon as the current step finishes (one GPU, one answer at a time)."))

        def work():
            try:
                answer = self.a.side_question(question)
            except Exception as e:
                answer = f"(could not answer: {e})"
            finally:
                self.btw_n -= 1
            lines = [magenta("btw: ") + bold(question)]
            st = {"code": False}
            lines += ["   " + render_md(no_emoji(l), st) for l in answer.splitlines()]
            out("\n".join(lines) + "\n")
            self.invalidate()

        threading.Thread(target=work, daemon=True).start()

    def set_confirmations(self, mode, save=True):
        """ask or auto for workers AND actions together (what /mode ask|auto does)."""
        a = self.a
        a.cfg["mode"] = mode
        a.cfg["agents_mode"] = mode
        a.auto = mode == "auto"
        if save:
            save_config_key("mode", mode)
            save_config_key("agents_mode", mode)

    def knowledge_cmd(self, arg):
        arg = arg.strip()
        if arg in ("auto on", "auto off"):
            on = arg.endswith("on")
            self.a.cfg["knowledge_auto"] = on
            save_config_key("knowledge_auto", on)
            print(green(f"  Knowledge for workers: {'on' if on else 'off'}") + dim(" (saved)"))
            return
        if not arg:
            wikis = KNOWLEDGE.listing()
            if not wikis:
                print(dim(f"  No wikis yet. Add folders with .md pages in {short_path(KNOWLEDGE_DIR)}"))
                return
            counts = collections.Counter(sec["wiki"] for sec in KNOWLEDGE.sections)
            print(bold("  Knowledge wikis") + dim(f"  {short_path(KNOWLEDGE_DIR)} · auto for workers: "
                                                  f"{'on' if self.a.cfg.get('knowledge_auto', True) else 'off'}"))
            for w, d in wikis.items():
                print(f"  {cyan(f'{w:<16}')} {counts[w]:>3} sections  {dim(_brief(d, 90))}")
            print(dim("  /knowledge <question> searches · /knowledge auto on|off"))
            return
        hits = KNOWLEDGE.search(arg, k=5)
        if not hits:
            print(dim("  Nothing found. Try other key terms in English."))
            return
        for sc, sec in hits:
            first = next((l for l in sec["text"].splitlines() if l.strip() and not l.startswith("```")), "")
            print(f"  {cyan(sec['wiki'] + '/' + sec['page'])} › {bold(sec['head'])} {dim(f'({sc:.1f})')}")
            print(dim(f"    {_brief(first.strip(), 110)}"))

    def mode_cmd(self, arg):
        a = self.a
        arg = arg.strip().lower()
        if arg not in MODE_INFO:
            print(bold("  Mode") + dim(f"  (now: {a.cfg.get('mode')} · {a.cfg.get('routing')})"))
            for k, v in MODE_INFO.items():
                cur = k in (a.cfg.get("mode"), a.cfg.get("routing"))
                print("  " + (green(f"{k:<16}") if cur else f"{k:<16}") + " " + dim(v))
            print(dim("  /mode ask|auto sets the confirmations; the others set who works."))
            return
        if arg in ("ask", "auto"):
            self.set_confirmations(arg)
        else:
            a.cfg["routing"] = arg
            save_config_key("routing", arg)
            a.cfg["model"] = a.cfg["heavy_model"] if arg == "bonsai-only" else a.base_model
            a.cfg["num_ctx"] = a.full_ctx()
        print(green(f"  Mode: {arg}") + dim(f" — {MODE_INFO[arg]} (saved)"))

    def allocation(self):
        """/allocation: what uses the GPU and the RAM right now."""
        a, v = self.a, self.a.vram
        used, total = v.gpu_mem()
        lim = v.budget()
        loaded = v.loaded()
        cols = {a.cfg["model"]: agent_color(None), a.cfg["heavy_model"]: "#e684d4", a.cfg.get("btw_model"): "#f5c542"}
        width = 48
        print(bold("  GPU") + dim(f"  {used:.1f} of {total:.1f} GB used · limit {a.cfg['vram_budget_gb']} = {lim:.1f} GB"))
        bar, x = "", 0
        for m, gb in loaded.items():
            n = max(1, round(gb / max(total, 1) * width))
            bar += hexc(cols.get(m, "#e3b458"), "█" * n)
            x += n
        other = max(0.0, used - sum(loaded.values()))
        n_o = round(other / max(total, 1) * width)
        bar += dim("▒" * n_o)
        x += n_o
        cut = round(lim / max(total, 1) * width)
        rest = "".join("│" if i == cut else "░" for i in range(x, width))
        print("  " + bar + dim(rest) + dim("  │ = limit"))
        for m, gb in loaded.items():
            ctx = v.ctx_of.get(m)
            p = v.profile(m)
            split = f"weights {p['base']:.1f} + context {max(0.0, gb - p['base']):.1f}" if ctx else ""
            role = ("orchestrator" if m == a.cfg["model"] and a.orchestrating() else "main" if m == a.cfg["model"]
                    else "/btw" if m == a.cfg.get("btw_model") else "worker")
            print(f"  {hexc(cols.get(m, '#e3b458'), '■')} {m:<22} {gb:5.1f} GB  " + dim(
                f"{role} · {split}" + (f" · {ctx // 1024}k tokens" if ctx else "") +
                (" · partly on the CPU!" if v.partial.get(m) else "")))
        if not loaded:
            print(dim("    no model loaded"))
        print(dim(f"  ▒ other programs {other:.1f} GB · free under the limit {max(0.0, lim - used):.1f} GB · "
                  f"free on the card {max(0.0, total - used):.1f} GB"))
        mem = {}
        try:
            for line in open("/proc/meminfo"):
                k, val = line.split(":", 1)
                mem[k] = int(val.split()[0]) / 1e6
        except OSError:
            pass
        if mem:
            print(bold("  RAM") + dim(f"  {mem['MemTotal'] - mem['MemAvailable']:.1f} of {mem['MemTotal']:.1f} GB used · "
                                     f"{mem['MemAvailable']:.1f} GB available · file cache {mem.get('Cached', 0):.1f} GB"))
            groups = {"Chicken": 0.0, "Ollama (models + server)": 0.0, "llama.cpp servers": 0.0}
            try:
                ps = subprocess.run(["ps", "-eo", "pid,rss,args"], capture_output=True, text=True, timeout=5).stdout
                for row in ps.splitlines()[1:]:
                    pid, rss, cmdline = row.split(None, 2)
                    gb = int(rss) / 1e6
                    if int(pid) == os.getpid():
                        groups["Chicken"] += gb
                    elif "ollama" in cmdline:
                        groups["Ollama (models + server)"] += gb
                    elif "llama-server" in cmdline:
                        groups["llama.cpp servers"] += gb
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
            for k, gb in groups.items():
                print(f"    {k:<26} {gb:5.1f} GB")
            print(dim("    (models on the GPU keep only small buffers in RAM; their files sit in the file cache)"))

    def queue_cmd(self, arg):
        with self.worker.cv:
            if arg == "clear":
                n = len(self.worker.q)
                self.worker.q.clear()
                print(dim(f"  Removed {n} queued message(s)."))
                return
            jobs = list(self.worker.q)
        if not jobs:
            print(dim("  The queue is empty. Messages you send while the AI works wait here."))
        for i, j in enumerate(jobs, 1):
            print(f"  {i}. {j['shown']}")

    def resolve_cmd(self, cmd):
        names = [n for n, *_ in self.command_entries()]
        if cmd in names:
            return cmd
        hits = [n for n in names if n.startswith(cmd)]
        return hits[0] if len(hits) == 1 else cmd

    def handle(self, line):
        a = self.a
        if line.startswith("!"):
            self.shell(line[1:].strip())
            return True
        if re.match(r"call\s+-", line, re.I):
            self.call_cmd(line)
            return True
        if not line.startswith("/"):
            a.run_turn(line)
            return True
        cmd, _, arg = line.partition(" ")
        arg = arg.strip()
        cmd = self.resolve_cmd(cmd.lower())

        if cmd in ("/exit", "/quit", "/q"):
            return False
        elif cmd in ("/help", "/?"):
            self.help()
        elif cmd == "/file":
            self.add_files(arg)
        elif cmd == "/files":
            for f in a.focus:
                print("  " + a.rel(f))
            if not a.focus:
                print(dim("  No files loaded. Use /file <path>"))
        elif cmd == "/drop":
            if not arg:
                a.focus = []
                print(dim("  Dropped all files."))
            else:
                p = a.resolve(arg)
                a.focus = [f for f in a.focus if f != p]
                print(dim(f"  Dropped {a.rel(p)}"))
        elif cmd == "/search":
            if not arg:
                print(yellow("  Usage: /search <what to look for>"))
            else:
                a.run_turn(f"Search the web for: {arg}\nRead the best sources with fetch_url and give me a clear answer, with the source links.")
        elif cmd == "/loop":
            self.loop(arg)
        elif cmd == "/plan":
            print("\n".join(a.plan_lines()))
        elif cmd == "/btw":
            self.btw(arg)
        elif cmd == "/mode":
            self.mode_cmd(arg)
        elif cmd == "/knowledge":
            self.knowledge_cmd(arg)
        elif cmd == "/allocation":
            self.allocation()
        elif cmd == "/queue":
            self.queue_cmd(arg)
        elif cmd == "/tasks":
            self.tasks = load_tasks()
            print(bold("  Preset tasks") + dim(f"  (edit or add .md files in {short_path(TASKS_DIR)})"))
            for name, t in self.tasks.items():
                tag = cyan(" [autonomous]") if t["loop"] and "autonomous" not in t["description"] else ""
                print(f"  {green('/' + name)}{' ' * (14 - len(name))} {t['description']}{tag}")
        elif cmd == "/cd":
            p = a.resolve(arg or str(a.root))
            if not p.is_dir():
                print(red(f"  Not a directory: {arg}"))
            else:
                a.cwd = p
                os.chdir(p)
                print(dim(f"  Working directory: {short_path(p)}"))
        elif cmd == "/model":
            self.model(arg)
        elif cmd == "/think":
            a.think = (arg == "on") if arg in ("on", "off") else not a.think
            print(dim(f"  Reasoning {'on (smarter, slower)' if a.think else 'off (faster)'}"))
        elif cmd == "/details":
            a.cfg["verbose"] = (arg == "on") if arg in ("on", "off") else not a.cfg["verbose"]
            print(dim(f"  Details {'on' if a.cfg['verbose'] else 'off'}  ({self.key_of('details')} toggles it anytime)"))
        elif cmd == "/temperature":
            self.temperature(arg)
        elif cmd == "/infocomputer":
            print("\n".join(a.info_lines()))
        elif cmd == "/auto":
            self.set_confirmations("ask" if a.auto else "auto")
            print(yellow("  ! Auto-approve ON: the AI will edit, delete and run commands without asking.") if a.auto
                  else green("  Auto-approve OFF: risky actions will ask first."))
        elif cmd == "/context":
            used = a.context_used()
            pct = 100 * used / a.cfg["num_ctx"]
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            print(f"  {bar} {used} / {a.cfg['num_ctx']} tokens ({pct:.0f}%) · {len(a.messages)} messages")
            how = (f"auto: what {a.cfg['model']} leaves under the {a.vram.budget():.1f} GB limit" if a.ctx_auto
                   else "fixed in config (num_ctx)")
            print(dim(f"  Context size: {how}. Auto-compact for helpers: "
                      f"{'on' if a.cfg['compact_for_helpers'] else 'off'} (/compact auto on|off)."))
            print(dim("  It auto-summarizes at ~72%. /compact to do it now, /clear to start fresh."))
        elif cmd == "/compact":
            if arg.split()[:1] == ["auto"]:
                v = arg.split()[1:2]
                on = v[0] == "on" if v and v[0] in ("on", "off") else not a.cfg["compact_for_helpers"]
                a.cfg["compact_for_helpers"] = on
                save_config_key("compact_for_helpers", on)
                print(green("  Auto-compact for helpers ON: Chicken compacts the conversation when that lets a "
                            "helper fit beside it.") if on else
                      dim("  Auto-compact for helpers OFF: helpers that don't fit beside Chicken swap it out "
                          "(eject and inject). /compact still compacts by hand."))
            else:
                a.maybe_compact(force=True)
        elif cmd == "/clear":
            a.messages, a.plan, a.focus, a.pending, a.last_tokens, a.wiki, a.run = [], "", [], [], 0, None, None
            print(dim("  Fresh conversation."))
        elif cmd == "/copy":
            last = next((m["content"] for m in reversed(a.messages) if m["role"] == "assistant" and m.get("content")), "")
            if not last:
                print(dim("  Nothing to copy yet."))
            else:
                Clipboard.copy(last)
                print(dim(f"  Copied the last answer ({len(last)} characters)."))
        elif cmd == "/save":
            self.save(arg)
        elif cmd == "/load":
            self.load(arg)
        elif cmd == "/setkeyboard":
            self.setkeyboard(arg)
        elif cmd == "/agents":
            self.agents_cmd(arg)
        elif cmd == "/system":
            self.system_cmd(arg)
        elif cmd == "/color":
            self.color(arg)
        elif cmd == "/theme":
            self.theme(arg)
        elif cmd == "/config":
            print(dim(f"  {short_path(CONF_FILE)}"))
            for k, v in a.cfg.items():
                print(f"  {k:<16} {v}")
        elif cmd[1:] in load_tasks():
            self.tasks = load_tasks()
            self.run_task(cmd[1:], arg)
        else:
            print(red(f"  Unknown command {cmd}. Type / to see the commands."))
        return True

    # ---------------- commands

    def help(self):
        entries = []
        for name, args, short, long, ex in CMDS:
            entries.append((f"{name} {args}".strip(), short, f"{name} {args}\n{long}\nExample: {ex}", name))
        for t, v in self.tasks.items():
            if "/" + t in CMD_ARGS:
                continue
            arg = "<text>" if t in FREE_TEXT_TASKS else "[path]"
            mode = "Autonomous: works until done." if v["loop"] else "Preset task."
            entries.append((f"/{t} {arg}", v["description"],
                            f"/{t} {arg}\n{mode} It asks the AI to:\n{clip(v['body'], 400)}\n"
                            f"Edit it in ~/.agent/tasks/{t}.md", "/" + t))
        k = self.key_of
        shortcuts = [
            (k("details"), "Show/hide details", "Shows the AI's reasoning as it thinks, full tool results and complete command output. Works while it's working too."),
            (k("plan"), "Show the plan", "Shows the AI's current checklist. Works while it's working too."),
            (k("last_output"), "Show last output", "Shows the full result of the last action (file read, search, command…)."),
            (k("info"), "Computer & AI info", "Same as /infocomputer."),
            (k("paste"), "Paste", "Pastes from the clipboard (your terminal's own paste also works)."),
            ("ctrl+c", "Copy · clear · stop · quit", "With text selected (shift+arrows): copies it. With text typed: clears the line. While the AI works: stops it. Twice on an empty line: quits."),
            ("ctrl+d", "Quit", "Quits on an empty line."),
            ("tab", "Complete", "Completes commands, file paths and options."),
            ("↑ ↓", "History", "Recall your previous messages."),
            (k("newline"), "New line", "Writes on several lines. Ending a line with \\ and pressing Enter also works."),
            ("!command", "Run a shell command", "Runs it yourself; the AI sees the output. Example: !git status"),
        ]
        for key, short, long in shortcuts:
            entries.append((key, short, f"{key}\n{long}\nChange shortcuts with /setkeyboard.", None))
        choice = menu("Commands & shortcuts", [(e[0], e[1]) for e in entries],
                      detail=lambda i: entries[i][2], filterable=True, height=14)
        if choice is not None and entries[choice][3]:
            name = entries[choice][3]
            self.prefill = name + (" " if CMD_ARGS.get(name, "x") else "")

    def setkeyboard(self, arg):
        a = self.a
        keys = a.cfg["keys"]
        if arg == "reset":
            keys.clear()
            keys.update({k: v[1] for k, v in ACTIONS.items()})
            save_config_key("keys", keys)
            self.refresh_keymap()
            print(green("  Shortcuts reset to defaults."))
            return
        parts = arg.split()
        if len(parts) == 2 and parts[0] in ACTIONS and parse_key(parts[1]):
            self._assign(parts[0], parse_key(parts[1]))
            return
        # anything else opens the table: on the named action (its key list open), or at the top
        focus = parts[0] if parts and parts[0] in ACTIONS else None
        msg = "" if not parts or focus and len(parts) == 1 else \
            f"'{arg}' isn't a shortcut command — edit the table instead (actions: {', '.join(ACTIONS)})"
        new = keyboard_editor(keys, focus, msg)
        if new is None:
            print(dim("  No changes."))
            return
        changed = [x for x in ACTIONS if new[x] != keys.get(x)]
        keys.update(new)
        save_config_key("keys", keys)
        self.refresh_keymap()
        if not changed:
            print(dim("  No changes."))
        for x in ACTIONS:
            print(f"  {keyname(keys[x]):<10} {x}" + (green("  ← changed (saved)") if x in changed else ""))

    def _assign(self, act, key):
        keys = self.a.cfg["keys"]
        why = key_ok(act, key)
        if why:
            print(red(f"  {why}. Choose another."))
            return
        other = next((x for x, k in keys.items() if k == key and x != act), None)
        if other:
            new = keys.get(act) if key_ok(other, keys.get(act)) is None else next(
                k for k in KEY_CHOICES if k not in keys.values() and k != key)
            keys[other] = new
            print(dim(f"  {other} moved to {keyname(new)}"))
        keys[act] = key
        save_config_key("keys", keys)
        self.refresh_keymap()
        print(green(f"  {act} → {keyname(key)} (saved)"))

    def system_cmd(self, arg):
        if arg == "reset":
            SYSTEM_FILE.write_text(SYSTEM_DEFAULT)
            print(green("  System prompt reset to the default."))
            return
        if arg == "edit":
            editor = os.environ.get("VISUAL") or os.environ.get("EDITOR") or "nano"
            subprocess.run([*editor.split(), str(SYSTEM_FILE)])
            print(green(f"  Saved {short_path(SYSTEM_FILE)}") + dim(" — it applies from your next message"))
            return
        full = self.a.system_prompt()
        base = system_template().replace("{date}", "…").replace("{cwd}", "…").replace("{root}", "…").rstrip()
        n_base = len(base.splitlines())
        lines = full.splitlines()
        print(bold("  System prompt") + dim(f"  ·  {len(full):,} characters (~{len(full) // 3.3 / 1000:.1f}k tokens)"
                                             f"  ·  editable part: {short_path(SYSTEM_FILE)}"))
        import textwrap
        width = max(term_width() - 4, 40)
        for i, line in enumerate(lines):
            if i == n_base:
                print(dim("  ── added automatically (loaded files, helpers, plan, wiki, autonomous mode) ──"))
            indent = "    " if line.lstrip().startswith("- ") else "  "
            for part in textwrap.wrap(line, width, subsequent_indent=indent[2:] + "  ") or [""]:
                print("  " + (part if i < n_base else dim(part)))
        print(dim("  /system edit to change it · /system reset for the default"))

    def call_cmd(self, line):
        m = re.match(r"call\s+-{1,2}agents?\b\s*(\S*)\s*(.*)$", line, re.I | re.S)
        roles = load_roles()
        name, task = (m.group(1), m.group(2).strip()) if m else ("", "")
        if not m or name not in roles or not task:
            if m and name and name not in roles:
                print(red(f"  No helper called '{name}'."))
            print(bold("  call -agent <helper> <task>") + dim("   e.g. call -agent coder add a footer to test/index.html"))
            for n, r in roles.items():
                print(f"  {hexc(r['color'], f'{n:<12}', True)} {dim(role_model(r, self.a.cfg) + ' · ' + r['kind'] + ' ·')} {r['description']}")
            return
        result = self.a.delegate(name, task, source="you")
        self.a.pending.append(f"[I ran the {name} helper myself with this task: {task}]\n{result}")
        print(dim("  Chicken will see this result with your next message."))

    def agents_cmd(self, arg):
        a = self.a
        parts = arg.split()
        if parts[:1] == ["mode"]:
            if len(parts) < 2 or parts[1] not in AGENT_MODES:
                for k, v in AGENT_MODES.items():
                    print(f"  {green(k) if k == a.cfg['agents_mode'] else k:<8} {v}")
                return
            a.cfg["agents_mode"] = parts[1]
            save_config_key("agents_mode", parts[1])
            print(green(f"  Helpers: {parts[1]}") + dim(f" — {AGENT_MODES[parts[1]]} (saved)"))
            return
        if parts[:1] == ["vram"]:
            loaded = a.vram.loaded()
            used = sum(loaded.values())
            print(bold(f"  GPU: {used:.1f} GB used by models · limit {a.vram.budget():.1f} GB ({a.cfg['vram_budget_gb']})"))
            for m, gb in loaded.items():
                ctx = a.vram.ctx_of.get(m)
                print(f"  {m:<24} {gb:5.1f} GB" + (f"  {ctx // 1024}k context" if ctx else "") +
                      (dim("  (on the CPU)") if gb == 0 else dim("  (partly on the CPU)") if a.vram.partial.get(m) else ""))
            if not loaded:
                print(dim("  No model loaded."))
            print(dim("  Measured sizes: " + ", ".join(f"{k.split('@')[0]} {v:g}" for k, v in sorted(a.vram.known.items()))))
            return
        if parts[:1] == ["resume"]:
            if self.worker.busy():
                print(yellow("  Wait for the current work to finish (or press Esc) first."))
                return
            text = a.resume_run()
            if not text:
                print(dim("  No stopped job to resume."))
                return
            print(dim(f"  Restored the job in {short_path(a.run['dir'])} ({len(a.run['steps'])} step(s) done)"))
            self.enqueue("turn", text, "(resume the stopped job)")
            return
        roles = load_roles()
        if parts[:1] == ["list"] or not (sys.stdin.isatty() and TTY):
            print(bold(f"  Helpers") + dim(f" · mode: {a.cfg['agents_mode']} — {AGENT_MODES[a.cfg['agents_mode']]}"))
            for n, r in roles.items():
                print(f"  {hexc(r['color'], f'{n:<12}', True)} {role_model(r, a.cfg):<20} {r['kind']:<7} {dim(r['description'])}")
            return
        models = self.model_meta()
        res = agents_editor(roles, a.cfg["agents_mode"], models)
        if res is None:
            print(dim("  No changes."))
            return
        cur, mode = res
        changed = []
        for n, v in cur.items():
            for k in ("model", "kind"):
                if v[k] != roles[n][k]:
                    set_role_meta(roles[n], k, v[k])
                    changed.append(f"{n} {k} → {v[k]}")
        if mode != a.cfg["agents_mode"]:
            a.cfg["agents_mode"] = mode
            save_config_key("agents_mode", mode)
            changed.append(f"mode → {mode}")
        print(green("  Saved: " + ", ".join(changed)) if changed else dim("  No changes."))

    def model_meta(self):
        """Installed chat models with what they can do and their GPU size: [(name, meta, capabilities)]."""
        out_ = []
        for m in self.model_names():
            if self.a.vram.servers.spec(m) and not self.a.vram.servers.spec(m).get("gpu", True):
                out_.append((m, f"{'CPU':<8} {self.a.vram.servers.spec(m).get('description', '')}", {"completion"}))
                continue
            if self.a.vram.servers.spec(m):
                cp = {"completion", "tools", "thinking"} | ({"vision"} if self.a.vram.servers.spec(m).get("mmproj") else set())
            else:
                try:
                    cp = set(requests.post(self.a.cfg["host"] + "/api/show", json={"model": m}, timeout=5).json()
                             .get("capabilities") or [])
                except Exception:
                    cp = set()
            if cp and "completion" not in cp:
                continue            # embedding models can't be helpers
            known = [v for k, v in self.a.vram.known.items() if k.startswith(m + "@")]
            size = f"{known[0]:.1f} GB" if known else f"~{self.a.vram.need(m, 8192):.1f} GB"
            tags = sorted(cp - {"completion", "insert"})
            if TOOLS_TESTED.get(m) is False:
                tags = [t if t != "tools" else "tools✗ (failed test)" for t in tags]
            elif TOOLS_TESTED.get(m):
                tags = [t if t != "tools" else "tools✓" for t in tags]
            out_.append((m, f"{size:<8} {' · '.join(tags)}", cp))
        return out_

    def restyle(self):
        apply_theme(self.a.cfg)
        self.session.style = MENU_STYLE
        self.invalidate()

    def color(self, arg):
        cfg = self.a.cfg
        arg = arg.strip().lower()
        names = list(PROMPT_COLORS)
        if not arg:
            cur = cfg.get("prompt_color") or "theme"
            styles = [f"{PROMPT_COLORS[n] or prompt_hex({**cfg, 'prompt_color': ''})} bold" for n in names]
            choice = menu("Prompt arrow colour", [(f"❯ {n}", "● current" if n == cur else PROMPT_COLORS[n] or "the theme's own")
                                                  for n in names],
                          styles=styles, start=names.index(cur) if cur in names else 0,
                          hint="↑↓ move · Enter choose · Esc close · or /color #rrggbb")
            if choice is None:
                return
            arg = names[choice]
        if arg in ("reset", "default"):
            arg = "theme"
        if arg not in PROMPT_COLORS and not re.fullmatch(r"#[0-9a-f]{6}", arg):
            print(yellow("  Usage: /color <name|#rrggbb>   colours: " + ", ".join(names)))
            return
        cfg["prompt_color"] = "" if arg == "theme" else arg
        save_config_key("prompt_color", cfg["prompt_color"])
        self.restyle()
        print(arrow(cfg) + dim(f"prompt arrow is now {arg} (saved)"))

    def theme(self, arg):
        cfg = self.a.cfg
        arg = arg.strip().lower()
        names = list(THEMES)
        if not arg:
            cur = cfg.get("theme", "terminal")
            styles = [f"{THEMES[n].get('accent', '#5fafd7')} bg:{THEMES[n]['bg']}" if "bg" in THEMES[n] else "bold"
                      for n in names]
            choice = menu("Colour theme", [(f" {n} ", ("● current · " if n == cur else "") + THEMES[n]["desc"])
                                           for n in names],
                          styles=styles, start=names.index(cur) if cur in names else 0,
                          hint="↑↓ move · Enter apply · Esc close")
            if choice is None:
                return
            arg = names[choice]
        if arg not in THEMES:
            print(yellow("  Usage: /theme <name>   themes: " + ", ".join(names)))
            return
        cfg["theme"] = arg
        save_config_key("theme", arg)
        self.restyle()
        print(green(f"  ✓ Theme '{arg}'") + dim(f" — {THEMES[arg]['desc']} (saved)"))
        print("  " + "  ".join(f(n) for f, n in ((red, "red"), (yellow, "yellow"), (green, "green"), (cyan, "cyan"),
                                                 (magenta, "magenta"))) + "  " + arrow(cfg) + dim("prompt"))
        if "bg" in THEMES[arg]:
            print(dim("  If the background didn't change, your terminal doesn't allow it — the text colours still apply."))

    def temperature(self, arg):
        a = self.a
        if not arg:
            print(f"  Temperature: {a.cfg['temperature']}" + dim("  (0.6 is recommended for qwen3 with reasoning, 0.7 without)"))
            return
        try:
            t = float(arg.replace(",", "."))
            assert 0 <= t <= 2
        except (ValueError, AssertionError):
            print(red("  Use a number between 0 and 2, e.g. /temperature 0.6"))
            return
        a.cfg["temperature"] = t
        save_config_key("temperature", t)
        print(green(f"  Temperature set to {t} (saved)"))

    def add_files(self, arg):
        a = self.a
        if not arg:
            print(yellow("  Usage: /file <path> [more paths]"))
            return
        for p in arg.split():
            try:
                f = a.resolve(p)
            except ToolError as e:
                print(red(f"  {e}"))
                continue
            if f.is_dir():
                a.focus.append(f)
                a.pending.append(f"I want to work in the folder {a.rel(f)}:\n{a.t_list_dir(str(f), 2)}")
                print(green(f"  + {a.rel(f)}/") + dim(" added"))
                continue
            if not f.is_file():
                print(red(f"  Not found: {p}"))
                continue
            if f not in a.focus:
                a.focus.append(f)
            try:
                content = a.t_read_file(str(f))
            except ToolError as e:
                print(red(f"  {e}"))
                continue
            a.pending.append(f"I want to work on this file:\n{content}")
            print(green(f"  + {a.rel(f)}") + dim(f" added ({content.count(chr(10))} lines)"))
        if a.pending:
            print(dim("  Now tell the AI what to do with it."))

    def shell(self, command):
        if not command:
            return
        proc = subprocess.run(command, shell=True, cwd=self.a.cwd, executable="/bin/bash",
                              capture_output=True, text=True, errors="replace")
        text = (proc.stdout + proc.stderr).rstrip()
        if text:
            print(text)
        if proc.returncode:
            print(dim(f"  (exit code {proc.returncode})"))
        self.a.pending.append(f"I ran this command myself: `{command}` (exit code {proc.returncode})\nOutput:\n{clip(text, 4000)}")

    def loop(self, arg):
        max_steps = None
        m = re.search(r"--max[= ](\d+)", arg)
        if m:
            max_steps = int(m.group(1))
            arg = (arg[:m.start()] + arg[m.end():]).strip()
        if arg in ("continue", "c", ""):
            if not self.last_loop_goal:
                print(yellow("  Usage: /loop <goal>   e.g. /loop build a python script that renames my photos by date"))
                return
            arg = self.last_loop_goal
            print(dim(f"  Continuing: {arg}"))
        self.last_loop_goal = arg
        self.a.run_loop(arg, max_steps)

    def run_task(self, name, arg):
        t = self.tasks[name]
        target = arg or f"the current directory ({self.a.rel(self.a.cwd)})"
        body = t["body"].replace("{args}", target)
        if "{args}" not in t["body"] and arg:
            body += f"\n\n{arg}"
        if t["loop"]:
            self.last_loop_goal = body
            self.a.run_loop(body)
        else:
            self.a.run_turn(body)

    def model(self, arg):
        a = self.a
        self._models = None
        models = self.model_names()
        if not models:
            print(red("  Could not list models — is Ollama running?"))
            return
        if not arg:
            meta = {m: d for m, d, _ in self.model_meta()}
            items = [(m, ("● current · " if m == a.cfg["model"] else "") + meta.get(m, "")) for m in models if m in meta]
            names = [m for m, _ in items]
            choice = menu("Main agent model", items, hint="↑↓ move · Enter switch · Esc close",
                          start=names.index(a.cfg["model"]) if a.cfg["model"] in names else 0)
            if choice is None:
                return
            arg = names[choice]
        if arg not in models and arg + ":latest" not in models:
            print(red(f"  Model {arg} is not installed. Install with: ollama pull {arg}"))
            return
        a.base_model = arg
        a.cfg["model"] = arg if a.cfg.get("routing") != "bonsai-only" else a.cfg["heavy_model"]
        a.think = a.cfg["think"]
        save_config_key("model", arg)
        a.cfg["num_ctx"] = a.full_ctx()
        if a.cfg.get("routing") == "bonsai-only":
            print(dim(f"  (bonsai-only is on: {a.cfg['heavy_model']} stays the orchestrator until /mode orchestrator)"))
        where = "local server, started when needed" if a.vram.servers.spec(arg) else "Ollama"
        print(green(f"  Main agent: {arg}") + dim(f" ({where}, saved as default)"))
        if TOOLS_TESTED.get(arg) is False:
            print(yellow(f"  ! {arg} failed the tool test here: it can't read or edit files properly as the main agent."))

    def progress(self, what):
        """Returns report(percent, detail): shows live progress on the status line while saving."""
        def report(pct, detail=""):
            ui.set_label(f"{what} {pct:3.0f}% {progress_bar(pct)}" + (f" · {detail}" if detail else ""))
            self.invalidate()
        return report

    @staticmethod
    def split_wiki_flag(arg):
        first, _, rest = arg.partition(" ")
        return (True, rest.strip()) if first.lower() in WIKI_FLAGS else (False, arg)

    def save(self, arg):
        wiki, name = self.split_wiki_flag(arg)
        a = self.a
        if not a.messages:
            print(dim("  Nothing to save yet."))
            return
        if wiki:
            name = re.sub(r"[^\w.-]", "_", name) or (a.wiki.name if a.wiki else time.strftime("%Y%m%d-%H%M%S"))
            exists = (WIKI_DIR / name / "index.md").exists()
            print(dim(f"  {'Updating' if exists else 'Creating'} wiki '{name}' in {short_path(WIKI_DIR / name)} "
                      f"(esc stops; pages already written are kept)"))
            try:
                r = a.save_wiki(name, self.progress("Saving wiki"))
            except Interrupted:
                print(yellow(f"  Wiki save stopped. The pages written so far are in {short_path(WIKI_DIR / name)}"))
                return
            print(green(f"  ✓ Wiki '{name}' saved 100% {progress_bar(100)}") + dim(
                f"  {len(r['written'])} page(s) written, {len(r['kept'])} unchanged · {_size(r['bytes'])} · "
                f"loads in ~{r['tokens'] / 1000:.1f}k tokens (log: ~{messages_chars(a.messages) / 3300:.1f}k)"))
            print(dim(f"  /load -wiki {name} to resume from it"))
            return
        name = re.sub(r"[^\w.-]", "_", name) or time.strftime("%Y%m%d-%H%M%S")
        report = self.progress("Saving log")
        raw, disk = write_log(SESS_DIR / f"{name}.json.gz", a.snapshot(), report)
        (SESS_DIR / f"{name}.json").unlink(missing_ok=True)
        print(green(f"  ✓ Saved log '{name}' 100% {progress_bar(100)}") +
              dim(f"  {_size(disk)} on disk ({_size(raw)} uncompressed, -{100 - 100 * disk / max(raw, 1):.0f}%)"))
        print(dim(f"  /load {name} to resume"))

    def load(self, arg):
        wiki, name = self.split_wiki_flag(arg)
        a = self.a
        if wiki:
            names = wiki_names()
            if not name:
                if not names:
                    print(dim("  No wikis yet. Create one with /save -wiki <name>"))
                    return
                items = [(n, time.strftime('%d %b %H:%M', time.localtime((WIKI_DIR / n / "index.md").stat().st_mtime)))
                         for n in names]
                choice = menu("Saved wikis", items, hint="↑↓ move · Enter load · Esc close")
                if choice is None:
                    return
                name = names[choice]
            if name not in names:
                print(red(f"  No wiki '{name}'"))
                return
            pages, tokens = a.load_wiki(name)
            print(dim(f"  Loaded wiki '{name}' ({pages} pages) in {short_path(a.cwd)} · ~{tokens / 1000:.1f}k tokens in "
                      f"context — the AI opens the other pages only when it needs them"))
            return
        files = session_files()
        if not name:
            recent = sorted(files.items(), key=lambda kv: kv[1].stat().st_mtime, reverse=True)[:20]
            if not recent:
                print(dim("  No saved conversations."))
                return
            choice = menu("Saved conversations",
                          [(n, time.strftime('%d %b %H:%M', time.localtime(f.stat().st_mtime))) for n, f in recent],
                          hint="↑↓ move · Enter load · Esc close")
            if choice is None:
                return
            name = recent[choice][0]
        if name not in files:
            print(red(f"  No saved conversation '{name}'"))
            return
        before, after = a.load_log(files[name])
        saved = f" · outdated outputs trimmed: ~{before / 1000:.1f}k → ~{after / 1000:.1f}k tokens" if after < before * 0.97 else ""
        print(dim(f"  Loaded '{name}' ({len(a.messages)} messages) in {short_path(a.cwd)}{saved}"))


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(prog="chicken", description="Chicken: a local AI agent for your files (Ollama).")
    ap.add_argument("prompt", nargs="*", help="Start with this request")
    ap.add_argument("-p", "--print", action="store_true", help="Answer once and exit (no interactive session)")
    ap.add_argument("-l", "--loop", action="store_true", help="Treat the prompt as a goal and work autonomously")
    ap.add_argument("-m", "--model", help="Model to use for this session")
    ap.add_argument("--auto", action="store_true", help="Auto-approve all actions (careful!)")
    args = ap.parse_args()

    ensure_setup()
    cfg = load_config()
    if args.model:
        cfg["model"] = args.model
    agent = Agent(cfg, auto=args.auto)
    agent.apply_mode()
    ui.agent = agent
    prompt = " ".join(args.prompt).strip()

    if args.print or not sys.stdin.isatty():
        if not prompt:
            prompt = sys.stdin.read().strip()
        if not prompt:
            ap.error("no prompt given")
        try:
            try:
                agent.run_loop(prompt) if args.loop else agent.run_turn(prompt)
            finally:
                agent.vram.servers.stop_started()
        except (Interrupted, KeyboardInterrupt):
            sys.exit(130)
        except RuntimeError as e:
            print(red(str(e)), file=sys.stderr)
            sys.exit(1)
        return

    if not HAVE_PT:
        sys.exit("chicken needs prompt_toolkit: ~/agent/.venv/bin/pip install prompt_toolkit")
    repl = Repl(agent)
    if prompt and args.loop:
        prompt = "/loop " + prompt
    repl.run(prompt or None)


if __name__ == "__main__":
    main()
