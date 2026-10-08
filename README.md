# Chicken

**A local AI agent that sizes memory to the task, runs its agents one at a time, and gets fast answers from small quantized models.**

Chicken is an AI agent for the terminal that works directly on your files, much like `claude`, and runs entirely on
your own machine: nothing leaves it except the web searches you ask for. Instead of loading one huge model and hoping
it fits, Chicken treats GPU memory as a resource to be scheduled:

- **Memory sized to the task.** Each step gets exactly the model and the context it needs, and nothing more.
- **One agent at a time.** Agents never run in parallel, so each one gets the whole GPU while it works.
- **Smart memory allocation.** Models are loaded, shrunk, compacted and ejected as the task moves on.
- **Fast answers.** A small, fast model decides what happens next; bigger models are called only when the task needs them.
- **Quantized models.** Every model runs fully on the GPU, never split onto the CPU.

The architecture is described in [docs/DESIGN.md](docs/DESIGN.md).

## Models: tuned for a 12 GB GPU, yours to change

Chicken is optimized with the models below for a **12 GB VRAM** card (developed on an RTX 3060 12 GB, with a 90%
memory limit). None of them is hard-wired: every model can be replaced to match your hardware (see
[Using other models](#using-other-models)).

| Role | Default model | Runs on | Why |
|---|---|---|---|
| Orchestrator (decides only) | `qwen3.5:4b` | Ollama | Small and fast: routing, plans, structured JSON |
| Planner, hard coding, review, recovery | `bonsai2:27b` (PrismML Bonsai 2 27B, 1.75-bit ternary Qwen3.8-27B) | llama.cpp (PrismML build) | A 27B model in 7.4 GB: the best results we measured |
| Coder | `qwen2.5-coder:14b` | Ollama | Code writing, run as a *writer* (Chicken applies its edits) |
| Vision | `gemma3:12b` | Ollama | Images and screenshots |
| Summarizer | `qwen2.5:7b` | Ollama | Long documents, low stakes |
| OCR | `glm-ocr`, `maternion/LightOnOCR-2` | Ollama | Document parsing |
| Trivial tasks | `qwen3.5:2b` | llama.cpp | Very fast short answers |
| `/btw` side questions | `qwen3.5:0.8b` | llama.cpp | Read-only, tiny, can stay loaded beside a worker |
| Classic mode main agent | `qwen3:14b` | Ollama | The model Chicken used before the orchestrator |

### Measured: Bonsai 2 27B vs Qwen3 14B

Bonsai 2 27B replaced Qwen3 14B as the model for hard work after this comparison, run on the same 12 GB card at a
24k-token context:

| | bonsai2:27b | qwen3:14b |
|---|---|---|
| Hard coding tasks (4, graded by tests), fast | **4/4** in 74 s | 2/4 in 27 s |
| Hard coding tasks (4, graded by tests), reasoning on | **4/4** in 269 s | 3/4 in 781 s |
| Tool calls | 5/5 | 5/5 |
| Read → edit loops | 3/3 | 3/3 |
| Speed | 28 tok/s | 34 tok/s |
| GPU memory at 24k context | **7.4 GB** | 11.1 GB |

Bonsai solved every task while using a third less memory than the 14B model, which is what makes room for the
orchestrator and the small models beside it. Alone in a 10 GB budget it gets about 57k tokens of context (only 1 layer
in 4 keeps a context cache), against about 11k for qwen3:14b.

### Using other models

Everything is set in `~/.agent/config.json` (created on the first run) and in the helper files:

- `model`: the orchestrator. `heavy_model`: planner and hard work. `btw_model`: the `/btw` model.
- `models`: add your own models to the list the orchestrator chooses from:
  ```json
  "models": {
    "phi4:14b": ["coding, reasoning", "high", "medium"],
    "llama3.2:3b": ["trivial tasks", "low", "very fast"]
  }
  ```
  Only models that are installed and fit fully under your GPU limit are offered, so on a smaller card Chicken simply
  skips the ones that don't fit.
- `servers`: models served by llama.cpp instead of Ollama (binary, GGUF file, port). Chicken starts and stops them.
- `vram_budget_gb`: the GPU limit, as a share of the card (`"90%"`) or a number of GB.
- `~/.agent/agents/<role>.md`: the model, kind and prompt of each helper role.
- Inside Chicken: `/model` to switch the main model, `/agents` to change helpers' models from a table.

On an 8 GB card, for example, keep `qwen3.5:4b` as orchestrator and use 7B-8B quantized models as workers; on a 24 GB
card you can make larger models the heavy model.

## Install

You need Linux, Python 3.10+, an NVIDIA GPU and [Ollama](https://ollama.com). Then:

```bash
git clone https://github.com/alemarzosw/Chicken.git
cd Chicken
./install.sh                     # private Python environment + the `chicken` and `call` commands
ollama pull qwen3.5:4b           # the orchestrator; pull the workers you want too
```

Bonsai 2 27B and the two small `qwen3.5` models run on llama.cpp: put PrismML's llama.cpp build in
`~/.local/share/prism-llama/bin/` and the GGUF files in `~/.local/share/prism-llama/models/` (paths and ports are in
`servers` in the config). Without them, Chicken works with the Ollama models you have.

## Using Chicken

### Orchestrator (default)

- **The orchestrator** (`model`, default qwen3.5:4b) has only two tools: `call_agent` (one role, one model, a stand-alone
  task, the reason and the complexity) and `update_plan`. It never touches files, the web or the shell.
- **Multi-step goals** go to the **planner** (Bonsai) first. Its JSON plan goes into the task state, and the orchestrator
  runs the steps one by one.
- **Workers** start with an empty memory, do one task and end with a JSON report (`status`, `summary`, `artifacts`,
  `important_facts`, `blocked_by`, `errors`, `suggested_followup`). The orchestrator reads a compact version; the full
  report stays in the run folder.
- **Validators** check every file a worker changed: Python must compile, JSON must parse, shell scripts must pass
  `bash -n`. A failure turns the step into `needs_review` and is shown to the orchestrator.
- **Task state** (`~/.agent/runs/<id>/state.json`) is authoritative: goal, plan with step status and model, artifacts,
  last result, errors, calls. Every decision is also logged in `~/.agent/logs/decisions.jsonl` for tuning the router later.
- **Escalation:** the orchestrator is told to escalate to Bonsai for hard work; after two failures of the same role the
  program adds "escalate to bonsai2:27b" to the result.
- **The model registry** (capabilities, intelligence, speed) is built in, plus your own `models` from the config. Only
  installed models that fit fully under the GPU limit are offered.

```
▸ planner  Plan: create hello.py and review it
    bonsai2:27b · medium · multi-step
└ ✓ planner  12s · Make hello.py and review it
● chicken  plan
    ○ 1. Create hello.py  coder
    ○ 2. Review hello.py  reviewer
▸ coder  Create hello.py printing hello
    bonsai2:27b · low · step 1 · coding step
└ ! coder  21s · Created hello.py
    ✗ hello.py '(' was never closed (hello.py, line 1)
```

| `/mode` | What it does |
|---|---|
| `ask` | Confirm each worker (Yes · Yes for the session · No, or type `/model <name>` or `/role <name>` to change it) and each change |
| `auto` | No confirmations for workers or actions |
| `orchestrator` | Default: the main model only decides |
| `bonsai-only` | Bonsai is the orchestrator and every worker |
| `bonsai-workers` | The orchestrator stays, every worker runs on Bonsai |
| `classic` | The old Chicken: the main model works itself with all tools and calls helpers |

**Knowledge wikis (optional).** `~/.agent/Built_skills_and_knowledge/` can hold one wiki per subject (one folder each,
with an `index.md`). Write each page from primary sources and list them on the page.
- Before each worker starts, Chicken searches the wikis with the worker's task (BM25 keyword search on the CPU, no
  GPU memory) and adds up to 3 clearly relevant sections (max `knowledge_chars`, 6000 characters) with their sources,
  plus the rule that the user's files, the project's docs and real command output take precedence.
- Tool workers also get `search_knowledge(query, wiki)`; the orchestrator sees the list of wikis.
- `/knowledge` lists the wikis, `/knowledge <question>` searches, `/knowledge auto off` stops adding sections.
- Add your own: a folder with `index.md` and one `.md` page per topic.

**GPU only, 90% limit.** Chicken never runs a model partly on the CPU: if Ollama puts a layer there, Chicken clears the
GPU and retries once, and otherwise refuses to run it. After each load it reads the real GPU memory; above 90% of the
card it pauses, shrinks the main agent's context (if the conversation still fits), and only then ejects models the
current step doesn't need. `/allocation` shows GPU memory by model (weights and context), other programs, the limit,
and RAM by program.

### Starting it

```bash
cd ~/Documents/my-project     # go to the folder you want to work in
call chicken                  # start Chicken (`chicken` works too)
```

`call chicken` (or `chicken`) takes the options in the table below (e.g. `call chicken -l "<goal>"`).
Inside Chicken, `call -agent <helper> <task>` runs a helper (see **Helpers**). In bash, `call -agent` only reminds you of this.
Inside the agent, what you type is coloured: `/commands` in blue, `-flags` in pink, `!shell` in yellow.

| Command | What it does |
|---|---|
| `call chicken` | Open an interactive session in the current folder |
| `call chicken "fix the typo in index.html"` | Open a session and start with this request |
| `call chicken -l "build a script that renames my photos by date"` | Start in autonomous mode (works until done) |
| `call chicken -p "what does main.py do?"` | Answer once and exit (useful in scripts) |
| `call chicken -m qwen2.5-coder:14b` | Use a different model for this session |
| `call chicken --auto` | Never ask permission (careful!) |

Type what you want in normal language, in Italian or English.

### The screen

```
 qwen3:14b · ~/project · ctx 12% (2.9k/25k) · RAM 11.2/27GB · GPU 52°C VRAM 10.5/12GB · temp 0.6 · 32.7 tok/s
❯ add a docstring to greet in hello.py
• Read hello.py · lines 1-4 of 4
• Edit hello.py
    lines 2:  +1 -0
      1   def greet(name):
      2 +     """Say hello to someone."""
      3       print("Hello " + name)
  └ Edited hello.py (1 replacement)
Added a docstring to the greet function.
✓ Done  41s · 812 tokens · 32.7 tok/s · ctx 13%
⠧ Thinking… 7s · 33.3 tok/s · ctx 8% · RAM 3.7/27GB · GPU 56°C · temp 0.6   ctrl+b details · ctrl+c stop
 / commands · ctrl+b details · ctrl+t plan · ctrl+o last output · ctrl+g info · ctrl+c ×2 quit
```

- **Status line** above your input: model, folder, context (the AI's memory) used, RAM, GPU temperature and VRAM, temperature setting, speed of the last answer.
- **Live line** while it works: what it's doing right now (Thinking / Writing / Reading x / Running y), seconds elapsed, tokens per second and the same system stats.
- **Actions** are listed one per line (`• Read…`, `• Edit…`, `• Run…`), with results under `└`.
- **✓ Done footer** after each answer: helpers used, time, tokens generated, speed and context used (`✻` instead when it was stopped).
- **Thinking** lines (`┊`, shown with `ctrl+b details`) take the colour of the agent thinking: cyan for Chicken, the helper's own colour for a helper.
- **Context compaction** shows as `↻ context 72% → 26% · auto-compacted`.
- **Hint bar** at the bottom: the main shortcuts.

### While the AI is working

The input line stays active the whole time, so you never have to wait:

- **Send more messages.** They wait in a queue (shown as `↳ queued: …` above your input) and run one after another as each finishes. **↑** on an empty line takes the last queued message back so you can edit it. `/queue` lists the queue and `/queue clear` empties it.
- **Run commands.** `/plan`, `/infocomputer`, `/context`, `/temperature`, `/details`, `/think`, `/model`, `/help`, `/tasks`, `/copy`, `/auto`, `/setkeyboard`, `/config` and `/btw` run immediately. Commands that change the conversation (`/file`, `/drop`, `/cd`, `/clear`, `/load`) and new work (`/save`, `/search`, `/loop`, tasks, `!commands`) join the queue.
- **`/btw <question>`** asks a quick side question. A small read-only model on the GPU (**qwen3.5:0.8b**) answers from a copy of the task state and the recent conversation, even while a worker runs. It stays loaded while there is room under the limit; if the GPU is full it says so instead of evicting the worker. It never changes anything.
- **Permission questions** appear right above your input. Press 1 2 3 (or ↑↓ + Enter). Whatever you were typing is kept.
- **Esc** (or **Ctrl+C**) interrupts. The AI stops immediately and won't continue the interrupted request later. Queued messages stay in the queue.

### Commands

Type **`/`** and a menu shows every command with its description. Keep typing to filter, use ↑↓ to choose, and press Enter.
Arguments complete too: file paths, model names, `on`/`off`…

Type **`/help`** to open the **command browser**. ↑↓ moves between commands and shows each one's full description and an example. Type to filter, Enter puts the command in your prompt, Esc closes it.

| Command | What it does |
|---|---|
| `/help` | Browse all commands and shortcuts with descriptions |
| `/file <path> [...]` | Load files or folders to work on. Their content is sent to the AI |
| `/files` · `/drop [path]` | Show loaded files · stop working on one (or all) |
| `/search <query>` | Search the web and get an answer with source links |
| `/btw <question>` | Read-only side question, answered by a small GPU model from the task state |
| `/mode [name]` | Show or switch: ask, auto, orchestrator, bonsai-only, bonsai-workers, classic |
| `/allocation` | GPU memory by model (weights + context), the 90% limit, free memory, RAM by program |
| `/knowledge [question]` | List or search the knowledge wikis; `auto on\|off` for adding sections to workers' tasks |
| `/queue [clear]` | Show or clear the messages waiting to run |
| `/loop <goal> [--max N]` | **Autonomous mode**: plan → work → check → repeat until done |
| `/loop continue` | Resume the last loop |
| `/plan` | Show the AI's current plan (checklist) |
| `/tasks` | List preset tasks (see below) |
| `/cd <dir>` | Change working folder |
| `/model [name]` | Choose Chicken's main model from a menu (Ollama models and bonsai2:27b), or switch directly. Saved as default |
| `/think [on\|off]` | Reasoning on (smarter, slower) or off (much faster) |
| `/temperature [0-2]` | Show or set creativity: low = precise, high = creative. Default 0.6 |
| `/details [on\|off]` | Show/hide reasoning and full outputs (same as ctrl+b) |
| `/infocomputer` | CPU, GPU, VRAM, RAM, disk, loaded model, GPU/CPU split, context, speed |
| `/context` · `/compact [auto on\|off]` | How full the memory is · summarize now to free it · `auto off` stops compacting just to fit a helper |
| `/clear` | Start a fresh conversation |
| `/copy` | Copy the AI's last answer to the clipboard |
| `/save [name]` · `/load [name]` | Save or resume the full conversation (log). `/load last` = the automatic save |
| `/save -wiki [name]` · `/load -wiki [name]` | Save the project as a wiki of .md pages · load its knowledge (few tokens) |
| `/auto` | Toggle auto-approve for everything |
| `/setkeyboard` | Change keyboard shortcuts (interactive table) |
| `/color [name\|#hex]` | Colour of the `❯` prompt arrow: a list shown in the real colours, or e.g. `/color pink`, `/color #ff8700` |
| `/theme [name]` | Colour theme: chicken (default, the promo's palette), terminal, dracula, catppuccin, nord, gruvbox, monokai, solarized, light |
| `/system [edit\|reset]` | Show the exact system prompt sent to the model. `edit` opens `~/.agent/system.md` in your editor (nano by default); `reset` restores the default |
| `/config` | Show settings |
| `!<command>` | Run a shell command yourself. The AI sees the output (e.g. `!git status`) |
| `/exit` | Quit |

### Colours

- **`/color`** changes only the `❯` prompt arrow. Pick from the list (each name is shown in its colour), or type a name or `#rrggbb`. `/color theme` goes back to the theme's colour.
- **`/theme`** changes every colour of the agent: status line, menus, messages, the shortcut table and command colouring.
  Themes other than `terminal` also set the terminal's **background and text colour** while the agent runs, and restore them when you quit.
  Most terminals allow this (Windows Terminal, kitty, iTerm2, GNOME Terminal, xterm…). If yours doesn't, the text colours still change.

Both are saved as defaults.

### Keyboard shortcuts

| Key | What it does |
|---|---|
| **ctrl+b** | Show/hide **details**: the AI's reasoning, streamed live under the status line, plus full tool results and complete command output. Also works while it's working |
| **ctrl+t** | Show the current plan (also while it works) |
| **ctrl+o** | Show the full output of the last action (file read, search, command…) |
| **ctrl+g** | Computer & AI info (same as `/infocomputer`) |
| **ctrl+v** | Paste |
| **esc** | Interrupt the AI while it works |
| **ctrl+c** | Copy selected text (select with shift+arrows) · clear the line · stop the AI while it works · **twice** on an empty line: quit |
| ↑ (empty line, while working) | Take the last queued message back to edit it |
| ctrl+d | Quit |
| tab | Complete commands, paths and options |
| ↑ ↓ | Previous messages |
| alt+enter (changeable), or `\` at the end of a line | New line. Change it in `/setkeyboard` (row *newline*): alt+enter, shift+enter, ctrl+j or any ctrl+letter |

**Shift+Enter for a new line.** Most terminals send exactly the same thing for Shift+Enter as for Enter, so no program
can tell them apart. Chicken understands the two codes that terminals send when they *do* tell them apart
(`ESC[13;2u` and `ESC[27;2;13~`). To check yours: `/setkeyboard newline` → **Push key** → press Shift+Enter.
If it says "plain Enter", your terminal doesn't send it; use alt+enter or ctrl+j, or set up the terminal:

| Terminal | Setting |
|---|---|
| Windows Terminal | Settings → Open JSON file, add to `"actions"`: `{ "command": { "action": "sendInput", "input": "\u001b[13;2u" }, "keys": "shift+enter" }` |
| VS Code terminal | keybindings.json: `{ "key": "shift+enter", "command": "workbench.action.terminal.sendSequence", "args": { "text": "\u001b[13;2u" }, "when": "terminalFocus" }` |
| iTerm2 | Settings → Profiles → Keys → Key Mappings → + → Shift+Enter → Send Escape Sequence → `[13;2u` |
| kitty | kitty.conf: `map shift+enter send_text all \x1b[13;2u` |
| WezTerm | `keys = { { key = "Enter", mods = "SHIFT", action = wezterm.action.SendString("\x1b[13;2u") } }` |
| MobaXterm, PuTTY | send a plain Enter; use alt+enter or ctrl+j |

When Shift+Enter is chosen, alt+enter keeps working as a backup.

**Changing shortcuts:** run `/setkeyboard` to open an interactive table:

```
 Keyboard shortcuts   preset: Default
 ╭─────────────┬──────────────────────────────────────┬──────────────────────────────╮
 │ Action      │ Shortcut                             │ What it does                 │
 ├─────────────┼──────────────────────────────────────┼──────────────────────────────┤
 │❯details     │ [ ctrl+b ▾ ] [ Push key ] [ Reset ]  │ Show/hide details: …         │
 │ plan        │   ctrl+t                             │ Show the AI's current plan   │
 ╰─────────────┴──────────────────────────────────────┴──────────────────────────────╯
  [ Preset: Default ▾ ]  [ Reset all ]  [ Save ]  [ Cancel ]
```

- **↑↓** picks a row. **←→** (or Tab) picks a box. **Enter** uses it.
- **`ctrl+b ▾`** opens the list of possible combinations. Each shows whether it's free, what typing action it replaces
  (e.g. `ctrl+a` = go to line start), or which shortcut it swaps with.
- **Push key**: press the new Ctrl+letter directly.
- **Reset** (or `r`) puts that row back to its default.
- **newline** is a row too: choose alt+enter (default), shift+enter (see below), ctrl+j (a real line feed, works in every terminal) or a ctrl+letter, or use Push key and press it. The hint bar and `/help` show the key you chose.
- **Preset** applies a whole set: *Default*, *Claude Code-like* (ctrl+o details, ctrl+b last output) or *Mnemonic* (ctrl+p plan, ctrl+o output).
- **Reset all**, **Save** (or `s`) and **Cancel** (or Esc). Changed rows are marked `•`, and nothing is saved until you choose Save.

You can also set one directly with `/setkeyboard plan ctrl+k`. `/setkeyboard reset` restores the defaults.
Ctrl+C, Ctrl+D, Ctrl+L, Ctrl+Z, Ctrl+S and Ctrl+Q are reserved.

**Clipboard over SSH:** copying uses your terminal's clipboard (OSC 52), which works in Windows Terminal, MobaXterm, iTerm2, kitty, WezTerm and others.
Your terminal's own copy and paste (usually ctrl+shift+c / ctrl+shift+v, or mouse selection) always work too.

### Permissions (like Claude Code)

Reading, searching and browsing happen freely. Anything that **changes** something shows exactly
what will change, with line numbers, and asks first:

```
• Edit todo.py
    lines 3-4:  +2 -1
      2   import sys
      3 - import os
      3 + import json
      4 + from pathlib import Path
 Apply edit to todo.py?  ↑↓ + Enter, or press 1 2 3 · Esc = no
 ❯ 1. Yes
   2. Yes, and don't ask again for file edits this session
   3. No, and tell the AI what to do instead
```

- **1** allows it once.
- **2** allows this kind of action (edits / writes / moves / deletions / commands) for the rest of the session.
- **3** blocks it, and you can type what it should do instead. The AI receives your note.

Terminal commands work the same way: the full command is shown before it runs, and its output appears live.

**Safety rules built in:**
- It can only touch files inside your home folder (`root` in the config). `~/.ssh`, `~/.gnupg` and its own config are protected.
- **Deleted files are not destroyed.** They're moved to `~/.agent/trash/`.
- In `/loop` mode it asks once whether to auto-approve file edits. Moves, deletions and commands still ask.
- It auto-corrects a common small-model mistake (re-typing lines that already exist after an edit) and warns you when an edit creates duplicated lines.
- When an edit **deletes** lines (for example a whole function), you get a yellow `! this edit DELETES …` warning before approving, and the AI is told about it afterwards.

### Preset tasks

Each preset is also a command:

| Command | What it does |
|---|---|
| `/review [path]` | Review files or a project: bugs, security, quality (read-only) |
| `/explain [path]` | Explain what a file or project does |
| `/summarize [path]` | Summarize documents (text, Markdown, PDF…) |
| `/research <topic>` | Search several web sources and write a report |
| `/organize [folder]` | Propose a tidier folder structure. Asks before moving anything |
| `/docs [path]` | Write or update a README |
| `/fix <problem>` | (autonomous) Find the cause, fix it, verify it |
| `/tests [path]` | (autonomous) Write tests, run them, fix them |
| `/continue [path]` | (autonomous) Continue a project until finished, keeping `PROGRESS.md` updated |

(autonomous) = runs in autonomous loop mode.

**Make your own:** create `~/.agent/tasks/<name>.md`, and it becomes `/<name>` (it also appears in the `/` menu):

```markdown
---
description: Translate a document to English
loop: false
---
Translate {args} to English and save it next to the original with the suffix _en.
```

`{args}` is replaced by whatever you type after the command. `loop: true` makes it autonomous.

### How autonomous mode works

1. The AI explores and writes a plan (checklist). You can view it any time with ctrl+t.
2. It works step by step: reads, edits, runs commands, searches the web.
3. After each round it is told to continue, until it declares the task done (or blocked).
4. It stops by itself at the step limit (80 by default), or if it stops making progress. `/loop continue` resumes.
5. The plan survives even when old messages are summarized to save memory.

The conversation is autosaved after every step (`/load last`).

### Bonsai 2 27B (optional main model)

`bonsai2:27b` is PrismML's 1.75-bit version of Qwen3.8-27B. Ollama can't run it, so Chicken starts PrismML's own
llama.cpp server when the model is needed, stops it to make room in the GPU, and stops it when you quit.
Pick it with `/model bonsai2:27b` (as the main agent) or in the `/agents` table (as a helper).

| | bonsai2:27b | qwen3:14b |
|---|---|---|
| Hard coding tasks (4, graded by tests) | 4/4 fast (74 s) · 4/4 reasoning (269 s) | 2/4 fast (27 s) · 3/4 reasoning (781 s) |
| Tool calls · read→edit loops | 5/5 · 3/3 | 5/5 · 3/3 |
| Speed | 28 tok/s | 34 tok/s |
| GPU memory at 24k context | 7.4 GB (under the 10 GB budget) | 11.1 GB |

Files: `~/.local/share/prism-llama/bin/` (PrismML's llama.cpp build for CUDA 12.8, using Ollama's CUDA libraries) and
`~/.local/share/prism-llama/models/Ternary-Bonsai-2-27B-PTQ1_0.gguf` (5.95 GB). Server logs: `~/.agent/servers/`.
Other llama.cpp-served models can be added under `servers` in `~/.agent/config.json`.

### Helpers (multi-agent)

Chicken, the main agent (the **boss**, qwen3:14b), can hand a step to a **helper** that starts with an empty memory. The helper
does the step and returns a short result. The boss checks it and decides what's next: another helper, the same helper
again with what to fix, or the answer to you. Each result is passed on to the next helper.

```
● chicken  plan
    ○ add the contact form
    ○ review it
▸ coder  Add a contact form to index.html …
  ⇄ ejected qwen3:14b · loaded qwen2.5-coder:14b (10.4 GB) in 4.0s
│ • Write index.html   +15 -0
└ ✓ coder  21s · wrote index.html  +15 −0
  ⇄ ejected qwen2.5-coder:14b · loaded qwen3:14b (11.1 GB) in 4.2s
▸ reviewer  Check the contact form …
│ VERDICT: needs fixes …
└ ✓ reviewer  14s · VERDICT: needs fixes …
✓ Done  2 helpers · one at a time · 58s · …
```

| Helper | Model | Kind | What it does |
|---|---|---|---|
| explorer | boss | tool | Maps a folder or project, read only |
| researcher | boss | tool | Web search, short report with sources |
| reviewer | boss | tool | Checks work, lists concrete problems, read only |
| coder | qwen2.5-coder:14b | writer | Writes and changes code |
| vision | gemma3:12b | writer | Reads images and screenshots |
| summarizer | qwen2.5:7b | writer | Shrinks long documents (low stakes) |

- **Tool** helpers use tools themselves (read, search, web…). **Writer** helpers are for models that can't use tools
  reliably: the program gives them the files, applies their answer (with the usual diff and approval), and writes the
  result card with the real changed lines.
- **boss** as a model means the same model as the main agent, so there's no GPU swap, just a fresh memory.
- Models never manage anything. The program moves them in and out of the GPU, saves the boss's state and passes results along.

**GPU memory rule:** models stay in the GPU together while their total fits in `vram_budget_gb` (10 GB); otherwise the
ones the next step doesn't need are ejected, least recently used first. The boss (11.1 GB) and the coder (10.4 GB) are
each over 10 GB, so they always run alone; swapping takes about 4–11 s. Sizes are measured when a model loads and saved in
`~/.agent/vram.json`. Ollama is currently set to keep one model loaded at a time (`OLLAMA_MAX_LOADED_MODELS=1`), so small
models don't share the GPU until that is raised.

**Context follows memory** (`num_ctx: "auto"`): the context is whatever GPU memory the model's weights leave in the
budget. Chicken reads each model's size and context cost per token from its own metadata (Ollama's model info, or the
GGUF file for llama.cpp servers) and corrects it with measured sizes. Alone in 10 GB, bonsai2:27b gets about 57k tokens
(it is a hybrid model: only 1 layer in 4 keeps a context cache); qwen3:14b gets about 11k.

When a helper on another model is called, Chicken first tries to keep the main agent in the GPU beside it:

1. It shrinks the main agent's context to what's left next to the helper (never below `ctx_min`, 8k).
2. If the conversation doesn't fit in that smaller context, it compacts it first (`/compact auto off` turns this off).
3. If that still isn't enough (or Ollama can't keep both models), it ejects the main agent and injects it back afterwards.
4. When the helper is done, the helper is ejected and the main agent grows back to its full context right away.

Nothing is ever put on the CPU: if Ollama had to place part of a model there, Chicken ejects and swaps instead.

```
  ⇄ chicken context 56k → 25k to make room for vision
  ⇄ loaded glm-ocr:latest (1.9 GB, 4k context) in 5.1s · kept bonsai2:27b (fits in 10 GB)
  …
  ⇄ ejected glm-ocr:latest · loaded bonsai2:27b (9.9 GB, 56k context) in 3.0s
```

| Command | What it does |
|---|---|
| `call -agent <helper> <task>` | Run a helper yourself. The main agent sees the result with your next message |
| `/agents` | Table of helpers: change model and kind with ↑↓ ←→ Enter; the bottom bar sets the mode |
| `/agents mode ask\|auto\|manual` | ask (default): you approve each helper · auto: the boss calls them on its own · manual: only `call -agent` |
| `/agents vram` | What's in the GPU now, and the measured size of each model |
| `/agents resume` | Continue a job that was stopped (Esc, crash) while a helper was working |
| `/agents list` | List the helpers |

Saying "use the coder" or "have the reviewer check it" in a message makes the boss use that helper.
Each helper is a file in `~/.agent/agents/<name>.md` (model, kind, tools, prompt), so you can edit them or add your own.
Every job with helpers gets a folder in `~/.agent/runs/`: one `.md` card per step (task and input), one `.result.md`
per result, and the boss's checkpoint.

### Saving: log or wiki

| | Log (`/save name`) | Wiki (`/save -wiki name`) |
|---|---|---|
| What it keeps | The whole conversation, word for word | A summary of it, plus one page per file worked on |
| Where | `~/.agent/sessions/name.json.gz` | `~/.agent/wiki/name/` |
| Load | `/load name`: continue exactly where you were | `/load -wiki name`: fresh conversation that knows the project |
| Memory used on load | Large (the whole chat, minus outdated outputs) | Small: only `index.md` + `chat.md` (usually under 1k tokens) |
| Save time | Instant | About 20–30 s per page (the AI writes it) |

**The wiki** is a folder of linked Markdown pages:

```
~/.agent/wiki/photos-project/
  index.md                  list of every page, with a one-line description
  chat.md                   goal, preferences, decisions, what's done, what's next, key facts
  plan.md                   the checklist (if there is one)
  files/test/index.html.md  one page per file: purpose, structure with line numbers, key details, changes
```

When you load it, the AI reads only the index and the chat summary. It opens a file page with `read_file` only when
it needs it, and the real file only to edit it. Saving again to the same wiki **updates** it: `chat.md` merges old and new,
and pages of files that haven't changed are kept (`= kept`). If nothing new was said, `chat.md` isn't rewritten.
`/save -wiki` without a name updates the wiki you loaded.

**Progress** is shown live while saving: `Saving wiki 42% █████░░░░░░░ · files/test/index.html.md` on the status line,
plus one line per finished page (`[ 67%] ✓ files/test/index.html.md`). Esc stops it, and pages already written are kept.

**Compression:**
- Logs are stored with gzip, which is lossless and makes them about 3–5× smaller on disk. Older plain `.json` saves still load.
- On `/load`, outputs that became outdated are trimmed: a file read that was later re-read or changed, a listing,
  search or page fetched again later, long outputs of old actions (start and end kept), and the full text of files written
  long ago (they're on disk). Nothing the AI can't get back by repeating the action. On the saved sessions here this saved 35–50% of memory.
- The wiki is the strongest compression, but it is a summary written by the local model. Exact details (IDs, numbers)
  can occasionally be lost or merged. Keep a log too when every detail matters.

### Settings: `~/.agent/config.json`

| Key | Default | Meaning |
|---|---|---|
| `model` | `qwen3.5:4b` | The main model: the orchestrator (in classic mode, the agent that works) |
| `mode` | `ask` | `ask` or `auto` (`/mode`) |
| `routing` | `orchestrator` | `orchestrator`, `bonsai-only`, `bonsai-workers` or `classic` (`/mode`) |
| `heavy_model` | `bonsai2:27b` | Planner, hard work, the only model in bonsai-only |
| `orchestrator_ctx` | `32768` | Largest context for the orchestrator |
| `num_ctx` | `"auto"` | Context size: `auto` = all the GPU memory the model leaves in the budget; a number = fixed |
| `ctx_min` | `8192` | Smallest context the main agent shrinks to so a helper fits beside it; below this it ejects |
| `compact_for_helpers` | `true` | Compact the conversation when that lets a helper fit beside the main agent (`/compact auto on\|off`) |
| `temperature` | `0.6` | Creativity |
| `think` | `true` | Reason before answering |
| `verbose` | `false` | Details mode at startup |
| `keys` | ctrl+b/t/o/g/v | Keyboard shortcuts (use `/setkeyboard`) |
| `root` | your home folder | The agent can't go outside this folder |
| `protected` | `.ssh`, `.gnupg`, … | Paths it can never modify |
| `max_tool_rounds` | `30` | Maximum actions per normal request |
| `loop_max_steps` | `80` | Maximum steps per `/loop` |
| `command_timeout` | `300` | Seconds before a command is killed |
| `max_answer_tokens` | `8192` | Maximum length of one answer (reasoning included), so nothing can hold the GPU for long |
| `theme` | `chicken` | Colour theme (`/theme`). `chicken` is the promo's palette; `terminal` keeps your terminal's own colours |
| `prompt_color` | `""` | Colour of the `❯` arrow (`/color`). Empty = the theme's (coral in `chicken`) |
| `agents_mode` | `ask` | When helpers run: ask, auto or manual (`/agents`) |
| `vram_budget_gb` | `"90%"` | GPU limit: a share of the card or a number of GB. Above it Chicken pauses, compresses, ejects |
| `models` | `{}` | Your own models for the orchestrator: `"name": ["good at", "intelligence", "speed"]` |
| `btw_model` | `qwen3.5:0.8b` | Read-only model for `/btw`, on the GPU; `""` uses the main model |
| `wiki_max_files` | `12` | Maximum file pages written per `/save -wiki` (changed files first, then read ones) |

### Files

```
agent.py                the program (single Python file)
install.sh              creates .venv and the `chicken` / `call` commands
requirements.txt        prompt_toolkit + requests
~/.local/bin/chicken    launcher (`chicken`)
~/.local/bin/call       launcher (`call chicken`)
~/.agent/config.json    settings
~/.agent/tasks/         preset tasks (edit or add your own)
~/.agent/sessions/      saved conversation logs, gzipped (+ last.json.gz autosave)
~/.agent/wiki/          saved project wikis (one folder each)
~/.agent/agents/        helpers, one .md file each (model, kind, tools, prompt)
~/.agent/runs/          one folder per job: task state, step cards, results
~/.agent/logs/          decisions.jsonl: every orchestrator decision, for tuning the router
~/.agent/vram.json      measured GPU memory per model
~/.agent/trash/         files the agent deleted (recoverable)
```

### Good to know

- **Speed:** about 33 tokens/s on the RTX 3060. With reasoning on, a simple edit takes about 1 minute. `/think off` is much faster.
- **Memory:** about 24k tokens (roughly 70 pages). When it fills up, older messages are summarized automatically. For big folders, point it at specific files with `/file`.
- **Other models:** `/model` opens a picker. `qwen2.5-coder:14b` is good for pure coding. 27B models don't fit in the GPU and run slowly.
- **It can make mistakes.** Read the changes it proposes (option 3 lets you correct it), and keep backups of important files.
- Web search uses DuckDuckGo, with no account or API key needed.

## License

- Code: [Apache License 2.0](LICENSE). See [NOTICE](NOTICE) for the required credit.
- Design documents in `docs/`: [Creative Commons Attribution 4.0](docs/LICENSE).

You can use, change, share and sell Chicken, as long as you keep the credit to its author in the NOTICE file
(for the code) or credit the author with a link to this repository (for the design).
The name "Chicken" and its logo are not covered by these licenses.

Created by [alemarzosw](https://github.com/alemarzosw).
