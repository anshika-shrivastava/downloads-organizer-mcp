# Downloads Folder Organizer (LLM + MCP)

An agent that tidies `~/Downloads` using an LLM as the brain and an MCP server as
the hands. The LLM lists files, decides categories, creates subfolders, and moves
files in — while a hard safety rule in the MCP server blocks any move of a file
younger than 7 days.

> Visual demo: open Finder on `~/Downloads`, sort by Name, and run the agent.
> Files will physically fly into freshly-created category folders.

## Architecture

```
talk2mcp.py  ── OpenRouter LLM ────┐
   │                                │ (OpenAI-style tool calls)
   │ spawns over stdio              │
   ▼                                ▼
mcp_server.py  ── exposes 3 tools ──┐
   │                                │
   ├─ list_downloads()              │
   ├─ create_subfolder(name)        │
   └─ move_file(filename, subfolder)│
                                    ▼
                               ~/Downloads/
```

Every LLM turn — system prompt, tool calls, tool results, final answer — is
logged to the console (color-coded) **and** to `logs/run-<timestamp>.log` plus a
machine-readable `logs/run-<timestamp>.jsonl`.

## File layout

```
EAG4/
├── .env.example          # Copy to .env and add your OpenRouter key
├── .gitignore            # Ignores .env, logs, __pycache__
├── README.md
├── requirements.txt
├── talk2mcp.py           # LLM client + MCP client (orchestrator)
├── mcp_server.py         # MCP server: 3 tools, enforces 7-day rule
├── utils/
│   ├── __init__.py
│   ├── logger.py         # Color console + file + JSONL logging
│   └── llm_client.py     # OpenRouter (via OpenAI SDK) wrapper
└── logs/                 # Per-run log files land here
```

## Setup

```bash
cd ~/EAG4
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env, paste your OpenRouter key
```

## Run

```bash
python talk2mcp.py
```

Optional flags:

```bash
python talk2mcp.py --dry-run                 # show plan, do not move anything
python talk2mcp.py --dir /path/to/folder     # organize a different directory
python talk2mcp.py --min-age 14              # only touch files older than 14 days
python talk2mcp.py --model anthropic/claude-3.5-sonnet
```

## Safety

- `move_file` in `mcp_server.py` re-checks file age on every call. If the file
  is younger than `MIN_AGE_DAYS` it raises and refuses — even if the LLM asks.
- The agent never deletes. There is no `delete_file` tool. Moves are reversible.
- Path traversal is blocked: `filename` and `subfolder` are basenames only and
  must resolve inside `DOWNLOADS_DIR`.
- `--dry-run` swaps `move_file` for a no-op that only logs intent.

## How the demo reads on YouTube

1. Show messy `~/Downloads` in Finder (sort by Date Added).
2. Split-screen the terminal running `python talk2mcp.py`.
3. The console shows the LLM thinking, then tool calls scroll past in yellow,
   tool results in green. Finder reflows as folders appear and files move.
4. Scroll back through `logs/run-*.log` to prove the LLM drove every move.
