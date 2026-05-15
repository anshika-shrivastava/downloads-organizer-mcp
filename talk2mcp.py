"""Orchestrator: LLM (OpenRouter) ↔ MCP server (mcp_server.py).

Flow per turn:
  1. Read the running state (initial user goal + tool history) and ask the
     LLM what to do next, passing the MCP tools as OpenAI function schemas.
  2. If the LLM emits tool_calls, dispatch each to the MCP server, append
     the result to the message history, and loop.
  3. If the LLM stops emitting tool_calls, log the final answer and exit.

Every step is logged through utils.logger so a viewer can replay the
decision-by-decision trace afterwards.

Usage:
  python talk2mcp.py
  python talk2mcp.py --dry-run
  python talk2mcp.py --dir /path/to/folder
  python talk2mcp.py --min-age 14
  python talk2mcp.py --model anthropic/claude-3.5-sonnet
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from utils.llm_client import LLMConfig, build_client, mcp_tools_to_openai
from utils.logger import get_logger


SYSTEM_PROMPT = """You are the Downloads Folder Organizer agent.

You have exactly three tools:
  - list_downloads()                    inventory of the Downloads folder
  - create_subfolder(name)              create a category folder
  - move_file(filename, subfolder)      move a file into a category folder

Your job in one run:
  1. Call list_downloads ONCE to see what is there.
  2. Decide a small set of clear category folders based on the files you see.
     Use Title-Case names (e.g. "Images", "PDFs", "Installers", "Code",
     "Archives", "Videos", "Audio", "Documents", "Other"). Reuse any
     `existing_subfolders` returned by list_downloads instead of duplicating.
  3. Call create_subfolder for each category you actually need. Skip
     categories with zero files.
  4. For every file where `movable` is true, call move_file with the right
     subfolder. Do NOT attempt to move files where `movable` is false — the
     server will refuse them anyway, but you should skip them to stay tidy.

Categorization rules of thumb (extension first, then look at the filename):
  - Images   : .png .jpg .jpeg .gif .webp .heic .svg .bmp .tiff
  - PDFs     : .pdf
  - Documents: .doc .docx .txt .rtf .md .pages .odt .xls .xlsx .csv .ppt .pptx .key
  - Code     : .py .js .ts .tsx .jsx .html .css .json .yaml .yml .sh .rb .go .rs .java .c .cpp .h
  - Installers: .dmg .pkg .exe .msi .deb .rpm .appimage
  - Archives : .zip .tar .gz .tgz .bz2 .7z .rar
  - Videos   : .mp4 .mov .mkv .webm .avi .m4v
  - Audio    : .mp3 .wav .flac .m4a .aac .ogg
  - Anything else → "Other"

Hard rules:
  - You may only call the three tools above.
  - You must not invent filenames; only act on files returned by list_downloads.
  - You must not retry a move that the server refused due to age — skip it.
  - When you are done, reply with a short plain-text summary: how many files
    moved, into which folders, and how many were skipped (and why). Do not
    emit any more tool calls in that final message.
"""

DEFAULT_USER_PROMPT = (
    "Please organize my Downloads folder now. List, categorize, create the "
    "subfolders you need, then move every eligible file. Finish with a short "
    "summary of what you did."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM-driven Downloads organizer")
    parser.add_argument("--dir", help="Override DOWNLOADS_DIR for this run")
    parser.add_argument("--min-age", type=int, help="Minimum age in days to move (default 7)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Server returns success but doesn't touch disk")
    parser.add_argument("--model", help="OpenRouter model id override")
    parser.add_argument("--prompt", help="Override the user prompt")
    parser.add_argument("--max-iterations", type=int,
                        help="Hard cap on LLM turns (default 25)")
    return parser.parse_args()


def build_server_env(args: argparse.Namespace) -> dict[str, str]:
    """Environment passed to the spawned MCP server subprocess."""
    env = os.environ.copy()
    if args.dir:
        env["DOWNLOADS_DIR"] = str(Path(args.dir).expanduser())
    if args.min_age is not None:
        env["MIN_AGE_DAYS"] = str(args.min_age)
    if args.dry_run:
        env["DRY_RUN"] = "1"
    # Keep PYTHONPATH so the subprocess can import from our project root.
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _stringify_tool_result(result) -> str:
    """Flatten an MCP CallToolResult into a single string the LLM can read."""
    if result is None:
        return ""
    chunks: list[str] = []
    for c in (result.content or []):
        text = getattr(c, "text", None)
        if text is not None:
            chunks.append(text)
    return "\n".join(chunks) if chunks else json.dumps(result.model_dump() if hasattr(result, "model_dump") else {}, default=str)


async def run_agent(args: argparse.Namespace) -> int:
    load_dotenv()
    log = get_logger()
    log.system("Downloads Organizer starting up")
    log.info(
        f"dir={args.dir or '~/Downloads'}  min_age={args.min_age or 7}  "
        f"dry_run={args.dry_run}  model={args.model or os.environ.get('OPENROUTER_MODEL', 'default')}"
    )

    try:
        llm_config = LLMConfig.from_env(model_override=args.model)
    except RuntimeError as e:
        log.error(str(e))
        return 2
    llm = build_client(llm_config)
    log.info(f"OpenRouter model: {llm_config.model}")

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(Path(__file__).parent / "mcp_server.py")],
        env=build_server_env(args),
    )

    max_iter = args.max_iterations or int(os.environ.get("MAX_ITERATIONS", "25"))
    user_prompt = args.prompt or DEFAULT_USER_PROMPT

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools_resp = await session.list_tools()
            tools = tools_resp.tools
            tool_names = ", ".join(t.name for t in tools)
            log.system(f"MCP tools available: {tool_names}")

            openai_tools = mcp_tools_to_openai(tools)

            messages: list[dict] = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
            log.user(user_prompt)

            # Gemini Flash sometimes narrates a plan ("I will now filter…") in a
            # text-only turn and forgets to emit the tool_call. Bail-on-first-
            # narrative would end the run too early. We nudge it up to N times
            # before treating no-tool-calls as a real final answer.
            successful_moves = 0
            nudges_used = 0
            MAX_NUDGES = 3

            for turn in range(1, max_iter + 1):
                log.info(f"— turn {turn}/{max_iter} —")
                response = llm.chat.completions.create(
                    model=llm_config.model,
                    messages=messages,
                    tools=openai_tools,
                    temperature=llm_config.temperature,
                )
                msg = response.choices[0].message

                if msg.content:
                    log.llm(msg.content)

                # Push the assistant's turn (with tool_calls) onto history.
                assistant_entry: dict = {"role": "assistant", "content": msg.content or ""}
                if msg.tool_calls:
                    assistant_entry["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in msg.tool_calls
                    ]
                messages.append(assistant_entry)

                if not msg.tool_calls:
                    # If real work has happened, this is the final summary turn.
                    if successful_moves > 0 or nudges_used >= MAX_NUDGES:
                        log.system("LLM produced no tool calls — treating as final answer.")
                        return 0
                    # Otherwise the model narrated without acting. Nudge it.
                    nudges_used += 1
                    nudge = (
                        "You produced text but no tool_call. You have not "
                        "completed the task yet. Per the procedure above, "
                        "immediately emit the NEXT tool call now "
                        "(create_subfolder or move_file) — do not reply with "
                        "narrative only."
                    )
                    log.system(
                        f"LLM narrated without tool calls; nudging "
                        f"({nudges_used}/{MAX_NUDGES})."
                    )
                    messages.append({"role": "user", "content": nudge})
                    continue

                for tc in msg.tool_calls:
                    name = tc.function.name
                    try:
                        arguments = json.loads(tc.function.arguments or "{}")
                    except json.JSONDecodeError as e:
                        log.error(f"Bad JSON args for {name}: {e}; raw={tc.function.arguments!r}")
                        arguments = {}

                    log.tool_call(name, arguments)
                    try:
                        result = await session.call_tool(name, arguments)
                        result_text = _stringify_tool_result(result)
                        is_error = bool(getattr(result, "isError", False))
                        log.tool_result(name, result_text, is_error=is_error)
                        if (
                            name == "move_file"
                            and not is_error
                            and '"ok": true' in result_text
                        ):
                            successful_moves += 1
                    except Exception as e:  # noqa: BLE001
                        result_text = json.dumps({"ok": False, "error": str(e)})
                        log.tool_result(name, result_text, is_error=True)

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "name": name,
                            "content": result_text,
                        }
                    )

            log.error(f"Hit max iterations ({max_iter}) without a final answer.")
            return 1

    return 0


def main() -> int:
    args = parse_args()
    try:
        return asyncio.run(run_agent(args))
    except KeyboardInterrupt:
        print("\n[talk2mcp] interrupted by user", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
