"""Logging setup for the Downloads Organizer agent.

Goals:
- Bright, color-coded console output so an LLM-driven run reads well on video.
- Plain text file log under logs/ for replay.
- Parallel JSONL log under logs/ for programmatic inspection.

Everything that matters during a run flows through `get_logger()`:
- LLM system / user / assistant messages
- Each tool call (name + arguments)
- Each tool result
- Errors

The JSONL stream uses one event per line with a stable schema so it can be
diffed or piped into another tool later.
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# ANSI color codes. Falls back to no-color if stdout isn't a TTY.
_COLORS = {
    "reset": "\033[0m",
    "dim": "\033[2m",
    "bold": "\033[1m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "gray": "\033[90m",
}

LEVEL_COLORS = {
    "DEBUG": _COLORS["gray"],
    "INFO": _COLORS["reset"],
    "WARNING": _COLORS["yellow"],
    "ERROR": _COLORS["red"],
    "CRITICAL": _COLORS["bold"] + _COLORS["red"],
}

# Semantic tags used by talk2mcp.py — each renders in a consistent color so
# scrolling through the log on video is easy to follow.
TAG_COLORS = {
    "SYSTEM": _COLORS["magenta"],
    "USER": _COLORS["blue"],
    "LLM": _COLORS["cyan"],
    "TOOL_CALL": _COLORS["yellow"],
    "TOOL_RESULT": _COLORS["green"],
    "ERROR": _COLORS["red"],
    "INFO": _COLORS["reset"],
}


def _supports_color() -> bool:
    return sys.stdout.isatty()


class _ColorFormatter(logging.Formatter):
    def __init__(self, use_color: bool):
        super().__init__("%(message)s")
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        tag = getattr(record, "tag", None)
        ts = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        prefix = f"[{ts}]"
        if tag:
            prefix += f" {tag:<11}"
        line = f"{prefix} {record.getMessage()}"
        if self.use_color:
            color = TAG_COLORS.get(tag, LEVEL_COLORS.get(record.levelname, ""))
            return f"{color}{line}{_COLORS['reset']}"
        return line


class _JsonlHandler(logging.Handler):
    """Writes one JSON object per line to a .jsonl file."""

    def __init__(self, path: Path):
        super().__init__()
        self._path = path
        self._fp = path.open("a", encoding="utf-8")

    def emit(self, record: logging.LogRecord) -> None:
        try:
            event = {
                "ts": datetime.fromtimestamp(record.created).isoformat(),
                "level": record.levelname,
                "tag": getattr(record, "tag", None),
                "message": record.getMessage(),
            }
            extra = getattr(record, "structured", None)
            if extra is not None:
                event["data"] = extra
            self._fp.write(json.dumps(event, default=str) + "\n")
            self._fp.flush()
        except Exception:  # noqa: BLE001
            self.handleError(record)

    def close(self) -> None:
        try:
            self._fp.close()
        finally:
            super().close()


class AgentLogger:
    """Thin wrapper over stdlib logging with tagged convenience methods."""

    def __init__(self, logger: logging.Logger, run_id: str, log_dir: Path):
        self._logger = logger
        self.run_id = run_id
        self.log_dir = log_dir

    def _log(self, tag: str, message: str, *, structured: Any = None,
             level: int = logging.INFO) -> None:
        extra = {"tag": tag}
        if structured is not None:
            extra["structured"] = structured
        self._logger.log(level, message, extra=extra)

    # Semantic helpers — each maps to a colored tag in the console.
    def system(self, message: str) -> None:
        self._log("SYSTEM", message)

    def user(self, message: str) -> None:
        self._log("USER", message)

    def llm(self, message: str, *, structured: Any = None) -> None:
        self._log("LLM", message, structured=structured)

    def tool_call(self, name: str, arguments: dict[str, Any]) -> None:
        pretty = json.dumps(arguments, ensure_ascii=False)
        self._log(
            "TOOL_CALL",
            f"→ {name}({pretty})",
            structured={"tool": name, "arguments": arguments},
        )

    def tool_result(self, name: str, result: str, *, is_error: bool = False) -> None:
        snippet = result if len(result) < 600 else result[:600] + " …(truncated)"
        tag = "ERROR" if is_error else "TOOL_RESULT"
        self._log(
            tag,
            f"← {name} {'FAILED' if is_error else 'ok'}: {snippet}",
            structured={"tool": name, "is_error": is_error, "result": result},
        )

    def info(self, message: str) -> None:
        self._log("INFO", message)

    def error(self, message: str) -> None:
        self._log("ERROR", message, level=logging.ERROR)


def get_logger(log_dir: Path | str = "logs") -> AgentLogger:
    """Create a logger that writes to console (color) + .log file + .jsonl file."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    text_path = log_dir / f"run-{run_id}.log"
    jsonl_path = log_dir / f"run-{run_id}.jsonl"

    logger = logging.getLogger(f"downloads-organizer.{run_id}")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    # Defensive: if someone re-imports, don't pile up handlers.
    if logger.handlers:
        for h in list(logger.handlers):
            logger.removeHandler(h)

    # Console
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(_ColorFormatter(use_color=_supports_color()))
    logger.addHandler(console)

    # Plain text file (no color)
    text_handler = logging.FileHandler(text_path, encoding="utf-8")
    text_handler.setLevel(logging.DEBUG)
    text_handler.setFormatter(_ColorFormatter(use_color=False))
    logger.addHandler(text_handler)

    # JSONL file
    jsonl_handler = _JsonlHandler(jsonl_path)
    jsonl_handler.setLevel(logging.DEBUG)
    logger.addHandler(jsonl_handler)

    wrapper = AgentLogger(logger, run_id=run_id, log_dir=log_dir)
    wrapper.info(f"run log → {text_path}")
    wrapper.info(f"run jsonl → {jsonl_path}")
    return wrapper
