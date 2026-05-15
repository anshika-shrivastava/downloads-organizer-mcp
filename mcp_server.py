"""MCP server: 3 tools for organizing a Downloads folder.

Exposed tools
-------------
- list_downloads()                       : list files in DOWNLOADS_DIR
- create_subfolder(name)                 : mkdir DOWNLOADS_DIR/name
- move_file(filename, subfolder)         : DOWNLOADS_DIR/filename -> DOWNLOADS_DIR/subfolder/filename

Safety rules baked into the server (not just the prompt):
  1. `move_file` refuses any file younger than MIN_AGE_DAYS (default 7).
  2. `filename` and `subfolder` must be basenames — no slashes, no `..`.
     Resolved paths must stay inside DOWNLOADS_DIR.
  3. No `delete` tool exists. Moves are reversible by hand.
  4. If --dry-run is set on launch, `move_file` logs intent only and does not
     touch the filesystem.

Environment knobs
-----------------
- DOWNLOADS_DIR   override target directory (defaults to ~/Downloads)
- MIN_AGE_DAYS    safety threshold in days (defaults to 7)
- DRY_RUN         "1" to make move_file a no-op

Run standalone (stdio transport):
    python mcp_server.py
"""
from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP


def _downloads_dir() -> Path:
    override = os.environ.get("DOWNLOADS_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return (Path.home() / "Downloads").resolve()


def _min_age_days() -> int:
    raw = os.environ.get("MIN_AGE_DAYS", "7")
    try:
        return max(0, int(raw))
    except ValueError:
        return 7


def _dry_run() -> bool:
    return os.environ.get("DRY_RUN", "").strip() in {"1", "true", "TRUE", "yes"}


def _age_days(p: Path) -> float:
    return (time.time() - p.stat().st_mtime) / 86400.0


def _human_size(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    f = float(n)
    for u in units:
        if f < 1024 or u == units[-1]:
            return f"{f:.1f}{u}" if u != "B" else f"{int(f)}{u}"
        f /= 1024
    return f"{n}B"


def _safe_basename(name: str, *, label: str) -> str:
    """Reject anything that isn't a plain file/folder name."""
    if not name or name in {".", ".."}:
        raise ValueError(f"{label} '{name}' is not a valid name")
    if "/" in name or "\\" in name or "\x00" in name:
        raise ValueError(f"{label} '{name}' must not contain path separators")
    if os.path.basename(name) != name:
        raise ValueError(f"{label} '{name}' must be a bare name, not a path")
    return name


def _inside(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


mcp = FastMCP("downloads-organizer")


@mcp.tool()
def list_downloads() -> dict[str, Any]:
    """List visible (non-hidden) files directly inside the Downloads directory.

    Returns one entry per regular file with: name, extension (lowercased, with
    leading dot), size_bytes, size_human, age_days (rounded to 1 decimal), and
    `movable` (True iff age_days >= MIN_AGE_DAYS).

    Subdirectories are reported separately under `existing_subfolders` so the
    LLM can reuse them instead of creating duplicates. Hidden entries
    (starting with `.`) are skipped entirely.
    """
    root = _downloads_dir()
    if not root.exists():
        return {"error": f"Downloads directory does not exist: {root}"}
    if not root.is_dir():
        return {"error": f"Not a directory: {root}"}

    min_age = _min_age_days()
    files: list[dict[str, Any]] = []
    subfolders: list[str] = []

    for entry in sorted(root.iterdir(), key=lambda p: p.name.lower()):
        if entry.name.startswith("."):
            continue
        if entry.is_dir():
            subfolders.append(entry.name)
            continue
        if not entry.is_file():
            continue
        try:
            stat = entry.stat()
        except FileNotFoundError:
            continue
        age = _age_days(entry)
        files.append(
            {
                "name": entry.name,
                "extension": entry.suffix.lower(),
                "size_bytes": stat.st_size,
                "size_human": _human_size(stat.st_size),
                "age_days": round(age, 1),
                "movable": age >= min_age,
            }
        )

    return {
        "downloads_dir": str(root),
        "min_age_days": min_age,
        "dry_run": _dry_run(),
        "file_count": len(files),
        "files": files,
        "existing_subfolders": subfolders,
    }


@mcp.tool()
def create_subfolder(name: str) -> dict[str, Any]:
    """Create a subfolder directly under the Downloads directory.

    `name` must be a bare folder name (no slashes, no `..`). Idempotent — if
    the folder already exists, returns `created=False` without error.
    """
    root = _downloads_dir()
    try:
        safe = _safe_basename(name, label="subfolder name")
    except ValueError as e:
        return {"ok": False, "error": str(e)}

    target = (root / safe).resolve()
    if not _inside(target, root):
        return {"ok": False, "error": f"resolved path escapes downloads dir: {target}"}

    if target.exists():
        if target.is_dir():
            return {"ok": True, "created": False, "path": str(target), "note": "already exists"}
        return {"ok": False, "error": f"path exists and is not a directory: {target}"}

    target.mkdir(parents=False, exist_ok=False)
    return {"ok": True, "created": True, "path": str(target)}


@mcp.tool()
def move_file(filename: str, subfolder: str) -> dict[str, Any]:
    """Move DOWNLOADS_DIR/{filename} into DOWNLOADS_DIR/{subfolder}/.

    Safety rules enforced here (the LLM cannot override them):
      - `filename` and `subfolder` must be bare names.
      - File must exist and be a regular file.
      - File must be at least MIN_AGE_DAYS old (default 7).
      - Destination subfolder must already exist (call create_subfolder first).
      - Refuses to overwrite: errors if a file with that name already exists
        in the destination.
      - If DRY_RUN=1, returns what would happen without touching disk.
    """
    root = _downloads_dir()
    min_age = _min_age_days()

    try:
        safe_file = _safe_basename(filename, label="filename")
        safe_dir = _safe_basename(subfolder, label="subfolder")
    except ValueError as e:
        return {"ok": False, "error": str(e)}

    src = (root / safe_file).resolve()
    dest_dir = (root / safe_dir).resolve()
    if not _inside(src, root) or not _inside(dest_dir, root):
        return {"ok": False, "error": "resolved path escapes downloads dir"}

    if not src.exists() or not src.is_file():
        return {"ok": False, "error": f"source file not found: {safe_file}"}
    if not dest_dir.exists() or not dest_dir.is_dir():
        return {
            "ok": False,
            "error": f"destination subfolder '{safe_dir}' does not exist — "
                     f"call create_subfolder first",
        }

    age = _age_days(src)
    if age < min_age:
        return {
            "ok": False,
            "error": (
                f"refusing to move '{safe_file}': only {age:.1f} days old, "
                f"minimum is {min_age} days"
            ),
            "age_days": round(age, 1),
            "min_age_days": min_age,
        }

    dest = dest_dir / safe_file
    if dest.exists():
        return {"ok": False, "error": f"destination already exists: {dest}"}

    if _dry_run():
        return {
            "ok": True,
            "dry_run": True,
            "from": str(src),
            "to": str(dest),
            "note": "DRY_RUN=1, no filesystem change",
        }

    shutil.move(str(src), str(dest))
    return {"ok": True, "dry_run": False, "from": str(src), "to": str(dest)}


if __name__ == "__main__":
    # stdio transport — the talk2mcp.py client spawns us as a subprocess.
    print(
        f"[mcp_server] downloads_dir={_downloads_dir()} "
        f"min_age_days={_min_age_days()} dry_run={_dry_run()}",
        file=sys.stderr,
        flush=True,
    )
    mcp.run(transport="stdio")
