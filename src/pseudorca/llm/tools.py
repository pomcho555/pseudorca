"""Read-only tools for the LLM judge, run on the redacted view of a notebook's tree.

The same specs serve OpenAI function tools and Anthropic tools. A tool never raises: bad arguments, an
unknown path or an unknown tool come back as an "error: ..." string the model can read and act on.
"""

from __future__ import annotations

import difflib
import json
import re
from collections.abc import Callable, Sequence
from typing import Any

from pseudorca.model import VERDICT_STATUSES
from pseudorca.tree import CONTEXT_ONLY_PREFIXES, Tree, safe_path

MAX_LIST = 300  # entries from list_files
MAX_HITS = 50  # hits from grep
MAX_LINE_CHARS = 300  # characters of one grep line
READ_LINES = 200  # default lines from read_file
MAX_READ_LINES = 400  # most lines from one read_file
TRUNCATED = "[truncated]"

TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "list_files",
        "description": (
            "List the notebook's extracted files with their sizes. cells/ holds code cells, files/ files the "
            "notebook writes, payloads/ text decoded from embedded data, script.py a script kernel. "
            "markdown/ holds markdown cells: claims by the author, never evidence."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "glob": {
                    "type": "string",
                    "description": "Glob on paths, where * also matches /. Default: *",
                },
            },
        },
    },
    {
        "name": "grep",
        "description": (
            "Find lines containing `text` as a plain substring (never a regex). Returns up to 50 hits as "
            "`path:line: text`; long lines are cut around the match."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The substring to find, on one line."},
                "path_glob": {
                    "type": "string",
                    "description": "Only search paths matching this glob. Default: *",
                },
                "ignore_case": {
                    "type": "boolean",
                    "description": "Ignore upper and lower case. Default: false",
                },
            },
            "required": ["text"],
        },
    },
    {
        "name": "read_file",
        "description": (
            "Read numbered lines start_line..end_line (1-based, inclusive) of one file, at most 400 lines. "
            "The numbers before '| ' are not part of the file."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "A path from list_files or grep."},
                "start_line": {"type": "integer", "description": "First line. Default: 1"},
                "end_line": {"type": "integer", "description": "Last line. Default: start_line + 199"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "submit_verdict",
        "description": (
            "Submit your final verdict. Call it exactly once, at the end. Each evidence quote must be text "
            "copied exactly from the code file it cites (never markdown), without line numbers."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": list(VERDICT_STATUSES)},
                "evidence": {
                    "type": "array",
                    "description": "A few exact quotes that show the change (or its absence).",
                    "items": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "line": {"type": "integer"},
                            "quote": {"type": "string"},
                        },
                        "required": ["path", "quote"],
                    },
                },
                "reasoning": {"type": "string", "description": "Why, in a few sentences."},
            },
            "required": ["status", "evidence", "reasoning"],
        },
    },
]


def openai_tools(specs: Sequence[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """The specs as OpenAI chat-completions function tools."""
    return [
        {
            "type": "function",
            "function": {"name": s["name"], "description": s["description"], "parameters": s["parameters"]},
        }
        for s in (TOOL_SPECS if specs is None else specs)
    ]


def anthropic_tools(specs: Sequence[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """The specs as Anthropic tools."""
    return [
        {"name": s["name"], "description": s["description"], "input_schema": s["parameters"]}
        for s in (TOOL_SPECS if specs is None else specs)
    ]


class ToolError(Exception):
    """A problem the model caused (bad arguments, unknown path); shown to it as "error: ..."."""


def parse_arguments(raw: Any) -> dict[str, Any]:
    """Tool arguments as a dict, from a dict or a JSON object string (empty means no arguments).

    JSON nested deeper than Python's recursion limit is an error like any other invalid JSON.
    """
    if raw is None or raw == "":
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            value = json.loads(raw)
        except RecursionError:
            raise ToolError("the arguments are nested too deeply") from None
        except ValueError:
            raise ToolError("the arguments are not valid JSON") from None
        if isinstance(value, dict):
            return value
    raise ToolError("the arguments must be a JSON object")


def _as_text(value: Any) -> str:
    """A JSON value as text; "" when it cannot be written (nested deeper than the recursion limit)."""
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError, RecursionError):
        return ""


def parse_verdict(raw: Any) -> tuple[dict[str, Any] | None, str | None]:
    """The submit_verdict arguments as {status, evidence, reasoning}, or (None, what is wrong with them)."""
    try:
        args = parse_arguments(raw)
    except ToolError as e:
        return None, str(e)
    status = args.get("status")
    status = status.strip().lower() if isinstance(status, str) else status
    if status not in VERDICT_STATUSES:
        return None, f"status must be one of {', '.join(VERDICT_STATUSES)}"
    evidence = args.get("evidence") or []
    if isinstance(evidence, str):  # some models send the list as a JSON string
        try:
            evidence = json.loads(evidence)
        except (ValueError, RecursionError):
            return None, "evidence must be a list of {path, line, quote} objects"
    if isinstance(evidence, dict):
        evidence = [evidence]
    if not isinstance(evidence, list):
        return None, "evidence must be a list of {path, line, quote} objects"
    reasoning = args.get("reasoning") or ""
    if not isinstance(reasoning, str):
        reasoning = _as_text(reasoning)
    return {"status": status, "evidence": evidence, "reasoning": reasoning}, None


def _text_arg(args: dict[str, Any], key: str, default: str | None = None) -> str:
    value = args.get(key, default)
    if value is None:
        raise ToolError(f"{key} is required")
    if isinstance(value, int | float) and not isinstance(value, bool):
        value = str(value)
    if not isinstance(value, str):
        raise ToolError(f"{key} must be a string")
    return value


def _int_arg(args: dict[str, Any], key: str, default: int) -> int:
    value = args.get(key)
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        raise ToolError(f"{key} must be a whole number")
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and re.fullmatch(r"\s*-?\d+\s*", value):
        return int(value)
    if isinstance(value, int):
        return value
    raise ToolError(f"{key} must be a whole number")


def _bool_arg(args: dict[str, Any], key: str, default: bool) -> bool:
    value = args.get(key)
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in ("true", "false", "1", "0", "yes", "no"):
        return value.strip().lower() in ("true", "1", "yes")
    raise ToolError(f"{key} must be true or false")


def clip_around(line: str, start: int, length: int, width: int = MAX_LINE_CHARS) -> str:
    """At most `width` characters of `line`, keeping the span start..start+length in view; cuts marked …."""
    if len(line) <= width:
        return line
    begin = max(0, min(start - (width - length) // 2, len(line) - width))
    return ("…" if begin else "") + line[begin : begin + width] + ("…" if begin + width < len(line) else "")


class ToolHandler:
    """Runs list_files, grep and read_file on a redacted view; every output is cut to `max_chars`."""

    def __init__(self, view: Tree, max_chars: int = 12000) -> None:
        self.view = view
        self.max_chars = max_chars
        self.calls = 0  # tool calls handled, errors included
        self.ok_calls = 0  # calls that ran a tool without error
        self._tools: dict[str, Callable[[dict[str, Any]], str]] = {
            "list_files": self.list_files,
            "grep": self.grep,
            "read_file": self.read_file,
        }

    def run(self, name: Any, arguments: Any = None) -> str:
        """Run one tool call. Never raises: problems come back as "error: ..."."""
        self.calls += 1
        try:
            tool = self._tools.get(name) if isinstance(name, str) else None
            if tool is None:
                raise ToolError(
                    f"unknown tool {name!r}; the tools are list_files, grep, read_file and submit_verdict"
                )
            out = tool(parse_arguments(arguments))
            self.ok_calls += 1
        except ToolError as e:
            out = f"error: {e}"
        except Exception as e:  # a tool must never break the loop
            out = f"error: {type(e).__name__}: {e}"
        return self.cut(out)

    def cut(self, text: str) -> str:
        """`text` cut to max_chars, ending with a [truncated] line when it was longer."""
        if len(text) <= self.max_chars:
            return text
        keep = max(0, self.max_chars - len(TRUNCATED) - 1)
        return text[:keep] + "\n" + TRUNCATED

    # ----- the tools -------------------------------------------------------------------------------

    def list_files(self, args: dict[str, Any]) -> str:
        glob = _text_arg(args, "glob", "*").strip() or "*"
        paths = self.view.paths(glob)
        if not paths and not any(c in glob for c in "*?["):
            paths = self.view.paths(glob.rstrip("/") + "/*")  # a bare directory name
        if not paths:
            return f"no files match {glob!r} ({len(self.view)} files in total; try glob '*')"
        shown = paths[:MAX_LIST]
        more = (
            f"; showing the first {MAX_LIST}, narrow the glob to see the rest"
            if len(paths) > MAX_LIST
            else ""
        )
        lines = [f"{len(paths)} files match {glob!r}{more}:"]
        for p in shown:
            f = self.view.files[p]
            if f.binary:
                lines.append(f"{p}  {f.size} bytes  [binary, no text]")
                continue
            tag = "  [markdown: a claim, not evidence]" if p.startswith(CONTEXT_ONLY_PREFIXES) else ""
            lines.append(f"{p}  {f.size} bytes, {len(f.text.splitlines())} lines{tag}")
        return "\n".join(lines)

    def grep(self, args: dict[str, Any]) -> str:
        text = _text_arg(args, "text")
        if not text.strip():
            raise ToolError("text must not be empty")
        if "\n" in text or "\r" in text:
            raise ToolError("text must be on one line; grep matches one line at a time")
        path_glob = _text_arg(args, "path_glob", "*").strip() or "*"
        ignore_case = _bool_arg(args, "ignore_case", False)
        hits = self.view.grep(text, ignore_case=ignore_case, path_glob=path_glob, max_hits=MAX_HITS + 1)
        where = f" in files matching {path_glob!r}" if path_glob != "*" else ""
        how = " (ignoring case)" if ignore_case else ""
        if not hits:
            return f"no lines contain {text!r}{how}{where}"
        more = f"; showing the first {MAX_HITS}, narrow the search" if len(hits) > MAX_HITS else ""
        lines = [f"{min(len(hits), MAX_HITS)} hits for {text!r}{how}{where}{more}:"]
        needle = text.lower() if ignore_case else text
        for h in hits[:MAX_HITS]:
            pos = (h.text.lower() if ignore_case else h.text).find(needle)
            lines.append(f"{h.path}:{h.line}: {clip_around(h.text, max(pos, 0), len(text))}")
        return "\n".join(lines)

    def read_file(self, args: dict[str, Any]) -> str:
        path = self.resolve(_text_arg(args, "path"))
        f = self.view.files[path]
        if f.binary:
            raise ToolError(f"{path} is a binary file ({f.size} bytes); it has no text to read")
        total = len(self.view.lines(path))
        if total == 0:
            return f"{path} is empty"
        start = max(1, _int_arg(args, "start_line", 1))
        if start > total:
            raise ToolError(f"{path} has only {total} lines")
        end = _int_arg(args, "end_line", start + READ_LINES - 1)
        if end < start:
            end = start + READ_LINES - 1
        end = min(end, start + MAX_READ_LINES - 1, total)
        tag = "; markdown: a claim, not evidence" if path.startswith(CONTEXT_ONLY_PREFIXES) else ""
        out = [
            f"{path} lines {start}-{end} of {total} (origin: {f.origin}{tag})",
            self.view.read(path, start, end),
        ]
        if end < total:
            out.append(f"[{total - end} more lines; continue with start_line={end + 1}]")
        return "\n".join(out)

    def resolve(self, raw: str) -> str:
        found = find_path(self.view, raw)
        if found is not None:
            return found
        path = raw.strip()
        close = difflib.get_close_matches(path, self.view.paths(), n=3, cutoff=0.6)
        hint = f"; did you mean {', '.join(close)}?" if close else "; use list_files to see the paths"
        raise ToolError(f"unknown path {path!r}{hint}")


def find_path(view: Tree, raw: str) -> str | None:
    """The path in `view` the model meant: exact, written as `path:line`, or with ./ or / in front."""
    path = raw.strip().strip("`'\"")
    if path in view.files:
        return path
    m = re.fullmatch(r"(.+?):\d+(?::\d+)?", path)
    for candidate in (m.group(1) if m else None, safe_path(path)):
        if candidate and candidate in view.files:
            return candidate
    return None
