"""IPython syntax in code cells (cell magics, line magics, `!` shell escapes) turned into plain Python.

IPython is never imported. The rules follow its input transformer closely enough that a cell IPython runs
still parses as Python once its special lines are commented out, and every line keeps its place, so line
numbers in the tree match the notebook.
"""

from __future__ import annotations

import posixpath
import re
import shlex
from dataclasses import dataclass

# IPython escapes at the start of a statement: !cmd, %magic, ?help, ,quote, ;quote and /call.
ESCAPES = frozenset("!%?,;/")
# Cell magics whose body is Python (run in the kernel or in a Python subprocess).
PYTHON_BODY = frozenset(
    {"time", "timeit", "capture", "prun", "debug", "memit", "python", "python2", "python3", "pypy"}
)
SHELLS = frozenset({"bash", "sh"})
WRITEFILE = frozenset({"writefile", "file"})  # %%file is IPython's alias of %%writefile

_LINE_RE = re.compile(r"[^\r\n]*(?:\r\n|\r|\n)|[^\r\n]+\Z")
_HEAD_RE = re.compile(r"%%(\S*)\s*(.*)", re.S)
_ENV_RE = re.compile(r"%(?:env|set_env)[ \t]+(.+)")
_NOT_PLAIN_ASSIGN = frozenset("=!<>:+-*/%&|^@")  # characters that make `=` part of another operator
_CODE_RE = re.compile(r"""[#'"()\[\]{}=\\]""")  # the characters that matter outside strings
_IN_SINGLE_RE = re.compile(r"[\\']")
_IN_DOUBLE_RE = re.compile(r'[\\"]')
_ESCAPE_AFTER_ASSIGN_RE = re.compile(r"[ \t]*(?:!(?!=)|%[^\W\d])")  # `= !cmd` or `= %magic`


def split_lines(text: str) -> list[str]:
    """Physical lines with their endings, split only at \\n, \\r\\n and \\r, as Python's tokenizer does."""
    return _LINE_RE.findall(text)


@dataclass(frozen=True)
class _State:
    quote: str = ""  # delimiter of a string still open at the end of the line
    depth: int = 0  # brackets still open
    cont: bool = False  # the line ended with a backslash continuation

    def at_statement_start(self) -> bool:
        return not (self.quote or self.depth or self.cont)


def _scan(line: str, state: _State) -> tuple[_State, bool, bool]:
    """Follow strings, brackets and comments through one physical line.

    Returns the state after the line, whether its first plain `=` outside brackets is followed by `!cmd`
    or `%magic` (IPython's `x = !ls`), and whether its code ends with `?` (IPython's `obj?`).
    """
    body = line.rstrip("\r\n")
    quote, depth, cont = state.quote, state.depth, False
    escape_assign = seen_assign = False
    n, i, code_end = len(body), 0, len(body)
    while i < n:
        if quote:  # jump to the next backslash or quote character inside the string
            m = (_IN_SINGLE_RE if quote[0] == "'" else _IN_DOUBLE_RE).search(body, i)
            if m is None:
                break
            j = m.start()
            if body[j] == "\\":
                cont = j + 1 == n
                i = j + 2
            elif body.startswith(quote, j):
                quote, i = "", j + len(quote)
            else:
                i = j + 1
            continue
        m = _CODE_RE.search(body, i)
        if m is None:
            break
        j = m.start()
        c = body[j]
        i = j + 1
        if c == "#":
            code_end = j
            break
        if c in "\"'":
            quote = c * 3 if body.startswith(c * 3, j) else c
            i = j + len(quote)
        elif c == "\\":
            if i == n:
                cont, code_end = True, j
        elif c in "([{":
            depth += 1
        elif c in ")]}":
            depth = max(0, depth - 1)
        elif (
            not depth
            and not seen_assign
            and body[i : i + 1] != "="
            and (j == 0 or body[j - 1] not in _NOT_PLAIN_ASSIGN)
        ):
            seen_assign = True
            escape_assign = _ESCAPE_AFTER_ASSIGN_RE.match(body, i) is not None
    if len(quote) == 1 and not cont:
        quote = ""  # a one-quote string cannot continue on the next line without a backslash
    help_end = not (quote or depth or cont) and body[:code_end].rstrip().endswith("?")
    return _State(quote, depth, cont), escape_assign, help_end


def _comment(line: str, *, statement: bool = False) -> str:
    """The line as a comment with the same indentation; `pass  # ...` for an indented `statement`, so a
    block whose only statement was a magic still parses."""
    body = line.rstrip("\r\n")
    eol = line[len(body) :]
    text = body.lstrip(" \t\f")
    indent = body[: len(body) - len(text)]
    if statement and indent:
        return f"{indent}pass  # {text}{eol}"
    return f"{indent}# {text}{eol}"


def _env_setting(line: str) -> tuple[str, str] | None:
    """(key, value) of `%env K=V`, `%env K V`, `%set_env K=V` or `%set_env K V` (IPython's parsing)."""
    m = _ENV_RE.match(line.strip())
    if not m:
        return None
    params = m.group(1).strip()
    key, sep, value = params.partition("=" if "=" in params else " ")
    key = key.strip()
    if not sep or not key or any(c.isspace() for c in key):
        return None
    return key, value.strip()


def transform_python(source: str) -> tuple[str, list[tuple[int, str, str]]]:
    """Comment out IPython-only lines so a code cell parses as Python, keeping every line in place.

    A line magic, `!cmd`, `x = !cmd`, `x = %magic` or `obj?` at the start of a statement becomes
    `# <line>` with its indentation kept, or `pass  # <line>` when indented (its backslash-continued
    lines become comments too). The same text inside strings or brackets is left alone. Returns the new
    text and the (line, key, value) of each `%env`/`%set_env` setting.
    """
    lines = split_lines(source)
    out: list[str] = []
    env: list[tuple[int, str, str]] = []
    state = _State()
    i = 0
    while i < len(lines):
        line = lines[i]
        after, escape_assign, help_end = _scan(line, state)
        text = line.lstrip(" \t\f")
        if state.at_statement_start() and text.strip() and (text[0] in ESCAPES or escape_assign or help_end):
            end = i
            while end + 1 < len(lines) and lines[end].rstrip("\r\n").endswith("\\"):
                end += 1
            out.extend(_comment(lines[j], statement=j == i) for j in range(i, end + 1))
            setting = _env_setting(text)
            if setting:
                env.append((i + 1, *setting))
            i = end + 1
            continue  # the state is unchanged: the special line is not Python
        out.append(line)
        state = after
        i += 1
    return "".join(out), env


@dataclass(frozen=True)
class CellMagic:
    name: str  # e.g. "writefile" for %%writefile
    args: str  # the rest of the magic line
    head: str  # the magic line, stripped
    body: str  # what IPython passes to the magic: the cell after the magic line, ending with a newline
    lead: str  # the cell up to the body: blank lines as they are, the magic line as `# %%bash`
    line: int  # the magic line's line in the cell; the body starts on the next one


def cell_magic(source: str) -> CellMagic | None:
    """The cell magic of a cell whose first non-blank line starts with %%, else None.

    `lead + body` has the cell's lines, so line N of a file built from it is line N of the cell.
    """
    lines = split_lines(source)
    k = next((k for k, line in enumerate(lines) if line.strip()), None)
    if k is None:
        return None
    line = lines[k]
    head = line.strip()
    m = _HEAD_RE.match(head)
    if m is None:
        return None
    indent = line[: len(line) - len(line.lstrip(" \t"))]
    rest = lines[k + 1 :]
    if indent:  # IPython removes the cell's common indentation before it looks for %%
        rest = [x[len(indent) :] if x.startswith(indent) else x for x in rest]
    body = "".join(rest)
    if body and not body.endswith("\n"):
        body += "\n"  # IPython ends a cell with a newline before the magic gets its body
    lead = "".join(lines[:k]) + f"# {head}" + line[len(line.rstrip("\r\n")) :]
    return CellMagic(m.group(1), m.group(2).strip(), head, body, lead, k + 1)


def split_args(args: str) -> list[str]:
    """Shell-like words of a magic line (quotes removed; backslashes and `#` kept as they are)."""
    lex = shlex.shlex(args, posix=True)
    lex.whitespace_split = True
    lex.commenters = ""
    lex.escape = ""
    try:
        return list(lex)
    except ValueError:  # unbalanced quotes
        return args.split()


def writefile_target(args: str) -> tuple[str, bool] | None:
    """(path, append) of `%%writefile [-a|--append] PATH`, or None without a path."""
    path, append = None, False
    for word in split_args(args):
        if word in ("-a", "--append"):
            append = True
        elif path is None and not word.startswith("-"):
            path = word
    return (path, append) if path else None


def script_program(args: str) -> str:
    """The interpreter a `%%script` cell runs (`/usr/bin/env bash` gives `bash`), or ""."""
    skip = False
    for word in split_args(args):
        if skip:
            skip = False
        elif word in ("--out", "--err", "--proc"):
            skip = True
        elif not word.startswith("-") and "=" not in word:  # options, and env's VAR=value
            name = posixpath.basename(word)
            if name != "env":
                return name
    return ""


def classify(magic: CellMagic) -> str:
    """How to treat a cell magic's body: "python", "writefile", "shell" or "other"."""
    if magic.name in PYTHON_BODY:
        return "python"
    if magic.name in WRITEFILE:
        return "writefile"
    if magic.name in SHELLS:
        return "shell"
    if magic.name == "script":
        program = script_program(magic.args)
        if program in SHELLS:
            return "shell"
        if program.startswith(("python", "pypy")):
            return "python"
    return "other"
