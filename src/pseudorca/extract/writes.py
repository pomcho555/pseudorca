"""Python source read with `ast` only (never compiled or run): literal file writes and environment settings.

A write counts only when both its path and its content are literals: string constants, `+` of literals,
f-strings without placeholders, `textwrap.dedent(...)` and `.strip()` of literals. Anything that depends
on a variable is skipped.
"""

from __future__ import annotations

import ast
import posixpath
import textwrap
import warnings
from dataclasses import dataclass

_PATH_TYPES = frozenset({"Path", "PosixPath", "WindowsPath"})
_PARSE_ERRORS = (SyntaxError, ValueError, RecursionError, MemoryError, OverflowError)
_MAX_FOLD_DEPTH = 40  # literal expressions nested deeper than this are not folded


def parse_python(text: str) -> tuple[ast.Module | None, str]:
    """(module, "") when `text` parses as Python, else (None, a short reason).

    A cell indented as a whole (which IPython dedents before running) is retried dedented; line numbers
    stay the same. Compiler warnings (invalid escapes and the like) are silenced.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return ast.parse(text), ""
        except _PARSE_ERRORS as e:
            error: Exception = e
        if isinstance(error, IndentationError):
            dedented = textwrap.dedent(text)
            if dedented != text:
                try:
                    return ast.parse(dedented), ""
                except _PARSE_ERRORS:
                    pass
    if isinstance(error, SyntaxError):
        where = f" at line {error.lineno}" if error.lineno else ""
        return None, f"{error.msg}{where}"
    return None, type(error).__name__


def _is_name(node: ast.AST, *names: str) -> bool:
    return isinstance(node, ast.Name) and node.id in names


def _is_attr(node: ast.AST, owner: str, attr: str) -> bool:
    return isinstance(node, ast.Attribute) and node.attr == attr and _is_name(node.value, owner)


def _keyword(call: ast.Call, name: str) -> ast.expr | None:
    return next((k.value for k in call.keywords if k.arg == name), None)


def flatten_add(node: ast.expr) -> list[ast.expr]:
    """The operands of a chain of `+`, left to right (iterative: chains can be long)."""
    out: list[ast.expr] = []
    stack = [node]
    while stack:
        n = stack.pop()
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Add):
            stack.extend((n.right, n.left))
        else:
            out.append(n)
    return out


def literal_str(node: ast.expr | None, _depth: int = 0) -> str | None:
    """The value of a string expression built only from literals, else None."""
    if node is None or _depth > _MAX_FOLD_DEPTH:
        return None
    if isinstance(node, ast.Constant):
        return node.value if isinstance(node.value, str) else None
    if isinstance(node, ast.JoinedStr):  # an f-string counts only without placeholders
        values = [v.value for v in node.values if isinstance(v, ast.Constant) and isinstance(v.value, str)]
        return "".join(values) if len(values) == len(node.values) else None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        parts = _literal_parts(flatten_add(node), _depth + 1)
        return None if parts is None else "".join(parts)
    if isinstance(node, ast.Call) and not node.keywords:
        func = node.func
        if (_is_name(func, "dedent") or _is_attr(func, "textwrap", "dedent")) and len(node.args) == 1:
            s = literal_str(node.args[0], _depth + 1)
            return None if s is None else textwrap.dedent(s)
        if (
            isinstance(func, ast.Attribute)
            and func.attr in ("strip", "lstrip", "rstrip")
            and len(node.args) <= 1
        ):
            s = literal_str(func.value, _depth + 1)
            chars = literal_str(node.args[0], _depth + 1) if node.args else None
            if s is None or (node.args and chars is None):
                return None
            return getattr(s, func.attr)(chars)
    return None


def _literal_parts(nodes: list[ast.expr], depth: int = 0) -> list[str] | None:
    """The values of literal string expressions, or None if one is not literal."""
    parts: list[str] = []
    for n in nodes:
        s = literal_str(n, depth)
        if s is None:
            return None
        parts.append(s)
    return parts


def literal_bytes(node: ast.expr | None) -> bytes | None:
    """The value of a bytes constant, or of `+` of bytes constants, else None."""
    if node is None:
        return None
    parts: list[bytes] = []
    for n in flatten_add(node):
        if not (isinstance(n, ast.Constant) and isinstance(n.value, bytes)):
            return None
        parts.append(n.value)
    return b"".join(parts)


def _path_object(node: ast.expr) -> str | None:
    """A literal pathlib path: `Path("a")`, `Path("a", "b")`, `pathlib.Path("a") / "b"`."""
    tail: list[str] = []
    while isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        part = literal_str(node.right)
        if part is None:
            return None
        tail.append(part)
        node = node.left
    if not isinstance(node, ast.Call) or node.keywords or not node.args:
        return None
    func = node.func
    if not (
        (isinstance(func, ast.Name) and func.id in _PATH_TYPES)
        or (isinstance(func, ast.Attribute) and func.attr in _PATH_TYPES and _is_name(func.value, "pathlib"))
    ):
        return None
    head = _literal_parts(node.args)
    return None if head is None else posixpath.join(*head, *reversed(tail))


def _literal_path(node: ast.expr | None) -> str | None:
    """A literal path given as a string, a pathlib path or `os.path.join` of literals."""
    if node is None:
        return None
    s = literal_str(node)
    if s is not None:
        return s
    p = _path_object(node)
    if p is not None:
        return p
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "join"
        and _is_attr(node.func.value, "os", "path")
        and node.args
        and not node.keywords
    ):
        parts = _literal_parts(node.args)
        if parts is not None:
            return posixpath.join(*parts)
    return None


def _open_target(node: ast.expr) -> tuple[str, bool, bool] | None:
    """(path, append, binary) for `open(<literal>, <write mode>)` or `Path(<literal>).open(<write mode>)`."""
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    if _is_name(func, "open") or _is_attr(func, "io", "open"):
        path = _literal_path(node.args[0] if node.args else _keyword(node, "file"))
        mode_node = node.args[1] if len(node.args) > 1 else _keyword(node, "mode")
    elif isinstance(func, ast.Attribute) and func.attr == "open":
        path = _path_object(func.value)
        mode_node = node.args[0] if node.args else _keyword(node, "mode")
    else:
        return None
    mode = "r" if mode_node is None else literal_str(mode_node)
    if path is None or mode is None:
        return None
    if "w" in mode or "x" in mode:
        return path, False, "b" in mode
    if "a" in mode:
        return path, True, "b" in mode
    return None


@dataclass(frozen=True)
class Write:
    line: int
    col: int
    path: str
    content: str | bytes
    append: bool
    via: str  # "write_text", "write_bytes" or "open().write"


def _content(node: ast.expr | None, binary: bool) -> str | bytes | None:
    return literal_bytes(node) if binary else literal_str(node)


def _concat(pieces: list[str | bytes | None], binary: bool) -> str | bytes | None:
    """Pieces joined, or None when one is missing or of the wrong type."""
    if binary:
        data = [p for p in pieces if isinstance(p, bytes)]
        return b"".join(data) if len(data) == len(pieces) else None
    text = [p for p in pieces if isinstance(p, str)]
    return "".join(text) if len(text) == len(pieces) else None


def _with_writes(node: ast.With) -> list[Write]:
    """`with open(p, "w") as f:` whose body only writes literals to `f` (several writes concatenate)."""
    out: list[Write] = []
    for item in node.items:
        target = _open_target(item.context_expr)
        if target is None or not isinstance(item.optional_vars, ast.Name):
            continue
        path, append, binary = target
        name = item.optional_vars.id
        parts: list[str | bytes | None] = []
        for stmt in node.body:
            call = stmt.value if isinstance(stmt, ast.Expr) else None
            if not (isinstance(call, ast.Call) and len(call.args) == 1 and not call.keywords):
                continue
            if _is_attr(call.func, name, "write"):
                value = _content(call.args[0], binary)
            elif _is_attr(call.func, name, "writelines") and isinstance(call.args[0], (ast.List, ast.Tuple)):
                value = _concat([_content(e, binary) for e in call.args[0].elts], binary)
            else:
                continue
            if value is None:
                parts = []
                break
            parts.append(value)
        # every use of the file object must be one of those literal writes
        uses = sum(isinstance(n, ast.Name) and n.id == name for stmt in node.body for n in ast.walk(stmt))
        content = _concat(parts, binary) if parts and uses == len(parts) else None
        if content is not None:
            out.append(Write(node.lineno, node.col_offset, path, content, append, "open().write"))
    return out


def find_writes(module: ast.AST) -> list[Write]:
    """Files written with a literal path and literal content, in source order."""
    found: list[Write] = []
    for node in ast.walk(module):
        if isinstance(node, ast.With):
            found.extend(_with_writes(node))
            continue
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        attr, owner = node.func.attr, node.func.value
        if attr in ("write_text", "write_bytes"):
            path = _path_object(owner)
            content = _content(node.args[0] if node.args else _keyword(node, "data"), attr == "write_bytes")
            if path is not None and content is not None:
                found.append(Write(node.lineno, node.col_offset, path, content, False, attr))
        elif attr == "write" and len(node.args) == 1:
            target = _open_target(owner)
            if target is not None:
                path, append, binary = target
                content = _content(node.args[0], binary)
                if content is not None:
                    found.append(Write(node.lineno, node.col_offset, path, content, append, "open().write"))
    found.sort(key=lambda w: (w.line, w.col))
    return found


def _is_environ(node: ast.AST) -> bool:
    return isinstance(node, ast.Attribute) and node.attr == "environ" and isinstance(node.value, ast.Name)


def find_env(module: ast.AST) -> list[tuple[int, int, str, str]]:
    """(line, col, key, value) of literal `os.environ[K] = V`, `.setdefault(K, V)` and `.update({K: V})`."""
    out: list[tuple[int, int, str, str]] = []

    def add(node: ast.AST, key: ast.expr | None, value: ast.expr | None) -> None:
        k, v = literal_str(key), literal_str(value)
        if k is not None and v is not None:
            out.append((getattr(node, "lineno", 0), getattr(node, "col_offset", 0), k, v))

    for node in ast.walk(module):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Subscript) and _is_environ(target.value):
                    add(node, target.slice, node.value)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and _is_environ(node.func.value)
        ):
            if node.func.attr == "setdefault" and len(node.args) == 2:
                add(node, node.args[0], node.args[1])
            elif node.func.attr == "update":
                if len(node.args) == 1 and isinstance(node.args[0], ast.Dict):
                    for key, value in zip(node.args[0].keys, node.args[0].values, strict=True):
                        add(node, key, value)
                for kw in node.keywords:
                    if kw.arg:
                        add(node, ast.Constant(kw.arg), kw.value)
    out.sort(key=lambda e: (e[0], e[1]))
    return out
