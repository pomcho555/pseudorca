"""Secret redaction for everything an LLM sees: prompts, tool outputs and transcripts.

Known key formats are replaced by [REDACTED]: OpenAI (sk-, sk-proj-), Anthropic (sk-ant-), GitHub (ghp_,
gho_, ghu_, ghs_, ghr_, github_pat_), Hugging Face (hf_), AWS access key ids (AKIA, ASIA), Google (AIza),
Slack (xox?-), JSON web tokens, bearer tokens, private key blocks, passwords in URLs, kaggle.json keys, and
generic assignments such as `api_key = "..."` or `%env MY_TOKEN=...` (the name stays, the value goes).

Replacements keep every line break, so a redacted file has the line numbers of the original, and quotes
verified against the redacted view point at the right lines.
"""

from __future__ import annotations

import re
import weakref
from collections.abc import Callable
from typing import Any

from pseudorca import util
from pseudorca.tree import Tree, TreeFile

REDACTED = "[REDACTED]"

Replace = Callable[[re.Match[str]], str]


def _whole(m: re.Match[str]) -> str:
    """Replace the whole match, keeping its line breaks."""
    return REDACTED + "\n" * m.group(0).count("\n")


def _value(m: re.Match[str]) -> str:
    """Replace only the `value` group, keeping the rest of the match (such as the variable name)."""
    text, base = m.group(0), m.start()
    start, end = m.start("value") - base, m.end("value") - base
    return text[:start] + REDACTED + "\n" * text[start:end].count("\n") + text[end:]


# a name that holds a secret: api_key, HF_TOKEN, client_secret, db_password (but not tokenizer, ...)
_SECRET_NAME = re.compile(r"key|secret|passw(?:or)?d|token(?!izer)", re.I)
# names that hold where a secret is, not the secret: KEY_PATH, token_file, api_key_env, SECRET_NAME
_NOT_SECRET_NAME = re.compile(
    r"(?:^|[_.-])(?:path|dir|file|url|uri|name|env|var|id|template|format|prefix|suffix|type)s?$", re.I
)
# a value that names an environment variable rather than holding a key: "OPENAI_API_KEY"
_ENV_VAR_VALUE = re.compile(r"[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+")


def _assignment(m: re.Match[str]) -> str:
    name, value = m.group("name"), m.group("value")
    if not _SECRET_NAME.search(name) or _NOT_SECRET_NAME.search(name) or _ENV_VAR_VALUE.fullmatch(value):
        return m.group(0)
    return _value(m)


# Each rule: pattern, replacement, and substrings of which one must occur for the rule to run (a fast
# pre-check; empty means always run). Order matters: specific formats first, generic assignments last.
_RULES: list[tuple[re.Pattern[str], Replace, tuple[str, ...]]] = [
    # private key blocks (PEM, OpenSSH, PGP), also inside a string literal with \n escapes
    (
        re.compile(
            r"-----BEGIN[A-Z0-9 ]*PRIVATE KEY[A-Z ]*-----.*?-----END[A-Z0-9 ]*PRIVATE KEY[A-Z ]*-----", re.S
        ),
        _whole,
        ("PRIVATE KEY",),
    ),
    # an unterminated block: everything up to the end of the string literal, or of the text
    (re.compile(r"-----BEGIN[A-Z0-9 ]*PRIVATE KEY[A-Z ]*-----[^\"'`]*"), _whole, ("PRIVATE KEY",)),
    # OpenAI sk-... and sk-proj-..., Anthropic sk-ant-...
    (re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{20,}"), _whole, ("sk-",)),
    # GitHub classic and fine-grained tokens
    (re.compile(r"(?<![A-Za-z0-9])gh[pousr]_[A-Za-z0-9]{30,}"), _whole, ("gh",)),
    (re.compile(r"(?<![A-Za-z0-9])github_pat_[A-Za-z0-9_]{20,}"), _whole, ("github_pat_",)),
    # Hugging Face
    (re.compile(r"(?<![A-Za-z0-9])hf_[A-Za-z0-9]{30,}"), _whole, ("hf_",)),
    # AWS access key ids
    (re.compile(r"(?<![A-Za-z0-9])(?:AKIA|ASIA)[A-Z0-9]{16}(?![A-Za-z0-9])"), _whole, ("AKIA", "ASIA")),
    # Google API keys
    (re.compile(r"(?<![A-Za-z0-9])AIza[A-Za-z0-9_-]{35}"), _whole, ("AIza",)),
    # Slack
    (re.compile(r"(?<![A-Za-z0-9])xox[abprs]-[A-Za-z0-9-]{10,}"), _whole, ("xox",)),
    # JSON web tokens
    (
        re.compile(r"(?<![A-Za-z0-9])eyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"),
        _whole,
        ("eyJ",),
    ),
    # kaggle.json: {"username": "...", "key": "<32 hex>"}
    (re.compile(r"""["']key["']\s*:\s*["'](?P<value>[0-9a-fA-F]{32})(?=["'])"""), _value, ("key",)),
    # Authorization: Bearer <token>
    (re.compile(r"(?i)\bbearer[ \t]+(?P<value>[A-Za-z0-9._~+/-]{16,}=*)"), _value, ("earer", "EARER")),
    # scheme://user:password@host
    (re.compile(r"(?<=://)(?P<value>[^\s/:@'\"]+:[^\s/@'\"]+)@"), _value, ("://",)),
    # api_key = "...", password: str = "...", "token": "...", os.environ["X_SECRET"] = "...",
    # login(token="..."), os.environ.setdefault("X_TOKEN", "...")
    (
        re.compile(
            r"""\b(?P<name>[A-Za-z_][\w.-]{0,80})["']?\]?(?:[ \t]*:[ \t]*[A-Za-z_][\w.\[\]| ]{0,40}?)?"""
            r"""[ \t]*(?::=|=>|[=:,])[ \t]*[rbuRBU]{0,2}(?P<q>["'])(?P<value>[^\s"'`\\<>|]{12,4096})(?P=q)"""
        ),
        _assignment,
        (),
    ),
    # %env NAME=value, %set_env NAME value, export NAME=value (magics may be commented out by extraction)
    (
        re.compile(
            r"(?:%env|%set_env|\bexport)[ \t]+(?P<name>[A-Za-z_]\w{0,80})(?:[ \t]*=[ \t]*|[ \t]+)"
            r"(?P<value>[^\s\"'`$(){}\[\],;<>|]{12,4096})"
        ),
        _assignment,
        ("%env", "%set_env", "export"),
    ),
    # NAME=value in .env files and shell commands (HF_TOKEN=... python run.py)
    (
        re.compile(
            r"(?:^|(?<=[\s;&|]))(?P<name>[A-Za-z_]\w{0,80})=(?P<value>[^\s\"'`$(){}\[\],;<>|]{12,4096})(?=\s|$)",
            re.M,
        ),
        _assignment,
        ("=",),
    ),
]


def redact(text: str) -> str:
    """`text` with secrets replaced by [REDACTED]. Line breaks are kept, so line numbers do not move."""
    if not text:
        return text
    for pattern, replace, hints in _RULES:
        if hints and not any(h in text for h in hints):
            continue
        text = pattern.sub(replace, text)
    return text


def redact_value(value: Any) -> Any:
    """A JSON-ready copy of `value` with every string redacted; other objects become redacted strings."""
    if isinstance(value, str):
        return redact(value)
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, dict):
        return {redact(str(k)): redact_value(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [redact_value(v) for v in value]
    return redact(str(value))


_VIEWS: weakref.WeakKeyDictionary[Tree, tuple[tuple[tuple[str, str], ...], Tree]] = (
    weakref.WeakKeyDictionary()
)


def redact_tree(tree: Tree) -> Tree:
    """A redacted view of `tree`: the same paths and line numbers, with secrets replaced by [REDACTED].

    The LLM's tools and the verification of its quotes both use this view, so the LLM never sees a secret
    and a quote that spans a redacted value still verifies. Views are cached while the tree is unchanged.
    """
    stamp = tuple((path, f.sha256) for path, f in tree.files.items())
    cached = _VIEWS.get(tree)
    if cached is not None and cached[0] == stamp:
        return cached[1]
    view = Tree(tree.source_sha256, [redact(w) for w in tree.warnings], redact_value(tree.meta))
    for path, f in tree.files.items():
        text = "" if f.binary else redact(f.text)
        if text == f.text:
            sha, size = f.sha256, f.size
        else:
            raw = text.encode("utf-8", "replace")
            sha, size = util.sha256_bytes(raw), len(raw)
        view.files[path] = TreeFile(path, text, redact(f.origin), sha, size, f.binary)
    _VIEWS[tree] = (stamp, view)
    return view
