"""Secret redaction for everything an LLM sees: prompts, tool outputs and transcripts.

Known key formats are replaced by [REDACTED]: OpenAI (sk-, sk-proj-), Anthropic (sk-ant-), GitHub (ghp_,
gho_, ghu_, ghs_, ghr_, github_pat_), Hugging Face (hf_), AWS access key ids (AKIA, ASIA) and secret access
keys, Google (AIza), Slack (xox?-), JSON web tokens, bearer tokens, Basic auth credentials, private key
blocks, passwords in URLs (also without a user name: redis://:password@host) and kaggle.json keys. Values
assigned to secret-like names are replaced too (the name stays, the value goes):
- quoted, with any string prefix or triple quotes: `api_key = "..."`, `client_secret = f"..."`;
- passwords and passphrases, which may be short or hold spaces: `password = "correct horse battery"`;
- unquoted, in YAML, INI and .env files: `api_key: ...`, `aws_secret_access_key = ...`, `DB_PASSWORD=changeme`;
- in magics and shell commands, also in a notebook's `!` lines and in strings handed to a shell:
  `%env MY_TOKEN=...`, `export MY_TOKEN=...`, `!MY_TOKEN=... python run.py`,
  `os.system("MY_TOKEN=... python run.py")`.

An unquoted value on a line whose name holds a password (it ends in password, passwd, pwd or passphrase:
DB_PASSWORD, self.password, but not password_auth) is the password when it has 6 or more characters, whatever
they are (`changeme`, `123456`), as it would be in quotes. On a settings line, values that read as code stay:
names of environment variables, dotted names, CamelCase class names and variables named for a password
(`password: SecretStr`, `self.password = password`, `password = args.password`). In a shell command or magic
a bare word is text, so only an environment variable's name stays, and, below 12 characters, a variable
named for a password, which a keyword argument looks like (`password=password`).

Replacements keep every line break (\\n, \\r\\n and \\r), so a redacted file has the line numbers of the
original, and quotes verified against the redacted view point at the right lines. Every rule takes time
linear in the text: where a match can fail late, its quantifiers are bounded or possessive, a shell
assignment never starts inside a value that an earlier attempt scanned, and private key blocks are found in
one pass over their BEGIN and END lines.
"""

from __future__ import annotations

import base64
import functools
import re
import weakref
from collections.abc import Callable
from typing import Any

from pseudorca import util
from pseudorca.tree import Tree, TreeFile

REDACTED = "[REDACTED]"
TOO_DEEP = "[nested too deeply]"
MAX_DEPTH = 64  # levels of lists and dicts redact_value keeps; deeper ones become TOO_DEEP

Replace = Callable[[re.Match[str]], str]

# the line breaks of tree.split_lines
_LINE_BREAK = re.compile(r"\r\n|\r|\n")


def _breaks(text: str, start: int = 0, end: int | None = None) -> str:
    """The line breaks in text[start:end], in order, as they are (\\n, \\r\\n or \\r)."""
    return "".join(_LINE_BREAK.findall(text, start, len(text) if end is None else end))


def _whole(m: re.Match[str]) -> str:
    """Replace the whole match, keeping its line breaks."""
    return REDACTED + _breaks(m.group(0))


def _span(m: re.Match[str], group: str) -> str:
    """Replace one group, keeping the rest of the match (such as the variable name) and its line breaks."""
    text, base = m.group(0), m.start()
    start, end = m.start(group) - base, m.end(group) - base
    return text[:start] + REDACTED + _breaks(text, start, end) + text[end:]


def _value(m: re.Match[str]) -> str:
    """Replace only the `value` group."""
    return _span(m, "value")


# a name that holds a secret: api_key, HF_TOKEN, client_secret, db_password (but not tokenizer, ...)
_SECRET_NAME = re.compile(r"key|secret|passw(?:or)?d|pwd|passphrase|credential|token(?!izer)", re.I)
# secrets that people choose: they can be short and can hold spaces
_PASSWORD_WORD = r"passw(?:or)?d|pwd|passphrase"
_PASSWORD_NAME = re.compile(_PASSWORD_WORD, re.I)
# a name that ends in a password word holds a password (DB_PASSWORD, self.password, admin_pwd, passwords);
# other names with one hold a setting about passwords (password_auth, password_hash, passphrase_iterations)
_HOLDS_PASSWORD = re.compile(r"(?:" + _PASSWORD_WORD + r")s?\d*$", re.I)
MIN_PASSWORD_CHARS = 6  # passwords can be short
MIN_KEY_CHARS = 12  # keys and tokens are long, and a shorter value is usually a setting
# unquoted values on a password line that are code, not the password: dotted names (args.password), CamelCase
# class names (SecretStr) and variables with a password word among their words (password, user_password,
# dbPassword, password_mgr), unless they end in a digit (password1 is a password)
_DOTTED_NAME = re.compile(r"[A-Za-z_]+(?:\.[A-Za-z_]+)+")
_CLASS_NAME = re.compile(r"(?:[A-Z][a-z]+){2,}")
_VARIABLE = re.compile(r"[A-Za-z_]\w*[A-Za-z_]")
_WORD = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])")  # the words of a snake_case or camelCase name
_PASSWORD_WORD_ONLY = re.compile(r"(?:" + _PASSWORD_WORD + r")s?", re.I)
# names that hold where a secret is, or text about it, not the secret: KEY_PATH, token_file, api_key_env,
# SECRET_NAME, password_prompt, key_size
_NOT_SECRET_NAME = re.compile(
    r"(?:^|[_.-])(?:path|dir|file|url|uri|name|env|var|id|template|format|prefix|suffix|type"
    r"|prompt|hint|policy|label|message|msg|help|header|field|column|length|len|size|count)s?$",
    re.I,
)
# a value that names an environment variable rather than holding a key: "OPENAI_API_KEY"
_ENV_VAR_VALUE = re.compile(r"[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+")
# unquoted values that are settings or code rather than secrets: booleans, numbers, words of up to 15
# letters and dotted names without digits (true, 4096, SecretStr, config.api_key). A longer run of letters
# can be a key.
_PLAIN_VALUE = re.compile(
    r"(?i:true|false|yes|no|on|off|null|none|nil)|[-+]?\d[\d_]*(?:\.\d+)?(?:[eE][-+]?\d+)?"
    r"|[A-Za-z_]{1,15}|" + _DOTTED_NAME.pattern
)
# names made of words, in snake_case or CamelCase without digits (resolved_api_key, InputTokensDetails):
# code, not a key or token, which are random. Passwords are exempt, as passphrases can look like this.
_WORDS_NAME = re.compile(r"_*[a-z]{1,16}(?:_+[a-z]{1,16})+_*|_*(?:[A-Z][a-z]{1,16})+")
# a template with replacement fields (f"{model}_{seed}", "${API_KEY}", "{{ token }}"): it builds a value
# from others rather than holding one
_TEMPLATE = re.compile(r"\{[^{}]*\}")


def _is_secret_name(name: str) -> bool:
    return bool(_SECRET_NAME.search(name)) and not _NOT_SECRET_NAME.search(name)


def _min_chars(name: str) -> int:
    """Passwords can be short; keys and tokens are long, and a short value is usually a setting."""
    return MIN_PASSWORD_CHARS if _PASSWORD_NAME.search(name) else MIN_KEY_CHARS


def _password_variable(value: str) -> bool:
    """A variable named for a password (password, user_password, dbPassword), not a password."""
    return bool(_VARIABLE.fullmatch(value)) and any(
        _PASSWORD_WORD_ONLY.fullmatch(word) for word in _WORD.findall(value)
    )


def _password_code(value: str) -> bool:
    """An unquoted value on a password line that reads as code rather than as the password itself."""
    return bool(_DOTTED_NAME.fullmatch(value) or _CLASS_NAME.fullmatch(value)) or _password_variable(value)


def _is_code(name: str, value: str) -> bool:
    """`value` names something (a variable, a class, an environment variable) or is a template, so it holds
    no secret itself. For a password only an environment variable's name counts: passphrases can be words."""
    if _ENV_VAR_VALUE.fullmatch(value):
        return True
    return not _PASSWORD_NAME.search(name) and bool(_WORDS_NAME.fullmatch(value) or _TEMPLATE.search(value))


def _assignment(m: re.Match[str]) -> str:
    name, value = m.group("name"), m.group("value")
    if not _is_secret_name(name) or _is_code(name, value):
        return m.group(0)
    return _value(m)


_QUOTED_GROUPS = ("v1", "v2", "v3", "v4")


def _password(m: re.Match[str]) -> str:
    name, group = m.group("name"), next(g for g in _QUOTED_GROUPS if m.group(g) is not None)
    if _NOT_SECRET_NAME.search(name) or _is_code(name, m.group(group)):
        return m.group(0)
    return _span(m, group)


def _unquoted(m: re.Match[str]) -> str:
    """A value on a settings line (YAML, INI, .env, properties).

    On a line that holds a password, words and numbers are passwords (DB_PASSWORD=changeme), as they are in
    quotes. On other secret lines they are settings: true, 4096, required, SecretStr.
    """
    name, value = m.group("name"), m.group("value")
    if not _is_secret_name(name) or len(value) < _min_chars(name) or _is_code(name, value):
        return m.group(0)
    if _HOLDS_PASSWORD.search(name):
        keep = _password_code(value)
    else:
        keep = bool(_PLAIN_VALUE.fullmatch(value))
    return m.group(0) if keep else _value(m)


def _command(m: re.Match[str]) -> str:
    """A value in a magic or a shell command: %env NAME=value, export NAME=value, NAME=value cmd.

    A shell reads a bare word as text, so a value of MIN_KEY_CHARS or more is redacted unless `_is_code`
    says it names something (an environment variable; for keys and tokens also a name made of words, or a
    template). On a line that holds a password, so is a shorter one of MIN_PASSWORD_CHARS or more, unless it
    is a variable named for a password: a keyword argument looks the same (password=password).
    """
    name, value = m.group("name"), m.group("value")
    if not _is_secret_name(name) or _is_code(name, value):
        return m.group(0)
    if len(value) >= MIN_KEY_CHARS:
        return _value(m)
    if len(value) < MIN_PASSWORD_CHARS or not _HOLDS_PASSWORD.search(name) or _password_variable(value):
        return m.group(0)
    return _value(m)


def _basic(m: re.Match[str]) -> str:
    """Basic auth sends base64 of user:password; words after "basic" in prose do not decode to that."""
    try:
        raw = base64.b64decode(m.group("value"), validate=True)
    except ValueError:  # binascii.Error
        return m.group(0)
    return _value(m) if b":" in raw else m.group(0)


_PEM_BEGIN = re.compile(r"-----BEGIN[A-Z0-9 ]{0,40}PRIVATE KEY[A-Z ]{0,20}-----")
_PEM_END = re.compile(r"-----END[A-Z0-9 ]{0,40}PRIVATE KEY[A-Z ]{0,20}-----")
_LITERAL_END = re.compile(r"[\"'`]")


def _private_keys(text: str) -> str:
    """Private key blocks (PEM, OpenSSH, PGP), also inside a string literal with \\n escapes.

    A block runs from its BEGIN line to the next END line. Without one it runs to the end of its string
    literal (the next quote), or of the text. A pattern like BEGIN.*?END would scan the rest of the text
    again for every BEGIN that has no END; here each header is looked at once.
    """
    parts: list[str] = []
    pos = 0
    more_ends = True  # False once a search found no END: none follows any later BEGIN either
    for begin in _PEM_BEGIN.finditer(text):
        if begin.start() < pos:
            continue  # inside the block before
        end = _PEM_END.search(text, begin.end()) if more_ends else None
        if end is not None:
            stop = end.end()
        else:
            more_ends = False
            quote = _LITERAL_END.search(text, begin.end())
            stop = len(text) if quote is None else quote.start()
        parts += [text[pos : begin.start()], REDACTED + _breaks(text, begin.start(), stop)]
        pos = stop
    parts.append(text[pos:])
    return "".join(parts)


def _sub(pattern: str, replace: Replace, flags: int = 0) -> Callable[[str], str]:
    return functools.partial(re.compile(pattern, flags).sub, replace)


# A name, and what follows it in an assignment: an optional closing quote or bracket and type annotation,
# then =, :=, => or :, or a comma after a quoted name ("NAME", "value"), and a string prefix (r, b, u, f, t
# and pairs such as rb or fr). After an unquoted name, a comma separates arguments: join(pwd, "data").
# None of the characters that can follow a name can be part of one, so the name is possessive ({0,80}+):
# it never gives characters back, which would only retry the same failure. The values below are possessive
# for the same reason: their closing quote, or the end of the line, cannot be part of them.
_NAME = r"(?P<name>[A-Za-z_][\w.-]{0,80}+)"
_ASSIGN = (
    r"""(?:["'][ \t]*,|["']?\]?(?:[ \t]*:[ \t]*[A-Za-z_][\w.\[\]| ]{0,40}?)?[ \t]*(?::=|=>|[=:]))"""
    r"""[ \t]*[rbuftRBUFT]{0,2}"""
)

# Each rule: the function that applies it, and substrings of which one must occur for it to run (a fast
# pre-check; empty means always run). Order matters: specific formats first, generic assignments last.
_RULES: list[tuple[Callable[[str], str], tuple[str, ...]]] = [
    (_private_keys, ("PRIVATE KEY",)),
    # AWS secret access keys: credentials files, code and `aws configure set aws_secret_access_key ...`
    (
        _sub(
            r"(?i)(?<![A-Za-z0-9])aws_?secret_?access_?key[\"']?\]?[ \t]*(?:[:=,][ \t]*)?[\"']?"
            r"(?P<value>[A-Za-z0-9/+]{40})(?![A-Za-z0-9/+=])",
            _value,
        ),
        ("ecret", "ECRET"),
    ),
    # OpenAI sk-... and sk-proj-..., Anthropic sk-ant-...
    (_sub(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{20,}", _whole), ("sk-",)),
    # GitHub classic and fine-grained tokens
    (_sub(r"(?<![A-Za-z0-9])gh[pousr]_[A-Za-z0-9]{30,}", _whole), ("gh",)),
    (_sub(r"(?<![A-Za-z0-9])github_pat_[A-Za-z0-9_]{20,}", _whole), ("github_pat_",)),
    # Hugging Face
    (_sub(r"(?<![A-Za-z0-9])hf_[A-Za-z0-9]{30,}", _whole), ("hf_",)),
    # AWS access key ids
    (_sub(r"(?<![A-Za-z0-9])(?:AKIA|ASIA)[A-Z0-9]{16}(?![A-Za-z0-9])", _whole), ("AKIA", "ASIA")),
    # Google API keys
    (_sub(r"(?<![A-Za-z0-9])AIza[A-Za-z0-9_-]{35}", _whole), ("AIza",)),
    # Slack
    (_sub(r"(?<![A-Za-z0-9])xox[abprs]-[A-Za-z0-9-]{10,}", _whole), ("xox",)),
    # JSON web tokens. A token starts where its first segment does (not after a letter, digit, _ or -), and
    # the possessive {8,}+ never gives characters back, so each segment is scanned a bounded number of times.
    (
        _sub(r"(?<![A-Za-z0-9_-])eyJ[A-Za-z0-9_-]{8,}+\.eyJ[A-Za-z0-9_-]{8,}+\.[A-Za-z0-9_-]{8,}+", _whole),
        ("eyJ",),
    ),
    # kaggle.json: {"username": "...", "key": "<32 hex>"}
    (_sub(r"""["']key["']\s*:\s*["'](?P<value>[0-9a-fA-F]{32})(?=["'])""", _value), ("key",)),
    # Authorization: Bearer <token>
    (_sub(r"(?i)\bbearer[ \t]+(?P<value>[A-Za-z0-9._~+/-]{16,}=*)", _value), ("earer", "EARER")),
    # Authorization: Basic <base64 of user:password>
    (
        _sub(r"(?i)\bbasic[ \t]+(?P<value>[A-Za-z0-9+/]{4,}+={0,2})(?![A-Za-z0-9+/=])", _basic),
        ("asic", "ASIC"),
    ),
    # scheme://user:password@host, and scheme://:password@host (Redis has no user name)
    (_sub(r"(?<=://)(?P<value>[^\s/:@'\"]*:[^\s/@'\"]+)@", _value), ("://",)),
    # api_key = "...", password: str = "...", "token": "...", os.environ["X_SECRET"] = "...",
    # login(token="..."), os.environ.setdefault("X_TOKEN", "..."), API_TOKEN = """...""", key = f"..."
    (
        _sub(
            r"\b"
            + _NAME
            + _ASSIGN
            + r'(?P<q>"""|'
            + r"'''|[\"'])"
            + r"""(?P<value>[^\s"'`\\<>|]{12,4096}+)(?P=q)""",
            _assignment,
        ),
        (),
    ),
    # passwords and passphrases: the value runs to its closing quote and may be short or hold spaces
    (
        _sub(
            r"(?<![\w.-])(?=[\w.-]{0,80}?(?i:"
            + _PASSWORD_WORD
            + r"))"
            + _NAME
            + _ASSIGN
            + r'(?:"""(?P<v1>[^"\n]{6,4096}+)"""'
            + r"|'''(?P<v2>[^'\n]{6,4096}+)'''"
            + r'|"(?P<v3>(?:[^"\\\n]|\\.){6,4096}+)"'
            + r"|'(?P<v4>(?:[^'\\\n]|\\.){6,4096}+)')",
            _password,
        ),
        ("assw", "ASSW", "wd", "WD", "assphrase", "ASSPHRASE"),
    ),
    # %env NAME=value, %set_env NAME value, export NAME=value (magics may be commented out by extraction)
    (
        _sub(
            r"(?:%env|%set_env|\bexport)[ \t]+(?P<name>[A-Za-z_]\w{0,80}+)(?:[ \t]*=[ \t]*|[ \t]+)"
            r"(?P<value>[^\s\"'`$(){}\[\],;<>|]{6,4096})",
            _command,
        ),
        ("%env", "%set_env", "export"),
    ),
    # unquoted settings, one per line (YAML, INI, .env, properties): `api_key: ...`, `password = ...`,
    # `- token: ...`, with an optional comment after the value. Before the shell rule below, which stops a
    # value at a shell separator such as &, so that a .env value holding one is redacted whole.
    (
        _sub(
            r"^[ \t]*(?:-[ \t]+)?"
            + _NAME
            + r"[ \t]*[:=][ \t]*(?P<value>[^\s\"'`\\<>|()\[\]{},$]{6,4096}+)(?:[ \t]+[#;][^\n]*)?[ \t]*\r?$",
            _unquoted,
            re.M,
        ),
        (),
    ),
    # NAME=value in shell commands: `HF_TOKEN=... python run.py`, also in a notebook's `!` line and in the
    # strings handed to a shell, `os.system("HF_TOKEN=... python run.py")`. A match starts at a line start
    # or after a shell separator (whitespace, ; & | or a quote), and a value holds none of them: no match can
    # start inside a value that an earlier attempt scanned, so every character is scanned once.
    (
        _sub(
            r"(?:^|(?<=[\s;&|'\"`]))!?(?P<name>[A-Za-z_]\w{0,80}+)="
            r"(?P<value>[^\s\"'`$(){}\[\],;<>|&]{6,4096}+)(?=[\s;&|'\"`]|$)",
            _command,
            re.M,
        ),
        ("=",),
    ),
]


def redact(text: str) -> str:
    """`text` with secrets replaced by [REDACTED]. Line breaks are kept, so line numbers do not move."""
    if not text:
        return text
    for apply, hints in _RULES:
        if hints and not any(h in text for h in hints):
            continue
        text = apply(text)
    return text


def redact_value(value: Any) -> Any:
    """A JSON-ready copy of `value` with every string redacted; other objects become redacted strings.

    Lists and dicts nested more than MAX_DEPTH levels deep (which a model's tool arguments can be) become
    "[nested too deeply]", so the copy can always be written as JSON.
    """

    def copy(value: Any, depth: int) -> Any:
        if isinstance(value, str):
            return redact(value)
        if value is None or isinstance(value, bool | int | float):
            return value
        if depth >= MAX_DEPTH:
            return TOO_DEEP
        if isinstance(value, dict):
            return {redact(str(k)): copy(v, depth + 1) for k, v in value.items()}
        if isinstance(value, list | tuple):
            return [copy(v, depth + 1) for v in value]
        try:
            return redact(str(value))
        except RecursionError:  # the repr of a deeply nested object
            return TOO_DEEP

    return copy(value, 0)


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
