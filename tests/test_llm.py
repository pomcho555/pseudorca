"""LLM settings, the remote guard, redaction and the judge's tools (no network anywhere)."""

from __future__ import annotations

import base64
import random
import string
import subprocess
import sys
import time

import pytest

from pseudorca.llm import LLMError, LLMSettings, ensure_allowed, load_settings
from pseudorca.llm.backends import parse_json_reply
from pseudorca.llm.redact import REDACTED, redact, redact_tree, redact_value
from pseudorca.llm.tools import (
    MAX_HITS,
    MAX_LIST,
    TOOL_SPECS,
    TRUNCATED,
    ToolHandler,
    anthropic_tools,
    openai_tools,
    parse_verdict,
)
from pseudorca.store import Store
from pseudorca.tree import Tree

ALNUM = string.ascii_letters + string.digits


def fake_secret(prefix: str, n: int, alphabet: str = ALNUM) -> str:
    """A key-shaped value built at run time, so this file holds nothing a secret scanner would flag."""
    rng = random.Random(f"{prefix}{n}{alphabet}")
    return prefix + "".join(rng.choice(alphabet) for _ in range(n))


def with_llm(store: Store, lines: str) -> Store:
    """The store with `lines` added under [llm] (the last table of the config template)."""
    path = store.root / "config.toml"
    path.write_text(path.read_text(encoding="utf-8") + lines, encoding="utf-8")
    return Store.at(store.root)


# ----- settings --------------------------------------------------------------------------------------


def test_settings_need_a_model(store: Store):
    assert load_settings(store) is None
    assert load_settings(store, model="local-model") == LLMSettings(provider="openai", model="local-model")


def test_settings_come_from_config_and_overrides_win(store: Store):
    st = with_llm(
        store,
        'provider = "anthropic"\nmodel = "m1"\nbase_url = "http://localhost:8000/v1"\napi_key_env = "MY_KEY"\n'
        "allow_remote = true\nmax_steps = 5\nmax_tool_chars = 3000\ntemperature = 0.5\n",
    )
    s = load_settings(st)
    assert s == LLMSettings("anthropic", "m1", "http://localhost:8000/v1", "MY_KEY", True, 5, 3000, 0.5)
    o = load_settings(st, provider="openai", model="m2", base_url=None, allow_remote=False, max_steps=None)
    assert (o.provider, o.model, o.base_url, o.allow_remote, o.max_steps) == (
        "openai",
        "m2",
        "http://localhost:8000/v1",
        False,
        5,
    )
    assert o.label == "openai:m2"


@pytest.mark.parametrize(
    "line",
    [
        'provider = "gemini"',
        "max_steps = 0",
        'max_tool_chars = "lots"',
        "allow_remote = 1",
        "temperature = 9",
        'base_url = "localhost:8000/v1"',
        'base_url = "file:///etc/hosts"',
    ],
)
def test_settings_reject_bad_values(store: Store, line: str):
    st = with_llm(store, f'model = "m"\n{line}\n')
    with pytest.raises(LLMError):
        load_settings(st)


def test_api_key_env_must_be_a_name_and_is_never_echoed(store: Store):
    key = fake_secret("sk-", 48)
    st = with_llm(store, f'model = "m"\napi_key_env = "{key}"\n')
    with pytest.raises(LLMError) as e:
        load_settings(st)
    assert key not in str(e.value) and "NAME" in str(e.value)


def test_unknown_override_is_a_programming_error(store: Store):
    with pytest.raises(TypeError):
        load_settings(store, modle="x")


# ----- remote guard ----------------------------------------------------------------------------------

LOCAL_URLS = [
    "http://localhost:8000/v1",
    "https://LOCALHOST./v1",
    "http://127.0.0.1:8000/v1",
    "http://127.4.5.6/v1",
    "http://[::1]:8000/v1",
    "http://0.0.0.0:8000/v1",
    "http://foo.localhost:11434/v1",
]
REMOTE_URLS = [
    "http://192.168.1.20:8000/v1",
    "http://10.0.0.5/v1",
    "https://api.example.com/v1",
    "http://localhost.example.com/v1",
    "http://localhost@example.com/v1",
    "http://127.0.0.1.example.com/v1",
    "http://[::]:8000/v1",
]


@pytest.mark.parametrize("url", LOCAL_URLS)
def test_local_endpoints_are_allowed(url: str):
    ensure_allowed(LLMSettings("openai", "m", base_url=url))


@pytest.mark.parametrize("url", [*REMOTE_URLS, None])
def test_remote_endpoints_need_allow_remote(url: str | None):
    with pytest.raises(LLMError) as e:
        ensure_allowed(LLMSettings("openai", "m", base_url=url))
    message = str(e.value)
    assert "allow_remote = true" in message and "--allow-remote" in message and "notebook code" in message
    if url is None:
        assert "api.openai.com" in message
    ensure_allowed(LLMSettings("openai", "m", base_url=url, allow_remote=True))


def test_anthropic_is_always_remote_and_fake_is_always_local():
    for url in (None, "http://localhost:8000"):
        with pytest.raises(LLMError):
            ensure_allowed(LLMSettings("anthropic", "m", base_url=url))
        ensure_allowed(LLMSettings("anthropic", "m", base_url=url, allow_remote=True))
    ensure_allowed(LLMSettings("fake", "m"))
    ensure_allowed(LLMSettings("fake", "m", base_url="https://api.example.com/v1"))
    with pytest.raises(LLMError):
        ensure_allowed(LLMSettings("other", "m", allow_remote=True))


# ----- redaction -------------------------------------------------------------------------------------

UPPER = string.ascii_uppercase + string.digits
TOKENS = {
    "openai": lambda: fake_secret("sk-", 48),
    "openai project": lambda: fake_secret("sk-" + "proj-", 100, ALNUM + "_-"),
    "anthropic": lambda: fake_secret("sk-" + "ant-api03-", 90, ALNUM + "_-"),
    "github ghp": lambda: fake_secret("gh" + "p_", 36),
    "github gho": lambda: fake_secret("gh" + "o_", 36),
    "github ghu": lambda: fake_secret("gh" + "u_", 36),
    "github ghs": lambda: fake_secret("gh" + "s_", 36),
    "github ghr": lambda: fake_secret("gh" + "r_", 36),
    "github fine-grained": lambda: fake_secret("github" + "_pat_", 82, ALNUM + "_"),
    "hugging face": lambda: fake_secret("hf" + "_", 34, string.ascii_letters),
    "aws akia": lambda: fake_secret("AK" + "IA", 16, UPPER),
    "aws asia": lambda: fake_secret("AS" + "IA", 16, UPPER),
    "google": lambda: fake_secret("AI" + "za", 35, ALNUM + "_-"),
    "slack": lambda: "xo" + "xb-" + fake_secret("", 12, string.digits) + "-" + fake_secret("", 24),
    "jwt": lambda: ".".join(["eyJ" + fake_secret("", 20), "eyJ" + fake_secret("", 30), fake_secret("", 40)]),
}


@pytest.mark.parametrize("kind", sorted(TOKENS))
def test_redact_known_key_formats(kind: str):
    secret = TOKENS[kind]()
    text = f'value = "{secret}"\nprint(1)\nother = {secret} # end\n'
    out = redact(text)
    assert secret not in out
    assert out == f'value = "{REDACTED}"\nprint(1)\nother = {REDACTED} # end\n'


def _private_key() -> str:
    body = "\n".join(fake_secret("", 64, ALNUM + "+/") for _ in range(3))
    return "-----BEGIN " + "OPENSSH PRIVATE KEY-----\n" + body + "\n-----END " + "OPENSSH PRIVATE KEY-----"


def test_redact_private_key_blocks_and_keep_line_numbers():
    pem = _private_key()
    text = f"a = 1\n{pem}\nb = 2\n"
    out = redact(text)
    assert "PRIVATE KEY" not in out and out.count("\n") == text.count("\n")
    assert (
        out.splitlines()[0] == "a = 1" and out.splitlines()[1] == REDACTED and out.splitlines()[-1] == "b = 2"
    )
    inline = 'KEY = "' + pem.replace("\n", "\\n") + '"\nnext_line = 1'
    assert redact(inline) == f'KEY = "{REDACTED}"\nnext_line = 1'
    unterminated = 'k = "' + pem.split("\n-----END")[0].replace("\n", "\\n") + '"\nx = 1'
    assert redact(unterminated) == f'k = "{REDACTED}"\nx = 1'


def test_redact_kaggle_json_and_credentials_in_urls():
    key = fake_secret("", 32, "0123456789abcdef")
    assert redact('{"username": "someone", "key": "' + key + '"}') == (
        '{"username": "someone", "key": "' + REDACTED + '"}'
    )
    password = fake_secret("", 14)
    assert redact(f'url = "https://user:{password}@example.com/repo.git"') == (
        f'url = "https://{REDACTED}@example.com/repo.git"'
    )
    token = fake_secret("", 30)
    assert redact(f'headers = {{"Authorization": "Bearer {token}"}}') == (
        f'headers = {{"Authorization": "Bearer {REDACTED}"}}'
    )


@pytest.mark.parametrize(
    "template",
    [
        'api_key = "{v}"',
        'API_TOKEN="{v}"',
        "client_secret: '{v}'",
        'db_password: str = "{v}"',
        'passwd = "{v}"',
        'os.environ["MY_SERVICE_TOKEN"] = "{v}"',
        'os.environ.setdefault("MY_SERVICE_KEY", "{v}")',
        'config = {{"api_key": "{v}"}}',
        'login(token="{v}")',
        "# %env MY_SERVICE_KEY={v}",
        "%set_env MY_SERVICE_KEY {v}",
        "export MY_SECRET={v}",
        "MY_TOKEN={v} python run.py",
    ],
)
def test_redact_generic_assignments_keep_the_name(template: str):
    value = fake_secret("", 20)
    out = redact(template.format(v=value))
    assert value not in out
    assert out == template.format(v=REDACTED)


@pytest.mark.parametrize(
    "text",
    [
        "max_tokens = 512",
        'eos_token = "<|eot_id|>"',
        'TOKENIZER_PATH = "/kaggle/input/some-model/tokenizer"',
        'api_key_env = "OPENAI_API_KEY"',
        'key = os.environ["OPENAI_API_KEY"]',
        'secret_name = "HF_TOKEN"',
        'tokenizer = AutoTokenizer.from_pretrained("org/model-7b-instruct")',
        'cache_dir = "/tmp/cache/somewhere/deep"',
        'print("task-1234567890abcdefghijklmnop")',
        'short_key = "abc123"',
        # code and settings that the rules for unquoted values, passwords and Basic auth must not touch
        "api_key = config.api_key",
        "token = tokenizer.eos_token",
        "password: SecretStr",
        "num_tokens = len(tokenizer.encode(text))",
        "    max_tokens: 4096",
        "key_size = 2048",
        "use_auth_token: true",
        "api_key: ${OPENAI_API_KEY}",
        "secret_key = SECRET_KEY",
        "pwd = os.getcwd()",
        'password = os.environ.get("DB_PASSWORD")',
        "password_file = /run/secrets/db-password-0001",
        'KEY_IDEA = "rerank the beam with a verifier, width 4"',
        'password_prompt = "Enter the password for the vault"',
        'print("token", "is not valid for this endpoint")',
        "A basic implementation of beam search",
        "Basic usage: call solve() on each task",
        "self.api_key = resolved_api_key",
        "input_tokens_details: InputTokensDetails",
        '__all__ = ["ClientSecrets", "AsyncClientSecrets"]',
        'getter = getattr(credentials, "get_frozen_credentials", None)',
        'train = pd.read_csv(join(pwd, "./data/train-set-2024.csv"))',
        'cache_key = f"{model_name}_{seed}_{temperature}"',
        'api_key: "${OPENAI_API_KEY}"',
    ],
)
def test_redact_leaves_ordinary_code_alone(text: str):
    assert redact(text) == text


def _words(n: int) -> str:
    rng = random.Random(n)
    return " ".join(
        "".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(4, 7))) for _ in range(n)
    )


# (template, value): redact(template.format(v=value)) must be template.format(v=REDACTED)
SECRET_FORMS = {
    "aws credentials file": (
        "[default]\naws_secret_access_key = {v}\nregion = eu-west-1\n",
        fake_secret("", 40, ALNUM + "/+"),
    ),
    "aws credentials file, letters only": (
        "[profile ci]\naws_secret_access_key = {v}\n",
        fake_secret("", 40, string.ascii_letters),
    ),
    "aws cli": ("# !aws configure set aws_secret_access_key {v}", fake_secret("", 40, ALNUM + "/+")),
    "triple double quotes": ('API_TOKEN = """{v}"""', fake_secret("", 24)),
    "triple single quotes (TOML literal)": ("api_key = '''{v}'''", fake_secret("", 24)),
    "f-string": ('client_secret = f"{v}"', fake_secret("", 20)),
    "raw f-string": ("CLIENT_SECRET = rf'{v}'", fake_secret("", 20)),
    "password with spaces": ('password = "{v}"', _words(4)),
    "short password": ('db_password = "{v}"', fake_secret("", 7, ALNUM + "!#%&")),
    "name pwd": ('pwd = "{v}"', fake_secret("", 16)),
    "name passphrase": ('ssh_passphrase = "{v}"', _words(5)),
    "name credential": ('db_credential = "{v}"', fake_secret("", 16)),
    "basic auth": (
        'headers = {{"Authorization": "Basic {v}"}}',
        base64.b64encode(b"someone:" + fake_secret("", 12).encode()).decode(),
    ),
    "yaml, unquoted": ("llm:\n  api_key: {v}\n  model: some-model\n", fake_secret("", 24)),
    "yaml list item, unquoted": ("- token: {v}", fake_secret("", 24)),
    "ini, unquoted": (
        "[database]\nuser = someone\npassword = {v}\n",
        fake_secret("", 6, string.ascii_letters) + "#" + fake_secret("", 3, string.digits),
    ),
    ".env with spaces around =": ("API_TOKEN = {v}\nDEBUG = 1\n", fake_secret("", 30)),
    "unquoted, then a comment": ("secret_key: {v}  # rotate monthly", fake_secret("", 24)),
}


@pytest.mark.parametrize("form", list(SECRET_FORMS))
def test_redact_more_secret_forms(form: str):
    template, value = SECRET_FORMS[form]
    out = redact(template.format(v=value))
    assert value not in out
    assert out == template.format(v=REDACTED)
    assert redact(out) == out  # prompts are redacted again as a whole: nothing more may change


PEM_HEADER = "-----BEGIN " + "PRIVATE KEY-----"
# inputs on which a backtracking pattern takes quadratic time: many places where a match starts and fails late
HOSTILE = {
    "private key headers without END": lambda n: (PEM_HEADER + "\n") * (n // 28),
    "a private key header that never closes": lambda n: "-----BEGIN " + "PRIVATE KEY " * (n // 12),
    "an END header that never closes": lambda n: PEM_HEADER + "\n-----END " + "PRIVATE KEY " * (n // 12),
    "a run of -eyJ": lambda n: "-eyJ" * (n // 4),
    "a dotted name that never ends": lambda n: "a." * (n // 2),
}


@pytest.mark.parametrize("kind", list(HOSTILE))
def test_redaction_takes_linear_time_on_hostile_text(kind: str):
    text = HOSTILE[kind](256_000)
    start = time.perf_counter()
    out = redact(text)
    elapsed = time.perf_counter() - start
    assert elapsed < 2.0, f"{elapsed:.2f} s for {len(text) // 1000} kB"  # linear: well under 0.1 s
    assert out.count("\n") == text.count("\n")
    if "PRIVATE KEY-----" in text:  # an unterminated block is redacted to the end of the text
        assert "PRIVATE KEY" not in out


def test_redact_value_handles_nested_data():
    secret = fake_secret("sk-", 48)
    data = {"messages": [{"content": f"key {secret}"}], "n": 3, "ok": True, "other": {secret}}
    out = redact_value(data)
    assert secret not in repr(out) and out["n"] == 3 and out["ok"] is True
    assert out["messages"][0]["content"] == f"key {REDACTED}"


def test_redacted_view_has_the_same_paths_and_lines():
    secret = fake_secret("sk-", 48)
    tree = Tree("src", meta={"env": [{"cell": 0, "key": "OPENAI_API_KEY", "value": secret}]})
    tree.add_text("cells/cell_000.py", f'import os\nKEY = "{secret}"\nprint(1)\n', "cell 0")
    tree.add_text("markdown/cell_001.md", f"my key is {secret}", "cell 1")
    tree.add_bytes("payloads/cell_000/0.bin", b"\x00\x01", "payload")
    view = redact_tree(tree)
    assert view.paths() == tree.paths()
    assert view.lines("cells/cell_000.py") == ["import os", f'KEY = "{REDACTED}"', "print(1)"]
    assert secret not in view.text("markdown/cell_001.md") and secret not in repr(view.meta)
    assert view.files["payloads/cell_000/0.bin"].binary
    assert redact_tree(tree) is view  # cached while the tree is unchanged
    assert secret in tree.text("cells/cell_000.py")  # the original is untouched
    tree.add_text("cells/cell_002.py", "x = 1\n", "cell 2")
    assert "cells/cell_002.py" in redact_tree(tree)


# ----- tools -----------------------------------------------------------------------------------------


def make_tree() -> Tree:
    t = Tree("abc")
    t.add_text(
        "cells/cell_000.py",
        "import os\nTEMPERATURE = 0.2\n\ndef call(prompt):\n    return client.chat(prompt, temperature=TEMPERATURE)\n",
        "cell 0",
    )
    t.add_text("markdown/cell_001.md", "We use temperature 0.2", "cell 1")
    t.add_text(
        "files/src/agent.py",
        "\n".join(f"line {i}" for i in range(1, 501)),
        "cell 2 > %%writefile src/agent.py",
    )
    t.add_bytes("payloads/cell_002/0.bin", b"\x00\x01\x02", "cell 2 > base64")
    return t


def handler(tree: Tree | None = None, max_chars: int = 12000) -> ToolHandler:
    return ToolHandler(redact_tree(make_tree() if tree is None else tree), max_chars)


def test_list_files_shows_sizes_and_tags():
    h = handler()
    out = h.run("list_files", {})
    assert out.splitlines()[0] == "4 files match '*':"
    assert "cells/cell_000.py  " in out and " bytes, 5 lines" in out
    assert "markdown/cell_001.md" in out and "[markdown: a claim, not evidence]" in out
    assert "payloads/cell_002/0.bin  3 bytes  [binary, no text]" in out
    only = h.run("list_files", '{"glob": "files/*"}')
    assert "files/src/agent.py" in only and "cells/" not in only
    assert "cells/cell_000.py" in h.run("list_files", {"glob": "cells"})  # a bare directory name
    assert h.run("list_files", {"glob": "nothing/*"}).startswith("no files match")
    assert h.calls == h.ok_calls == 4


def test_list_files_is_capped():
    big = Tree("x")
    for i in range(MAX_LIST + 50):
        big.add_text(f"files/f{i:04d}.py", "x = 1\n", "o")
    out = handler(big).run("list_files", {})
    assert len(out.splitlines()) == MAX_LIST + 1
    assert out.startswith(f"{MAX_LIST + 50} files match '*'; showing the first {MAX_LIST}")


def test_grep_is_substring_only_even_for_regex_like_text():
    t = Tree("x")
    t.add_text("cells/cell_000.py", "abc = 1\na.c = 2\nx = re.compile('.*')\nABC = 3\n", "c")
    h = handler(t)
    out = h.run("grep", {"text": "a.c"})
    assert out.splitlines() == ["1 hits for 'a.c':", "cells/cell_000.py:2: a.c = 2"]
    assert h.run("grep", {"text": ".*"}).splitlines()[1:] == ["cells/cell_000.py:3: x = re.compile('.*')"]
    assert h.run("grep", {"text": "(unclosed["}).startswith("no lines contain")
    assert h.run("grep", {"text": "a+c"}).startswith("no lines contain")
    hits = h.run("grep", {"text": "abc", "ignore_case": "true"}).splitlines()[1:]
    assert hits == ["cells/cell_000.py:1: abc = 1", "cells/cell_000.py:4: ABC = 3"]
    assert h.run("grep", {"text": "abc", "path_glob": "files/*"}).startswith("no lines contain")


def test_grep_caps_hits_and_clips_long_lines():
    t = Tree("x")
    t.add_text("files/many.py", "\n".join(f"hit {i}" for i in range(80)), "o")
    t.add_text("files/long.py", "x" * 1000 + "NEEDLE" + "y" * 1000, "o")
    h = handler(t)
    out = h.run("grep", {"text": "hit", "path_glob": "files/many.py"})
    assert out.count("\nfiles/many.py:") == MAX_HITS and "narrow the search" in out
    line = h.run("grep", {"text": "NEEDLE"}).splitlines()[1]
    shown = line.split(": ", 1)[1]
    assert "NEEDLE" in shown and shown.startswith("…") and shown.endswith("…") and len(shown) == 302


def test_read_file_numbers_lines_and_caps_them():
    h = handler()
    lines = h.run("read_file", {"path": "files/src/agent.py"}).splitlines()
    assert lines[0] == "files/src/agent.py lines 1-200 of 500 (origin: cell 2 > %%writefile src/agent.py)"
    assert lines[1] == "  1| line 1" and lines[200] == "200| line 200"
    assert lines[-1] == "[300 more lines; continue with start_line=201]"
    out = h.run("read_file", {"path": "files/src/agent.py", "start_line": 50, "end_line": 1000})
    assert out.startswith("files/src/agent.py lines 50-449 of 500")  # at most 400 lines
    two = h.run("read_file", {"path": "./cells/cell_000.py", "start_line": "2", "end_line": 2.0}).splitlines()
    assert two[1:] == ["2| TEMPERATURE = 0.2", "[3 more lines; continue with start_line=3]"]
    md = h.run("read_file", {"path": "markdown/cell_001.md"})
    assert "markdown: a claim, not evidence" in md.splitlines()[0]


def test_tool_problems_are_error_strings_never_exceptions():
    h = handler()
    outs = [
        h.run("read_file", {"path": "cells/nope.py"}),
        h.run("read_file", {"path": "cells/cell_00.py"}),
        h.run("read_file", {"path": "payloads/cell_002/0.bin"}),
        h.run("read_file", {}),
        h.run("read_file", {"path": "cells/cell_000.py", "start_line": 99}),
        h.run("read_file", {"path": "cells/cell_000.py", "start_line": "first"}),
        h.run("rm_rf", {"path": "/"}),
        h.run(None, None),
        h.run("grep", "{not json"),
        h.run("grep", {"text": "  "}),
        h.run("grep", {"text": "two\nlines"}),
        h.run("list_files", ["glob"]),
    ]
    assert all(o.startswith("error: ") for o in outs), outs
    assert outs[0].startswith("error: unknown path 'cells/nope.py'")
    assert "did you mean cells/cell_000.py" in outs[1]
    assert "binary" in outs[2] and "path is required" in outs[3] and "only 5 lines" in outs[4]
    assert outs[6].startswith("error: unknown tool 'rm_rf'")
    assert h.calls == len(outs) and h.ok_calls == 0


def test_outputs_are_cut_to_max_tool_chars():
    h = handler(max_chars=300)
    out = h.run("read_file", {"path": "files/src/agent.py"})
    assert len(out) == 300 and out.endswith("\n" + TRUNCATED)
    assert len(h.run("list_files", {})) <= 300


def test_tool_outputs_never_contain_a_planted_secret():
    secret = fake_secret("hf" + "_", 34, string.ascii_letters)
    password = fake_secret("", 18)
    t = make_tree()
    t.add_text(
        "cells/cell_003.py", f'HF = "{secret}"\nlogin(token=HF)\nDB_PASSWORD = "{password}"\n', "cell 3"
    )
    t.add_text("files/.env", f"HF_TOKEN={secret}\n", "cell 3 > %%writefile .env")
    h = handler(t)
    outs = [
        h.run("list_files", {}),
        h.run("grep", {"text": "HF"}),
        h.run("grep", {"text": "PASSWORD"}),
        h.run("read_file", {"path": "cells/cell_003.py"}),
        h.run("read_file", {"path": "files/.env"}),
    ]
    for out in outs:
        assert secret not in out and password not in out
    assert outs[3].count(REDACTED) == 2 and outs[4].endswith(f"HF_TOKEN={REDACTED}")


def test_tool_specs_fit_both_apis():
    names = [s["name"] for s in TOOL_SPECS]
    assert names == ["list_files", "grep", "read_file", "submit_verdict"]
    oa = openai_tools()
    assert [t["function"]["name"] for t in oa] == names
    assert all(t["type"] == "function" and t["function"]["parameters"]["type"] == "object" for t in oa)
    an = anthropic_tools()
    assert [t["name"] for t in an] == names and all(t["input_schema"]["type"] == "object" for t in an)
    verdict = TOOL_SPECS[-1]["parameters"]
    assert verdict["properties"]["status"]["enum"] == ["present", "partial", "absent", "unknown"]
    assert verdict["required"] == ["status", "evidence", "reasoning"]


def test_verdict_arguments_and_json_replies():
    v, problem = parse_verdict(
        '{"status": "Present", "evidence": "[{\\"path\\": \\"a\\", \\"quote\\": \\"b\\"}]", "reasoning": "r"}'
    )
    assert problem is None and v == {
        "status": "present",
        "evidence": [{"path": "a", "quote": "b"}],
        "reasoning": "r",
    }
    assert parse_verdict({"status": "maybe"}) == (
        None,
        "status must be one of present, partial, absent, unknown",
    )
    assert parse_verdict("not json")[0] is None
    assert parse_verdict({"status": "absent", "evidence": 5})[0] is None
    v, _ = parse_json_reply(
        '<think>{"status": "absent"}</think>\n```json\n{"status": "partial", "evidence": [], "reasoning": "half"}\n```'
    )
    assert v is not None and v["status"] == "partial"
    v, _ = parse_json_reply('Sure. {"status": "absent", "evidence": [], "reasoning": "none"} Bye {')
    assert v is not None and v["status"] == "absent"
    code_then_verdict = (
        'Code:\n```python\nx = {"a": 1}\n```\nVerdict: {"status": "maybe"} {"status": "present"}'
    )
    v, _ = parse_json_reply(code_then_verdict)
    assert v is not None and v["status"] == "present"
    assert parse_json_reply('{"status": "maybe"}') == (
        None,
        "status must be one of present, partial, absent, unknown",
    )
    assert parse_json_reply("I think it is present.") == (
        None,
        "the reply was not a valid JSON object with a status",
    )


def test_importing_the_judge_loads_no_llm_sdk():
    code = "import sys, pseudorca.llm.judge; print(sorted(m for m in ('openai', 'anthropic') if m in sys.modules))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"
