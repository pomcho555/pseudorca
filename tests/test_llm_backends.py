"""The OpenAI-compatible and Anthropic backends against fake SDK modules (no network, no real SDK calls).

The exception is the proxy test, which drives the real openai SDK with every connection attempt intercepted
and refused before it leaves the process.
"""

from __future__ import annotations

import copy
import json
import random
import socket
import ssl
import string
import sys
import types
from dataclasses import replace
from types import SimpleNamespace as NS
from typing import Any

import pytest

from pseudorca.llm import LLMError, LLMSettings, judge_idea
from pseudorca.llm.backends import STEP_LIMIT
from pseudorca.llm.tools import TOOL_SPECS
from pseudorca.model import Idea
from pseudorca.tree import Tree

CODE = "def make_params():\n    return dict(\n        temperature=0.2,\n        top_p=0.95,\n    )\n"
IDEA = Idea(
    id="low-temp",
    title="Lower temperature",
    description="Use temperature 0.2.",
    fingerprints=["temperature=0.2"],
)
VERDICT = {
    "status": "present",
    "evidence": [{"path": "cells/cell_000.py", "line": 3, "quote": "temperature=0.2,"}],
    "reasoning": "make_params sets it.",
}
JSON_VERDICT = json.dumps(VERDICT)
NAMES = [s["name"] for s in TOOL_SPECS]
LOCAL = LLMSettings("openai", "local-model", base_url="http://localhost:8000/v1")
ANTHROPIC = LLMSettings("anthropic", "claude-model", allow_remote=True)
FINAL = "Your tool budget is used up. Call submit_verdict now, quoting exactly what you have read."


@pytest.fixture(autouse=True)
def no_real_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "OPENAI_BASE_URL",
        "ANTHROPIC_BASE_URL",
        "MY_LLM_KEY",
    ):
        monkeypatch.delenv(name, raising=False)


def tree() -> Tree:
    t = Tree("nb")
    t.add_text("cells/cell_000.py", CODE, "cell 0")
    return t


def fake_key() -> str:
    rng = random.Random(7)
    return "sk-" + "".join(rng.choice(string.ascii_letters + string.digits) for _ in range(48))


class APIError(Exception):
    """Shaped like the SDKs' APIStatusError: a message, an HTTP status and a parsed body."""

    def __init__(self, message: str, status_code: int = 400, body: Any = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class Recorder:
    """Plays canned replies (or raises canned errors) and records every request and client."""

    def __init__(self, replies: list[Any]) -> None:
        self.replies = list(replies)
        self.requests: list[dict[str, Any]] = []
        self.clients: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.requests.append(copy.deepcopy(kwargs))
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply


# ----- OpenAI ----------------------------------------------------------------------------------------


def install_openai(monkeypatch: pytest.MonkeyPatch, replies: list[Any]) -> Recorder:
    rec = Recorder(replies)
    module = types.ModuleType("openai")

    class DefaultHttpxClient:
        def __init__(self, **options: Any) -> None:
            self.options = options

    class OpenAI:
        def __init__(
            self,
            *,
            base_url: str | None = None,
            api_key: str | None = None,
            http_client: DefaultHttpxClient | None = None,
            **kwargs: Any,
        ) -> None:
            options = None if http_client is None else http_client.options
            rec.clients.append({"base_url": base_url, "api_key": api_key, "http_client": options})
            self.chat = NS(completions=NS(create=rec.create))

    module.OpenAI = OpenAI  # type: ignore[attr-defined]
    module.DefaultHttpxClient = DefaultHttpxClient  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", module)
    return rec


DIRECT = {"trust_env": False, "follow_redirects": False}  # the HTTP client options for a local endpoint


def oa_calls(*calls: tuple[str, dict[str, Any]], text: str = "") -> Any:
    tool_calls = [
        NS(id=f"call{i}", type="function", function=NS(name=name, arguments=json.dumps(args)))
        for i, (name, args) in enumerate(calls)
    ]
    message = NS(content=text, tool_calls=tool_calls)
    return NS(choices=[NS(message=message)], usage=NS(prompt_tokens=100, completion_tokens=20))


def oa_text(text: str) -> Any:
    message = NS(content=text, tool_calls=None)
    return NS(choices=[NS(message=message)], usage=NS(prompt_tokens=50, completion_tokens=5))


def test_openai_tool_loop(monkeypatch: pytest.MonkeyPatch):
    rec = install_openai(
        monkeypatch, [oa_calls(("grep", {"text": "temperature"})), oa_calls(("submit_verdict", VERDICT))]
    )
    f = judge_idea(IDEA, None, tree(), [], LOCAL)
    assert f.status == "present" and f.evidence[0].line == 3 and f.model == "openai:local-model"
    assert f.detail["steps"] == 1 and f.detail["usage"] == {"input_tokens": 200, "output_tokens": 40}
    assert rec.clients == [
        {"base_url": "http://localhost:8000/v1", "api_key": "EMPTY", "http_client": DIRECT}
    ]
    first, second = rec.requests
    assert first["model"] == "local-model" and first["tool_choice"] == "auto" and first["temperature"] == 0.0
    assert [t["function"]["name"] for t in first["tools"]] == NAMES
    assert [m["role"] for m in first["messages"]] == ["system", "user"]
    assert [m["role"] for m in second["messages"]] == ["system", "user", "assistant", "tool"]
    call = second["messages"][2]["tool_calls"][0]
    assert call["id"] == "call0" and call["function"] == {
        "name": "grep",
        "arguments": '{"text": "temperature"}',
    }
    result = second["messages"][3]
    assert result["tool_call_id"] == "call0" and "cells/cell_000.py:3:" in result["content"]


def test_openai_forces_the_verdict_after_max_steps(monkeypatch: pytest.MonkeyPatch):
    settings = LLMSettings("openai", "m", base_url="http://127.0.0.1:8000/v1", max_steps=2)
    rec = install_openai(
        monkeypatch,
        [
            oa_calls(("grep", {"text": "temperature"})),
            oa_calls(("read_file", {"path": "cells/cell_000.py"}), ("grep", {"text": "top_p"})),
            oa_calls(("submit_verdict", VERDICT)),
        ],
    )
    f = judge_idea(IDEA, None, tree(), [], settings)
    assert f.status == "present" and f.detail["steps"] == 2
    assert [r["tool_choice"] for r in rec.requests[:2]] == ["auto", "auto"]
    last = rec.requests[-1]
    assert last["tool_choice"] == {"type": "function", "function": {"name": "submit_verdict"}}
    assert last["messages"][-1] == {"role": "user", "content": FINAL}
    assert [m["content"] for m in last["messages"] if m["role"] == "tool"][-1] == STEP_LIMIT


def test_openai_nudges_once_when_the_model_does_not_call_tools(monkeypatch: pytest.MonkeyPatch):
    rec = install_openai(
        monkeypatch,
        [
            oa_text("I believe it is present."),
            oa_calls(("grep", {"text": "temp"})),
            oa_calls(("submit_verdict", VERDICT)),
        ],
    )
    assert judge_idea(IDEA, None, tree(), [], LOCAL).status == "present"
    messages = rec.requests[1]["messages"]
    assert messages[-2] == {"role": "assistant", "content": "I believe it is present."}
    assert messages[-1]["role"] == "user" and "submit_verdict" in messages[-1]["content"]


def test_openai_takes_a_json_verdict_written_as_text_in_the_forced_turn(monkeypatch: pytest.MonkeyPatch):
    rec = install_openai(monkeypatch, [oa_text("Hmm."), oa_text("Let me think."), oa_text(JSON_VERDICT)])
    f = judge_idea(IDEA, None, tree(), [], LOCAL)
    assert rec.requests[2]["tool_choice"]["function"]["name"] == "submit_verdict"
    assert (
        rec.requests[2]["messages"][-1]["content"]
        == "Call submit_verdict now, quoting exactly what you have read."
    )
    assert f.status == "present" and f.detail["steps"] == 0


def test_openai_falls_back_to_a_single_prompt_without_tool_support(monkeypatch: pytest.MonkeyPatch):
    refusal = APIError(
        '"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set'
    )
    rec = install_openai(monkeypatch, [refusal, oa_text(f"```json\n{JSON_VERDICT}\n```")])
    f = judge_idea(IDEA, None, tree(), [], LOCAL)
    assert f.status == "present" and f.detail["mode"] == "single" and f.evidence[0].line == 3
    single = rec.requests[1]
    assert "tools" not in single and "tool_choice" not in single and single["temperature"] == 0.0
    assert "<snippets>" in single["messages"][1]["content"]


def test_openai_retries_without_temperature_when_the_model_rejects_it(monkeypatch: pytest.MonkeyPatch):
    rejected = APIError("Unsupported value: 'temperature' does not support 0.0 with this model.")
    rec = install_openai(
        monkeypatch, [rejected, oa_calls(("grep", {"text": "t"})), oa_calls(("submit_verdict", VERDICT))]
    )
    assert judge_idea(IDEA, None, tree(), [], LOCAL).status == "present"
    assert "temperature" in rec.requests[0] and all("temperature" not in r for r in rec.requests[1:])


def test_openai_errors_never_show_the_key(monkeypatch: pytest.MonkeyPatch):
    key = fake_key()
    monkeypatch.setenv("MY_LLM_KEY", key)
    settings = LLMSettings("openai", "gpt-model", api_key_env="MY_LLM_KEY", allow_remote=True)
    masked = key[:8] + "*" * 30 + key[-4:]
    rec = install_openai(monkeypatch, [APIError(f"Incorrect API key provided: {key} ({masked})", 401)])
    with pytest.raises(LLMError) as e:
        judge_idea(IDEA, None, tree(), [], settings)
    message = str(e.value)
    assert key not in message and masked not in message and key[-4:] + ")" not in message
    assert "HTTP 401" in message and "api.openai.com" in message and "\n" not in message
    assert rec.clients == [{"base_url": None, "api_key": key, "http_client": None}]


def test_openai_network_errors_become_llm_errors(monkeypatch: pytest.MonkeyPatch):
    install_openai(monkeypatch, [ConnectionError("Connection refused")])
    with pytest.raises(LLMError) as e:
        judge_idea(IDEA, None, tree(), [], LOCAL)
    assert "localhost" in str(e.value) and "Connection refused" in str(e.value)


def test_an_error_body_nested_too_deeply_is_still_an_llm_error(monkeypatch: pytest.MonkeyPatch):
    body: Any = {}
    for _ in range(100_000):
        body = {"error": body}
    install_openai(monkeypatch, [APIError("internal server error", 500, body)])
    with pytest.raises(LLMError) as e:
        judge_idea(IDEA, None, tree(), [], LOCAL)
    assert "HTTP 500" in str(e.value) and "internal server error" in str(e.value)


def test_openai_keys(monkeypatch: pytest.MonkeyPatch):
    install_openai(monkeypatch, [])
    with pytest.raises(LLMError) as e:
        judge_idea(IDEA, None, tree(), [], LLMSettings("openai", "m", allow_remote=True))
    assert "OPENAI_API_KEY" in str(e.value)
    with pytest.raises(LLMError) as e:
        judge_idea(
            IDEA,
            None,
            tree(),
            [],
            LLMSettings("openai", "m", base_url=LOCAL.base_url, api_key_env="MY_LLM_KEY"),
        )
    assert "MY_LLM_KEY" in str(e.value) and "not set" in str(e.value)


def test_a_missing_sdk_says_how_to_add_the_extra(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(sys.modules, "openai", None)
    with pytest.raises(LLMError) as e:
        judge_idea(IDEA, None, tree(), [], LOCAL)
    message = str(e.value)
    # both ways to install from git; a bare "pseudorca[llm]" would be looked up on PyPI
    assert 'pip install "pseudorca[llm] @ git+https://github.com/pomcho555/pseudorca"' in message
    assert (
        'uv tool install --reinstall "pseudorca[llm] @ git+https://github.com/pomcho555/pseudorca"' in message
    )


# ----- local endpoints are reached directly ----------------------------------------------------------

PROXY_VARIABLES = [f"{scheme}_proxy" for scheme in ("http", "https", "all", "no")]
PROXY_VARIABLES += [name.upper() for name in PROXY_VARIABLES]


class Dialed(BaseException):
    """Raised in place of opening a connection. A BaseException, so neither httpx nor the SDK retries it."""


def refuse_connections(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Any, ...]]:
    """Record the address of every connection attempt and stop it there: nothing reaches the network."""
    dialed: list[tuple[Any, ...]] = []

    def dial(address: Any, *args: Any, **kwargs: Any) -> Any:
        dialed.append(tuple(address)[:2])
        raise Dialed

    monkeypatch.setattr(socket, "create_connection", dial)
    monkeypatch.setattr(socket.socket, "connect", lambda self, address: dial(address))
    return dialed


@pytest.mark.parametrize("variable", ["HTTP_PROXY", "ALL_PROXY", "http_proxy"])
def test_a_local_endpoint_is_reached_directly_despite_a_proxy_in_the_environment(
    monkeypatch: pytest.MonkeyPatch, variable: str
):
    pytest.importorskip("openai")  # the real SDK and its HTTP client
    for name in PROXY_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(variable, "http://192.0.2.1:3128")  # TEST-NET-1: an address that is never in use
    dialed = refuse_connections(monkeypatch)
    with pytest.raises(Dialed):
        judge_idea(IDEA, None, tree(), [], LOCAL)
    assert dialed == [("localhost", 8000)]  # not the proxy, which would receive the notebook's code
    # an allowed remote endpoint still goes through the proxy, which a company network may require
    monkeypatch.setenv("OPENAI_API_KEY", "test-value-not-a-key")
    remote = LLMSettings("openai", "m", base_url="http://192.0.2.7:8000/v1", allow_remote=True)
    with pytest.raises(Dialed):
        judge_idea(IDEA, None, tree(), [], remote)
    assert dialed[1:] == [("192.0.2.1", 3128)]


def test_only_a_local_endpoint_gets_a_direct_client_without_redirects(monkeypatch: pytest.MonkeyPatch):
    rec = install_openai(monkeypatch, [oa_calls(("submit_verdict", VERDICT))] * 3)
    assert judge_idea(IDEA, None, tree(), [], LOCAL).status == "present"
    assert judge_idea(IDEA, None, tree(), [], replace(LOCAL, allow_remote=True)).status == "present"
    monkeypatch.setenv("OPENAI_API_KEY", "test-value-not-a-key")
    remote = LLMSettings("openai", "m", base_url="https://llm.example.com/v1", allow_remote=True)
    assert judge_idea(IDEA, None, tree(), [], remote).status == "present"
    assert [c["http_client"] for c in rec.clients] == [DIRECT, DIRECT, None]


def test_a_local_https_endpoint_keeps_the_certificate_settings(monkeypatch: pytest.MonkeyPatch):
    certifi = pytest.importorskip("certifi")  # any file of PEM certificates will do
    monkeypatch.setenv("SSL_CERT_FILE", certifi.where())
    rec = install_openai(monkeypatch, [oa_calls(("submit_verdict", VERDICT))] * 2)
    judge_idea(IDEA, None, tree(), [], replace(LOCAL, base_url="https://localhost:8443/v1"))
    options = rec.clients[0]["http_client"]
    assert options["trust_env"] is False and isinstance(options["verify"], ssl.SSLContext)
    assert options["verify"].cert_store_stats()["x509_ca"] > 0  # the certificates from SSL_CERT_FILE
    judge_idea(IDEA, None, tree(), [], LOCAL)  # plain http has no certificates to check
    assert rec.clients[1]["http_client"] == DIRECT


# ----- Anthropic -------------------------------------------------------------------------------------


def install_anthropic(
    monkeypatch: pytest.MonkeyPatch, replies: list[Any], *, temperature: bool = True
) -> Recorder:
    rec = Recorder(replies)
    module = types.ModuleType("anthropic")

    def create_without_temperature(
        *,
        model: str,
        max_tokens: int,
        messages: Any,
        system: Any = None,
        tools: Any = None,
        tool_choice: Any = None,
    ) -> Any:
        given = {"system": system, "tools": tools, "tool_choice": tool_choice}
        return rec.create(
            model=model, max_tokens=max_tokens, messages=messages, **{k: v for k, v in given.items() if v}
        )

    class Anthropic:
        def __init__(self, *, api_key: str | None = None, base_url: str | None = None, **kwargs: Any) -> None:
            rec.clients.append({"api_key": api_key, "base_url": base_url})
            self.messages = NS(create=rec.create if temperature else create_without_temperature)

    module.Anthropic = Anthropic  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "anthropic", module)
    return rec


def an_reply(*blocks: Any) -> Any:
    return NS(content=list(blocks), usage=NS(input_tokens=100, output_tokens=20), stop_reason="tool_use")


def tool_use(block_id: str, name: str, args: dict[str, Any]) -> Any:
    return NS(type="tool_use", id=block_id, name=name, input=args)


def text_block(text: str) -> Any:
    return NS(type="text", text=text)


def test_anthropic_tool_loop_and_forced_final(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-value-not-a-key")
    settings = LLMSettings("anthropic", "claude-model", allow_remote=True, max_steps=1)
    rec = install_anthropic(
        monkeypatch,
        [
            an_reply(text_block("Let me look."), tool_use("tu1", "grep", {"text": "temperature"})),
            an_reply(tool_use("tu2", "submit_verdict", VERDICT)),
        ],
    )
    f = judge_idea(IDEA, None, tree(), [], settings)
    assert f.status == "present" and f.evidence[0].line == 3 and f.model == "anthropic:claude-model"
    assert f.detail["usage"] == {"input_tokens": 200, "output_tokens": 40}
    assert rec.clients == [{"api_key": "test-value-not-a-key", "base_url": None}]
    first, second = rec.requests
    assert first["model"] == "claude-model" and first["max_tokens"] == 2048 and first["temperature"] == 0.0
    assert first["system"].startswith("You are an auditor") and first["tool_choice"] == {"type": "auto"}
    assert [t["name"] for t in first["tools"]] == NAMES and all("input_schema" in t for t in first["tools"])
    assert [m["role"] for m in first["messages"]] == ["user"]
    assert second["tool_choice"] == {"type": "tool", "name": "submit_verdict"}
    assistant, user = second["messages"][1], second["messages"][2]
    assert assistant == {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Let me look."},
            {"type": "tool_use", "id": "tu1", "name": "grep", "input": {"text": "temperature"}},
        ],
    }
    result = user["content"][0]
    assert user["role"] == "user" and result["type"] == "tool_result" and result["tool_use_id"] == "tu1"
    assert "cells/cell_000.py:3:" in result["content"] and "is_error" not in result
    assert user["content"][-1] == {"type": "text", "text": FINAL}  # roles alternate: one user turn


def test_anthropic_nudge_and_error_results(monkeypatch: pytest.MonkeyPatch):
    rec = install_anthropic(
        monkeypatch,
        [
            an_reply(text_block("It is there, I think.")),
            an_reply(tool_use("tu1", "read_file", {"path": "nope.py"})),
            an_reply(tool_use("tu2", "submit_verdict", VERDICT)),
        ],
    )
    assert judge_idea(IDEA, None, tree(), [], ANTHROPIC).status == "present"
    nudged = rec.requests[1]["messages"]
    assert [m["role"] for m in nudged] == ["user", "assistant", "user"] and "submit_verdict" in nudged[-1][
        "content"
    ]
    error = rec.requests[2]["messages"][-1]["content"][0]
    assert error["is_error"] is True and error["content"].startswith("error: unknown path")


def test_anthropic_sends_temperature_only_when_the_sdk_takes_it(monkeypatch: pytest.MonkeyPatch):
    unknown = {"status": "unknown", "evidence": [], "reasoning": "?"}
    rec = install_anthropic(
        monkeypatch, [an_reply(tool_use("tu1", "submit_verdict", unknown))], temperature=False
    )
    assert judge_idea(IDEA, None, tree(), [], ANTHROPIC).status == "unknown"
    assert "temperature" not in rec.requests[0] and rec.requests[0]["tool_choice"] == {"type": "auto"}


def test_anthropic_falls_back_without_tool_support(monkeypatch: pytest.MonkeyPatch):
    absent = '{"status": "absent", "evidence": [], "reasoning": "not in make_params"}'
    rec = install_anthropic(
        monkeypatch, [APIError("tools are not supported by this proxy"), an_reply(text_block(absent))]
    )
    f = judge_idea(IDEA, None, tree(), [], ANTHROPIC)
    assert f.status == "absent" and f.detail["mode"] == "single"
    single = rec.requests[1]
    assert "tools" not in single and single["max_tokens"] == 2048 and single["messages"][0]["role"] == "user"


def test_anthropic_errors_become_llm_errors(monkeypatch: pytest.MonkeyPatch):
    install_anthropic(monkeypatch, [APIError("overloaded", 529)])
    with pytest.raises(LLMError) as e:
        judge_idea(IDEA, None, tree(), [], ANTHROPIC)
    assert "HTTP 529" in str(e.value) and "api.anthropic.com" in str(e.value)
    with pytest.raises(LLMError) as e:  # remote and not allowed: refused before any client exists
        judge_idea(IDEA, None, tree(), [], LLMSettings("anthropic", "claude-model"))
    assert "allow_remote" in str(e.value)
