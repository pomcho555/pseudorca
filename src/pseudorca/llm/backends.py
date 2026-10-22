"""LLM backends: run the judge's tool loop, or one plain prompt, against an endpoint.

OpenAIBackend talks to any OpenAI-compatible chat-completions endpoint (OpenAI, vLLM, Ollama, LM Studio,
llama.cpp, ...), AnthropicBackend to the Anthropic API, and FakeBackend plays a script for tests. SDKs are
imported only when a backend first sends a request, and their errors become LLMError messages that never
contain an API key. A local endpoint is reached directly: never through a proxy, never by a redirect.
"""

from __future__ import annotations

import inspect
import json
import os
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from pseudorca import util
from pseudorca.llm import LLMError, LLMSettings, endpoint_host, is_local
from pseudorca.llm.redact import REDACTED, redact
from pseudorca.llm.tools import ToolHandler, anthropic_tools, openai_tools, parse_verdict

MAX_TOKENS = 2048  # per Anthropic response
NUDGE = (
    "Use the tools to look at the notebook's code, then call submit_verdict exactly once with your verdict."
)
FINAL = "Call submit_verdict now, quoting exactly what you have read."
BUDGET_SPENT = "Your tool budget is used up. "
STEP_LIMIT = "error: the tool budget is used up; call submit_verdict now"
_TOOL_WORDS = re.compile(r"tool|function[ _-]?call", re.I)
_MASKED_KEY = re.compile(r"\b(?:sk|hf|gh[pousr])[-_][A-Za-z0-9_-]*\*{3,}[A-Za-z0-9_*-]*")

# (call id, tool name, arguments as a dict or JSON string)
Call = tuple[str, str, Any]


class ToolsUnsupported(Exception):
    """The endpoint rejected tool calling; the judge falls back to a single prompt."""


@dataclass
class LoopResult:
    """What a backend returns: the submit_verdict arguments (None if none came) and run statistics."""

    verdict: dict[str, Any] | None
    mode: str = "tools"  # "tools" or "single"
    steps: int = 0  # tool calls run
    usage: dict[str, int] = field(default_factory=dict)  # input_tokens, output_tokens, when reported
    messages: list[dict[str, Any]] = field(default_factory=list)  # the transcript
    note: str = ""  # what went wrong, if anything


@dataclass
class Conversation:
    system: str
    tools: list[dict[str, Any]]
    messages: list[dict[str, Any]] = field(default_factory=list)  # without the system prompt
    usage: dict[str, int] = field(default_factory=dict)

    def add_usage(self, **counts: Any) -> None:
        for key, value in counts.items():
            if isinstance(value, int) and not isinstance(value, bool):
                self.usage[key] = self.usage.get(key, 0) + value

    def transcript(self) -> list[dict[str, Any]]:
        return [{"role": "system", "content": self.system}, *self.messages]


class Backend:
    """Runs the tool loop: ask, run the tools the model asks for, repeat until it calls submit_verdict.

    After one reply without a tool call the model is nudged once; when `max_steps` tool calls are spent (or
    it still does not use the tools) the next request forces submit_verdict. Subclasses implement `_turn`
    (send one request, append the assistant message, return its text and tool calls), `_say` (add user
    text), `_answer` (add tool results) and `complete` (one plain prompt, for the single-shot fallback).
    """

    def run_tools(
        self,
        system: str,
        user: str,
        tools: Sequence[dict[str, Any]],
        handler: ToolHandler,
        settings: LLMSettings,
    ) -> LoopResult:
        """Raises ToolsUnsupported when the endpoint has no tool calling, LLMError when a request fails."""
        convo = Conversation(system, list(tools))
        self._say(convo, user)
        steps, nudged, force, forced = 0, False, False, 0
        for _ in range(settings.max_steps + 4):
            if steps >= settings.max_steps:
                force = True
            if force:
                if forced == 2:
                    break
                if forced == 0:
                    self._say(convo, (BUDGET_SPENT if steps >= settings.max_steps else "") + FINAL)
                forced += 1
            text, calls = self._turn(convo, force, settings)
            if not calls:
                if force:  # last resort: a verdict written as JSON text
                    verdict, _ = parse_json_reply(text)
                    if verdict is not None:
                        return LoopResult(verdict, "tools", steps, convo.usage, convo.transcript())
                    break
                if nudged:
                    force = True
                else:
                    nudged = True
                    self._say(convo, NUDGE)
                continue
            answers: list[tuple[str, str]] = []
            for call_id, name, arguments in calls:
                if name == "submit_verdict":
                    verdict, problem = parse_verdict(arguments)
                    if verdict is not None:
                        return LoopResult(verdict, "tools", steps, convo.usage, convo.transcript())
                    out = f"error: {problem}; call submit_verdict again"
                elif steps >= settings.max_steps:
                    out = STEP_LIMIT
                else:
                    out = handler.run(name, arguments)
                    steps += 1
                answers.append((call_id, out))
            self._answer(convo, answers)
        return LoopResult(
            None, "tools", steps, convo.usage, convo.transcript(), "the LLM did not submit a verdict"
        )

    def complete(self, system: str, user: str, settings: LLMSettings) -> tuple[str, dict[str, int]]:
        """One plain prompt without tools: (reply text, token usage)."""
        raise NotImplementedError

    def _turn(self, convo: Conversation, force: bool, settings: LLMSettings) -> tuple[str, list[Call]]:
        raise NotImplementedError

    def _say(self, convo: Conversation, text: str) -> None:
        raise NotImplementedError

    def _answer(self, convo: Conversation, answers: list[tuple[str, str]]) -> None:
        raise NotImplementedError


# ----- SDK backends --------------------------------------------------------------------------------


def _api_key(settings: LLMSettings, default_env: str) -> str | None:
    """The key from the variable named by api_key_env (else `default_env`); "EMPTY" for a local endpoint."""
    name = settings.api_key_env or default_env
    value = os.environ.get(name, "").strip()
    if value:
        return value
    if settings.api_key_env:
        raise LLMError(f"the environment variable {name} (api_key_env under [llm]) is not set")
    return "EMPTY" if is_local(settings) else None


def _error_text(e: BaseException) -> str:
    body = getattr(e, "body", None)
    if body is None:
        return str(e)
    try:
        return f"{e} {json.dumps(body, default=str)}"
    except (TypeError, ValueError, RecursionError):  # a body nested deeper than the recursion limit
        return str(e)


def _rejects(e: BaseException, word: str) -> bool:
    """A 400/422 response that complains about a request parameter."""
    return getattr(e, "status_code", None) in (400, 422) and word in _error_text(e).lower()


def _rejects_tools(e: BaseException) -> bool:
    """An error response that says the endpoint does not do tools, tool_choice or function calling."""
    status = getattr(e, "status_code", None)
    return (
        isinstance(status, int)
        and status not in (401, 403, 408, 429)
        and bool(_TOOL_WORDS.search(_error_text(e)))
    )


def _accepts(fn: Callable[..., Any], name: str) -> bool:
    """Whether `fn` takes a keyword argument `name` (newer Anthropic SDKs dropped `temperature`)."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return name in params or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def describe_error(e: BaseException, settings: LLMSettings, key: str | None = None) -> str:
    """A one-line description of an SDK or network error, without the API key."""
    text = str(e) or type(e).__name__
    if key and key != "EMPTY":
        text = text.replace(key, REDACTED)
    text = " ".join(_MASKED_KEY.sub(REDACTED, redact(text)).split())
    if len(text) > 300:
        text = text[:299] + "…"
    status = getattr(e, "status_code", None)
    code = f" (HTTP {status})" if isinstance(status, int) else ""
    return f"{settings.label} at {endpoint_host(settings)} failed{code}: {type(e).__name__}: {text}"


def _direct_http_client(openai: Any, base_url: str | None) -> Any:
    """The openai SDK's HTTP client for a local endpoint: no proxy from the environment, no redirects.

    By default httpx sends requests through HTTP_PROXY, HTTPS_PROXY or ALL_PROXY (or the system proxy on
    macOS and Windows) unless NO_PROXY names the host, so a "local" request, with the notebook's code, would
    reach the proxy's host. trust_env=False stops that. A redirect could lead to another host as well, so
    it is not followed: the request fails with its HTTP status. trust_env=False also drops SSL_CERT_FILE
    and SSL_CERT_DIR, which are applied here instead, for https endpoints with their own certificates.
    """
    options: dict[str, Any] = {"trust_env": False, "follow_redirects": False}
    cafile = os.environ.get("SSL_CERT_FILE") or None
    capath = os.environ.get("SSL_CERT_DIR") or None
    if (base_url or "").lower().startswith("https:") and (cafile or capath):
        import ssl

        options["verify"] = ssl.create_default_context(cafile=cafile, capath=None if cafile else capath)
    return openai.DefaultHttpxClient(**options)


class _SDKBackend(Backend):
    sdk = ""  # the package to import
    key_env = ""  # the default environment variable for the key

    def __init__(self, settings: LLMSettings) -> None:
        self.settings = settings
        self._client: Any = None
        self._key: str | None = None
        self._temperature = True  # cleared when the endpoint rejects the parameter

    def client(self) -> Any:
        if self._client is None:
            self._key = self._find_key()
            try:
                self._client = self._connect(self._key)
            except ImportError:
                raise LLMError(
                    f"the {self.sdk} package is not installed; {util.install_hint('llm')}"
                ) from None
            except Exception as e:
                raise LLMError(describe_error(e, self.settings, self._key)) from None
        return self._client

    def _find_key(self) -> str | None:
        return _api_key(self.settings, self.key_env)

    def _connect(self, key: str | None) -> Any:
        raise NotImplementedError

    def _request(self, create: Callable[..., Any], **kwargs: Any) -> Any:
        """Call the SDK; errors become LLMError, or ToolsUnsupported when the endpoint refuses tools."""
        if self._temperature and _accepts(create, "temperature"):
            kwargs["temperature"] = self.settings.temperature
        try:
            try:
                return create(**kwargs)
            except Exception as e:
                if "temperature" not in kwargs or not _rejects(e, "temperature"):
                    raise
                self._temperature = False  # e.g. reasoning models that only take the default
                kwargs.pop("temperature")
                return create(**kwargs)
        except Exception as e:
            if "tools" in kwargs and _rejects_tools(e):
                raise ToolsUnsupported(describe_error(e, self.settings, self._key)) from None
            raise LLMError(describe_error(e, self.settings, self._key)) from None


class OpenAIBackend(_SDKBackend):
    """Any OpenAI-compatible chat-completions endpoint."""

    sdk, key_env = "openai", "OPENAI_API_KEY"

    def _find_key(self) -> str | None:
        key = _api_key(self.settings, self.key_env)
        if key is None:
            raise LLMError(
                f"set the environment variable {self.key_env} to the API key for {endpoint_host(self.settings)}, "
                "or name another variable with api_key_env under [llm]"
            )
        return key

    def _connect(self, key: str | None) -> Any:
        import openai

        if not is_local(self.settings):
            return openai.OpenAI(base_url=self.settings.base_url, api_key=key)
        http_client = _direct_http_client(openai, self.settings.base_url)
        return openai.OpenAI(base_url=self.settings.base_url, api_key=key, http_client=http_client)

    def _create(self, **kwargs: Any) -> Any:
        resp = self._request(self.client().chat.completions.create, model=self.settings.model, **kwargs)
        if not getattr(resp, "choices", None):
            raise LLMError(f"{self.settings.label} returned no choices")
        return resp

    @staticmethod
    def _usage(resp: Any) -> dict[str, Any]:
        usage = getattr(resp, "usage", None)
        return {
            "input_tokens": getattr(usage, "prompt_tokens", None),
            "output_tokens": getattr(usage, "completion_tokens", None),
        }

    def _turn(self, convo: Conversation, force: bool, settings: LLMSettings) -> tuple[str, list[Call]]:
        choice = {"type": "function", "function": {"name": "submit_verdict"}} if force else "auto"
        resp = self._create(messages=convo.transcript(), tools=openai_tools(convo.tools), tool_choice=choice)
        convo.add_usage(**self._usage(resp))
        message = resp.choices[0].message
        text = getattr(message, "content", None) or ""
        calls: list[Call] = []
        for i, call in enumerate(getattr(message, "tool_calls", None) or []):
            fn = getattr(call, "function", None)
            call_id = getattr(call, "id", None) or f"call_{len(convo.messages)}_{i}"
            calls.append((call_id, getattr(fn, "name", "") or "", getattr(fn, "arguments", "") or ""))
        entry: dict[str, Any] = {"role": "assistant", "content": text}
        if calls:
            entry["tool_calls"] = [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": args if isinstance(args, str) else json.dumps(args),
                    },
                }
                for call_id, name, args in calls
            ]
        convo.messages.append(entry)
        return text, calls

    def _say(self, convo: Conversation, text: str) -> None:
        convo.messages.append({"role": "user", "content": text})

    def _answer(self, convo: Conversation, answers: list[tuple[str, str]]) -> None:
        convo.messages.extend(
            {"role": "tool", "tool_call_id": call_id, "content": out} for call_id, out in answers
        )

    def complete(self, system: str, user: str, settings: LLMSettings) -> tuple[str, dict[str, int]]:
        resp = self._create(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}]
        )
        convo = Conversation(system, [])
        convo.add_usage(**self._usage(resp))
        return getattr(resp.choices[0].message, "content", None) or "", convo.usage


class AnthropicBackend(_SDKBackend):
    """The Anthropic Messages API (always remote)."""

    sdk, key_env = "anthropic", "ANTHROPIC_API_KEY"

    def _connect(self, key: str | None) -> Any:
        from anthropic import Anthropic

        return Anthropic(api_key=key, base_url=self.settings.base_url)

    def _create(self, **kwargs: Any) -> Any:
        create = self.client().messages.create
        return self._request(create, model=self.settings.model, max_tokens=MAX_TOKENS, **kwargs)

    @staticmethod
    def _usage(resp: Any) -> dict[str, Any]:
        usage = getattr(resp, "usage", None)
        return {
            "input_tokens": getattr(usage, "input_tokens", None),
            "output_tokens": getattr(usage, "output_tokens", None),
        }

    def _turn(self, convo: Conversation, force: bool, settings: LLMSettings) -> tuple[str, list[Call]]:
        choice = {"type": "tool", "name": "submit_verdict"} if force else {"type": "auto"}
        resp = self._create(
            system=convo.system,
            messages=convo.messages,
            tools=anthropic_tools(convo.tools),
            tool_choice=choice,
        )
        convo.add_usage(**self._usage(resp))
        blocks: list[dict[str, Any]] = []
        texts: list[str] = []
        calls: list[Call] = []
        for block in getattr(resp, "content", None) or []:
            kind = getattr(block, "type", "")
            if kind == "text":
                text = getattr(block, "text", "") or ""
                texts.append(text)
                if text:
                    blocks.append({"type": "text", "text": text})
            elif kind == "tool_use":
                calls.append((block.id, block.name, block.input))
                blocks.append({"type": "tool_use", "id": block.id, "name": block.name, "input": block.input})
            elif hasattr(block, "model_dump"):  # e.g. thinking blocks, which must be sent back unchanged
                blocks.append(block.model_dump(mode="json", exclude_none=True))
        convo.messages.append(
            {"role": "assistant", "content": blocks or [{"type": "text", "text": "(no reply)"}]}
        )
        return "".join(texts), calls

    def _say(self, convo: Conversation, text: str) -> None:
        last = convo.messages[-1] if convo.messages else None
        if last is not None and last["role"] == "user":  # roles must alternate: extend the last user turn
            content = last["content"]
            blocks = [{"type": "text", "text": content}] if isinstance(content, str) else list(content)
            last["content"] = [*blocks, {"type": "text", "text": text}]
        else:
            convo.messages.append({"role": "user", "content": text})

    def _answer(self, convo: Conversation, answers: list[tuple[str, str]]) -> None:
        results = []
        for call_id, out in answers:
            result: dict[str, Any] = {"type": "tool_result", "tool_use_id": call_id, "content": out}
            if out.startswith("error:"):
                result["is_error"] = True
            results.append(result)
        convo.messages.append({"role": "user", "content": results})

    def complete(self, system: str, user: str, settings: LLMSettings) -> tuple[str, dict[str, int]]:
        resp = self._create(system=system, messages=[{"role": "user", "content": user}])
        convo = Conversation(system, [])
        convo.add_usage(**self._usage(resp))
        blocks = getattr(resp, "content", None) or []
        text = "".join(getattr(b, "text", "") or "" for b in blocks if getattr(b, "type", "") == "text")
        return text, convo.usage


# ----- fake backend for tests ----------------------------------------------------------------------


class FakeBackend(Backend):
    """A scripted model for tests. Its tool calls run through the real handler, so the tools are exercised.

    Each script item is one model turn: a (tool, arguments) pair, a list of pairs (several calls in one
    turn) or a str (a text reply without tool calls). When the script ends, or the loop forces the final
    call, it calls submit_verdict with `verdict`; with verdict=None it never submits. With tools=False it
    acts like an endpoint without tool calling, and `complete` returns `reply`.
    """

    def __init__(
        self,
        script: Sequence[tuple[str, Any] | list[tuple[str, Any]] | str] = (),
        verdict: dict[str, Any] | None = None,
        *,
        reply: str | None = None,
        tools: bool = True,
    ) -> None:
        self.script = list(script)
        self.verdict = verdict
        self.reply = reply
        self.tools = tools
        self.prompts: list[tuple[str, str]] = []  # (system, user) it was sent
        self.outputs: list[str] = []  # tool results it was given
        self.forced = 0  # turns in which submit_verdict was forced
        self._pos = 0

    def run_tools(
        self,
        system: str,
        user: str,
        tools: Sequence[dict[str, Any]],
        handler: ToolHandler,
        settings: LLMSettings,
    ) -> LoopResult:
        if not self.tools:
            raise ToolsUnsupported("this fake endpoint has no tool calling")
        self.prompts.append((system, user))
        return super().run_tools(system, user, tools, handler, settings)

    def complete(self, system: str, user: str, settings: LLMSettings) -> tuple[str, dict[str, int]]:
        self.prompts.append((system, user))
        default = (
            '{"status": "unknown", "evidence": [], "reasoning": "the fake provider has no scripted reply"}'
        )
        return (default if self.reply is None else self.reply), {}

    def _turn(self, convo: Conversation, force: bool, settings: LLMSettings) -> tuple[str, list[Call]]:
        if force:
            self.forced += 1
        if force or self._pos >= len(self.script):
            item: Any = ("submit_verdict", self.verdict) if self.verdict is not None else "I cannot tell."
        else:
            item = self.script[self._pos]
            self._pos += 1
        if isinstance(item, str):
            convo.messages.append({"role": "assistant", "content": item})
            return item, []
        pairs = [item] if isinstance(item, tuple) else list(item)
        calls = [(f"fake_{len(convo.messages)}_{i}", name, args) for i, (name, args) in enumerate(pairs)]
        convo.messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": c, "name": n, "arguments": a} for c, n, a in calls],
            }
        )
        return "", calls

    def _say(self, convo: Conversation, text: str) -> None:
        convo.messages.append({"role": "user", "content": text})

    def _answer(self, convo: Conversation, answers: list[tuple[str, str]]) -> None:
        for call_id, out in answers:
            self.outputs.append(out)
            convo.messages.append({"role": "tool", "tool_call_id": call_id, "content": out})


FAKE_VERDICT = {"status": "unknown", "evidence": [], "reasoning": "the fake provider has no scripted answer"}


def make_backend(settings: LLMSettings) -> Backend:
    """The backend for `settings.provider`. Nothing is imported or sent until the first request."""
    if settings.provider == "openai":
        return OpenAIBackend(settings)
    if settings.provider == "anthropic":
        return AnthropicBackend(settings)
    if settings.provider == "fake":
        return FakeBackend(verdict=dict(FAKE_VERDICT))
    raise LLMError(f"unknown LLM provider {settings.provider!r}")


# ----- replies written as text ---------------------------------------------------------------------

_THINK_TAG = re.compile(r"<(/?)think>", re.I)
# the language tag and the spaces after it are possessive: giving them back cannot make a fence close, and
# retrying would scan the rest of the reply once for every character of an unclosed fence's tag
_FENCE = re.compile(r"```[A-Za-z]*+[ \t]*+\n?(.*?)```", re.S)


def _without_think(text: str) -> str:
    """`text` without its <think>...</think> blocks, each from a <think> to the next </think>.

    A block that never closes stays, and so does a </think> without a start. One pass over the tags, so a
    reply full of unclosed tags takes linear time (a lazy <think>.*?</think> scans to the end from each).
    """
    parts: list[str] = []
    pos = 0
    start: int | None = None
    for tag in _THINK_TAG.finditer(text):
        if not tag.group(1):
            if start is None:
                start = tag.start()
        elif start is not None:
            parts.append(text[pos:start])
            pos, start = tag.end(), None
    parts.append(text[pos:])
    return "".join(parts)


def parse_json_reply(text: str) -> tuple[dict[str, Any] | None, str | None]:
    """A {status, evidence, reasoning} verdict from a text reply: code fences first, then the whole reply.

    <think> blocks are skipped. Returns (None, what is wrong) when no valid verdict object is found, also
    when the reply's JSON is nested deeper than Python's recursion limit.
    """
    body = _without_think(text or "")
    problem = "the reply was not a valid JSON object with a status"
    decoder = json.JSONDecoder()
    for part in [*_FENCE.findall(body), body]:
        for tries, m in enumerate(re.finditer(r"\{", part)):
            if tries == 20:
                break
            try:
                value, _ = decoder.raw_decode(part, m.start())
            except (ValueError, RecursionError):
                continue
            if isinstance(value, dict) and "status" in value:
                verdict, why = parse_verdict(value)
                if verdict is not None:
                    return verdict, None
                problem = why or problem
    return None, problem
