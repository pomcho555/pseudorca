"""The LLM judge for "is this idea in this notebook?".

The judge reads a redacted view of a notebook's tree through read-only tools (list_files, grep, read_file)
and answers with submit_verdict. Every quote it cites is looked up in that same redacted text, and claims
that cannot be verified drop to "unknown". Notebook code leaves the machine only for a local endpoint, or
when `allow_remote` is set.

    redact.py    secret redaction and the redacted view of a tree
    tools.py     tool specs and the handler that runs them
    backends.py  OpenAI-compatible, Anthropic and fake backends; the tool loop
    judge.py     prompts, the single-prompt fallback, verification and transcripts
"""

from __future__ import annotations

import ipaddress
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from pseudorca.audit.types import Finding
    from pseudorca.llm.backends import Backend
    from pseudorca.model import Idea
    from pseudorca.store import Store
    from pseudorca.tree import Tree

PROMPT_VERSION = "1"  # bump when the prompt or tools change, so cached LLM verdicts are redone
PROVIDERS = ("openai", "anthropic", "fake")
DEFAULT_HOSTS = {"openai": "api.openai.com", "anthropic": "api.anthropic.com"}
SETTING_KEYS = (
    "provider",
    "model",
    "base_url",
    "api_key_env",
    "allow_remote",
    "max_steps",
    "max_tool_chars",
    "temperature",
)
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_LOOPBACK_V4 = ipaddress.ip_network("127.0.0.0/8")
_ANY_V4 = ipaddress.ip_address("0.0.0.0")
_LOOPBACK_V6 = ipaddress.ip_address("::1")


class LLMError(Exception):
    """The LLM cannot be used: not configured, not allowed (remote), missing SDK, or the call failed."""


@dataclass(frozen=True)
class LLMSettings:
    provider: str  # "openai" (any OpenAI-compatible endpoint) | "anthropic" | "fake" (tests)
    model: str
    base_url: str | None = None
    api_key_env: str | None = None  # NAME of the environment variable holding the key
    allow_remote: bool = False
    max_steps: int = 12  # tool calls per idea
    max_tool_chars: int = 12000  # characters returned by one tool call
    temperature: float = 0.0

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model}"


def load_settings(store: Store, **overrides: Any) -> LLMSettings | None:
    """[llm] from config.toml with overrides (provider, model, base_url, allow_remote); None if no model is set.

    An override wins unless it is None, so a caller passes None for an option the user did not give (a
    `--allow-remote` flag that is off must arrive as None, or it would hide `allow_remote = true`).
    Raises LLMError for invalid values.
    """
    unknown = sorted(set(overrides) - set(SETTING_KEYS))
    if unknown:
        raise TypeError(f"unknown LLM setting(s): {', '.join(unknown)}")
    raw = {key: store.setting("llm", key) for key in SETTING_KEYS}
    raw.update((key, value) for key, value in overrides.items() if value is not None)
    model = raw["model"]
    if model is None or (isinstance(model, str) and not model.strip()):
        return None
    if not isinstance(model, str):
        raise LLMError("[llm] model must be a string")

    provider = raw["provider"] if raw["provider"] not in (None, "") else "openai"
    provider = provider.strip().lower() if isinstance(provider, str) else provider
    if provider not in PROVIDERS:
        raise LLMError(f"[llm] provider must be one of {', '.join(PROVIDERS)}, not {provider!r}")
    values: dict[str, Any] = {"provider": provider, "model": model.strip()}
    if raw["base_url"] not in (None, ""):
        values["base_url"] = _url(raw["base_url"])
    if raw["api_key_env"] not in (None, ""):
        values["api_key_env"] = _env_name(raw["api_key_env"])
    if raw["allow_remote"] is not None:
        if not isinstance(raw["allow_remote"], bool):
            raise LLMError("[llm] allow_remote must be true or false")
        values["allow_remote"] = raw["allow_remote"]
    for key, minimum in (("max_steps", 1), ("max_tool_chars", 200)):
        value = raw[key]
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise LLMError(f"[llm] {key} must be a whole number of at least {minimum}")
        values[key] = value
    temperature = raw["temperature"]
    if temperature is not None:
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, int | float)
            or not 0 <= temperature <= 2
        ):
            raise LLMError("[llm] temperature must be a number from 0 to 2")
        values["temperature"] = float(temperature)
    return LLMSettings(**values)


def _url(value: Any) -> str:
    if isinstance(value, str):
        url = value.strip()
        try:
            parts = urlsplit(url)
            if parts.scheme in ("http", "https") and parts.hostname:
                return url
        except ValueError:
            pass
    raise LLMError("[llm] base_url must be an http:// or https:// URL, such as http://localhost:8000/v1")


def _env_name(value: Any) -> str:
    # never echo the value: a key pasted here by mistake must not end up in a terminal log
    if isinstance(value, str) and _ENV_NAME.fullmatch(value.strip()):
        return value.strip()
    raise LLMError(
        "[llm] api_key_env must be the NAME of an environment variable (such as OPENAI_API_KEY), "
        "never the key itself"
    )


def is_local_host(host: str | None) -> bool:
    """localhost, *.localhost, 127.0.0.0/8, ::1 and 0.0.0.0 are local; every other host is remote."""
    if not host:
        return False
    name = host.strip().strip("[]").rstrip(".").lower()
    if name == "localhost" or name.endswith(".localhost"):
        return True
    try:
        ip = ipaddress.ip_address(name)
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    if isinstance(ip, ipaddress.IPv4Address):
        return ip in _LOOPBACK_V4 or ip == _ANY_V4
    return ip == _LOOPBACK_V6


def is_local(settings: LLMSettings) -> bool:
    """True when nothing leaves the machine: the fake provider, or an OpenAI-compatible endpoint on a local host."""
    if settings.provider == "fake":
        return True
    if settings.provider != "openai" or not settings.base_url:
        return False  # anthropic is always remote; no base_url means api.openai.com
    try:
        parts = urlsplit(settings.base_url)
        return parts.scheme in ("http", "https") and is_local_host(parts.hostname)
    except ValueError:
        return False


def endpoint_host(settings: LLMSettings) -> str:
    """The host notebook code is sent to (without any user:password part of the URL)."""
    if settings.base_url:
        try:
            host = urlsplit(settings.base_url).hostname
        except ValueError:
            host = None
        return host or "an unparsable base_url"
    return DEFAULT_HOSTS.get(settings.provider, "localhost")


def ensure_allowed(settings: LLMSettings) -> None:
    """Raise LLMError unless the endpoint is local or allow_remote is true. Call before any network use."""
    if settings.provider not in PROVIDERS:
        raise LLMError(f"unknown LLM provider {settings.provider!r} (use one of {', '.join(PROVIDERS)})")
    if settings.allow_remote or is_local(settings):
        return
    host = endpoint_host(settings)
    raise LLMError(
        f"{settings.label} runs at {host}, which is not this machine: notebook code (with secrets redacted) "
        f"would be sent to {host}. If that is acceptable, set allow_remote = true under [llm] in "
        ".pseudorca/config.toml, or pass --allow-remote."
    )


def judge_idea(
    idea: Idea,
    patch_text: str | None,
    tree: Tree,
    findings: list[Finding],
    settings: LLMSettings,
    *,
    cache_dir: Path | None = None,
    backend: Backend | None = None,
) -> Finding:
    """Ask the LLM. Returns a Finding with method="llm" whose evidence quotes were all verified in `tree`.

    Quotes are verified in the redacted view of `tree`, the same text the LLM read. Raises LLMError when the
    endpoint is not allowed or a call fails. `backend` replaces the one chosen by `settings.provider`
    (tests pass a FakeBackend); `cache_dir` receives a redacted transcript.
    """
    ensure_allowed(settings)
    from pseudorca.llm.judge import judge

    return judge(idea, patch_text, tree, findings, settings, cache_dir=cache_dir, backend=backend)
