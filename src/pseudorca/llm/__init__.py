"""The LLM judge for "is this idea in this notebook?".

CONTRACT STUB: the LLM module replaces the bodies below and keeps these names and signatures.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pseudorca.audit.types import Finding
    from pseudorca.model import Idea
    from pseudorca.store import Store
    from pseudorca.tree import Tree

PROMPT_VERSION = "1"  # bump when the prompt or tools change, so cached LLM verdicts are redone


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
    """[llm] from config.toml with overrides (provider, model, base_url, allow_remote); None if no model is set."""
    raise NotImplementedError


def ensure_allowed(settings: LLMSettings) -> None:
    """Raise LLMError unless the endpoint is local or allow_remote is true. Call before any network use."""
    raise NotImplementedError


def judge_idea(
    idea: Idea,
    patch_text: str | None,
    tree: Tree,
    findings: list[Finding],
    settings: LLMSettings,
    *,
    cache_dir: Path | None = None,
) -> Finding:
    """Ask the LLM. Returns a Finding with method="llm" whose evidence quotes were all verified in `tree`."""
    raise NotImplementedError
