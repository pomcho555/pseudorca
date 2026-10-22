"""Results shared by the audit methods (fingerprint, patch, LLM, human)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pseudorca.model import Idea


@dataclass
class Evidence:
    path: str  # a path in the notebook's Tree
    line: int | None  # 1-based line of the quote's first line, when known
    quote: str  # text that occurs in that file (whitespace differences allowed)

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "line": self.line, "quote": self.quote}


@dataclass
class Finding:
    """One method's verdict on whether an idea is in a notebook."""

    status: str  # present | partial | absent | unknown
    method: str  # fingerprint | patch | llm | human
    evidence: list[Evidence] = field(default_factory=list)
    note: str = ""
    model: str = ""  # for method="llm": provider:model
    detail: dict[str, Any] = field(default_factory=dict)  # method-specific, JSON-serialisable

    def event_data(self, nb_id: str, idea: Idea) -> dict[str, Any]:
        """The data of the audit.verdict event that records this finding."""
        return {
            "nb": nb_id,
            "idea": idea.id,
            "idea_rev": idea.rev,
            "status": self.status,
            "method": self.method,
            "evidence": [e.to_dict() for e in self.evidence],
            "note": self.note,
            "model": self.model,
            "detail": self.detail,
        }
