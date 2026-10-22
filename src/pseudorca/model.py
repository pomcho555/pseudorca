"""Ledger entities, and the state rebuilt by replaying events in id order.

Concurrent edits from different people resolve as last-writer-wins per field, by event id (= time).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from pseudorca import util
from pseudorca.store import Event

IDEA_STATUSES = ("idea", "testing", "adopted", "rejected", "parked")
VERDICT_STATUSES = ("present", "partial", "absent", "unknown")
METHODS = ("fingerprint", "patch", "llm", "human")
DETERMINISTIC = ("fingerprint", "patch")


@dataclass
class Idea:
    """An improvement someone wants in (or out of) the notebooks: a prompt change, a serving knob, a patch."""

    id: str
    title: str = ""
    description: str = ""
    owner: str = ""
    status: str = "idea"
    tags: list[str] = field(default_factory=list)
    links: list[str] = field(default_factory=list)
    fingerprints: list[str] = field(default_factory=list)
    patch: str | None = None  # blob sha256 of a unified diff
    created: str = ""
    created_by: str = ""
    updated: str = ""
    updated_by: str = ""

    @property
    def rev(self) -> str:
        """Hash of the fields an audit depends on. Verdicts made for another rev are stale."""
        payload = json.dumps(
            [self.title, self.description, sorted(self.fingerprints), self.patch], ensure_ascii=False
        )
        return util.sha256_bytes(payload.encode())[:16]


IDEA_FIELDS = ("title", "description", "owner", "status", "tags", "links", "fingerprints", "patch")


@dataclass
class Notebook:
    """A notebook (or script) version, identified by the sha256 of its bytes."""

    id: str
    name: str = ""
    path: str = ""
    kernel: str = ""  # Kaggle kernel ref "owner/slug", if any
    version: str = ""  # Kaggle kernel version, if known
    parent: str | None = None
    applies: list[str] = field(default_factory=list)  # ideas its author says are in it
    tags: list[str] = field(default_factory=list)
    note: str = ""
    stored: bool = True  # the bytes are in blobs/
    size: int = 0
    added: str = ""
    added_by: str = ""
    seq: int = 0  # registration order; "latest" is the highest

    @property
    def label(self) -> str:
        return self.name or util.short(self.id)


NOTEBOOK_FIELDS = ("name", "path", "kernel", "version", "parent", "applies", "tags", "note", "stored", "size")


@dataclass
class Submission:
    """One leaderboard submission and its scores."""

    id: str
    competition: str = ""
    nb: str | None = None
    description: str = ""
    date: str = ""
    status: str = ""
    public_score: float | None = None
    private_score: float | None = None
    submitted_by: str = ""
    source: str = "manual"  # manual | kaggle
    extra: dict[str, Any] = field(default_factory=dict)
    updated: str = ""


SUBMISSION_FIELDS = (
    "competition",
    "nb",
    "description",
    "date",
    "status",
    "public_score",
    "private_score",
    "submitted_by",
    "source",
    "extra",
)


@dataclass
class Run:
    """A local evaluation run: a results table with one row per (item, draw)."""

    id: str
    results: str  # blob sha256 of the table
    format: str = "csv"  # csv | jsonl
    nb: str | None = None
    label: str = ""
    protocol: str = ""
    note: str = ""
    n_rows: int = 0
    added: str = ""
    added_by: str = ""


RUN_FIELDS = ("results", "format", "nb", "label", "protocol", "note", "n_rows")


@dataclass
class VerdictRecord:
    """One method's answer to "is this idea in this notebook?", as recorded by an audit.verdict event."""

    event_id: str
    nb: str
    idea: str
    idea_rev: str
    status: str
    method: str
    evidence: list[dict[str, Any]] = field(default_factory=list)  # [{"path", "line", "quote"}]
    note: str = ""
    model: str = ""
    detail: dict[str, Any] = field(default_factory=dict)
    time: str = ""
    actor: str = ""


@dataclass
class Effective:
    """The verdict to show for an (notebook, idea) pair after combining all methods."""

    status: str  # present | partial | absent | unknown | conflict
    method: str  # e.g. "human", "patch", "fingerprint+patch", "llm", "det"
    stale: bool
    records: list[VerdictRecord]


def _set_fields(obj: Any, data: dict[str, Any], allowed: Iterable[str]) -> None:
    for key in allowed:
        if key in data:
            value = data[key]
            if isinstance(getattr(obj, key, None), list) and value is None:
                value = []
            setattr(obj, key, list(value) if isinstance(value, (list, tuple)) else value)


class State:
    """Everything in the ledger, as of the last replayed event."""

    def __init__(self) -> None:
        self.ideas: dict[str, Idea] = {}
        self.notebooks: dict[str, Notebook] = {}
        self.subs: dict[str, Submission] = {}
        self.runs: dict[str, Run] = {}
        self.verdicts: dict[tuple[str, str], list[VerdictRecord]] = {}
        self.events: list[Event] = []
        self.warnings: list[str] = []

    @classmethod
    def replay(cls, events: Iterable[Event]) -> State:
        st = cls()
        for ev in events:
            st.events.append(ev)
            handler = getattr(st, "_on_" + ev.type.replace(".", "_"), None)
            if handler is None:
                st.warnings.append(f"ignored event {ev.id} of unknown type {ev.type!r} (newer Pseudorca?)")
                continue
            try:
                handler(ev)
            except (KeyError, TypeError, ValueError) as e:
                st.warnings.append(f"ignored malformed event {ev.id} ({ev.type}): {e}")
        return st

    # ----- event handlers --------------------------------------------------------------------------

    def _on_idea_add(self, ev: Event) -> None:
        iid = str(ev.data["id"])
        idea = self.ideas.get(iid)
        if idea is None:
            idea = self.ideas[iid] = Idea(id=iid, created=ev.time, created_by=ev.actor)
        else:
            self.warnings.append(
                f"idea {iid!r} was added twice; the later add ({ev.id}) is applied as an update"
            )
        _set_fields(idea, ev.data, IDEA_FIELDS)
        idea.updated, idea.updated_by = ev.time, ev.actor

    def _on_idea_update(self, ev: Event) -> None:
        iid = str(ev.data["id"])
        idea = self.ideas.get(iid)
        if idea is None:
            self.warnings.append(f"update for unknown idea {iid!r} ({ev.id}) created it")
            idea = self.ideas[iid] = Idea(id=iid, created=ev.time, created_by=ev.actor)
        _set_fields(idea, ev.data, IDEA_FIELDS)
        idea.updated, idea.updated_by = ev.time, ev.actor

    def _on_nb_add(self, ev: Event) -> None:
        nid = str(ev.data["id"])
        nb = self.notebooks.get(nid)
        if nb is None:
            nb = self.notebooks[nid] = Notebook(
                id=nid, added=ev.time, added_by=ev.actor, seq=len(self.notebooks) + 1
            )
        _set_fields(nb, ev.data, NOTEBOOK_FIELDS)

    def _on_nb_update(self, ev: Event) -> None:
        nid = str(ev.data["id"])
        nb = self.notebooks.get(nid)
        if nb is None:
            self.warnings.append(f"update for unknown notebook {nid[:12]} ({ev.id}) ignored")
            return
        _set_fields(nb, ev.data, NOTEBOOK_FIELDS)

    def _on_sub_upsert(self, ev: Event) -> None:
        sid = str(ev.data["id"])
        sub = self.subs.get(sid)
        if sub is None:
            sub = self.subs[sid] = Submission(id=sid)
        _set_fields(sub, ev.data, SUBMISSION_FIELDS)
        for key in ("public_score", "private_score"):
            value = getattr(sub, key)
            setattr(sub, key, None if value in (None, "") else float(value))
        sub.updated = ev.time

    def _on_run_add(self, ev: Event) -> None:
        rid = str(ev.data["id"])
        run = self.runs[rid] = Run(id=rid, results=str(ev.data["results"]), added=ev.time, added_by=ev.actor)
        _set_fields(run, ev.data, RUN_FIELDS)

    def _on_audit_verdict(self, ev: Event) -> None:
        d = ev.data
        status, method = str(d["status"]), str(d["method"])
        if status not in VERDICT_STATUSES or method not in METHODS:
            raise ValueError(f"bad status/method {status!r}/{method!r}")
        rec = VerdictRecord(
            event_id=ev.id,
            nb=str(d["nb"]),
            idea=str(d["idea"]),
            idea_rev=str(d.get("idea_rev", "")),
            status=status,
            method=method,
            evidence=list(d.get("evidence") or []),
            note=str(d.get("note", "")),
            model=str(d.get("model", "")),
            detail=dict(d.get("detail") or {}),
            time=ev.time,
            actor=ev.actor,
        )
        self.verdicts.setdefault((rec.nb, rec.idea), []).append(rec)

    # ----- lookups ---------------------------------------------------------------------------------

    def notebooks_in_order(self) -> list[Notebook]:
        return sorted(self.notebooks.values(), key=lambda n: n.seq)

    def ideas_in_order(self) -> list[Idea]:
        rank = {s: i for i, s in enumerate(IDEA_STATUSES)}
        return sorted(self.ideas.values(), key=lambda i: (rank.get(i.status, len(rank)), i.id))

    def resolve_nb(self, ref: str) -> Notebook:
        """A notebook by full sha256, `latest`, `latest:<tag>`, name, Kaggle kernel ref or sha prefix (6+)."""
        ref = ref.strip()
        ordered = self.notebooks_in_order()
        if not ordered:
            raise LookupError(
                "no notebooks registered yet (use `pseudorca nb add` or `pseudorca kaggle pull`)"
            )
        if ref in self.notebooks:
            return self.notebooks[ref]
        if ref == "latest" or ref.startswith("latest:"):
            tag = ref.partition(":")[2]
            candidates = [n for n in ordered if not tag or tag in n.tags]
            if not candidates:
                raise LookupError(f"no notebook has the tag {tag!r}")
            return candidates[-1]
        named = [n for n in ordered if n.name == ref] or [n for n in ordered if n.kernel and n.kernel == ref]
        if named:
            return named[-1]  # a reused name means its newest notebook
        if len(ref) >= 6 and util.HEX_RE.match(ref.lower()):
            hits = [n for n in ordered if n.id.startswith(ref.lower())]
            if len(hits) == 1:
                return hits[0]
            if len(hits) > 1:
                raise LookupError(f"{ref!r} matches {len(hits)} notebooks; use more characters")
        raise LookupError(f"no notebook matches {ref!r}")

    def resolve_idea(self, ref: str) -> Idea:
        ref = ref.strip()
        if ref in self.ideas:
            return self.ideas[ref]
        hits = [i for i in self.ideas.values() if i.id.startswith(ref)]
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            raise LookupError(f"{ref!r} matches several ideas: {', '.join(sorted(i.id for i in hits))}")
        raise LookupError(f"no idea matches {ref!r}")

    # ----- verdicts --------------------------------------------------------------------------------

    def effective_verdict(self, nb_id: str, idea_id: str) -> Effective | None:
        """Combine every recorded method into the one verdict to show.

        1. The latest human verdict wins.
        2. Otherwise, among verdicts made for the idea's current rev: the deterministic methods
           (fingerprint, patch) win when they agree on present or absent and none says partial;
           if they contradict each other the result is "conflict".
        3. Otherwise a non-unknown LLM verdict.
        4. Otherwise "partial" if a deterministic method said so, else "unknown".
        If only verdicts for an older rev exist, the latest one is shown with stale=True.
        """
        records = self.verdicts.get((nb_id, idea_id))
        if not records:
            return None
        idea = self.ideas.get(idea_id)
        rev = idea.rev if idea else ""
        humans = [r for r in records if r.method == "human"]
        if humans:
            r = humans[-1]
            return Effective(r.status, "human", bool(rev) and r.idea_rev != rev, [r])
        latest: dict[str, VerdictRecord] = {}
        for r in records:
            if r.idea_rev == rev:
                latest[r.method] = r
        if not latest:
            r = records[-1]
            return Effective(r.status, r.method, True, [r])
        det = [latest[m] for m in DETERMINISTIC if m in latest]
        conclusive = {r.status for r in det if r.status in ("present", "absent")}
        has_partial = any(r.status == "partial" for r in det)
        if len(conclusive) == 2:
            return Effective("conflict", "det", False, det)
        if len(conclusive) == 1 and not has_partial:
            status = conclusive.pop()
            return Effective(status, "+".join(r.method for r in det if r.status == status), False, det)
        llm = latest.get("llm")
        if llm is not None and llm.status != "unknown":
            return Effective(llm.status, "llm", False, [llm, *det])
        if has_partial:
            return Effective("partial", "det", False, det)
        return Effective("unknown", "+".join(latest), False, list(latest.values()))
