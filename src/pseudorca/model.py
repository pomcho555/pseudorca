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
    file: str = ""  # name of the file it was imported from
    added: str = ""
    added_by: str = ""


RUN_FIELDS = ("results", "format", "nb", "label", "protocol", "note", "n_rows", "file")


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


def _score(value: Any) -> float | None:
    """A leaderboard score from event data: a finite number, or None for empty."""
    if value is None or value == "":
        return None
    score = float(value)
    if score != score or score in (float("inf"), float("-inf")):
        raise ValueError(f"score {value!r} is not a finite number")
    return score


VERDICT_FIELDS = ("nb", "idea", "idea_rev", "status", "method", "evidence", "note", "model", "detail")
KNOWN_FIELDS = {
    "idea.add": {"id", *IDEA_FIELDS},
    "idea.update": {"id", *IDEA_FIELDS},
    "nb.add": {"id", *NOTEBOOK_FIELDS},
    "nb.update": {"id", *NOTEBOOK_FIELDS},
    "sub.upsert": {"id", *SUBMISSION_FIELDS},
    "run.add": {"id", *RUN_FIELDS},
    "audit.verdict": set(VERDICT_FIELDS),
}


def _version(record: VerdictRecord, key: str) -> int:
    try:
        return int(record.detail.get(key) or 0)
    except (TypeError, ValueError):
        return 0


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
        self._warned: set[tuple[str, str]] = set()

    @classmethod
    def replay(cls, events: Iterable[Event]) -> State:
        st = cls()
        for ev in events:
            st.apply(ev)
        return st

    def _warn_once(self, key: tuple[str, str], message: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            self.warnings.append(message)

    def apply(self, ev: Event) -> None:
        """Apply one event. Events must come in id order (Store.append guarantees the new one is last)."""
        from pseudorca.store import SCHEMA

        self.events.append(ev)
        if ev.schema > SCHEMA:
            self._warn_once(
                ("schema", str(ev.schema)),
                f"some events were written by a newer Pseudorca (schema {ev.schema}); upgrade to read them fully",
            )
        handler = getattr(self, "_on_" + ev.type.replace(".", "_"), None)
        if handler is None:
            self._warn_once(
                ("type", ev.type), f"ignored events of unknown type {ev.type!r} (newer Pseudorca?)"
            )
            return
        for key in sorted(set(ev.data) - KNOWN_FIELDS.get(ev.type, set())):
            self._warn_once(
                (ev.type, key), f"ignored field {key!r} of {ev.type} events (written by a newer Pseudorca?)"
            )
        try:
            handler(ev)
        except (KeyError, TypeError, ValueError) as e:
            self.warnings.append(f"ignored malformed event {ev.id} ({ev.type}): {e}")

    # ----- event handlers (validate first, then change state) --------------------------------------

    def _idea_data(self, ev: Event) -> dict[str, Any]:
        data = dict(ev.data)
        patch = data.get("patch")
        if patch is not None and not util.is_sha256(patch):
            self.warnings.append(f"ignored invalid patch id in event {ev.id} (not a sha256)")
            data.pop("patch")
        return data

    def _on_idea_add(self, ev: Event) -> None:
        iid = str(ev.data["id"])
        data = self._idea_data(ev)
        idea = self.ideas.get(iid)
        if idea is None:
            idea = self.ideas[iid] = Idea(id=iid, created=ev.time, created_by=ev.actor)
        else:
            self.warnings.append(
                f"idea {iid!r} was added twice; the later add ({ev.id}) is applied as an update"
            )
        _set_fields(idea, data, IDEA_FIELDS)
        idea.updated, idea.updated_by = ev.time, ev.actor

    def _on_idea_update(self, ev: Event) -> None:
        iid = str(ev.data["id"])
        data = self._idea_data(ev)
        idea = self.ideas.get(iid)
        if idea is None:
            self.warnings.append(f"update for unknown idea {iid!r} ({ev.id}) created it")
            idea = self.ideas[iid] = Idea(id=iid, created=ev.time, created_by=ev.actor)
        _set_fields(idea, data, IDEA_FIELDS)
        idea.updated, idea.updated_by = ev.time, ev.actor

    def _on_nb_add(self, ev: Event) -> None:
        nid = str(ev.data["id"])
        if not util.is_sha256(nid):
            raise ValueError("notebook id is not a sha256")
        nb = self.notebooks.get(nid)
        if nb is None:
            nb = self.notebooks[nid] = Notebook(
                id=nid, added=ev.time, added_by=ev.actor, seq=len(self.notebooks) + 1
            )
            _set_fields(nb, ev.data, NOTEBOOK_FIELDS)
            return
        # registered again (two clones before a pull): merge, never wipe what the first add recorded
        d = ev.data
        for key in ("tags", "applies"):
            extra = [v for v in d.get(key) or [] if v not in getattr(nb, key)]
            setattr(nb, key, [*getattr(nb, key), *extra])
        for key in ("name", "path", "kernel", "version", "note", "parent"):
            if not getattr(nb, key) and d.get(key):
                setattr(nb, key, d[key])
        nb.stored = bool(nb.stored or d.get("stored"))
        nb.size = nb.size or int(d.get("size") or 0)

    def _on_nb_update(self, ev: Event) -> None:
        nid = str(ev.data["id"])
        nb = self.notebooks.get(nid)
        if nb is None:
            self.warnings.append(f"update for unknown notebook {nid[:12]} ({ev.id}) ignored")
            return
        _set_fields(nb, ev.data, NOTEBOOK_FIELDS)

    def _on_sub_upsert(self, ev: Event) -> None:
        sid = str(ev.data["id"])
        data = dict(ev.data)
        for key in ("public_score", "private_score"):
            if key in data:
                try:
                    data[key] = _score(data[key])
                except (TypeError, ValueError):
                    self.warnings.append(f"ignored invalid {key} {data[key]!r} in event {ev.id}")
                    data.pop(key)
        sub = self.subs.get(sid)
        if sub is None:
            sub = self.subs[sid] = Submission(id=sid)
        _set_fields(sub, data, SUBMISSION_FIELDS)
        sub.updated = ev.time

    def _on_run_add(self, ev: Event) -> None:
        rid = str(ev.data["id"])
        results = ev.data["results"]
        if not util.is_sha256(results):
            raise ValueError("results is not a blob id (sha256)")
        run = self.runs[rid] = Run(id=rid, results=results, added=ev.time, added_by=ev.actor)
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
            evidence=[e for e in d.get("evidence") or [] if isinstance(e, dict)],
            note=str(d.get("note") or ""),
            model=str(d.get("model") or ""),
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

    def current_records(self, nb_id: str, idea_id: str) -> dict[str, VerdictRecord]:
        """Per machine method, the record that counts for the idea's current rev: the one made with the
        newest extractor (deterministic) or prompt (llm) version, latest first among equals. So a teammate
        on an older Pseudorca cannot override a newer one's verdict."""
        idea = self.ideas.get(idea_id)
        rev = idea.rev if idea else ""
        best: dict[str, VerdictRecord] = {}
        for r in self.verdicts.get((nb_id, idea_id), []):
            if r.method == "human" or r.idea_rev != rev:
                continue
            key = "prompt_version" if r.method == "llm" else "extractor_version"
            cur = best.get(r.method)
            if cur is None or _version(r, key) >= _version(cur, key):
                best[r.method] = r
        return best

    def effective_verdict(self, nb_id: str, idea_id: str) -> Effective | None:
        """Combine every recorded method into the one verdict to show.

        1. The latest human verdict wins.
        2. Otherwise, among verdicts for the idea's current rev (see `current_records`): the deterministic
           methods (fingerprint, patch) win when they agree on present or absent and none says partial.
        3. Otherwise a non-unknown LLM verdict (this also settles a fingerprint/patch contradiction).
        4. Otherwise "conflict" if the deterministic methods contradict each other, "partial" if one said
           so, else "unknown".
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
        latest = self.current_records(nb_id, idea_id)
        if not latest:
            r = records[-1]
            return Effective(r.status, r.method, True, [r])
        det = [latest[m] for m in DETERMINISTIC if m in latest]
        conclusive = {r.status for r in det if r.status in ("present", "absent")}
        has_partial = any(r.status == "partial" for r in det)
        if len(conclusive) == 1 and not has_partial:
            status = conclusive.pop()
            return Effective(status, "+".join(r.method for r in det if r.status == status), False, det)
        llm = latest.get("llm")
        if llm is not None and llm.status != "unknown":
            return Effective(llm.status, "llm", False, [llm, *det])
        if len(conclusive) == 2:
            return Effective("conflict", "det", False, det)
        if has_partial:
            return Effective("partial", "det", False, det)
        return Effective("unknown", "+".join(latest), False, list(latest.values()))
