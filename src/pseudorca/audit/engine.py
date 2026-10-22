"""Audit orchestration: deterministic methods first, the LLM when they are inconclusive, one event per finding.

For each idea:
1. fingerprint (when the idea has fingerprints) and patch (when it has a patch);
2. the LLM: never, always, or (auto) only when the deterministic verdicts are not conclusive;
3. each new finding is written as one audit.verdict event (nothing is written in a dry run).
A method is skipped, unless `force` is given, when what is recorded for it is at least as new as what this
run would record. For fingerprint and patch: the verdict that counts (see `State.current_records`) has a
(`detail.extractor_version`, `detail.matcher_version`) >= (EXTRACTOR_VERSION, the method's MATCHER_VERSION),
compared in that order. For the LLM: a verdict for this rev from the same model has `detail.prompt_version`
>= PROMPT_VERSION, or one from any model has a newer prompt version. Versions compare as integers, and a
missing one counts as 0. So an older Pseudorca never redoes a newer one's verdicts, an upgrade of the
extractor or of a matcher redoes older ones, and each model is asked once per prompt version.

The ledger is read once per audit: the effective verdicts are computed at the end from that snapshot plus
this run's own findings, never by replaying the events after each one is written.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from types import ModuleType
from typing import TYPE_CHECKING, Any

from pseudorca.audit import fingerprint, patch
from pseudorca.audit.types import Evidence, Finding
from pseudorca.model import (
    DETERMINISTIC,
    VERDICT_STATUSES,
    Effective,
    Idea,
    Notebook,
    State,
    VerdictRecord,
    record_version,
)
from pseudorca.store import Event, Store, StoreError
from pseudorca.tree import Tree

if TYPE_CHECKING:
    from pseudorca.llm import LLMSettings

LLM_MODES = ("auto", "always", "never")
MATCHERS: dict[str, ModuleType] = {"fingerprint": fingerprint, "patch": patch}  # each has a MATCHER_VERSION

# judge(idea, patch_text, tree, findings, settings, *, cache_dir) -> Finding, like pseudorca.llm.judge_idea
Judge = Callable[..., Finding]


@dataclass
class AuditRow:
    """What one audit did for one idea."""

    idea: Idea
    findings: list[Finding] = field(default_factory=list)  # new ones: written, or to be written in a dry run
    skipped: list[str] = field(default_factory=list)  # methods whose current verdict was already recorded
    effective: Effective | None = None  # after writing; in a dry run, as if the findings had been written
    llm_note: str = ""  # why the LLM was wanted but not asked
    llm_error: str = ""  # the LLM was asked and failed
    notes: list[str] = field(default_factory=list)  # other problems, such as a patch blob that is missing


class AuditRows(list[AuditRow]):
    """The rows of one audit run, plus messages about the run as a whole (each reported once)."""

    def __init__(self, rows: Iterable[AuditRow] = (), messages: Iterable[str] = ()) -> None:
        super().__init__(rows)
        self.messages: list[str] = list(messages)


def conclusive(statuses: Iterable[str]) -> bool:
    """Deterministic statuses settle the question: they agree on present or absent, and none is partial."""
    s = set(statuses)
    return len(s & {"present", "absent"}) == 1 and "partial" not in s


def audit_notebook(
    store: Store,
    nb: Notebook,
    ideas: Sequence[Idea],
    *,
    llm_mode: str = "auto",
    settings: LLMSettings | None = None,
    force: bool = False,
    dry_run: bool = False,
    tree: Tree | None = None,
    judge: Judge | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> AuditRows:
    """Audit `ideas` in `nb` and return one row per idea, in order.

    `tree` defaults to `pseudorca.extract.tree_for(store, nb)`, loaded only when a method needs it, and
    `judge` to `pseudorca.llm.judge_idea`. `on_progress` receives a line before each (slow) LLM call.
    """
    if llm_mode not in LLM_MODES:
        raise ValueError(f"llm mode must be one of {', '.join(LLM_MODES)}, not {llm_mode!r}")
    run = _Run(store, nb, llm_mode, settings, force, dry_run, tree, judge, on_progress)
    for idea in ideas:
        run.rows.append(run.audit(idea))
    for row in run.rows:  # once, at the end: as recorded before this run, plus this run's findings
        row.effective = run.view(row.idea).effective_verdict(nb.id, row.idea.id)
    if run.unconfigured:
        n = run.unconfigured
        if llm_mode == "auto":
            run.rows.messages.append(
                f"{_ideas(n)} {'is' if n == 1 else 'are'} inconclusive and [llm] is not configured"
            )
        else:
            run.rows.messages.append(f"[llm] is not configured, so {_ideas(n)} did not go to the LLM")
    return run.rows


def _ideas(n: int) -> str:
    return f"{n} idea" if n == 1 else f"{n} ideas"


def _int(value: Any) -> int:
    """A version from event data as an integer (0 when it is missing or not a number)."""
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


class _Run:
    def __init__(
        self,
        store: Store,
        nb: Notebook,
        llm_mode: str,
        settings: LLMSettings | None,
        force: bool,
        dry_run: bool,
        tree: Tree | None,
        judge: Judge | None,
        on_progress: Callable[[str], None] | None,
    ) -> None:
        self.store, self.nb, self.llm_mode, self.settings = store, nb, llm_mode, settings
        self.force, self.dry_run, self.judge, self.on_progress = force, dry_run, judge, on_progress
        self.state = store.state()  # the one snapshot this run reads; its own findings are kept in `new`
        self.new: dict[str, list[VerdictRecord]] = {}
        self._tree = tree
        self._patch: tuple[str, str | None] | None = None  # (idea id, patch text) of the current idea
        self.rows = AuditRows()
        self.unconfigured = 0  # ideas that wanted the LLM while no settings were given
        self.allowed: bool | None = None  # outcome of llm.ensure_allowed, once checked

    def tree(self) -> Tree:
        if self._tree is None:
            from pseudorca import extract

            self._tree = extract.tree_for(self.store, self.nb)
        return self._tree

    def records(self, idea: Idea) -> list[VerdictRecord]:
        """The verdicts on `idea` in this notebook: recorded before this run, then this run's."""
        return [*self.state.verdicts.get((self.nb.id, idea.id), []), *self.new.get(idea.id, [])]

    def view(self, idea: Idea) -> State:
        """The verdicts on `idea` (recorded before this run, then this run's), judged for `idea` as given."""
        st = State()
        st.ideas[idea.id] = idea
        st.verdicts[(self.nb.id, idea.id)] = self.records(idea)
        return st

    def audit(self, idea: Idea) -> AuditRow:
        from pseudorca import extract

        row = AuditRow(idea)
        current = self.view(idea).current_records(self.nb.id, idea.id)
        det = {m: _as_finding(current[m]) for m in DETERMINISTIC if m in current}
        methods = []
        if any(f.strip() for f in idea.fingerprints):
            methods.append("fingerprint")
        if idea.patch:
            methods.append("patch")
        for method in methods:
            # in the order model.record_version compares them
            versions = {
                "extractor_version": int(extract.EXTRACTOR_VERSION),
                "matcher_version": int(MATCHERS[method].MATCHER_VERSION),
            }
            rec = current.get(method)
            if rec is not None and not self.force and record_version(rec) >= tuple(versions.values()):
                row.skipped.append(method)
                continue
            if method == "fingerprint":
                found = fingerprint.check(idea.fingerprints, self.tree())
            else:
                text = self.patch_text(idea, row)
                found = None if text is None else patch.check(text, self.tree())
            if found is not None:
                found.detail = {**found.detail, **versions}
                self.record(row, idea, found)
                det[method] = found
        if not methods:
            row.notes.append("no fingerprints or patch to check")

        findings = [det[m] for m in DETERMINISTIC if m in det]
        if self.llm_mode == "always" or (
            self.llm_mode == "auto" and not conclusive(f.status for f in findings)
        ):
            self.ask_llm(idea, row, current.get("llm"), findings)
        return row

    def patch_text(self, idea: Idea, row: AuditRow) -> str | None:
        if self._patch is None or self._patch[0] != idea.id:
            try:
                # utf-8-sig: a byte order mark would otherwise hide the first file header
                text: str | None = self.store.get_blob(idea.patch or "").decode("utf-8-sig", "replace")
            except StoreError as e:
                text = None
                row.notes.append(f"patch unavailable: {e}")
            self._patch = (idea.id, text)
        return self._patch[1]

    def record(self, row: AuditRow, idea: Idea, finding: Finding) -> None:
        data = finding.event_data(self.nb.id, idea)
        event = None if self.dry_run else self.store.append("audit.verdict", data)
        self.new.setdefault(idea.id, []).append(_record(data, event))
        row.findings.append(finding)

    def ask_llm(self, idea: Idea, row: AuditRow, current: VerdictRecord | None, det: list[Finding]) -> None:
        settings = self.settings
        if settings is None:
            # a teammate's LLM verdict for this rev is enough; otherwise the idea waits for an LLM
            if current is None:
                self.unconfigured += 1
                row.llm_note = "LLM not configured"
            return
        from pseudorca import llm

        # every LLM verdict for this rev, not only the one that counts: with two models at the same prompt
        # version the latest one counts, and asking the other model again would only flip it back
        asked = [r for r in self.records(idea) if r.method == "llm" and r.idea_rev == idea.rev]
        if not self.force and any(_llm_is_current(r, settings.label, llm.PROMPT_VERSION) for r in asked):
            row.skipped.append("llm")
            return
        if self.allowed is None:
            try:
                llm.ensure_allowed(settings)
                self.allowed = True
            except llm.LLMError as e:
                self.allowed = False
                self.rows.messages.append(f"LLM disabled for this run: {e}")
        if not self.allowed:
            row.llm_note = "LLM not allowed"
            return

        if self.on_progress is not None:
            self.on_progress(f"{idea.id}: asking {settings.label}")
        judge = self.judge or llm.judge_idea
        patch_text = self.patch_text(idea, row) if idea.patch else None
        cache_dir = None if self.dry_run else self.store.cache_dir("llm")
        try:
            found = judge(idea, patch_text, self.tree(), det, settings, cache_dir=cache_dir)
        except llm.LLMError as e:
            row.llm_error = str(e) or type(e).__name__
            return
        if found.status not in VERDICT_STATUSES:
            row.llm_error = f"the LLM gave an invalid status {found.status!r}"
            return
        # the keys that make a later run skip this verdict come from here, never from the judge
        found.method, found.model = "llm", settings.label
        found.detail = {**found.detail, "prompt_version": llm.PROMPT_VERSION}
        self.record(row, idea, found)


def _llm_is_current(r: VerdictRecord, label: str, prompt_version: str) -> bool:
    """Whether an LLM verdict for this rev makes asking `label` now pointless: it came from the same model
    with this prompt version or a newer one, or from any model with a newer one (an older prompt's answer
    would never count over it)."""
    recorded, ours = record_version(r)[0], _int(prompt_version)
    return recorded > ours or (recorded == ours and r.model == label)


def _record(data: dict[str, Any], event: Event | None) -> VerdictRecord:
    """The record a finding's event makes (in a dry run, without an event id)."""
    return VerdictRecord(
        event_id=event.id if event else "",
        nb=data["nb"],
        idea=data["idea"],
        idea_rev=data["idea_rev"],
        status=data["status"],
        method=data["method"],
        evidence=list(data["evidence"]),
        note=data["note"],
        model=data["model"],
        detail=dict(data["detail"]),
        time=event.time if event else "",
        actor=event.actor if event else "",
    )


def _as_finding(r: VerdictRecord) -> Finding:
    evidence = [
        Evidence(
            str(e.get("path", "")),
            e.get("line"),
            str(e.get("quote", "")),
            e["col"] if type(e.get("col")) is int else None,
        )
        for e in r.evidence
        if isinstance(e, dict)
    ]
    return Finding(r.status, r.method, evidence, note=r.note, model=r.model, detail=dict(r.detail))
