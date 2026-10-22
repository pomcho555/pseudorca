"""Audit orchestration: deterministic methods first, the LLM when they are inconclusive, one event per finding.

For each idea:
1. fingerprint (when the idea has fingerprints) and patch (when it has a patch);
2. the LLM: never, always, or (auto) only when the deterministic verdicts are not conclusive;
3. each new finding is written as one audit.verdict event (nothing is written in a dry run).
A method is skipped when an equal verdict is already recorded (same notebook, idea, rev and method; for the
LLM also the same model and PROMPT_VERSION), unless `force` is given.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pseudorca.audit import fingerprint, patch
from pseudorca.audit.types import Evidence, Finding
from pseudorca.model import DETERMINISTIC, VERDICT_STATUSES, Effective, Idea, Notebook, State, VerdictRecord
from pseudorca.store import Store, StoreError
from pseudorca.tree import Tree

if TYPE_CHECKING:
    from pseudorca.llm import LLMSettings

LLM_MODES = ("auto", "always", "never")

# judge(idea, patch_text, tree, findings, settings, *, cache_dir) -> Finding, like pseudorca.llm.judge_idea
Judge = Callable[..., Finding]


@dataclass
class AuditRow:
    """What one audit did for one idea."""

    idea: Idea
    findings: list[Finding] = field(default_factory=list)  # new ones: written, or to be written in a dry run
    skipped: list[str] = field(default_factory=list)  # methods whose equal verdict was already recorded
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

    def audit(self, idea: Idea) -> AuditRow:
        row = AuditRow(idea)
        records = self.store.state().verdicts.get((self.nb.id, idea.id), [])
        rev = idea.rev
        # latest still-valid recorded verdict per deterministic method, replaced below by new findings
        current = {
            r.method: _as_finding(r) for r in records if r.idea_rev == rev and r.method in DETERMINISTIC
        }
        methods = []
        if any(f.strip() for f in idea.fingerprints):
            methods.append("fingerprint")
        if idea.patch:
            methods.append("patch")
        for method in methods:
            if method in current and not self.force:
                row.skipped.append(method)
                continue
            if method == "fingerprint":
                found = fingerprint.check(idea.fingerprints, self.tree())
            else:
                text = self.patch_text(idea, row)
                found = None if text is None else patch.check(text, self.tree())
            if found is not None:
                self.record(row, idea, found)
                current[method] = found
        if not methods:
            row.notes.append("no fingerprints or patch to check")

        det = list(current.values())
        if self.llm_mode == "always" or (self.llm_mode == "auto" and not conclusive(f.status for f in det)):
            self.ask_llm(idea, row, records, det)

        if self.dry_run:
            row.effective = _simulated(self.store.state(), self.nb.id, idea, row.findings)
        else:
            row.effective = self.store.state().effective_verdict(self.nb.id, idea.id)
        return row

    def patch_text(self, idea: Idea, row: AuditRow) -> str | None:
        if self._patch is None or self._patch[0] != idea.id:
            try:
                text: str | None = self.store.get_blob(idea.patch or "").decode("utf-8", "replace")
            except StoreError as e:
                text = None
                row.notes.append(f"patch unavailable: {e}")
            self._patch = (idea.id, text)
        return self._patch[1]

    def record(self, row: AuditRow, idea: Idea, finding: Finding) -> None:
        if not self.dry_run:
            self.store.append("audit.verdict", finding.event_data(self.nb.id, idea))
        row.findings.append(finding)

    def ask_llm(self, idea: Idea, row: AuditRow, records: list[VerdictRecord], det: list[Finding]) -> None:
        settings = self.settings
        if settings is None:
            # a teammate's LLM verdict for this rev is enough; otherwise the idea waits for an LLM
            if not any(r.method == "llm" and r.idea_rev == idea.rev for r in records):
                self.unconfigured += 1
                row.llm_note = "LLM not configured"
            return
        from pseudorca import llm

        if not self.force and any(
            _same_llm(r, idea.rev, settings.label, llm.PROMPT_VERSION) for r in records
        ):
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


def _same_llm(r: VerdictRecord, rev: str, label: str, prompt_version: str) -> bool:
    return (
        r.method == "llm"
        and r.idea_rev == rev
        and r.model == label
        and str(r.detail.get("prompt_version", "")) == str(prompt_version)
    )


def _as_finding(r: VerdictRecord) -> Finding:
    evidence = [
        Evidence(str(e.get("path", "")), e.get("line"), str(e.get("quote", "")))
        for e in r.evidence
        if isinstance(e, dict)
    ]
    return Finding(r.status, r.method, evidence, note=r.note, model=r.model, detail=dict(r.detail))


def _simulated(state: State, nb_id: str, idea: Idea, findings: list[Finding]) -> Effective | None:
    """The effective verdict as if `findings` had been written after the recorded verdicts."""
    sim = State()
    sim.ideas[idea.id] = idea
    new = [
        VerdictRecord(
            event_id="",
            nb=nb_id,
            idea=idea.id,
            idea_rev=idea.rev,
            status=f.status,
            method=f.method,
            evidence=[e.to_dict() for e in f.evidence],
            note=f.note,
            model=f.model,
            detail=dict(f.detail),
        )
        for f in findings
    ]
    sim.verdicts[(nb_id, idea.id)] = [*state.verdicts.get((nb_id, idea.id), []), *new]
    return sim.effective_verdict(nb_id, idea.id)
