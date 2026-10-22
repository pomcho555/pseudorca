from __future__ import annotations

from pathlib import Path

import pytest

import pseudorca.extract
import pseudorca.llm
from pseudorca.audit import engine
from pseudorca.audit.engine import audit_notebook, conclusive
from pseudorca.audit.types import Evidence, Finding
from pseudorca.llm import LLMError, LLMSettings
from pseudorca.model import Idea, Notebook, State
from pseudorca.notebooks import register_bytes
from pseudorca.store import Store
from pseudorca.tree import Tree

SETTINGS = LLMSettings(provider="fake", model="m")

PATCH = """--- a/src/agent.py
+++ b/src/agent.py
@@ -1,2 +1,2 @@
 def answer(q):
-    return client.generate(q, temperature=0.7)
+    return client.generate(q, temperature=0.2)
"""


class FakeJudge:
    """Stands in for pseudorca.llm.judge_idea."""

    def __init__(self, status: str = "present", fail_for: tuple[str, ...] = ()) -> None:
        self.status = status
        self.fail_for = fail_for
        self.calls: list[str] = []
        self.seen: dict[str, list[Finding]] = {}
        self.cache_dirs: list[Path | None] = []

    def __call__(self, idea, patch_text, tree, findings, settings, *, cache_dir=None) -> Finding:
        self.calls.append(idea.id)
        self.seen[idea.id] = findings
        self.cache_dirs.append(cache_dir)
        if idea.id in self.fail_for:
            raise LLMError("endpoint timed out")
        return Finding(
            self.status, "llm", [Evidence("cells/cell_000.py", 1, "def answer(q):")], note="judged"
        )


@pytest.fixture(autouse=True)
def no_real_dependencies(monkeypatch: pytest.MonkeyPatch) -> list[LLMSettings]:
    """The LLM guard allows everything and records its calls; extraction must come from the tests."""
    allowed: list[LLMSettings] = []
    monkeypatch.setattr(pseudorca.llm, "ensure_allowed", allowed.append)

    def no_tree_for(store, nb):
        raise AssertionError("tree_for must not be called here")

    monkeypatch.setattr(pseudorca.extract, "tree_for", no_tree_for)
    return allowed


@pytest.fixture
def nb(store: Store) -> Notebook:
    return register_bytes(store, b'{"cells": []}', name="main.ipynb")[0]


def make_tree(code: str) -> Tree:
    t = Tree("synthetic")
    t.add_text("cells/cell_000.py", code, "cell 0")
    return t


TREE_CODE = "def answer(q):\n    return client.generate(q, temperature=0.2)\nUSE_CACHE = True\n"


def add_idea(store: Store, iid: str, *, fingerprints: tuple[str, ...] = (), patch: str | None = None) -> Idea:
    data = {"id": iid, "title": iid, "status": "adopted", "fingerprints": list(fingerprints)}
    if patch is not None:
        data["patch"] = store.put_blob(patch.encode())
    store.append("idea.add", data)
    return store.state().ideas[iid]


def verdict_events(store: Store) -> list[dict]:
    return [e.data for e in Store.at(store.root).events() if e.type == "audit.verdict"]


@pytest.fixture
def ideas(store: Store) -> list[Idea]:
    """One conclusive idea (fingerprint and patch agree), one partial, one with nothing to check."""
    return [
        add_idea(store, "low-temp", fingerprints=("temperature=0.2",), patch=PATCH),
        add_idea(store, "cache", fingerprints=("USE_CACHE = True", "CACHE_SIZE")),
        add_idea(store, "vague"),
    ]


def test_conclusive_rule():
    assert conclusive(["present"]) and conclusive(["absent", "unknown"])
    assert not conclusive(["present", "absent"])
    assert not conclusive(["present", "partial"])
    assert not conclusive(["unknown"]) and not conclusive([])


def test_deterministic_findings_are_recorded(store: Store, nb: Notebook, ideas: list[Idea]):
    rows = audit_notebook(store, nb, ideas, llm_mode="never", tree=make_tree(TREE_CODE))
    assert [r.idea.id for r in rows] == ["low-temp", "cache", "vague"]
    low, cache, vague = rows
    assert [f.method for f in low.findings] == ["fingerprint", "patch"]
    assert (low.effective.status, low.effective.method) == ("present", "fingerprint+patch")
    assert cache.effective.status == "partial"
    assert (
        vague.findings == []
        and vague.effective is None
        and vague.notes == ["no fingerprints or patch to check"]
    )
    events = verdict_events(store)
    assert [(e["idea"], e["method"], e["status"]) for e in events] == [
        ("low-temp", "fingerprint", "present"),
        ("low-temp", "patch", "present"),
        ("cache", "fingerprint", "partial"),
    ]
    assert events[0]["nb"] == nb.id and events[0]["idea_rev"] == ideas[0].rev
    assert rows.messages == []


@pytest.mark.parametrize(
    ("mode", "asked"),
    [("never", []), ("auto", ["cache", "vague"]), ("always", ["low-temp", "cache", "vague"])],
)
def test_llm_modes(store: Store, nb: Notebook, ideas: list[Idea], mode: str, asked: list[str]):
    judge = FakeJudge()
    rows = audit_notebook(
        store, nb, ideas, llm_mode=mode, settings=SETTINGS, tree=make_tree(TREE_CODE), judge=judge
    )
    assert judge.calls == asked
    llm_events = [e for e in verdict_events(store) if e["method"] == "llm"]
    assert [e["idea"] for e in llm_events] == asked
    assert all(
        e["model"] == "fake:m" and e["detail"]["prompt_version"] == pseudorca.llm.PROMPT_VERSION
        for e in llm_events
    )
    if mode == "auto":
        assert rows[1].effective.status == "present" and rows[1].effective.method == "llm"
        # the judge sees what the deterministic methods found
        assert [f.method for f in judge.seen["cache"]] == ["fingerprint"]


def test_unconfigured_llm_is_reported_once(store: Store, nb: Notebook, ideas: list[Idea]):
    rows = audit_notebook(store, nb, ideas, llm_mode="auto", settings=None, tree=make_tree(TREE_CODE))
    assert [r.llm_note for r in rows] == ["", "LLM not configured", "LLM not configured"]
    assert rows.messages == ["2 ideas are inconclusive and [llm] is not configured"]
    assert len(verdict_events(store)) == 3  # the deterministic results are still recorded
    rows = audit_notebook(store, nb, ideas[:1], llm_mode="always", tree=make_tree(TREE_CODE))
    assert rows.messages == ["[llm] is not configured, so 1 idea did not go to the LLM"]


def test_recorded_methods_are_skipped_unless_forced(store: Store, nb: Notebook, ideas: list[Idea]):
    tree = make_tree(TREE_CODE)
    audit_notebook(store, nb, ideas, llm_mode="never", tree=tree)
    before = len(verdict_events(store))
    rows = audit_notebook(store, nb, ideas, llm_mode="never", tree=tree)
    assert [r.skipped for r in rows] == [["fingerprint", "patch"], ["fingerprint"], []]
    assert all(r.findings == [] for r in rows)
    assert rows[0].effective.status == "present"
    assert len(verdict_events(store)) == before
    rows = audit_notebook(store, nb, ideas, llm_mode="never", tree=tree, force=True)
    assert [len(r.findings) for r in rows] == [2, 1, 0]
    assert len(verdict_events(store)) == before + 3

    # a changed idea is a new rev: its methods run again
    store.append("idea.update", {"id": "cache", "fingerprints": ["USE_CACHE = True"]})
    changed = store.state().ideas["cache"]
    rows = audit_notebook(store, nb, [changed], llm_mode="never", tree=tree)
    assert rows[0].skipped == [] and rows[0].effective.status == "present"


def test_llm_is_skipped_only_for_the_same_model_and_prompt_version(
    store: Store, nb: Notebook, ideas: list[Idea], monkeypatch: pytest.MonkeyPatch
):
    tree = make_tree(TREE_CODE)
    vague = ideas[2:]
    judge = FakeJudge()
    audit_notebook(store, nb, vague, settings=SETTINGS, tree=tree, judge=judge)
    rows = audit_notebook(store, nb, vague, settings=SETTINGS, tree=tree, judge=judge)
    assert judge.calls == ["vague"] and rows[0].skipped == ["llm"]
    assert rows[0].effective.status == "present"
    audit_notebook(
        store, nb, vague, settings=LLMSettings(provider="fake", model="other"), tree=tree, judge=judge
    )
    assert judge.calls == ["vague", "vague"]
    monkeypatch.setattr(pseudorca.llm, "PROMPT_VERSION", "2")
    audit_notebook(store, nb, vague, settings=SETTINGS, tree=tree, judge=judge)
    assert judge.calls == ["vague"] * 3
    audit_notebook(store, nb, vague, settings=SETTINGS, tree=tree, judge=judge, force=True)
    assert judge.calls == ["vague"] * 4


def test_an_older_pseudorca_never_redoes_a_newer_ones_verdicts(
    store: Store, nb: Notebook, ideas: list[Idea], monkeypatch: pytest.MonkeyPatch
):
    """Bob runs a newer extractor and prompt; Alice, on an older Pseudorca, audits after him."""
    low = ideas[:1]
    present, reverted = make_tree(TREE_CODE), make_tree("def answer(q):\n    return 1\n")
    monkeypatch.setattr(pseudorca.extract, "EXTRACTOR_VERSION", 2)
    rows = audit_notebook(store, nb, low, llm_mode="never", tree=present)
    assert rows[0].effective.status == "present"
    monkeypatch.setattr(pseudorca.extract, "EXTRACTOR_VERSION", 1)  # Alice: skips, records nothing
    rows = audit_notebook(store, nb, low, llm_mode="never", tree=reverted)
    assert rows[0].skipped == ["fingerprint", "patch"] and rows[0].findings == []
    assert rows[0].effective.status == "present" and len(verdict_events(store)) == 2
    monkeypatch.setattr(pseudorca.extract, "EXTRACTOR_VERSION", 3)  # an upgrade redoes them
    rows = audit_notebook(store, nb, low, llm_mode="never", tree=reverted)
    assert [f.method for f in rows[0].findings] == ["fingerprint", "patch"]
    assert rows[0].effective.status == "absent"

    vague = ideas[2:]
    judge = FakeJudge()
    monkeypatch.setattr(pseudorca.llm, "PROMPT_VERSION", "10")
    audit_notebook(store, nb, vague, settings=SETTINGS, tree=present, judge=judge)
    monkeypatch.setattr(pseudorca.llm, "PROMPT_VERSION", "9")  # compared as integers, not as text
    rows = audit_notebook(store, nb, vague, settings=SETTINGS, tree=present, judge=judge)
    assert judge.calls == ["vague"] and rows[0].skipped == ["llm"]
    # even with another model: its answer could never count over a newer prompt's
    other = LLMSettings(provider="fake", model="other")
    audit_notebook(store, nb, vague, settings=other, tree=present, judge=judge)
    assert judge.calls == ["vague"]
    monkeypatch.setattr(pseudorca.llm, "PROMPT_VERSION", "10")  # same version, other model: asked
    rows = audit_notebook(store, nb, vague, settings=other, tree=present, judge=judge)
    assert judge.calls == ["vague", "vague"] and rows[0].effective.records[0].model == "fake:other"


def test_an_audit_replays_the_ledger_at_most_once(
    store: Store, nb: Notebook, monkeypatch: pytest.MonkeyPatch
):
    many = [add_idea(store, f"idea-{i:02d}", fingerprints=(f"knob_{i} = 1",), patch=PATCH) for i in range(30)]
    for i in range(2900):  # a ledger of about 3,000 events
        store.append("idea.update", {"id": "idea-00", "owner": f"o{i % 7}"})
    many = [store.state().ideas[i.id] for i in many]
    replays: list[int] = []
    replay = State.replay.__func__

    def counting(cls, events):
        replays.append(1)
        return replay(cls, events)

    monkeypatch.setattr(State, "replay", classmethod(counting))
    code = "".join(f"knob_{i} = 1\n" for i in range(0, 30, 2)) + TREE_CODE
    rows = audit_notebook(store, nb, many, llm_mode="never", tree=make_tree(code))
    assert len(replays) <= 1
    assert len(rows) == 30 and len(verdict_events(store)) == 60
    assert [r.effective.status for r in rows[:2]] == ["present", "conflict"]
    assert rows[1].effective.records[0].event_id  # the records of a real run carry their event ids
    # a dry run replays nothing more either
    rows = audit_notebook(store, nb, many, llm_mode="never", tree=make_tree(code), force=True, dry_run=True)
    assert len(replays) <= 2 and rows[0].effective.status == "present"


def test_an_llm_verdict_settles_a_conflict(store: Store, nb: Notebook):
    conflict = add_idea(store, "conflict", fingerprints=("USE_CACHE",), patch=PATCH)
    reverted = make_tree("def answer(q):\n    return client.generate(q, temperature=0.7)\nUSE_CACHE = 1\n")
    judge = FakeJudge("absent")
    rows = audit_notebook(store, nb, [conflict], settings=SETTINGS, tree=reverted, judge=judge)
    assert judge.calls == ["conflict"]
    eff = rows[0].effective
    assert (eff.status, eff.method) == ("absent", "llm")
    assert [r.method for r in eff.records] == ["llm", "fingerprint", "patch"]


def test_a_patch_with_a_byte_order_mark_is_read(store: Store, nb: Notebook):
    data = {"id": "bom", "title": "bom", "patch": store.put_blob(b"\xef\xbb\xbf" + PATCH.encode())}
    store.append("idea.add", data)
    rows = audit_notebook(
        store, nb, [store.state().ideas["bom"]], llm_mode="never", tree=make_tree(TREE_CODE)
    )
    assert rows[0].effective.status == "present"


def test_conclusive_counts_recorded_deterministic_verdicts(store: Store, nb: Notebook, ideas: list[Idea]):
    tree = make_tree(TREE_CODE)
    audit_notebook(store, nb, ideas[:1], llm_mode="never", tree=tree)
    judge = FakeJudge()
    rows = audit_notebook(store, nb, ideas[:1], settings=SETTINGS, tree=tree, judge=judge)
    assert rows[0].skipped == ["fingerprint", "patch"] and judge.calls == []
    # fingerprint says present, patch says absent: a conflict, so the LLM is asked
    conflict = add_idea(store, "conflict", fingerprints=("USE_CACHE",), patch=PATCH)
    reverted = make_tree("def answer(q):\n    return client.generate(q, temperature=0.7)\nUSE_CACHE = 1\n")
    rows = audit_notebook(store, nb, [conflict], llm_mode="never", tree=reverted)
    assert rows[0].effective.status == "conflict"
    rows = audit_notebook(store, nb, [conflict], settings=SETTINGS, tree=reverted, judge=judge)
    assert judge.calls == ["conflict"] and rows[0].skipped == ["fingerprint", "patch"]
    assert [f.status for f in judge.seen["conflict"]] == ["present", "absent"]


def test_llm_guard_error_disables_the_llm_for_the_run(
    store: Store, nb: Notebook, ideas: list[Idea], monkeypatch: pytest.MonkeyPatch
):
    calls: list[str] = []

    def refuse(settings: LLMSettings) -> None:
        calls.append(settings.label)
        raise LLMError("https://llm.example.com is remote; set allow_remote = true")

    monkeypatch.setattr(pseudorca.llm, "ensure_allowed", refuse)
    judge = FakeJudge()
    rows = audit_notebook(store, nb, ideas, settings=SETTINGS, tree=make_tree(TREE_CODE), judge=judge)
    assert calls == ["fake:m"] and judge.calls == []
    assert rows.messages == [
        "LLM disabled for this run: https://llm.example.com is remote; set allow_remote = true"
    ]
    assert [r.llm_note for r in rows] == ["", "LLM not allowed", "LLM not allowed"]
    assert len(verdict_events(store)) == 3


def test_guard_runs_once_and_only_before_a_call(
    store: Store, nb: Notebook, ideas: list[Idea], no_real_dependencies: list[LLMSettings]
):
    tree = make_tree(TREE_CODE)
    audit_notebook(store, nb, ideas[:1], settings=SETTINGS, tree=tree, judge=FakeJudge())
    assert no_real_dependencies == []  # conclusive: no LLM call, so no guard either
    audit_notebook(store, nb, ideas, settings=SETTINGS, tree=tree, judge=FakeJudge())
    assert no_real_dependencies == [SETTINGS]


def test_a_failed_llm_call_is_kept_on_its_row(store: Store, nb: Notebook, ideas: list[Idea]):
    judge = FakeJudge(fail_for=("cache",))
    rows = audit_notebook(store, nb, ideas, settings=SETTINGS, tree=make_tree(TREE_CODE), judge=judge)
    assert judge.calls == ["cache", "vague"]
    assert rows[1].llm_error == "endpoint timed out" and rows[1].effective.status == "partial"
    assert [f.method for f in rows[2].findings] == ["llm"] and rows[2].llm_error == ""

    class Odd(FakeJudge):
        def __call__(self, *args, **kwargs) -> Finding:
            return Finding("maybe", "llm")

    rows = audit_notebook(store, nb, ideas[1:2], settings=SETTINGS, tree=make_tree(TREE_CODE), judge=Odd())
    assert "invalid status" in rows[0].llm_error and rows[0].findings == []


def test_engine_sets_the_keys_that_make_llm_verdicts_skippable(store: Store, nb: Notebook, ideas: list[Idea]):
    class Sloppy(FakeJudge):
        def __call__(self, *args, **kwargs) -> Finding:
            super().__call__(*args, **kwargs)
            return Finding(
                "absent", "judge", model="resolved-name", detail={"prompt_version": "old", "steps": 3}
            )

    judge = Sloppy()
    tree = make_tree(TREE_CODE)
    audit_notebook(store, nb, ideas[2:], settings=SETTINGS, tree=tree, judge=judge)
    (event,) = [e for e in verdict_events(store) if e["method"] == "llm"]
    assert event["model"] == "fake:m"
    assert event["detail"] == {"prompt_version": pseudorca.llm.PROMPT_VERSION, "steps": 3}
    audit_notebook(store, nb, ideas[2:], settings=SETTINGS, tree=tree, judge=judge)
    assert judge.calls == ["vague"]


def test_dry_run_writes_nothing(store: Store, nb: Notebook, ideas: list[Idea]):
    judge = FakeJudge()
    rows = audit_notebook(
        store, nb, ideas, settings=SETTINGS, tree=make_tree(TREE_CODE), judge=judge, dry_run=True
    )
    assert verdict_events(store) == []
    assert not (store.root / "cache" / "llm").exists() and judge.cache_dirs == [None, None]
    assert [len(r.findings) for r in rows] == [2, 2, 1]
    assert [r.effective.status for r in rows] == ["present", "present", "present"]
    assert rows[2].effective.method == "llm"
    # a real run passes a transcript directory
    audit_notebook(store, nb, ideas[2:], settings=SETTINGS, tree=make_tree(TREE_CODE), judge=judge)
    assert judge.cache_dirs[-1] == store.root / "cache" / "llm"


def test_dry_run_effective_includes_recorded_verdicts(store: Store, nb: Notebook, ideas: list[Idea]):
    store.append(
        "audit.verdict",
        {"nb": nb.id, "idea": "cache", "idea_rev": ideas[1].rev, "status": "absent", "method": "human"},
    )
    rows = audit_notebook(store, nb, ideas[1:2], llm_mode="never", tree=make_tree(TREE_CODE), dry_run=True)
    assert rows[0].findings[0].status == "partial"
    assert (rows[0].effective.status, rows[0].effective.method) == ("absent", "human")


def test_tree_is_extracted_lazily_and_once(
    store: Store, nb: Notebook, ideas: list[Idea], monkeypatch: pytest.MonkeyPatch
):
    loads: list[str] = []

    def tree_for(store_: Store, nb_: Notebook) -> Tree:
        loads.append(nb_.id)
        return make_tree(TREE_CODE)

    monkeypatch.setattr(pseudorca.extract, "tree_for", tree_for)
    audit_notebook(store, nb, ideas, llm_mode="never")
    assert loads == [nb.id]
    audit_notebook(store, nb, ideas, llm_mode="never")  # everything is recorded: nothing to extract
    assert loads == [nb.id]


def test_missing_patch_blob_is_a_note_not_a_verdict(store: Store, nb: Notebook):
    store.append(
        "idea.add", {"id": "lost", "title": "t", "fingerprints": ["temperature=0.2"], "patch": "0" * 64}
    )
    idea = store.state().ideas["lost"]
    rows = audit_notebook(store, nb, [idea], llm_mode="never", tree=make_tree(TREE_CODE))
    assert [f.method for f in rows[0].findings] == ["fingerprint"]
    assert rows[0].notes and rows[0].notes[0].startswith("patch unavailable:")


def test_bad_llm_mode_is_rejected(store: Store, nb: Notebook):
    with pytest.raises(ValueError, match="llm mode"):
        audit_notebook(store, nb, [], llm_mode="sometimes")
    assert engine.LLM_MODES == ("auto", "always", "never")
