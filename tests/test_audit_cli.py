from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner, Result

import pseudorca.extract
import pseudorca.llm
from pseudorca import util
from pseudorca.audit.types import Evidence, Finding
from pseudorca.cli import main
from pseudorca.commands._common import LEGEND
from pseudorca.llm import LLMSettings
from pseudorca.notebooks import register_bytes
from pseudorca.store import Store
from pseudorca.tree import Tree


def ipynb(*code: str) -> bytes:
    cells = [{"cell_type": "markdown", "metadata": {}, "source": "Claims: max_retries = 3"}]
    cells += [
        {"cell_type": "code", "metadata": {}, "source": c, "outputs": [], "execution_count": None}
        for c in code
    ]
    return json.dumps({"nbformat": 4, "nbformat_minor": 5, "metadata": {}, "cells": cells}).encode()


GOOD = ipynb('SYSTEM = "Be brief."\n', "max_retries = 3\ntemperature=0.2\n")
BAD = ipynb('SYSTEM = "Be brief."\n', "temperature=1.2\n")


def fake_extract(data: bytes, name: str = "notebook.ipynb", limits=None) -> Tree:
    """A minimal stand-in for the extractor: code and markdown cells of a JSON notebook, as text."""
    tree = Tree(util.sha256_bytes(data))
    for i, cell in enumerate(json.loads(data)["cells"]):
        folder, ext = ("cells", "py") if cell["cell_type"] == "code" else ("markdown", "md")
        tree.add_text(f"{folder}/cell_{i:03d}.{ext}", cell["source"], f"cell {i}")
    return tree


class Calls:
    def __init__(self) -> None:
        self.extract_bytes = 0
        self.tree_for = 0
        self.load_settings: list[dict] = []


@pytest.fixture(autouse=True)
def fakes(monkeypatch: pytest.MonkeyPatch) -> Calls:
    """No real extractor or LLM: [llm] is unconfigured unless a test says otherwise."""
    calls = Calls()

    def extract_bytes(data: bytes, name: str = "notebook.ipynb", limits=None) -> Tree:
        calls.extract_bytes += 1
        return fake_extract(data, name)

    def tree_for(store: Store, nb) -> Tree:
        calls.tree_for += 1
        return fake_extract(store.get_blob(nb.id), nb.name)

    def load_settings(store: Store, **overrides) -> LLMSettings | None:
        calls.load_settings.append(overrides)
        return None

    def judge_idea(*args, **kwargs) -> Finding:
        raise AssertionError("the LLM must not be called")

    monkeypatch.setattr(pseudorca.extract, "extract_bytes", extract_bytes)
    monkeypatch.setattr(pseudorca.extract, "tree_for", tree_for)
    monkeypatch.setattr(pseudorca.llm, "load_settings", load_settings)
    monkeypatch.setattr(pseudorca.llm, "ensure_allowed", lambda settings: None)
    monkeypatch.setattr(pseudorca.llm, "judge_idea", judge_idea)
    return calls


@pytest.fixture
def ledger(store: Store) -> Store:
    for iid, status, fps in [
        ("short-prompt", "adopted", ['SYSTEM = "Be brief."']),
        ("retry-loop", "adopted", [r"re:max_retries\s*=\s*3"]),
        ("hot-sampling", "rejected", ["temperature=1.2"]),
        ("kv-cache", "testing", []),
        ("old-trick", "parked", ["old_trick()"]),
    ]:
        store.append("idea.add", {"id": iid, "title": iid, "status": status, "fingerprints": fps})
    return store


def run(store: Store, *args: str, code: int = 0) -> Result:
    result = CliRunner().invoke(main, ["--store", str(store.root), *args])
    assert result.exit_code == code, result.output + repr(result.exception)
    return result


def count_files(store: Store, folder: str) -> int:
    return sum(1 for p in (store.root / folder).rglob("*") if p.is_file() and p.name != ".gitkeep")


def verdicts(store: Store) -> list[dict]:
    return [e.data for e in Store.at(store.root).events() if e.type == "audit.verdict"]


def table_row(output: str, idea_id: str) -> str:
    return next(line for line in output.splitlines() if line.strip().startswith(idea_id + " "))


def test_audit_prints_a_table_counts_and_the_legend(ledger: Store, fakes: Calls):
    register_bytes(ledger, GOOD, name="good.ipynb")
    out = run(ledger, "audit", "--llm", "never").stdout
    assert "audit of good.ipynb" in out
    header = next(line for line in out.splitlines() if line.startswith("IDEA"))
    assert header.split() == ["IDEA", "STATUS", "VERDICT", "METHOD", "EVIDENCE/NOTE"]
    assert table_row(out, "short-prompt").split()[:5] == [
        "short-prompt",
        "adopted",
        "✓",
        "present",
        "fingerprint",
    ]
    assert 'cells/cell_001.py:1 SYSTEM = "Be brief."' in table_row(out, "short-prompt")
    assert table_row(out, "hot-sampling").split()[:4] == ["hot-sampling", "rejected", "✗", "absent"]
    assert "no fingerprints or patch" in table_row(out, "kv-cache")
    assert "old-trick" not in out  # parked ideas are left out by default
    assert "2 present, 1 absent, 1 none" in out and "3 new verdicts recorded" in out
    assert out.rstrip().endswith(LEGEND)
    assert len(verdicts(ledger)) == 3 and fakes.load_settings == []

    again = run(ledger, "audit", "good.ipynb", "--llm", "never").stdout
    assert "0 new verdicts recorded, 3 skipped as already recorded" in again


def test_audit_selection_dry_run_and_unconfigured_llm(ledger: Store, fakes: Calls):
    register_bytes(ledger, BAD, name="bad.ipynb")
    result = run(ledger, "audit", "--dry-run")
    assert "dry run: nothing recorded" in result.stdout and verdicts(ledger) == []
    assert "1 idea is inconclusive and [llm] is not configured" in result.stderr
    assert fakes.load_settings == [{}]
    assert "LLM not configured" in table_row(result.stdout, "kv-cache")

    out = run(ledger, "audit", "--idea", "retry", "--idea", "old-trick", "--llm", "never").stdout
    assert [line.split()[0] for line in out.splitlines()[4:6]] == ["retry-loop", "old-trick"]
    out = run(ledger, "audit", "--status", "rejected", "--llm", "never").stdout
    assert "hot-sampling" in out and "short-prompt" not in out
    assert "no idea matches 'nope'" in run(ledger, "audit", "--idea", "nope", code=1).output
    assert "no notebook matches" in run(ledger, "audit", "missing.ipynb", code=1).output


def test_audit_asks_a_configured_llm(ledger: Store, fakes: Calls, monkeypatch: pytest.MonkeyPatch):
    register_bytes(ledger, GOOD, name="good.ipynb")
    seen: list[dict] = []

    def load_settings(store: Store, **overrides) -> LLMSettings:
        seen.append(overrides)
        return LLMSettings(provider="fake", model=overrides.get("model", "m"))

    def judge_idea(idea, patch_text, tree, findings, settings, *, cache_dir=None) -> Finding:
        return Finding("present", "llm", [Evidence("cells/cell_002.py", 2, "temperature=0.2")], note="seen")

    monkeypatch.setattr(pseudorca.llm, "load_settings", load_settings)
    monkeypatch.setattr(pseudorca.llm, "judge_idea", judge_idea)
    result = run(ledger, "audit", "--idea", "kv-cache", "--model", "m2", "--allow-remote")
    assert seen == [{"model": "m2", "allow_remote": True}]
    assert table_row(result.stdout, "kv-cache").split()[:5] == ["kv-cache", "testing", "✓", "present", "llm"]
    assert "kv-cache: asking fake:m2" in result.stderr
    assert verdicts(ledger)[-1]["model"] == "fake:m2"


def test_broken_llm_settings(ledger: Store, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    register_bytes(ledger, GOOD, name="good.ipynb")

    def load_settings(store: Store, **overrides) -> LLMSettings:
        raise pseudorca.llm.LLMError("unknown provider 'nope' in [llm]")

    monkeypatch.setattr(pseudorca.llm, "load_settings", load_settings)
    # only the config asks for the LLM: warn and audit without it
    result = run(ledger, "audit")
    assert "LLM not used: unknown provider 'nope' in [llm]" in result.stderr
    assert "✓ present" in table_row(result.stdout, "short-prompt")
    # asked for on the command line: a one-line error, before anything is registered
    path = tmp_path / "new.ipynb"
    path.write_bytes(BAD)
    result = run(ledger, "check", str(path), "--llm", "always", code=1)
    assert "cannot use the LLM: unknown provider" in result.output and "Traceback" not in result.output
    assert len(Store.at(ledger.root).state().notebooks) == 1


def test_check_exit_codes(ledger: Store):
    register_bytes(ledger, GOOD, name="good.ipynb")
    register_bytes(ledger, BAD, name="bad.ipynb")
    out = run(ledger, "check", "good.ipynb", "--llm", "never").stdout
    assert "OK (2)" in out and "MISSING" not in out and "FORBIDDEN PRESENT" not in out
    assert out.rstrip().splitlines()[-1] == "check passed"

    out = run(ledger, "check", "bad.ipynb", "--llm", "never", code=1).stdout
    sections = out.split("\n\n")
    missing = next(s for s in sections if s.startswith("MISSING"))
    forbidden = next(s for s in sections if s.startswith("FORBIDDEN PRESENT"))
    assert "retry-loop" in missing and "✗ absent" in missing
    assert "hot-sampling" in forbidden and "cells/cell_002.py:1 temperature=1.2" in forbidden
    assert out.rstrip().splitlines()[-1] == "check failed: 1 missing, 1 forbidden present"


def test_check_strict_fails_on_unknown(ledger: Store):
    register_bytes(ledger, GOOD, name="good.ipynb")
    out = run(ledger, "check", "good.ipynb", "--expect", "kv-cache").stdout
    assert "UNKNOWN (1)" in out and "kv-cache" in out
    assert out.rstrip().splitlines()[-1].startswith("check passed")
    out = run(ledger, "check", "good.ipynb", "--expect", "kv-cache", "--strict", code=1).stdout
    assert out.rstrip().splitlines()[-1] == "check failed: 1 unknown (--strict)"


def test_check_dry_run_on_a_file_writes_nothing(ledger: Store, fakes: Calls, tmp_path: Path):
    path = tmp_path / "candidate.ipynb"
    path.write_bytes(BAD)
    events, blobs = count_files(ledger, "events"), count_files(ledger, "blobs")
    out = run(ledger, "check", str(path), "--dry-run", "--llm", "never", code=1).stdout
    assert "check of candidate.ipynb" in out and "dry run" in out
    assert "MISSING (1)" in out and "FORBIDDEN PRESENT (1)" in out
    assert (count_files(ledger, "events"), count_files(ledger, "blobs")) == (events, blobs)
    assert (fakes.extract_bytes, fakes.tree_for) == (1, 0)
    assert Store.at(ledger.root).state().notebooks == {}


def test_check_registers_a_file(ledger: Store, tmp_path: Path):
    path = tmp_path / "candidate.ipynb"
    path.write_bytes(GOOD)
    out = run(ledger, "check", str(path), "--name", "cand-1", "--tag", "candidate", "--llm", "never").stdout
    assert "check of cand-1" in out and "newly registered" in out
    nb = Store.at(ledger.root).state().resolve_nb("latest:candidate")
    assert nb.name == "cand-1" and nb.id == util.sha256_bytes(GOOD)
    assert {v["nb"] for v in verdicts(ledger)} == {nb.id}
    assert "already registered" in run(ledger, "check", str(path), "--llm", "never").stdout


def test_check_lists_claimed_ideas_that_are_absent(ledger: Store):
    register_bytes(ledger, BAD, name="bad.ipynb", applies=["retry-loop", "short-prompt"])
    out = run(ledger, "check", "bad.ipynb", "--llm", "never", code=1).stdout
    claimed = next(s for s in out.split("\n\n") if s.startswith("CLAIMED BUT NOT FOUND"))
    assert "retry-loop" in claimed and "short-prompt" not in claimed
    assert "found 0/1 (missing: re:max_retries" in claimed


def test_verdict_records_a_human_verdict(ledger: Store):
    nb = register_bytes(ledger, GOOD, name="good.ipynb")[0]
    out = run(
        ledger,
        "verdict",
        "good.ipynb",
        "kv-cache",
        "absent",
        "--note",
        "checked by hand",
        "--evidence",
        "cells/cell_002.py:1:max_retries = 3",
        "--evidence",
        "files/run.sh::python main.py --no-cache",
    ).stdout
    assert out.strip() == "recorded: kv-cache in good.ipynb is ✗ absent (human verdict)"
    (v,) = verdicts(ledger)
    assert (v["method"], v["status"], v["note"], v["nb"]) == ("human", "absent", "checked by hand", nb.id)
    assert v["evidence"] == [
        {"path": "cells/cell_002.py", "line": 1, "quote": "max_retries = 3"},
        {"path": "files/run.sh", "line": None, "quote": "python main.py --no-cache"},
    ]
    eff = Store.at(ledger.root).state().effective_verdict(nb.id, "kv-cache")
    assert (eff.status, eff.method) == ("absent", "human")
    # the human verdict wins over a later machine audit
    out = run(ledger, "audit", "--idea", "kv-cache", "--llm", "never").stdout
    assert table_row(out, "kv-cache").split()[2:5] == ["✗", "absent", "human"]


def test_verdict_rejects_bad_input(ledger: Store):
    register_bytes(ledger, GOOD, name="good.ipynb")
    assert "'maybe' is not one of" in run(ledger, "verdict", "good.ipynb", "kv-cache", "maybe", code=2).output
    out = run(
        ledger, "verdict", "good.ipynb", "kv-cache", "present", "--evidence", "no-line-here", code=1
    ).output
    assert "is not PATH:LINE:QUOTE" in out and "Traceback" not in out
    assert "no idea matches" in run(ledger, "verdict", "good.ipynb", "nope", "present", code=1).output
    assert verdicts(ledger) == []
