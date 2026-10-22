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
from pseudorca.model import State
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
    # asked for on the command line: a one-line error (check could not run), before anything is registered
    path = tmp_path / "new.ipynb"
    path.write_bytes(BAD)
    result = run(ledger, "check", str(path), "--llm", "always", code=2)
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


# ----- check: configuration and exit codes ---------------------------------------------------------


def set_config(store: Store, old: str, new: str) -> None:
    config = store.root / "config.toml"
    text = config.read_text(encoding="utf-8")
    assert old in text
    config.write_text(text.replace(old, new), encoding="utf-8")


def test_check_rejects_statuses_that_are_not_idea_statuses(ledger: Store):
    register_bytes(ledger, BAD, name="bad.ipynb")
    set_config(ledger, 'expect_status = ["adopted"]', 'expect_status = ["adopt"]')
    out = run(ledger, "check", "bad.ipynb", "--llm", "never", code=2).output
    assert out.strip() == (
        "Error: [audit] expect_status: 'adopt' is not an idea status (idea, testing, adopted, rejected, parked)"
    )
    set_config(ledger, 'expect_status = ["adopt"]', 'expect_status = ["adopted"]')
    set_config(ledger, 'forbid_status = ["rejected"]', 'forbid_status = "reject"')
    out = run(ledger, "check", "bad.ipynb", "--llm", "never", code=2).output
    assert "[audit] forbid_status: 'reject' is not an idea status" in out
    assert verdicts(ledger) == []  # nothing was audited


def test_check_with_nothing_expected_names_the_statuses(ledger: Store):
    register_bytes(ledger, GOOD, name="good.ipynb")
    set_config(ledger, 'expect_status = ["adopted"]', 'expect_status = ["idea"]')
    out = run(ledger, "check", "good.ipynb", "--llm", "never").stdout
    assert "check passed (nothing is expected: no idea has status idea; pass --expect ID" in out
    assert "OK" not in out
    set_config(ledger, 'expect_status = ["idea"]', "expect_status = []")
    out = run(ledger, "check", "good.ipynb", "--llm", "never").stdout
    assert "nothing is expected: [audit] expect_status is empty;" in out


def test_check_exits_2_when_it_cannot_run(ledger: Store, tmp_path: Path):
    register_bytes(ledger, GOOD, name="good.ipynb")
    assert "no notebook matches 'nope.ipynb'" in run(ledger, "check", "nope.ipynb", code=2).output
    broken = tmp_path / "broken.ipynb"
    broken.write_bytes(b"version https://git-lfs.github.com/spec/v1\n")
    for args in ([str(broken)], [str(broken), "--dry-run"]):
        out = run(ledger, "check", *args, "--llm", "never", code=2).output
        assert out.startswith("Error: ") and "Traceback" not in out
    assert "no idea matches 'nope'" in run(ledger, "check", "good.ipynb", "--expect", "nope", code=2).output
    assert "Missing argument" in run(ledger, "check", code=2).output  # a usage error
    assert "is not one of" in run(ledger, "check", "good.ipynb", "--llm", "sometimes", code=2).output
    elsewhere = tmp_path / "no-store"
    elsewhere.mkdir()
    result = CliRunner().invoke(main, ["--store", str(elsewhere), "check", "good.ipynb"])
    assert result.exit_code == 2 and "is not a Pseudorca store" in result.output
    # a failed check is 1, a passed one 0
    run(ledger, "check", "good.ipynb", "--llm", "never", "--expect", "kv-cache", "--strict", code=1)
    run(ledger, "check", "good.ipynb", "--llm", "never")
    help_text = " ".join(run(ledger, "check", "--help").stdout.split())
    assert "0 the check passed 1 the check failed" in help_text
    assert "2 the check could not run: any error, including a usage error" in help_text


def test_check_exits_2_on_an_unexpected_error(ledger: Store, monkeypatch: pytest.MonkeyPatch):
    register_bytes(ledger, GOOD, name="good.ipynb")
    monkeypatch.delenv("PSEUDORCA_DEBUG", raising=False)

    def broken(*args, **kwargs):
        raise KeyError("bug")

    monkeypatch.setattr("pseudorca.commands.audit.audit_notebook", broken)
    result = run(ledger, "check", "good.ipynb", "--llm", "never", code=2)
    assert "check could not run: KeyError" in result.output and "Traceback" in result.stderr


# ----- audit of a file ---------------------------------------------------------------------------


def test_audit_of_a_file_audits_that_files_version(
    ledger: Store, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    Path("agent.ipynb").write_bytes(BAD)
    register_bytes(ledger, BAD, name="agent.ipynb")
    Path("agent.ipynb").write_bytes(GOOD)  # edited after it was registered
    result = run(ledger, "audit", "agent.ipynb", "--llm", "never")
    assert "agent.ipynb on disk is not the registered version" in result.stderr
    assert "pseudorca nb add agent.ipynb" in result.stderr and "pseudorca check agent.ipynb" in result.stderr
    assert f"audit of agent.ipynb ({util.short(util.sha256_bytes(BAD))})" in result.stdout
    # once the file's version is registered, under any name, that version is audited
    register_bytes(ledger, GOOD, name="agent-v2.ipynb")
    result = run(ledger, "audit", "agent.ipynb", "--llm", "never")
    assert "on disk" not in result.stderr and "audit of agent-v2.ipynb" in result.stdout
    Path("other.ipynb").write_bytes(ipynb("x = 1\n"))
    out = run(ledger, "audit", "other.ipynb", code=1).output
    assert "other.ipynb is not registered" in out and "pseudorca check other.ipynb" in out


# ----- what the tables show ----------------------------------------------------------------------


def test_the_evidence_column_shows_the_matched_part_of_a_long_line(ledger: Store):
    long_prompt = (
        'PROMPT = "' + "You are a careful assistant who explains each step. " * 3 + 'Think briefly."\n'
    )
    japanese = (
        'PROMPT_JA = "あなたは数学の文章題を解きます。答えは数字だけで書きます。手順を短く考えてください"\n'
    )
    register_bytes(ledger, ipynb(long_prompt, japanese), name="long.ipynb")
    for iid, fp in (("brief", "Think briefly."), ("brief-ja", "手順を短く考えてください")):
        ledger.append("idea.add", {"id": iid, "title": iid, "status": "adopted", "fingerprints": [fp]})
    out = run(ledger, "audit", "--idea", "brief", "--idea", "brief-ja", "--llm", "never").stdout
    assert "cells/cell_001.py:1 …" in table_row(out, "brief") and "Think briefly." in table_row(out, "brief")
    assert "手順を短く" in table_row(out, "brief-ja")
    assert all(util.display_width(line) <= 160 for line in out.splitlines())
    out = run(ledger, "check", "long.ipynb", "--llm", "never", code=1).stdout
    ok = next(s for s in out.split("\n\n") if s.startswith("OK"))
    assert "Think briefly." in ok and "手順を短く" in ok


def configure_llm(monkeypatch: pytest.MonkeyPatch, finding: Finding) -> list[str]:
    asked: list[str] = []

    def load_settings(store: Store, **overrides) -> LLMSettings:
        return LLMSettings(provider="fake", model="m")

    def judge_idea(idea, patch_text, tree, findings, settings, *, cache_dir=None) -> Finding:
        asked.append(idea.id)
        return finding

    monkeypatch.setattr(pseudorca.llm, "load_settings", load_settings)
    monkeypatch.setattr(pseudorca.llm, "judge_idea", judge_idea)
    return asked


CONFLICT_PATCH = "--- a/solver.py\n+++ b/solver.py\n@@ -1,2 +1,2 @@\n max_retries = 3\n-temperature=0.2\n+temperature=0.0\n"


def test_an_llm_verdict_that_settles_a_conflict_is_shown_with_its_quote(
    ledger: Store, monkeypatch: pytest.MonkeyPatch
):
    register_bytes(ledger, GOOD, name="good.ipynb")
    data = {"id": "zero-temp", "title": "t", "status": "adopted", "fingerprints": ["max_retries = 3"]}
    ledger.append("idea.add", {**data, "patch": ledger.put_blob(CONFLICT_PATCH.encode())})
    quote = Evidence("cells/cell_002.py", 2, "temperature=0.2")
    asked = configure_llm(monkeypatch, Finding("present", "llm", [quote], note="seen"))
    out = run(ledger, "audit", "--idea", "zero-temp").stdout
    assert asked == ["zero-temp"]
    assert table_row(out, "zero-temp").split()[2:5] == ["✓", "present", "llm"]
    assert "cells/cell_002.py:2 temperature=0.2" in table_row(out, "zero-temp")
    out = run(ledger, "check", "good.ipynb").stdout
    ok = next(s for s in out.split("\n\n") if s.startswith("OK"))
    assert "zero-temp" in ok and "cells/cell_002.py:2 temperature=0.2" in ok


def test_an_llm_absent_without_a_note_does_not_show_a_contradicting_note(
    ledger: Store, monkeypatch: pytest.MonkeyPatch
):
    register_bytes(ledger, GOOD, name="good.ipynb")
    data = {"id": "zero-temp", "title": "t", "status": "testing", "fingerprints": ["max_retries = 3"]}
    ledger.append("idea.add", {**data, "patch": ledger.put_blob(CONFLICT_PATCH.encode())})
    configure_llm(monkeypatch, Finding("absent", "llm", note=""))
    row = table_row(run(ledger, "audit", "--idea", "zero-temp").stdout, "zero-temp")
    assert row.split()[2:5] == ["✗", "absent", "llm"]
    assert "found 1/1" not in row and "0/1 hunks applied, 1 reverted" in row


# ----- JSON ----------------------------------------------------------------------------------------


def test_audit_json(ledger: Store):
    nb = register_bytes(ledger, GOOD, name="good.ipynb")[0]
    result = run(ledger, "audit", "--llm", "never", "--json")
    data = json.loads(result.stdout)
    assert data["notebook"]["id"] == nb.id and data["dry_run"] is False
    by_id = {i["idea"]: i for i in data["ideas"]}
    assert list(by_id) == ["kv-cache", "retry-loop", "short-prompt", "hot-sampling"]
    short = by_id["short-prompt"]
    assert (short["idea_status"], short["claimed"], short["verdict"]["status"]) == (
        "adopted",
        False,
        "present",
    )
    assert short["new"][0]["method"] == "fingerprint" and short["new"][0]["evidence"][0]["col"] == 0
    assert by_id["kv-cache"]["verdict"] is None
    assert by_id["kv-cache"]["notes"] == ["no fingerprints or patch to check"]
    assert data["counts"] == {"present": 2, "absent": 1, "none": 1} and data["messages"] == []
    again = json.loads(run(ledger, "audit", "--llm", "never", "--json").stdout)
    again_short = next(i for i in again["ideas"] if i["idea"] == "short-prompt")
    assert again_short["skipped"] == ["fingerprint"] and again_short["new"] == []


def test_check_json(ledger: Store):
    register_bytes(ledger, BAD, name="bad.ipynb", applies=["retry-loop"])
    data = json.loads(run(ledger, "check", "bad.ipynb", "--llm", "never", "--json", code=1).stdout)
    assert data["passed"] is False and data["problems"] == ["1 missing", "1 forbidden present"]
    assert data["sections"] == {
        "missing": ["retry-loop"],
        "unknown": [],
        "forbidden_present": ["hot-sampling"],
        "claimed_but_not_found": ["retry-loop"],
        "ok": ["short-prompt"],
    }
    assert (data["expected"], data["forbidden"]) == (["retry-loop", "short-prompt"], ["hot-sampling"])
    verdict = {i["idea"]: i["verdict"] for i in data["ideas"]}
    assert verdict["hot-sampling"]["status"] == "present" and verdict["retry-loop"]["status"] == "absent"
    register_bytes(ledger, GOOD, name="good.ipynb")
    data = json.loads(run(ledger, "check", "good.ipynb", "--llm", "never", "--json").stdout)
    assert data["passed"] is True and data["problems"] == [] and data["notebook"]["name"] == "good.ipynb"
    assert data["registration"] == "" and data["strict"] is False and data["dry_run"] is False


# ----- speed ---------------------------------------------------------------------------------------


def test_audit_and_check_replay_the_ledger_a_few_times_not_once_per_idea(
    ledger: Store, monkeypatch: pytest.MonkeyPatch
):
    for i in range(30):
        ledger.append(
            "idea.add", {"id": f"knob-{i}", "title": "t", "status": "adopted", "fingerprints": [f"k{i} = 1"]}
        )
    register_bytes(ledger, GOOD, name="good.ipynb")
    replays: list[int] = []
    replay = State.replay.__func__

    def counting(cls, events):
        replays.append(1)
        return replay(cls, events)

    monkeypatch.setattr(State, "replay", classmethod(counting))
    run(ledger, "audit", "--llm", "never")
    assert len(verdicts(ledger)) == 33 and len(replays) <= 2
    replays.clear()
    run(ledger, "check", "good.ipynb", "--llm", "never", code=1)  # the knobs are missing
    assert len(replays) <= 3
