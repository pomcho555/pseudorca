from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from collections.abc import Callable
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
from pseudorca.notebooks import register_bytes, register_file
from pseudorca.store import Store, StoreError
from pseudorca.tree import Tree

TREE_FOR = pseudorca.extract.tree_for  # the real ones, before the `fakes` fixture replaces them
EXTRACT_BYTES = pseudorca.extract.extract_bytes


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


def test_audit_of_a_file_whose_suffix_is_a_label_audits_that_files_version(
    ledger: Store, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Any existing file is a file, as in `check`: `solver-v1.5` is not only a notebook name."""
    monkeypatch.chdir(tmp_path)
    Path("solver-v1.5").write_bytes(BAD)
    register_bytes(ledger, BAD, name="solver-v1.5")
    Path("solver-v1.5").write_bytes(GOOD)  # edited after it was registered
    result = run(ledger, "audit", "solver-v1.5", "--llm", "never")
    assert "solver-v1.5 on disk is not the registered version" in result.stderr
    assert "pseudorca nb add solver-v1.5" in result.stderr
    register_bytes(ledger, GOOD, name="solver-next")
    result = run(ledger, "audit", "solver-v1.5", "--llm", "never")
    assert "on disk" not in result.stderr and "audit of solver-next" in result.stdout
    # the default NB_REF is the latest registered notebook, whatever files the directory holds
    Path("latest").write_bytes(ipynb("x = 1\n"))
    result = run(ledger, "audit", "--llm", "never")
    assert "audit of solver-next" in result.stdout and "on disk" not in result.stderr


# ----- check into a pipe that closes early -------------------------------------------------------


def run_into_closed_pipe(store: Store, *args: str) -> subprocess.CompletedProcess[bytes]:
    """Run pseudorca in a child process whose stdout is a pipe nobody reads (like `| head -c 0`)."""
    read_end, write_end = os.pipe()
    os.close(read_end)
    try:
        return subprocess.run(
            [sys.executable, "-m", "pseudorca", "--store", str(store.root), *args],
            stdout=write_end,
            stderr=subprocess.PIPE,
            timeout=120,
            check=False,
        )
    finally:
        os.close(write_end)


def test_check_keeps_its_exit_code_when_stdout_is_closed_early(ledger: Store, tmp_path: Path):
    for i in range(40):
        ledger.append(
            "idea.add", {"id": f"knob-{i}", "title": "t", "status": "adopted", "fingerprints": [f"k{i} = 1"]}
        )
    bad, good = tmp_path / "bad.py", tmp_path / "good.py"
    bad.write_text("temperature=1.2\n", encoding="utf-8")
    good.write_text(
        "".join(f"k{i} = 1\n" for i in range(40)) + 'SYSTEM = "Be brief."\nmax_retries = 3\n',
        encoding="utf-8",
    )
    for args, code in (
        ([str(bad)], 1),
        ([str(bad), "--json"], 1),
        ([str(good)], 0),
        ([str(good), "--json"], 0),
    ):
        result = run_into_closed_pipe(ledger, "check", *args, "--llm", "never", "--dry-run")
        assert result.returncode == code, (args, result.returncode, result.stderr.decode()[-2000:])
        assert b"Traceback" not in result.stderr


def test_check_that_loses_its_output_before_the_outcome_is_known_could_not_run(
    ledger: Store, monkeypatch: pytest.MonkeyPatch
):
    register_bytes(ledger, BAD, name="bad.ipynb")

    def closed(store: Store) -> None:  # stderr piped into `head`, which has exited
        raise BrokenPipeError(32, "Broken pipe")

    monkeypatch.setattr("pseudorca.commands.audit.show_warnings", closed)
    run(ledger, "check", "bad.ipynb", "--llm", "never", code=2)  # not 0: nothing was checked
    assert verdicts(ledger) == []


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


# ----- audit --submitted ---------------------------------------------------------------------------


def submit(store: Store, sub_id: str, nb_id: str | None, day: int) -> None:
    """A scored submission, as `sub add` or `kaggle sync` records it, made from `nb_id` (None: not linked)."""
    data = {"id": sub_id, "date": f"2026-01-{day:02d}", "public_score": 0.5}
    store.append("sub.upsert", data if nb_id is None else {**data, "nb": nb_id})


INCONCLUSIVE = (
    "1 idea is inconclusive and [llm] is not configured (set [llm] model in config.toml or pass --model)"
)


def test_audit_submitted_audits_each_submitted_notebook_once(ledger: Store, fakes: Calls):
    good = register_bytes(ledger, GOOD, name="good.ipynb")[0]
    bad = register_bytes(ledger, BAD, name="bad.ipynb")[0]
    register_bytes(ledger, ipynb("x = 1\n"), name="never-submitted.ipynb")
    submit(ledger, "12", good.id, day=2)
    submit(ledger, "11", bad.id, day=1)
    submit(ledger, "13", good.id, day=3)  # a resubmission
    result = run(ledger, "audit", "--submitted")
    lines = result.stdout.splitlines()
    assert lines == [
        "2 submitted notebooks to audit, oldest submission first",
        "",
        f"bad.ipynb   {util.short(bad.id)}  2 present, 1 absent, 1 none; 3 new verdicts",
        f"good.ipynb  {util.short(good.id)}  2 present, 1 absent, 1 none; 3 new verdicts",
        "",
        "2 notebooks: 4 present, 2 absent, 2 none; 6 new verdicts recorded",
        LEGEND,
    ]
    assert Counter(v["nb"] for v in verdicts(ledger)) == {bad.id: 3, good.id: 3}
    # each notebook is read once, the LLM settings are loaded once, and a message shared by both shows once
    assert fakes.tree_for == 2 and fakes.load_settings == [{}]
    assert result.stderr.splitlines() == [f"warning: in each of 2 notebooks: {INCONCLUSIVE}"]

    out = run(ledger, "audit", "--submitted", "--llm", "never").stdout
    assert f"good.ipynb  {util.short(good.id)}  2 present, 1 absent, 1 none; 0 new verdicts" in out
    assert "0 new verdicts recorded, 6 skipped as already recorded (--force redoes them)" in out
    out = run(ledger, "audit", "--submitted", "--idea", "retry", "--force", "--llm", "never").stdout
    assert f"bad.ipynb   {util.short(bad.id)}  1 absent; 1 new verdict" in out
    assert "2 notebooks: 1 present, 1 absent; 2 new verdicts recorded" in out
    assert len(verdicts(ledger)) == 8


def test_audit_submitted_skips_what_it_cannot_audit_with_one_warning_each(ledger: Store, fakes: Calls):
    good = register_bytes(ledger, GOOD, name="good.ipynb")[0]
    lost = register_bytes(ledger, BAD, name="lost.ipynb")[0]
    ledger.blob_path(lost.id).unlink()  # registered, but its bytes never arrived
    submit(ledger, "1", None, day=1)
    submit(ledger, "2", "e" * 64, day=2)  # a notebook this ledger does not know
    submit(ledger, "3", lost.id, day=3)
    submit(ledger, "4", lost.id, day=4)
    submit(ledger, "5", good.id, day=5)
    result = run(ledger, "audit", "--submitted", "--llm", "never")
    assert result.stderr.splitlines() == [
        "warning: skipped submission 1: it has no notebook (link one with `pseudorca sub link 1 NB`)",
        "warning: skipped submission 2: its notebook eeeeeeeeee is not registered (add it with "
        "`pseudorca nb add FILE` or `pseudorca kaggle pull OWNER/SLUG`, or link another with "
        "`pseudorca sub link 2 NB`)",
        f"warning: skipped lost.ipynb ({util.short(lost.id)}): its bytes are not in this store (pull the "
        "latest store, or add the file again with `pseudorca nb add FILE`)",
    ]
    assert "1 submitted notebook to audit, oldest submission first" in result.stdout
    assert {v["nb"] for v in verdicts(ledger)} == {good.id}
    assert fakes.tree_for == 2  # lost.ipynb looked for in the cache, good.ipynb read for its audit
    data = json.loads(run(ledger, "audit", "--submitted", "--llm", "never", "--json").stdout)
    assert [(e["reason"], e["submissions"], e["nb"]) for e in data["skipped"]] == [
        ("no_notebook", ["1"], None),
        ("not_registered", ["2"], "e" * 64),
        ("not_stored", ["3", "4"], lost.id),
    ]
    assert [e["message"] for e in data["skipped"]] == [
        w[len("warning: ") :] for w in result.stderr.splitlines()
    ]

    # what the warnings say fixes it
    register_bytes(ledger, BAD, name="lost.ipynb")
    run(ledger, "sub", "link", "1", "good.ipynb")
    result = run(ledger, "audit", "--submitted", "--llm", "never")
    assert "skipped submission 2" in result.stderr and "submission 1" not in result.stderr
    assert "lost.ipynb" not in result.stderr and f"lost.ipynb  {util.short(lost.id)}  " in result.stdout


def test_audit_submitted_reads_a_notebook_added_without_its_bytes_from_its_file(
    ledger: Store, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(pseudorca.extract, "tree_for", TREE_FOR)  # reads the file at the recorded path
    path = ledger.root.parent / "notebooks" / "local.ipynb"
    path.parent.mkdir()
    path.write_bytes(GOOD)
    nb = register_file(ledger, path, keep_bytes=False)[0]
    submit(ledger, "1", nb.id, day=1)
    out = run(ledger, "audit", "--submitted", "--llm", "never").stdout
    assert f"local.ipynb  {util.short(nb.id)}  2 present, 1 absent, 1 none; 3 new verdicts" in out
    path.write_bytes(BAD)  # edited since it was added
    result = run(ledger, "audit", "--submitted", "--llm", "never", "--force")
    assert result.stderr.strip() == (
        f"warning: skipped local.ipynb ({util.short(nb.id)}): it was added with --no-store and "
        "notebooks/local.ipynb is gone or has changed (restore that version, or add it with "
        "`pseudorca nb add FILE` so the store keeps its bytes)"
    )
    assert result.stdout.strip() == "no submitted notebooks to audit: 1 submission skipped"
    assert len(verdicts(ledger)) == 3


def test_audit_submitted_dry_run_and_its_usage(ledger: Store):
    good = register_bytes(ledger, GOOD, name="good.ipynb")[0]
    out = run(ledger, "audit", "--submitted").stdout
    assert out.strip() == (
        "no submitted notebooks to audit: no submissions yet (`pseudorca sub add` or `pseudorca kaggle sync`)"
    )
    submit(ledger, "1", good.id, day=1)
    result = run(ledger, "audit", "--submitted", "--dry-run")
    out = result.stdout
    assert out.startswith("1 submitted notebook to audit, oldest submission first, dry run: nothing recorded")
    assert "1 notebook: 2 present, 1 absent, 1 none; 3 new verdicts not recorded (dry run)" in out
    assert result.stderr.splitlines() == [f"warning: in good.ipynb ({util.short(good.id)}): {INCONCLUSIVE}"]
    assert run(ledger, "audit", "--submitted", "--status", "idea").stdout.strip() == "no ideas to audit"
    out = run(ledger, "audit", "good.ipynb", "--submitted", code=1).output
    assert out.strip() == "Error: use NB_REF or --submitted, not both"
    assert verdicts(ledger) == []


def test_audit_submitted_json(ledger: Store):
    good = register_bytes(ledger, GOOD, name="good.ipynb")[0]
    bad = register_bytes(ledger, BAD, name="bad.ipynb")[0]
    empty = json.loads(run(ledger, "audit", "--submitted", "--json").stdout)
    totals = {"notebooks": 0, "submissions": 0, "counts": {}, "new_verdicts": 0, "skipped": 0}
    assert empty == {"dry_run": False, "notebooks": [], "skipped": [], "totals": totals}
    submit(ledger, "1", bad.id, day=1)
    submit(ledger, "2", good.id, day=2)
    submit(ledger, "3", good.id, day=3)
    submit(ledger, "4", None, day=4)
    args = ("--llm", "never", "--json")
    data = json.loads(run(ledger, "audit", "--submitted", "--dry-run", *args).stdout)
    # each notebook as `audit NB --json` shows it
    single = [
        json.loads(run(ledger, "audit", nb, "--dry-run", *args).stdout) for nb in ("bad.ipynb", "good.ipynb")
    ]
    assert data["notebooks"] == single and data["dry_run"] is True
    assert data["skipped"] == [
        {
            "reason": "no_notebook",
            "submissions": ["4"],
            "nb": None,
            "message": "skipped submission 4: it has no notebook (link one with `pseudorca sub link 4 NB`)",
        }
    ]
    counts = {"present": 4, "absent": 2, "none": 2}
    totals = {"notebooks": 2, "submissions": 3, "counts": counts, "new_verdicts": 6, "skipped": 0}
    assert data["totals"] == totals and verdicts(ledger) == []
    run(ledger, "audit", "--submitted", *args)
    again = json.loads(run(ledger, "audit", "--submitted", *args).stdout)
    assert again["totals"] == {**totals, "new_verdicts": 0, "skipped": 6} and len(verdicts(ledger)) == 6


def test_audit_submitted_sets_the_llm_up_once(ledger: Store, monkeypatch: pytest.MonkeyPatch):
    for i in range(4):
        nb = register_bytes(ledger, ipynb(f"x = {i}\n"), name=f"v{i}.ipynb")[0]
        submit(ledger, str(i), nb.id, day=i + 1)
    out = run(ledger, "audit", "--submitted", "--idea", "kv-cache").stderr
    assert out.splitlines() == [f"warning: in each of 4 notebooks: {INCONCLUSIVE}"]

    loaded: list[dict] = []
    asked: list[str] = []

    def load_settings(store: Store, **overrides) -> LLMSettings:
        loaded.append(overrides)
        return LLMSettings(provider="fake", model=overrides.get("model", "m"))

    def not_allowed(settings: LLMSettings) -> None:
        raise pseudorca.llm.LLMError(f"{settings.label} is remote; pass --allow-remote")

    def judge_idea(idea, patch_text, tree, findings, settings, *, cache_dir=None) -> Finding:
        asked.append(idea.id)
        return Finding("absent", "llm", note="no cache flags")

    monkeypatch.setattr(pseudorca.llm, "load_settings", load_settings)
    monkeypatch.setattr(pseudorca.llm, "ensure_allowed", not_allowed)
    monkeypatch.setattr(pseudorca.llm, "judge_idea", judge_idea)
    result = run(ledger, "audit", "--submitted", "--idea", "kv-cache", "--model", "m2")
    assert loaded == [{"model": "m2"}] and asked == []
    assert result.stderr.splitlines() == [
        "warning: in each of 4 notebooks: LLM disabled for this run: fake:m2 is remote; pass --allow-remote"
    ]
    monkeypatch.setattr(pseudorca.llm, "ensure_allowed", lambda settings: None)
    result = run(ledger, "audit", "--submitted", "--idea", "kv-cache", "--model", "m2")
    assert len(loaded) == 2 and asked == ["kv-cache"] * 4
    assert [line for line in result.stderr.splitlines() if "asking" in line] == [
        f"v{i}.ipynb: kv-cache: asking fake:m2" for i in range(4)
    ]
    assert result.stdout.count("1 absent; 1 new verdict\n") == 4
    assert {v["model"] for v in verdicts(ledger)} == {"fake:m2"}


def test_audit_submitted_skips_a_notebook_the_extractor_cannot_read(
    ledger: Store, monkeypatch: pytest.MonkeyPatch
):
    """Its blob is sound, so this is found only when the audit reads it: skipped there, and the run goes on."""
    monkeypatch.setattr(pseudorca.extract, "tree_for", TREE_FOR)
    monkeypatch.setattr(pseudorca.extract, "extract_bytes", EXTRACT_BYTES)
    first = register_bytes(ledger, GOOD, name="first.ipynb")[0]
    script = b"max_retries = 3\n"  # registered by hand as a notebook: `nb add` would refuse it
    broken = ledger.put_blob(script)
    ledger.append("nb.add", {"id": broken, "name": "broken.ipynb", "size": len(script)})
    last = register_bytes(ledger, ipynb("max_retries = 3\n"), name="last.ipynb")[0]
    for i, nb_id in enumerate((first.id, broken, broken, last.id), 1):
        submit(ledger, str(i), nb_id, day=i)
    result = run(ledger, "audit", "--submitted", "--llm", "never")
    (warning,) = result.stderr.splitlines()
    assert warning.startswith(
        f"warning: skipped broken.ipynb ({util.short(broken)}): broken.ipynb is not a valid notebook: its JSON "
        "cannot be read"
    )
    assert result.stdout.splitlines()[2:] == [
        f"first.ipynb   {util.short(first.id)}  2 present, 1 absent, 1 none; 3 new verdicts",
        f"last.ipynb    {util.short(last.id)}  1 present, 2 absent, 1 none; 3 new verdicts",
        "",
        "2 notebooks: 3 present, 3 absent, 2 none; 6 new verdicts recorded",
        LEGEND,
    ]
    assert {v["nb"] for v in verdicts(ledger)} == {first.id, last.id}
    data = json.loads(run(ledger, "audit", "--submitted", "--llm", "never", "--json").stdout)
    assert [n["notebook"]["id"] for n in data["notebooks"]] == [first.id, last.id]
    assert data["skipped"] == [
        {
            "reason": "unreadable",
            "submissions": ["2", "3"],
            "nb": broken,
            "message": warning[len("warning: ") :],
        }
    ]
    assert (data["totals"]["notebooks"], data["totals"]["submissions"]) == (2, 2)

    def reads_nothing(store: Store, nb) -> Tree:
        raise ValueError(f"{nb.name} is not a Jupyter notebook: it has no cells")

    monkeypatch.setattr(pseudorca.extract, "tree_for", reads_nothing)
    result = run(ledger, "audit", "--submitted", "--llm", "never", "--force")
    assert len(result.stderr.splitlines()) == 3
    assert result.stdout.splitlines()[-1] == "no submitted notebook could be read"


def test_audit_submitted_stops_at_any_other_error_and_names_the_notebook(
    ledger: Store, monkeypatch: pytest.MonkeyPatch
):
    first = register_bytes(ledger, GOOD, name="first.ipynb")[0]
    full = register_bytes(ledger, BAD, name="full\x1b[2J.ipynb")[0]  # a name that would clear the screen
    submit(ledger, "1", first.id, day=1)
    submit(ledger, "2", full.id, day=2)
    append = Store.append

    def disk_full(self: Store, type: str, data: dict):
        if type == "audit.verdict" and data["nb"] == full.id:
            raise StoreError("cannot write events/x.json: No space left on device")
        return append(self, type, data)

    monkeypatch.setattr(Store, "append", disk_full)
    result = run(ledger, "audit", "--submitted", "--llm", "never", code=1)
    assert (
        f"first.ipynb     {util.short(first.id)}  2 present, 1 absent, 1 none; 3 new verdicts"
        in result.stdout
    )
    assert result.stderr.splitlines() == [
        f"Error: full?[2J.ipynb ({util.short(full.id)}): cannot write events/x.json: No space left on device"
    ]
    assert {v["nb"] for v in verdicts(ledger)} == {first.id}  # what came before it is recorded


def test_audit_submitted_reads_each_notebook_as_audit_does(ledger: Store, monkeypatch: pytest.MonkeyPatch):
    """A blob that is not the notebook's bytes (a Git LFS pointer), and the lost file of a notebook added with
    --no-store, are skipped with the store's message, and the rest is audited; unless a fresh cached tree
    stands in for them, as it does for `audit NB`."""
    monkeypatch.setattr(pseudorca.extract, "tree_for", TREE_FOR)
    monkeypatch.setattr(pseudorca.extract, "extract_bytes", EXTRACT_BYTES)
    path = ledger.root.parent / "local.ipynb"
    path.write_bytes(GOOD)
    local = register_file(ledger, path, keep_bytes=False)[0]
    lfs = register_bytes(ledger, BAD, name="lfs.ipynb")[0]
    fine = register_bytes(ledger, ipynb("max_retries = 3\n"), name="fine.ipynb")[0]
    for i, nb in enumerate((local, lfs, fine), 1):
        submit(ledger, str(i), nb.id, day=i)
    run(ledger, "audit", "--submitted", "--llm", "never")  # extracts each notebook and caches its tree
    path.unlink()
    pointer = f"version https://git-lfs.github.com/spec/v1\noid sha256:{lfs.id}\nsize {len(BAD)}\n"
    ledger.blob_path(lfs.id).write_text(pointer, encoding="utf-8")
    result = run(ledger, "audit", "--submitted", "--llm", "never", "--force")
    assert (
        result.stderr == ""
        and "3 notebooks: " in result.stdout
        and "9 new verdicts recorded" in result.stdout
    )
    for nb in (local, lfs):
        assert (
            "3 new verdicts recorded" in run(ledger, "audit", nb.id[:12], "--llm", "never", "--force").stdout
        )

    shutil.rmtree(ledger.root / "cache" / "trees")  # nothing stands in for the lost bytes any more
    result = run(ledger, "audit", "--submitted", "--llm", "never", "--force")
    lost, damaged = result.stderr.splitlines()
    assert lost == (
        f"warning: skipped local.ipynb ({util.short(local.id)}): it was added with --no-store and local.ipynb "
        "is gone or has changed (restore that version, or add it with `pseudorca nb add FILE` so the store "
        "keeps its bytes)"
    )
    assert damaged.startswith(
        f"warning: skipped lfs.ipynb ({util.short(lfs.id)}): blob {lfs.id[:12]} does not match its sha256: "
        "it is probably a Git LFS pointer (run `git lfs pull`)"
    )
    assert result.stdout.startswith("1 submitted notebook to audit, oldest submission first")
    assert f"fine.ipynb  {util.short(fine.id)}  1 present, 2 absent, 1 none; 3 new verdicts" in result.stdout
    data = json.loads(run(ledger, "audit", "--submitted", "--llm", "never", "--json").stdout)
    assert [(e["reason"], e["submissions"], e["nb"]) for e in data["skipped"]] == [
        ("not_stored", ["1"], local.id),
        ("unreadable", ["2"], lfs.id),
    ]
    register_bytes(ledger, BAD, name="lfs.ipynb")  # adding the file again restores its blob
    result = run(ledger, "audit", "--submitted", "--llm", "never")
    assert "lfs.ipynb" not in result.stderr and f"lfs.ipynb   {util.short(lfs.id)}  " in result.stdout


DOWN = "fake:m at 127.0.0.1 failed: APIConnectionError: Connection error."


def llm_that_fails(monkeypatch: pytest.MonkeyPatch, error: Callable[[str, Tree], str]) -> None:
    """A configured LLM whose calls fail with `error(idea id, tree)`, or answer absent when that is ""."""

    def load_settings(store: Store, **overrides) -> LLMSettings:
        return LLMSettings(provider="fake", model="m")

    def judge_idea(idea, patch_text, tree, findings, settings, *, cache_dir=None) -> Finding:
        if text := error(idea.id, tree):
            raise pseudorca.llm.LLMError(text)
        return Finding("absent", "llm", note="not seen")

    monkeypatch.setattr(pseudorca.llm, "load_settings", load_settings)
    monkeypatch.setattr(pseudorca.llm, "judge_idea", judge_idea)


def warnings_of(result: Result) -> list[str]:
    return [line for line in result.stderr.splitlines() if line.startswith("warning: ")]


def test_audit_submitted_shows_llm_errors_once_each(ledger: Store, monkeypatch: pytest.MonkeyPatch):
    """`audit NB` shows an LLM error in its table. --submitted counts them on each notebook's line and in
    the total, and warns once per error text, saying how many notebooks it came from."""
    for i, data in enumerate((GOOD, BAD, ipynb("x = 1\n"))):
        submit(ledger, str(i), register_bytes(ledger, data, name=f"v{i}.ipynb")[0].id, day=i + 1)
    llm_that_fails(monkeypatch, lambda idea, tree: DOWN)
    result = run(ledger, "audit", "--submitted", "--idea", "kv-cache")
    assert result.stdout.count("  1 none; 0 new verdicts; 1 LLM error\n") == 3
    assert "\n3 notebooks: 3 none; 0 new verdicts recorded; 3 LLM errors\n" in result.stdout
    assert warnings_of(result) == [f"warning: in each of 3 notebooks: the LLM failed on 1 idea: {DOWN}"]
    assert "llm error: fake:m at 127.0.0.1" in table_row(run(ledger, "audit", "v0.ipynb").stdout, "kv-cache")

    def code_with_retries(idea: str, tree: Tree) -> str:  # fails twice in v0, once in v1 and v2
        code = "".join(f.text for path, f in tree.files.items() if path.startswith("cells/"))
        return DOWN if idea == "kv-cache" or "max_retries" in code else ""

    llm_that_fails(monkeypatch, code_with_retries)
    args = ("--idea", "kv-cache", "--idea", "retry-loop", "--llm", "always")
    result = run(ledger, "audit", "--submitted", *args)
    lines = result.stdout.splitlines()
    assert [line.rsplit("; ", 1)[1] for line in lines[2:5]] == ["2 LLM errors", "1 LLM error", "1 LLM error"]
    assert lines[-2].endswith("; 4 LLM errors")
    assert warnings_of(result) == [f"warning: in 3 notebooks: the LLM failed on 4 ideas in all: {DOWN}"]
    data = json.loads(run(ledger, "audit", "--submitted", *args, "--json").stdout)
    assert [i["llm_error"] for n in data["notebooks"] for i in n["ideas"]] == [DOWN, DOWN, DOWN, "", DOWN, ""]


def test_audit_submitted_shows_the_most_frequent_llm_errors(ledger: Store, monkeypatch: pytest.MonkeyPatch):
    """Error texts can differ on every call (a rate limit's wait): the most frequent are shown, the others
    counted."""
    nbs = [register_bytes(ledger, ipynb(f"x = {i}\n"), name=f"v{i}.ipynb")[0] for i in range(8)]
    for i, nb in enumerate(nbs):
        submit(ledger, str(i), nb.id, day=i + 1)
    texts = [f"rate limited: retry in {i} ms" for i in range(6)] + [DOWN, DOWN]  # one call per notebook
    llm_that_fails(monkeypatch, lambda idea, tree: texts.pop(0))
    result = run(ledger, "audit", "--submitted", "--idea", "kv-cache")
    assert warnings_of(result) == [
        f"warning: in each of 2 notebooks: the LLM failed on 1 idea: {DOWN}",
        *[
            f"warning: in v{i}.ipynb ({util.short(nbs[i].id)}): the LLM failed on 1 idea: rate limited: "
            f"retry in {i} ms"
            for i in range(4)
        ],
        "warning: 2 more LLM errors, with 2 other texts (--json shows each one)",
    ]
    assert "8 notebooks: 8 none; 0 new verdicts recorded; 8 LLM errors" in result.stdout


def test_strings_from_the_ledger_reach_the_terminal_without_control_characters(
    ledger: Store, monkeypatch: pytest.MonkeyPatch
):
    """Names and ids come from teammates' events, or hand-edited ones: characters that could move the
    cursor, retitle the terminal or reorder the text are printed as "?"."""
    sha = ledger.put_blob(GOOD)
    ledger.append(
        "nb.add", {"id": sha, "name": "evil\x1b]0;pwned\x07\x1b[31m\u202e.ipynb", "size": len(GOOD)}
    )
    ledger.append("idea.add", {"id": "bell\x07", "title": "t", "status": "testing", "fingerprints": []})
    submit(ledger, "1", sha, day=1)
    configure_llm(monkeypatch, Finding("absent", "llm", note="not seen"))
    results = [
        run(ledger, "audit", "--submitted"),
        run(ledger, "audit", sha[:12], "--force"),
        run(ledger, "check", sha[:12]),
        run(ledger, "verdict", sha[:12], "bell\x07", "absent"),
    ]
    shown = "evil?]0;pwned??[31m?.ipynb"
    assert f"\n{shown}  {util.short(sha)}  " in results[0].stdout
    assert f"{shown}: bell?: asking fake:m" in results[0].stderr
    assert results[1].stdout.startswith(f"audit of {shown} ") and "bell?: asking" in results[1].stderr
    assert results[2].stdout.startswith(f"check of {shown} ")
    assert results[3].stdout == f"recorded: bell? in {shown} is ✗ absent (human verdict)\n"
    for result in results:
        assert not any(c in result.stdout + result.stderr for c in "\x1b\x07\u202e")
