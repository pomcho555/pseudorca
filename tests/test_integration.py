"""End to end through the CLI with the synthetic demo files (examples/make_demo_files.py)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from pseudorca.cli import main
from pseudorca.store import Store

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def demo(store: Store) -> Path:
    spec = importlib.util.spec_from_file_location("make_demo_files", ROOT / "examples" / "make_demo_files.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    repo = store.root.parent
    module.main(repo)
    return repo


def _run(store: Store, *args: str) -> tuple[int, str]:
    out = CliRunner().invoke(main, ["--store", str(store.root), *args])
    return out.exit_code, out.output


def test_demo_flow(store: Store, demo: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(demo)
    assert (
        _run(
            store,
            "idea",
            "add",
            "short-cot",
            "--title",
            "Shorter reasoning",
            "--status",
            "adopted",
            "--fingerprint",
            "Think step by step, briefly.",
        )[0]
        == 0
    )
    assert (
        _run(
            store,
            "idea",
            "add",
            "retry-parse",
            "--title",
            "Retry",
            "--status",
            "adopted",
            "--patch",
            "retry-parse.diff",
        )[0]
        == 0
    )
    assert (
        _run(
            store,
            "idea",
            "add",
            "temp-06",
            "--title",
            "Temperature 0.6",
            "--status",
            "rejected",
            "--fingerprint",
            r"re:temperature\s*=\s*0\.6",
        )[0]
        == 0
    )
    assert _run(store, "nb", "add", "base.ipynb")[0] == 0
    assert _run(store, "nb", "add", "next.ipynb", "--parent", "base.ipynb", "--applies", "short-cot")[0] == 0

    code, out = _run(store, "audit", "next.ipynb")
    assert code == 0, out
    assert "files/solver.py" in out  # evidence comes from the %%writefile file

    code, out = _run(store, "check", "next.ipynb")
    assert code == 0 and "check passed" in out, out
    code, out = _run(store, "check", "base.ipynb")
    assert code == 1 and "MISSING" in out, out

    code, out = _run(store, "nb", "tree", "next.ipynb")
    assert "payloads/cell_003/0/utils/io.py" in out

    code, out = _run(store, "status", "--json")
    assert code == 0
    json.loads(out)

    for name in ("base", "next"):
        code, out = _run(
            store,
            "run",
            "add",
            f"runs/{name}.csv",
            "--nb",
            f"{name}.ipynb",
            "--label",
            name,
            "--protocol",
            "p60x3",
        )
        assert code == 0, out
    code, out = _run(store, "compare", "base", "next", "--json")
    assert code == 0, out
    result = json.loads(out)
    assert result["n_items"] == 60 and result["ci_low"] < result["diff"] < result["ci_high"]

    code, out = _run(store, "report")
    assert code == 0 and (store.root / "cache" / "report.html").exists()


def test_store_problems_are_one_line_errors_in_every_command(store: Store, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("PSEUDORCA_DEBUG", raising=False)
    (store.root / "config.toml").write_bytes(b"[store\nname = \xff\n")
    assert CliRunner().invoke(main, ["--store", str(store.root), "status"]).exit_code == 0  # needs no config
    for args in (["sub", "add", "1", "--score", "0.5"], ["report"]):
        out = CliRunner().invoke(main, ["--store", str(store.root), *args])
        assert out.exit_code == 1, (args, out.output)
        assert out.output.startswith("Error: ") and "Traceback" not in out.output, (args, out.output)
        assert "config.toml" in out.output or "store" in out.output


def test_replay_warnings_are_printed_once_by_any_command(store: Store):
    store.append("idea.add", {"id": "i", "title": "t", "priority": "high"})
    out = CliRunner().invoke(main, ["--store", str(store.root), "idea", "list"])
    assert out.exit_code == 0
    assert out.output.count("ignored field 'priority'") == 1


def test_a_closed_pipe_ends_quietly(store: Store):
    import subprocess
    import sys

    for i in range(400):
        store.append("idea.add", {"id": f"idea-{i}", "title": "x" * 200})
    p = subprocess.Popen(
        [sys.executable, "-m", "pseudorca", "--store", str(store.root), "log", "-n", "0", "--json"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert p.stdout is not None and p.stderr is not None
    p.stdout.read(10)
    p.stdout.close()
    err = p.stderr.read().decode()
    assert p.wait(timeout=60) == 0, err
    assert "Traceback" not in err and "Exception ignored" not in err
