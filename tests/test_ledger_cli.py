"""The ledger commands (init, idea, nb, status, sub, log) through the CLI, on synthetic data only."""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable, Iterator
from dataclasses import fields
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner, Result

from pseudorca import util
from pseudorca.cli import main
from pseudorca.commands._common import LEGEND
from pseudorca.commands.ledger import PatchFile, summarize_patch
from pseudorca.model import Idea, Notebook, State, Submission
from pseudorca.store import Store
from pseudorca.tree import Tree

PATCH = """\
diff --git a/src/agent.py b/src/agent.py
index 0000000..1111111 100644
--- a/src/agent.py
+++ b/src/agent.py
@@ -1,3 +1,4 @@
 import os
-MAX_TOKENS = 1024
+MAX_TOKENS = 512
+TEMPERATURE = 0.0
 def run():
@@ -10,2 +11,2 @@ def run():
-    return call(prompt)
+    return call(short_prompt)

--- a/src/util.py
+++ b/src/util.py
@@ -1 +1 @@
-x = 1
+x = 2
"""


# ----- helpers -------------------------------------------------------------------------------------


def cli(store: Store, *args: str, code: int = 0, input: str | None = None) -> Result:
    """Run `pseudorca --store <store> ARGS` and check the exit code; unexpected exceptions propagate."""
    result = CliRunner().invoke(main, ["--store", str(store.root), *args], input=input)
    if result.exception is not None and not isinstance(result.exception, SystemExit):
        raise result.exception
    assert result.exit_code == code, result.output
    return result


def fresh(store: Store) -> State:
    """The state as the CLI left it (the fixture's Store object caches what it has read)."""
    return Store.at(store.root).state()


def table(out: str) -> list[list[str]]:
    """The rows of the first table in `out`, split into cells on runs of two or more spaces."""
    lines = out.splitlines()
    start = next(i for i, line in enumerate(lines) if "-" in line and set(line) <= {"-", " "})
    rows = []
    for line in lines[start + 1 :]:
        if not line.strip() or line.startswith("("):
            break
        rows.append(re.split(r" {2,}", line.strip()))
    return rows


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode())
    return path


def add_nb(store: Store, data: bytes, **fields: Any) -> str:
    """Register a notebook straight through an event."""
    sha = store.put_blob(data)
    store.append("nb.add", {"id": sha, "size": len(data), **fields})
    return sha


def add_verdict(store: Store, nb: str, idea_id: str, status: str, method: str, **kw: Any) -> None:
    rev = fresh(store).ideas[idea_id].rev
    store.append(
        "audit.verdict",
        {"nb": nb, "idea": idea_id, "idea_rev": rev, "status": status, "method": method, **kw},
    )


def field_names(cls: type) -> set[str]:
    return {f.name for f in fields(cls)}


def put_event(store: Store, type: str, data: dict[str, Any], when: str, actor: str) -> None:
    """Write an event file the way a teammate's clone does: with that machine's clock and UTC offset."""
    eid = util.new_ulid()
    folder = store.root / "events" / when[:7]
    folder.mkdir(parents=True, exist_ok=True)
    event = {"schema": 1, "id": eid, "type": type, "time": when, "actor": actor, "data": data}
    (folder / f"{eid}-{actor}.json").write_text(json.dumps(event), encoding="utf-8")


def flat(text: str) -> str:
    """Text with every run of whitespace made one space (help text wraps at the terminal width)."""
    return " ".join(text.split())


@pytest.fixture
def viewer_tz() -> Iterator[Callable[[str], None]]:
    """Sets the viewer's time zone (a POSIX TZ string, so no time zone database is needed)."""
    if not hasattr(time, "tzset"):
        pytest.skip("needs time.tzset")
    old = os.environ.get("TZ")

    def set_tz(tz: str) -> None:
        os.environ["TZ"] = tz
        time.tzset()

    yield set_tz
    if old is None:
        os.environ.pop("TZ", None)
    else:
        os.environ["TZ"] = old
    time.tzset()


# ----- init ----------------------------------------------------------------------------------------


def test_init_creates_a_store_inside_a_git_repository(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("PSEUDORCA_STORE", raising=False)
    repo = tmp_path / "repo"
    (repo / "sub").mkdir(parents=True)
    (repo / ".git").write_text("gitdir: elsewhere\n")  # a worktree's .git is a file
    runner = CliRunner()
    r = runner.invoke(main, ["init", str(repo / "sub"), "--name", "team", "--competition", "some-comp"])
    assert r.exit_code == 0, r.output
    assert "created" in r.stdout and "config.toml" in r.stdout
    assert "git add .pseudorca" in r.stdout and "cd " in r.stdout and "private" in r.stdout
    assert "git repository" not in r.stderr
    store = Store.at(repo / "sub")
    assert store.setting("store", "name") == "team" and store.setting("store", "competition") == "some-comp"

    again = runner.invoke(main, ["init", str(repo / "sub")])
    assert again.exit_code == 1 and "already exists" in again.output


def test_init_defaults_to_the_current_directory_and_warns_outside_git(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv("PSEUDORCA_STORE", raising=False)
    plain = tmp_path / "plain"
    plain.mkdir()
    if any((d / ".git").exists() for d in (plain, *plain.parents)):
        pytest.skip("the temporary directory is inside a git repository")
    monkeypatch.chdir(plain)
    r = CliRunner().invoke(main, ["init"])
    assert r.exit_code == 0, r.output
    assert (plain / ".pseudorca" / "events").is_dir()
    assert "not inside a git repository" in r.stderr
    assert "cd " not in r.stdout


def init_listing(out: str) -> list[str]:
    """The names init lists under its first line, up to the blank line before the next steps."""
    names = []
    for line in out.splitlines()[1:]:
        if not line.strip():
            break
        names.append(line.split()[0])
    return names


@pytest.mark.parametrize("empty_folder", [False, True])
def test_init_lists_exactly_what_it_created(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, empty_folder: bool
):
    monkeypatch.delenv("PSEUDORCA_STORE", raising=False)
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    if empty_folder:  # made by hand: nothing to keep, so still "created"
        (repo / ".pseudorca").mkdir()
    r = CliRunner().invoke(main, ["init", str(repo)])
    assert r.exit_code == 0, r.output
    assert r.stdout.splitlines()[0] == f"created {repo.resolve() / '.pseudorca'}"
    assert init_listing(r.stdout) == [
        "config.toml",
        "events/",
        "blobs/",
        "cache/",
        ".gitignore",
        ".gitattributes",
    ]


def test_init_completes_a_partial_store_without_touching_its_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A clone holds the committed config.toml and .gitignore but no events/ (nothing was recorded yet)."""
    monkeypatch.delenv("PSEUDORCA_STORE", raising=False)
    repo = tmp_path / "clone"
    (repo / ".git").mkdir(parents=True)
    config = '[store]\nname = "team"\ncompetition = "team-comp"\n\n[llm]\nmodel = "team-model"\n'
    write(repo / ".pseudorca" / "config.toml", config)
    write(repo / ".pseudorca" / ".gitignore", "cache/\n")
    root = (repo / ".pseudorca").resolve()

    r = CliRunner().invoke(main, ["init", str(repo), "--competition", "other"])
    assert r.exit_code == 0, r.output
    assert r.stdout.splitlines()[0].startswith(f"completed {root}")
    assert init_listing(r.stdout) == ["events/", "blobs/", "cache/", ".gitattributes"]
    assert "config.toml" not in r.stdout.split("\n\n")[0]
    assert (repo / ".pseudorca" / "config.toml").read_text() == config
    assert "--competition was not applied" in flat(r.stderr)
    assert Store.at(repo).setting("store", "competition") == "team-comp"

    again = CliRunner().invoke(main, ["init", str(repo)])
    assert again.exit_code == 1
    assert again.output == f"Error: a store already exists at {root}\n"  # one line, no traceback


# ----- idea ----------------------------------------------------------------------------------------


def test_idea_add_records_every_field(store: Store, tmp_path: Path):
    patch = write(tmp_path / "fast.diff", PATCH)
    desc = write(tmp_path / "desc.md", "Cut the prompt.\n\nKeep the examples.\n")
    r = cli(
        store,
        *("idea", "add", "fast-prompt", "--title", " Shorter prompt ", "--desc-file", str(desc)),
        *("--tag", "prompt", "--tag", "prompt", "--link", "https://example.com/notes"),
        *("--fingerprint", "MAX_TOKENS = 512", "--fingerprint", r"re:TEMPERATURE\s*=\s*0"),
        *("--fingerprint", "!old_prompt", "--patch", str(patch)),
    )
    assert r.output.strip() == "added idea fast-prompt (idea, owner tester, checks 3f+p)"
    events = fresh(store).events
    assert len(events) == 1 and events[0].type == "idea.add"
    assert events[0].data == {
        "id": "fast-prompt",
        "title": "Shorter prompt",
        "description": "Cut the prompt.\n\nKeep the examples.",
        "owner": "tester",
        "status": "idea",
        "tags": ["prompt"],
        "links": ["https://example.com/notes"],
        "fingerprints": ["MAX_TOKENS = 512", r"re:TEMPERATURE\s*=\s*0", "!old_prompt"],
        "patch": util.sha256_bytes(PATCH.encode()),
    }
    assert store.get_blob(util.sha256_bytes(PATCH.encode())) == PATCH.encode()

    cli(store, "idea", "add", "kv", "--title", "t", "--owner", "bob", "--status", "testing", "--desc-file", "-",
        input="from stdin\n")  # fmt: skip
    idea = fresh(store).ideas["kv"]
    assert (idea.owner, idea.status, idea.description, idea.patch) == ("bob", "testing", "from stdin", None)


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["bad id", "--title", "t"], "invalid idea id"),
        (["x" * 65, "--title", "t"], "invalid idea id"),
        (["ok", "--title", "t", "--fingerprint", "re:(unclosed"], "invalid regex"),
        (["ok", "--title", "t", "--fingerprint", "!re:[a-"], "invalid regex"),
        (["ok", "--title", "t", "--fingerprint", "re:"], "nothing to look for"),
        (["ok", "--title", "t", "--fingerprint", "!"], "nothing to look for"),
        (["ok", "--title", "t", "--fingerprint", " \n \n"], "nothing to look for"),
        (["ok", "--title", "t", "--fingerprint", "re:a\n(b"], "invalid regex"),
        (["ok", "--title", "  "], "--title cannot be empty"),
        (["ok", "--title", "t", "--tag", " "], "a tag cannot be empty"),
        (["ok", "--title", "t", "--desc", "x", "--desc-file", "-"], "not both"),
    ],
)
def test_idea_add_rejects_bad_input_without_writing(
    store: Store, tmp_path: Path, args: list[str], message: str
):
    patch = write(tmp_path / "p.diff", PATCH)
    r = cli(store, "idea", "add", *args, "--patch", str(patch), code=1)
    assert message in r.output
    assert fresh(store).events == []
    assert [p.name for p in (store.root / "blobs").rglob("*")] == [".gitkeep"]


def test_idea_add_refuses_a_duplicate_id_and_a_bad_status(store: Store):
    cli(store, "idea", "add", "fast-prompt", "--title", "t")
    assert "already exists" in cli(store, "idea", "add", "fast-prompt", "--title", "t2", code=1).output
    r = cli(store, "idea", "add", "x", "--title", "t", "--status", "done", code=2)  # a usage error
    assert "Invalid value" in r.output
    assert len(fresh(store).events) == 1


def test_multi_line_fingerprints_are_kept_and_shown_on_indented_lines(store: Store):
    """A snippet copied from the notebook across lines is one fingerprint (the audit matches it as a block),
    and so is a regex with a line break (searched over whole files)."""
    snippet = "for _ in range(retries + 1):\n    out = llm.generate("
    regex = "re:range\\(retries \\+ 1\\):\n\\s+out"
    fps = (snippet, regex, "!solve_old(")
    r = cli(store, "idea", "add", "retry", "--title", "t", *(a for fp in fps for a in ("--fingerprint", fp)))
    assert r.output.strip() == "added idea retry (idea, owner tester, checks 3f)"
    assert fresh(store).ideas["retry"].fingerprints == list(fps)

    lines = cli(store, "idea", "show", "retry").stdout.splitlines()
    i = next(n for n, line in enumerate(lines) if line.startswith("fingerprints"))
    indent = " " * len("fingerprints  ")
    assert lines[i : i + 5] == [
        "fingerprints  for _ in range(retries + 1):",
        indent + "      out = llm.generate(",  # its own indentation, two columns in from the first line
        indent + "re:range\\(retries \\+ 1\\):",
        indent + "  \\s+out",
        indent + "!solve_old(",
    ]
    r = cli(store, "idea", "update", "retry", "--remove-fingerprint", snippet, "--remove-fingerprint", regex)
    assert r.output == "updated idea retry: fingerprints=!solve_old(\n"
    r = cli(store, "idea", "update", "retry", "--add-fingerprint", "if ok:\n    return out")
    assert r.output == "updated idea retry: fingerprints=!solve_old(,if ok: return out\n"  # on one line

    for command in (["idea", "add"], ["idea", "update"]):
        help_text = flat(cli(store, *command, "--help").output)
        assert "text (may span lines), re:REGEX, !text or !re:REGEX" in help_text


def test_a_patch_that_is_not_a_diff_is_stored_with_a_warning(store: Store, tmp_path: Path):
    notes = write(tmp_path / "notes.txt", "just some notes\n")
    r = cli(store, "idea", "add", "x", "--title", "t", "--patch", str(notes))
    assert "does not look like a unified diff" in r.stderr
    sha = fresh(store).ideas["x"].patch
    assert sha == util.sha256_bytes(b"just some notes\n") and store.has_blob(sha)


def test_idea_update_writes_only_the_changed_fields(store: Store, tmp_path: Path):
    cli(store, "idea", "add", "fast-prompt", "--title=Shorter prompt", "--tag=prompt", "--fingerprint=A=1")
    r = cli(
        store,
        *("idea", "update", "fast-prompt", "--status", "adopted", "--title", "Shorter prompt"),
        *("--add-tag", "speed", "--add-tag", "prompt"),
    )
    assert r.output.strip() == "updated idea fast-prompt: status=adopted, tags=prompt,speed"
    assert fresh(store).events[-1].data == {
        "id": "fast-prompt",
        "status": "adopted",
        "tags": ["prompt", "speed"],
    }

    patch = write(tmp_path / "p.diff", PATCH)
    cli(
        store,
        *("idea", "update", "fast", "--remove-tag", "prompt", "--remove-fingerprint", "A=1"),
        *("--add-fingerprint", r"re:B\s*=\s*2", "--patch", str(patch), "--desc", "new text"),
    )
    idea = fresh(store).ideas["fast-prompt"]
    assert (idea.tags, idea.fingerprints, idea.description) == (["speed"], [r"re:B\s*=\s*2"], "new text")
    assert idea.patch == util.sha256_bytes(PATCH.encode())
    cli(store, "idea", "update", "fast-prompt", "--no-patch")
    assert fresh(store).ideas["fast-prompt"].patch is None

    r = cli(store, "idea", "update", "fast-prompt", "--remove-tag", "missing", "--status", "adopted", code=1)
    assert "has no tag 'missing'" in r.stderr and "already has these values" in r.output


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["nope", "--status", "adopted"], "no idea matches 'nope'"),
        (["fast-prompt"], "give at least one option"),
        (["fast-prompt", "--status", "idea"], "nothing to change"),
        (["fast-prompt", "--add-fingerprint", "re:("], "invalid regex"),
        (["fast-prompt", "--add-tag", "x", "--remove-tag", "x"], "both added and removed"),
        (["fast-prompt", "--patch", "-", "--no-patch"], "not both"),
        (["fast-prompt", "--title", ""], "--title cannot be empty"),
    ],
)
def test_idea_update_errors(store: Store, args: list[str], message: str):
    cli(store, "idea", "add", "fast-prompt", "--title", "t")
    assert message in cli(store, "idea", "update", *args, code=1).output
    assert len(fresh(store).events) == 1


def test_changing_an_ideas_checks_says_its_verdicts_are_stale(store: Store):
    cli(store, "idea", "add", "fast-prompt", "--title", "t", "--fingerprint", "A=1")
    add_verdict(store, add_nb(store, b"one", name="main.ipynb"), "fast-prompt", "present", "fingerprint")
    assert "stale" not in cli(store, "idea", "update", "fast-prompt", "--owner", "bob").output
    r = cli(store, "idea", "update", "fast-prompt", "--add-fingerprint", "B")
    assert "the verdicts on 1 notebook are now stale" in r.output


def test_idea_list_filters_and_hides_parked_ideas(store: Store, tmp_path: Path):
    assert "no ideas yet" in cli(store, "idea", "list").output
    patch = write(tmp_path / "p.diff", PATCH)
    cli(store, "idea", "add", "fast-prompt", "--title", "Shorter prompt", "--tag", "prompt",
        "--fingerprint", "a", "--fingerprint", "b", "--patch", str(patch))  # fmt: skip
    cli(store, "idea", "add", "kv-cache", "--title=Quantised KV cache", "--status=testing", "--owner=bob")
    cli(store, "idea", "add", "old-trick", "--title", "Beam search", "--status", "parked")
    day = fresh(store).ideas["fast-prompt"].updated[:10]

    out = cli(store, "idea", "list").output
    rows = table(out)
    assert out.splitlines()[0].split() == ["ID", "STATUS", "OWNER", "TITLE", "CHECKS", "UPDATED"]
    assert rows == [
        ["fast-prompt", "idea", "tester", "Shorter prompt", "2f+p", day],
        ["kv-cache", "testing", "bob", "Quantised KV cache", "-", day],
    ]
    assert "(1 parked idea hidden; use --all)" in out

    def ids(*args: str) -> list[str]:
        return [row[0] for row in table(cli(store, "idea", "list", *args).output)]

    assert ids("--all") == ["fast-prompt", "kv-cache", "old-trick"]
    assert ids("--status", "parked") == ["old-trick"]
    assert ids("--status", "idea", "--status", "testing") == ["fast-prompt", "kv-cache"]
    assert ids("--tag", "prompt") == ["fast-prompt"]
    assert ids("--owner", "bob") == ["kv-cache"]
    assert "no ideas match" in cli(store, "idea", "list", "--tag", "nothing").output

    data = json.loads(cli(store, "idea", "list", "--all", "--json").stdout)
    assert [d["id"] for d in data] == ["fast-prompt", "kv-cache", "old-trick"]
    assert set(data[0]) == field_names(Idea)
    assert data[0]["fingerprints"] == ["a", "b"] and data[0]["patch"] == util.sha256_bytes(PATCH.encode())


def test_idea_show_prints_fields_patch_and_verdicts(store: Store, tmp_path: Path):
    patch = write(tmp_path / "p.diff", PATCH)
    cli(store, "idea", "add", "fast-prompt", "--title", "Shorter prompt", "--desc", "Line one.\nLine two.",
        "--fingerprint", "MAX_TOKENS = 512", "--patch", str(patch), "--link", "https://example.com/a")  # fmt: skip
    n1 = add_nb(store, b"one", name="main.ipynb")
    add_nb(store, b"unrelated", name="other.ipynb")
    n2 = add_nb(store, b"two", name="main.ipynb", applies=["fast-prompt"])
    evidence = [{"path": "cells/cell_001.py", "line": 3, "quote": "MAX_TOKENS   =   512"}]
    add_verdict(store, n1, "fast-prompt", "present", "fingerprint", evidence=evidence)

    out = cli(store, "idea", "show", "fast").output
    assert "Shorter prompt" in out and "Line two." in out and "https://example.com/a" in out
    assert "2 files, 3 hunks, +4 -3" in out
    assert re.search(r"src/agent\.py\s+2 hunks\s+\+3 -2", out)
    assert re.search(r"src/util\.py\s+1 hunk\s+\+1 -1", out)
    rows = table(out)
    assert [row[:3] for row in rows] == [
        [f"main.ipynb {n1[:8]}", "✓ present", "fingerprint"],
        [f"main.ipynb {n2[:8]}", "c claimed", "-"],
    ]
    assert rows[0][4] == "MAX_TOKENS = 512" and rows[1][3:] == ["-", "-"]

    data = json.loads(cli(store, "idea", "show", "fast-prompt", "--json").stdout)
    assert set(data) == {"idea", "rev", "patch_summary", "notebooks"}
    assert data["idea"]["id"] == "fast-prompt" and data["rev"] == fresh(store).ideas["fast-prompt"].rev
    summary = data["patch_summary"]
    assert (summary["available"], summary["hunks"], summary["added"], summary["removed"]) == (True, 3, 4, 3)
    assert [f["path"] for f in summary["files"]] == ["src/agent.py", "src/util.py"]
    first, second = data["notebooks"]
    assert first["nb"] == n1 and first["verdict"]["status"] == "present" and not first["claimed"]
    assert first["verdict"]["records"][0]["evidence"] == evidence
    assert second["nb"] == n2 and second["verdict"] is None and second["claimed"]

    cli(store, "idea", "update", "fast-prompt", "--desc", "changed")
    assert table(cli(store, "idea", "show", "fast-prompt").output)[0][1] == "✓* present, stale"


def test_idea_show_lists_only_the_latest_ten_notebooks(store: Store):
    store.append("idea.add", {"id": "x", "title": "t"})
    shas = [add_nb(store, f"nb {i}".encode(), name=f"n{i:02}.ipynb", applies=["x"]) for i in range(12)]
    out = cli(store, "idea", "show", "x").output
    assert "latest 10 of 12 notebooks" in out
    assert [row[0] for row in table(out)] == [f"n{i:02}.ipynb {shas[i][:8]}" for i in range(2, 12)]
    assert len(json.loads(cli(store, "idea", "show", "x", "--json").stdout)["notebooks"]) == 12


def test_idea_show_without_verdicts_or_patch_blob(store: Store, tmp_path: Path):
    patch = write(tmp_path / "p.diff", PATCH)
    cli(store, "idea", "add", "x", "--title", "t", "--patch", str(patch))
    store.blob_path(util.sha256_bytes(PATCH.encode())).unlink()
    out = cli(store, "idea", "show", "x").output
    assert "(blob not in this store)" in out and "no verdicts yet" in out
    assert json.loads(cli(store, "idea", "show", "x", "--json").stdout)["patch_summary"]["available"] is False
    assert "no idea matches 'nope'" in cli(store, "idea", "show", "nope", code=1).output


def test_summarize_patch_handles_plain_git_new_and_deleted_files():
    assert summarize_patch(PATCH) == [PatchFile("src/agent.py", 2, 3, 2), PatchFile("src/util.py", 1, 1, 1)]
    sql = "--- a/q.sql\n+++ b/q.sql\n@@ -1,2 +1,2 @@\n--- old comment\n+-- new comment\n SELECT 1;\n"
    assert summarize_patch(sql) == [PatchFile("q.sql", 1, 1, 1)]
    created = "--- /dev/null\n+++ b/new.py\n@@ -0,0 +1,2 @@\n+x = 1\n+y = 2\n"
    deleted = "--- a/old.py\t2026-01-01 00:00:00\n+++ /dev/null\n@@ -1 +0,0 @@\n-x = 1\n\\ No newline at end of file\n"
    assert summarize_patch(created + deleted) == [PatchFile("new.py", 1, 2, 0), PatchFile("old.py", 1, 0, 1)]
    loose = "--- a.py\n+++ a.py\n@@ ... @@\n-a\n+b\n+c\n"
    assert summarize_patch(loose) == [PatchFile("a.py", 1, 2, 1)]
    assert summarize_patch("not a diff") == []


# ----- nb ------------------------------------------------------------------------------------------


def test_nb_add_records_a_repo_relative_path_never_an_absolute_one(
    store: Store, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    repo = store.root.parent
    main_nb = write(repo / "notebooks" / "main.ipynb", '{"cells": []}')
    r = cli(store, "nb", "add", str(main_nb))
    sha = util.sha256_file(main_nb)
    assert r.output.strip() == f"registered main.ipynb {sha[:10]}"
    assert fresh(store).notebooks[sha].path == "notebooks/main.ipynb"

    script = write(repo / "notebooks" / "sub" / "agent.py", "print('hi')\n")
    monkeypatch.chdir(repo / "notebooks")
    cli(store, "nb", "add", "sub/agent.py")
    assert fresh(store).notebooks[util.sha256_file(script)].path == "notebooks/sub/agent.py"

    outside = write(tmp_path / "elsewhere" / "draft.ipynb", '{"cells": [1]}')
    r = cli(store, "nb", "add", str(outside), "--no-store", code=1)
    assert "outside the repository" in r.output  # audits could never find it again
    assert util.sha256_file(outside) not in fresh(store).notebooks
    r = cli(store, "nb", "add", str(outside))
    assert fresh(store).notebooks[util.sha256_file(outside)].path == "draft.ipynb"

    for p in (store.root / "events").rglob("*.json"):
        text = p.read_text(encoding="utf-8")
        assert str(tmp_path) not in text and str(tmp_path.resolve()) not in text
        data = json.loads(text)["data"]
        assert not Path(data.get("path") or "x").is_absolute()


def test_nb_add_no_store_reads_the_file_from_inside_the_repository(store: Store, tmp_path: Path):
    repo = store.root.parent
    draft = write(repo / "notebooks" / "draft.ipynb", '{"cells": []}')
    r = cli(store, "nb", "add", str(draft), "--no-store")
    nb = fresh(store).notebooks[util.sha256_file(draft)]
    assert not nb.stored and (repo / nb.path).resolve() == draft.resolve()  # audits can find it again
    assert r.output.splitlines()[1] == (
        "  bytes not stored: audits read notebooks/draft.ipynb in the repository while it keeps this content"
    )
    outside = write(tmp_path / "elsewhere" / "other.ipynb", '{"cells": [2]}')
    r = cli(store, "nb", "add", str(outside), "--no-store", code=1)
    assert r.output.startswith("Error: ") and r.output.count("\n") == 1
    assert "outside the repository" in r.output and "audits read" not in r.output
    assert "must be inside the repository" in flat(cli(store, "nb", "add", "--help").output)


def test_nb_add_refuses_an_ipynb_that_is_not_notebook_json(store: Store):
    """A Git LFS pointer checked out without `git lfs pull` must not become the latest notebook."""
    repo = store.root.parent
    pointer = f"version https://git-lfs.github.com/spec/v1\noid sha256:{'ab' * 32}\nsize 48213\n"
    lfs = write(repo / "notebooks" / "big.ipynb", pointer)
    for extra in ([], ["--no-store"]):
        r = cli(store, "nb", "add", str(lfs), *extra, code=1)
        assert r.output.count("\n") == 1 and "Traceback" not in r.output  # one line
        assert r.output.startswith("Error: notebooks/big.ipynb is not notebook JSON")
        assert "git-lfs" in r.output and "Git LFS pointer" in r.output
    assert fresh(store).events == []
    assert [p.name for p in (store.root / "blobs").rglob("*")] == [".gitkeep"]


def test_nb_add_again_says_when_it_stored_the_bytes(store: Store):
    repo = store.root.parent
    cand = write(repo / "cand.ipynb", '{"cells": []}')
    sha = util.sha256_file(cand)
    cli(store, "nb", "add", str(cand))
    store.blob_path(sha).unlink()  # the event was pushed without its blob
    r = cli(store, "nb", "add", str(cand))
    assert r.output == f"already registered as cand.ipynb {sha[:10]}; stored its bytes, which were missing\n"
    assert store.get_blob(sha) == cand.read_bytes()

    draft = write(repo / "draft.ipynb", '{"cells": [1]}')
    dsha = util.sha256_file(draft)
    cli(store, "nb", "add", str(draft), "--no-store")
    r = cli(store, "nb", "add", str(draft), "--tag", "candidate")
    assert r.output == f"already registered as draft.ipynb {dsha[:10]}; updated tags; stored its bytes\n"
    assert fresh(store).notebooks[dsha].stored and store.has_blob(dsha)
    assert cli(store, "nb", "add", str(draft), "--no-store").output.endswith(" (no changes)\n")


def test_nb_add_merges_metadata_and_validates_references(store: Store):
    cli(store, "idea", "add", "fast-prompt", "--title", "t")
    repo = store.root.parent
    base = write(repo / "base.ipynb", '{"cells": []}')
    nb_path = write(repo / "main.ipynb", '{"cells": [1]}')
    cli(store, "nb", "add", str(base), "--tag", "baseline")
    cli(
        store,
        *("nb", "add", str(nb_path), "--parent", "base.ipynb", "--applies", "fast"),
        *("--kernel", "someone/main", "--version", "3", "--note", "first try"),
    )
    sha = util.sha256_file(nb_path)
    nb = fresh(store).notebooks[sha]
    assert (nb.parent, nb.applies, nb.kernel) == (util.sha256_file(base), ["fast-prompt"], "someone/main")
    assert (nb.version, nb.note) == ("3", "first try")

    r = cli(store, "nb", "add", str(nb_path), "--tag", "candidate")
    assert r.output.strip() == f"already registered as main.ipynb {sha[:10]}; updated tags"
    r = cli(store, "nb", "add", str(nb_path), "--name", "renamed.ipynb")
    assert r.stdout.strip() == f"already registered as main.ipynb {sha[:10]} (no changes)"
    assert "kept the name 'main.ipynb'" in r.stderr

    def error(*args: str) -> str:
        return cli(store, "nb", "add", str(nb_path), *args, code=1).output

    assert "no idea matches 'nope'" in error("--applies", "nope")
    assert "no notebook matches 'nope'" in error("--parent", "nope")
    assert "own parent" in error("--parent", sha[:8])
    assert "OWNER/SLUG" in error("--kernel", "no-slash")
    assert "--name cannot be empty" in error("--name", " ")

    notes = write(repo / "notes.txt", "hello")
    r = cli(store, "nb", "add", str(notes))
    assert "neither a .ipynb nor a .py" in r.stderr and "registered notes.txt" in r.stdout


def test_nb_list_columns_filters_and_json(store: Store):
    assert "no notebooks yet" in cli(store, "nb", "list").output
    a = add_nb(store, b"one", name="base.ipynb", tags=["baseline"])
    b = add_nb(store, b"two", name="main.ipynb", parent=a, applies=["x", "y"], tags=["candidate", "gpu"])
    c = add_nb(store, b"three")
    day = fresh(store).notebooks[a].added[:10]

    out = cli(store, "nb", "list").output
    assert out.splitlines()[0].split() == ["#", "SHA", "NAME", "TAGS", "PARENT", "APPLIES", "ADDED", "BY"]
    assert table(out) == [
        ["1", a[:10], "base.ipynb", "baseline", "-", "0", day, "tester"],
        ["2", b[:10], "main.ipynb", "candidate, gpu", "base.ipynb", "2", day, "tester"],
        ["3", c[:10], "-", "-", "-", "0", day, "tester"],
    ]
    out = cli(store, "nb", "list", "-n", "1").output
    assert [row[0] for row in table(out)] == ["3"] and "(2 older notebooks not shown)" in out
    assert [row[0] for row in table(cli(store, "nb", "list", "--tag", "candidate").output)] == ["2"]

    data = json.loads(cli(store, "nb", "list", "--json").stdout)
    assert [d["id"] for d in data] == [a, b, c] and set(data[0]) == field_names(Notebook)
    assert data[1]["parent"] == a and data[1]["applies"] == ["x", "y"]


def test_nb_show_lists_claimed_and_audited_ideas(store: Store):
    for iid, status in (("fast-prompt", "adopted"), ("kv-cache", "testing"), ("unrelated", "idea")):
        store.append("idea.add", {"id": iid, "title": "t", "status": status})
    a = add_nb(store, b"one", name="base.ipynb")
    b = add_nb(store, b"two", name="main.ipynb", parent=a, applies=["kv-cache"], path="nbs/main.ipynb")
    add_verdict(store, b, "fast-prompt", "absent", "fingerprint", note="no match")
    store.append("sub.upsert", {"id": "101", "nb": b, "public_score": 0.45})

    out = cli(store, "nb", "show", "main.ipynb").output
    assert "nbs/main.ipynb" in out and f"base.ipynb {a[:10]}" in out and "101 (public 0.45)" in out
    assert out.splitlines()[0].startswith("notebook") and b in out
    assert table(out) == [
        ["kv-cache", "testing", "c claimed", "-", "-"],
        ["fast-prompt", "adopted", "✗ absent", "fingerprint", "no match"],
    ]

    data = json.loads(cli(store, "nb", "show", b[:8], "--json").stdout)
    assert set(data) == {"notebook", "children", "submissions", "ideas"}
    assert set(data["notebook"]) == field_names(Notebook)
    assert (data["notebook"]["id"], data["notebook"]["name"], data["notebook"]["parent"]) == (
        b,
        "main.ipynb",
        a,
    )
    assert [s["id"] for s in data["submissions"]] == ["101"]
    assert [(i["idea"], i["claimed"]) for i in data["ideas"]] == [("kv-cache", True), ("fast-prompt", False)]
    assert data["ideas"][0]["verdict"] is None and data["ideas"][1]["verdict"]["status"] == "absent"
    assert json.loads(cli(store, "nb", "show", "base.ipynb", "--json").stdout)["children"] == [b]
    assert "no verdicts yet" in cli(store, "nb", "show", "base.ipynb").output
    assert "no notebook matches 'nope'" in cli(store, "nb", "show", "nope", code=1).output


def test_evidence_columns_keep_the_matched_text_in_view(store: Store):
    """A long quote is cut around the match (the evidence's col), not from the start of the line."""
    en = 'SYSTEM_PROMPT = "You solve math word problems. Think step by step, briefly. Answer with a number."'
    ja = 'PROMPT = "あなたは数学の文章題を解きます。手順を短く考えてください。答えだけを書いてください。"'
    nb = add_nb(store, b"one", name="next.ipynb")
    for iid, quote, needle in (("short-cot", en, "Think step"), ("ja-cot", ja, "手順を短く")):
        store.append("idea.add", {"id": iid, "title": "t", "fingerprints": [needle]})
        evidence = [{"path": "files/solver.py", "line": 3, "quote": quote, "col": quote.index(needle)}]
        add_verdict(store, nb, iid, "present", "fingerprint", evidence=evidence)
    odd_cols = {"no-col": None, "str-col": "5", "bool-col": True, "neg-col": -3, "far-col": 10**9}
    for iid, col in odd_cols.items():  # older verdicts have no col; a teammate's could hold anything
        store.append("idea.add", {"id": iid, "title": "t"})
        ev = {"path": "files/solver.py", "line": 3, "quote": en, **({} if col is None else {"col": col})}
        add_verdict(store, nb, iid, "present", "fingerprint", evidence=[ev])

    cells = {row[0]: row[4] for row in table(cli(store, "nb", "show", "next.ipynb").stdout)}
    assert cells["short-cot"].startswith("…") and "Think step by step, briefly." in cells["short-cot"]
    assert cells["ja-cot"].startswith("…") and "手順を短く" in cells["ja-cot"]
    for iid in odd_cols:  # cut from the start, as before
        assert cells[iid].startswith('SYSTEM_PROMPT = "You solve'), iid
    assert all(util.display_width(cell) <= 50 for cell in cells.values())

    shown = table(cli(store, "idea", "show", "short-cot").stdout)[0][4]
    assert shown.startswith("…") and "Think step by step, briefly." in shown


def test_nb_show_before_any_notebook_is_registered(store: Store):
    assert "no notebooks registered yet" in cli(store, "nb", "show", "latest", code=1).output


def test_nb_update_records_only_the_changed_fields(store: Store):
    store.append("idea.add", {"id": "fast-prompt", "title": "t"})
    store.append("idea.add", {"id": "kv-cache", "title": "t"})
    a = add_nb(store, b"one", name="base.ipynb")
    b = add_nb(store, b"two", name="main.ipynb", applies=["kv-cache"], tags=["gpu"])
    r = cli(
        store,
        *("nb", "update", "main.ipynb", "--name", "main.ipynb", "--parent", "base.ipynb"),
        *("--add-applies", "fast", "--remove-applies", "kv-cache", "--add-tag", "candidate"),
        *("--kernel", "someone/main"),
    )
    assert r.output.startswith(f"updated main.ipynb {b[:10]}:")
    assert fresh(store).events[-1].data == {
        "id": b,
        "parent": a,
        "applies": ["fast-prompt"],
        "tags": ["gpu", "candidate"],
        "kernel": "someone/main",
    }
    cli(store, "nb", "update", b[:8], "--no-parent", "--kernel", "", "--note", "kept for reference")
    nb = fresh(store).notebooks[b]
    assert (nb.parent, nb.kernel, nb.note) == (None, "", "kept for reference")


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["main.ipynb"], "give at least one option"),
        (["main.ipynb", "--add-tag", "gpu"], "nothing to change"),
        (["main.ipynb", "--parent", "main.ipynb"], "cannot be its own parent"),
        (["base.ipynb", "--parent", "main.ipynb"], "would create a loop"),
        (["main.ipynb", "--add-applies", "nope"], "no idea matches 'nope'"),
        (["main.ipynb", "--parent", "base.ipynb", "--no-parent"], "not both"),
        (["main.ipynb", "--kernel", "bad"], "OWNER/SLUG"),
        (["main.ipynb", "--name", ""], "--name cannot be empty"),
        (["nope", "--note", "x"], "no notebook matches 'nope'"),
    ],
)
def test_nb_update_errors(store: Store, args: list[str], message: str):
    store.append("idea.add", {"id": "fast-prompt", "title": "t"})
    a = add_nb(store, b"one", name="base.ipynb")
    add_nb(store, b"two", name="main.ipynb", parent=a, tags=["gpu"])
    count = len(fresh(store).events)
    assert message in cli(store, "nb", "update", *args, code=1).output
    assert len(fresh(store).events) == count


def fake_tree() -> Tree:
    t = Tree("synthetic", warnings=["skipped a payload over the size limit"])
    t.add_text("cells/cell_000.py", "import os\nMAX_TOKENS = 512\nprint(MAX_TOKENS)\n", "cell 0")
    t.add_text("markdown/cell_001.md", "Notes on max_tokens\n", "cell 1")
    t.add_text("files/src/agent.py", "def run(x):\n    return x.*y\n", "cell 2 > %%writefile src/agent.py")
    t.add_bytes("payloads/cell_003/0.bin", b"\x00\x01\x02", "cell 3 > line 1 str > base64")
    return t


@pytest.fixture
def tree_calls(store: Store, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Registers one notebook and replaces the extractor with a hand-built tree; records the ids asked for."""
    add_nb(store, b"notebook bytes", name="main.ipynb")
    calls: list[str] = []

    def tree_for(_store: Store, nb: Notebook) -> Tree:
        calls.append(nb.id)
        return fake_tree()

    monkeypatch.setattr("pseudorca.extract.tree_for", tree_for)
    return calls


def test_nb_tree_lists_files_and_extraction_warnings(store: Store, tree_calls: list[str]):
    r = cli(store, "nb", "tree", "main.ipynb")
    assert tree_calls == [util.sha256_bytes(b"notebook bytes")]
    assert r.stdout.splitlines()[0].split() == ["PATH", "SIZE", "ORIGIN"]
    rows = {row[0]: row for row in table(r.stdout)}
    assert list(rows) == [
        "cells/cell_000.py",
        "files/src/agent.py",
        "markdown/cell_001.md",
        "payloads/cell_003/0.bin",
    ]
    assert rows["payloads/cell_003/0.bin"][1:] == ["3 B (binary)", "cell 3 > line 1 str > base64"]
    assert rows["files/src/agent.py"][2] == "cell 2 > %%writefile src/agent.py"
    assert "warning: skipped a payload over the size limit" in r.stderr
    assert [row[0] for row in table(cli(store, "nb", "tree", "latest", "--path", "cells/*").stdout)] == [
        "cells/cell_000.py"
    ]
    assert "no files match 'nothing/*'" in cli(store, "nb", "tree", "latest", "--path", "nothing/*").output


def test_nb_tree_grep_searches_plain_text(store: Store, tree_calls: list[str]):
    def grep(*args: str) -> list[str]:
        return cli(store, "nb", "tree", "latest", "--grep", *args).stdout.splitlines()

    assert grep("MAX_TOKENS") == [
        "cells/cell_000.py:2: MAX_TOKENS = 512",
        "cells/cell_000.py:3: print(MAX_TOKENS)",
    ]
    assert "markdown/cell_001.md:1: Notes on max_tokens" in grep("max_tokens", "--ignore-case")
    assert grep("max_tokens") == ["markdown/cell_001.md:1: Notes on max_tokens"]
    assert grep("x.*y") == ["files/src/agent.py:2:     return x.*y"]  # a literal, not a regex
    assert grep("print(") == ["cells/cell_000.py:3: print(MAX_TOKENS)"]
    assert grep("MAX_TOKENS", "--path", "files/*") == []
    r = cli(store, "nb", "tree", "latest", "--grep", "x.+y")
    assert r.stdout == "" and "no line contains" in r.stderr
    assert "--grep needs some text" in cli(store, "nb", "tree", "latest", "--grep", "", code=1).output
    assert "no notebook matches 'nope'" in cli(store, "nb", "tree", "nope", code=1).output


# ----- status --------------------------------------------------------------------------------------


@pytest.fixture
def board(store: Store) -> list[str]:
    """Four ideas and three notebooks with one verdict of every kind; returns the notebook ids."""
    for iid, status, tags in (
        ("a-idea", "adopted", ["prompt"]),
        ("b-idea", "testing", []),
        ("c-idea", "idea", ["prompt"]),
        ("d-idea", "parked", []),
    ):
        store.append(
            "idea.add", {"id": iid, "title": iid, "status": status, "tags": tags, "fingerprints": ["X"]}
        )
    n1 = add_nb(store, b"one", name="base.ipynb")
    n2 = add_nb(store, b"two", name="main.ipynb")
    n3 = add_nb(store, b"three", name="a-very-long-notebook-name.ipynb", applies=["b-idea", "a-idea"])
    add_verdict(store, n1, "a-idea", "present", "fingerprint")
    add_verdict(store, n2, "a-idea", "absent", "fingerprint")
    add_verdict(store, n3, "a-idea", "present", "fingerprint")
    add_verdict(store, n3, "a-idea", "absent", "patch")
    add_verdict(store, n1, "b-idea", "partial", "fingerprint")
    add_verdict(store, n2, "b-idea", "unknown", "llm")
    add_verdict(store, n1, "c-idea", "present", "fingerprint")
    store.append("idea.update", {"id": "c-idea", "description": "changed after the audit"})  # -> stale
    return [n1, n2, n3]


def test_status_board_shows_every_symbol(store: Store, board: list[str]):
    n1, _, n3 = board
    r = cli(store, "status")
    header = re.split(r" {2,}", r.stdout.splitlines()[0])
    assert header == ["IDEA", "STATUS", "1 base.ipynb", "2 main.ipynb", "3 a-very-long…"]
    assert table(r.stdout) == [
        ["c-idea", "idea", "✓*", "·", "·"],
        ["b-idea", "testing", "◐", "?", "c"],
        ["a-idea", "adopted", "✓", "✗", "!"],
    ]
    assert "(1 parked idea hidden; use --all)" in r.stdout
    assert LEGEND in r.stdout
    assert f"3  a-very-long-notebook-name.ipynb  {n3[:10]}" in r.stdout
    assert re.search(rf"1\s+base\.ipynb\s+{n1[:10]}", r.stdout)


def test_status_columns_and_filters(store: Store, board: list[str]):
    n3 = board[2]

    def rows(*args: str) -> dict[str, list[str]]:
        return {row[0]: row[2:] for row in table(cli(store, "status", *args).stdout)}

    assert rows("-n", "2")["a-idea"] == ["✗", "!"]
    assert rows("-n", "0")["a-idea"] == ["✓", "✗", "!"]  # 0 = all, as in the other commands
    assert rows("--nb", "base.ipynb", "--nb", n3[:8], "--nb", "base.ipynb")["a-idea"] == ["✓", "!"]
    assert rows("--nb", "latest", "--nb", "base.ipynb")["a-idea"] == ["!", "✓"]
    assert rows("--all")["d-idea"] == ["·", "·", "·"]
    assert list(rows("--status", "parked")) == ["d-idea"]
    assert list(rows("--tag", "prompt")) == ["c-idea", "a-idea"]
    assert "no ideas match" in cli(store, "status", "--tag", "nothing").output
    assert "no notebook matches 'nope'" in cli(store, "status", "--nb", "nope", code=1).output


def test_status_json_shape(store: Store, board: list[str]):
    n1, n2, n3 = board
    data = json.loads(cli(store, "status", "--json").stdout)
    assert set(data) == {"notebooks", "rows"}
    assert [nb["id"] for nb in data["notebooks"]] == [n1, n2, n3]
    assert set(data["notebooks"][0]) == field_names(Notebook)
    rows = {row["idea"]["id"]: row["cells"] for row in data["rows"]}
    assert list(rows) == ["c-idea", "b-idea", "a-idea"]
    assert set(data["rows"][0]["idea"]) == field_names(Idea)
    assert rows["b-idea"][2] == {
        "nb": n3,
        "symbol": "c",
        "status": None,
        "method": None,
        "stale": False,
        "claimed": True,
    }
    assert rows["c-idea"][0] == {
        "nb": n1,
        "symbol": "✓*",
        "status": "present",
        "method": "fingerprint",
        "stale": True,
        "claimed": False,
    }
    assert rows["a-idea"][2]["status"] == "conflict" and rows["a-idea"][2]["claimed"]


def test_status_shows_store_warnings_and_empty_ledgers(store: Store):
    assert "no ideas yet" in cli(store, "status").output
    store.append("idea.add", {"id": "x", "title": "t"})
    month = next(p for p in (store.root / "events").iterdir() if p.is_dir())
    future = {"schema": 2, "id": "ZZZZ", "type": "idea.rename", "time": "t", "actor": "a", "data": {}}
    (month / "ZZZZ-future.json").write_text(json.dumps(future))
    r = cli(store, "status")
    assert table(r.stdout) == [["x", "idea"]] and "no notebooks yet" in r.stdout
    assert "idea.rename" in r.stderr


# ----- sub -----------------------------------------------------------------------------------------


def test_sub_add_list_and_link(store: Store):
    assert "no submissions yet" in cli(store, "sub", "list").output
    a = add_nb(store, b"one", name="base.ipynb")
    add_nb(store, b"two", name="main.ipynb")
    r = cli(
        store,
        *("sub", "add", "101", "--nb", "base.ipynb", "--score", "0.451", "--desc", "first try"),
        *("--date", "2026-01-02"),
    )
    assert r.output.strip() == "added submission 101: public=0.451, nb=base.ipynb, description=first try"
    s = fresh(store).subs["101"]
    assert (s.nb, s.public_score, s.source, s.submitted_by, s.competition, s.date) == (
        a,
        0.451,
        "manual",
        "tester",
        "some-competition",
        "2026-01-02",
    )
    cli(store, "sub", "add", "99", "--score", "0.46")
    assert fresh(store).subs["99"].date == date.today().isoformat()
    cli(store, "sub", "add", "100", "--date", "2026-01-02", "--competition", "other", "--status", "complete")
    assert fresh(store).subs["100"].competition == "other"

    r = cli(store, "sub", "add", "101", "--private-score", "0.44", "--score", "0.451")
    assert r.output.strip() == "updated submission 101: private=0.44"
    assert fresh(store).events[-1].data == {"id": "101", "private_score": 0.44}
    assert fresh(store).subs["101"].source == "manual"

    out = cli(store, "sub", "list").output
    assert out.splitlines()[0].split() == [
        "ID",
        "DATE",
        "NOTEBOOK",
        "PUBLIC",
        "PRIVATE",
        "STATUS",
        "BY",
        "DESCRIPTION",
    ]
    rows = table(out)
    assert [row[0] for row in rows] == ["100", "101", "99"]  # by date, then id; newest last
    assert rows[1] == ["101", "2026-01-02", "base.ipynb", "0.451", "0.44", "-", "tester", "first try"]
    assert rows[0][5] == "complete" and rows[2][2] == "-"

    assert "no submission '555'" in cli(store, "sub", "link", "555", "main.ipynb", code=1).output
    assert "no notebook matches 'nope'" in cli(store, "sub", "link", "99", "nope", code=1).output
    sha = fresh(store).resolve_nb("main.ipynb").id
    assert (
        cli(store, "sub", "link", "99", "main.ipynb").output.strip()
        == f"linked submission 99 to main.ipynb {sha[:10]}"
    )
    assert fresh(store).events[-1].data == {"id": "99", "nb": sha}
    assert "already linked" in cli(store, "sub", "link", "99", "main.ipynb").output
    assert [row[0] for row in table(cli(store, "sub", "list", "--nb", "main.ipynb").output)] == ["99"]
    out = cli(store, "sub", "list", "-n", "1").output
    assert [row[0] for row in table(out)] == ["99"] and "(2 older submissions not shown)" in out

    data = json.loads(cli(store, "sub", "list", "--json").stdout)
    assert [d["id"] for d in data] == ["100", "101", "99"] and set(data[0]) == field_names(Submission)
    assert data[1]["public_score"] == 0.451 and data[1]["private_score"] == 0.44


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["a b"], "invalid submission id"),
        (["x1", "--score", "nan"], "--score must be a finite number"),
        (["x1", "--private-score", "inf"], "--private-score must be a finite number"),
        (["x1", "--date", "yesterday"], "is not an ISO date"),
        (["x1", "--nb", "nope"], "no notebooks registered yet"),
    ],
)
def test_sub_add_errors(store: Store, args: list[str], message: str):
    assert message in cli(store, "sub", "add", *args, code=1).output
    assert fresh(store).subs == {}


def test_sub_add_refuses_a_no_op_change(store: Store):
    cli(store, "sub", "add", "7", "--score", "0.5")
    assert "nothing to change" in cli(store, "sub", "add", "7", "--score", "0.5", code=1).output


def test_submissions_written_by_other_tools_with_empty_fields_still_list(store: Store):
    store.append(
        "sub.upsert", {"id": "k1", "date": None, "public_score": "", "nb": "f" * 64, "source": "kaggle"}
    )
    store.append("sub.upsert", {"id": "k2", "date": "2026-01-03T10:00:00", "public_score": "0.5"})
    assert table(cli(store, "sub", "list").output) == [
        ["k1", "-", "ffffffffff", "-", "-", "-", "-", "-"],
        ["k2", "2026-01-03 10:00", "-", "0.5", "-", "-", "-", "-"],
    ]


# ----- log -----------------------------------------------------------------------------------------


def test_log_shows_one_line_per_event_newest_first(store: Store):
    assert "no events" in cli(store, "log").output
    assert json.loads(cli(store, "log", "--json").stdout) == []
    cli(store, "idea", "add", "fast-prompt", "--title", "Shorter prompt")
    cli(store, "idea", "update", "fast-prompt", "--status", "adopted")
    sha = add_nb(store, b"one", name="main.ipynb", tags=["candidate"])
    add_verdict(store, sha, "fast-prompt", "present", "patch")
    store.append("sub.upsert", {"id": "123", "public_score": 0.451})
    cli(store, "--as", "bob", "sub", "link", "123", "main.ipynb")
    cli(store, "nb", "update", "main.ipynb", "--add-applies", "fast-prompt")

    out = cli(store, "log").stdout
    assert out.splitlines()[0].split() == ["TIME", "ACTOR", "TYPE", "SUMMARY"]
    rows = table(out)
    assert [row[3] for row in rows] == [
        f"nb main.ipynb {sha[:10]}: applies=fast-prompt",
        "sub 123: nb=main.ipynb",
        "sub 123: public=0.451",
        "verdict present (patch): idea fast-prompt on main.ipynb",
        f"nb main.ipynb {sha[:10]}: tags=candidate",
        "idea fast-prompt: status=adopted",
        "idea fast-prompt added: Shorter prompt",
    ]
    assert [row[1] for row in rows[:3]] == ["tester", "bob", "tester"]
    assert [row[2] for row in rows[:2]] == ["nb.update", "sub.upsert"]
    assert re.fullmatch(r"\d{4}-\d\d-\d\d \d\d:\d\d", rows[0][0])

    def types(*args: str) -> list[str]:
        return [row[2] for row in table(cli(store, "log", *args).stdout)]

    assert types("--type", "idea") == ["idea.update", "idea.add"]
    assert types("--type", "idea.add") == ["idea.add"]
    assert types("--type", "audit") == ["audit.verdict"]
    assert types("--actor", "bob") == ["sub.upsert"]
    assert len(types("-n", "2")) == 2 and len(types("-n", "0")) == 7
    assert "unknown event type 'nope'" in cli(store, "log", "--type", "nope", code=1).output

    data = json.loads(cli(store, "log", "--json", "-n", "2").stdout)
    assert [(d["type"], d["actor"], d["summary"]) for d in data] == [
        ("nb.update", "tester", f"nb main.ipynb {sha[:10]}: applies=fast-prompt"),
        ("sub.upsert", "bob", "sub 123: nb=main.ipynb"),
    ]
    assert set(data[0]) == {"schema", "id", "type", "time", "actor", "data", "summary"}
    assert data[1]["data"] == {"id": "123", "nb": sha} and data[0]["id"] > data[1]["id"]
    assert [d["type"] for d in json.loads(cli(store, "log", "--json", "--type", "idea").stdout)] == [
        "idea.update",
        "idea.add",
    ]


def test_times_show_in_the_viewers_time_zone_and_order_by_instant(
    store: Store, viewer_tz: Callable[[str], None]
):
    """Berlin, Tokyo and Los Angeles write events seconds apart, each stamped with its own UTC offset."""
    put_event(store, "idea.add", {"id": "retry", "title": "Retry", "fingerprints": ["X"]},
              "2026-10-22T11:30:23+02:00", "berlin")  # fmt: skip
    nb, late = util.sha256_bytes(b"main"), util.sha256_bytes(b"late")
    put_event(store, "nb.add", {"id": nb, "name": "main.ipynb"}, "2026-10-22T11:30:24+02:00", "berlin")
    verdict = {"nb": nb, "idea": "retry", "idea_rev": Idea("retry", "Retry", fingerprints=["X"]).rev}
    put_event(store, "audit.verdict", {**verdict, "status": "present", "method": "fingerprint"},
              "2026-10-22T18:30:26+09:00", "tokyo")  # 09:30:26 UTC  # fmt: skip
    put_event(store, "audit.verdict", {**verdict, "status": "present", "method": "patch"},
              "2026-10-22T11:31:00+02:00", "berlin")  # 09:31 UTC: the later one  # fmt: skip
    put_event(store, "idea.update", {"id": "retry", "owner": "la"}, "2026-10-22T02:32:00-07:00", "la")
    put_event(store, "nb.add", {"id": late, "name": "late.ipynb"}, "2026-10-23T01:00:00+09:00", "tokyo")
    put_event(store, "sub.upsert", {"id": "7", "date": "2026-10-23T08:00:00+09:00"},
              "2026-10-23T08:00:05+09:00", "tokyo")  # 2026-10-22 23:00 UTC  # fmt: skip
    put_event(store, "sub.upsert", {"id": "8", "date": "2026-10-22T20:00:00-05:00"},
              "2026-10-22T20:00:05-05:00", "la")  # 2026-10-23 01:00 UTC  # fmt: skip

    viewer_tz("UTC0")
    assert [row[:3] for row in table(cli(store, "log", "-n", "0").stdout)] == [
        ["2026-10-23 01:00", "la", "sub.upsert"],
        ["2026-10-22 23:00", "tokyo", "sub.upsert"],
        ["2026-10-22 16:00", "tokyo", "nb.add"],
        ["2026-10-22 09:32", "la", "idea.update"],
        ["2026-10-22 09:31", "berlin", "audit.verdict"],
        ["2026-10-22 09:30", "tokyo", "audit.verdict"],
        ["2026-10-22 09:30", "berlin", "nb.add"],
        ["2026-10-22 09:30", "berlin", "idea.add"],
    ]
    out = cli(store, "idea", "show", "retry").stdout
    assert "2026-10-22 09:30 by berlin" in out and "2026-10-22 09:32 by la" in out
    assert table(out)[0][1:4] == ["✓ present", "fingerprint+patch", "2026-10-22 09:31"]  # WHEN: the newest
    assert table(cli(store, "idea", "list").stdout)[0][5] == "2026-10-22"
    assert [row[6] for row in table(cli(store, "nb", "list").stdout)] == ["2026-10-22", "2026-10-22"]
    assert "2026-10-22 16:00 by tokyo" in cli(store, "nb", "show", "late.ipynb").stdout
    key = cli(store, "status").stdout.split(LEGEND)[1]
    assert [row[3] for row in table(key)] == ["2026-10-22", "2026-10-22"]
    subs = [row[:2] for row in table(cli(store, "sub", "list").stdout)]
    assert subs == [["7", "2026-10-22 23:00"], ["8", "2026-10-23 01:00"]]  # by instant, not by text

    viewer_tz("JST-9")
    assert table(cli(store, "log", "-n", "1").stdout)[0][0] == "2026-10-23 10:00"
    assert table(cli(store, "idea", "show", "retry").stdout)[0][3] == "2026-10-22 18:31"
    assert [row[6] for row in table(cli(store, "nb", "list").stdout)] == ["2026-10-22", "2026-10-23"]
    assert [row[:2] for row in table(cli(store, "sub", "list").stdout)] == [
        ["7", "2026-10-23 08:00"],
        ["8", "2026-10-23 10:00"],
    ]


def test_times_that_cannot_be_converted_are_shown_as_written(store: Store):
    put_event(store, "idea.add", {"id": "x", "title": "t"}, "0001-01-01T00:00:00+05:00", "old")
    put_event(store, "sub.upsert", {"id": "9", "date": "9999-12-31T23:59:59-05:00"}, "sometime", "odd")
    assert [row[0] for row in table(cli(store, "log").stdout)] == ["sometime", "0001-01-01 00:00"]
    assert table(cli(store, "sub", "list").stdout)[0][1] == "9999-12-31 23:59"
    assert "0001-01-01 00:00 by old" in cli(store, "idea", "show", "x").stdout


@pytest.mark.parametrize(
    ("command", "wanted"),
    [
        (["init"], ["--name NAME"]),
        (["idea", "add"], ["--owner NAME", "--fingerprint FINGERPRINT"]),
        (
            ["idea", "update"],
            ["--owner NAME", "--add-fingerprint FINGERPRINT", "--remove-fingerprint FINGERPRINT"],
        ),
        (["idea", "list"], ["--tag TAG", "--owner NAME"]),
        (["nb", "add"], ["--name NAME", "--tag TAG"]),
        (["nb", "list"], ["--tag TAG", "-n N", "(0 = all)"]),
        (["nb", "update"], ["--name NAME", "--add-tag TAG"]),
        (["status"], ["--tag TAG", "-n N", "(0 = all)"]),
        (["sub", "add"], ["--status TEXT", "Kaggle"]),
        (["sub", "list"], ["-n N", "(0 = all)"]),
        (["sub", "link"], ["sub link [OPTIONS] ID REF"]),
        (["log"], ["-n N", "(0 = all)", "--json"]),
    ],
)
def test_options_follow_one_convention(store: Store, command: list[str], wanted: list[str]):
    """-n N with 0 = all, --json on the listings, and word metavars (TAG, NAME, REF) rather than letters."""
    raw = cli(store, *command, "--help").output
    for text in wanted:
        assert text in flat(raw), text
    assert not re.search(r"--[\w-]+ [A-Z]\b(?!/)", raw)  # no one-letter metavar such as `--tag T`
