from __future__ import annotations

import ast
import enum
import importlib.util
import json
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from click.testing import CliRunner

from pseudorca import kaggle_api
from pseudorca.cli import main
from pseudorca.kaggle_api import (
    ALLOWED_CALLS,
    KaggleError,
    list_submissions,
    normalize_submission,
    pull_kernel,
    sync_submissions,
)
from pseudorca.notebooks import register_bytes
from pseudorca.store import Store

COMPETITION = "some-competition"  # the store fixture's [store].competition
METADATA = "kernel-metadata.json"
REAL_GET_API = kaggle_api.get_api


@pytest.fixture(autouse=True)
def _no_real_kaggle(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail any test that would reach the real Kaggle client."""

    def refuse() -> Any:
        raise AssertionError("a test reached the real Kaggle client")

    monkeypatch.setattr(kaggle_api, "get_api", refuse)


# ----- fakes shaped like the kaggle client -----------------------------------------------------------


class Status(enum.Enum):
    """Shaped like kagglesdk's SubmissionStatus."""

    PENDING = 0
    COMPLETE = 1
    ERROR = 2


class FakeSubmission:
    """Shaped like kagglesdk's ApiSubmission: "" for unset text and scores, naive UTC dates, an enum status."""

    def __init__(
        self,
        ref: int,
        description: str = "",
        public_score: str = "",
        private_score: str = "",
        status: Status = Status.COMPLETE,
        day: int = 1,
        **extra: str,
    ) -> None:
        self.ref = ref
        self.total_bytes = 2048
        self.date = datetime(2026, 9, day, 12, 30, 15, 123000)
        self.description = description
        self.error_description = extra.get("error_description", "")
        self.file_name = extra.get("file_name", "submission.csv")
        self.public_score = public_score
        self.private_score = private_score
        self.status = status
        self.submitted_by = extra.get("submitted_by", "alice")
        self.submitted_by_ref = ""
        self.team_name = extra.get("team_name", "Team Orca")
        self.url = extra.get("url", f"/submissions/{ref}")


class StrictApi:
    """A fake KaggleApi with only ALLOWED_CALLS: touching any other attribute fails the test.

    It pages like the real client (page_number -1 means page 1). `server_cap` makes the server send fewer
    than page_size items; `ignores_page` makes it always send the first page.
    """

    def __init__(
        self,
        submissions: list[FakeSubmission] | tuple[()] = (),
        *,
        server_cap: int | None = None,
        ignores_page: bool = False,
        files: dict[str, str] | None = None,
        pull_error: Exception | None = None,
    ) -> None:
        fake = SimpleNamespace(
            subs=list(submissions),
            cap=server_cap,
            ignores_page=ignores_page,
            files=files or {},
            pull_error=pull_error,
            calls=[],
        )
        object.__setattr__(self, "_fake", fake)

    def __getattribute__(self, name: str) -> Any:
        if name in ALLOWED_CALLS or name == "_fake" or (name.startswith("__") and name.endswith("__")):
            return object.__getattribute__(self, name)
        raise AssertionError(f"api.{name} is not in ALLOWED_CALLS")

    def __setattr__(self, name: str, value: Any) -> None:
        raise AssertionError(f"api.{name} was set")

    def authenticate(self) -> None:
        self._fake.calls.append(("authenticate",))

    def competition_submissions(
        self, competition, group=None, sort=None, page_number=-1, page_token="", page_size=20
    ):
        f = self._fake
        f.calls.append(("competition_submissions", competition, page_number, page_size))
        page = 1 if page_number == -1 or f.ignores_page else page_number
        size = min(page_size, f.cap or page_size)
        return f.subs[(page - 1) * size : page * size]

    def kernels_pull(self, kernel, path, metadata=False, quiet=True):
        f = self._fake
        f.calls.append(("kernels_pull", kernel, path, metadata))
        if f.pull_error is not None:
            raise f.pull_error
        for name, text in f.files.items():
            Path(path, name).write_text(text, encoding="utf-8")
        return path


class OldApi(StrictApi):
    """A client whose competition_submissions cannot page."""

    def competition_submissions(self, competition):
        self._fake.calls.append(("competition_submissions", competition))
        return self._fake.subs[:2]


def _subs() -> list[FakeSubmission]:
    """Newest first, as Kaggle lists them."""
    return [
        FakeSubmission(5003, "agent-v2.ipynb: shorter prompt", status=Status.PENDING, day=3),
        FakeSubmission(5002, "baseline rerun", "0.512", "0.498", day=2),
        FakeSubmission(5001, "first try", "0.488", day=1),
    ]


def _notebook(code: str) -> str:
    cell = {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None, "source": code}
    return json.dumps({"cells": [cell], "metadata": {}, "nbformat": 4, "nbformat_minor": 5})


def _kernel_files(text: str | None = None, code_file: str = "demo-agent.ipynb") -> dict[str, str]:
    meta = {
        "id": "someone/demo-agent",
        "id_no": 1,
        "title": "Demo Agent",
        "code_file": code_file,
        "language": "python",
        "kernel_type": "notebook",
    }
    return {code_file: text or _notebook("print('hello')\n"), METADATA: json.dumps(meta)}


def _events(store: Store) -> int:
    return len(list((store.root / "events").glob("*/*.json")))


def _nb(store: Store, data: bytes, name: str) -> str:
    """Register a minimal notebook whose bytes are unique to `data`."""
    if name.endswith(".ipynb"):
        data = json.dumps({"cells": [], "metadata": {"label": data.decode()}}).encode()
    return register_bytes(store, data, name=name)[0].id


def _invoke(store: Store, *args: str):
    return CliRunner().invoke(main, ["--store", str(store.root), *args])


# ----- the client guard ------------------------------------------------------------------------------


def test_only_allowed_calls_reach_the_client():
    api = StrictApi()
    for method in ("competition_submit", "kernels_push", "dataset_create_new", "__init__"):
        with pytest.raises(KaggleError, match="only reads"):
            kaggle_api._call(api, method)
    assert not api._fake.calls


def test_source_calls_no_other_client_methods():
    source = Path(kaggle_api.__file__).read_text(encoding="utf-8")
    direct = set(re.findall(r"(?<![\w.])api\.(\w+)\s*\(", source))
    via_helper = set(re.findall(r"_call\(\s*api\s*,\s*[\"'](\w+)[\"']", source))
    assert direct <= set(ALLOWED_CALLS)
    assert via_helper and via_helper <= set(ALLOWED_CALLS)
    assert len(re.findall(r"getattr\(\s*api\b", source)) == 1  # only the guarded helper looks methods up


def test_calls_fit_the_installed_client():
    """The keywords we pass, checked against the installed client's source (parsed, never imported)."""
    already_imported = "kaggle" in sys.modules
    spec = importlib.util.find_spec("kaggle")
    if spec is None or not spec.submodule_search_locations:
        pytest.skip("the kaggle package is not installed")
    source = Path(next(iter(spec.submodule_search_locations)), "api", "kaggle_api_extended.py")
    tree = ast.parse(source.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "KaggleApi")
    params = {
        f.name: [a.arg for a in f.args.args + f.args.kwonlyargs]
        for f in cls.body
        if isinstance(f, ast.FunctionDef) and f.name in ALLOWED_CALLS
    }
    assert set(params) == set(ALLOWED_CALLS)
    subs = params["competition_submissions"]
    assert subs[1] == "competition" and ("page_number" not in subs or "page_size" in subs)
    assert params["kernels_pull"][1] == "kernel" and {"path", "metadata"} <= set(params["kernels_pull"])
    assert ("kaggle" in sys.modules) == already_imported


def test_importing_the_commands_does_not_import_kaggle():
    code = (
        "import sys, pseudorca.commands.kaggle_cmd\n"
        "print(sorted(m for m in sys.modules if m.split('.')[0] in ('kaggle', 'kagglesdk')))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


def _fake_kaggle_package(monkeypatch: pytest.MonkeyPatch, api_class: type) -> None:
    leaf = ModuleType("kaggle.api.kaggle_api_extended")
    leaf.KaggleApi = api_class  # type: ignore[attr-defined]
    for name, module in (
        ("kaggle", ModuleType("kaggle")),
        ("kaggle.api", ModuleType("kaggle.api")),
        ("kaggle.api.kaggle_api_extended", leaf),
    ):
        monkeypatch.setitem(sys.modules, name, module)


def test_get_api_authenticates_once_and_keeps_stdout_clean(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    class Api:
        calls = 0

        def authenticate(self) -> None:
            type(self).calls += 1
            print("Warning: your Kaggle API key is readable by other users")

    _fake_kaggle_package(monkeypatch, Api)
    api = REAL_GET_API()
    assert isinstance(api, Api) and Api.calls == 1
    out = capsys.readouterr()
    assert out.out == "" and "readable by other users" in out.err


def test_get_api_without_credentials_is_one_clear_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    monkeypatch.setenv("KAGGLE_KEY", "fake-key-1234")

    class NoCredentials:
        def authenticate(self) -> None:
            print("Authentication required. KAGGLE_KEY=fake-key-1234")
            raise SystemExit(1)

    _fake_kaggle_package(monkeypatch, NoCredentials)
    with pytest.raises(KaggleError) as err:
        REAL_GET_API()
    message = str(err.value)
    assert "kaggle.json" in message and "KAGGLE_USERNAME" in message and "kaggle.com/docs/api" in message
    out = capsys.readouterr()
    assert "fake-key-1234" not in message + out.out + out.err


def test_get_api_without_the_client(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(sys.modules, "kaggle.api.kaggle_api_extended", None)
    with pytest.raises(KaggleError, match=r'pip install "pseudorca\[kaggle\] @ git\+https://'):
        REAL_GET_API()


# ----- listing submissions ---------------------------------------------------------------------------


def test_normalize_submission():
    item = normalize_submission(FakeSubmission(42, " tuned  ", "0.71", "", day=5))
    assert item == {
        "id": "42",
        "date": "2026-09-05T12:30:15+00:00",
        "description": "tuned",
        "status": "complete",
        "public_score": 0.71,
        "private_score": None,
        "submitted_by": "alice",
        "team_name": "Team Orca",
        "file_name": "submission.csv",
        "url": "/submissions/42",
        "error_description": "",
    }
    # older clients: dicts with camelCase keys, string statuses, other time zones; junk scores become None
    old = {
        "ref": 7,
        "date": "2026-09-05T21:30:15+09:00",
        "status": "SubmissionStatus.ERROR",
        "publicScore": "nan",
        "privateScore": "n/a",
        "fileName": "x.csv",
    }
    item = normalize_submission(old)
    assert item is not None
    assert item["date"] == "2026-09-05T12:30:15+00:00"
    assert (item["status"], item["file_name"]) == ("error", "x.csv")
    assert item["public_score"] is None and item["private_score"] is None
    assert normalize_submission(None) is None and normalize_submission(FakeSubmission(0)) is None


def test_normalize_real_kagglesdk_submission():
    service = pytest.importorskip("kagglesdk.competitions.types.competition_api_service")
    sub = service.ApiSubmission()
    sub.ref = 99
    sub.date = datetime(2026, 9, 1, 8, 0)
    sub.description = "real type"
    sub.public_score = "0.5"
    sub.status = service.SubmissionStatus.ERROR
    item = normalize_submission(sub)
    assert item is not None
    assert (item["id"], item["status"], item["public_score"]) == ("99", "error", 0.5)
    assert item["private_score"] is None and item["url"] == ""
    assert item["date"] == "2026-09-01T08:00:00+00:00"


@pytest.mark.parametrize("cap", [None, 3])
def test_list_submissions_follows_pages(cap: int | None):
    subs = [FakeSubmission(1000 + i, f"s{i}") for i in range(250)]
    api = StrictApi(subs, server_cap=cap)
    assert [i["id"] for i in list_submissions(api, "comp")] == [str(1000 + i) for i in range(250)]
    calls = api._fake.calls
    assert calls[0][2] == 1 and all(c[3] == kaggle_api.PAGE_SIZE for c in calls)


def test_a_server_that_ignores_pages_ends_the_listing_with_a_note(store: Store):
    api = StrictApi(_subs(), server_cap=2, ignores_page=True)
    assert len(list_submissions(api, "comp")) == 2 and len(api._fake.calls) == 2
    report = sync_submissions(store, COMPETITION, api=StrictApi(_subs(), server_cap=2, ignores_page=True))
    assert len(report.new) == 2 and report.notes == [
        "Kaggle did not page past 2 submissions; older ones may be missing"
    ]
    assert not sync_submissions(store, COMPETITION, api=StrictApi(_subs())).notes  # a full listing


def test_client_without_paging_reads_one_page(store: Store):
    report = sync_submissions(store, COMPETITION, api=OldApi(_subs()))
    assert len(report.new) == 2 and "only the first page" in report.notes[0]


def test_bad_competition_slug(store: Store):
    with pytest.raises(KaggleError, match="competition slug"):
        sync_submissions(store, "https://www.kaggle.com/competitions/x", api=StrictApi())


# ----- sync ------------------------------------------------------------------------------------------


def test_sync_writes_once_then_nothing(store: Store):
    report = sync_submissions(store, COMPETITION, api=StrictApi(_subs()))
    assert report.new == ["5001", "5002", "5003"] and not report.updated and not report.unchanged
    assert _events(store) == 3
    sub = store.state().subs["5002"]
    assert (sub.public_score, sub.private_score, sub.status) == (0.512, 0.498, "complete")
    assert (sub.source, sub.competition, sub.submitted_by) == ("kaggle", COMPETITION, "alice")
    assert sub.date == "2026-09-02T12:30:15+00:00" and sub.description == "baseline rerun"
    assert sub.extra == {"team_name": "Team Orca", "file_name": "submission.csv", "url": "/submissions/5002"}
    assert store.state().subs["5003"].status == "pending"

    again = sync_submissions(store, COMPETITION, api=StrictApi(_subs()))
    assert again.unchanged == ["5001", "5002", "5003"] and not again.new and not again.updated
    assert _events(store) == 3


def test_new_score_is_one_update(store: Store):
    sync_submissions(store, COMPETITION, api=StrictApi(_subs()))
    subs = _subs()
    subs[0].status, subs[0].public_score = Status.COMPLETE, "0.533"
    report = sync_submissions(store, COMPETITION, api=StrictApi(subs))
    assert report.updated == ["5003"] and report.changed == {"5003": ["status", "public_score"]}
    assert _events(store) == 4
    last = Store.at(store.root).events()[-1]
    assert last.data == {"id": "5003", "status": "complete", "public_score": 0.533}
    assert store.state().subs["5003"].public_score == 0.533


def test_sync_keeps_existing_links_and_extra_keys(store: Store):
    _nb(store, b"nb-a", "agent-v2.ipynb")
    manual = _nb(store, b"nb-b", "other.ipynb")
    store.append("sub.upsert", {"id": "5003", "nb": manual, "extra": {"note": "kept"}})  # as `sub link` would
    report = sync_submissions(store, COMPETITION, api=StrictApi(_subs()))
    sub = store.state().subs["5003"]
    assert sub.nb == manual and "5003" in report.updated and "5003" not in report.linked
    assert sub.extra["note"] == "kept" and sub.extra["team_name"] == "Team Orca"
    assert "nb" not in Store.at(store.root).events()[-1].data
    assert not sync_submissions(store, COMPETITION, api=StrictApi(_subs())).updated


def test_links_by_name_and_by_sha_prefix(store: Store):
    hashed = _nb(store, b"nb-hashed", "candidate.ipynb")
    subs = [
        FakeSubmission(1, "agent-v2.ipynb: shorter prompt", day=1),
        FakeSubmission(2, f"retry of {hashed[:8].upper()}", day=2),
        FakeSubmission(3, "agent-v2.ipynbx is another name", day=3),
        FakeSubmission(4, "no notebook named here", day=4),
    ]
    first = sync_submissions(store, COMPETITION, api=StrictApi(subs))
    assert first.linked == ["2"]
    # a notebook registered later links the submission that names it, as an update of `nb` only
    named = _nb(store, b"nb-named", "agent-v2.ipynb")
    report = sync_submissions(store, COMPETITION, api=StrictApi(subs))
    assert report.linked == ["1"] and report.changed == {"1": ["nb"]}
    st = store.state().subs
    assert (st["1"].nb, st["2"].nb, st["3"].nb, st["4"].nb) == (named, hashed, None, None)


def test_ambiguous_descriptions_are_not_linked(store: Store):
    first = _nb(store, b"agent v1", "agent.ipynb")
    _nb(store, b"agent v2", "agent.ipynb")
    _nb(store, b"other", "other.ipynb")
    _nb(store, b"short", "v2")  # names under 4 characters never link
    store.append("nb.add", {"id": "abcdef12" + "0" * 56, "name": "p.ipynb"})
    store.append("nb.add", {"id": "abcdef12" + "1" * 56, "name": "q.ipynb"})
    subs = [
        FakeSubmission(1, "agent.ipynb again", day=1),  # one name, two notebooks
        FakeSubmission(2, "other.ipynb vs agent.ipynb", day=2),  # two names
        FakeSubmission(3, "v2 run", day=3),
        FakeSubmission(4, f"agent.ipynb at {first[:10]}", day=4),  # the sha prefix decides
        FakeSubmission(5, "built from abcdef12", day=5),  # prefix shared by two notebooks
        FakeSubmission(6, f"prefix {first[:7]} is too short", day=6),
    ]
    report = sync_submissions(store, COMPETITION, api=StrictApi(subs))
    st = store.state().subs
    assert [st[i].nb for i in "12356"] == [None] * 5 and st["4"].nb == first
    assert report.ambiguous == ["1", "2", "5"] and report.linked == ["4"]


def test_failed_submission_keeps_its_error(store: Store):
    failed = FakeSubmission(9, "oom", status=Status.ERROR, error_description="Notebook Threw Exception")
    sync_submissions(store, COMPETITION, api=StrictApi([failed]))
    sub = store.state().subs["9"]
    assert (sub.status, sub.public_score) == ("error", None)
    assert sub.extra["error_description"] == "Notebook Threw Exception"


def test_dry_run_writes_nothing(store: Store):
    named = _nb(store, b"nb", "agent-v2.ipynb")
    before = _events(store)
    report = sync_submissions(store, COMPETITION, api=StrictApi(_subs()), dry_run=True)
    assert report.dry_run and len(report.new) == 3 and report.linked == ["5003"]
    assert report.rows["5003"]["nb"] == named
    assert _events(store) == before and not Store.at(store.root).state().subs


# ----- pull ------------------------------------------------------------------------------------------


def test_pull_registers_the_source_without_running_it(store: Store, tmp_path: Path):
    store.append("idea.add", {"id": "short-prompt", "title": "Shorter prompt"})
    marker = tmp_path / "ran.txt"
    text = _notebook(f"open({str(marker)!r}, 'w').write('ran')\n")
    api = StrictApi(files=_kernel_files(text))
    nb, created = pull_kernel(store, "someone/demo-agent", api=api, tags=["kaggle"], applies=["short-prompt"])
    assert created and not marker.exists()
    assert (nb.name, nb.path, nb.kernel) == ("demo-agent.ipynb", "demo-agent.ipynb", "someone/demo-agent")
    assert (nb.tags, nb.applies, nb.stored) == (["kaggle"], ["short-prompt"], True)
    assert store.get_blob(nb.id) == text.encode()
    ((method, kernel, path, metadata),) = api._fake.calls
    assert (method, kernel, metadata) == ("kernels_pull", "someone/demo-agent", True)
    assert Path(path).parent == store.root / "cache" / "kaggle" / "someone__demo-agent"

    api = StrictApi(files=_kernel_files(text))
    again, created = pull_kernel(store, "someone/demo-agent", api=api, tags=["v2"])
    assert not created and again.id == nb.id and again.tags == ["kaggle", "v2"]


def test_pull_without_metadata_takes_the_only_code_file(store: Store):
    base = _nb(store, b"base", "base.ipynb")
    api = StrictApi(files={"agent.py": "print('x')\n", "notes.txt": "hello"})
    nb, created = pull_kernel(store, "someone/demo-agent", api=api, name="agent-script", parent="base.ipynb")
    assert created and (nb.name, nb.path, nb.parent) == ("agent-script", "agent.py", base)


@pytest.mark.parametrize(
    ("files", "message"),
    [
        ({"a.ipynb": "{}", "b.py": ""}, "several code files"),
        ({"notes.txt": "x"}, "no .ipynb or .py"),
        ({METADATA: json.dumps({"code_file": "../evil.py"}), "../evil.py": "x"}, "not a downloaded"),
        ({METADATA: json.dumps({"code_file": "gone.ipynb"}), "other.ipynb": "{}"}, "not a downloaded"),
        ({METADATA: "{not json", "a.ipynb": "{}"}, "cannot read"),
    ],
)
def test_pull_refuses_unclear_downloads(store: Store, files: dict[str, str], message: str):
    with pytest.raises(KaggleError, match=message):
        pull_kernel(store, "someone/demo-agent", api=StrictApi(files=files))
    assert not store.state().notebooks
    kernel_dir = store.root / "cache" / "kaggle" / "someone__demo-agent"
    assert not [p for p in kernel_dir.iterdir() if p.is_dir()]  # the download folder was removed


@pytest.mark.parametrize("ref", ["demo-agent", "a/b/c", "../x", "someone/", "/x", "some one/x", ""])
def test_pull_rejects_bad_refs(store: Store, ref: str):
    api = StrictApi()
    with pytest.raises(KaggleError, match="OWNER/SLUG"):
        pull_kernel(store, ref, api=api)
    assert not api._fake.calls


def test_pull_checks_ideas_and_parent_before_downloading(store: Store):
    api = StrictApi(files=_kernel_files())
    with pytest.raises(LookupError, match="unknown idea"):
        pull_kernel(store, "someone/demo-agent", api=api, applies=["nope"])
    with pytest.raises(LookupError):
        pull_kernel(store, "someone/demo-agent", api=api, parent="missing")
    assert not api._fake.calls


def test_pull_failure_is_a_kaggle_error(store: Store):
    class HTTPError(Exception):
        response = SimpleNamespace(status_code=404)

    api = StrictApi(pull_error=HTTPError("404 Client Error: Not Found"))
    with pytest.raises(KaggleError, match=r"cannot pull someone/demo-agent: 404 .*check OWNER/SLUG"):
        pull_kernel(store, "someone/demo-agent", api=api)
    assert list((store.root / "cache" / "kaggle" / "someone__demo-agent").iterdir()) == []


# ----- commands --------------------------------------------------------------------------------------


def test_cli_sync_prints_counts_and_table(store: Store, monkeypatch: pytest.MonkeyPatch):
    _nb(store, b"nb-named", "agent-v2.ipynb")
    monkeypatch.setattr(kaggle_api, "get_api", lambda: StrictApi(_subs()))
    result = _invoke(store, "kaggle", "sync")
    assert result.exit_code == 0, result.output
    lines = result.stdout.splitlines()
    assert lines[0] == f"{COMPETITION}: 3 new, 0 updated, 0 unchanged; 1 linked to notebooks"
    assert lines[1].split() == ["ID", "DATE", "PUBLIC", "PRIVATE", "STATUS", "NOTEBOOK", "DESCRIPTION"]
    assert [line.split()[0] for line in lines[3:]] == ["5001", "5002", "5003"]
    local = datetime(2026, 9, 2, 12, 30, 15, tzinfo=UTC).astimezone().strftime("%Y-%m-%d %H:%M")
    assert lines[4].split()[3:] == ["0.512", "0.498", "complete", "baseline", "rerun"]
    assert local in lines[4]
    assert "pending" in lines[5] and "agent-v2.ipynb  agent-v2.ipynb: shorter prompt" in lines[5]

    again = _invoke(store, "kaggle", "sync", "--competition", COMPETITION)
    assert again.stdout == f"{COMPETITION}: 0 new, 0 updated, 3 unchanged\n"


def test_cli_sync_dry_run_ambiguity_and_missing_competition(
    store: Store, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    _nb(store, b"one", "agent-v2.ipynb")
    _nb(store, b"two", "agent-v2.ipynb")
    monkeypatch.setattr(kaggle_api, "get_api", lambda: StrictApi(_subs()))
    result = _invoke(store, "kaggle", "sync", "--dry-run")
    assert result.exit_code == 0, result.output
    head = result.stdout.splitlines()[0]
    assert head == f"{COMPETITION}: 3 new, 0 updated, 0 unchanged (dry run: nothing written)"
    assert "1 submission(s) not linked" in result.stderr and "5003" in result.stderr
    assert not Store.at(store.root).state().subs

    bare = Store.init(tmp_path / "bare")
    result = CliRunner().invoke(main, ["--store", str(bare.root), "kaggle", "sync"])
    assert result.exit_code == 1 and result.output.startswith("Error: no competition")


def test_cli_pull(store: Store, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(kaggle_api, "get_api", lambda: StrictApi(files=_kernel_files()))
    result = _invoke(store, "kaggle", "pull", "someone/demo-agent", "--name", "demo", "--tag", "kaggle")
    assert result.exit_code == 0, result.output
    nb = Store.at(store.root).state().resolve_nb("someone/demo-agent")
    assert result.stdout == f"registered demo {nb.id[:10]} from someone/demo-agent\n"
    again = _invoke(store, "kaggle", "pull", "someone/demo-agent")
    assert again.stdout == f"already registered as demo {nb.id[:10]}\n"
    unknown = _invoke(store, "kaggle", "pull", "someone/demo-agent", "--applies", "nope")
    assert unknown.exit_code == 1 and unknown.output.startswith("Error: unknown idea(s): nope")


def test_cli_shows_kaggle_errors_in_one_line(store: Store, monkeypatch: pytest.MonkeyPatch):
    def no_credentials() -> Any:
        raise KaggleError(f"no Kaggle credentials found: {kaggle_api.CREDENTIALS_HELP}")

    monkeypatch.setattr(kaggle_api, "get_api", no_credentials)
    for args in (["kaggle", "sync"], ["kaggle", "pull", "someone/demo-agent"]):
        result = _invoke(store, *args)
        assert result.exit_code == 1 and result.output.count("\n") == 1
        assert result.output.startswith("Error: no Kaggle credentials found")
    bad = _invoke(store, "kaggle", "pull", "not-a-ref")
    assert bad.exit_code == 1 and bad.output.startswith("Error: not a Kaggle kernel")
