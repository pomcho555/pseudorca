from __future__ import annotations

import json
from pathlib import Path

import pytest

from pseudorca import util
from pseudorca.model import State
from pseudorca.store import Store, StoreError
from pseudorca.tree import Tree, decode_text, safe_path


def test_ulids_sort_in_creation_order():
    ids = [util.new_ulid() for _ in range(2000)]
    assert ids == sorted(ids)
    assert len(set(ids)) == len(ids)
    assert all(len(i) == 26 for i in ids)


def test_format_table_aligns_wide_characters():
    out = util.format_table(["id", "title"], [["a", "改善"], ["bb", "x"]])
    lines = out.splitlines()
    assert util.display_width(lines[2]) == util.display_width("a   改善")
    assert lines[3].startswith("bb  x")


def test_init_find_and_config(store: Store, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    assert (store.root / "config.toml").exists()
    assert store.setting("store", "competition") == "some-competition"
    assert store.setting("audit", "expect_status") == ["adopted"]
    nested = store.root.parent / "a" / "b"
    nested.mkdir(parents=True)
    assert Store.find(nested).root == store.root
    with pytest.raises(StoreError):
        Store.init(store.root.parent)
    with pytest.raises(StoreError):
        Store.find(tmp_path)


def test_events_are_one_file_each_and_replay(store: Store):
    store.append("idea.add", {"id": "fast-prompt", "title": "Shorter prompt", "status": "testing"})
    store.append("idea.update", {"id": "fast-prompt", "status": "adopted", "tags": ["prompt"]})
    files = sorted((store.root / "events").glob("*/*.json"))
    assert len(files) == 2
    raw = json.loads(files[0].read_text())
    assert raw["actor"] == "tester" and raw["type"] == "idea.add" and raw["schema"] == 1
    idea = Store.at(store.root).state().ideas["fast-prompt"]
    assert idea.status == "adopted" and idea.tags == ["prompt"] and idea.title == "Shorter prompt"


def test_unreadable_and_unknown_events_become_warnings(store: Store):
    store.append("idea.add", {"id": "x"})
    month = next(d for d in (store.root / "events").iterdir() if d.is_dir())  # not .gitkeep
    (month / "broken.json").write_text("{not json")
    (month / "zzzz-future.json").write_text(
        json.dumps({"schema": 2, "id": "ZZZZ", "type": "idea.rename", "time": "t", "actor": "a", "data": {}})
    )
    st = Store.at(store.root).state()
    assert "x" in st.ideas
    assert any("broken.json" in w for w in st.warnings)
    assert any("idea.rename" in w for w in st.warnings)


def test_blobs_are_content_addressed(store: Store):
    sha = store.put_blob(b"hello")
    assert sha == util.sha256_bytes(b"hello")
    assert store.put_blob(b"hello") == sha
    assert store.get_blob(sha) == b"hello"
    with pytest.raises(StoreError):
        store.get_blob("0" * 64)


def _nb(store: Store, data: bytes, **kw) -> str:
    sha = store.put_blob(data)
    store.append("nb.add", {"id": sha, "size": len(data), **kw})
    return sha


def test_resolve_notebooks(store: Store):
    a = _nb(store, b"one", name="main.ipynb", tags=["candidate"])
    b = _nb(store, b"two", name="main.ipynb")
    c = _nb(store, b"three", name="other.ipynb", kernel="me/other")
    st = store.state()
    assert st.resolve_nb("latest").id == c
    assert st.resolve_nb("latest:candidate").id == a
    assert st.resolve_nb("main.ipynb").id == b
    assert st.resolve_nb("me/other").id == c
    assert st.resolve_nb(a[:8]).id == a
    with pytest.raises(LookupError):
        st.resolve_nb("nope")
    with pytest.raises(LookupError):
        st.resolve_nb("latest:missing")


def _verdict(store: Store, nb: str, idea_id: str, status: str, method: str, **kw):
    idea = store.state().ideas[idea_id]
    store.append(
        "audit.verdict",
        {
            "nb": nb,
            "idea": idea_id,
            "idea_rev": kw.pop("rev", idea.rev),
            "status": status,
            "method": method,
            **kw,
        },
    )


def test_effective_verdict_rules(store: Store):
    nb = _nb(store, b"nb")
    store.append("idea.add", {"id": "i1", "title": "t", "fingerprints": ["X=1"]})
    st: State = store.state()
    assert st.effective_verdict(nb, "i1") is None

    _verdict(store, nb, "i1", "present", "fingerprint")
    eff = store.state().effective_verdict(nb, "i1")
    assert (eff.status, eff.method, eff.stale) == ("present", "fingerprint", False)

    _verdict(store, nb, "i1", "absent", "patch")
    assert store.state().effective_verdict(nb, "i1").status == "conflict"

    _verdict(store, nb, "i1", "partial", "patch")
    _verdict(store, nb, "i1", "absent", "llm", model="fake:m")
    eff = store.state().effective_verdict(nb, "i1")
    assert (eff.status, eff.method) == ("absent", "llm")

    _verdict(store, nb, "i1", "present", "human")
    assert store.state().effective_verdict(nb, "i1").method == "human"

    # changing the idea makes machine verdicts stale; the human one is kept but marked stale
    store.append("idea.update", {"id": "i1", "fingerprints": ["X=2"]})
    eff = store.state().effective_verdict(nb, "i1")
    assert (eff.status, eff.method, eff.stale) == ("present", "human", True)


def test_stale_machine_verdict_is_shown_as_stale(store: Store):
    nb = _nb(store, b"nb")
    store.append("idea.add", {"id": "i2", "title": "t"})
    _verdict(store, nb, "i2", "absent", "llm")
    store.append("idea.update", {"id": "i2", "description": "more detail"})
    eff = store.state().effective_verdict(nb, "i2")
    assert (eff.status, eff.stale) == ("absent", True)


def test_tree_search_and_snippets():
    t = Tree("abc")
    t.add_text("cells/cell_000.py", "import os\n\ndef f(x):\n    return  x + 1\n", "cell 0")
    t.add_text("markdown/cell_001.md", "def f(x):", "cell 1")
    t.add_text("../../etc/passwd", "nope", "evil")
    assert "etc/passwd" in t
    assert t.add_text("cells/cell_000.py", "dup", "again").path == "cells/cell_000.py~2"
    assert [h.line for h in t.grep("return")] == [4]
    assert t.find_snippet("def f(x):\n  return x + 1") == [("cells/cell_000.py", 3)]
    assert t.find_snippet("x + 1", partial_ends=True) == [("cells/cell_000.py", 4)]
    assert t.find_snippet("x + 1", partial_ends=False) == []
    assert "markdown/cell_001.md" not in t.evidence_paths()
    assert t.read("cells/cell_000.py", 3, 4).splitlines()[0].endswith("| def f(x):")


def test_tree_cache_round_trip(tmp_path: Path):
    t = Tree("src", warnings=["w"], meta={"extractor_version": 1})
    t.add_text("a.py", "print('hi')\n", "cell 0")
    t.add_bytes("blob.bin", b"\x00\x01\x02", "payload")
    t.save(tmp_path / "c")
    u = Tree.load(tmp_path / "c")
    assert u is not None
    assert u.text("a.py") == "print('hi')\n" and u.files["blob.bin"].binary
    assert u.meta == {"extractor_version": 1} and u.warnings == ["w"]


def test_safe_path_and_decode_text():
    assert safe_path("/abs/../x/./y.py") == "x/y.py"
    assert safe_path("") == "_"
    assert decode_text(b"hello\n") == "hello\n"
    assert decode_text(b"\x00\xff\xfe") is None


def test_register_notebook_merges_metadata(store: Store, tmp_path: Path):
    from pseudorca.notebooks import register_file

    store.append("idea.add", {"id": "i1", "title": "t"})
    f = store.root.parent / "nbs" / "main.ipynb"
    f.parent.mkdir()
    f.write_bytes(b'{"cells": []}')
    nb, created = register_file(store, f, tags=["a"])
    assert created and nb.path == "nbs/main.ipynb" and nb.stored and store.has_blob(nb.id)
    nb2, created2 = register_file(store, f, tags=["a", "b"], applies=["i1"])
    assert not created2 and nb2.tags == ["a", "b"] and nb2.applies == ["i1"]
    store.blob_path(nb.id).unlink()  # a blob that never got pushed
    register_file(store, f)
    assert store.has_blob(nb.id)  # re-adding the file repairs it
    inside = store.root.parent / "nbs" / "draft.py"
    inside.write_bytes(b"print(1)")
    nb3, _ = register_file(store, inside, keep_bytes=False, parent=nb.id)
    assert nb3.path == "nbs/draft.py" and not nb3.stored and nb3.parent == nb.id
    outside = tmp_path / "elsewhere.ipynb"
    outside.write_bytes(b'{"cells": [1]}')
    with pytest.raises(ValueError, match="outside the repository"):
        register_file(store, outside, keep_bytes=False)
    with pytest.raises(LookupError):
        register_file(store, outside, applies=["missing"])
    lfs = tmp_path / "pointer.ipynb"
    lfs.write_bytes(b"version https://git-lfs.github.com/spec/v1\noid sha256:abc\n")
    with pytest.raises(ValueError, match="not notebook JSON"):
        register_file(store, lfs)


def test_split_lines_counts_like_python():
    from pseudorca.tree import split_lines

    assert split_lines("a\fb\nc\r\nd\re\n") == ["a\fb", "c", "d", "e"]
    t = Tree()
    t.add_text("x.py", "x = 1\f\ny = 2\n", "cell 0")
    assert [h.line for h in t.grep("y = 2")] == [2]


def test_null_note_and_model_are_empty_strings(store: Store):
    store.append("idea.add", {"id": "i", "title": "t"})
    store.append(
        "audit.verdict",
        {
            "nb": "a" * 64,
            "idea": "i",
            "idea_rev": "r",
            "status": "present",
            "method": "llm",
            "note": None,
            "model": None,
        },
    )
    rec = store.state().verdicts[("a" * 64, "i")][0]
    assert rec.note == "" and rec.model == ""


def test_format_table_column_widths():
    out = util.format_table(["a", "b"], [["x" * 30, "y" * 30]], col_widths={0: 10})
    assert "x" * 9 + "…" in out and "y" * 30 in out


def test_blob_ids_must_be_sha256_and_bytes_must_match(store: Store):
    with pytest.raises(StoreError, match="not a blob id"):
        store.get_blob("../../outside/marker.txt")
    assert not store.has_blob("/etc/passwd")
    sha = store.put_blob(b"real bytes")
    store.blob_path(sha).write_bytes(b"version https://git-lfs.github.com/spec/v1\n")
    with pytest.raises(StoreError, match="Git LFS"):
        store.get_blob(sha)
    store.append("idea.add", {"id": "evil", "title": "t", "patch": "../../outside/marker.txt"})
    state = Store.at(store.root).state()
    assert state.ideas["evil"].patch is None
    assert any("invalid patch=" in w for w in state.warnings)


def test_new_events_sort_after_everything_seen_even_with_a_slow_clock(store: Store, monkeypatch):
    store.append("idea.add", {"id": "fast", "title": "t", "status": "testing"})
    future_ms = util.ulid_ms(store.events()[-1].id) + 10 * 60_000
    future_id = util.new_ulid(future_ms)
    month = next(d for d in (store.root / "events").iterdir() if d.is_dir())  # not .gitkeep
    (month / f"{future_id}-other.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "id": future_id,
                "type": "idea.update",
                "time": "t",
                "actor": "other",
                "data": {"id": "fast", "status": "parked"},
            }
        )
    )
    s2 = Store.at(store.root)
    s2.append("idea.update", {"id": "fast", "status": "adopted"})  # made after seeing the future event
    assert s2.events()[-1].id > future_id
    assert Store.at(store.root).state().ideas["fast"].status == "adopted"
    assert any("clock" in w for w in s2.state().warnings)


def test_init_completes_a_partial_store_without_overwriting(tmp_path: Path):
    repo = tmp_path / "repo"
    (repo / ".pseudorca").mkdir(parents=True)
    (repo / ".pseudorca" / "config.toml").write_text('[store]\nname = "kept"\n')
    with pytest.raises(StoreError, match="no events/ folder"):
        Store.find(repo)
    s = Store.init(repo, name="new")
    assert "events/" in s.created and "config.toml" not in s.created and ".gitattributes" in s.created
    assert s.setting("store", "name") == "kept"
    assert "blobs/** -text" in (s.root / ".gitattributes").read_text()


def test_replay_merges_duplicate_registrations_and_warns_on_unknown_fields(store: Store):
    sha = store.put_blob(b"nb")
    store.append(
        "nb.add", {"id": sha, "name": "a.ipynb", "tags": ["candidate"], "applies": ["x"], "kernel": "me/a"}
    )
    store.append("nb.add", {"id": sha, "name": "b.ipynb", "tags": [], "applies": [], "kernel": ""})
    store.append("idea.add", {"id": "i", "title": "t", "priority": "high"})
    store.append("sub.upsert", {"id": "9", "public_score": "pending", "description": "rerun"})
    st = Store.at(store.root).state()
    nb = st.notebooks[sha]
    assert (nb.name, nb.tags, nb.applies, nb.kernel) == ("a.ipynb", ["candidate"], ["x"], "me/a")
    assert any("'priority'" in w for w in st.warnings)
    assert st.subs["9"].public_score is None and st.subs["9"].description == "rerun"


def test_newer_extractor_verdict_beats_a_later_one_from_an_older_client(store: Store):
    nb = _nb(store, b"nb")
    store.append("idea.add", {"id": "i", "title": "t", "fingerprints": ["X"]})
    _verdict(store, nb, "i", "present", "fingerprint", detail={"extractor_version": 2})
    _verdict(store, nb, "i", "absent", "fingerprint", detail={"extractor_version": 1})
    assert store.state().effective_verdict(nb, "i").status == "present"


def test_llm_settles_a_deterministic_conflict(store: Store):
    nb = _nb(store, b"nb")
    store.append("idea.add", {"id": "i", "title": "t"})
    _verdict(store, nb, "i", "present", "fingerprint")
    _verdict(store, nb, "i", "absent", "patch")
    assert store.state().effective_verdict(nb, "i").status == "conflict"
    _verdict(store, nb, "i", "present", "llm", model="m")
    eff = store.state().effective_verdict(nb, "i")
    assert (eff.status, eff.method) == ("present", "llm")


def test_excerpt_keeps_a_wide_character_match_visible():
    q = 'PROMPT = "あなたは数学の文章題を解きます。手順を短く考えてください。答えだけを書いてください。"'
    out = util.excerpt(q, q.index("手順"), 40)
    assert "手順を短く" in out and util.display_width(out) <= 40


def test_times_from_many_zones_and_bad_values():
    assert util.instant("2026-01-05T11:30:00+02:00") > util.instant("2026-01-05T18:29:00+09:00")
    assert util.instant(None) < util.instant("2026-01-05")
    assert util.local_time("0001-01-01T00:00:00+05:00")  # no OverflowError
    assert util.local_time("2026-01-05") == "2026-01-05"


def test_event_fields_of_the_wrong_kind_are_dropped_with_a_warning(store: Store):
    sha = store.put_blob(b"x")
    store.append("nb.add", {"id": sha, "name": "a.ipynb"})
    store.append(
        "sub.upsert", {"id": "1004", "nb": ["x"], "date": None, "public_score": 0.5, "status": {"a": 1}}
    )
    store.append("idea.add", {"id": "i", "title": "t", "tags": "not-a-list"})
    st = Store.at(store.root).state()
    sub = st.subs["1004"]
    assert (sub.nb, sub.date, sub.public_score, sub.status) == (None, "", 0.5, "")
    assert st.ideas["i"].tags == []
    assert sum("ignored invalid" in w for w in st.warnings) == 3


def test_deeply_nested_json_is_not_a_notebook(tmp_path: Path, store: Store):
    from pseudorca.notebooks import register_file

    deep = tmp_path / "deep.ipynb"
    deep.write_bytes(b"[" * 100_000 + b"]" * 100_000)
    with pytest.raises(ValueError, match="not notebook JSON"):
        register_file(store, deep)


def test_cache_paths_refuse_symlinks_and_escapes(store: Store, tmp_path: Path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (store.root / "cache" / "trees").symlink_to(outside, target_is_directory=True)
    with pytest.raises(StoreError, match="symlink"):
        store.cache_dir("trees", "a" * 64)
    with pytest.raises(StoreError, match="invalid cache path"):
        store.cache_dir("..", "x")


def test_tree_cache_with_non_hash_names_or_symlinks_is_ignored(tmp_path: Path):
    t = Tree("src", meta={"extractor_version": 1})
    t.add_text("a.py", "x = 1\n", "cell 0")
    t.save(tmp_path / "c")
    manifest = json.loads((tmp_path / "c" / "manifest.json").read_text())
    manifest["files"][0]["sha256"] = "../../../etc/hosts"
    (tmp_path / "c" / "manifest.json").write_text(json.dumps(manifest))
    assert Tree.load(tmp_path / "c") is None
    t.save(tmp_path / "d")
    content = next((tmp_path / "d" / "files").iterdir())
    content.unlink()
    content.symlink_to(tmp_path / "c" / "manifest.json")
    assert Tree.load(tmp_path / "d") is None


def test_a_damaged_blob_is_repaired_by_adding_the_file_again(store: Store):
    from pseudorca.notebooks import register_file

    f = store.root.parent / "nb.ipynb"
    f.write_bytes(b'{"cells": []}')
    nb, _ = register_file(store, f)
    store.blob_path(nb.id).write_bytes(b'{"cells": []}\r\n')  # a line-ending conversion
    assert store.has_blob(nb.id) and not store.blob_ok(nb.id)
    register_file(store, f)
    assert store.blob_ok(nb.id) and store.get_blob(nb.id) == f.read_bytes()


def test_registrations_on_two_clones_merge_without_losing_anything(store: Store):
    from pseudorca.notebooks import register_bytes

    store.append("idea.add", {"id": "a", "title": "t"})
    store.append("idea.add", {"id": "b", "title": "t"})
    data = b'{"cells": [], "metadata": {}}'
    nb, _ = register_bytes(
        store, data, name="next.ipynb", tags=["candidate"], applies=["a"], note="alice: v7"
    )
    # what bob's clone wrote before pulling: his own first registration of the same bytes
    store.append(
        "nb.add", {"id": nb.id, "name": "cand-7", "tags": ["fast"], "applies": ["b"], "note": "bob: c7"}
    )
    register_bytes(store, data, name="next.ipynb", tags=["ci"])  # a later re-add only adds
    st = Store.at(store.root).state()
    merged = st.notebooks[nb.id]
    assert merged.tags == ["candidate", "fast", "ci"] and merged.applies == ["a", "b"]
    assert merged.name == "next.ipynb" and "cand-7" in merged.aliases
    assert merged.note == "alice: v7 / bob: c7"
    assert st.resolve_nb("cand-7").id == nb.id
    assert [e.type for e in st.events if e.data.get("id") == nb.id].count("nb.update") == 0


def test_init_without_a_name_leaves_it_out_of_the_config(tmp_path: Path):
    a = Store.init(tmp_path / "alice")
    b = Store.init(tmp_path / "bob")
    assert (a.root / "config.toml").read_bytes() == (b.root / "config.toml").read_bytes()
    assert a.setting("store", "name") is None


def test_a_future_store_does_not_leak_future_ids_into_other_stores(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("PSEUDORCA_ACTOR", "t")
    ahead = Store.init(tmp_path / "ahead")
    ahead.append("idea.add", {"id": "x", "title": "t"})
    future_id = util.new_ulid(util.ulid_ms(ahead.events()[-1].id) + 30 * 60_000)
    month = next(d for d in (ahead.root / "events").iterdir() if d.is_dir())  # not .gitkeep
    (month / f"{future_id}-o.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "id": future_id,
                "type": "idea.update",
                "time": "t",
                "actor": "o",
                "data": {"id": "x"},
            }
        )
    )
    Store.at(ahead.root).append("idea.update", {"id": "x", "status": "testing"})
    other = Store.init(tmp_path / "other")
    other.append("idea.add", {"id": "y", "title": "t"})
    assert util.ulid_ms(other.events()[-1].id) < util.ulid_ms(future_id)
    assert not any("clock" in w for w in Store.at(other.root).state().warnings)


def test_snippet_search_is_linear_and_exact():
    import time

    t = Tree()
    body = "".join(f"x{i} = compute({i})\n" for i in range(60_000))
    t.add_text("big.py", body + "def target():\n    return  42\n", "cell 0")
    started = time.perf_counter()
    for _ in range(20):
        assert t.find_snippet("def target():\n return 42", partial_ends=False) == [("big.py", 60_001)]
    assert time.perf_counter() - started < 2.0
    assert t.find_snippet("compute(59999)\ndef tar") == [("big.py", 60_000)]
    assert t.find_snippet("ute(1)") == [("big.py", 2)]
    assert t.find_snippet("x1 = compute(1)", partial_ends=False) == [("big.py", 2)]


def test_tree_cache_keeps_line_endings_and_never_writes_through_links(tmp_path: Path):
    t = Tree("s", meta={"extractor_version": 1})
    t.add_text("a.sh", "# %%bash\r\necho hi\r\n", "cell 0")
    (tmp_path / "c" / "files").mkdir(parents=True)
    victim = tmp_path / "victim.txt"
    (tmp_path / "c" / "files" / t.files["a.sh"].sha256).symlink_to(victim)
    (tmp_path / "c" / "manifest.json").symlink_to(tmp_path / "victim2.txt")
    t.save(tmp_path / "c")
    assert not victim.exists() and not (tmp_path / "victim2.txt").exists()
    u = Tree.load(tmp_path / "c")
    assert u is not None and u.text("a.sh") == "# %%bash\r\necho hi\r\n"
