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
    month = next((store.root / "events").iterdir())
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
