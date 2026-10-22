from __future__ import annotations

from click.testing import CliRunner

from pseudorca.cli import main
from pseudorca.report import render
from pseudorca.store import Store


def _setup(store: Store) -> tuple[str, str]:
    store.append("idea.add", {"id": "short-prompt", "title": "Shorter <b>prompt</b>", "status": "adopted"})
    store.append("idea.add", {"id": "old-knob", "title": "Old knob", "status": "parked"})
    a = store.put_blob(b"nb-a")
    b = store.put_blob(b"nb-b")
    store.append("nb.add", {"id": a, "name": "base.ipynb", "tags": ["main"]})
    store.append("nb.add", {"id": b, "name": "next.ipynb", "parent": a, "applies": ["short-prompt"]})
    idea = store.state().ideas["short-prompt"]
    store.append(
        "audit.verdict",
        {
            "nb": a,
            "idea": "short-prompt",
            "idea_rev": idea.rev,
            "status": "absent",
            "method": "patch",
            "evidence": [{"path": "files/a.py", "line": 3, "quote": "x = '<script>'"}],
        },
    )
    for sid, nb, score in (("1", a, 0.40), ("2", a, 0.44), ("3", b, 0.47)):
        store.append("sub.upsert", {"id": sid, "nb": nb, "public_score": score, "date": f"2026-01-0{sid}"})
    return a, b


def test_render_escapes_and_shows_board(store: Store):
    _setup(store)
    page = render(store.state(), store_name="team & co", generated="2026-01-09T00:00:00+00:00")
    assert "team &amp; co" in page
    assert "Shorter &lt;b&gt;prompt&lt;/b&gt;" in page
    assert "<script>" not in page and "&lt;script&gt;" in page
    assert ">✗<" in page and ">c<" in page  # absent on base, claimed on next
    assert "old-knob" not in page.split("Recent activity")[0]  # parked ideas hidden from board and list
    assert "http" not in page.split("<footer>")[0]  # no external resources


def test_report_command_writes_to_cache_by_default(store: Store):
    _setup(store)
    out = CliRunner().invoke(main, ["--store", str(store.root), "report"])
    assert out.exit_code == 0, out.output
    path = store.root / "cache" / "report.html"
    assert path.exists() and "Which ideas are in which notebook" in path.read_text()
