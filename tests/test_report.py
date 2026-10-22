from __future__ import annotations

import html
import math
import re
from collections.abc import Sequence
from datetime import date, timedelta
from typing import Any

import pytest
from click.testing import CliRunner

from pseudorca import charts, effects, report, util
from pseudorca.cli import main
from pseudorca.commands.report import RESAMPLES
from pseudorca.commands.results import _load
from pseudorca.effects import Cell, Row
from pseudorca.report import Effects, render
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


def test_long_linear_lineage_is_flat_and_complete(store: Store):
    parent = None
    for i in range(80):
        sha = store.put_blob(f"v{i}".encode())
        store.append("nb.add", {"id": sha, "name": f"v{i}.ipynb", "parent": parent})
        parent = sha
    page = render(store.state(), store_name="t")
    lineage = page.split('<h2 class="sr">Notebooks</h2>')[1].split("<h2")[0]
    assert all(f"v{i}.ipynb" in lineage for i in range(80))
    assert "margin-left:18px" not in lineage  # a single chain never indents


# ----- what each idea added ------------------------------------------------------------------------


def _effects_ledger(store: Store, *, verdicts: bool = True, runs: bool = False) -> dict[str, str]:
    """Two ideas over four notebooks, seven scored submissions (some resubmitted), and optionally local runs
    of three of the notebooks. Returns the notebook ids by name."""
    for iid in ("retry", "cache"):
        store.append("idea.add", {"id": iid, "title": f"{iid} idea", "status": "testing"})
    ideas = store.state().ideas
    nbs = {}
    for name, have in (("base", ()), ("r", ("retry",)), ("c", ("cache",)), ("rc", ("retry", "cache"))):
        sha = nbs[name] = store.put_blob(name.encode())
        store.append("nb.add", {"id": sha, "name": f"{name}.ipynb"})
        for iid in ("retry", "cache") if verdicts else ():
            status = "present" if iid in have else "absent"
            data = {"nb": sha, "idea": iid, "idea_rev": ideas[iid].rev, "status": status, "method": "patch"}
            store.append("audit.verdict", data)
    scores = [("base", 0.400), ("base", 0.404), ("r", 0.431), ("c", 0.409), ("rc", 0.440), ("rc", 0.436)]
    for i, (name, score) in enumerate([*scores, ("r", 0.428)]):
        store.append(
            "sub.upsert",
            {"id": str(101 + i), "nb": nbs[name], "public_score": score, "date": f"2026-03-0{i + 1}"},
        )
    for name, rate in (("base", 3), ("r", 6), ("rc", 7)) if runs else ():
        rows = [f"q{i},{d},{int((i * 7 + d * 3) % 10 < rate)}" for i in range(24) for d in range(2)]
        sha = store.put_blob(("item,draw,score\n" + "\n".join(rows) + "\n").encode())
        data = {"id": util.new_ulid(), "results": sha, "format": "csv", "nb": nbs[name], "protocol": "p1"}
        store.append("run.add", {**data, "label": name, "n_rows": len(rows)})
    return nbs


def _report(store: Store, *args: str) -> str:
    out = CliRunner().invoke(main, ["--store", str(store.root), "report", *args])
    assert out.exit_code == 0, out.output
    return (store.root / "cache" / "report.html").read_text(encoding="utf-8")


def _section(page: str) -> str:
    """The Effects tab's panel (its views are tabs of their own, with h3 headings)."""
    return page.split('<h2 class="sr">What each idea added</h2>')[1].split("<h2")[0]


def _signed(x: float, dec: int) -> str:
    text = f"{abs(x):.{dec}f}"
    return text if not float(text) else ("+" if x > 0 else "−") + text


def _numbers(terms: list[Any]) -> list[str]:
    """What the forest should print for each term: decimals from the narrowest interval, as the CLI does."""
    dec = min(6, max(0, 1 - math.floor(math.log10(min(t.ci_high - t.ci_low for t in terms)))))
    return [
        f"{_signed(t.estimate, dec)} [{_signed(t.ci_low, dec)}, {_signed(t.ci_high, dec)}]" for t in terms
    ]


def _ideas(store: Store, ids: str) -> None:
    for iid in ids.split():
        store.append("idea.add", {"id": iid, "title": f"idea {iid}"})


def _verdict(store: Store, nb: str, idea: str, status: str) -> None:
    rev = store.state().ideas[idea].rev
    store.append(
        "audit.verdict", {"nb": nb, "idea": idea, "idea_rev": rev, "status": status, "method": "human"}
    )


def _notebook(store: Store, name: str, **verdicts: str) -> str:
    """Register a notebook with human verdicts (idea=status); returns its id."""
    sha = store.put_blob(name.encode())
    store.append("nb.add", {"id": sha, "name": name})
    for idea, status in verdicts.items():
        _verdict(store, sha, idea, status)
    return sha


def _subs(store: Store, ideas: str, subs: Sequence[tuple[str, str | None, float, str]]) -> dict[str, str]:
    """The ideas (space-separated ids), then one scored submission per entry: (id, notebook name or None,
    public score, one verdict code per idea: 1 present, 0 absent, - none). A notebook's verdicts are recorded
    where it first appears; the submissions are a day apart. Returns the notebook ids by name."""
    _ideas(store, ideas)
    nbs: dict[str, str] = {}
    for k, (sid, name, score, codes) in enumerate(subs):
        if name is not None and name not in nbs:
            status = {"1": "present", "0": "absent"}
            verdicts = {i: status[c] for i, c in zip(ideas.split(), codes, strict=True) if c != "-"}
            nbs[name] = _notebook(store, name, **verdicts)
        day = (date(2026, 2, 1) + timedelta(days=k)).isoformat()
        store.append("sub.upsert", {"id": sid, "nb": nbs.get(name or ""), "public_score": score, "date": day})
    return nbs


def _table(
    store: Store, nb: str, values: Sequence[float], label: str, draws: int, metric: str = "score"
) -> None:
    """A local run of `nb` (protocol p): `values` in item order, `draws` per item."""
    lines = [f"item,draw,{metric}"] + [f"q{i // draws},{i % draws},{v}" for i, v in enumerate(values)]
    sha = store.put_blob(("\n".join(lines) + "\n").encode())
    data = {"id": util.new_ulid(), "results": sha, "format": "csv", "nb": nb, "protocol": "p", "label": label}
    store.append("run.add", data)


def _effects_config(store: Store, table: str) -> None:
    """Set config.toml's [effects] table."""
    path = store.root / "config.toml"
    text = path.read_text().split("\n[effects]\n")[0]
    path.write_text(f"{text}\n[effects]\n{table}\n")


def _visible(section: str) -> str:
    """The section's text as a reader sees it: no tooltips, no tags."""
    text = re.sub(r"<title>.*?</title>|</?code>", "", section, flags=re.S).replace("<br>", "\n")
    return html.unescape(re.sub(r"<[^>]+>", " ", text))


def _chart(section: str, label: str) -> str:
    """The SVG whose aria-label starts with `label`."""
    return section.split(f'aria-label="{label}')[1].split("</svg>")[0]


def test_effects_section_has_matrix_charts_and_breakdown(store: Store):
    _effects_ledger(store)
    store.append("sub.upsert", {"id": "108", "public_score": 0.5, "date": "2026-03-08"})  # no notebook
    page = _report(store)
    section = _section(page)
    lb = effects.lb_fit(store.state())
    assert page.index('<h2 class="sr">What each idea added</h2>') < page.index(
        '<h2 class="sr">Submissions</h2>'
    )
    assert section.count("<svg") == 3  # the scores over time, the forest plot and the breakdown
    assert all('role="img" aria-label="' in svg for svg in section.split("<svg")[1:])
    assert "<script" not in page and "xmlns" not in page and "http" not in page.split("<footer>")[0]
    assert "resubmits 101" in section and "+retry" in section and "+cache −retry" in section
    assert section.count('class="c ok chg"') == 3 and section.count('class="c no chg"') == 2  # in / out
    assert "From 101 to 105, idea by idea" in section and "105 fitted (best in the model)" in section
    assert "noise band" in section and "resubmitting a notebook moved the score" in section
    fitted = util.number(lb.rows[0].fitted, max(3, util.score_decimals([r.fitted for r in lb.rows])))
    assert f'<td class="num">{fitted}</td>' in section  # 101's fitted score, with a mean's extra digits
    assert "! methods disagree" in section and "· · " not in section  # no separator that reads as a symbol
    assert '<tr class="off" title="not used in the model: no notebook' in section
    assert "not in the model" in section  # 108 is a hollow point over the scores
    assert "What each idea added" not in render(store.state(), store_name="t")  # only when computed


def test_forest_numbers_match_the_fit(store: Store):
    _effects_ledger(store)
    state = store.state()
    lb = effects.lb_fit(state)
    page = render(state, store_name="t", effects=Effects(lb=lb))
    forest = page.split('aria-label="What each idea adds to the public score')[1].split("</svg>")[0]
    terms = lb.measured()
    assert [t.label for t in terms] == ["retry", "cache"]  # best first
    for t, text in zip(terms, _numbers(terms), strict=True):
        assert f">{text}</text>" in forest and f">{t.label}</text>" in forest
    assert ">helps</text>" in forest  # retry clears the noise
    assert terms[0].verdict == effects.HELPS


def test_effects_section_escapes_ledger_strings(store: Store):
    hostile_idea = 'x"><script>alert(1)</script>'
    store.append("idea.add", {"id": hostile_idea, "title": "<b>t</b>"})
    rev = store.state().ideas[hostile_idea].rev
    nbs = []
    for i, name in enumerate(["<img src=x onerror=alert(2)>.ipynb", "plain.ipynb"]):
        sha = store.put_blob(f"nb{i}".encode())
        store.append("nb.add", {"id": sha, "name": name})
        status = "present" if i else "absent"
        store.append(
            "audit.verdict",
            {"nb": sha, "idea": hostile_idea, "idea_rev": rev, "status": status, "method": "human"},
        )
        nbs.append(sha)
    for i, (sid, nb, score) in enumerate(
        [("<b>s1</b>", 0, 0.40), ("s2", 0, 0.41), ("s3", 1, 0.45), ("s4", 1, 0.44)]
    ):
        store.append(
            "sub.upsert", {"id": sid, "nb": nbs[nb], "public_score": score, "date": f"2026-04-0{i + 1}"}
        )
    page = _report(store)
    section = _section(page)
    assert "<script" not in page and "<img" not in page and "<b>s1" not in page and "<b>t</b>" not in page
    assert "http" not in page.split("<footer>")[0]
    escaped = "x&quot;&gt;&lt;script&gt;alert(1)&lt;/script&gt;"
    svgs = section.split("<svg")[1:]
    assert len(svgs) == 3
    assert any(f">+{escaped}</text>" in s for s in svgs)  # the change label over the scores
    assert any(f"<title>s3: +{escaped}</title>" in s for s in svgs)
    assert any(f">{escaped}</text>" in s for s in svgs)  # the forest row
    assert any(f"<title>{escaped}\n" in s for s in svgs)
    assert any("&lt;img src=x onerror=alert(2)&gt;.ipynb" in s for s in svgs)  # a tooltip names the notebook
    assert "<code>&lt;b&gt;s1&lt;/b&gt;</code>" in section  # the matrix row


def test_effects_section_notes_instead_of_failing(store: Store, monkeypatch: pytest.MonkeyPatch):
    section = _section(_report(store))
    assert "No submissions yet" in section and "<svg" not in section

    _effects_ledger(store, verdicts=False)
    section = _section(_report(store))
    assert "No verdicts on these submissions" in section
    assert section.count("<svg") == 1  # the scores still have a chart
    assert "no idea has a verdict on the notebooks of these submissions" in section
    assert "no idea is in some" not in section and "same ideas" not in section  # neither is known

    def boom(*args: Any, **kwargs: Any) -> None:
        raise ValueError("the ledger is odd")

    monkeypatch.setattr(effects, "lb_fit", boom)
    section = _section(_report(store))
    assert "Leaderboard: the ledger is odd" in section and "<svg" not in section


def test_effects_of_several_competitions(store: Store):
    nbs = _effects_ledger(store)
    store.append("sub.upsert", {"id": "999", "nb": nbs["rc"], "public_score": 0.9, "competition": "other"})
    store.append("sub.upsert", {"id": "101", "competition": "some-competition"})  # the store's competition
    section = _section(_report(store))
    assert "The submissions are from several competitions" in section
    assert "<code>999</code>" not in section and "<code>101</code>" in section

    config = store.root / "config.toml"
    config.write_text(config.read_text().replace('competition = "some-competition"', 'competition = ""'))
    section = _section(_report(store))
    assert "Leaderboard: the submissions are from 2 competitions" in section and "config.toml" in section


def test_effects_of_local_runs(store: Store):
    _effects_ledger(store, runs=True)
    section = _section(_report(store))
    assert (
        "Local runs (protocol p1): score" in section and "3 of 3 runs used, 24 items in every one" in section
    )
    assert section.count("<svg") == 4
    local = section.split("Local runs (protocol p1)")[1]
    fits = effects.run_fits(store.state(), lambda run: _load(store, run), resamples=RESAMPLES)
    terms = fits[0].measured()
    assert terms and "<svg" in local
    for text in _numbers(terms):
        assert f">{text}</text>" in local
    assert f"({RESAMPLES} resamples, seed 0)" in local


def test_change_labels_never_overlap():
    cells = [
        {"a": Cell("present" if (i // 2) % 2 else "absent"), "b": Cell("present" if i % 3 else "absent")}
        for i in range(120)
    ]
    rows = [Row(str(i), f"s{i}", "nb", "nb", "", 0.5, c) for i, c in enumerate(cells)]
    bounds = [40 + i * 7.0 for i in range(len(rows))]
    moved = effects.all_changes(rows)
    labels = charts._change_labels(rows, bounds, moved)
    assert len(labels) == sum(1 for m in moved if m)
    placed = [lab for lab in labels if lab.lane is not None]
    assert placed and len(placed) < len(labels)  # crowded: some keep only their tick
    for lab in placed:
        assert 0 <= lab.x0 < lab.x1 <= charts.WIDTH
        same = [o for o in placed if o is not lab and o.lane == lab.lane]
        assert all(lab.x1 < o.x0 or o.x1 < lab.x0 for o in same)


# ----- review fixes --------------------------------------------------------------------------------

RMSE = [
    ("1", "base.py", 0.50, "0"),
    ("2", "base.py", 0.52, "0"),
    ("3", "feat.py", 0.41, "1"),
    ("4", "feat.py", 0.43, "1"),
]


def test_lower_is_better_comes_from_the_config(store: Store):
    _subs(store, "feat", RMSE)  # feat lowers an error: better only when lower is better
    section = _section(_report(store))
    assert ">hurts</text>" in section and "Higher scores count as better" in section  # the default
    _effects_config(store, "lower_is_better = true")
    section = _section(_report(store))
    forest = _chart(section, "What each idea adds")
    assert ">helps</text>" in forest and 'class="dot ok"' in forest
    assert (
        "Lower scores count as better" in section and "Leaderboard: public score (lower is better)" in section
    )
    assert (
        "From 1 to 3, idea by idea" in section and "3 fitted (best in the model)" in section
    )  # the lowest score
    legend = section.split("From 1 to 3, idea by idea")[1]  # the breakdown's legend, not the tab names
    assert "lowers it (better)" in legend and "adds</" not in legend


def test_a_bad_config_value_is_a_note_and_the_other_still_counts(store: Store):
    _subs(store, "feat", RMSE)
    _effects_config(store, 'lower_is_better = true\nnoise = "x"')
    section = _section(_report(store))
    assert "[effects] noise in config.toml must be a positive number" in section and "ignored here" in section
    assert "Lower scores count as better" in section and "noise: residual SD" in section
    _effects_config(store, "noise = 0.01")
    section = _section(_report(store))
    assert "noise: SD 0.01 as given ([effects] noise in .pseudorca/config.toml); normal intervals" in section


def test_parked_ideas_only_with_all(store: Store):
    _subs(store, "a old", [("1", "n1", 0.5, "00"), ("2", "n2", 0.6, "11"), ("3", "n1", 0.51, "")])
    store.append("idea.update", {"id": "old", "status": "parked"})
    assert "<span>old</span>" not in _section(_report(store))  # as on the board
    assert "<span>old</span>" in _section(_report(store, "--all"))


def test_local_runs_of_another_metric(store: Store):
    _ideas(store, "x")
    a, b = _notebook(store, "a.py", x="absent"), _notebook(store, "b.py", x="present")
    _table(store, a, [0, 1, 0, 0] * 10, "a", 2, metric="acc")
    _table(store, b, [1, 1, 0, 1] * 10, "b", 2, metric="acc")
    section = _section(_report(store))
    assert "0 of 2 runs used" in section and "no metric &#x27;score&#x27; (metrics: acc)" in section
    assert "<code>pseudorca report --metric NAME</code>" in section
    section = _section(_report(store, "--metric", "acc"))
    assert "Local runs (protocol p): acc" in section and "2 of 2 runs used" in section


def test_local_intervals_match_effects(store: Store):
    """With effects' 10,000 resamples the verdict is effects' (2000 made this interval touch 0)."""
    a = "001101100100000110010101000000011000010000101110110100001110111001100101100111100011010000111110100111010000111001001110"
    b = "010010101011101000011011000110001011010000111110110111101111111110101110101111110101111110001111011001100111111001100110"
    _ideas(store, "x")
    for name, values, status in (("a.py", a, "absent"), ("b.py", b, "present")):
        _table(store, _notebook(store, name, x=status), [int(c) for c in values], name, 3)
    (fit,) = effects.run_fits(store.state(), lambda run: _load(store, run))
    assert RESAMPLES == 10_000 and fit.resamples == RESAMPLES and fit.measured()[0].verdict == effects.HELPS
    section = _section(_report(store))
    assert ">helps</text>" in _chart(section, "What each idea adds to the score")
    assert "(10000 resamples, seed 0)" in section
    quick = _section(_report(store, "--resamples", "2000"))
    assert "(2000 resamples, seed 0)" in quick and ">within noise</text>" in quick


def test_an_empty_forest_says_why(store: Store):
    _subs(store, "x", [("1", "a.py", 0.40, "0"), ("2", "b.py", 0.45, "1"), ("3", "c.py", 0.44, "-")])
    section = _section(_report(store))
    assert "no idea can be measured yet: 1 idea is left out for missing verdicts (see below)" in section
    assert (
        "no idea is in some" not in section
        and "<code>pseudorca audit --submitted</code> fills the gaps" in section
    )


def test_an_empty_forest_of_tangled_ideas(store: Store):
    subs = [("1", "n1", 0.40, "1000"), ("2", "n2", 0.42, "0110"), ("3", "n3", 0.45, "1101")]
    _subs(store, "cot retry cache temp", subs)
    section = _section(_report(store))
    assert "no idea can be measured on its own: the ideas that vary cannot be separated" in section
    assert "cot: cannot be separated from" in section and "no idea is in some" not in section
    # its noise note points to the setting the report reads, not to an option it lacks
    assert "--noise" not in section and "<code>noise = SD</code> under <code>[effects]</code>" in section


def test_effects_without_a_noise_estimate(store: Store):
    _subs(store, "x", [("1", "a.py", 0.4, "0"), ("2", "b.py", 0.5, "1")])
    section = _section(_report(store))
    assert (
        ">unknown</text>" in section
        and "; within noise otherwise, and unknown without an interval" in section
    )
    assert "--noise" not in section and "Resubmit a notebook, or set a known LB noise SD" in section


def test_an_empty_forest_when_no_idea_varies(store: Store):
    _subs(store, "x", [("1", "a.py", 0.40, "1"), ("2", "b.py", 0.45, "1"), ("3", "a.py", 0.41, "")])
    section = _section(_report(store))
    assert "no idea is in some submissions but not all" in section
    assert "in every one (cannot be measured here): x" in section


def test_the_breakdown_never_claims_the_same_ideas_wrongly(store: Store):
    _subs(store, "x", [("1", "a.py", 0.40, "0"), ("2", "b.py", 0.45, "1"), ("3", "c.py", 0.44, "-")])
    section = _section(_report(store))
    assert (
        "no modelled idea differs between 1 and 2, so there is nothing to break down; x differs, or may, but "
        "is not in the model" in section
    )
    assert "same ideas" not in section


def test_the_breakdown_names_what_its_steps_leave_out(store: Store):
    subs = [("1", "n1", 0.50, "00"), ("2", "n2", 0.55, "1-"), ("3", "n3", 0.56, "11"), ("4", "n1", 0.51, "")]
    _subs(store, "a gap", subs)
    section = _section(_report(store))
    assert "From 1 to 3, idea by idea" in section
    assert "not in the model, so not in these steps: gap (it differs, or may, between 1 and 3)" in section


def test_the_breakdown_of_a_resubmission_or_the_same_ideas(store: Store):
    subs = [("1", "a.py", 0.40, "10"), ("2", "a.py", 0.41, "10"), ("3", "b.py", 0.39, "01")]
    _subs(store, "x y", subs)
    assert "2 resubmits the notebook of 1: nothing to break down" in _section(_report(store))
    nb = _notebook(store, "c.py", x="present", y="absent")  # another notebook with the ideas of a.py
    store.append("sub.upsert", {"id": "4", "nb": nb, "public_score": 0.42, "date": "2026-03-01"})
    assert "4 has the same ideas as 1: nothing to break down" in _section(_report(store))


def test_a_bound_near_zero_keeps_its_sign(store: Store):
    subs = [(str(i + 1), "a.py", s, "0") for i, s in enumerate((0.395, 0.408, 0.390))]
    subs += [(str(i + 4), "b.py", s, "1") for i, s in enumerate((0.383, 0.381, 0.383))]
    _subs(store, "x", subs)
    forest = _chart(_section(_report(store)), "What each idea adds")
    assert ">hurts</text>" in forest and "−0.0003]" in forest and ", 0.000]" not in forest


def test_a_downgraded_verdict_says_why_where_it_can_be_read(store: Store):
    _ideas(store, "x")
    for name, values, status in (
        ("a.py", "001101100100110000001001", "absent"),
        ("b.py", "110011110111101111101111", "present"),
    ):
        _table(store, _notebook(store, name, x=status), [int(v) for v in values], name, 4)
    shown = _visible(_section(_report(store)))
    assert "x: the interval excludes 0, but the sign-flip p = 0.0625 is not below 0.05" in shown
    assert (
        "helps or hurts when the 95% interval excludes 0 and the sign-flip p < 0.05, as in compare" in shown
    )
    assert "no two runs have the same ideas, so noise shared by a whole run cannot be measured" in shown


def test_run_to_run_noise_is_described(store: Store):
    _ideas(store, "x")
    nbs = {"a.py": _notebook(store, "a.py", x="absent"), "b.py": _notebook(store, "b.py", x="present")}
    for k, (name, rate) in enumerate((("a.py", 3), ("a.py", 4), ("b.py", 6), ("b.py", 5))):
        _table(store, nbs[name], [int((i * 7 + k) % 10 < rate) for i in range(60)], f"r{k}", 2)
    (fit,) = effects.run_fits(store.state(), lambda run: _load(store, run))
    assert fit.tau is not None and fit.tau_df == 2 and fit.lof_p is None  # all the residuals are repeats
    shown = _visible(_section(_report(store)))
    if fit.tau > 0:
        assert (
            f"plus a run-to-run SD of {charts.sig(fit.tau)} from runs with the same ideas (2 degrees of freedom)"
            in shown
        )
    else:
        assert "runs with the same ideas show no run-to-run noise beyond that" in shown


def test_a_submission_without_verdicts_hides_no_change(store: Store):
    subs = [
        ("1", "a.py", 0.40, "0"),
        ("2", "a.py", 0.41, "0"),
        ("3", None, 0.43, ""),
        ("4", "b.py", 0.45, "1"),
    ]
    _subs(store, "x", [*subs, ("5", "b.py", 0.46, "1")])
    section = _section(_report(store))
    row = section.split("<code>4</code>")[1].split("</tr>")[0]
    assert "<td class='chs'>+x</td>" in row and 'class="c ok chg" title="x: present; came in since 2"' in row
    chart = _chart(section, "public score of 5 submissions")
    assert ">+x</text>" in chart and "<title>4: +x</title>" in chart


def test_several_competitions_point_to_the_config(store: Store):
    _subs(store, "x", [("1", "n1", 0.41, "0"), ("2", "n1", 0.43, "0")])
    for sid, competition in (("1", "comp-a"), ("2", "comp-b")):
        store.append("sub.upsert", {"id": sid, "competition": competition})
    section = _section(_report(store))
    assert "choose one with --competition" not in section
    assert "set <code>competition = &quot;SLUG&quot;</code> under <code>[store]</code>" in section
    assert "the store&#x27;s competition (some-competition) is none of them" in section
    config = store.root / "config.toml"
    config.write_text(
        config.read_text().replace('competition = "some-competition"', 'competition = "comp-a"')
    )
    section = _section(_report(store))
    assert (
        "these are comp-a&#x27;s" in section
        and "<code>pseudorca effects --competition comp-a</code>" in section
    )


def test_scores_with_five_decimals_keep_them(store: Store):
    _subs(store, "x", [("1", "a.py", 0.91231, "0"), ("2", "a.py", 0.91234, "0"), ("3", "b.py", 0.91262, "1")])
    section = _section(_report(store))
    assert '<td class="num">0.91231</td>' in section and '<td class="num">0.91234</td>' in section


def test_fitted_values_keep_their_digits(store: Store):
    _subs(
        store,
        "tir",
        [
            ("1", "base.py", 29, "0"),
            ("2", "base.py", 30, "0"),
            ("3", "tir.py", 32, "1"),
            ("4", "tir.py", 33, "1"),
        ],
    )
    section = _section(_report(store))
    assert '<td class="num">29</td><td class="num">29.5</td>' in section
    assert '<td class="num">33</td><td class="num">32.5</td>' in section


NUMBER = re.compile(r"\d{10,}")  # a number printed digit by digit, far past what anyone reads


def test_huge_scores_read_well(store: Store):
    subs = [("1", "a.py", 1e30, "0"), ("2", "a.py", 1.1e30, "0"), ("3", "b.py", 1.5e30, "1")]
    _subs(store, "x", [*subs, ("4", "b.py", 1.45e30, "1")])
    section = _section(_report(store))
    assert not NUMBER.search(_visible(section)) and ">helps</text>" in section
    forest = _chart(section, "What each idea adds")
    xs = [float(x) for x in re.findall(r'<line class="(?:grid|zero)" x1="([\d.]+)"', forest)]
    assert xs == sorted(xs) and xs[-1] - xs[0] > 200  # left to right, and wide enough to read
    labels = re.findall(r'text-anchor="end">([^<]+)</text>', _chart(section, "public score of"))
    assert labels and len(set(labels)) == len(labels)


def test_tiny_differences_have_tick_labels_apart(store: Store):
    subs = [("1", "a.py", 0.5, "0"), ("2", "a.py", 0.500000000001, "0"), ("3", "b.py", 0.500000003, "1")]
    _subs(store, "x", [*subs, ("4", "b.py", 0.5000000031, "1")])
    chart = _chart(_section(_report(store)), "public score of")
    labels = re.findall(r'<text x="[\d.]+" y="[\d.]+" text-anchor="end">([^<]+)</text>', chart)
    assert len(labels) >= 3 and len(set(labels)) == len(labels)


def test_identical_resubmissions_have_no_noise_band(store: Store):
    _subs(store, "x", [("1", "a.py", 0.5, "0"), ("2", "a.py", 0.5, "0"), ("3", "b.py", 0.6, "1")])
    section = _section(_report(store))
    assert "noise band" not in section and "wide intervals" not in section
    assert "no noise estimate: the residuals are 0" in section
    fit = effects.lb_fit(store.state())
    fit.sigma = 1.92e-16  # rounding, not noise
    assert charts.noise_sd(fit) == 0 and charts.noise_halfwidth(fit) is None


@pytest.mark.filterwarnings("ignore::RuntimeWarning")  # numpy's overflow warnings
def test_scores_near_the_float_limit_still_give_a_report(store: Store):
    _subs(store, "x", [("1", "a.py", 1.75e308, "0")])
    section = _section(_report(store))
    assert "1.75e+308" in section and "<svg" in section
    _subs(store, "", [("2", "b.py", 0.0, ""), ("3", "a.py", 1e300, ""), ("4", "a.py", 1.1e300, "")])
    page = _report(store)  # the LB noise of a.py overflows
    assert (
        "LB noise: submit the same notebook" in page  # the resubmission SD overflows: not shown
        and "Leaderboard: a score is too large to model (beyond 1e+150)" in page
    )
    for lo, hi in ((0.0, 1.79e308), (math.inf, 1.0), (-1e300, 1e300), (5.0, 5.0)):
        a, b, ticks, step = charts._axis(lo, hi, 4)
        assert all(math.isfinite(v) for v in (a, b, step, *ticks)) and ticks
    assert charts._f(math.inf) == "0.0" and charts._f(-1e300) == "-100000.0"


def test_a_part_that_cannot_be_drawn_becomes_a_note(store: Store, monkeypatch: pytest.MonkeyPatch):
    _effects_ledger(store, runs=True)

    def overflow(*args: Any, **kwargs: Any) -> str:
        raise OverflowError("cannot convert float infinity to integer")

    monkeypatch.setattr(charts, "lb_chart", overflow)
    section = _section(_report(store))
    assert "the chart of the scores cannot be drawn (OverflowError: cannot convert" in section
    assert section.count("<svg") == 3 and "Local runs (protocol p1): score" in section  # the rest is drawn
    monkeypatch.setattr(report, "_matrix", overflow)
    section = _section(_report(store))
    assert "the table of submissions cannot be drawn (OverflowError" in section
    assert section.count("<svg") == 3  # only that view fails: the effects, the breakdown, the local runs

    def deep(*args: Any, **kwargs: Any) -> None:
        raise RecursionError("maximum recursion depth exceeded")

    monkeypatch.setattr(effects, "run_fits", deep)
    section = _section(_report(store))
    assert "Local runs: cannot be computed (RecursionError: maximum recursion depth exceeded)" in section


def test_a_results_table_nested_too_deeply_never_stops_the_report(store: Store):
    nbs = _subs(store, "x", [("1", "a.py", 0.4, "0"), ("2", "b.py", 0.5, "1")])
    _table(store, nbs["a.py"], [0, 1, 0, 1] * 10, "a", 1)
    blob = store.put_blob(('{"item":"q1","score":' + "[" * 100_000 + "]" * 100_000 + "}\n").encode())
    data = {"id": util.new_ulid(), "results": blob, "format": "jsonl", "nb": nbs["b.py"], "protocol": "p"}
    store.append("run.add", data)
    section = _section(_report(store))
    assert "Leaderboard: public score" in section and "Local runs" in section


def test_a_difference_reads_as_one_in_its_tooltip(store: Store):
    subs = [
        ("1", "n1", 0.40, "110"),
        ("2", "n2", 0.42, "101"),
        ("3", "n3", 0.45, "011"),
        ("4", "n1", 0.41, ""),
    ]
    _subs(store, "a b c", subs)  # every notebook has two of the three ideas
    diffs = [t for t in effects.lb_fit(store.state()).measured() if t.kind == effects.DIFFERENCE]
    assert diffs and all(" vs " in t.label for t in diffs)  # also used together
    after = _section(_report(store)).split('aria-label="What each idea adds')[1]
    forest, legend = after.split("</svg>")[0], after.split('<div class="legend">')[1].split("</div>")[0]
    for t in diffs:
        left, right = " + ".join(t.left), " + ".join(t.right)
        assert f"<title>{html.escape(t.label)}\n" in forest
        assert f"a difference: what {left} adds minus what {right} adds" in forest
    assert "instead of" not in forest and "instead of" not in legend


def test_the_page_is_tabs_that_switch_without_scripts(store: Store):
    _effects_ledger(store, runs=True)
    page = _report(store)
    assert "<script" not in page.lower() and "javascript:" not in page.lower()
    groups = {}
    for name, ident, checked in re.findall(
        r'<input class="tab" type="radio" name="([a-z]+)" id="([a-z]+-\d+)"( checked)?>', page
    ):
        groups.setdefault(name, []).append((ident, bool(checked)))
    assert set(groups) == {"tab", "fx"}
    for name, tabs in groups.items():
        assert sum(checked for _, checked in tabs) == 1, name  # one view at a time
        for ident, _ in tabs:  # each radio has its label in the bar, its panel, and CSS that shows the panel
            assert f'<label for="{ident}">' in page and f'id="{ident}-panel"' in page
            assert f"#{ident}:checked ~ #{ident}-panel {{ display:block; }}" in page
    labels = re.findall(r'<label for="tab-\d+">([^<]+)', page)
    assert labels == ["Effects", "Board", "Submissions", "Notebooks", "Ideas", "Activity"]
    assert groups["tab"][0] == ("tab-0", True)  # the effects first, when there are some
    views = re.findall(r'<label for="fx-\d+">([^<]+)', page)
    assert views == [
        "Ideas per submission",
        "What each idea adds",
        "Breakdown",
        "Score chart",
        "Local runs · p1",
    ]
    assert "@media print" in page and ".tabs > .panel { display:block !important; }" in page
    css = page.split("<style>")[1].split("</style>")[0]
    assert "retry" not in css and "p1" not in css  # the CSS holds only the ids it made, no ledger string
    plain = render(store.state(), store_name="t")  # no effects computed: the board comes first
    assert re.findall(r'<label for="tab-\d+">([^<]+)', plain)[0] == "Board" and 'id="tab-0" checked' in plain


def test_hostile_tab_labels_are_escaped(store: Store):
    _effects_ledger(store, runs=True)
    run = next(iter(store.state().runs.values()))
    store.append(
        "run.add",
        {
            **{k: getattr(run, k) for k in ("results", "format", "nb", "label")},
            "id": util.new_ulid(),
            "protocol": "<b>x</b>\x1b[2J",
        },
    )
    page = _report(store)
    assert "Local runs · &lt;b&gt;x&lt;/b&gt;?[2J" in page and "<b>x</b>" not in page


def test_a_long_tab_label_is_cut_with_a_tooltip(store: Store):
    _effects_ledger(store, runs=True)
    run = next(iter(store.state().runs.values()))
    long = "a-protocol-whose-name-goes-on-and-on-for-a-long-while"
    fields = {k: getattr(run, k) for k in ("results", "format", "nb", "label")}
    store.append("run.add", {**fields, "id": util.new_ulid(), "protocol": long})
    page = _report(store)
    label = re.search(r'<label for="fx-\d+" title="([^"]+)">([^<]+)</label>', page)
    assert label is not None and label.group(1) == f"Local runs · {long}"
    assert label.group(2).endswith("…") and len(label.group(2)) <= 40
