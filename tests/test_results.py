from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from pseudorca.cli import main
from pseudorca.notebooks import register_bytes
from pseudorca.results import format_for, load_table, per_item, pool
from pseudorca.store import Store

# ----- parsing -------------------------------------------------------------------------------------


def _lists(values: dict[str, np.ndarray]) -> dict[str, list[float]]:
    return {k: v.tolist() for k, v in values.items()}


def test_csv_without_draw_numbers_repeats_in_file_order():
    t = load_table(b"item,score\nq1,1\nq2,0\nq1,0\nq1,1\n", "csv")
    assert t.items == ["q1", "q2", "q1", "q1"] and t.draws == [0, 0, 1, 2]
    assert list(t.metrics) == ["score"] and t.ignored == []
    values = per_item(t, "score")
    assert values["q1"].tolist() == [1.0, 0.0, 1.0] and values["q2"].tolist() == [0.0]
    assert t.draw_counts() == {"q1": 3, "q2": 1} and len(t) == 4


def test_csv_explicit_draws_are_integers_and_ordered():
    t = load_table(b"\xef\xbb\xbfdraw,item,score\n2,q1,0.2\n0,q1,0.0\n1.0,q1,0.1\n7,q2,1\n", "csv")
    assert t.draws == [2, 0, 1, 7]
    assert per_item(t, "score")["q1"].tolist() == [0.0, 0.1, 0.2]  # in draw order
    for bad in (b"item,draw,score\nq1,1.5,0\n", b"item,draw,score\nq1,x,0\n", b"item,draw,score\nq1,,0\n"):
        with pytest.raises(ValueError, match="draw must be an integer"):
            load_table(bad, "csv")


def test_missing_item_column_is_named():
    with pytest.raises(ValueError, match=r'no "item" column \(columns: id, score\)'):
        load_table(b"id,score\nq1,1\n", "csv")
    with pytest.raises(ValueError, match='no "item" column'):
        load_table(b'{"id": "q1", "score": 1}\n', "jsonl")
    with pytest.raises(ValueError, match="line 3: empty item"):
        load_table(b"item,score\nq1,1\n,0\n", "csv")


def test_duplicate_item_and_draw_names_the_item():
    with pytest.raises(ValueError, match="item 'q7' has draw 1 twice"):
        load_table(b"item,draw,score\nq7,0,1\nq7,1,0\nq7,1,1\n", "csv")
    with pytest.raises(ValueError, match="item 'q7' has draw 0 twice"):
        load_table(b'{"item": "q7", "draw": 0, "score": 1}\n{"item": "q7", "draw": 0, "score": 0}\n', "jsonl")


def test_text_columns_are_ignored_and_empty_values_are_nan():
    data = (
        b"item,score,latency_s,ok,answer,notes,,model\n"
        b"q1,1,12.5,true,42,,0,gpt\n"
        b"q2,,NA,False,43,,1,gpt\n"
        b"q3,0.5,nan,TRUE,x,,2,gpt\n"
    )
    t = load_table(data, "csv")
    assert list(t.metrics) == ["score", "latency_s", "ok", "notes"]  # an all-empty column is all NaN
    assert t.ignored == ["answer", "(column 7)", "model"]  # text, an unnamed column, text
    assert np.isnan(t.metrics["score"][1]) and t.metrics["ok"].tolist() == [1.0, 0.0, 1.0]
    assert np.isnan(t.metrics["latency_s"][1:]).all()
    assert _lists(per_item(t, "score")) == {"q1": [1.0], "q2": [], "q3": [0.5]}
    with pytest.raises(ValueError, match="no metric 'nope'"):
        per_item(t, "nope")


def test_csv_structure_errors():
    with pytest.raises(ValueError, match="no header"):
        load_table(b"", "csv")
    with pytest.raises(ValueError, match="no rows"):
        load_table(b"item,score\n\n", "csv")
    with pytest.raises(ValueError, match="duplicate column"):
        load_table(b"item,score,score\nq1,1,2\n", "csv")
    with pytest.raises(ValueError, match="line 3: 3 fields"):
        load_table(b"item,score\nq1,1\nq2,0,9\n", "csv")
    with pytest.raises(ValueError, match="not UTF-8"):
        load_table(b"item,score\nq\xff,1\n", "csv")
    with pytest.raises(ValueError, match="unknown results format"):
        load_table(b"item\nq1\n", "parquet")
    long_text = "x" * 200_000  # longer than the csv module's default field limit
    t = load_table(f'item,score,output\nq1,1,"{long_text}"\n'.encode(), "csv")
    assert t.ignored == ["output"]


def test_jsonl_values():
    lines = [
        {"item": "q1", "score": True, "cost": 0.5, "tags": ["a"], "answer": "yes"},
        {"item": "q1", "score": False, "cost": None, "answer": "no"},
        {"item": 3, "score": 1, "cost": "0.25", "extra": 4},
    ]
    data = "﻿" + "\n".join(json.dumps(x) for x in lines[:2]) + "\n\n" + json.dumps(lines[2]) + "\n"
    t = load_table(data.encode(), "jsonl")
    assert t.items == ["q1", "q1", "3"] and t.draws == [0, 1, 0]
    assert list(t.metrics) == ["score", "cost", "extra"] and t.ignored == ["tags", "answer"]
    assert t.metrics["score"].tolist() == [1.0, 0.0, 1.0]
    assert t.metrics["cost"][2] == 0.25 and np.isnan(t.metrics["cost"][1])
    assert np.isnan(t.metrics["extra"][:2]).all()  # a key missing from a line is NaN
    with pytest.raises(ValueError, match="line 2: not valid JSON"):
        load_table(b'{"item": "q1"}\n{oops\n', "jsonl")
    with pytest.raises(ValueError, match="line 1: expected a JSON object"):
        load_table(b'["q1", 1]\n', "jsonl")
    with pytest.raises(ValueError, match="line 2: missing item"):
        load_table(b'{"item": "q1", "score": 1}\n{"score": 0}\n', "jsonl")


def test_format_for():
    assert format_for("runs/a.csv") == "csv"
    assert format_for(Path("a.JSONL")) == "jsonl" and format_for("b.ndjson") == "jsonl"
    with pytest.raises(ValueError, match="--format"):
        format_for("results.txt")


def test_pool_renumbers_draws_across_runs():
    t1 = load_table(b"item,draw,score,cost\nq1,5,1,1\nq1,3,0,1\nq2,0,1,1\n", "csv")
    t2 = load_table(b'{"item": "q1", "score": 0.5}\n{"item": "q3", "score": 0.25, "note": "x"}\n', "jsonl")
    pooled = pool([t1, t2])
    assert sorted(zip(pooled.items, pooled.draws, strict=True)) == [
        ("q1", 0),
        ("q1", 1),
        ("q1", 2),
        ("q2", 0),
        ("q3", 0),
    ]
    assert per_item(pooled, "score")["q1"].tolist() == [0.0, 1.0, 0.5]  # t1 in draw order, then t2
    assert _lists(per_item(pooled, "cost")) == {"q1": [1.0, 1.0], "q2": [1.0], "q3": []}
    assert pooled.ignored == ["note"]
    assert pool([t1]) is t1
    with pytest.raises(ValueError):
        pool([])


# ----- CLI -----------------------------------------------------------------------------------------


def _cli(store: Store, *args: str) -> object:
    return CliRunner().invoke(main, ["--store", str(store.root), *args])


def _write_run(path: Path, *, shift: float, draws: int, seed: int, n_items: int = 30) -> Path:
    """A synthetic results table: item levels 0.2..0.8, B-style shift, small per-draw noise."""
    rng = np.random.default_rng(seed)
    lines = ["item,draw,score,latency_s,answer"]
    for i, level in enumerate(np.linspace(0.2, 0.8, n_items)):
        for d in range(draws):
            score = level + shift + rng.normal(0, 0.05)
            lines.append(f"q{i:03d},{d},{score:.4f},{rng.uniform(1, 2):.3f},text {i}")
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.fixture
def two_notebooks(store: Store) -> Store:
    register_bytes(store, b'{"cells": ["base"]}', name="base.ipynb")
    register_bytes(store, b'{"cells": ["new"]}', name="new.ipynb")
    return store


def _add(store: Store, path: Path, *args: str) -> str:
    """Add a run through the CLI and return its full id."""
    before = set(Store.at(store.root).state().runs)
    result = _cli(store, "run", "add", str(path), *args)
    assert result.exit_code == 0, result.output
    (new_id,) = set(Store.at(store.root).state().runs) - before
    return new_id


def test_run_add_list_show(two_notebooks: Store, tmp_path: Path):
    store = two_notebooks
    path = _write_run(tmp_path / "base.csv", shift=0.0, draws=3, seed=1)
    result = _cli(
        store, "run", "add", str(path), "--nb", "base.ipynb", "--label", "base", "--protocol", "dev-30"
    )
    assert result.exit_code == 0, result.output
    out = result.output
    assert "label base, notebook base.ipynb, protocol dev-30" in out
    assert "90 rows, 30 items, draws per item: min 3, median 3, max 3" in out
    assert re.search(r"metrics \(mean over items\): score 0\.\d+, latency_s 1\.\d+", out)
    assert "ignored columns: answer" in out

    run = next(iter(Store.at(store.root).state().runs.values()))
    assert (run.label, run.protocol, run.n_rows, run.format) == ("base", "dev-30", 90, "csv")
    assert store.get_blob(run.results) == path.read_bytes()

    listing = _cli(store, "run", "list")
    assert listing.exit_code == 0
    header, _, row = listing.output.splitlines()[:3]
    assert header.split() == ["ID", "LABEL", "NOTEBOOK", "PROTOCOL", "ROWS", "ADDED", "BY"]
    assert row.split()[:5] == [run.id[:10], "base", "base.ipynb", "dev-30", "90"] and row.endswith("tester")
    data = json.loads(_cli(store, "run", "list", "--json").stdout)
    assert data[0]["id"] == run.id and data[0]["nb"] == run.nb
    assert "no runs for that notebook" in _cli(store, "run", "list", "--nb", "new.ipynb").output

    shown = _cli(store, "run", "show", run.id[:12])
    assert shown.exit_code == 0, shown.output
    assert f"run {run.id}" in shown.output and "protocol  dev-30" in shown.output
    metric_rows = {
        line.split()[0]: line.split()
        for line in shown.output.splitlines()
        if line[:6] in ("score ", "latenc")
    }
    assert metric_rows["score"][1:4] == ["30", "90", "0"]  # items, values, missing
    assert float(metric_rows["score"][4]) == pytest.approx(0.5, abs=0.02)
    assert _cli(store, "run", "show", "base").exit_code == 0  # a label only one run has


def test_run_add_rejects_bad_tables(store: Store, tmp_path: Path):
    bad = tmp_path / "bad.csv"
    bad.write_text("id,score\nq1,1\n")
    result = _cli(store, "run", "add", str(bad))
    assert result.exit_code == 1 and "Traceback" not in result.output
    assert 'no "item" column' in result.output

    odd = tmp_path / "results.txt"
    odd.write_text('{"item": "q1", "score": 1}\n')
    result = _cli(store, "run", "add", str(odd))
    assert result.exit_code == 1 and "--format" in result.output
    result = _cli(store, "run", "add", str(odd), "--format", "jsonl")
    assert result.exit_code == 0, result.output
    assert "no 'score' column" not in result.output

    textonly = tmp_path / "text.csv"
    textonly.write_text("item,answer\nq1,yes\n")
    assert "no metric columns" in _cli(store, "run", "add", str(textonly)).output
    assert "no notebook" in _cli(store, "run", "add", str(odd), "--format", "jsonl", "--nb", "nope").output

    again = _cli(store, "run", "add", str(odd), "--format", "jsonl")
    assert again.exit_code == 0 and "already added as run" in again.output


def test_compare_by_run_id_label_and_notebook(two_notebooks: Store, tmp_path: Path):
    store = two_notebooks
    base1 = _add(
        store,
        _write_run(tmp_path / "b1.csv", shift=0, draws=3, seed=1),
        "--nb",
        "base.ipynb",
        "--label",
        "base",
    )
    base2 = _add(
        store,
        _write_run(tmp_path / "b2.csv", shift=0, draws=2, seed=2),
        "--nb",
        "base.ipynb",
        "--label",
        "base",
    )
    new = _add(
        store,
        _write_run(tmp_path / "n.csv", shift=0.1, draws=3, seed=3),
        "--nb",
        "new.ipynb",
        "--label",
        "new",
    )

    by_id = _cli(store, "compare", base1, new, "--resamples", "2000")
    assert by_id.exit_code == 0, by_id.output
    lines = by_id.output.splitlines()
    assert lines[0] == f"A  run {base1[:10]} (base): 30 items, 3 draws per item"
    assert lines[1] == f"B  run {new[:10]} (new): 30 items, 3 draws per item"
    assert "items compared: 30 (only in A: 0, only in B: 0)" in by_id.output
    assert "B − A: +0.1" in by_id.output and "95% CI [+0.0" in by_id.output
    assert "verdict: B better (the 95% CI excludes 0)" in by_id.output
    assert "A/A on A (draws split in halves)" in by_id.output and "MDE80" in by_id.output

    prefix = new[:16].lower()  # case-insensitive unique prefix
    assert _cli(store, "compare", base1, prefix, "--resamples", "2000").output == by_id.output

    by_label = _cli(store, "compare", "base", "new", "--resamples", "2000")
    assert by_label.exit_code == 0, by_label.output
    assert by_label.output.splitlines()[0] == "A  label base: 2 runs, 30 items, 5 draws per item"
    by_nb = _cli(store, "compare", "base.ipynb", "new.ipynb", "--resamples", "2000")
    assert by_nb.output.splitlines()[0] == "A  notebook base.ipynb: 2 runs, 30 items, 5 draws per item"
    assert by_nb.output.splitlines()[2:] == by_label.output.splitlines()[2:]  # the same runs either way

    lower = _cli(
        store, "compare", "base", "new", "--metric", "latency_s", "--lower-is-better", "--resamples", "500"
    )
    assert lower.exit_code == 0 and "mean latency_s" in lower.output and "(lower is better)" in lower.output

    missing = _cli(store, "compare", "base", "new", "--metric", "acc")
    assert (
        missing.exit_code == 1
        and "no metric 'acc'" in missing.output
        and "score, latency_s" in missing.output
    )
    unknown = _cli(store, "compare", "base", "nothing-here")
    assert unknown.exit_code == 1 and "no run, run label or notebook matches 'nothing-here'" in unknown.output
    assert base2 != base1


def test_compare_protocol_mismatch_and_force(two_notebooks: Store, tmp_path: Path):
    store = two_notebooks
    _add(store, _write_run(tmp_path / "a.csv", shift=0, draws=2, seed=1), "--label", "a", "--protocol", "dev")
    _add(
        store, _write_run(tmp_path / "b.csv", shift=0, draws=2, seed=2), "--label", "b", "--protocol", "full"
    )
    result = _cli(store, "compare", "a", "b")
    assert result.exit_code == 1
    assert "different protocols (A: 'dev' (1 run); B: 'full' (1 run))" in result.output
    assert "--force" in result.output and "Traceback" not in result.output
    forced = _cli(store, "compare", "a", "b", "--force", "--resamples", "1000")
    assert forced.exit_code == 0, forced.output
    assert "warning: comparing runs with different protocols" in forced.stderr
    assert "verdict: within noise" in forced.stdout

    # a notebook with runs under two protocols: --protocol keeps one of them in both arms
    _add(
        store,
        _write_run(tmp_path / "c.csv", shift=0, draws=2, seed=3),
        "--nb",
        "base.ipynb",
        "--protocol",
        "dev",
    )
    _add(
        store,
        _write_run(tmp_path / "d.csv", shift=0, draws=2, seed=4),
        "--nb",
        "base.ipynb",
        "--protocol",
        "full",
    )
    _add(
        store,
        _write_run(tmp_path / "e.csv", shift=0.1, draws=2, seed=5),
        "--nb",
        "new.ipynb",
        "--protocol",
        "dev",
    )
    assert "different protocols" in _cli(store, "compare", "base.ipynb", "new.ipynb").output
    picked = _cli(store, "compare", "base.ipynb", "new.ipynb", "--protocol", "dev", "--resamples", "1000")
    assert picked.exit_code == 0, picked.output
    assert picked.output.startswith("A  notebook base.ipynb: 30 items, 2 draws per item")
    none = _cli(store, "compare", "base.ipynb", "new.ipynb", "--protocol", "other")
    assert none.exit_code == 1
    assert (
        "notebook base.ipynb has no runs with protocol 'other' (its protocols: 'dev', 'full')" in none.output
    )


def test_compare_json(two_notebooks: Store, tmp_path: Path):
    store = two_notebooks
    a = _add(store, _write_run(tmp_path / "a.csv", shift=0, draws=2, seed=1, n_items=12), "--label", "a")
    path_b = tmp_path / "b.csv"
    _write_run(path_b, shift=0.2, draws=1, seed=2, n_items=14)  # two extra items, one draw each
    b = _add(store, path_b, "--label", "b")
    result = _cli(
        store, "compare", "a", "b", "--json", "--resamples", "1000", "--seed", "3", "--confidence", "0.9"
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    assert data["metric"] == "score" and data["arm_a"]["runs"] == [a] and data["arm_b"]["runs"] == [b]
    assert data["n_items"] == 12 and data["only_b"] == ["q012", "q013"] and data["only_a"] == []
    assert data["verdict"] == "B better" and data["ci_low"] > 0
    assert (data["resamples"], data["seed"], data["confidence"]) == (1000, 3, 0.9)
    assert data["draws_a"] == [2, 2.0, 2] and data["draws_b"] == [1, 1.0, 1]
    assert data["aa_a"]["n_items"] == 12 and data["aa_b"] is None
    assert "only 12" not in result.stdout  # warnings go to stderr


def _sub(
    store: Store, sid: str, nb: str | None, score: float | None, date: str, status: str = "complete"
) -> None:
    nb_id = Store.at(store.root).state().resolve_nb(nb).id if nb else None
    store.append(
        "sub.upsert", {"id": sid, "nb": nb_id, "public_score": score, "date": date, "status": status}
    )


def test_lb_without_repeats(two_notebooks: Store):
    store = two_notebooks
    assert "no submissions yet" in _cli(store, "lb").output
    _sub(store, "901", "base.ipynb", 0.61, "2026-10-02 08:00:00")
    _sub(store, "900", "new.ipynb", 0.64, "2026-10-01 08:00:00")
    result = _cli(store, "lb")
    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    assert lines[0].split() == ["ID", "DATE", "NOTEBOOK", "PUBLIC", "PRIVATE", "STATUS"]
    assert lines[2].split()[0] == "900" and lines[3].split()[0] == "901"  # newest last
    assert "LB noise: unknown" in result.output and "submitted at least twice" in result.output
    diff = _cli(store, "lb", "--diff", "900", "901")
    assert diff.exit_code == 0 and "B − A: −0.03" in diff.output and "verdict: unknown" in diff.output


def test_lb_with_repeats_and_diff(two_notebooks: Store):
    store = two_notebooks
    _sub(store, "1001", "base.ipynb", 0.70, "2026-10-01 10:00:00")
    _sub(store, "1002", "base.ipynb", 0.72, "2026-10-02 10:00:00")
    _sub(store, "1003", "new.ipynb", 0.75, "2026-10-03 10:00:00")
    _sub(store, "1004", "new.ipynb", 0.74, "2026-10-04 10:00:00")
    _sub(store, "1005", None, 0.90, "2026-10-05 10:00:00")
    _sub(store, "1006", None, None, "2026-10-06 10:00:00", status="error")
    result = _cli(store, "lb")
    assert result.exit_code == 0, result.output
    sd = math.sqrt((0.0002 + 0.00005) / 2)
    assert f"pooled SD {sd:.3g} of the public score, from 2 notebooks and 4 submissions" in result.output
    assert "1006" in result.output and "error" in result.output

    limited = _cli(store, "lb", "-n", "1", "--nb", "base.ipynb")
    rows = [line for line in limited.output.splitlines() if line.startswith("100")]
    assert [r.split()[0] for r in rows] == ["1002"] and "the newest 1 of 2" in limited.output

    beyond = _cli(store, "lb", "--diff", "1001", "1003")
    assert beyond.exit_code == 0, beyond.output
    assert "B − A: +0.05 (public score)" in beyond.output
    assert f"z = +{0.05 / (sd * math.sqrt(2)):.2f}" in beyond.output
    assert "verdict: B higher, beyond LB noise" in beyond.output
    within = _cli(store, "lb", "--diff", "1003", "1004")
    assert "z = −0.63" in within.output and "verdict: within LB noise" in within.output

    no_score = _cli(store, "lb", "--diff", "1001", "1006")
    assert no_score.exit_code == 1 and "1006 has no public score" in no_score.output
    unknown = _cli(store, "lb", "--diff", "1001", "77")
    assert unknown.exit_code == 1 and "no submission matches '77'" in unknown.output
