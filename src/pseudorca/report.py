"""A single-file HTML report of the ledger: no scripts, no external resources. Open it locally.

The report holds the team's data. Keep it private: do not publish it (GitHub Pages sites are public even
when the repository is private).
"""

from __future__ import annotations

import html
from collections.abc import Iterable
from typing import Any

from pseudorca import __version__, util
from pseudorca.model import Effective, Idea, Notebook, State

SYMBOLS = {"present": "✓", "partial": "◐", "absent": "✗", "unknown": "?", "conflict": "!"}
CELL_CLASS = {"present": "ok", "partial": "part", "absent": "no", "unknown": "unk", "conflict": "conf"}

CSS = """
:root { --bg:#fff; --fg:#1d232a; --muted:#5f6b76; --line:#d9dee3; --head:#f4f6f8;
        --ok:#1f8a4c; --ok-bg:#e3f4ea; --part:#9a6700; --part-bg:#fdf3d8; --no:#b42318; --no-bg:#fde8e7;
        --unk:#5f6b76; --unk-bg:#eef1f4; --conf:#7a2bbf; --conf-bg:#f1e7fb; color-scheme: light; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#14181c; --fg:#e4e8ec; --muted:#9aa5b1; --line:#2c343c; --head:#1c2228;
          --ok:#5fd18f; --ok-bg:#16301f; --part:#e8b94a; --part-bg:#33290f; --no:#ff8a80; --no-bg:#3a1714;
          --unk:#9aa5b1; --unk-bg:#232a31; --conf:#c79cf0; --conf-bg:#2c1d3a; color-scheme: dark; }
}
body { background:var(--bg); color:var(--fg); font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif;
       margin:0; padding:24px max(16px, 3vw); }
h1 { font-size:22px; margin:0 0 4px; } h2 { font-size:17px; margin:32px 0 8px; }
.muted { color:var(--muted); } code { font-size:12.5px; }
.scroll { overflow-x:auto; }
table { border-collapse:collapse; } th, td { border:1px solid var(--line); padding:4px 8px; vertical-align:top; }
th { background:var(--head); text-align:left; font-weight:600; }
td.c { text-align:center; font-weight:700; min-width:3.2em; }
td.ok { color:var(--ok); background:var(--ok-bg); } td.part { color:var(--part); background:var(--part-bg); }
td.no { color:var(--no); background:var(--no-bg); } td.unk { color:var(--unk); background:var(--unk-bg); }
td.conf { color:var(--conf); background:var(--conf-bg); } td.none { color:var(--muted); }
td.num { text-align:right; font-variant-numeric:tabular-nums; }
ul.tree { margin:0; padding-left:20px; } ul.tree li { margin:2px 0; } ul.flat { list-style:none; padding-left:0; }
.tag { display:inline-block; border:1px solid var(--line); border-radius:9px; padding:0 7px; font-size:12px; margin-right:3px; }
footer { margin-top:40px; color:var(--muted); font-size:12.5px; }
"""


def _e(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def _tags(tags: Iterable[str]) -> str:
    return "".join(f'<span class="tag">{_e(t)}</span>' for t in tags)


def _cell(eff: Effective | None, claimed: bool) -> str:
    if eff is None:
        mark = "c" if claimed else "·"
        title = "claimed by the notebook, not audited yet" if claimed else "not audited"
        return f'<td class="c none" title="{_e(title)}">{mark}</td>'
    lines = [f"{eff.status} via {eff.method}" + (" (stale: the idea changed since)" if eff.stale else "")]
    for rec in eff.records[:3]:
        for ev in rec.evidence[:2]:
            where = f"{ev.get('path', '')}:{ev.get('line') or ''}"
            lines.append(f"{rec.method}: {where} {util.clip(str(ev.get('quote', '')), 120)}")
        if rec.note:
            lines.append(f"{rec.method}: {util.clip(rec.note, 160)}")
        lines.append(f"{util.local_time(rec.time)} by {rec.actor}")
    mark = SYMBOLS.get(eff.status, "?") + ("*" if eff.stale else "")
    return (
        f'<td class="c {CELL_CLASS.get(eff.status, "unk")}" title="{_e(chr(10).join(lines))}">{_e(mark)}</td>'
    )


def _board(state: State, ideas: list[Idea], notebooks: list[Notebook]) -> str:
    if not ideas or not notebooks:
        return '<p class="muted">Nothing to show yet: add ideas and notebooks, then run <code>pseudorca audit</code>.</p>'
    head = "".join(
        f'<th title="{_e(nb.id)}">{_e(util.clip(nb.label, 18))}<br><span class="muted"><code>{_e(util.short(nb.id, 8))}</code></span></th>'
        for nb in notebooks
    )
    rows = []
    for idea in ideas:
        cells = "".join(
            _cell(state.effective_verdict(nb.id, idea.id), idea.id in nb.applies) for nb in notebooks
        )
        rows.append(
            f'<tr><th><code>{_e(idea.id)}</code><br><span class="muted">{_e(util.clip(idea.title, 48))}</span></th>'
            f"<td>{_e(idea.status)}</td><td>{_e(idea.owner)}</td>{cells}</tr>"
        )
    legend = " ".join(f"{s} {name}" for name, s in SYMBOLS.items())
    return (
        f'<div class="scroll"><table><tr><th>idea</th><th>status</th><th>owner</th>{head}</tr>{"".join(rows)}</table></div>'
        f'<p class="muted">{_e(legend)} · c claimed, not audited · · none · * stale. Hover a cell for the evidence.</p>'
    )


def _lineage(state: State) -> str:
    """Notebooks as a family tree. A single child continues at its parent's indent and only branches
    indent, so a long linear history stays flat; nothing is ever dropped."""
    ordered = state.notebooks_in_order()
    if not ordered:
        return '<p class="muted">No notebooks yet.</p>'
    children: dict[str | None, list[Notebook]] = {}
    for nb in ordered:
        parent = nb.parent if nb.parent in state.notebooks and nb.parent != nb.id else None
        children.setdefault(parent, []).append(nb)
    rows: list[tuple[int, Notebook]] = []
    seen: set[str] = set()
    stack = [(nb, 0) for nb in reversed(children.get(None, []))]
    while stack:
        nb, depth = stack.pop()
        if nb.id in seen:
            continue
        seen.add(nb.id)
        rows.append((depth, nb))
        kids = children.get(nb.id, [])
        for kid in reversed(kids):
            stack.append((kid, depth if len(kids) == 1 else depth + 1))
    rows.extend((0, nb) for nb in ordered if nb.id not in seen)  # notebooks caught in a parent loop
    items = []
    for depth, nb in rows:
        parent = state.notebooks.get(nb.parent or "")
        bits = [f"{_e(util.local_day(nb.added))} by {_e(nb.added_by)}"]
        if parent is not None:
            bits.append(f"from {_e(parent.label)}")
        if nb.applies:
            bits.append(f"applies {', '.join(_e(a) for a in nb.applies)}")
        items.append(
            f'<li style="margin-left:{min(depth, 12) * 18}px"><b>{_e(nb.label)}</b> <code>{_e(util.short(nb.id))}</code> '
            f'{_tags(nb.tags)}<span class="muted">{" · ".join(bits)}</span></li>'
        )
    return f'<ul class="tree flat">{"".join(items)}</ul>'


def _submissions(state: State, lb_noise: Any) -> str:
    subs = sorted(state.subs.values(), key=lambda s: (util.instant(s.date), s.id))
    if not subs:
        return '<p class="muted">No submissions yet (<code>pseudorca kaggle sync</code> or <code>pseudorca sub add</code>).</p>'
    rows = []
    for s in subs:
        nb = state.notebooks.get(s.nb or "")
        rows.append(
            f"<tr><td>{_e(util.local_time(s.date))}</td><td><code>{_e(s.id)}</code></td><td>{_e(nb.label if nb else '')}</td>"
            f'<td class="num">{_e(s.public_score if s.public_score is not None else "")}</td>'
            f'<td class="num">{_e(s.private_score if s.private_score is not None else "")}</td>'
            f"<td>{_e(s.status)}</td><td>{_e(util.clip(s.description, 80))}</td></tr>"
        )
    noise = ""
    if lb_noise is not None:
        noise = (
            f"<p>LB noise: pooled SD <b>{lb_noise.sd:.3g}</b> from {lb_noise.n_groups} notebook(s) submitted more than "
            f"once ({lb_noise.n_subs} submissions). Two single submissions differ by noise alone with SD "
            f"{lb_noise.sd * 2**0.5:.3g}.</p>"
        )
    else:
        noise = '<p class="muted">LB noise: submit the same notebook at least twice to estimate it.</p>'
    return (
        '<div class="scroll"><table><tr><th>date</th><th>id</th><th>notebook</th><th>public</th><th>private</th>'
        f"<th>status</th><th>description</th></tr>{''.join(rows)}</table></div>{noise}"
    )


def _ideas(ideas: list[Idea]) -> str:
    if not ideas:
        return '<p class="muted">No ideas yet.</p>'
    rows = []
    for i in ideas:
        checks = [f"{len(i.fingerprints)} fingerprint(s)"] if i.fingerprints else []
        if i.patch:
            checks.append("patch")
        links = " ".join(
            f'<a href="{_e(u)}">{_e(util.clip(u, 50))}</a>'
            for u in i.links
            if u.startswith(("http://", "https://"))
        )
        rows.append(
            f"<tr><td><code>{_e(i.id)}</code></td><td>{_e(i.title)}</td><td>{_e(i.status)}</td><td>{_e(i.owner)}</td>"
            f"<td>{_tags(i.tags)}</td><td>{_e(', '.join(checks) or '-')}</td>"
            f"<td>{_e(util.clip(i.description, 160))} {links}</td></tr>"
        )
    return (
        '<div class="scroll"><table><tr><th>id</th><th>title</th><th>status</th><th>owner</th><th>tags</th>'
        f"<th>checks</th><th>description</th></tr>{''.join(rows)}</table></div>"
    )


def _activity(state: State, n: int = 30) -> str:
    rows = []
    for ev in reversed(state.events[-n:]):
        d = ev.data
        what = d.get("id") or f"{d.get('idea', '')} on {util.short(d.get('nb'), 8)}"
        extra = d.get("status", "")
        rows.append(
            f"<tr><td>{_e(util.local_time(ev.time))}</td><td>{_e(ev.actor)}</td><td>{_e(ev.type)}</td>"
            f"<td>{_e(util.clip(str(what), 60))} {_e(extra)}</td></tr>"
        )
    if not rows:
        return '<p class="muted">No activity yet.</p>'
    return f'<div class="scroll"><table><tr><th>time</th><th>who</th><th>event</th><th>what</th></tr>{"".join(rows)}</table></div>'


def render(
    state: State,
    *,
    store_name: str,
    notebooks: list[Notebook] | None = None,
    include_parked: bool = False,
    lb_noise: Any = None,
    generated: str | None = None,
) -> str:
    """The whole report as an HTML string (the body of a page; it has its own <title> and <style>)."""
    ideas = [i for i in state.ideas_in_order() if include_parked or i.status != "parked"]
    shown = notebooks if notebooks is not None else state.notebooks_in_order()[-8:]
    generated = generated or util.now_iso()
    counts = (
        f"{len(state.ideas)} ideas · {len(state.notebooks)} notebooks · {len(state.subs)} submissions · "
        f"{len(state.runs)} runs · generated {generated}"
    )
    return (
        f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>{_e(store_name)} · Pseudorca</title><style>{CSS}</style></head><body>"
        f"<h1>{_e(store_name)}</h1><div class='muted'>{_e(counts)}</div>"
        f"<h2>Which ideas are in which notebook</h2>{_board(state, ideas, shown)}"
        f"<h2>Notebooks</h2>{_lineage(state)}"
        f"<h2>Submissions</h2>{_submissions(state, lb_noise)}"
        f"<h2>Ideas</h2>{_ideas(ideas)}"
        f"<h2>Recent activity</h2>{_activity(state)}"
        f"<footer>Pseudorca {_e(__version__)}. This file contains your team's data: keep it private and do not "
        f"publish it (GitHub Pages sites are public even for private repositories).</footer></body></html>"
    )
