"""`audit`, `check` and `verdict`: which ideas are in a notebook, and does it hold what the team expects?"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click

from pseudorca import util
from pseudorca.audit.engine import LLM_MODES, AuditRow, AuditRows, audit_notebook
from pseudorca.audit.types import Evidence, Finding
from pseudorca.commands._common import (
    LEGEND,
    SYMBOLS,
    Ctx,
    fail,
    pass_ctx,
    show_warnings,
    user_errors,
    verdict_symbol,
    warn,
)
from pseudorca.model import IDEA_STATUSES, VERDICT_STATUSES, Effective, Idea, Notebook, State

if TYPE_CHECKING:
    from pseudorca.llm import LLMSettings
    from pseudorca.store import Store
    from pseudorca.tree import Tree

HEADERS = ["IDEA", "STATUS", "VERDICT", "METHOD", "EVIDENCE/NOTE"]
COUNT_ORDER = ("present", "partial", "absent", "unknown", "conflict", "none")
SECTIONS = ("MISSING", "UNKNOWN", "FORBIDDEN PRESENT", "CLAIMED BUT NOT FOUND", "OK")
_EVIDENCE_RE = re.compile(r"^(?P<path>.+?):(?P<line>\d*):(?P<quote>.*)$", re.DOTALL)


def llm_options(f: Callable[..., Any]) -> Callable[..., Any]:
    """--llm, --model, --base-url, --provider and --allow-remote."""
    options = [
        click.option(
            "--llm",
            "llm_mode",
            type=click.Choice(LLM_MODES),
            help="When to ask the LLM: auto (only when fingerprints and patch are inconclusive), always or "
            "never (default: [audit] llm in config.toml, else auto).",
        ),
        click.option("--model", metavar="M", help="LLM model (default: [llm] model)."),
        click.option("--base-url", metavar="URL", help="LLM endpoint (default: [llm] base_url)."),
        click.option("--provider", metavar="P", help="openai (any OpenAI-compatible endpoint) or anthropic."),
        click.option(
            "--allow-remote",
            is_flag=True,
            help="Allow sending notebook code to an endpoint that is not local.",
        ),
    ]
    for option in reversed(options):
        f = option(f)
    return f


def _llm(
    store: Store,
    llm_mode: str | None,
    model: str | None,
    base_url: str | None,
    provider: str | None,
    allow_remote: bool,
) -> tuple[str, LLMSettings | None]:
    """The LLM mode of this run and, unless it is "never", the settings ([llm] plus the options)."""
    mode = llm_mode or store.setting("audit", "llm", "auto") or "auto"
    if mode not in LLM_MODES:
        fail(f"[audit] llm in config.toml must be one of {', '.join(LLM_MODES)}, not {mode!r}")
    if mode == "never":
        return mode, None
    from pseudorca import llm

    overrides: dict[str, Any] = {
        k: v for k, v in (("model", model), ("base_url", base_url), ("provider", provider)) if v
    }
    if allow_remote:
        overrides["allow_remote"] = True
    try:
        return mode, llm.load_settings(store, **overrides)
    except llm.LLMError as e:
        if llm_mode or overrides:  # asked for on the command line: do not silently go without
            fail(f"cannot use the LLM: {e}")
        warn(f"LLM not used: {e}")
        return "never", None


def _progress(message: str) -> None:
    click.secho(message, err=True, dim=True)


def _report_messages(rows: AuditRows) -> None:
    for message in rows.messages:
        if "not configured" in message:
            message += " (set [llm] model in config.toml or pass --model)"
        warn(message)


# ----- rendering -----------------------------------------------------------------------------------


def _verdict_cell(eff: Effective | None, claimed: bool) -> str:
    word = eff.status if eff else ("claimed" if claimed else "none")
    return f"{verdict_symbol(eff, claimed)} {word}"


def _evidence_text(e: dict[str, Any]) -> str:
    where = f"{e.get('path', '')}:{e['line']}" if e.get("line") else str(e.get("path", ""))
    return f"{where} {e.get('quote', '')}".strip()


def _explain(eff: Effective | None, row: AuditRow | None) -> str:
    """The EVIDENCE/NOTE cell: where the idea is, or why it is not (or not known to be) there."""
    parts: list[str] = []
    if row is not None and row.llm_error:
        parts.append(f"llm error: {row.llm_error}")
    if eff is not None:
        evidence = [e for r in eff.records for e in r.evidence if isinstance(e, dict)]
        note = next((r.note for r in eff.records if r.note), "")
        if eff.status == "conflict":
            parts.append(", ".join(f"{r.method} says {r.status}" for r in eff.records))
        elif eff.status == "present" and evidence:
            parts.append(_evidence_text(evidence[0]))
        elif note or evidence:
            parts.append(note or _evidence_text(evidence[0]))
    if row is not None and (eff is None or eff.status not in ("present", "absent")):
        parts += [*row.notes, row.llm_note]
    return "; ".join(p for p in parts if p)


def _cells(idea: Idea, eff: Effective | None, row: AuditRow | None, claimed: bool) -> list[object]:
    return [idea.id, idea.status, _verdict_cell(eff, claimed), eff.method if eff else "", _explain(eff, row)]


def _summary(rows: AuditRows, dry_run: bool) -> str:
    counts = Counter(r.effective.status if r.effective else "none" for r in rows)
    verdicts = ", ".join(f"{counts[k]} {k}" for k in COUNT_ORDER if counts[k])
    new = sum(len(r.findings) for r in rows)
    skipped = sum(len(r.skipped) for r in rows)
    done = f"{new} new verdict{'' if new == 1 else 's'}" + (
        " not recorded (dry run)" if dry_run else " recorded"
    )
    if skipped:
        done += f", {skipped} skipped as already recorded (--force redoes them)"
    return f"{verdicts}; {done}"


# ----- audit ---------------------------------------------------------------------------------------


def _select(st: State, idea_refs: tuple[str, ...], statuses: tuple[str, ...]) -> list[Idea]:
    if idea_refs:
        ideas = [st.ideas[i] for i in dict.fromkeys(st.resolve_idea(r).id for r in idea_refs)]
    else:
        ideas = st.ideas_in_order()
    if statuses:
        return [i for i in ideas if i.status in statuses]
    return ideas if idea_refs else [i for i in ideas if i.status != "parked"]


@click.command("audit")
@click.argument("nb_ref", required=False, default="latest")
@click.option("--idea", "idea_refs", multiple=True, metavar="ID", help="Audit only this idea (repeatable).")
@click.option(
    "--status",
    "statuses",
    multiple=True,
    type=click.Choice(IDEA_STATUSES),
    help="Audit only ideas with this status (repeatable; default: all but parked).",
)
@llm_options
@click.option("--force", is_flag=True, help="Redo methods whose verdict is already recorded.")
@click.option("--dry-run", is_flag=True, help="Show the verdicts without recording them.")
@pass_ctx
def audit_cmd(
    c: Ctx,
    nb_ref: str,
    idea_refs: tuple[str, ...],
    statuses: tuple[str, ...],
    llm_mode: str | None,
    model: str | None,
    base_url: str | None,
    provider: str | None,
    allow_remote: bool,
    force: bool,
    dry_run: bool,
) -> None:
    """Check which ideas are in a notebook (default: the latest registered one).

    Fingerprints and patches are checked first, without running anything; the LLM is asked only when
    they are inconclusive (see --llm). Each new verdict is recorded as an event, and methods already
    recorded for the same idea revision are skipped.
    """
    store = c.store()
    show_warnings(store)
    with user_errors():
        st = store.state()
        nb = st.resolve_nb(nb_ref)
        ideas = _select(st, idea_refs, statuses)
    if not ideas:
        click.echo("no ideas to audit")
        return
    mode, settings = _llm(store, llm_mode, model, base_url, provider, allow_remote)
    with user_errors():
        rows = audit_notebook(
            store,
            nb,
            ideas,
            llm_mode=mode,
            settings=settings,
            force=force,
            dry_run=dry_run,
            on_progress=_progress,
        )
    _report_messages(rows)
    click.echo(
        f"audit of {nb.label} ({util.short(nb.id)})" + (", dry run: nothing recorded" if dry_run else "")
    )
    click.echo()
    click.echo(
        util.format_table(HEADERS, [_cells(r.idea, r.effective, r, r.idea.id in nb.applies) for r in rows])
    )
    click.echo()
    click.echo(_summary(rows, dry_run))
    click.echo(LEGEND)


# ----- check ---------------------------------------------------------------------------------------


def _statuses(store: Store, key: str, default: str) -> list[str]:
    value = store.setting("audit", key, [default])
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        fail(f"[audit] {key} in config.toml must be a list of idea statuses")
    return value


def _target(
    store: Store, target: str, name: str | None, tags: tuple[str, ...], dry_run: bool
) -> tuple[Notebook, Tree | None, str]:
    """(notebook, tree or None to extract it later, how it was found) for a file path or notebook ref."""
    path = Path(target)
    if not path.is_file():
        if name or tags:
            warn("--name and --tag apply only when TARGET is a file")
        return store.state().resolve_nb(target), None, "dry run: nothing recorded" if dry_run else ""
    from pseudorca.notebooks import display_path, register_file

    if not dry_run:
        nb, created = register_file(store, path, name=name, tags=list(tags))
        return nb, None, "newly registered" if created else "already registered"
    from pseudorca import extract

    data = path.read_bytes()
    sha = util.sha256_bytes(data)
    nb = store.state().notebooks.get(sha) or Notebook(
        id=sha,
        name=name or path.name,
        path=display_path(store, path),
        tags=list(tags),
        stored=False,
        size=len(data),
    )
    return nb, extract.extract_bytes(data, path.name), "dry run: nothing registered or recorded"


@click.command("check")
@click.argument("target")
@click.option("--expect", "expects", multiple=True, metavar="ID", help="Also expect this idea (repeatable).")
@click.option("--strict", is_flag=True, help="Also fail when an expected idea is unknown.")
@click.option("--dry-run", is_flag=True, help="Register nothing and record nothing.")
@click.option("--name", metavar="N", help="Name to register a file under (default: its file name).")
@click.option(
    "--tag", "tags", multiple=True, metavar="T", help="Tag for a file being registered (repeatable)."
)
@llm_options
@pass_ctx
def check_cmd(
    c: Ctx,
    target: str,
    expects: tuple[str, ...],
    strict: bool,
    dry_run: bool,
    name: str | None,
    tags: tuple[str, ...],
    llm_mode: str | None,
    model: str | None,
    base_url: str | None,
    provider: str | None,
    allow_remote: bool,
) -> None:
    """Check that a notebook holds the expected ideas and none of the forbidden ones.

    TARGET is a notebook file (registered first, unless --dry-run) or a registered notebook reference.

    \b
    expected:  status in [audit] expect_status (default: adopted), and --expect
    forbidden: status in [audit] forbid_status (default: rejected)

    Exits with 1 when an expected idea is absent, partial or disputed, or a forbidden idea is present;
    with --strict, also when an expected idea is unknown.
    """
    store = c.store()
    show_warnings(store)
    expect_status = _statuses(store, "expect_status", "adopted")
    forbid_status = _statuses(store, "forbid_status", "rejected")
    mode, settings = _llm(store, llm_mode, model, base_url, provider, allow_remote)
    with user_errors():
        extra = [store.state().resolve_idea(e).id for e in expects]
        try:
            nb, tree, how = _target(store, target, name, tags, dry_run)
        except OSError as e:
            fail(f"cannot read {target}: {e.strerror or e}")
    st = store.state()
    order = st.ideas_in_order()
    expected = list(dict.fromkeys([i.id for i in order if i.status in expect_status] + extra))
    forbidden = [i.id for i in order if i.status in forbid_status and i.id not in expected]
    ideas = [i for i in order if i.status != "parked" or i.id in expected or i.id in forbidden]

    with user_errors():
        rows = audit_notebook(
            store,
            nb,
            ideas,
            llm_mode=mode,
            settings=settings,
            dry_run=dry_run,
            tree=tree,
            on_progress=_progress,
        )
    _report_messages(rows)

    by_id = {r.idea.id: r for r in rows}
    st = store.state()

    def eff(iid: str) -> Effective | None:
        return by_id[iid].effective if iid in by_id else st.effective_verdict(nb.id, iid)

    def status(iid: str) -> str | None:
        e = eff(iid)
        return e.status if e else None

    sections = {
        "MISSING": [i for i in expected if status(i) in ("absent", "partial", "conflict")],
        "UNKNOWN": [i for i in expected if status(i) in (None, "unknown")],
        "FORBIDDEN PRESENT": [i for i in forbidden if status(i) == "present"],
        "CLAIMED BUT NOT FOUND": [i for i in nb.applies if i in st.ideas and status(i) == "absent"],
        "OK": [i for i in expected if status(i) == "present"],
    }
    listed = [iid for title in SECTIONS for iid in sections[title]]
    lines = util.format_table(
        HEADERS, [_cells(st.ideas[i], eff(i), by_id.get(i), i in nb.applies) for i in listed]
    ).splitlines()[2:]

    click.echo(f"check of {nb.label} ({util.short(nb.id)})" + (f", {how}" if how else ""))
    k = 0
    for title in SECTIONS:
        if sections[title]:
            click.echo()
            click.echo(f"{title} ({len(sections[title])})")
            for _ in sections[title]:
                click.echo("  " + lines[k])
                k += 1
    problems = []
    if sections["MISSING"]:
        problems.append(f"{len(sections['MISSING'])} missing")
    if sections["FORBIDDEN PRESENT"]:
        problems.append(f"{len(sections['FORBIDDEN PRESENT'])} forbidden present")
    if strict and sections["UNKNOWN"]:
        problems.append(f"{len(sections['UNKNOWN'])} unknown (--strict)")
    click.echo()
    if problems:
        click.echo("check failed: " + ", ".join(problems))
        click.get_current_context().exit(1)
    if not expected:
        click.echo("check passed (nothing is expected: set [audit] expect_status or pass --expect)")
    elif sections["UNKNOWN"]:
        click.echo(f"check passed ({len(sections['UNKNOWN'])} expected unknown; --strict fails on them)")
    else:
        click.echo("check passed")


# ----- verdict -------------------------------------------------------------------------------------


def _parse_evidence(text: str) -> Evidence:
    m = _EVIDENCE_RE.match(text)
    if m is None or not m["path"].strip():
        raise ValueError(f"--evidence {text!r} is not PATH:LINE:QUOTE (LINE may be empty)")
    line = int(m["line"]) if m["line"] else None
    if line == 0:
        raise ValueError(f"--evidence {text!r}: lines start at 1")
    return Evidence(m["path"].strip(), line, m["quote"].strip())


@click.command("verdict")
@click.argument("nb_ref")
@click.argument("idea_id")
@click.argument("status", type=click.Choice(VERDICT_STATUSES))
@click.option("--note", default="", help="Why; shown next to the verdict.")
@click.option(
    "--evidence",
    "evidence",
    multiple=True,
    metavar="PATH:LINE:QUOTE",
    help="Where in the notebook's tree the idea is (repeatable; LINE may be empty).",
)
@pass_ctx
def verdict_cmd(c: Ctx, nb_ref: str, idea_id: str, status: str, note: str, evidence: tuple[str, ...]) -> None:
    """Record a human verdict on whether an idea is in a notebook.

    A human verdict overrides fingerprints, patches and the LLM for this notebook and idea.
    """
    store = c.store()
    with user_errors():
        st = store.state()
        nb = st.resolve_nb(nb_ref)
        idea = st.resolve_idea(idea_id)
        items = [_parse_evidence(e) for e in evidence]
    store.append("audit.verdict", Finding(status, "human", items, note=note).event_data(nb.id, idea))
    click.echo(f"recorded: {idea.id} in {nb.label} is {SYMBOLS[status]} {status} (human verdict)")
