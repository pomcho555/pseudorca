"""`audit`, `check` and `verdict`: which ideas are in a notebook, and does it hold what the team expects?"""

from __future__ import annotations

import json
import os
import re
import shlex
import sys
import traceback
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO

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
from pseudorca.store import StoreError

if TYPE_CHECKING:
    from pseudorca.llm import LLMSettings
    from pseudorca.store import Store
    from pseudorca.tree import Tree

HEADERS = ["IDEA", "STATUS", "VERDICT", "METHOD", "EVIDENCE/NOTE"]
COUNT_ORDER = ("present", "partial", "absent", "unknown", "conflict", "none")
SECTIONS = ("MISSING", "UNKNOWN", "FORBIDDEN PRESENT", "CLAIMED BUT NOT FOUND", "OK")
SECTION_KEYS = {title: title.lower().replace(" ", "_") for title in SECTIONS}  # in --json
EVIDENCE_WIDTH = 60  # display columns of the EVIDENCE/NOTE cell (util.format_table's default)
CHECK_FAILED, CHECK_ERROR = 1, 2  # check's exit codes besides 0
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
        click.option("--model", metavar="MODEL", help="LLM model (default: [llm] model)."),
        click.option("--base-url", metavar="URL", help="LLM endpoint (default: [llm] base_url)."),
        click.option(
            "--provider", metavar="PROVIDER", help="openai (any OpenAI-compatible endpoint) or anthropic."
        ),
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


def _echo_json(obj: Any) -> None:
    click.echo(json.dumps(obj, ensure_ascii=False, indent=2))


# ----- rendering -----------------------------------------------------------------------------------


def _verdict_cell(eff: Effective | None, claimed: bool) -> str:
    word = eff.status if eff else ("claimed" if claimed else "none")
    return f"{verdict_symbol(eff, claimed)} {word}"


def _excerpt(quote: str, col: int | None, width: int) -> str:
    """util.excerpt, with the quote first started a little before `col`: util.excerpt places its window in
    characters, so on a line of wide (East Asian) characters the match could fall outside it."""
    if col is None or util.display_width(quote) <= width:
        return util.excerpt(quote, col, width)
    start = max(0, col - (width - 1) // 4)
    # a match near the end leaves room: show more of what comes before it
    while start > 0 and util.display_width(quote[start - 1 :]) <= width - 1:
        start -= 1
    if not start:
        return util.excerpt(quote, col, width)
    return "…" + util.excerpt(quote[start:], col - start, width - 1)


def _evidence_text(e: dict[str, Any], width: int = EVIDENCE_WIDTH) -> str:
    """PATH:LINE and the quote, cut to `width` around where the match starts (`col`) when that is known."""
    where = f"{e.get('path', '')}:{e['line']}" if e.get("line") else str(e.get("path", ""))
    quote = str(e.get("quote") or "")
    col = e.get("col")
    if type(col) is not int or not 0 <= col <= len(quote):
        col = None
    room = max(16, width - util.display_width(where) - 1)
    return f"{where} {_excerpt(quote, col, room)}".strip()


def _explain(eff: Effective | None, row: AuditRow | None) -> str:
    """The EVIDENCE/NOTE cell: where the idea is, or why it is not (or not known to be) there."""
    parts: list[str] = []
    if row is not None and row.llm_error:
        parts.append(f"llm error: {row.llm_error}")
    if eff is not None:
        # the records behind the verdict, such as the LLM's when it settled a fingerprint/patch conflict
        backing = [r for r in eff.records if r.status == eff.status] or eff.records
        evidence = [e for r in backing for e in r.evidence if isinstance(e, dict)]
        note = next((r.note for r in backing if r.note), "")
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


def _counts(rows: AuditRows) -> Counter[str]:
    return Counter(r.effective.status if r.effective else "none" for r in rows)


def _summary(rows: AuditRows, dry_run: bool) -> str:
    counts = _counts(rows)
    verdicts = ", ".join(f"{counts[k]} {k}" for k in COUNT_ORDER if counts[k])
    new = sum(len(r.findings) for r in rows)
    skipped = sum(len(r.skipped) for r in rows)
    done = f"{new} new verdict{'' if new == 1 else 's'}" + (
        " not recorded (dry run)" if dry_run else " recorded"
    )
    if skipped:
        done += f", {skipped} skipped as already recorded (--force redoes them)"
    return f"{verdicts}; {done}"


def _finding_json(f: Finding) -> dict[str, Any]:
    return {
        "method": f.method,
        "status": f.status,
        "evidence": [e.to_dict() for e in f.evidence],
        "note": f.note,
        "model": f.model,
        "detail": f.detail,
    }


def _idea_json(idea: Idea, eff: Effective | None, row: AuditRow | None, claimed: bool) -> dict[str, Any]:
    """One idea in --json: its effective verdict and, when it was audited now, what the audit did."""
    return {
        "idea": idea.id,
        "idea_status": idea.status,
        "claimed": claimed,
        "verdict": asdict(eff) if eff is not None else None,
        "new": [_finding_json(f) for f in row.findings] if row else [],
        "skipped": list(row.skipped) if row else [],
        "notes": list(row.notes) if row else [],
        "llm_note": row.llm_note if row else "",
        "llm_error": row.llm_error if row else "",
    }


# ----- audit ---------------------------------------------------------------------------------------


def _select(st: State, idea_refs: tuple[str, ...], statuses: tuple[str, ...]) -> list[Idea]:
    if idea_refs:
        ideas = [st.ideas[i] for i in dict.fromkeys(st.resolve_idea(r).id for r in idea_refs)]
    else:
        ideas = st.ideas_in_order()
    if statuses:
        return [i for i in ideas if i.status in statuses]
    return ideas if idea_refs else [i for i in ideas if i.status != "parked"]


def _audit_target(st: State, nb_ref: str | None) -> Notebook:
    """The notebook NB_REF names (default: the latest registered one). When NB_REF is also an existing file,
    whatever its suffix (as in `check`: `solver-v1.5` may name a file), that file's version is audited if it
    is registered (under any name); a file that is not is pointed out, never silently replaced by the
    registered notebook of the same name."""
    if nb_ref is None:
        return st.resolve_nb("latest")
    path = Path(nb_ref)
    if not path.is_file():
        return st.resolve_nb(nb_ref)
    sha = util.sha256_file(path)
    if sha in st.notebooks:
        return st.notebooks[sha]
    q = shlex.quote(nb_ref)
    todo = f"run `pseudorca nb add {q}` (or `pseudorca check {q}`) to audit the file"
    try:
        nb = st.resolve_nb(nb_ref)
    except LookupError:
        raise LookupError(f"{nb_ref} is not registered; {todo}") from None
    warn(f"{nb_ref} on disk is not the registered version {util.short(nb.id)} audited here; {todo}")
    return nb


@click.command("audit")
@click.argument("nb_ref", required=False)
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
@click.option("--json", "as_json", is_flag=True, help="Print JSON.")
@pass_ctx
def audit_cmd(
    c: Ctx,
    nb_ref: str | None,
    idea_refs: tuple[str, ...],
    statuses: tuple[str, ...],
    llm_mode: str | None,
    model: str | None,
    base_url: str | None,
    provider: str | None,
    allow_remote: bool,
    force: bool,
    dry_run: bool,
    as_json: bool,
) -> None:
    """Check which ideas are in a notebook (default: the latest registered one).

    NB_REF is a notebook reference, or a notebook file whose version is registered (an existing file is
    always read as a file). Fingerprints and patches are checked first, without running anything; the LLM is
    asked only when they are inconclusive (see --llm). Each new verdict is recorded as an event, and methods
    already recorded for the same idea revision are skipped.
    """
    store = c.store()
    show_warnings(store)
    with user_errors():
        st = store.state()
        nb = _audit_target(st, nb_ref)
        ideas = _select(st, idea_refs, statuses)
    rows = AuditRows()
    if ideas:
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
    if as_json:
        counts = _counts(rows)
        _echo_json(
            {
                "notebook": asdict(nb),
                "dry_run": dry_run,
                "ideas": [_idea_json(r.idea, r.effective, r, r.idea.id in nb.applies) for r in rows],
                "counts": {k: counts[k] for k in COUNT_ORDER if counts[k]},
                "new_verdicts": sum(len(r.findings) for r in rows),
                "skipped": sum(len(r.skipped) for r in rows),
                "messages": list(rows.messages),
            }
        )
        return
    if not ideas:
        click.echo("no ideas to audit")
        return
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


class CheckError(click.ClickException):
    """`check` could not run, which CI must tell apart from a check that failed."""

    exit_code = CHECK_ERROR


def _statuses(store: Store, key: str, default: str) -> list[str]:
    value = store.setting("audit", key, [default])
    if isinstance(value, str):
        value = [value]
    known = ", ".join(IDEA_STATUSES)
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        fail(f"[audit] {key} in config.toml must be a list of idea statuses ({known})")
    for v in value:
        if v not in IDEA_STATUSES:
            fail(f"[audit] {key}: {v!r} is not an idea status ({known})")
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
@click.option("--name", metavar="NAME", help="Name to register a file under (default: its file name).")
@click.option(
    "--tag", "tags", multiple=True, metavar="TAG", help="Tag for a file being registered (repeatable)."
)
@llm_options
@click.option("--json", "as_json", is_flag=True, help="Print JSON: the sections, the verdicts and `passed`.")
@pass_ctx
def check_cmd(c: Ctx, **options: Any) -> None:
    """Check that a notebook holds the expected ideas and none of the forbidden ones.

    TARGET is a notebook file (registered first, unless --dry-run) or a registered notebook reference.

    \b
    expected:  status in [audit] expect_status (default: adopted), and --expect
    forbidden: status in [audit] forbid_status (default: rejected)

    \b
    Exit codes:
      0  the check passed
      1  the check failed: an expected idea is absent, partial or disputed, or a forbidden idea is
         present; with --strict, also when an expected idea is unknown
      2  the check could not run: any error, including a usage error
    """
    try:
        _check(c, **options)  # a pipe closed early while the outcome is printed keeps that outcome
    except click.exceptions.Exit:  # passed (0) or failed (1)
        raise
    except BrokenPipeError:  # stderr closed before the outcome was known: the check did not run to its end
        _drop_output(sys.stdout, sys.stderr)
        raise click.exceptions.Exit(CHECK_ERROR) from None
    except click.ClickException as e:  # one-line errors, usage errors included
        e.exit_code = CHECK_ERROR
        raise
    except (KeyboardInterrupt, click.Abort):
        raise CheckError("interrupted") from None
    except Exception as e:
        if os.environ.get("PSEUDORCA_DEBUG"):
            raise
        if isinstance(e, (StoreError, LookupError, ValueError, OSError)) and not isinstance(e, KeyError):
            raise CheckError(str(e) or type(e).__name__) from e
        traceback.print_exc()  # a bug: show it, but never let it look like a failed check
        raise CheckError(f"check could not run: {type(e).__name__}: {e}") from e


def _check(
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
    as_json: bool,
) -> None:
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
    problems = []
    if sections["MISSING"]:
        problems.append(f"{len(sections['MISSING'])} missing")
    if sections["FORBIDDEN PRESENT"]:
        problems.append(f"{len(sections['FORBIDDEN PRESENT'])} forbidden present")
    if strict and sections["UNKNOWN"]:
        problems.append(f"{len(sections['UNKNOWN'])} unknown (--strict)")
    listed = [iid for title in SECTIONS for iid in sections[title]]
    code = CHECK_FAILED if problems else 0  # decided before anything is printed: a cut pipe cannot change it

    try:
        if as_json:
            shown = list(dict.fromkeys([*by_id, *listed]))
            _echo_json(
                {
                    "notebook": asdict(nb),
                    "registration": how,  # newly or already registered, a dry run, or "" for a reference
                    "dry_run": dry_run,
                    "strict": strict,
                    "passed": not problems,
                    "problems": problems,
                    "expected": expected,
                    "forbidden": forbidden,
                    "sections": {SECTION_KEYS[title]: sections[title] for title in SECTIONS},
                    "ideas": [_idea_json(st.ideas[i], eff(i), by_id.get(i), i in nb.applies) for i in shown],
                    "messages": list(rows.messages),
                }
            )
        else:
            table = util.format_table(
                HEADERS, [_cells(st.ideas[i], eff(i), by_id.get(i), i in nb.applies) for i in listed]
            ).splitlines()[2:]
            click.echo(f"check of {nb.label} ({util.short(nb.id)})" + (f", {how}" if how else ""))
            _show_sections(sections, table)
            click.echo(_check_outcome(problems, expected, expect_status, sections))
    except BrokenPipeError:  # stdout closed early (`| head`): the outcome stands
        _drop_output(sys.stdout)
    if code:
        click.get_current_context().exit(code)


def _show_sections(sections: dict[str, list[str]], table: list[str]) -> None:
    """Each non-empty section with its rows of `table` (one row per idea, in section order)."""
    lines = iter(table)
    for title in SECTIONS:
        if sections[title]:
            click.echo()
            click.echo(f"{title} ({len(sections[title])})")
            for _ in sections[title]:
                click.echo("  " + next(lines))
    click.echo()


def _check_outcome(
    problems: list[str], expected: list[str], expect_status: list[str], sections: dict[str, list[str]]
) -> str:
    """The last line of a check: why it failed, or that it passed (and what that pass rests on)."""
    if problems:
        return "check failed: " + ", ".join(problems)
    if not expected:
        why = "[audit] expect_status is empty"
        if expect_status:
            why = f"no idea has status {' or '.join(expect_status)}"
        return f"check passed (nothing is expected: {why}; pass --expect ID to expect one)"
    if sections["UNKNOWN"]:
        return f"check passed ({len(sections['UNKNOWN'])} expected unknown; --strict fails on them)"
    return "check passed"


def _drop_output(*streams: TextIO) -> None:
    """Point these streams' file descriptors at os.devnull once a reader closed the pipe early (`| head`),
    so that nothing more fails on it, not even the flush at exit (which would make the exit code 120)."""
    for stream in streams:
        try:
            fd = stream.fileno()
            devnull = os.open(os.devnull, os.O_WRONLY)
            try:
                os.dup2(devnull, fd)
            finally:
                os.close(devnull)
        except (OSError, ValueError):  # no file descriptor (a test runner's buffer): nothing to flush there
            pass


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
