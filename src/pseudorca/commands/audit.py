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
from dataclasses import asdict, dataclass, field
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
LABEL_WIDTH = 40  # display columns of a notebook's label in the lines of `audit --submitted`
MAX_LLM_ERRORS = 5  # distinct LLM error texts `audit --submitted` shows; the others are counted
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
    click.secho(util.clean(message), err=True, dim=True)  # idea ids and labels come from the ledger


def _report_messages(rows: AuditRows, prefix: str = "") -> None:
    for message in rows.messages:
        if "not configured" in message:
            message += " (set [llm] model in config.toml or pass --model)"
        warn(prefix + message)


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


def _verdict_counts(rows: AuditRows) -> str:
    """How many ideas have each verdict, as "2 present, 1 absent, 1 none"."""
    counts = _counts(rows)
    return ", ".join(f"{counts[k]} {k}" for k in COUNT_ORDER if counts[k])


def _summary(rows: AuditRows, dry_run: bool) -> str:
    verdicts = _verdict_counts(rows)
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


def _tallies(rows: AuditRows) -> dict[str, Any]:
    """In --json: how many ideas have each verdict, how many verdicts are new, how many methods were skipped."""
    counts = _counts(rows)
    return {
        "counts": {k: counts[k] for k in COUNT_ORDER if counts[k]},
        "new_verdicts": sum(len(r.findings) for r in rows),
        "skipped": sum(len(r.skipped) for r in rows),
    }


def _audit_json(nb: Notebook, rows: AuditRows, dry_run: bool) -> dict[str, Any]:
    """One notebook's audit in --json."""
    return {
        "notebook": asdict(nb),
        "dry_run": dry_run,
        "ideas": [_idea_json(r.idea, r.effective, r, r.idea.id in nb.applies) for r in rows],
        **_tallies(rows),
        "messages": list(rows.messages),
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
@click.option(
    "--submitted",
    is_flag=True,
    help="Audit every notebook a submission links to, oldest submission first, instead of NB_REF.",
)
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
    submitted: bool,
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

    --submitted audits every registered notebook that a submission links to instead, each once, oldest
    submission first, so that `effects` has verdicts for all of them. A submission without a registered
    notebook, and a notebook that cannot be read (its bytes are not in this store, its blob does not match
    its sha256, or it is not a notebook), are skipped with a warning.
    """
    if submitted:
        if nb_ref is not None:
            fail("use NB_REF or --submitted, not both")
        llm_args = (llm_mode, model, base_url, provider, allow_remote)
        _audit_submitted(c, idea_refs, statuses, llm_args, force=force, dry_run=dry_run, as_json=as_json)
        return
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
        _echo_json(_audit_json(nb, rows, dry_run))
        return
    if not ideas:
        click.echo("no ideas to audit")
        return
    click.echo(
        f"audit of {util.clean(nb.label)} ({util.short(nb.id)})"
        + (", dry run: nothing recorded" if dry_run else "")
    )
    click.echo()
    click.echo(
        util.format_table(HEADERS, [_cells(r.idea, r.effective, r, r.idea.id in nb.applies) for r in rows])
    )
    click.echo()
    click.echo(_summary(rows, dry_run))
    click.echo(LEGEND)


# ----- audit --submitted ---------------------------------------------------------------------------


@dataclass
class SubmittedNotebook:
    """A notebook that submissions link to, with their ids (oldest first) and, once audited, its rows."""

    nb: Notebook
    subs: list[str]
    rows: AuditRows = field(default_factory=AuditRows)


def _audit_submitted(
    c: Ctx,
    idea_refs: tuple[str, ...],
    statuses: tuple[str, ...],
    llm_args: tuple[str | None, str | None, str | None, str | None, bool],
    *,
    force: bool,
    dry_run: bool,
    as_json: bool,
) -> None:
    """`audit --submitted`: each submitted notebook as `audit` would, one line each instead of a table. The
    LLM endpoint and its remote guard are set up once, for all of them."""
    store = c.store()
    show_warnings(store)
    with user_errors():
        st = store.state()
        ideas = _select(st, idea_refs, statuses)
        targets, skipped = _submitted(store, st)
    for entry in skipped:
        warn(entry["message"])
    mode, settings = "never", None
    if ideas and targets:
        mode, settings = _llm(store, *llm_args)
    if not as_json:
        if not targets:
            n = sum(len(entry["submissions"]) for entry in skipped)
            why = f"{n} submission{'' if n == 1 else 's'} skipped" if n else "no submissions yet"
            hint = "" if n else " (`pseudorca sub add` or `pseudorca kaggle sync`)"
            click.echo(f"no submitted notebooks to audit: {why}{hint}")
            return
        if not ideas:
            click.echo("no ideas to audit")
            return
        click.echo(
            f"{len(targets)} submitted notebook{'' if len(targets) == 1 else 's'} to audit, oldest submission "
            "first" + (", dry run: nothing recorded" if dry_run else "")
        )
        click.echo()
    width = max((util.display_width(_label(t.nb)) for t in targets), default=0)
    audited: list[SubmittedNotebook] = []
    for t in targets:
        if ideas:
            try:
                with user_errors():
                    t.rows = audit_notebook(
                        store,
                        t.nb,
                        ideas,
                        llm_mode=mode,
                        settings=settings,
                        force=force,
                        dry_run=dry_run,
                        on_progress=_progress_of(t.nb),
                    )
            except click.ClickException as e:
                # The audit reads the notebook when a method first needs it, before it records anything for
                # it. If the extractor, asked again, cannot read it either, the notebook is skipped like those
                # found before the run; any other error (an event that cannot be written, say) stops the run.
                reason, why = "", ""
                if isinstance(e.__cause__, (StoreError, ValueError)):
                    reason, why = _unreadable(store, t.nb)
                if not reason:  # say which notebook (the verdicts before it are recorded)
                    e.message = util.clean(f"{t.nb.label} ({util.short(t.nb.id)}): {e.message}")
                    raise
                skipped.append(_skipped_notebook(t.nb, t.subs, reason, why))
                warn(skipped[-1]["message"])
                continue
        audited.append(t)
        if not as_json:
            click.echo(_notebook_line(t.nb, t.rows, width))
    _report_each_once(audited)
    every = AuditRows(r for t in audited for r in t.rows)
    if as_json:
        totals = {"notebooks": len(audited), "submissions": sum(len(t.subs) for t in audited)}
        _echo_json(
            {
                "dry_run": dry_run,
                "notebooks": [_audit_json(t.nb, t.rows, dry_run) for t in audited],
                "skipped": skipped,
                "totals": {**totals, **_tallies(every)},
            }
        )
        return
    click.echo()
    if not audited:  # each one was skipped, with a warning
        click.echo("no submitted notebook could be read")
        return
    n = len(audited)
    click.echo(f"{n} notebook{'' if n == 1 else 's'}: {_summary(every, dry_run)}{_llm_errors(every)}")
    click.echo(LEGEND)


def _submitted(store: Store, st: State) -> tuple[list[SubmittedNotebook], list[dict[str, Any]]]:
    """The registered notebooks that submissions link to, each once, in the order of its oldest submission
    (as `effects` lists them); and what cannot be audited, each with the warning that says how to fix it:
    a submission without a registered notebook, and a notebook the extractor cannot read (`_unreadable`)."""
    from pseudorca.effects import sub_order

    targets: dict[str, SubmittedNotebook] = {}
    unreadable: dict[str, list[str]] = {}  # notebook id -> its submissions (shared with its skipped entry)
    skipped: list[dict[str, Any]] = []
    for sub in sorted(st.subs.values(), key=sub_order):
        nb_id = sub.nb if isinstance(sub.nb, str) and sub.nb else None
        nb = st.notebooks.get(nb_id) if nb_id else None
        if nb is None:
            skipped.append(_unlinked(sub.id, nb_id))
        elif nb.id in targets:
            targets[nb.id].subs.append(sub.id)
        elif nb.id in unreadable:
            unreadable[nb.id].append(sub.id)
        else:
            # a sound blob is read (and extracted) only when the audit needs it; without one, the extractor
            # may still read the notebook, from a fresh cached tree or from its file
            reason, why = ("", "") if store.blob_ok(nb.id) else _unreadable(store, nb)
            if reason:
                unreadable[nb.id] = [sub.id]
                skipped.append(_skipped_notebook(nb, unreadable[nb.id], reason, why))
            else:
                targets[nb.id] = SubmittedNotebook(nb, [sub.id])
    return list(targets.values()), skipped


def _unlinked(sub_id: str, nb_id: str | None) -> dict[str, Any]:
    """The skipped entry of a submission without a notebook (nb_id None) or with one that is not registered."""
    link = f"`pseudorca sub link {shlex.quote(sub_id)} NB`"
    if nb_id is None:
        reason, why = "no_notebook", f"it has no notebook (link one with {link})"
    else:
        reason = "not_registered"
        why = (
            f"its notebook {util.short(nb_id)} is not registered (add it with `pseudorca nb add FILE` or "
            f"`pseudorca kaggle pull OWNER/SLUG`, or link another with {link})"
        )
    message = f"skipped submission {sub_id}: {why}"
    return {"reason": reason, "submissions": [sub_id], "nb": nb_id, "message": message}


def _skipped_notebook(nb: Notebook, subs: list[str], reason: str, why: str) -> dict[str, Any]:
    """The skipped entry of a notebook that cannot be read (reason not_stored or unreadable)."""
    message = f"skipped {nb.label} ({util.short(nb.id)}): {why}"
    return {"reason": reason, "submissions": subs, "nb": nb.id, "message": message}


def _unreadable(store: Store, nb: Notebook) -> tuple[str, str]:
    """(reason, why) when the extractor cannot read the notebook, else ("", ""). It is read as the audit
    reads it (`extract.tree_for`): a fresh cached tree, else its blob, checked against its sha256, else the
    file at its recorded path while that still has the same sha256 (a notebook added with --no-store). A
    tree extracted here is cached, so the audit does not extract it again.

    - not_stored: the bytes are not in this store, and no fresh cached tree or file stands in for them.
    - unreadable: a blob that does not match its sha256 (a Git LFS pointer, say), bytes that are not a
      notebook, or a file that cannot be read; `why` is the store's or the extractor's message.
    """
    from pseudorca import extract

    try:
        extract.tree_for(store, nb)
    except StoreError as e:
        if store.has_blob(nb.id):
            return "unreadable", str(e) or type(e).__name__
        return "not_stored", _not_stored(nb)
    except (ValueError, OSError) as e:
        return "unreadable", str(e) or type(e).__name__
    return "", ""


def _not_stored(nb: Notebook) -> str:
    """Why a notebook whose bytes are not in this store cannot be read, with what to do about it."""
    if nb.stored:
        return (
            "its bytes are not in this store (pull the latest store, or add the file again with "
            "`pseudorca nb add FILE`)"
        )
    where = f"{nb.path} is gone or has changed" if nb.path else "no file path was recorded for it"
    return (
        f"it was added with --no-store and {where} (restore that version, or add it with "
        "`pseudorca nb add FILE` so the store keeps its bytes)"
    )


def _label(nb: Notebook) -> str:
    """A notebook's label on one line, cut to LABEL_WIDTH display columns (util.clip makes it safe to print:
    labels come from the ledger)."""
    return util.clip(" ".join(nb.label.split()), LABEL_WIDTH)


def _llm_errors(rows: AuditRows) -> str:
    """How many of these ideas the LLM failed on (each is in the table of `audit NB`), as "; N LLM errors"
    to end a line; "" when it failed on none."""
    n = sum(1 for r in rows if r.llm_error)
    return f"; {n} LLM error{'' if n == 1 else 's'}" if n else ""


def _notebook_line(nb: Notebook, rows: AuditRows, width: int) -> str:
    """A notebook's line in `audit --submitted`: its label (padded to `width`), short sha, verdicts and LLM
    errors."""
    label = _label(nb)
    pad = " " * (width - util.display_width(label))
    new = sum(len(r.findings) for r in rows)
    verdicts = f"{_verdict_counts(rows)}; {new} new verdict{'' if new == 1 else 's'}{_llm_errors(rows)}"
    return f"{label}{pad}  {util.short(nb.id)}  {verdicts}"


def _progress_of(nb: Notebook) -> Callable[[str], None]:
    """Progress lines that name the notebook they are about."""
    label = _label(nb)
    return lambda message: _progress(f"{label}: {message}")


def _where(nbs: list[Notebook]) -> str:
    """The notebooks a warning is about: "each of N notebooks", or the one notebook's label and short sha."""
    if len(nbs) > 1:
        return f"each of {len(nbs)} notebooks"
    return f"{nbs[0].label} ({util.short(nbs[0].id)})"


def _report_each_once(targets: list[SubmittedNotebook]) -> None:
    """The notebooks' messages and LLM errors, each text once, saying how many notebooks it came from: the
    LLM setup is shared, so the same message (the LLM is not configured, not allowed or not reachable) would
    otherwise repeat for each. Notebooks are counted, not named: versions pulled from one kernel often share
    a name. Error texts can differ on every call (a rate limit's wait), so the MAX_LLM_ERRORS most frequent
    are shown, and the others counted; --json has each one."""
    sources: dict[str, list[Notebook]] = {}
    for t in targets:
        for message in t.rows.messages:
            sources.setdefault(message, []).append(t.nb)
    for message, nbs in sources.items():
        _report_messages(AuditRows(messages=[message]), prefix=f"in {_where(nbs)}: ")
    failed: dict[str, list[Notebook]] = {}  # error text -> its notebook, once per idea the LLM failed on
    for t in targets:
        for r in t.rows:
            if r.llm_error:
                failed.setdefault(r.llm_error, []).append(t.nb)
    ranked = sorted(failed.items(), key=lambda item: -len(item[1]))  # stable: the first seen first
    for error, nbs in ranked[:MAX_LLM_ERRORS]:
        distinct = list({nb.id: nb for nb in nbs}.values())
        per_nb = set(Counter(nb.id for nb in nbs).values())
        if len(per_nb) == 1:  # as many ideas in each notebook
            k = per_nb.pop()
            warn(f"in {_where(distinct)}: the LLM failed on {k} idea{'' if k == 1 else 's'}: {error}")
        else:
            warn(f"in {len(distinct)} notebooks: the LLM failed on {len(nbs)} ideas in all: {error}")
    if rest := ranked[MAX_LLM_ERRORS:]:
        n = sum(len(nbs) for _, nbs in rest)
        texts = f"{len(rest)} other text{'' if len(rest) == 1 else 's'}"
        warn(f"{n} more LLM error{'' if n == 1 else 's'}, with {texts} (--json shows each one)")


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
            click.echo(f"check of {util.clean(nb.label)} ({util.short(nb.id)})" + (f", {how}" if how else ""))
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
    click.echo(util.clean(f"recorded: {idea.id} in {nb.label} is {SYMBOLS[status]} {status} (human verdict)"))
