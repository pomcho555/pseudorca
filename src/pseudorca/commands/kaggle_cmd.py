"""`pseudorca kaggle`: read-only Kaggle access (sync submissions, pull kernel sources)."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from typing import Any

import click

from pseudorca import util
from pseudorca.commands._common import Ctx, fail, pass_ctx, user_errors, warn
from pseudorca.kaggle_api import KaggleError, SyncReport, pull_kernel, sync_submissions
from pseudorca.model import Notebook

SUB_HEADERS = ["ID", "DATE", "PUBLIC", "PRIVATE", "STATUS", "NOTEBOOK", "DESCRIPTION"]


@contextmanager
def _errors() -> Iterator[None]:
    """User errors and Kaggle failures as one-line messages."""
    try:
        with user_errors():
            yield
    except KaggleError as e:
        raise click.ClickException(str(e)) from e


def _local_time(iso: str) -> str:
    try:
        return datetime.fromisoformat(iso).astimezone().strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return iso


def _score(value: float | None) -> str:
    return "" if value is None else f"{value:.6g}"


def _nb_label(notebooks: dict[str, Notebook], nb_id: str | None) -> str:
    if not nb_id:
        return ""
    nb = notebooks.get(nb_id)
    return nb.label if nb else util.short(nb_id)


def _rows(report: SyncReport, notebooks: dict[str, Notebook]) -> list[list[Any]]:
    return [
        [
            sid,
            _local_time(r["date"]),
            _score(r["public_score"]),
            _score(r["private_score"]),
            r["status"],
            _nb_label(notebooks, r["nb"]),
            r["description"],
        ]
        for sid, r in report.rows.items()
    ]


@click.group("kaggle")
def kaggle_group() -> None:
    """Read-only Kaggle access: sync your team's submissions and pull kernel sources.

    Uses the kaggle package's own credentials (kaggle.json, or KAGGLE_USERNAME and KAGGLE_KEY). Pseudorca
    never submits, pushes or creates anything on Kaggle.
    """


@kaggle_group.command("sync")
@click.option(
    "-c", "--competition", metavar="SLUG", help="Competition slug (default: competition under [store])."
)
@click.option("--dry-run", is_flag=True, help="Show what would change without writing anything.")
@pass_ctx
def sync_cmd(ctx: Ctx, competition: str | None, dry_run: bool) -> None:
    """Record new and changed submissions, and link them to notebooks.

    An unlinked submission is linked when its description contains a notebook's sha prefix (8+ hex
    characters) or registered name (4+ characters), and only one notebook matches. Links are never changed.
    """
    store = ctx.store()
    slug = (competition or str(store.setting("store", "competition") or "")).strip()
    if not slug:
        fail("no competition: pass --competition SLUG or set competition under [store] in config.toml")
    with _errors():
        report = sync_submissions(store, slug, dry_run=dry_run)
    head = (
        f"{report.competition}: {len(report.new)} new, {len(report.updated)} updated, "
        f"{len(report.unchanged)} unchanged"
    )
    if report.linked:
        head += f"; {len(report.linked)} linked to notebooks"
    click.echo(head + (" (dry run: nothing written)" if dry_run else ""))
    if report.rows:
        click.echo(util.format_table(SUB_HEADERS, _rows(report, store.state().notebooks)))
    if report.ambiguous:
        ids = ", ".join(report.ambiguous[:10]) + (" ..." if len(report.ambiguous) > 10 else "")
        warn(
            f"{len(report.ambiguous)} submission(s) not linked because the description names several "
            f"notebooks: {ids} (link them with `pseudorca sub link`)"
        )
    for note in report.notes:
        warn(note)


@kaggle_group.command("pull")
@click.argument("kernel", metavar="OWNER/SLUG")
@click.option("--name", metavar="NAME", help="Name to register it under (default: the code file's name).")
@click.option("--tag", "tags", multiple=True, metavar="TAG", help="Tag it (repeatable).")
@click.option("--parent", metavar="REF", help="The notebook it was derived from.")
@click.option("--applies", multiple=True, metavar="ID", help="An idea its author says is in it (repeatable).")
@pass_ctx
def pull_cmd(
    ctx: Ctx,
    kernel: str,
    name: str | None,
    tags: tuple[str, ...],
    parent: str | None,
    applies: tuple[str, ...],
) -> None:
    """Download a kernel's latest source and register it.

    The source goes to .pseudorca/cache/kaggle/ (local only) and into the ledger; it is never run.
    """
    store = ctx.store()
    with _errors():
        nb, created = pull_kernel(store, kernel, name=name, tags=tags, parent=parent, applies=applies)
    if created:
        click.echo(f"registered {nb.label} {util.short(nb.id)} from {nb.kernel}")
    else:
        click.echo(f"already registered as {nb.label} {util.short(nb.id)}")
