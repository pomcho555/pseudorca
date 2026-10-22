"""`pseudorca report`: the ledger as one HTML file."""

from __future__ import annotations

from pathlib import Path

import click

from pseudorca.commands._common import Ctx, pass_ctx, show_warnings, user_errors


@click.command("report")
@click.option(
    "-o",
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Where to write the HTML (default: .pseudorca/cache/report.html, which git ignores).",
)
@click.option(
    "--nb", "nb_refs", multiple=True, metavar="REF", help="A notebook to show as a board column (repeatable)."
)
@click.option(
    "-n",
    "last",
    type=click.IntRange(min=0),
    default=8,
    show_default=True,
    metavar="N",
    help="Latest notebooks to show (0 = all).",
)
@click.option("--all", "include_parked", is_flag=True, help="Include parked ideas.")
@pass_ctx
def report_cmd(
    ctx: Ctx, output: Path | None, nb_refs: tuple[str, ...], last: int, include_parked: bool
) -> None:
    """Write a single-file HTML report: board, lineage, submissions, ideas and recent activity.

    The file contains your team's data. Keep it private.
    """
    store = ctx.store()
    with user_errors():
        state = store.state()
        ordered = state.notebooks_in_order()
        notebooks = (
            [state.resolve_nb(r) for r in nb_refs] if nb_refs else (ordered[-last:] if last else ordered)
        )
        try:
            from pseudorca import stats

            noise = stats.lb_noise(list(state.subs.values()))
        except (ImportError, AttributeError):
            noise = None
        from pseudorca.report import render

        page = render(
            state,
            store_name=store.setting("store", "name") or store.root.parent.name,
            notebooks=notebooks,
            include_parked=include_parked,
            lb_noise=noise,
        )
    target = output or store.cache_dir() / "report.html"
    target.write_text(page, encoding="utf-8")
    click.echo(f"wrote {target}")
    show_warnings(store)
