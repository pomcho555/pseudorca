"""`pseudorca report`: the ledger as one HTML file."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import click

from pseudorca.commands._common import Ctx, pass_ctx, show_warnings, user_errors
from pseudorca.store import StoreError

if TYPE_CHECKING:
    from pseudorca.effects import Fit, Waterfall
    from pseudorca.model import State
    from pseudorca.report import Effects
    from pseudorca.store import Store

RESAMPLES = 10_000  # bootstrap resamples for the local runs: the default of `pseudorca effects`
# what computing the effects section may raise on an odd ledger (hand-edited events or blobs, a results
# table nested too deeply to parse, scores near the float limits); that part then becomes a note
EFFECT_ERRORS = (StoreError, ValueError, LookupError, OSError, ArithmeticError, RecursionError)
CONFIG = ".pseudorca/config.toml"


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
@click.option(
    "--all",
    "include_parked",
    is_flag=True,
    help="Include parked ideas (on the board and in the effects section).",
)
@click.option("--metric", default="score", show_default=True, help="Metric of the local runs.")
@click.option(
    "--resamples",
    type=click.IntRange(min=100),
    default=RESAMPLES,
    show_default=True,
    help="Bootstrap resamples for the local runs (fewer is quicker on big tables).",
)
@pass_ctx
def report_cmd(
    ctx: Ctx,
    output: Path | None,
    nb_refs: tuple[str, ...],
    last: int,
    include_parked: bool,
    metric: str,
    resamples: int,
) -> None:
    """Write a single-file HTML report: board, lineage, submissions, what each idea added, ideas and
    recent activity.

    The effects section shows what `pseudorca effects` prints with the same options (public scores; with
    submissions from several competitions, the store's competition). Whether lower scores are better, and
    a known LB noise SD, come from the [effects] table in .pseudorca/config.toml (lower_is_better, noise),
    which `effects` reads too. The file contains your team's data. Keep it private.
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
        except (ImportError, AttributeError, ArithmeticError):  # scores near the float limits overflow
            noise = None
        from pseudorca.report import render

        page = render(
            state,
            store_name=store.setting("store", "name") or store.root.parent.name,
            notebooks=notebooks,
            include_parked=include_parked,
            lb_noise=noise,
            effects=_effects(store, state, include_parked=include_parked, metric=metric, resamples=resamples),
        )
    target = output or store.cache_dir() / "report.html"
    target.write_text(page, encoding="utf-8")
    click.echo(f"wrote {target}")
    show_warnings(store)


def _effects(store: Store, state: State, *, include_parked: bool, metric: str, resamples: int) -> Effects:
    """The fits for the effects section, as `pseudorca effects` computes them with the same options and
    config.toml. A problem becomes a note in that section; it never stops the report."""
    from pseudorca import effects
    from pseudorca.commands.results import _load
    from pseudorca.report import Effects

    out = Effects()
    lower, noise = _settings(store, out.notes)
    try:
        out.lb = _lb_fit(
            store,
            state,
            out.notes,
            noise=noise,
            higher_is_better=not lower,
            include_parked=include_parked,
        )
        out.lb.notes = [_reworded(note) for note in out.lb.notes]
        out.breakdown = _breakdown(out.lb)
    except EFFECT_ERRORS as e:
        out.notes.append(f"Leaderboard: {_message(e)}")
    if state.runs:
        try:
            out.runs = effects.run_fits(
                state,
                lambda run: _load(store, run),
                metric=metric,
                higher_is_better=not lower,
                resamples=resamples,
                include_parked=include_parked,
            )
        except EFFECT_ERRORS as e:
            out.notes.append(f"Local runs: {_message(e)}")
    return out


def _settings(store: Store, notes: list[str]) -> tuple[bool, float | None]:
    """(lower_is_better, noise) from config.toml's [effects] table, as `pseudorca effects` reads them. A bad
    value is left out with a note, and the other one still counts."""
    from pseudorca import effects

    values: list[Any] = [False, None]
    keys = (
        ("lower_is_better", "higher scores count as better"),
        ("noise", "the noise comes from the residuals"),
    )
    for i, (key, instead) in enumerate(keys):

        def one(section: str, name: str, default: Any, key: str = key) -> Any:
            return store.setting(section, name, default) if name == key else default

        try:
            values[i] = effects.config_defaults(one)[i]
        except ValueError as e:
            notes.append(f"{e}; it is ignored here, so {instead}")
    return values[0], values[1]


def _lb_fit(store: Store, state: State, notes: list[str], **options: Any) -> Fit:
    """The LB fit; with submissions from several competitions, the store's competition's."""
    from pseudorca import effects

    names = sorted({s.competition for s in state.subs.values() if s.competition})
    if len(names) < 2:
        return effects.lb_fit(state, **options)
    competition = store.setting("store", "competition")
    if not isinstance(competition, str) or competition not in names:
        which = (
            f"the store's competition ({competition}) is none of them"
            if isinstance(competition, str) and competition
            else "the store has no competition"
        )
        raise ValueError(
            f"the submissions are from {len(names)} competitions ({', '.join(names)}), and {which}: set "
            f'`competition = "SLUG"` under `[store]` in `{CONFIG}` to show one here (`pseudorca effects '
            "--competition SLUG` prints any of them)"
        )
    fit = effects.lb_fit(state, competition=competition, **options)
    notes.append(
        f"The submissions are from several competitions: these are {competition}'s (the store's competition) "
        f"and those without one, as `pseudorca effects --competition {competition}` prints them."
    )
    return fit


def _reworded(note: str) -> str:
    """A note of the LB fit, with advice to pass an option of `pseudorca effects` given as the setting the
    report reads instead."""
    from pseudorca import effects

    if note.startswith(effects.NO_NOISE):
        return (
            f"{effects.NO_NOISE}. Resubmit a notebook, or set a known LB noise SD as `noise = SD` under "
            f"`[effects]` in `{CONFIG}`"
        )
    return note


def _message(e: Exception) -> str:
    if isinstance(e, (ArithmeticError, RecursionError)):  # not a message meant for users
        return f"cannot be computed ({type(e).__name__}: {e})"
    return str(e) or type(e).__name__


def _breakdown(lb: Fit) -> Waterfall | None:
    """The best submission from the first one in the model, as `pseudorca effects` shows by default (None
    when they are the same, or there is no model)."""
    from pseudorca import effects

    used, best = lb.used, lb.best()
    if lb._model is None or best is None or not used or best.key == used[0].key:
        return None
    return effects.waterfall(lb, used[0].key, best.key)
