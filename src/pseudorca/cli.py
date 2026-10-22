"""Command-line entry point. Subcommands are imported only when used, so `pseudorca` starts fast."""

from __future__ import annotations

import contextlib
import importlib
import os
import sys
from pathlib import Path

import click

from pseudorca import __version__
from pseudorca.commands._common import Ctx, show_warnings
from pseudorca.store import StoreError

# name -> "module:attribute", in the order `pseudorca --help` lists them
COMMANDS: dict[str, str] = {
    "init": "pseudorca.commands.ledger:init_cmd",
    "idea": "pseudorca.commands.ledger:idea_group",
    "nb": "pseudorca.commands.ledger:nb_group",
    "status": "pseudorca.commands.ledger:status_cmd",
    "audit": "pseudorca.commands.audit:audit_cmd",
    "check": "pseudorca.commands.audit:check_cmd",
    "verdict": "pseudorca.commands.audit:verdict_cmd",
    "sub": "pseudorca.commands.ledger:sub_group",
    "kaggle": "pseudorca.commands.kaggle_cmd:kaggle_group",
    "run": "pseudorca.commands.results:run_group",
    "compare": "pseudorca.commands.results:compare_cmd",
    "lb": "pseudorca.commands.results:lb_cmd",
    "report": "pseudorca.commands.report:report_cmd",
    "log": "pseudorca.commands.ledger:log_cmd",
}


class LazyGroup(click.Group):
    def list_commands(self, ctx: click.Context) -> list[str]:
        return list(COMMANDS)

    def invoke(self, ctx: click.Context) -> object:
        """Run the command, then print ledger warnings it has not shown. Expected failures become one-line
        errors (set PSEUDORCA_DEBUG=1 for tracebacks)."""
        if os.environ.get("PSEUDORCA_DEBUG"):
            return self._run(ctx)
        try:
            return self._run(ctx)
        except BrokenPipeError:  # output piped into `head` and closed early: exit quietly, no flush error
            with contextlib.suppress(OSError, ValueError):
                fd = os.open(os.devnull, os.O_WRONLY)
                try:
                    for stream in (sys.stdout, sys.stderr):  # stderr too, for `2>&1 | head`
                        os.dup2(fd, stream.fileno())
                finally:
                    os.close(fd)
            raise SystemExit(0) from None
        except KeyError:
            raise
        except (StoreError, LookupError, ValueError, OSError) as e:
            raise click.ClickException(str(e) or type(e).__name__) from e

    def _run(self, ctx: click.Context) -> object:
        try:
            return super().invoke(ctx)
        finally:
            if sys.exc_info()[0] in (None, click.exceptions.Exit, SystemExit):
                obj = ctx.find_object(Ctx)
                if obj is not None and obj._store is not None:
                    with contextlib.suppress(Exception):  # never let a warning hide the command's result
                        show_warnings(obj._store)

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        target = COMMANDS.get(cmd_name)
        if target is None:
            return None
        module_name, attr = target.split(":")
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as e:
            if e.name == module_name:  # that command is not part of this build
                return None
            raise
        return getattr(module, attr, None)


@click.group(
    cls=LazyGroup, context_settings={"help_option_names": ["-h", "--help"], "max_content_width": 100}
)
@click.version_option(__version__, "-V", "--version", prog_name="pseudorca")
@click.option(
    "--store",
    "store_path",
    type=click.Path(file_okay=False, path_type=Path),
    envvar="PSEUDORCA_STORE",
    help="Store to use (default: the nearest .pseudorca above the current directory).",
)
@click.option(
    "--as",
    "actor",
    metavar="NAME",
    envvar="PSEUDORCA_ACTOR",
    help="Author of new events (default: git user.name).",
)
@click.pass_context
def main(ctx: click.Context, store_path: Path | None, actor: str | None) -> None:
    """Pseudorca: which improvements are in which notebook, and does a change beat the noise?

    The ledger lives in .pseudorca/ inside your team's private git repository; commit and push it like code.
    """
    ctx.obj = Ctx(store_path=store_path, actor=actor)
