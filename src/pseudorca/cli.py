"""Command-line entry point. Subcommands are imported only when used, so `pseudorca` starts fast."""

from __future__ import annotations

import importlib
from pathlib import Path

import click

from pseudorca import __version__
from pseudorca.commands._common import Ctx

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
