"""Built-in slash commands.

Slash commands are intercepted in ``Session.send()`` before the text reaches
the LLM.  Each command is a simple function that receives a ``CommandContext``
and returns a ``CommandResult``.

The ``Session`` returns the result inside a normal ``Response`` with
``state=IDLE`` and the command output in ``.text``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from openvibe.api import Session


# ---------------------------------------------------------------------------
# Context passed to every command handler
# ---------------------------------------------------------------------------


@dataclass
class CommandContext:
    """Everything a slash command needs to inspect or mutate state."""

    session: "Session"
    args: str  # everything after the command name, stripped


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class CommandResult:
    """What a slash command returns."""

    output: str  # Rich markup to display
    quit: bool = False  # signal the app to exit
    clear: bool = False  # signal the screen to clear messages


# ---------------------------------------------------------------------------
# Command registry
# ---------------------------------------------------------------------------


@dataclass
class _CommandEntry:
    handler: Callable
    description: str
    subcommands: dict[str, tuple[Callable, str]] = field(default_factory=dict)


_COMMANDS: dict[str, _CommandEntry] = {}


def command(name: str, description: str):
    """Decorator to register a slash command."""

    def decorator(fn):
        _COMMANDS[name] = _CommandEntry(handler=fn, description=description)
        return fn

    return decorator


def subcommand(parent: str, name: str, description: str):
    """Decorator to register a subcommand under *parent*."""

    def decorator(fn):
        if parent not in _COMMANDS:
            raise ValueError(
                f"Parent command /{parent} must be registered before subcommands"
            )
        _COMMANDS[parent].subcommands[name] = (fn, description)
        return fn

    return decorator


def is_command(text: str) -> bool:
    """Return True if *text* looks like a slash command."""
    return text.startswith("/") and len(text) > 1 and not text.startswith("//")


def get_command(text: str) -> tuple[str, str] | None:
    """Parse ``/name args`` and return ``(name, args)`` or None."""
    if not is_command(text):
        return None
    parts = text[1:].split(None, 1)
    name = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""
    return name, args


def _fmt_subcommands(entry: _CommandEntry, name: str) -> str:
    """Format subcommand list for display."""
    lines = []
    for sub_name, (_, sub_desc) in sorted(entry.subcommands.items()):
        lines.append(
            f"  [bold cyan]/{name} {sub_name}[/bold cyan]  [dim]{sub_desc}[/dim]"
        )
    return "\n".join(lines)


def execute(name: str, ctx: CommandContext) -> CommandResult:
    """Execute a command by name.  Returns an error result for unknown commands."""
    from rich.markup import escape

    entry = _COMMANDS.get(name)
    if entry is None:
        known = ", ".join(f"/{n}" for n in sorted(_COMMANDS))
        return CommandResult(
            output=f"[red]Unknown command:[/red] /{escape(name)}\n"
            f"[dim]Available: {known}[/dim]",
        )

    # Check if the first arg is a subcommand.
    if entry.subcommands and ctx.args.strip():
        parts = ctx.args.strip().split(None, 1)
        sub_name = parts[0].lower()
        sub_entry = entry.subcommands.get(sub_name)
        if sub_entry is not None:
            sub_handler, _ = sub_entry
            sub_ctx = CommandContext(
                session=ctx.session, args=parts[1] if len(parts) > 1 else ""
            )
            return sub_handler(sub_ctx)
        # Unknown subcommand — let the main handler deal with it (it may
        # treat args as parameters, like /model does).

    return entry.handler(ctx)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _project_dir(ctx: CommandContext) -> Path:
    return Path(ctx.session.info.directory)


def _config(ctx: CommandContext):
    return ctx.session._config


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@command("help", "Show available commands and skills")
def cmd_help(ctx: CommandContext) -> CommandResult:
    lines = ["[bold]Available commands:[/bold]\n"]
    for name in sorted(_COMMANDS):
        entry = _COMMANDS[name]
        lines.append(
            f"  [bold cyan]/{name}[/bold cyan]  [dim]{entry.description}[/dim]"
        )
        for sub_name, (_, sub_desc) in sorted(entry.subcommands.items()):
            lines.append(f"    [bold cyan]/{name} {sub_name}[/bold cyan]  [dim]{sub_desc}[/dim]")

    # Append skills section
    try:
        from rich.markup import escape

        from openvibe.skill.registry import get_registry
        skills = get_registry().user_invocable()
        if skills:
            lines.append("\n[bold]Skills[/bold] [dim](route through the LLM):[/dim]\n")
            for skill in skills:
                aliases = (
                    f"  [dim]alias: {', '.join(f'/{a}' for a in skill.aliases)}[/dim]"
                    if skill.aliases
                    else ""
                )
                hint = f" [dim]{escape(skill.argument_hint)}[/dim]" if skill.argument_hint else ""
                lines.append(
                    f"  [bold cyan]/{escape(skill.name)}[/bold cyan]{hint}"
                    f"  [dim]{escape(skill.description)}[/dim]{aliases}"
                )
    except Exception:
        pass

    return CommandResult(output="\n".join(lines))


@command("skills", "List available skills")
def cmd_skills(ctx: CommandContext) -> CommandResult:
    """Show all user-invocable skills with metadata."""
    try:
        from openvibe.skill.registry import get_registry
    except ImportError:
        return CommandResult(output="[dim]Skills system not available.[/dim]")

    skills = get_registry().user_invocable()
    if not skills:
        return CommandResult(output="[dim]No skills registered.[/dim]")

    from rich.markup import escape

    lines = ["[bold]Available skills:[/bold]\n"]
    for skill in skills:
        lines.append(f"[bold cyan]/{escape(skill.name)}[/bold cyan]")
        if skill.aliases:
            lines[-1] += f"  [dim](aliases: {', '.join(f'/{a}' for a in skill.aliases)})[/dim]"
        lines.append(f"  [dim]{escape(skill.description)}[/dim]")
        if skill.when_to_use:
            lines.append(f"  [yellow]When to use:[/yellow] [dim]{escape(skill.when_to_use)}[/dim]")
        if skill.argument_hint:
            lines.append(
                f"  [yellow]Usage:[/yellow] [dim]/{escape(skill.name)} {escape(skill.argument_hint)}[/dim]"
            )
        if skill.tags:
            lines.append(f"  [yellow]Tags:[/yellow] [dim]{escape(', '.join(skill.tags))}[/dim]")
        lines.append("")

    return CommandResult(output="\n".join(lines))


@command("clear", "Clear conversation display")
def cmd_clear(ctx: CommandContext) -> CommandResult:
    return CommandResult(output="", clear=True)


@command("cost", "Show token usage and cost for this session")
def cmd_cost(ctx: CommandContext) -> CommandResult:
    info = ctx.session.info
    lines = [f"[bold]Session cost[/bold]  [dim]{info.id[:12]}…[/dim]\n"]

    def _fmt_tokens(n: int) -> str:
        if n >= 1_000_000:
            return f"{n / 1_000_000:.2f}M"
        if n >= 1_000:
            v = n / 1_000
            return f"{int(v)}k" if v == int(v) else f"{v:.1f}k"
        return str(n)

    lines.append(f"  Input tokens:       [bold]{_fmt_tokens(info.input_tokens)}[/bold]")
    lines.append(
        f"  Output tokens:      [bold]{_fmt_tokens(info.output_tokens)}[/bold]"
    )
    if info.cache_read_tokens:
        lines.append(
            f"  Cache read tokens:  [bold]{_fmt_tokens(info.cache_read_tokens)}[/bold]"
        )
    if info.cache_write_tokens:
        lines.append(
            f"  Cache write tokens: [bold]{_fmt_tokens(info.cache_write_tokens)}[/bold]"
        )
    total_tokens = info.input_tokens + info.output_tokens
    lines.append(f"  Total tokens:       [bold]{_fmt_tokens(total_tokens)}[/bold]")
    if info.cost:
        lines.append(f"  Cost:               [bold]${info.cost:.4f}[/bold]")
    else:
        lines.append(f"  Cost:               [dim]n/a[/dim]")
    return CommandResult(output="\n".join(lines))


@command("compact", "Summarize conversation to reduce context")
def cmd_compact(ctx: CommandContext) -> CommandResult:
    # TODO: implement actual compaction (summarize via LLM and replace history)
    return CommandResult(
        output="[dim]Compaction is not yet implemented. Coming soon.[/dim]"
    )


@command("permissions", "Show or manage permission rules")
def cmd_permissions(ctx: CommandContext) -> CommandResult:
    # No args or "list" → show rules + available subcommands.
    return _permissions_list(ctx)


@subcommand("permissions", "reset", "Clear all stored permission rules")
def cmd_permissions_reset(ctx: CommandContext) -> CommandResult:
    info = ctx.session.info
    permissions_svc = ctx.session._permissions
    if not permissions_svc:
        return CommandResult(output="[dim]No permission service available.[/dim]")

    count = permissions_svc.clear_rules(info.project_id)
    if count:
        return CommandResult(
            output=f"[green]Cleared {count} stored permission rule(s).[/green]"
        )
    return CommandResult(output="[dim]No stored rules to clear.[/dim]")


@subcommand("permissions", "list", "Show all permission rules")
def cmd_permissions_list(ctx: CommandContext) -> CommandResult:
    return _permissions_list(ctx)


def _permissions_list(ctx: CommandContext) -> CommandResult:
    info = ctx.session.info
    config = _config(ctx)

    from openvibe.agent.agent import resolve

    agent_name = ctx.session._agent_name
    agent = resolve(config, agent_name)
    agent_rules = agent.permission_rules

    lines = ["[bold]Permission rules[/bold]\n"]

    if config.permission:
        lines.append("[bold dim]Project config rules:[/bold dim]")
        for r in config.permission:
            pattern = f" [dim]({r.pattern})[/dim]" if r.pattern else ""
            lines.append(f"  {r.tool}: [bold]{r.action}[/bold]{pattern}")
    else:
        lines.append("[dim]No project config rules.[/dim]")

    if agent_rules:
        lines.append(f"\n[bold dim]Agent '{agent_name}' rules:[/bold dim]")
        for r in agent_rules:
            pattern_str = f" [dim]({r.pattern})[/dim]" if r.pattern else ""
            lines.append(f"  {r.tool}: [bold]{r.action}[/bold]{pattern_str}")

    # Load stored (allow-always) rules from DB
    permissions_svc = ctx.session._permissions
    if permissions_svc:
        stored = permissions_svc.load_rules(info.project_id)
        if stored:
            lines.append("\n[bold dim]Stored rules (allow always):[/bold dim]")
            for r in stored:
                pattern = f" [dim]({r.pattern})[/dim]" if r.pattern else ""
                lines.append(f"  {r.tool}: [bold]{r.action}[/bold]{pattern}")
        else:
            lines.append("\n[dim]No stored rules.[/dim]")

    # Show available subcommands
    entry = _COMMANDS.get("permissions")
    if entry and entry.subcommands:
        lines.append(f"\n[bold dim]Subcommands:[/bold dim]")
        lines.append(_fmt_subcommands(entry, "permissions"))

    return CommandResult(output="\n".join(lines))


@command("model", "Show or switch the active model")
def cmd_model(ctx: CommandContext) -> CommandResult:
    config = _config(ctx)
    args = ctx.args.strip()

    if not args:
        # Show current model
        model = config.model
        if model:
            lines = [
                f"[bold]Current model:[/bold] {model.provider_id}/{model.model_id}"
            ]
        else:
            lines = ["[bold]Current model:[/bold] [dim]default (set by agent)[/dim]"]

        # Show configured providers
        if config.provider:
            lines.append("\n[bold dim]Configured providers:[/bold dim]")
            for pid in sorted(config.provider):
                lines.append(f"  [dim]{pid}[/dim]")

        lines.append(
            "\n[dim]Usage: /model provider/model_id [--session|--project|--global][/dim]"
        )
        return CommandResult(output="\n".join(lines))

    # Parse scope flag from the end of the args string
    scope = "session"
    for flag in ("--session", "--global", "--project"):
        if args.endswith(flag):
            scope = flag[2:]
            args = args[: -len(flag)].strip()
            break

    if not args:
        return CommandResult(output="[red]Missing model argument.[/red]")

    # Switch model: /model provider/model_id
    if "/" in args:
        provider_id, model_id = args.split("/", 1)
    else:
        # Assume current provider or default
        provider_id = config.model.provider_id if config.model else "anthropic"
        model_id = args

    from openvibe.config import (ModelRef, save_model_to_global,
                                 save_model_to_project)

    new_model = ModelRef(provider_id=provider_id, model_id=model_id)
    model_dict = {"model": {"provider_id": provider_id, "model_id": model_id}}

    if scope == "global":
        path = save_model_to_global(new_model)
        # Also apply to current session in-memory
        ctx.session.update_session_config(model_dict)
        return CommandResult(
            output=f"[green]Model switched to:[/green] {provider_id}/{model_id}\n"
            f"[dim]Saved to {path}[/dim]"
        )

    if scope == "project":
        project_dir = _project_dir(ctx)
        path = save_model_to_project(new_model, project_dir)
        # Also apply to current session in-memory
        ctx.session.update_session_config(model_dict)
        return CommandResult(
            output=f"[green]Model switched to:[/green] {provider_id}/{model_id}\n"
            f"[dim]Saved to {path}[/dim]"
        )

    # session (default) — persists to DB so model survives session resume
    ctx.session.update_session_config(model_dict)
    return CommandResult(
        output=f"[green]Model switched to:[/green] {provider_id}/{model_id}\n"
        f"[dim]Saved to session.[/dim]"
    )


@command("screenshot", "Take a screenshot and display info about the current screen")
def cmd_screenshot(ctx: CommandContext) -> CommandResult:
    """Capture the screen and show dimensions (does not embed the image in TUI)."""
    try:
        from openvibe.computer.capture import capture_screen, screen_size
    except ImportError:
        return CommandResult(
            output="[red]Computer-use extras not installed.[/red]\n"
            "[dim]Run: pip install mss pillow[/dim]"
        )

    try:
        w, h = screen_size()
        lines = [
            "[bold]Screen info[/bold]\n",
            f"  [dim]Primary monitor:[/dim] [bold]{w}×{h}[/bold] pixels",
            "\n[dim]Use the 'screenshot' tool inside a computer-use session to "
            "capture the screen and pass the image to the model.[/dim]",
        ]
        return CommandResult(output="\n".join(lines))
    except Exception as exc:
        return CommandResult(output=f"[red]Screenshot failed:[/red] {exc}", )


@command("computer", "Show computer-use session info or manage the sandbox")
def cmd_computer(ctx: CommandContext) -> CommandResult:
    """Display audit log summary for the current session's computer-use sandbox."""
    try:
        from openvibe.computer.sandbox import get_sandbox
    except ImportError:
        return CommandResult(
            output="[red]Computer-use module not available.[/red]"
        )

    sandbox = get_sandbox(ctx.session.info.id)
    lines = [
        "[bold]Computer-use sandbox[/bold]\n",
        f"  [dim]Session:[/dim]      {sandbox.session_id[:16]}…",
        f"  [dim]Actions logged:[/dim] {len(sandbox.audit_log)}",
    ]
    if sandbox.allowed_apps:
        lines.append(f"  [dim]Allowed apps:[/dim]  {', '.join(sandbox.allowed_apps)}")
    else:
        lines.append("  [dim]Allowed apps:[/dim]  (all)")
    if sandbox.screen_region:
        x, y, w, h = sandbox.screen_region
        lines.append(f"  [dim]Screen region:[/dim] x={x} y={y} w={w} h={h}")
    else:
        lines.append("  [dim]Screen region:[/dim] (full screen)")

    if sandbox.audit_log:
        lines.append("\n[bold dim]Recent actions:[/bold dim]")
        for entry in sandbox.audit_log[-10:]:
            ts = entry.timestamp
            status = "[green]ok[/green]" if entry.error is None else "[red]err[/red]"
            lines.append(
                f"  [{ts:.0f}] {status} {entry.action_type.value}  "
                f"[dim]{(entry.result or entry.error or '')[:60]}[/dim]"
            )

    return CommandResult(output="\n".join(lines))


@subcommand("computer", "reset", "Clear the computer-use audit log for this session")
def cmd_computer_reset(ctx: CommandContext) -> CommandResult:
    try:
        from openvibe.computer.sandbox import clear_sandbox, get_sandbox
    except ImportError:
        return CommandResult(output="[red]Computer-use module not available.[/red]")

    count = len(get_sandbox(ctx.session.info.id).audit_log)
    clear_sandbox(ctx.session.info.id)
    return CommandResult(
        output=f"[green]Cleared {count} computer-use audit entries.[/green]"
    )


@command("quit", "Exit the application")
def cmd_quit(ctx: CommandContext) -> CommandResult:
    return CommandResult(output="", quit=True)


@command("exit", "Exit the application")
def cmd_exit(ctx: CommandContext) -> CommandResult:
    return CommandResult(output="", quit=True)


@command("config", "Show current configuration")
def cmd_config(ctx: CommandContext) -> CommandResult:
    config = _config(ctx)
    project = _project_dir(ctx)

    lines = ["[bold]Current configuration[/bold]\n"]

    # Model
    if config.model:
        lines.append(
            f"  [bold dim]model:[/bold dim] {config.model.provider_id}/{config.model.model_id}"
        )
    else:
        lines.append(f"  [bold dim]model:[/bold dim] [dim]default[/dim]")

    # Default agent
    lines.append(f"  [bold dim]default_agent:[/bold dim] {config.default_agent}")

    # Providers
    if config.provider:
        lines.append(
            f"  [bold dim]providers:[/bold dim] {', '.join(sorted(config.provider))}"
        )

    # Agents
    if config.agent:
        lines.append(
            f"  [bold dim]agents:[/bold dim] {', '.join(sorted(config.agent))}"
        )

    # MCP servers
    if config.mcp:
        lines.append(
            f"  [bold dim]mcp servers:[/bold dim] {', '.join(sorted(config.mcp))}"
        )

    # Instructions
    if config.instructions:
        lines.append(
            f"  [bold dim]instructions:[/bold dim] {len(config.instructions)} fragment(s)"
        )

    # Permission rules
    if config.permission:
        lines.append(
            f"  [bold dim]permission rules:[/bold dim] {len(config.permission)}"
        )

    # Config file locations
    lines.append("\n[bold dim]Config sources:[/bold dim]")
    for candidate in [
        project / "openvibe.json",
        project / "openvibe.jsonc",
        project / ".openvibe" / "openvibe.json",
        project / ".openvibe" / "openvibe.jsonc",
    ]:
        if candidate.exists():
            lines.append(f"  [green]●[/green] {candidate}")
            break
    else:
        lines.append(f"  [dim]No project config found in {project}[/dim]")

    global_cfg = Path.home() / ".config" / "openvibe" / "openvibe.json"
    if global_cfg.exists():
        lines.append(f"  [green]●[/green] {global_cfg}")
    else:
        lines.append(f"  [dim]No global config at {global_cfg}[/dim]")

    return CommandResult(output="\n".join(lines))


@command("sim", "Stress-test a process by running openvibe against a real environment")
def cmd_sim(ctx: CommandContext) -> CommandResult:
    """Stress-test a process. The harness builds a realistic environment from
    your description, runs openvibe against it, and evaluates the result.

    /sim run <process description>       — run any process (description is the goal)
    /sim run --keep <description>        — keep the temp environment for inspection
    /sim run --hard <description>        — plant adversarial traps in the environment
    /sim help                            — show difficulty levels and usage tips
    """
    args = ctx.args.strip()
    if not args:
        return CommandResult(
            output=(
                "[bold]Process Stress-Test Harness[/bold]\n\n"
                "Builds a realistic environment from your description, runs openvibe\n"
                "against it, then evaluates the result. Nothing is hardcoded.\n\n"
                "[bold]Usage:[/bold]\n"
                "  [cyan]/sim run[/cyan] <process description>\n"
                "  [cyan]/sim run --keep[/cyan] <description>   — preserve env after run\n"
                "  [cyan]/sim run --hard[/cyan] <description>   — adversarial mode\n"
                "  [cyan]/sim help[/cyan]                       — difficulty guide\n\n"
                "[bold]Examples:[/bold]\n"
                "  /sim run process purchase requisitions through the full P2P workflow\n"
                "  /sim run --keep investigate the incident logs and write a runbook\n"
                "  /sim run --hard migrate legacy CRM export to normalised schema\n"
            )
        )
    return CommandResult(output="[dim]Use /sim run or /sim help[/dim]")


@subcommand("sim", "help", "Show difficulty levels and usage guide")
def cmd_sim_help(ctx: CommandContext) -> CommandResult:
    return CommandResult(
        output=(
            "[bold]Process Stress-Test — Difficulty Guide[/bold]\n\n"
            "[green]foundational[/green]  Linear process, clear inputs/outputs.\n"
            "               ~5–10 tool calls. Use for basic workflow validation.\n\n"
            "[yellow]complex[/yellow]       Branching logic, ambiguous edges, judgment required.\n"
            "               ~10–20 tool calls. Default level for /sim run.\n\n"
            "[red]adversarial[/red]   Planted traps, conflicting data, wrong assumptions.\n"
            "               Use --hard flag to enable.\n\n"
            "[bold]Flags:[/bold]\n"
            "  --keep   Preserve the generated environment after the run\n"
            "  --hard   Enable adversarial mode (trap-planting)\n\n"
            "[bold]The harness:[/bold]\n"
            "  1. Sends your goal to the LLM → generates setup.sh\n"
            "  2. Runs setup.sh to build a realistic filesystem environment\n"
            "  3. Runs openvibe against that environment with your goal\n"
            "  4. LLM evaluates the result using the process blueprint + filesystem\n\n"
            "[dim]Nothing is hardcoded. Every environment is generated from your description.[/dim]"
        )
    )


@subcommand("sim", "run", "Run a process stress-test")
def cmd_sim_run(ctx: CommandContext) -> CommandResult:
    import asyncio
    from openvibe.sim import ProcessHarness, ProcessSpec

    args = ctx.args.strip()
    keep = False
    adversarial = False
    if "--keep" in args:
        keep = True
        args = args.replace("--keep", "").strip()
    if "--hard" in args:
        adversarial = True
        args = args.replace("--hard", "").strip()

    if not args:
        return CommandResult(output="[red]Usage: /sim run <process description>[/red]")

    difficulty = "adversarial" if adversarial else "complex"
    spec = ProcessSpec(
        name=args[:40].replace(" ", "_").lower(),
        goal=args,
        difficulty=difficulty,
    )

    progress_lines: list[str] = []

    def on_progress(msg: str) -> None:
        progress_lines.append(f"[dim]{msg}[/dim]")

    config = _config(ctx)
    harness = ProcessHarness(
        config=config,
        on_progress=on_progress,
        keep_env=keep,
    )

    try:
        run = asyncio.run(harness.run(spec))
    except Exception as exc:
        return CommandResult(
            output=f"[red]Simulation failed:[/red] {exc}\n" + "\n".join(progress_lines)
        )

    status_color = "green" if run.passed else "red"
    status = "PASS" if run.passed else "FAIL"
    lines = [
        f"[bold]{run.spec_name}[/bold]  [{run.difficulty}]",
        f"[{status_color}]{status}[/{status_color}]  "
        f"score: [bold]{run.score:.2f}[/bold]  |  "
        f"tool calls: {run.tool_calls}  |  "
        f"branches: {run.dag_branches}  |  "
        f"{run.elapsed_seconds:.1f}s",
        "",
    ]
    if run.evaluation:
        lines.append(run.evaluation)
        lines.append("")
    if run.strengths:
        lines.append("[bold dim]What worked:[/bold dim]")
        for s in run.strengths:
            lines.append(f"  [green]✓[/green] {s}")
        lines.append("")
    if run.gaps:
        lines.append("[bold dim]Gaps:[/bold dim]")
        for g in run.gaps:
            lines.append(f"  [yellow]✗[/yellow] {g}")
        lines.append("")
    if run.env_dir:
        lines.append(f"[dim]Environment preserved at: {run.env_dir}[/dim]")
    if run.run_error:
        lines.append(f"[red]Run error:[/red] {run.run_error}")

    return CommandResult(output="\n".join(lines))


@command("blueprint", "View or export the process blueprint for this session")
def cmd_blueprint(ctx: CommandContext) -> CommandResult:
    """Show the current process blueprint summary.

    /blueprint          — show step summary
    /blueprint export   — save process-blueprint.json to the project directory
    /blueprint undo     — mark the last step undone (starts a new branch)
    """
    from openvibe.session.blueprint import get_blueprint

    bp = get_blueprint(ctx.session.info.id)
    if bp is None or not bp.nodes:
        return CommandResult(
            output="[dim]No steps recorded yet. The blueprint builds as the process runs.[/dim]"
        )

    lines = [
        f"[bold]Process blueprint[/bold]  [dim]goal: {bp.goal[:80]}[/dim]\n"
    ]
    branch_colors = ["cyan", "yellow", "magenta", "blue", "green"]
    for n in bp.nodes.values():
        status_icon = {
            "completed": "[green]●[/green]",
            "running": "[yellow]◎[/yellow]",
            "failed": "[red]✗[/red]",
            "undone": "[dim]○[/dim]",
        }.get(n.status, "?")
        color = branch_colors[n.branch % len(branch_colors)]
        branch_tag = f" [dim](branch {n.branch})[/dim]" if n.branch else ""
        lines.append(
            f"  {status_icon} [{color}]{n.action}[/{color}]{branch_tag}"
        )
        if n.purpose:
            lines.append(f"     [dim]↳ {n.purpose}[/dim]")

    completed = sum(1 for n in bp.nodes.values() if n.status == "completed")
    total = len(bp.nodes)
    lines.append(f"\n[dim]{completed}/{total} steps completed[/dim]")
    lines.append(
        "\n[dim]Use [bold]/blueprint export[/bold] to save a resumable file, "
        "or [bold]/blueprint undo[/bold] to branch from the last step.[/dim]"
    )
    return CommandResult(output="\n".join(lines))


@subcommand("blueprint", "export", "Export process blueprint to project-blueprint.json")
def cmd_blueprint_export(ctx: CommandContext) -> CommandResult:
    from openvibe.session.blueprint import get_blueprint

    bp = get_blueprint(ctx.session.info.id)
    if bp is None or not bp.nodes:
        return CommandResult(output="[dim]No steps to export yet.[/dim]")

    project = _project_dir(ctx)
    path = bp.save(project)
    return CommandResult(
        output=(
            f"[green]Process blueprint saved:[/green] {path}\n\n"
            "[dim]To resume this process in a new session, start openvibe and paste "
            "the contents of that file as your first message. The agent will read "
            "the completed steps and pick up from where it left off.[/dim]"
        )
    )


@subcommand("blueprint", "undo", "Mark the last completed step as undone")
def cmd_blueprint_undo(ctx: CommandContext) -> CommandResult:
    from openvibe.session.blueprint import get_blueprint

    bp = get_blueprint(ctx.session.info.id)
    if bp is None or not bp.current_node_id:
        return CommandResult(output="[dim]No step to undo.[/dim]")

    node = bp.nodes.get(bp.current_node_id)
    if node is None:
        return CommandResult(output="[dim]No step to undo.[/dim]")

    bp.undo_node(node.id)
    return CommandResult(
        output=(
            f"[yellow]Marked as undone:[/yellow] {node.action}\n"
            f"[dim]The process is now on branch {bp.current_branch}. "
            "Tell the agent to try a different approach.[/dim]"
        )
    )


@command("init", "Create or edit project openvibe.json")
def cmd_init(ctx: CommandContext) -> CommandResult:
    project = _project_dir(ctx)

    # Check existing config
    for candidate in [
        project / "openvibe.json",
        project / "openvibe.jsonc",
        project / ".openvibe" / "openvibe.json",
        project / ".openvibe" / "openvibe.jsonc",
    ]:
        if candidate.exists():
            content = candidate.read_text(encoding="utf-8")
            return CommandResult(
                output=f"[bold]Project config exists:[/bold] {candidate}\n\n"
                f"[dim]{content}[/dim]\n\n"
                f"[dim]Edit this file directly to change settings.[/dim]"
            )

    # Create a minimal config
    config_path = project / "openvibe.json"
    template = {
        "model": {"provider_id": "anthropic", "model_id": "claude-sonnet-4-5"},
        "permission": [
            {"tool": "bash", "action": "ask"},
            {"tool": "file.*", "action": "allow"},
        ],
    }
    config_path.write_text(json.dumps(template, indent=2) + "\n", encoding="utf-8")
    return CommandResult(
        output=f"[green]Created:[/green] {config_path}\n\n"
        f"[dim]{json.dumps(template, indent=2)}[/dim]\n\n"
        f"[dim]Edit this file to customize your project settings.[/dim]"
    )


# ---------------------------------------------------------------------------
# /learn — record and replay screen interactions
# ---------------------------------------------------------------------------



@command("learn", "Record and replay screen interactions")
def cmd_learn(ctx: CommandContext) -> CommandResult:
    """Record human screen interactions and replay them exactly.

    /learn start "task description"  — start recording
    /learn stop                      — stop and save to ./openvibe_recordings/
    /learn list                      — list saved recordings
    /learn replay <name>             — replay (click mouse to interrupt)
    /learn delete <name>             — delete a recording
    """
    from openvibe.computer.recorder import get_recorder, recordings_dir
    recorder = get_recorder()
    status = (
        "[yellow]● Recording in progress[/yellow]" if recorder.is_recording
        else "[dim]○ Not recording[/dim]"
    )
    d = recordings_dir()
    return CommandResult(
        output=(
            f"[bold]learn[/bold]  {status}\n\n"
            "  [cyan]/learn start[/cyan] [italic]\"task\"[/italic]  — begin recording\n"
            "  [cyan]/learn stop[/cyan]              — finish and save\n"
            "  [cyan]/learn list[/cyan]              — show saved recordings\n"
            "  [cyan]/learn replay[/cyan] [italic]<name>[/italic]  — replay exactly (click to cancel)\n"
            "  [cyan]/learn delete[/cyan] [italic]<name>[/italic]  — remove recording\n\n"
            f"[dim]Saved to: {d}[/dim]"
        )
    )




@subcommand("learn", "start", "Start recording screen interactions")
def cmd_learn_start(ctx: CommandContext) -> CommandResult:
    from openvibe.computer.recorder import get_recorder, name_from_prompt

    recorder = get_recorder()
    if recorder.is_recording:
        return CommandResult(
            output="[yellow]Already recording.[/yellow] Use [cyan]/learn stop[/cyan] to finish."
        )

    prompt = ctx.args.strip().strip('"').strip("'")
    if not prompt:
        return CommandResult(
            output="[red]Usage:[/red] /learn start [italic]\"task description\"[/italic]"
        )

    name = name_from_prompt(prompt)
    try:
        recorder.start(prompt, name)
    except Exception as exc:
        return CommandResult(output=f"[red]Error:[/red] {exc}")

    return CommandResult(
        output=(
            f"[green]● Recording started[/green]  [bold]{name}[/bold]\n"
            f"  Task: {prompt}\n\n"
            "Perform your task now, then run [cyan]/learn stop[/cyan]."
        )
    )


@subcommand("learn", "stop", "Stop recording and save")
def cmd_learn_stop(ctx: CommandContext) -> CommandResult:
    from openvibe.computer.recorder import get_recorder

    recorder = get_recorder()
    if not recorder.is_recording:
        return CommandResult(
            output="[yellow]Not currently recording.[/yellow] Use [cyan]/learn start[/cyan] first."
        )

    try:
        recording = recorder.stop()
    except Exception as exc:
        return CommandResult(output=f"[red]Error stopping:[/red] {exc}")

    path = recording.save()
    return CommandResult(
        output=(
            f"[green]Saved:[/green] [bold]{recording.name}[/bold]\n"
            f"  Events:   {len(recording.events)}\n"
            f"  Duration: {recording.duration:.1f}s\n"
            f"  File:     {path}\n\n"
            f"Replay: [cyan]/learn replay[/cyan] [italic]{recording.name}[/italic]"
        )
    )


@subcommand("learn", "list", "List saved recordings")
def cmd_learn_list(ctx: CommandContext) -> CommandResult:
    from openvibe.computer.recorder import Recording, recordings_dir

    recs = Recording.list_all()
    if not recs:
        return CommandResult(
            output=(
                "[dim]No recordings yet.[/dim]\n"
                "Start one: [cyan]/learn start[/cyan] [italic]\"your task\"[/italic]\n\n"
                f"[dim]Storage: {recordings_dir()}[/dim]"
            )
        )

    lines = [f"[bold]Recordings[/bold] ({len(recs)})  [dim]{recordings_dir()}[/dim]\n"]
    for r in recs:
        lines.append(f"  [bold cyan]{r.name}[/bold cyan]")
        lines.append(f"    [italic]{r.prompt}[/italic]")
        lines.append(f"    [dim]{r.recorded_at[:19]}  {r.duration:.1f}s[/dim]")
        lines.append("")
    lines.append("[dim]Replay: /learn replay <name>[/dim]")
    return CommandResult(output="\n".join(lines))


@subcommand("learn", "replay", "Replay a saved recording")
def cmd_learn_replay(ctx: CommandContext) -> CommandResult:
    from openvibe.computer.recorder import Recording

    query = ctx.args.strip().strip('"').strip("'")
    if not query:
        return CommandResult(
            output="[red]Usage:[/red] /learn replay [italic]<name>[/italic]"
        )

    recording = Recording.load(query)
    if recording is None:
        return CommandResult(
            output=f"[red]Not found:[/red] {query!r}\n"
                   "Run [cyan]/learn list[/cyan] to see saved recordings."
        )
    if not recording.events:
        return CommandResult(
            output="[yellow]Recording has no events.[/yellow] Re-record with [cyan]/learn start[/cyan]."
        )

    try:
        outcome = recording.replay_with_vision()
    except Exception as exc:
        return CommandResult(output=f"[red]Replay error:[/red] {exc}")

    clicks = sum(1 for e in recording.events if e.type == "click" and e.data.get("pressed"))
    color = "green" if outcome == "done" else "yellow"
    note = "" if outcome == "done" else "  [dim](stopped — human input detected)[/dim]"
    return CommandResult(
        output=(
            f"[{color}]{outcome.capitalize()}.[/{color}] "
            f"[bold]{recording.name}[/bold]  "
            f"[dim]({len(recording.events)} events, {clicks} clicks, {recording.duration:.1f}s)[/dim]"
            f"{note}"
        )
    )


@subcommand("learn", "delete", "Delete a saved recording")
def cmd_learn_delete(ctx: CommandContext) -> CommandResult:
    from openvibe.computer.recorder import recordings_dir

    name = ctx.args.strip().strip('"').strip("'")
    if not name:
        return CommandResult(
            output="[red]Usage:[/red] /learn delete [italic]<name>[/italic]"
        )

    d = recordings_dir()
    for path in [d / name, d / f"{name}.json"]:
        if path.exists():
            path.unlink()
            return CommandResult(output=f"[green]Deleted:[/green] {name}")

    return CommandResult(
        output=f"[red]Not found:[/red] {name!r}  (run [cyan]/learn list[/cyan])"
    )
