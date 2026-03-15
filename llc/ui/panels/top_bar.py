from __future__ import annotations

from pathlib import Path

from rich.console import Group
from rich.text import Text
from textual.widgets import Static

from llc.ui.state import MissionControlState


class TopBarPanel(Static):
    def render_state(self, state: MissionControlState) -> None:
        top = state.top_bar
        repo_name = Path.cwd().name

        line1 = Text()
        line1.append("[", style="#667270")
        line1.append("LLC", style="bold #79F2C0")
        line1.append("] ", style="#667270")
        line1.append("MODEL ", style="#93A19E")
        line1.append(top.model_name, style="bold #66DED3")
        line1.append("  |  ", style="#667270")
        line1.append("REPO ", style="#93A19E")
        line1.append(repo_name, style="bold #E3ECE9")

        phase_style = "#79F2C0"
        if top.phase.value == "verify":
            phase_style = "#F1C27B"
        line2 = Text()
        line2.append(">", style="#79F2C0")
        line2.append(" PHASE ", style="#93A19E")
        line2.append(top.phase.value.upper(), style=f"bold {phase_style}")
        line2.append("  |  ", style="#667270")
        line2.append("TOK ", style="#93A19E")
        line2.append(top.tokens_label, style="bold #E3ECE9")
        line2.append("  |  ", style="#667270")
        line2.append("COST ", style="#93A19E")
        line2.append(f"${top.session_cost:.4f}", style="bold #66DED3")
        line2.append("  |  ", style="#667270")
        line2.append("SUB ", style="#93A19E")
        line2.append(
            f"{top.active_subagents}/{max(top.total_workers, top.active_subagents)}",
            style="bold #E3ECE9",
        )
        line2.append("  |  ", style="#667270")
        line2.append("UP ", style="#93A19E")
        line2.append(top.elapsed_label, style="bold #E3ECE9")

        self.update(Group(line1, line2))
