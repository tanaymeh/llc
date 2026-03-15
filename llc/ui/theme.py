from __future__ import annotations

from pydantic import BaseModel, ConfigDict

try:
    from textual.app import App
    from textual.theme import Theme
except Exception:  # pragma: no cover
    App = object  # type: ignore[assignment]
    Theme = None  # type: ignore[assignment]


class MissionPalette(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    dark: bool
    primary: str
    secondary: str
    accent: str
    background: str
    surface: str
    panel: str
    foreground: str
    success: str
    warning: str
    error: str
    variables: dict[str, str]


PHOSPHOR_MINT = MissionPalette(
    name="mission-phosphor-mint",
    dark=True,
    primary="#79F2C0",
    secondary="#66DED3",
    accent="#B8FFE8",
    background="#0A0D0E",
    surface="#101617",
    panel="#101617",
    foreground="#E3ECE9",
    success="#79F2C0",
    warning="#F1C27B",
    error="#F090A6",
    variables={
        "divider": "#253133",
        "text_secondary": "#93A19E",
        "text_dim": "#667270",
        "info": "#66DED3",
    },
)

GRAPHITE_AQUA = MissionPalette(
    name="mission-graphite-aqua",
    dark=True,
    primary="#67E5D8",
    secondary="#7DEFD4",
    accent="#C8FFF3",
    background="#0B1012",
    surface="#12191C",
    panel="#12191C",
    foreground="#DCE8E7",
    success="#7DEFD4",
    warning="#E6BE74",
    error="#E88EA8",
    variables={
        "divider": "#26343A",
        "text_secondary": "#869695",
        "text_dim": "#61706F",
        "info": "#67E5D8",
    },
)

PAPER_TERMINAL = MissionPalette(
    name="mission-paper-terminal",
    dark=False,
    primary="#14967D",
    secondary="#1D8D97",
    accent="#2FB79C",
    background="#F3F7F5",
    surface="#E7EFEC",
    panel="#E7EFEC",
    foreground="#14211E",
    success="#14967D",
    warning="#B67A22",
    error="#BC596F",
    variables={
        "divider": "#C9D5D1",
        "text_secondary": "#5E716D",
        "text_dim": "#7A8885",
        "info": "#1D8D97",
    },
)


def _to_textual_theme(palette: MissionPalette) -> Theme | None:
    if Theme is None:
        return None
    try:
        return Theme(
            name=palette.name,
            dark=palette.dark,
            primary=palette.primary,
            secondary=palette.secondary,
            accent=palette.accent,
            background=palette.background,
            surface=palette.surface,
            panel=palette.panel,
            foreground=palette.foreground,
            success=palette.success,
            warning=palette.warning,
            error=palette.error,
            variables=palette.variables,
        )
    except Exception:
        return None


def register_mission_control_themes(app: App) -> None:
    theme_names: list[str] = []
    for palette in (PHOSPHOR_MINT, GRAPHITE_AQUA, PAPER_TERMINAL):
        theme = _to_textual_theme(palette)
        if theme is None:
            continue
        try:
            app.register_theme(theme)
            theme_names.append(palette.name)
        except Exception:
            continue
    if theme_names:
        app.theme = PHOSPHOR_MINT.name
