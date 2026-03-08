from __future__ import annotations

from collections.abc import Callable, Iterable

from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document

from models import AvailableModel


class ModelCompleter(Completer):
    def __init__(self, get_models: Callable[[], Iterable[AvailableModel]]) -> None:
        self._get_models = get_models

    def get_completions(
        self,
        document: Document,
        complete_event: object,
    ) -> Iterable[Completion]:
        text = document.text_before_cursor
        if not text.startswith("/model "):
            return

        query = text[len("/model ") :]
        start_position = -len(query)

        for model in self._get_models():
            yield Completion(
                text=model.id,
                start_position=start_position,
                display=model.id,
                display_meta=model.name if model.name != model.id else "",
            )
