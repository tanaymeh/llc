from dataclasses import dataclass


@dataclass(frozen=True)
class TurnUsage:
    input_tokens: int
    output_tokens: int


class TokenTracker:
    def __init__(self) -> None:
        self._turn_input = 0
        self._turn_output = 0
        self._session_input = 0
        self._session_output = 0

    def add(self, input_tokens: int, output_tokens: int) -> None:
        self._turn_input += input_tokens
        self._turn_output += output_tokens

    def finish_turn(self) -> TurnUsage:
        turn = TurnUsage(
            input_tokens=self._turn_input,
            output_tokens=self._turn_output,
        )
        self._session_input += self._turn_input
        self._session_output += self._turn_output
        self._turn_input = 0
        self._turn_output = 0
        return turn

    @property
    def session_input_tokens(self) -> int:
        return self._session_input

    @property
    def session_output_tokens(self) -> int:
        return self._session_output

    @staticmethod
    def compute_cost(
        turn: TurnUsage,
        prompt_price: float,
        completion_price: float,
    ) -> float:
        return turn.input_tokens * prompt_price + turn.output_tokens * completion_price
