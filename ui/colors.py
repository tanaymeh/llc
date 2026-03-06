RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
ITALIC = "\033[3m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
GRAY = "\033[90m"
WHITE = "\033[97m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"


def style(text: str, *codes: str) -> str:
    return "".join(codes) + text + RESET
