from __future__ import annotations

import logging


_DEFAULT_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"


def configure_logging(*, level: int = logging.INFO) -> None:
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=level, format=_DEFAULT_FORMAT)
        return
    if root.level > level:
        root.setLevel(level)
