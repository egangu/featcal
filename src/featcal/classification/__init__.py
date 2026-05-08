from __future__ import annotations

import importlib
from collections.abc import Callable

MODULES = {
    "sun397": "sun397",
    "stanford-cars": "stanford_cars",
    "resisc45": "resisc45",
    "eurosat": "eurosat",
    "svhn": "svhn",
    "gtsrb": "gtsrb",
    "mnist": "mnist",
    "dtd": "dtd",
}


def get_classnames_and_templates(task: str) -> tuple[list[str], list[Callable]]:
    if task not in MODULES:
        raise KeyError(f"Unknown CLIP classification task {task!r}.")
    module = importlib.import_module(f"{__name__}.{MODULES[task]}")
    return list(module.classnames), list(module.templates)

