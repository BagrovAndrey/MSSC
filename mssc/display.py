from __future__ import annotations

DISPLAY_NAMES = {
    "C": "Cdetail",
    "Jglob": "Jglobal",
    "JlocQ": "Jnested",
}

LEGACY_NAMES = {
    "O": "O",
    "Odiv": "Odiv",
    "Jloc": "Jloc",
    "Q": "Q",
    "D": "D",
}


def display_name(metric: str) -> str:
    return DISPLAY_NAMES.get(metric, LEGACY_NAMES.get(metric, metric))


def phase_name(metric: str) -> str:
    if metric == "JlocQ":
        return "Jphase"
    return f"{display_name(metric)} phase excess"
