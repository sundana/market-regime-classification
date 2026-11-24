"""Human-readable names, colours, and descriptions for volatility-ordered states.

Every detector in this project sorts states by mean realised volatility, so
state 0 is always the calmest and state ``n_states - 1`` the most volatile.
These helpers turn those integer indices into something a junior trader can
read without knowing the model internals.
"""

from __future__ import annotations

_COLOR_HEX = {
    "green": "#1a7f37",
    "amber": "#bf8700",
    "red": "#cf222e",
}


def state_names(n_states: int) -> list[str]:
    """Return low-to-high volatility labels for ``n_states`` regimes."""
    if n_states < 1:
        raise ValueError("n_states must be >= 1")
    if n_states == 1:
        return ["NORMAL"]
    if n_states == 2:
        return ["CALM", "STRESSED"]
    if n_states == 3:
        return ["CALM", "NORMAL", "STRESSED"]
    if n_states == 4:
        return ["CALM", "NORMAL", "ELEVATED", "STRESSED"]
    return ["CALM"] + [f"VOL BAND {i}" for i in range(2, n_states)] + ["STRESSED"]


def state_color_names(n_states: int) -> list[str]:
    """Return ``green`` / ``amber`` / ``red`` traffic-light names per state."""
    if n_states < 1:
        raise ValueError("n_states must be >= 1")
    if n_states == 1:
        return ["amber"]
    if n_states == 2:
        return ["green", "red"]
    return ["green"] + ["amber"] * (n_states - 2) + ["red"]


def state_colors(n_states: int) -> list[str]:
    """Return hex colours per state."""
    return [_COLOR_HEX[name] for name in state_color_names(n_states)]


def state_description(index: int, n_states: int) -> str:
    """Return a one-line, non-directional description of a volatility state."""
    names = state_names(n_states)
    name = names[index] if 0 <= index < n_states else f"STATE {index}"
    descriptions = {
        "CALM": "low realised volatility; trends are usually orderly",
        "NORMAL": "normal volatility; no strong risk signal",
        "ELEVATED": "above-normal volatility; risk is rising",
        "STRESSED": "high realised volatility; risk is elevated, not a direction call",
    }
    if name in descriptions:
        return descriptions[name]
    return f"volatility band {index} of {n_states}, ordered calm to stressed"


__all__ = [
    "state_color_names",
    "state_colors",
    "state_description",
    "state_names",
]
