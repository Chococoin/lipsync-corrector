"""Track A: Face-swap quick win. See docs/superpowers/specs/2026-04-11-lipsync-corrector-design.md."""

from __future__ import annotations


def select_providers(available: list[str]) -> list[str]:
    preferred = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    result = [p for p in preferred if p in available]
    if "CPUExecutionProvider" not in result:
        result.append("CPUExecutionProvider")
    return result
