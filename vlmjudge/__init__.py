# Name: Jayesh Pandey
# Summary: vlmjudge

"""
vlmjudge

Phase 1 modular architecture for building modern image scoring + comparison pipelines
on top of the existing `ImageReward` repository.

Why this name?
    On Windows (default case-insensitive filesystems), a folder named `imagereward/`
    conflicts with the existing `ImageReward/` package. To keep `import ImageReward`
    fully backward compatible, the new modular package lives under `vlmjudge/`.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"

