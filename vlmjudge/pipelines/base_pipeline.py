"""
Pipeline interfaces.

In Phase 1, pipelines are only structured; no full orchestration logic is implemented yet.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BasePipeline(ABC):
    """Base interface for a pipeline that runs on arbitrary inputs."""

    @abstractmethod
    def run(self, inputs: Any):
        """Run the pipeline. Logic intentionally not implemented in Phase 1."""
        raise NotImplementedError

