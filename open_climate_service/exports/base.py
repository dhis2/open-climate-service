"""Public contract for pure export renderers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RenderedExport:
    """Serialized payload and counts; the framework owns file creation."""

    content: bytes
    record_count: int
    skipped_count: int = 0

    def __post_init__(self) -> None:
        if type(self.content) is not bytes:
            raise TypeError("Export content must be bytes")
        for count in (self.record_count, self.skipped_count):
            if type(count) is not int or count < 0:
                raise ValueError("Export counts must be non-negative integers")


class BaseExportPlugin(ABC):
    """Render a computed result without network calls or delivery side effects.

    Plugin modules expose an instance as ``plugin``. Configuration validation must
    also be pure. Instances may be shared between threads; keep invocation state
    in local variables. Delivery will be introduced as a separate capability.
    """

    id: str
    format: str
    extension: str
    media_type: str

    @abstractmethod
    def validate_mapping(self, mapping: dict[str, Any]) -> dict[str, Any]:
        """Return a validated mapping or raise ValueError."""

    @abstractmethod
    def render(self, data: Any, mapping: dict[str, Any]) -> RenderedExport:
        """Serialize a computed result using a validated mapping."""
