"""Public contract for export renderers and optional delivery."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from open_climate_service.exports.report import ExportReport


@dataclass(frozen=True)
class RenderedExport:
    """Serialized payload and counts; the framework owns file creation."""

    content: bytes
    record_count: int
    skipped_count: int = 0
    periods: tuple[str, ...] | None = None

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
    # Renderer authors bump this when changing payload semantics. An unversioned
    # renderer may produce files but cannot supply a verified delivery input.
    version: str | None = None
    # Render-only plugins leave this False; the delivery endpoint refuses them.
    supports_delivery: bool = False

    @abstractmethod
    def validate_mapping(self, mapping: dict[str, Any]) -> dict[str, Any]:
        """Return a validated mapping or raise ValueError."""

    @abstractmethod
    def render(self, data: Any, mapping: dict[str, Any]) -> RenderedExport:
        """Serialize a computed result using a validated mapping."""

    def send(self, payload: bytes, target: Any, *, dry_run: bool = False) -> ExportReport:
        """Deliver a rendered payload and report what the far end accepted.

        Delivery plugins override this and set ``supports_delivery = True``. The
        default raises so a render-only plugin fails loudly instead of being
        silently treated as a no-op delivery.
        """
        raise NotImplementedError(f"Export plugin '{self.id}' does not support delivery")
