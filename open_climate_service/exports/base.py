"""Public contract for export renderers and optional delivery."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Protocol

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


class DeliveryContext(Protocol):
    """Execution hooks the framework hands a delivery plugin during ``send``.

    The protocol mirrors the native job's ``JobExecutionContext`` without coupling
    external plugins to job internals: progress reporting, cooperative
    cancellation, and durable per-chunk checkpoints. Plugins must remain usable
    with a no-op implementation so their delivery code works standalone.
    """

    def report_progress(self, done: int | None = None, total: int | None = None, message: str | None = None) -> None:
        """Persist coarse progress (e.g. chunks sent) for one delivery job."""
        ...

    def is_cancel_requested(self) -> bool:
        """Return True when the delivery job has been asked to stop."""
        ...

    def save_checkpoint(self, key: str, state: dict[str, Any]) -> None:
        """Persist one resumable chunk checkpoint under a plugin-chosen key."""
        ...

    def load_checkpoint(self, key: str) -> dict[str, Any] | None:
        """Return a previously saved chunk checkpoint, or None."""
        ...


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

    def send(
        self,
        payload: bytes,
        target: Any,
        *,
        dry_run: bool = False,
        context: DeliveryContext | None = None,
    ) -> ExportReport:
        """Deliver a rendered payload and report what the far end accepted.

        Delivery plugins override this and set ``supports_delivery = True``. The
        ``context`` carries progress, cancellation, and checkpoint hooks; a plugin
        must tolerate ``context=None`` so it remains callable standalone. The
        default raises so a render-only plugin fails loudly instead of being
        silently treated as a no-op delivery.
        """
        raise NotImplementedError(f"Export plugin '{self.id}' does not support delivery")
