"""Public contract for export renderers and optional delivery."""

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from open_climate_service.exports.report import ExportReport


@dataclass(frozen=True)
class PipelineDataset:
    """A managed dataset a pipeline's workflow reads, as a destination sees it."""

    id: str
    variable: str | None
    cadence: str | None


@dataclass(frozen=True)
class DestinationContext:
    """What a pipeline binds, handed to its destination plugin.

    The pipeline core derives this from the workflow arguments without knowing the
    destination: the datasets bound to ``collection-id`` parameters and the feature
    collections bound with ``from_features``. ``export_id`` is the named export the
    pipeline's results are saved and delivered through.
    """

    export_id: str
    datasets: tuple[PipelineDataset, ...]
    feature_collections: tuple[str, ...]


@dataclass(frozen=True)
class DestinationCheck:
    """One destination-side validation step and its verdict."""

    id: str
    status: Literal["pass", "fail", "skip"]
    message: str


@dataclass(frozen=True)
class SettingsFormField:
    """Presentation hints for one schema property, or a column below an array property.

    ``name`` is a property name such as ``connection`` or a dotted row-column path such
    as ``series.data_element``. The JSON Schema still decides the field's type, whether it
    is required, and its allowed values. Unknown or incompatible widget hints are ignored.
    """

    name: str
    label: str | None = None
    help_text: str | None = None
    widget: str | None = None
    option_labels: dict[str, str] | None = None
    lookup: str | None = None


@dataclass(frozen=True)
class LookupOption:
    """One choice a destination offers for a lookup field: the stored value and how it reads."""

    value: str
    label: str
    detail: str = ""


@dataclass(frozen=True)
class SettingsFormSection:
    """A titled, ordered group of top-level plugin settings."""

    title: str
    fields: tuple[str, ...]
    description: str = ""


@dataclass(frozen=True)
class PipelineSettingsForm:
    """Plugin-owned presentation of its settings in the pipeline form.

    Presentation is deliberately data, not HTML. Properties omitted from ``sections`` are
    appended safely, so adding a schema property cannot make it impossible to configure.

    A string field with ``widget="lookup"`` and a ``lookup`` name is chosen from the
    destination: the form searches it through ``BaseExportPlugin.lookup_options`` and shows
    stored values by the labels ``BaseExportPlugin.lookup_labels`` resolves.
    """

    title: str
    description: str = ""
    sections: tuple[SettingsFormSection, ...] = ()
    fields: tuple[SettingsFormField, ...] = ()


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
    # True when ``send(dry_run=True)`` has the far end validate without storing anything.
    # A pipeline whose destination lacks it dry-runs by rendering only.
    supports_dry_run: bool = False
    # True when sending the same payload twice leaves the destination as sending it once
    # (an upsert). Only then may an operator send a delivery again after an unconfirmed outcome.
    redelivery_safe: bool = False

    # --- delivery target hooks (delivery plugins) -----------------------------------------

    def target_binding(self, references: dict[str, str]) -> dict[str, str] | None:
        """Credential-free identity of the target an export's references name, or None.

        Recorded in the saved export's manifest and compared again before delivery, so a
        target that changed in between is refused. Must not contain secrets or URLs with
        credentials; a digest of an address is the usual identity.

        The default keeps the contract delivery plugins have always had: a ``connection``
        reference names one of the instance's named connections, bound by id and URL digest.
        Override both target hooks for any other kind of target.
        """
        from open_climate_service.exports.manifest import named_connection_binding

        return named_connection_binding(references)

    def delivery_target(self, binding: dict[str, str]) -> Any:
        """What ``send`` receives for a bound target. Default: the named connection's id."""
        return binding["connection_id"]

    # --- pipeline destination hooks (optional) ------------------------------------------
    # A plugin becomes a pipeline destination by returning a settings schema. The core
    # never interprets settings; it hands them back to these hooks.

    def destination_settings_schema(self) -> dict[str, Any] | None:
        """JSON Schema of the pipeline destination settings, or None when not a destination.

        Read when rendering the pipeline form, so it may list instance choices such as
        configured connections. It must be pure.
        """
        return None

    def pipeline_settings_form(self) -> PipelineSettingsForm | None:
        """Presentation metadata for this plugin's pipeline settings, or schema defaults.

        This hook may derive labels and choices from instance configuration but must not
        contact the destination. JSON Schema remains the data and validation contract.
        """
        return None

    def lookup_options(self, lookup: str, settings: dict[str, Any], query: str) -> list[LookupOption]:
        """Search the destination for the choices of a lookup field, as the form types.

        ``lookup`` is the name a ``SettingsFormField`` declared; ``settings`` are the form's
        current values, incomplete and unvalidated, so the plugin reads from them only what the
        search needs, such as the connection. May contact the destination. Raise ValueError
        naming what is missing or what the destination answered.
        """
        raise NotImplementedError(f"Export plugin '{self.id}' has no lookup '{lookup}'")

    def lookup_labels(self, lookup: str, settings: dict[str, Any], values: Sequence[str]) -> dict[str, str]:
        """How stored lookup values read, by value; a value the destination does not know is left out.

        Called when a pipeline is validated, so the labels are kept with the validation result
        and pages never wait for the destination. May contact the destination; raise on failure.
        """
        return {}

    def export_definition(self, settings: dict[str, Any], context: DestinationContext) -> dict[str, Any]:
        """Compile destination settings into this plugin's export definition, without ``id`` or ``plugin``.

        The result is resolved like a configured named export, so it holds the mapping plus
        any ``dataset``/``connection`` references. Raise ValueError naming what is wrong.
        Must be pure: no network calls.
        """
        raise NotImplementedError(f"Export plugin '{self.id}' is not a pipeline destination")

    def check_destination(self, settings: dict[str, Any], context: DestinationContext) -> list[DestinationCheck]:
        """Destination-side checks run on save, on re-check and before going live.

        Unlike the other hooks this may contact the destination, and must turn its
        failures into ``fail`` checks rather than raise.
        """
        return []

    def preview(
        self,
        content: bytes,
        settings: dict[str, Any],
        context: DestinationContext,
        labels: Mapping[str, Mapping[str, str]] | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """A few rendered records for a dry-run summary; empty when the payload has no row shape.

        A cell is a scalar, or ``{"value": ..., "label": ...}`` when the plugin can say how an
        identifier reads: the page shows the label with the value beside it. ``labels`` are the
        names ``lookup_labels`` resolved at the last validation, by setting path and value, so
        the plugin need not ask the destination again for those. Names the plugin has at hand,
        such as a bound feature collection's, cost nothing; the destination may be asked for the
        rest, with its failure tolerated.
        """
        return []

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
