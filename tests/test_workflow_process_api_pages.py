"""The workflow, process and API pages (CLIM-940).

Each is HTML for a browser and leaves the machine-readable form where it was: a workflow's is
`GET /process_graphs/{id}`, a process's is the same URL answering JSON unless the client ranks
HTML higher. The API page is built from the instance's own OpenAPI schema, so it lists what
this instance serves rather than what the code happens to define.
"""

from __future__ import annotations

from html.parser import HTMLParser
from typing import Any

import pytest
from fastapi.testclient import TestClient

from open_climate_service import config as api_config
from open_climate_service.system import templates as landing


def _template(template_id: str, **fields: Any) -> dict[str, Any]:
    return {"id": template_id, "name": template_id.replace("_", " "), "sync": {"kind": "static"}, **fields}


def _record(dataset_id: str = "chirps_monthly", **fields: Any) -> Any:
    from open_climate_service.ingestions.schemas import DatasetDetailRecord

    values: dict[str, Any] = {
        "dataset_id": dataset_id,
        "source_dataset_id": dataset_id,
        "dataset_name": "Precipitation (CHIRPS, monthly)",
        "description": "First line\nwraps here.\n\nSecond paragraph.",
        "variable": "precip",
        "period_type": "monthly",
        "units": "mm/d",
        "source": "CHIRPS v3",
        "source_url": "https://example.org/chirps",
        "extent": {
            "spatial": {"xmin": 80.0, "ymin": 26.0, "xmax": 88.0, "ymax": 30.0},
            "temporal": {"start": "2020-01", "end": "2026-07"},
        },
        "last_updated": "2026-09-01T07:58:03Z",
        "links": [
            {"href": f"/datasets/{dataset_id}", "rel": "self", "title": "Dataset detail"},
            {"href": f"/zarr/{dataset_id}", "rel": "zarr", "title": "Zarr store"},
        ],
        "publication": {"status": "published", "published_at": "2026-09-01T07:58:03Z"},
        "versions": [],
        **fields,
    }
    return DatasetDetailRecord.model_validate(values)


BROWSER_ACCEPT = "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"


class _VisibleText(HTMLParser):
    """Collects the text a reader sees: no markup, scripts, styles or comments."""

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style"}:
            self._skip += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"}:
            self._skip -= 1

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self.parts.append(data)


def _visible_text(html: str) -> str:
    parser = _VisibleText()
    parser.feed(html)
    return " ".join(parser.parts)


# --- the workflow page -------------------------------------------------------------------


def _workflow_record(**fields: Any) -> Any:
    from open_climate_service.openeo.schemas import WorkflowRecord

    values: dict[str, Any] = {
        "id": "aggregate_to_chap_csv",
        "summary": "Aggregate and export",
        "description": 'Loads `dataset_id` and exports.\n\nUsage:\n\n{\n  "agg": {"process_id": "x"}\n}',
        "parameters": [
            {"name": "dataset_id", "schema": {"type": "string"}, "description": "The `dataset` to read."},
            {
                "name": "period",
                "optional": True,
                "default": "month",
                "schema": {"type": "string", "enum": ["month", "week"]},
            },
            {"name": "temporal_extent", "schema": {"type": "array", "subtype": "temporal-interval"}},
        ],
        "process_graph": {
            "load": {"process_id": "load_collection", "arguments": {}},
            "save": {"process_id": "save_result", "arguments": {"format": "CHAPCSV"}, "result": True},
        },
        **fields,
    }
    return WorkflowRecord.model_validate(values)


def test_the_workflow_page_describes_the_workflow() -> None:
    context = landing._workflow_page_context(_workflow_record(), [], [], [])

    assert context["workflow"]["title"] == "Aggregate to CHAP CSV"
    assert context["workflow"]["results"] == [("export", "Exports CHAP CSV")]
    assert str(context["blocks"][0]["html"]) == "Loads <code>dataset_id</code> and exports."
    assert context["blocks"][2]["code"].startswith("{")
    params = {p["name"]: p for p in context["parameters"]}
    assert params["dataset_id"]["required"] is True
    assert params["period"] == {**params["period"], "required": False, "type": "month | week", "default": '"month"'}
    assert params["temporal_extent"]["type"] == "temporal-interval"


def test_workflow_descriptions_are_escaped() -> None:
    context = landing._workflow_page_context(_workflow_record(description="<script>x</script> `<b>`"), [], [], [])

    assert str(context["blocks"][0]["html"]) == "&lt;script&gt;x&lt;/script&gt; <code>&lt;b&gt;</code>"


def test_the_workflow_page_lists_outputs_and_triggers() -> None:
    from open_climate_service.automation.config import WorkflowTrigger

    record = _workflow_record(id="climate_normal")
    templates = [
        _template("normal_b", name="B normal", produced_by="climate_normal"),
        _template("normal_a", name="A normal", produced_by="climate_normal"),
        _template("other", produced_by="temporal_change"),
    ]
    triggers = [
        WorkflowTrigger(id="refresh", on_update_of="chirps_monthly", workflow_id="climate_normal", arguments={"x": 1}),
        WorkflowTrigger(id="elsewhere", on_update_of="chirps_monthly", workflow_id="temporal_change"),
    ]

    context = landing._workflow_page_context(record, templates, [_record("normal_b")], triggers)

    assert [(o["id"], o["ingested"]) for o in context["outputs"]] == [("normal_a", False), ("normal_b", True)]
    assert [t["id"] for t in context["triggers"]] == ["refresh"]
    assert context["triggers"][0]["held"] is False


def test_the_workflow_page_is_served(client: TestClient) -> None:
    response = client.get("/workflows/climate_normal")

    assert response.status_code == 200
    assert "Climate normal" in response.text
    assert 'href="/process_graphs/climate_normal"' in response.text
    assert client.get("/workflows/does_not_exist").status_code == 404


# --- processes ---------------------------------------------------------------------------


def test_processes_are_tagged_with_their_origin() -> None:
    processes = {p["id"]: p for p in landing._load_processes()}

    assert processes["load_collection"]["origin"] == "core"
    assert processes["add"]["origin"] == "core"
    assert processes["spi"]["origin"] == "ocs"
    assert {"xclim", "earthkit"} <= {p["origin"] for p in processes.values()}
    # OCS first, openEO core last, so the default view leads with what is specific to OCS.
    origins = [p["origin"] for p in processes.values()]
    assert origins[0] == "ocs" and origins[-1] == "core"


def test_a_plugin_overriding_an_indicator_counts_as_ocs(monkeypatch: pytest.MonkeyPatch) -> None:
    from open_climate_service.openeo import earthkit_processes, plugin_processes, processes, xclim_processes

    def indicator() -> None: ...

    def override() -> None: ...

    monkeypatch.setattr(processes, "list_openeo_processes", lambda: [{"id": "tg_mean"}, {"id": "abs"}])
    monkeypatch.setattr(xclim_processes, "scan", lambda: [indicator])
    monkeypatch.setattr(earthkit_processes, "scan", lambda: [])
    monkeypatch.setattr(plugin_processes, "load_plugin_processes", lambda: [("tg_mean", override)])

    assert {p["id"]: p["origin"] for p in landing._load_processes()} == {"tg_mean": "ocs", "abs": "core"}


# --- the process page --------------------------------------------------------------------


def test_process_descriptions_render_their_markdown_safely() -> None:
    blocks = landing._description_blocks(
        "Uses ``eq()`` and **bold**; see [docs](https://openeo.org) or [x](javascript:alert(1)).\n\n"
        "* First <item>\n* Second `code`\n\n"
        "```\nprint(1)\n```"
    )

    assert str(blocks[0]["html"]) == (
        'Uses <code>eq()</code> and <strong>bold</strong>; see <a href="https://openeo.org">docs</a> '
        "or [x](javascript:alert(1))."
    )
    assert [str(item) for item in blocks[1]["bullets"]] == ["First &lt;item&gt;", "Second <code>code</code>"]
    assert blocks[2] == {"code": "print(1)\n"}


def test_a_numpydoc_docstring_is_rendered_rather_than_leaked() -> None:
    """Process descriptions are docstrings, and earthkit's are reStructuredText.

    Rendered as Markdown they leaked: the section underline showed as a row of dashes, the
    `.. math::` directive as literal text, and every role name before its argument.
    """
    blocks = landing._description_blocks(
        "Compute a thing.\n\n"
        "Parameters\n----------\nt: array-like\n    Temperature (K)\n\n"
        "Returns\n-------\narray-like\n    Dewpoint (K). For zero ``r`` returns nan.\n\n\n"
        "The pressure at the dewpoint:\n\n.. math::\n\n    e(td) = r/100\n\n"
        "where :math:`e` is the :func:`vapour_pressure`."
    )
    rendered = [b.get("code") or str(b.get("html") or b.get("bullets")) for b in blocks]

    # Parameters and Returns are tables further down the page, so they are not repeated here.
    assert not any("Parameters" in str(r) or "----" in str(r) for r in rendered)
    # The formula keeps its own block rather than being folded into a sentence.
    assert any(b.get("code", "").strip() == "e(td) = r/100" for b in blocks)
    # Roles and ``literals`` become code, with no role name left showing.
    assert any("<code>e</code> is the <code>vapour_pressure</code>" in str(r) for r in rendered)
    assert not any(":math:" in str(r) or ":func:" in str(r) for r in rendered)


def test_a_description_that_is_already_markdown_is_untouched() -> None:
    """The guard on the literal pattern: a fence is backticks too."""
    blocks = landing._description_blocks("Text.\n\n```\nprint(1)\n```")

    assert blocks[1] == {"code": "print(1)\n"}


def test_the_lead_sentence_is_not_repeated_as_the_first_paragraph() -> None:
    """A docstring's first line is its summary, so the page led with it twice."""
    context = landing._process_page_context(
        {"id": "x", "summary": "Compute a thing.", "description": "Compute a thing.\n\nThen more."},
        "earthkit",
        [],
    )

    assert context["process"]["summary"] == "Compute a thing."
    assert [str(b["html"]) for b in context["blocks"]] == ["Then more."]


def test_the_process_page_lists_the_workflows_that_use_it() -> None:
    process = {
        "id": "reduce_dimension",
        "summary": "Reduce",
        "categories": ["cubes"],
        "parameters": [{"name": "data", "schema": {"type": "object", "subtype": "datacube"}}],
        "returns": {"description": "A cube.", "schema": {"type": "object", "subtype": "datacube"}},
        "links": [{"href": "https://example.org", "title": "About"}, {"href": "javascript:x", "rel": "bad"}],
    }
    nested = _workflow_record(
        id="uses_it",
        process_graph={
            "r": {
                "process_id": "apply",
                "arguments": {"process": {"process_graph": {"x": {"process_id": "reduce_dimension"}}}},
            }
        },
    )
    unrelated = _workflow_record(id="does_not")

    context = landing._process_page_context(process, "openEO core", [nested, unrelated])

    assert [w["id"] for w in context["used_by"]] == ["uses_it"]
    assert context["returns"]["type"] == "datacube"
    assert context["parameters"][0]["type"] == "datacube"
    assert context["links"] == [{"href": "https://example.org", "title": "About"}]


@pytest.mark.parametrize(("accept", "html"), [(BROWSER_ACCEPT, True), ("*/*", False), ("application/json", False)])
def test_the_process_endpoint_serves_a_page_only_to_browsers(client: TestClient, accept: str, html: bool) -> None:
    response = client.get("/processes/load_collection", headers={"Accept": accept})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html" if html else "application/json")
    if html:
        assert 'href="/workflows/climate_normal"' in response.text
        # The rail names no Processes entry yet — that is an area of the landing page, which
        # arrives in the next slice — but the page is served under the shared chrome.
        assert 'class="rail"' in response.text
        assert '<a class="home" href="/">' in response.text
    else:
        assert response.json()["id"] == "load_collection"


# --- the API page ------------------------------------------------------------------------


def test_the_api_page_lists_the_instance_s_own_endpoints(client: TestClient) -> None:
    html = client.get("/api").text
    schema = client.get("/openapi.json").json()
    served = {(method.upper(), path) for path, ops in schema["paths"].items() for method in ops}

    for method, path in served:
        assert f"<code>{path}</code>" in html, path
        assert f"<code>{method}</code>" in html
    # Entry points for the catalogues and the docs, which have no OpenAPI path of their own.
    for href in ("/stac/catalog.json", "/docs", "/openapi.json", "/collections", "/?f=json"):
        assert f'href="{href}"' in html
    assert '<a href="/api" aria-current="page">API</a>' in html


def test_endpoint_rows_prefer_the_docstring_over_the_generated_summary(client: TestClient) -> None:
    html = client.get("/api").text

    assert "Read Index" not in html, "FastAPI's function-name summary is not what a reader wants"
    assert "Return openEO capabilities (JSON) or the landing page (HTML)." in html


def test_the_api_page_marks_what_read_only_closes(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    monkeypatch.setattr(api_config, "is_read_only", lambda: True)

    context = landing._api_page_context(schema, read_only=True)
    closed = {(e["method"], e["path"]) for group in context["groups"] for e in group["endpoints"] if e["closed"]}
    everything = {(e["method"], e["path"]) for group in context["groups"] for e in group["endpoints"]}

    assert ("POST", "/ingestions") in closed
    assert ("GET", "/jobs") in closed
    assert ("GET", "/datasets") in everything - closed
    assert "closed" in landing.render_api_page(schema, "")


def test_a_writable_instance_marks_nothing_closed(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()

    context = landing._api_page_context(schema, read_only=False)

    assert not [e for group in context["groups"] for e in group["endpoints"] if e["closed"]]


# --- the list pages ------------------------------------------------------------------------


def test_the_workflow_list_is_its_own_page(client: TestClient) -> None:
    """HTML only: the machine-readable list stays at `GET /process_graphs`."""
    response = client.get("/workflows")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert 'href="/workflows/climate_normal"' in response.text
    assert client.get("/process_graphs").status_code == 200


@pytest.mark.parametrize(
    ("accept", "html"),
    [(BROWSER_ACCEPT, True), ("*/*", False), ("", False), ("application/json", False)],
)
def test_the_process_list_serves_a_page_only_to_browsers(client: TestClient, accept: str, html: bool) -> None:
    """openEO clients keep the JSON catalogue they have always had at this URL."""
    headers = {"Accept": accept} if accept else {}

    response = client.get("/processes", headers=headers)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html" if html else "application/json")
    assert response.headers["vary"] == "Accept"
    if html:
        assert 'data-views="processes"' in response.text
        assert "data-origin=" in response.text, "the origin filter needs the origins passed in"
    else:
        assert "processes" in response.json()


def test_a_fenced_block_survives_a_blank_line_inside_it() -> None:
    """Fences are taken whole before paragraphs are.

    Splitting on blank lines first cut such a block in two: the opening half became code and
    the rest was rendered as prose, carrying the closing fence into the text.
    """
    blocks = landing._description_blocks("Text.\n\n```\nline one\n\nline two\n```\n\nAfter.")

    assert [b.get("code") or str(b.get("html")) for b in blocks] == [
        "Text.",
        "line one\n\nline two\n",
        "After.",
    ]


def test_the_negotiated_process_endpoints_still_publish_a_json_schema(client: TestClient) -> None:
    """Serving two representations must not cost the machine-readable contract.

    `response_model=None` silences FastAPI's inference, which would drop the JSON schema these
    endpoints published before they learned to answer HTML.
    """
    paths = client.get("/openapi.json").json()["paths"]

    for path in ("/processes", "/processes/{process_id}"):
        content = paths[path]["get"]["responses"]["200"]["content"]
        assert "application/json" in content, path
        assert "text/html" in content, path
