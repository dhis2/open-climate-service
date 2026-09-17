// Runs a form against one of the streaming endpoints (/manage/ingest, /manage/sync) and
// reports progress in place. A request that cannot start is refused with a JSON error; one that
// starts answers with server-sent events: progress updates, then `finished` or `error`.
function runJobForm(form, options) {
  if (!form) return;
  var progress = form.querySelector("[data-job-progress]");
  var fill = progress.querySelector(".progress-fill");
  var label = progress.querySelector(".progress-label");
  var error = form.querySelector("[data-job-error]");
  var controls = Array.prototype.slice.call(form.querySelectorAll("input:not([type=hidden]), button"));

  function setBusy(busy) {
    controls.forEach(function (control) {
      control.disabled = busy;
    });
  }

  function fail(message) {
    setBusy(false);
    progress.hidden = true;
    error.textContent = message || "The request failed.";
    error.hidden = false;
  }

  form.addEventListener("submit", async function (event) {
    event.preventDefault();
    var data = new FormData(form);
    error.hidden = true;
    progress.hidden = false;
    fill.className = "progress-fill indeterminate";
    fill.style.width = "";
    label.textContent = options.startLabel || "Starting…";
    setBusy(true);

    var response;
    try {
      response = await fetch(form.action, { method: "POST", body: data });
    } catch (e) {
      fail("The request could not be sent.");
      return;
    }
    if (!response.ok || !response.body) {
      var detail = null;
      try {
        detail = (await response.json()).error;
      } catch (e) {}
      fail(detail || "The request could not start (HTTP " + response.status + ").");
      return;
    }

    var reader = response.body.getReader();
    var decoder = new TextDecoder();
    var buffer = "";
    while (true) {
      var chunk = await reader.read();
      if (chunk.done) break;
      buffer += decoder.decode(chunk.value, { stream: true });
      var lines = buffer.split("\n");
      buffer = lines.pop() || "";
      for (var i = 0; i < lines.length; i++) {
        if (lines[i].indexOf("data: ") !== 0) continue;
        var evt;
        try {
          evt = JSON.parse(lines[i].slice(6));
        } catch (e) {
          continue;
        }
        if (evt.error) {
          fail(evt.error);
          return;
        }
        if (evt.finished) {
          options.onDone(form);
          return;
        }
        if (typeof evt.done === "number" && typeof evt.total === "number" && evt.total > 0) {
          fill.className = "progress-fill";
          fill.style.width = Math.round((evt.done / evt.total) * 100) + "%";
        }
        if (evt.message) label.textContent = evt.message;
      }
    }
    fail("The request ended without a result.");
  });
}
