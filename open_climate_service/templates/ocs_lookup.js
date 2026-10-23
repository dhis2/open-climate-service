// Progressive enhancement for lookup fields: without JavaScript each is a text input holding the
// stored id. With it, the input searches the destination through the form's lookup URL and the
// id travels in a hidden input, so the form posts exactly what it would have without the script.
(function () {
  var DEBOUNCE_MS = 250;

  function init(input) {
    var form = input.form;
    var url = form ? form.getAttribute("data-lookup-url") : null;
    var lookup = input.getAttribute("data-lookup");
    if (!url || !lookup || !input.name) return;

    var hidden = document.createElement("input");
    hidden.type = "hidden";
    hidden.name = input.name;
    hidden.value = input.value;
    input.removeAttribute("name");
    input.insertAdjacentElement("afterend", hidden);

    var chosenLabel = input.getAttribute("data-label") || "";
    if (chosenLabel) input.value = chosenLabel;

    var list = document.createElement("ul");
    list.className = "lookup-results";
    list.setAttribute("role", "listbox");
    list.id = input.id + "-results";
    list.hidden = true;
    hidden.insertAdjacentElement("afterend", list);

    input.setAttribute("autocomplete", "off");
    input.setAttribute("role", "combobox");
    input.setAttribute("aria-autocomplete", "list");
    input.setAttribute("aria-expanded", "false");
    input.setAttribute("aria-controls", list.id);

    var timer = null;
    var controller = null;
    var active = -1;

    function close() {
      list.hidden = true;
      list.innerHTML = "";
      active = -1;
      input.setAttribute("aria-expanded", "false");
      input.removeAttribute("aria-activedescendant");
    }

    function open() {
      list.hidden = false;
      input.setAttribute("aria-expanded", "true");
    }

    function status(text, cls) {
      list.innerHTML = "";
      var item = document.createElement("li");
      item.className = cls || "lookup-status";
      item.textContent = text;
      list.appendChild(item);
      open();
    }

    function choose(option) {
      hidden.value = option.value;
      chosenLabel = option.label;
      input.value = option.label;
      close();
    }

    function render(options) {
      list.innerHTML = "";
      if (!options.length) {
        status("No match");
        return;
      }
      options.forEach(function (option, index) {
        var item = document.createElement("li");
        item.setAttribute("role", "option");
        item.id = list.id + "-" + index;
        item.setAttribute("aria-selected", "false");
        var label = document.createElement("span");
        label.textContent = option.label;
        var detail = document.createElement("small");
        detail.textContent = option.detail ? option.value + " · " + option.detail : option.value;
        item.appendChild(label);
        item.appendChild(detail);
        // mousedown, not click: the input's blur would close the list before a click lands.
        item.addEventListener("mousedown", function (event) {
          event.preventDefault();
          choose(option);
        });
        list.appendChild(item);
      });
      active = -1;
      open();
    }

    function highlight(index) {
      var items = list.querySelectorAll('[role="option"]');
      if (!items.length) return;
      active = (index + items.length) % items.length;
      Array.prototype.forEach.call(items, function (item, i) {
        item.setAttribute("aria-selected", i === active ? "true" : "false");
      });
      input.setAttribute("aria-activedescendant", items[active].id);
      items[active].scrollIntoView({ block: "nearest" });
    }

    function search(query) {
      if (controller) controller.abort();
      controller = new AbortController();
      var body = new FormData(form);
      body.set("q", query);
      status("Searching…");
      fetch(url + "/" + encodeURIComponent(lookup), {
        method: "POST",
        body: body,
        headers: { Accept: "application/json" },
        signal: controller.signal,
      })
        .then(function (response) {
          return response.json().then(function (data) {
            if (!response.ok) throw new Error(data.detail || response.statusText);
            return data.options || [];
          });
        })
        .then(render)
        .catch(function (error) {
          if (error.name === "AbortError") return;
          status(error.message || "The lookup failed", "lookup-error");
        });
    }

    function schedule() {
      if (timer) clearTimeout(timer);
      timer = setTimeout(function () {
        search(input.value === chosenLabel ? "" : input.value.trim());
      }, DEBOUNCE_MS);
    }

    input.addEventListener("input", function () {
      var text = input.value.trim();
      if (text !== chosenLabel) {
        // Typed text is the value until an option is chosen, so a pasted id still works.
        chosenLabel = "";
        hidden.value = text;
      }
      schedule();
    });
    input.addEventListener("focus", function () {
      if (list.hidden) schedule();
    });
    input.addEventListener("blur", function () {
      if (timer) clearTimeout(timer);
      if (controller) controller.abort();
      close();
    });
    input.addEventListener("keydown", function (event) {
      if (event.key === "ArrowDown") {
        event.preventDefault();
        if (list.hidden) schedule();
        else highlight(active + 1);
      } else if (event.key === "ArrowUp") {
        event.preventDefault();
        highlight(active - 1);
      } else if (event.key === "Enter" && !list.hidden && active >= 0) {
        event.preventDefault();
        list.querySelectorAll('[role="option"]')[active].dispatchEvent(new Event("mousedown"));
      } else if (event.key === "Escape" && !list.hidden) {
        event.preventDefault();
        close();
      }
    });

  }

  Array.prototype.forEach.call(document.querySelectorAll("input[data-lookup]"), init);
})();
