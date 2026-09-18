// Progressive enhancement: without JavaScript every area is shown in sequence and every
// list is complete, and the rail links scroll to their section.
(function () {
  document.documentElement.classList.add("js");

  var areas = Array.prototype.slice.call(document.querySelectorAll("[data-area]"));
  var links = Array.prototype.slice.call(document.querySelectorAll("[data-area-link]"));

  function showArea(id, moveFocus) {
    var known = areas.some(function (area) {
      return area.id === id;
    });
    if (!known) id = "overview";
    areas.forEach(function (area) {
      area.hidden = area.id !== id;
    });
    links.forEach(function (link) {
      if (link.getAttribute("data-area-link") === id) link.setAttribute("aria-current", "page");
      else link.removeAttribute("aria-current");
    });
    window.scrollTo(0, 0);
    if (moveFocus) {
      var heading = document.getElementById(id + "-title");
      if (heading) heading.focus({ preventScroll: true });
    }
  }

  window.addEventListener("hashchange", function () {
    showArea(location.hash.slice(1), true);
  });
  showArea(location.hash.slice(1), false);

  function initList(root) {
    var items = Array.prototype.slice.call(root.querySelectorAll("[data-item]"));
    var pageSize = parseInt(root.getAttribute("data-page-size"), 10) || 12;
    var text = root.querySelector("[data-filter-text]");
    var selects = Array.prototype.slice.call(root.querySelectorAll("[data-filter-field]"));
    var status = root.querySelector("[data-status]");
    var empty = root.querySelector("[data-empty]");
    var pager = root.querySelector("[data-pager]");
    var prev = root.querySelector("[data-prev]");
    var next = root.querySelector("[data-next]");
    var page = 0;

    function apply() {
      var query = text ? text.value.trim().toLowerCase() : "";
      var matched = items.filter(function (item) {
        if (query && item.getAttribute("data-search").indexOf(query) === -1) return false;
        return selects.every(function (select) {
          var field = select.getAttribute("data-filter-field");
          var value = select.value;
          var actual = item.getAttribute("data-" + field);
          // A leading "!" excludes a value, so a filter can default to "everything but".
          if (value.charAt(0) === "!") return actual !== value.slice(1);
          return !value || actual === value;
        });
      });
      var pages = Math.max(1, Math.ceil(matched.length / pageSize));
      page = Math.min(page, pages - 1);
      var start = page * pageSize;
      var visible = matched.slice(start, start + pageSize);
      items.forEach(function (item) {
        item.hidden = visible.indexOf(item) === -1;
      });
      if (status) {
        status.textContent = matched.length
          ? "Showing " + (start + 1) + "–" + (start + visible.length) + " of " + matched.length
          : "";
      }
      if (empty) empty.hidden = matched.length > 0;
      if (pager) pager.hidden = pages <= 1;
      if (prev) prev.disabled = page === 0;
      if (next) next.disabled = page >= pages - 1;
    }

    if (text)
      text.addEventListener("input", function () {
        page = 0;
        apply();
      });
    selects.forEach(function (select) {
      select.addEventListener("change", function () {
        page = 0;
        apply();
      });
    });
    if (prev)
      prev.addEventListener("click", function () {
        page -= 1;
        apply();
      });
    if (next)
      next.addEventListener("click", function () {
        page += 1;
        apply();
      });
    var views = root.getAttribute("data-views");
    var target = root.querySelector("[data-view-target]");
    var viewButtons = Array.prototype.slice.call(root.querySelectorAll("[data-view-button]"));
    function setView(view) {
      if (!target) return;
      target.classList.toggle("view-tiles", view !== "list");
      target.classList.toggle("view-list", view === "list");
      viewButtons.forEach(function (button) {
        button.setAttribute("aria-pressed", String(button.getAttribute("data-view-button") === view));
      });
    }
    if (views) {
      var stored = null;
      try {
        stored = window.localStorage.getItem("ocs.view." + views);
      } catch (e) {}
      setView(stored === "list" ? "list" : "tiles");
      viewButtons.forEach(function (button) {
        button.addEventListener("click", function () {
          var view = button.getAttribute("data-view-button");
          setView(view);
          try {
            window.localStorage.setItem("ocs.view." + views, view);
          } catch (e) {}
        });
      });
    }

    apply();
  }

  Array.prototype.slice.call(document.querySelectorAll("[data-list]")).forEach(initList);
})();

