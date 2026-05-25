/* nav.js — shared nav active-state + status banner wiring.
 * Drop this script + the nav HTML into any page.
 */
(function () {
  "use strict";

  function highlightActive() {
    const path = location.pathname.split("/").pop() || "index.html";
    document.querySelectorAll(".nav-links a[data-page]").forEach((a) => {
      if (a.dataset.page === path) a.classList.add("active");
    });
  }

  function wireConfigLinks() {
    const cfg = window.SITE_CONFIG || {};
    const apiBase = (cfg.API_BASE || "").replace(/\/+$/, "");
    const githubUrl = cfg.GITHUB_URL || "";

    function setLink(id, href, hideIfEmpty) {
      const el = document.getElementById(id);
      if (!el) return;
      if (!href) {
        if (hideIfEmpty) el.style.display = "none";
        return;
      }
      el.href = href;
    }

    setLink("nav-api-docs", apiBase ? apiBase + "/docs" : "", true);
    setLink("nav-github",   githubUrl, true);
    setLink("footer-github",   githubUrl, true);
    setLink("footer-api",      apiBase, true);
    setLink("footer-api-docs", apiBase ? apiBase + "/docs" : "", true);
    setLink("footer-ref",      "reference/", false);
  }

  function wireStatusIndicator() {
    const dot = document.getElementById("live-indicator");
    if (!dot || !window.API_STATUS) return;

    function render(s) {
      // mode tells us if the API call reached the server; dataSource tells
      // us if the server is reading from a real warehouse or its baked-in
      // synthetic fallback. Both matter — distinguish them in the label.
      let text;
      let mode;
      if (s.mode === "loading" || s.mode === "unknown") {
        text = "loading…"; mode = "loading";
      } else if (s.mode === "sample") {
        text = "synthetic (no API)"; mode = "sample";
      } else if (s.dataSource === "warehouse") {
        text = "live · real pipeline"; mode = "live";
      } else {
        // API up but serving synthetic fallback because the warehouse
        // hasn't been populated on the server. Honest label.
        text = "live API · synthetic data"; mode = "sample";
      }
      dot.dataset.mode = mode;
      dot.querySelector(".text").textContent = text;
      dot.title = s.dataSource
        ? `API: ${s.mode} · data_source: ${s.dataSource}`
        : `API: ${s.mode}`;
    }
    render(window.API_STATUS);
    window.API_STATUS.subscribe(render);
  }

  function init() {
    highlightActive();
    wireConfigLinks();
    wireStatusIndicator();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
