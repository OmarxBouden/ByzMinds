/* app.js — load data, manage tick/mode state, wire controls + tabs. */
(function () {
  const S = { runs: [], data: null, idx: 0, mode: "glassbox", timer: null };
  const $ = (id) => document.getElementById(id);

  function tickNum() { return S.data ? (S.data.ticks[S.idx] || {}).tick : 0; }

  function render() {
    if (!S.data) return;
    window.Sim3D && Sim3D.scene(S.data, tickNum(), S.mode);   // 3D owns the canvas
    window.Sim && Sim.render(S.data, tickNum(), S.mode);       // drives the side panel (+ 2D fallback)
    if ($("tab-replay").classList.contains("active") && window.Replay) Replay.render(S.data, S.mode);
    if ($("tab-metrics").classList.contains("active") && window.Metrics) Metrics.render();
    $("tick-label").textContent = "tick " + tickNum();
  }

  function setData(data) {
    S.data = data; S.idx = 0;
    const n = data.ticks.length;
    const sc = $("scrubber"); sc.max = Math.max(0, n - 1); sc.value = 0;
    window.Replay && Replay.setData(data);
    render();
  }

  function loadRun(name) {
    fetch("data/" + name + ".json").then(r => r.json()).then(setData)
      .catch(() => msg('Could not fetch data/' + name + '.json — serve with "python -m http.server", or use “load JSON”.'));
  }
  function msg(t) {
    const host = $("stage3d") || $("stage");
    if (!host) return;
    let o = host.querySelector(".stage-msg");
    if (!o) { o = document.createElement("div"); o.className = "stage-msg"; host.appendChild(o); }
    o.textContent = t;
  }

  // ---- controls ----
  function stop() { if (S.timer) { clearInterval(S.timer); S.timer = null; $("btn-play").textContent = "▶"; } }
  function play() {
    if (S.timer) return stop();
    $("btn-play").textContent = "⏸";
    S.timer = setInterval(() => {
      if (!S.data || S.idx >= S.data.ticks.length - 1) return stop();
      S.idx++; $("scrubber").value = S.idx; render();
    }, 1400);
  }
  function step(d) { stop(); if (!S.data) return; S.idx = Math.max(0, Math.min(S.data.ticks.length - 1, S.idx + d)); $("scrubber").value = S.idx; render(); }

  function init() {
    $("scrubber").addEventListener("input", e => { stop(); S.idx = +e.target.value; render(); });
    $("btn-play").addEventListener("click", play);
    $("btn-step").addEventListener("click", () => step(1));
    $("btn-step-back").addEventListener("click", () => step(-1));
    $("mode-toggle").addEventListener("click", e => {
      S.mode = S.mode === "glassbox" ? "public" : "glassbox";
      e.currentTarget.dataset.mode = S.mode;
      $("mode-label").textContent = S.mode === "glassbox" ? "Glass-box (researcher)" : "Public observer";
      render();
    });
    $("run-select").addEventListener("change", e => loadRun(e.target.value));
    $("file-input").addEventListener("change", e => {
      const f = e.target.files[0]; if (!f) return;
      const rd = new FileReader();
      rd.onload = () => { try { setData(JSON.parse(rd.result)); } catch (err) { msg("bad JSON: " + err); } };
      rd.readAsText(f);
    });
    document.querySelectorAll(".tab").forEach(t => t.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach(x => x.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach(x => x.classList.remove("active"));
      t.classList.add("active"); $("tab-" + t.dataset.tab).classList.add("active");
      render();
    }));

    // load the run index (works when served); file-input is the file:// fallback
    fetch("data/index.json").then(r => r.json()).then(j => {
      S.runs = j.runs || [];
      const sel = $("run-select");
      S.runs.forEach(r => { const o = document.createElement("option"); o.value = r.name; o.textContent = r.name; sel.appendChild(o); });
      window.Metrics && Metrics.init(S.runs);
      if (S.runs.length) loadRun(S.runs[0].name);
      else msg('No runs indexed. Use “load JSON”.');
    }).catch(() => msg('Could not load data/index.json. Serve with "python -m http.server" from frontend/, or use “load JSON” to open an exported run.'));
  }
  document.addEventListener("DOMContentLoaded", init);
})();
