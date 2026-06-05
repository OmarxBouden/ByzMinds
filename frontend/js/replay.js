/* replay.js — the replay & ledger dashboard. Shows the committed, hash-chained
   event log with a per-ledger filter and a replay-verified badge. */
(function () {
  const LEDGERS = ["L_pub", "L_prv", "L_cog_ind", "L_cog_eli", "L_ctrl"];
  const S = { data: null, on: new Set(LEDGERS) };
  const $ = (id) => document.getElementById(id);

  function chips() {
    const box = $("ledger-filters"); box.innerHTML = "";
    for (const l of LEDGERS) {
      const c = document.createElement("span");
      c.className = "lf on"; c.dataset.l = l; c.textContent = l;
      c.onclick = () => { c.classList.toggle("on"); S.on.has(l) ? S.on.delete(l) : S.on.add(l); table(); };
      box.appendChild(c);
    }
  }
  function badge() {
    const b = $("replay-badge"), r = S.data.replay || {};
    b.className = "badge " + (r.ok ? "ok" : "bad");
    b.textContent = (r.ok ? "✓ replay verified" : "✗ replay mismatch")
      + ` · ${r.n_events} events · head ${r.final_chain_hash}`;
  }
  function table() {
    const t = $("replay-table"); if (!S.data) return;
    const rows = (S.data.events || []).filter(e => S.on.has(e.ledger));
    let html = "<thead><tr><th>seq</th><th>ledger</th><th>event</th><th>emitter</th><th>chain</th></tr></thead><tbody>";
    for (const e of rows) {
      html += `<tr><td>${e.seq}</td><td><span class="lpill" style="border-color:${col(e.ledger)};color:${col(e.ledger)}">${e.ledger}</span></td>`
        + `<td>${e.type}</td><td>${e.agent ? e.agent.replace("reviewer_", "R") : "kernel"}</td>`
        + `<td style="color:#7e8aa6">${e.chain}…</td></tr>`;
    }
    t.innerHTML = html + "</tbody>";
  }
  function col(l) {
    return l === "L_pub" ? "#38e1ff" : l === "L_prv" ? "#ff4d6d"
      : l.startsWith("L_cog") ? "#ffd166" : "#7e8aa6";
  }

  window.Replay = {
    setData(d) { S.data = d; if (!$("ledger-filters").children.length) chips(); },
    render() { if (!S.data) return; badge(); table(); },
  };
})();
