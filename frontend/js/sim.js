/* sim.js — the agentic-civilization scene. Deterministic render of one tick.
   mode "public" hides L_prv + L_cog_* (the public-observer view); mode "glassbox"
   reveals everything (the researcher view) — this is the M5/M7 demonstration. */
(function () {
  const NS = "http://www.w3.org/2000/svg";
  const DIAL_COLOR = {           // matches the paper's Wong palette
    deceive: "#D55E00", authority: "#0072B2", bandwagon: "#009E73",
    sycophancy: "#CC79A7", free_ride: "#F0E442", collude: "#56B4E9",
  };
  const HONEST = "#5b6b8c", PRV = "#ff4d6d", COG = "#ffd166", PUB = "#38e1ff";
  const stage = () => document.getElementById("stage");

  function el(tag, attrs, parent) {
    const n = document.createElementNS(NS, tag);
    for (const k in (attrs || {})) n.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(n);
    return n;
  }
  function bubble(g, x, y, color, text, w = 190) {
    const grp = el("g", { transform: `translate(${x},${y})` }, g);
    const fo = el("foreignObject", { x: -w / 2, y: 0, width: w, height: 76 }, grp);
    const div = document.createElement("div");
    div.style.cssText = `border:1px solid ${color};border-radius:8px;background:rgba(18,26,43,.78);`
      + `color:#cdd9f0;font:11px/1.35 ui-monospace,monospace;padding:5px 8px;`
      + `box-shadow:0 0 10px ${color}44;max-height:66px;overflow:hidden`;
    div.textContent = text;
    fo.appendChild(div);
    return grp;
  }
  const esc = (s) => (s || "").replace(/[&<>]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
  const trunc = (s, n) => s && s.length > n ? s.slice(0, n) + "…" : (s || "");

  // per-agent running tallies up to and including `tick`
  function tallies(data, tick) {
    const t = {};
    data.agents.forEach(a => t[a.id] = { vote: null, pub: 0, prv: 0, intents: 0 });
    for (const tk of data.ticks) {
      if (tk.tick > tick) break;
      for (const e of tk.events) {
        const r = t[e.agent]; if (!r) continue;
        if (e.type === "Speak") (e.ledger === "L_prv" ? r.prv++ : r.pub++);
        else if (e.type === "Vote") r.vote = e.vote;
        else if (e.type === "DeclareIntent") r.intents++;
      }
    }
    return t;
  }

  function agentColor(a, mode) {
    if (mode === "public" || !a.profile.biased) return HONEST;
    return DIAL_COLOR[a.profile.dial] || "#ff6b3d";
  }

  window.Sim = {
    render(data, tick, mode) {
      // The 3D scene (sim3d.js) owns the canvas; drive only the side panel.
      // The SVG block below remains as a legacy 2D fallback if #stage exists.
      if (!stage()) {
        const tk0 = data.ticks.find(t => t.tick === tick) || { tick, phase: "—", events: [] };
        const pb = document.getElementById("phase-banner");
        if (pb) pb.textContent = (tk0.phase || "—").replace(/_/g, " ");
        let hidden = 0;
        if (mode === "public") for (const e of tk0.events)
          if ((e.type === "Speak" && e.ledger === "L_prv") || e.type === "DeclareIntent") hidden++;
        return this.sidebar(data, tk0, mode, tallies(data, tick), hidden);
      }
      const svg = stage();
      svg.innerHTML = "";
      const W = 900, H = 560, cx = 440, cy = 285, R = 185;
      const tk = data.ticks.find(t => t.tick === tick) || { tick, phase: "—", events: [] };
      document.getElementById("phase-banner").textContent = (tk.phase || "—").replace(/_/g, " ");

      // agent positions (ring)
      const pos = {};
      data.agents.forEach((a, i) => {
        const ang = -Math.PI / 2 + i * 2 * Math.PI / data.agents.length;
        pos[a.id] = { x: cx + R * Math.cos(ang), y: cy + R * Math.sin(ang) };
      });

      // coalition halos for private channels (glass-box only)
      if (mode === "glassbox") {
        for (const ch of data.channels) {
          if (ch.id === "public" || ch.members.length < 2) continue;
          const pts = ch.members.map(m => pos[m]).filter(Boolean);
          if (pts.length < 2) continue;
          const mx = pts.reduce((s, p) => s + p.x, 0) / pts.length;
          const my = pts.reduce((s, p) => s + p.y, 0) / pts.length;
          for (const p of pts)
            el("line", { x1: mx, y1: my, x2: p.x, y2: p.y, class: "coalition" }, svg);
          el("circle", { cx: mx, cy: my, r: 16, class: "coalition" }, svg);
          const lbl = el("text", { x: mx, y: my - 22, class: "coalition-label", "text-anchor": "middle" }, svg);
          lbl.textContent = "🔒 " + ch.id;
        }
      }

      // agents
      const evByAgent = {};
      tk.events.forEach(e => { (evByAgent[e.agent] = evByAgent[e.agent] || []).push(e); });
      for (const a of data.agents) {
        const p = pos[a.id], col = agentColor(a, mode);
        const g = el("g", { class: "agent", transform: `translate(${p.x},${p.y})` }, svg);
        el("circle", { class: "body", r: 26, fill: "rgba(18,26,43,.9)", stroke: col,
          style: `filter:drop-shadow(0 0 8px ${col}66)` }, g);
        el("circle", { r: 26, fill: "none", stroke: col, "stroke-width": a.profile.biased && mode === "glassbox" ? 4 : 2 }, g);
        const name = el("text", { y: 4 }, g); name.textContent = a.id.replace("reviewer_", "R");
        const role = el("text", { y: 42, class: "role" }, g); role.textContent = a.role;
        // disposition badge (glass-box only)
        if (a.profile.biased && mode === "glassbox") {
          const b = el("text", { y: -34, class: "badge", fill: col }, g);
          b.textContent = "▲ " + a.profile.dial + "/" + a.profile.strength;
        } else if (mode === "public") {
          const b = el("text", { y: -34, class: "badge", fill: HONEST }, g);
          b.textContent = "?";
        }
        // vote token
        const ev = evByAgent[a.id] || [];
        const vote = ev.find(e => e.type === "Vote");
        if (vote) {
          const vt = el("text", { x: 26, y: -18, class: "ballot",
            fill: vote.vote === "accept" ? "#22c55e" : "#ef4444" }, g);
          vt.textContent = vote.vote === "accept" ? "✓" : "✗";
        }
      }

      // speech / thought bubbles for this tick's events
      let hiddenCount = 0;
      for (const e of tk.events) {
        const p = pos[e.agent]; if (!p) continue;
        const above = p.y < cy;                       // place bubble away from centre
        const by = above ? p.y - 95 : p.y + 34;
        if (e.type === "Speak" && e.ledger === "L_pub")
          bubble(svg, p.x, by, PUB, trunc(e.content, 120));
        else if (e.type === "Speak" && e.ledger === "L_prv") {
          if (mode === "glassbox") bubble(svg, p.x, by, PRV, "🔒 " + trunc(e.content, 120));
          else hiddenCount++;
        } else if (e.type === "DeclareIntent") {
          if (mode === "glassbox") bubble(svg, p.x + 70, by, COG, "💭 " + trunc(e.content, 90), 170);
          else hiddenCount++;
        }
      }

      this.sidebar(data, tk, mode, tallies(data, tick), hiddenCount);
    },

    sidebar(data, tk, mode, tal, hidden) {
      document.getElementById("side-tick").textContent = tk.tick;
      document.getElementById("side-phase").textContent = (tk.phase || "—").replace(/_/g, " ");
      const ul = document.getElementById("side-events"); ul.innerHTML = "";
      for (const e of tk.events) {
        if (e.type === "CogIndSnapshot") continue;
        const lg = e.ledger === "L_prv" ? "prv" : e.ledger.startsWith("L_cog") ? "cog" : "pub";
        const hide = mode === "public" && (e.ledger === "L_prv" || e.ledger.startsWith("L_cog"));
        const li = document.createElement("li");
        li.className = lg + (hide ? " hidden" : "");
        const what = e.type === "Speak" ? (hide ? "[private — hidden]" : esc(trunc(e.content, 80)))
          : e.type === "Vote" ? "voted " + e.vote
          : e.type === "DeclareIntent" ? (hide ? "[intent — hidden]" : "💭 " + esc(trunc(e.content, 70)))
          : e.type;
        li.innerHTML = `<span class="who">${e.agent ? e.agent.replace("reviewer_", "R") : "kernel"}</span> ${what}`;
        ul.appendChild(li);
      }
      const tbl = document.getElementById("side-agents"); tbl.innerHTML = "";
      for (const a of data.agents) {
        const r = tal[a.id]; const tr = document.createElement("tr");
        const disp = mode === "glassbox" && a.profile.biased ? a.profile.dial : "—";
        const v = r.vote ? `<span class="vt-${r.vote}">${r.vote}</span>` : "—";
        tr.innerHTML = `<td>${a.id.replace("reviewer_", "R")}</td><td>${disp}</td>`
          + `<td>${r.pub}🗣</td><td>${mode === "glassbox" ? r.prv + "🔒" : "·"}</td><td>${v}</td>`;
        tbl.appendChild(tr);
      }
      const note = document.getElementById("public-note");
      if (mode === "public")
        note.textContent = hidden > 0
          ? `Public-observer view: ${hidden} private/cognition event(s) this tick are hidden. From the public record alone you cannot see the collusion or the agents' true dispositions.`
          : "Public-observer view: private channels and induced dispositions are hidden.";
      else note.textContent = "Glass-box view: every ledger is visible — private coordination and ground-truth dispositions included.";
    },
  };
})();
