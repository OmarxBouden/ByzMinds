/* metrics.js — the analytics deck: the paper's figures as a captioned gallery
   (self-contained PNGs in frontend/figures/). */
(function () {
  const FIGS = [
    ["fig2_bias_rates", "M1 — biases under situational pressure", "Bias-aligned rate per dial; only free-riding clears the bar — biases are largely resisted."],
    ["fig3_correlation", "M2 — probe validation", "Contrast-pair vs independent-probe rates per dial."],
    ["fig4_dose_response", "M3 — prompted-persona dose response", "Persona strength monotonically modulates the rate (both models)."],
    ["fig5_propagation", "M4 — dyadic propagation", "A biased peer shifts an honest agent's vote; authority strongest."],
    ["fig6_money_plot", "M5 — public vs private (the headline)", "Public-only detection can't separate colluders; the private ledger proves intent."],
    ["fig7_cost_to_attack", "M6 — protocol robustness", "An aligned majority resists; cost-to-attack is the vote threshold (3/5, 4/5)."],
    ["fig8_detection", "M7 — detection feasibility", "From the transcript alone a judge has 41% false positives and ~0 dial recall."],
    ["fig9_crossmodel", "M8 — cross-model", "The collude headline replicates on Apertus 8B."],
    ["fig11_crosslingual", "Cross-lingual transfer", "Biases transfer unevenly across EN/DE/FR/IT/ZH; natural baseline varies by language."],
  ];

  window.Metrics = {
    init() {
      const deck = document.getElementById("metrics-deck");
      if (deck.children.length) return;
      for (const [f, title, cap] of FIGS) {
        const card = document.createElement("div");
        card.className = "fig-card";
        card.innerHTML = `<h4>${title}</h4>`
          + `<img loading="lazy" src="figures/${f}.png" alt="${title}" `
          + `onerror="this.replaceWith(Object.assign(document.createElement('p'),{textContent:'(figure ${f}.png not found — run the figure scripts / copy paper/figures)'}))">`
          + `<p>${cap}</p>`;
        deck.appendChild(card);
      }
    },
    render() { this.init(); },
  };
})();
