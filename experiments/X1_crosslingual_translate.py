"""X1 -- translate the contrast pairs into target languages with a defensible
equivalence gate, for the cross-lingual bias-transfer study (Part 1).

Translator: NLLB-200 (open, published, reproducible MT). For each EN pair
(p_plus = cue present, p_minus = cue absent) we translate both into language L,
then back-translate L->EN and gate on:
  (1) round-trip chrF >= --chrf-threshold for BOTH p_plus and p_minus (sacrebleu);
  (2) minimal-pair preserved: p_plus_L != p_minus_L AND the back-translations
      still differ (chrF(p_plus_back,p_minus_back) < 100), so the bias cue was not
      translated away.
Kept pairs -> agent/data/contrast_pairs_xling/<lang>/<dial>.jsonl (with scores);
pass-rates -> analysis/data/crosslingual_gate.json.

The downstream comparison (X2) is the WITHIN-language asymmetry p_+ - p_-, so the
critical invariant is the within-language minimal pair, not exact cross-language
equivalence; the gate enforces exactly that and reports what it drops.

    python experiments/X1_crosslingual_translate.py --dials deceive,authority \
        --langs de,fr,it,zh --limit 20 --beams 4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PAIRS_DIR = REPO / "agent" / "data" / "contrast_pairs"
OUT_DIR = REPO / "agent" / "data" / "contrast_pairs_xling"
DATA = REPO / "analysis" / "data"

LANG_CODE = {"en": "eng_Latn", "de": "deu_Latn", "fr": "fra_Latn",
             "it": "ita_Latn", "zh": "zho_Hans"}


def _load_nllb(name: str):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSeq2SeqLM.from_pretrained(name)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    return tok, model.to(dev), dev


def _translate_batch(tok, model, dev, texts: list[str], src: str, tgt: str,
                     beams: int = 4, chunk: int = 12) -> list[str]:
    """Batched NLLB translation src->tgt. Chunks to bound MPS memory."""
    tok.src_lang = src
    bos = tok.convert_tokens_to_ids(tgt)
    out: list[str] = []
    for i in range(0, len(texts), chunk):
        batch = texts[i:i + chunk]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(dev)
        gen = model.generate(**enc, forced_bos_token_id=bos, max_length=512, num_beams=beams)
        out += tok.batch_decode(gen, skip_special_tokens=True)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dials", default="deceive,authority")
    ap.add_argument("--langs", default="de,fr,it,zh")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--nllb-model", default="facebook/nllb-200-distilled-600M")
    ap.add_argument("--chrf-threshold", type=float, default=45.0)
    ap.add_argument("--beams", type=int, default=4)
    args = ap.parse_args(argv)

    import sacrebleu
    tok, model, dev = _load_nllb(args.nllb_model)
    print(f"loaded {args.nllb_model} on {dev}", flush=True)

    gate = {}
    for dial in args.dials.split(","):
        pairs = [json.loads(l) for l in (PAIRS_DIR / f"{dial}.jsonl").read_text().splitlines() if l.strip()][: args.limit]
        en_plus = [p["p_plus"] for p in pairs]
        en_minus = [p["p_minus"] for p in pairs]
        for lang in args.langs.split(","):
            code = LANG_CODE[lang]
            pp_L = _translate_batch(tok, model, dev, en_plus, "eng_Latn", code, args.beams)
            pm_L = _translate_batch(tok, model, dev, en_minus, "eng_Latn", code, args.beams)
            pp_b = _translate_batch(tok, model, dev, pp_L, code, "eng_Latn", args.beams)
            pm_b = _translate_batch(tok, model, dev, pm_L, code, "eng_Latn", args.beams)
            kept, chrfs = [], []
            for p, ppL, pmL, ppb, pmb in zip(pairs, pp_L, pm_L, pp_b, pm_b):
                c_plus = sacrebleu.sentence_chrf(ppb, [p["p_plus"]]).score
                c_minus = sacrebleu.sentence_chrf(pmb, [p["p_minus"]]).score
                minimal_pair_ok = (ppL.strip() != pmL.strip()
                                   and sacrebleu.sentence_chrf(ppb, [pmb]).score < 100.0)
                chrfs += [c_plus, c_minus]
                if c_plus >= args.chrf_threshold and c_minus >= args.chrf_threshold and minimal_pair_ok:
                    kept.append({"pair_id": p["pair_id"], "p_plus": ppL, "p_minus": pmL,
                                 "chrf_plus": round(c_plus, 1), "chrf_minus": round(c_minus, 1)})
            outd = OUT_DIR / lang
            outd.mkdir(parents=True, exist_ok=True)
            (outd / f"{dial}.jsonl").write_text("\n".join(json.dumps(k) for k in kept) + "\n")
            mean_chrf = sum(chrfs) / len(chrfs) if chrfs else 0.0
            gate[f"{lang}/{dial}"] = {"kept": len(kept), "total": len(pairs), "mean_chrf": round(mean_chrf, 1)}
            print(f"  {lang}/{dial}: kept {len(kept)}/{len(pairs)}  mean round-trip chrF={mean_chrf:.1f}", flush=True)

    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "crosslingual_gate.json").write_text(json.dumps(
        {"experiment": "X1_crosslingual_translate", "translator": args.nllb_model,
         "chrf_threshold": args.chrf_threshold, "beams": args.beams, "gate": gate}, indent=2))
    print(f"\nWrote {DATA/'crosslingual_gate.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
