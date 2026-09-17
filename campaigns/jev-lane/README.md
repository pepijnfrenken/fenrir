# Campaign: Jev lane — the closed judge (TypeSafe Jev as a guard instrument)

**Status (2026-09-17): STARTED.** Smoke + first readout on the guard-lane
slice-2 test bank (160 pairs). No conclusions until the smoke passes.

## Why this lane exists

Everything in this repo is about judges and how they fail: guard-lane
abliterates an **open** generative safety guard and measures what weight
surgery does to its judgment. This lane is the opposite polarity — a judge you
cannot open. Jev (TypeSafe's System One model) is an API: there are no weights
to edit, so the only attack surface is the **state**. Its product surface
explicitly includes guardrail duty ("detect jailbreaks", hazard screening),
which makes it a fair comparison arm for the guard-lane banks.

## The three questions (agreement / calibration / robustness)

1. **Agreement** — read by the same bank the open guards read (harmful/benign
   prompt–response pairs, slice-2 test n=80+80): does Jev flag the same pairs,
   and how do its flag rate / separation compare with the open guards
   (pristine qwen3guard 78/80 · 0 FP; granite 75–80/80; ablated 10/80)?
2. **Calibration** — do its probabilities/confidence track correctness on OUR
   data (their docs say to test exactly this)? Planned; needs the bank read first.
3. **Robustness** — can the verdict be moved by state framing (dilution /
   order / prompt-tier / multi-turn), and is the damage a *threshold shift*
   (ordering survives — recoverable) or *destruction* (ordering collapses)?
   This is the ablation-analog for a model with no weights to edit. Their own
   jaggedness page states the surface: "content written to adversarially steer
   the model ... can move the answer."

## Discipline (inherited from guard-lane)

- **Instrument first:** `--smoke` (canned compliance / refusal / benign) must
  behave before any bank number is interpreted.
- **Per-item records, never aggregates alone** (`items_*.jsonl`).
- **Two reads per pair**, mirroring the guard's native reads: prompt-side and
  response-side.
- **Defensive framing:** this is guard-robustness measurement.

## Publication discipline (read before sharing anything)

TypeSafe's preview Interfaces ToS (§1(v)): **no publishing benchmarks or
performance information about the Interfaces** without their OK; no
distillation; preview access is for evaluation. Therefore:

- every raw Jev number in this lane stays **local** — `results/` is gitignored;
- findings go to TypeSafe **first** (they explicitly invite failure-mode
  reports — jaggedness page + launch post);
- `outreach-typesafe.md` holds the standing ask (send before any publication).

## Files

- `jev_client.py` — minimal System One client (vendored from the wave-5
  beer-game harness); key: env `TYPESAFE_API_KEY` or `.env.local` (gitignored).
- `jev_probe.py` — smoke + bank readout.
- `outreach-typesafe.md` — the publication/coordination note (draft).

## Run order

1. `python3 jev_probe.py --smoke` — instrument sanity (canned pairs).
2. `python3 jev_probe.py` — slice-2 test readout (default bank; `--n` for a quick pass).
3. NEXT (planned): port guard-lane's dilution cells (best-evade: benign prompt +
   250-char span first + benign flood) and the order matrix onto the Jev battery.
4. LATER: calibration curve (probability vs correctness), judge-shootout vs the
   thinking-model judge stack, elicitation-analog arms.

## What this is NOT

- Not weight surgery — Jev has no weights to abliterate; only the state moves.
- Not a product benchmark — small local banks, one model version, honest caveats.
