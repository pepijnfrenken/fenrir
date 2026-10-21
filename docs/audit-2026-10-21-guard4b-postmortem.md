# Fenrir upgrade audit — post guard-4B rung (2026-10-21)

Scope: what the guard-4B rung exposed about the toolkit, and what is worth
building. Evidence file: this session's artifacts (`results4b/`, volume mirror
`/checkpoints/guard4b-abl-l32-a6.0`).

## Verdict table

| # | Upgrade | Why | Class | Priority |
|---|---|---|---|---|
| G1 | Dual-read gate support (`--read` on gates/acceptance/visibility-price) | `guard_readout.read_pair()` (L341) hardcodes the response-moderation scaffold, and `gates._guard_read_all()` (L572) calls it with no read selector — so `run_guard_gates.py`, `guard_excise.py` (its gates) and `guard_visibility_price.py` all measure **resp_mod only**. The harvest side DOES support both (`guard_probe.py --read resp_mod|user_only`, `scaffold_messages(prompt, response, read)` L75), so a dual-read rung currently harvests two directions but can only certify one read. Edition: pass `read` through `read_pair` → `_guard_read_all` → the gate trio, then report both gates per read. | capability | HIGH |
| G2 | Derive directions from BOTH native reads at the next rung | Rule 9: the edit's footprint tracks the read its direction was derived from. The 4B run derived from resp_mod only (`probe_guard4b.npz: read=resp_mod`) — the published recipe is correct but read-scoped. With G1 landed, a rung can harvest+distill+excise per read and measure the cross-read footprint in one pass. | capability | HIGH (needs G1) |
| G3 | Prompt-moderation acceptance arm in the standard gate bundle | The off40 fresh-slice repl ran resp_mod only. Before anyone repeats "the flag is gone", every rung should ship a two-read gate statement by default (rule 6: "never conclude from ONE readout"). Cheap: one extra gates pass with `--read user_only`. | instrumentation | HIGH (needs G1) |
| G4 | Fresh-slice replication runner named + scripted | We ran it ad hoc (offset 40). It should be a standard `--repl-offset N` stage anyone's rung closes with (claim-discipline rule 13). | standardization | MED |
| G5 | Cross-family matched-effect-size comparator | Rule 14 needs an α-sweep comparator that picks α by matching own-read flag rate, then reports the other read. Currently hand-run; 10-line wrapper around the excise grid. | standardization | MED |
| G6 | When integrated: guard lane recipes → `docs/guard-lane-spec.md` append a "4B scale rung" section: saturated separation profiles, α≠family-constant, unit-norm decoupling, steer grid asymmetric read | otherwise the scale knowledge lives only in the agent skill. | documentation | MED |
| G7 | README guard-lane section | README mentions guard lane only via history (line 172); the lane now has TWO families + a 4B scale rung + a published model. Deserves its own short section w/ pointer to `docs/guard-lane-spec.md`. | documentation | LOW |
| G8 | Run Guard runner tracked | `run_guard4b_modal.py` is untracked in git; the artifact-persistence pattern should be committed. | hygiene | LOW |
| G9 | HK/LM head caveat | The excise wrote 2 tensors only (o_proj+down_proj). For guard models with no lm_head tie, set `include_lm_head false` default in configs; check config default. | hygiene | LOW |

## Confirmed facts behind the audit

- `read_pair()` hardened to resp_mod: L341-350 of `guard_readout.py`.
- `scaffold_messages()` already carries the read selector: L75-85 of `guard_probe.py`
  (same file family config), so the G1 change is a pass-through, not a redesign.
- Gate aggregation (`_guard_aggregate` L605) is read-agnostic; only the read loop
  needs the parameter.
- The 4B rung's steering grid ran on the same resp_mod-only basis, so the L32
  choice is validated only for that read at 4B (0.6B measured both; footprint
  asymmetry was family- and read-specific there).

## Notes

- Nothing in the audit blocks the release: L32 a6 checkpoint is deterministic on
  its recipe and the released card scopes the claim to the response read.
- G1+G3 together change the next-guard-rung cost estimate from ~1.5 A100 h to
  ~3 h (double reads at harvest+gate), still trivial.
- Optional stretch: sentence-max "hardened reader" from the jev lane could be
  lifted into the guard-lane gate bundle (union read kills flooding cells).
  Only worth it if someone deploys a guard in anger.
