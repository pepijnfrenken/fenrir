# LFM3-1.7B abliteration campaign — desktop run report

Run: 2026-11-13, desktop `DESKTOP-9Q9U7PV` (WSL Ubuntu-24.04, user `pino`), repo `~/absolver`
at HEAD `6358b0d` (clean tree), GPU RTX 3060 Ti / driver 561.17 / 8192 MiB.

**Outcome: STOPPED at phase 1 (`inspect`), rc=1.** No model work happened: no directions, no
pristine baseline, no steer gate, no weight edit, no gate bundles. There are no real gate numbers
in this campaign yet. The cause is a config/schema contract violation in the repo at the exact
commit under test, reproduced without touching the GPU (see Blocker).

## Per-phase rc and wall time

Launched 17:42:33Z. Transport: `ssh -o BatchMode=yes 100.122.67.126 'wsl.exe -d Ubuntu-24.04 -u pino -- bash -s' < run_campaign.sh`.

| phase | rc | wall time | log | done marker |
|---|---|---|---|---|
| setup | 0 | 148 s (17:42:33Z → 17:45:01Z) | `desktop/logs/setup.log` | not written (see Defect 2) |
| inspect | 1 | ~0.1 s (17:45:01.99Z → 17:45:02.09Z) | `desktop/logs/inspect.log` | not written |
| directions | — not run | — | — | — |
| baseline | — not run | — | — | — |
| steer | — not run | — | — | — |
| abl | — not run | — | — | — |
| collect | — not run | — | — | — |

Runner output (verbatim, from the launch stream):

```
== setup: 2026-11-13T17:42:33Z
== setup rc=0
== inspect: 2026-11-13T17:45:01Z
== inspect rc=1
!! stopping at p1_inspect (rc=1)
```

Setup was green — venv 6.9 GB, CUDA visible:

```
torch 2.14.1+cu126 | cuda True | NVIDIA GeForce RTX 3060 Ti
transformers 5.23.0
```

## Blocker

`models/lfm3-1.7b.yaml` line 49 sets `gate_first_token_kl_max: 0.1`. `ModelConfig` (`config.py`)
has **no such field** — it accepts `gate_kl_max: float = 0.10` (`config.py:259`) and
`gate_first_token_kl_enabled: bool = True` (`config.py:267`). Pydantic runs with
`extra=forbidden`, so config load raises before any model is opened. `gates.py:376` reads the
threshold as `getattr(cfg, "gate_kl_max", 0.1)`, i.e. the schema name is authoritative and the
yaml key is simply wrong.

Verbatim tail of the failing phase log (`desktop/logs/inspect.log`):

```
    raise SystemExit(main())
                     ^^^^^^
  File "/home/pino/absolver/harness/abl.py", line 1167, in main
    return args.fn(args)
           ^^^^^^^^^^^^^
  File "/home/pino/absolver/harness/abl.py", line 1101, in <lambda>
    p.set_defaults(fn=lambda a: cmd_inspect(a.config))
                                ^^^^^^^^^^^^^^^^^^^^^
  File "/home/pino/absolver/harness/abl.py", line 178, in cmd_inspect
    cfg = _load_cfg(config_path)
          ^^^^^^^^^^^^^^^^^^^^^^
  File "/home/pino/absolver/harness/abl.py", line 82, in _load_cfg
    return load_config(config_path)
           ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/pino/absolver/config.py", line 493, in load_config
    raise ValueError(
ValueError: Config validation failed for /home/pino/absolver/models/lfm3-1.7b.yaml.
Schema hint: ModelConfig fields are documented in config.py.
1 validation error for ModelConfig
gate_first_token_kl_max
  Extra inputs are not permitted [type=extra_forbidden, input_value=0.1, input_type=float]
    For further information visit https://errors.pydantic.dev/2.14/v/extra_forbidden
```

Isolated reproduction on the desktop (no GPU, no harness, config load only):

```
$ .venv/bin/python -c "from config import load_config; load_config('models/lfm3-1.7b.yaml')"
ValueError
Config validation failed for /home/pino/absolver/models/lfm3-1.7b.yaml.
... 1 validation error for ModelConfig
gate_first_token_kl_max
  Extra inputs are not permitted [type=extra_forbidden, input_value=0.1, input_type=float]
```

Correct name confirmed against the other campaign config in-tree: `models/qwen2.5-1.5b-instruct.yaml:100`
reads `gate_kl_max: 0.10`.

### Minimal fix (NOT APPLIED — outside this lane's authority)

`models/lfm3-1.7b.yaml:49`

```diff
-gate_first_token_kl_max: 0.1
+gate_kl_max: 0.1
```

Semantics unchanged (same 0.1 threshold; the schema default is 0.10). No alpha, layer, weight,
method or other threshold touches. Not applied because this lane is forbidden to edit the config
and forbidden to edit files outside `campaigns/liquidai-lfm3-1.7b/desktop/` on the hub. The same
edit is needed on whichever side launches next; the desktop clone is a clean checkout of the hub,
so fix at `6358b0d`'s successor and re-clone/`git pull` on the desktop.

### Defect 2 — skip-if-done is inert at this commit

`run_campaign.sh:28` defines `mark()` but it is **never called**; `done_()` only tests
`$L/$name.done` and nothing ever writes those files. Evidence: after `setup` returned rc=0, the log
directory holds no marker at all —

```
logs/ birth=2026-11-13 17:42:33Z
  setup.log   80 bytes   mtime 17:45:01Z
  inspect.log 1991 bytes mtime 17:45:02Z
```

Consequence: the documented "relaunching after a dropped ssh is safe and idempotent" does not
hold — a relaunch re-runs `p0_setup` (pip re-resolve, minutes) and every later phase, and would
re-run `p3_baseline`, overwriting `collect-pristine/bundle.json` and its transcript with a second
draw. Fix is one line inside `phase()`: `[ "$rc" -eq 0 ] && mark "$name"` after the `tee`
(`mark` must not fire on rc≠0). Needs `mark` to run before `return`.

## Gate table

No bundle exists for either target, so every cell is `not measured`. `ls campaigns/liquidai-lfm3-1.7b`
on the desktop shows only `README.md` and `desktop/` — no `collect-pristine/`, no
`collect-mpoa-a2.0-all24/`, no `bundle.json`, no `transcript.json`.

| gate | collect-pristine | collect-mpoa-a2.0-all24 |
|---|---|---|
| eval_pass | not measured | not measured |
| refusal (n/N) | not measured | not measured |
| coherence | not measured | not measured |
| degeneracy | not measured | not measured |
| capability | not measured | not measured |
| ppl_increase | not measured | not measured |
| first_token_kl | not measured | not measured |
| baseline_sanity | not measured | not measured |

## Steer test

Not run. Effect alphas: **not measured**. `--require-effect` verdict: **not measured** (the phase
never started, so no `steer-test-*.json` exists either).

## Inspect profile — three most informative lines

Not measured. `cmd_inspect` died in `_load_cfg` before loading the model, so it printed no
`arch=`/`layers=` line, no weight audit, no per-layer profile, no separation ranking. The
architecture claims in the config header (`24 layers`, `6 conv + 5 full_attention + 5
sliding_attention + 8 conv`, hidden 2048) remain **unverified by this run** — do not cite them as
measured.

## State left on the desktop

- Venv exists and is usable: `~/absolver/.venv` (6.9 GB, torch 2.14.1+cu126, transformers 5.23.0,
  CUDA available on the 3060 Ti).
- `desktop/logs/` holds `setup.log` and `inspect.log` only; no `.done` markers, so a corrected
  relaunch starts from `setup` (pip will hit warm caches).
- No files edited on the desktop. Hub tree untouched by this lane (`git status` already showed a
  pre-existing ` M behavior.py` and untracked `results4b/` before the run — not mine).

## Exact next step

1. Apply the one-line yaml fix above at the next commit, plus the `mark "$name"` fix in
   `phase()`.
2. `git pull` on the desktop (`~/absolver`), confirm `git rev-parse --short HEAD`.
3. Relaunch verbatim:
   `ssh -o BatchMode=yes 100.122.67.126 'wsl.exe -d Ubuntu-24.04 -u pino -- bash -s' < campaigns/liquidai-lfm3-1.7b/desktop/run_campaign.sh`
4. The first evidence to check is `desktop/logs/inspect.log` — it must print
   `arch=dense layers=24 hidden=2048`, then a weight audit with **no `MISSING`** lines and an
   out-projection coverage count of 2 per layer (attn layers resolve `o_proj`+`w2`, conv layers
   resolve `conv_out`+`w2`, 24 layers → expect 48 `resolved`); a `MISSING` on any `w2` or a
   `conv out_proj NOT 2D square` line means the claimed trio is not projectable and the campaign
   must not proceed. Then read `collect-pristine/bundle.json` `baseline_sanity`.
