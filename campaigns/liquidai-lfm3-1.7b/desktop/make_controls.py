#!/usr/bin/env python3
"""Build null-direction controls for the LFM3-1.7B campaign (rung 5).

Why: rung 4 concluded "the direction is not a refusal axis" from the fact that
steering it at +/-1 destroys the decoder at BOTH signs. That inference is only
valid if a semantically empty direction at the same magnitude does NOT destroy
the decoder. That control was never run. Without it, "not sign-selective" may
just mean "this model dies at alpha 1 no matter what you add to it".

Two controls, both built from the real t320 harvest (no model run needed):

  ctrl-random : per-layer gaussian unit vector. Same layer set, same magnitude
                (alpha is absolute hidden-state units — sweep._apply_steering
                normalises d and applies -d*alpha, so stored norms do not
                matter), no relation to refusal.
  ctrl-shuf   : the real 24 vectors with the layer assignment deranged. Every
                vector and every magnitude is real; only the correspondence
                between a direction and the layer it was measured at is broken.
                Tests whether layer placement carries the effect.

Both are written in the same bundle schema steer-test already reads via
--from-directions, so the control runs through the identical code path.

Seeds are drawn once and recorded in the bundle: seed=20261113 for both.
"""

from __future__ import annotations

import json
import pathlib
import sys

import torch

D = pathlib.Path(__file__).resolve().parents[1]
SRC = D / "directions-chat-paired-t320.pt"
SEED = 20261113


def derangement(n: int, g: torch.Generator) -> list[int]:
    """A permutation of range(n) with no fixed points (so no vector stays put)."""
    while True:
        perm = torch.randperm(n, generator=g).tolist()
        if all(i != perm[i] for i in range(n)):
            return perm


def main() -> int:
    if not SRC.exists():
        print(f"FATAL: missing {SRC} — harvest rung 4 first", file=sys.stderr)
        return 2
    bundle = torch.load(SRC, map_location="cpu", weights_only=False)
    dirs = {int(k): torch.as_tensor(v).float().flatten()
            for k, v in bundle["dirs"].items()}
    layers = sorted(dirs)
    hidden = dirs[layers[0]].numel()
    print(f"source: {SRC.name} layers={len(layers)} hidden={hidden} "
          f"dir_method={bundle.get('dir_method')}")

    g = torch.Generator().manual_seed(SEED)

    # --- control 1: random unit vectors, one per layer -------------------
    rand = {}
    for li in layers:
        v = torch.randn(hidden, generator=g)
        rand[li] = v / v.norm().clamp(min=1e-8)

    # --- control 2: real vectors, layer assignment deranged -------------
    perm = derangement(len(layers), g)
    shuf = {layers[i]: dirs[layers[perm[i]]].clone()
            for i in range(len(layers))}
    print(f"derangement: {dict(zip(layers, [layers[p] for p in perm]))}")

    def write(path: pathlib.Path, d: dict, kind: str) -> None:
        payload = {
            "dirs": {str(k): v for k, v in d.items()},
            "scores": {str(k): float(bundle["scores"][str(k)])
                       for k in d if str(k) in bundle.get("scores", {})},
            "model_id": bundle.get("model_id"),
            "dir_method": f"control-{kind}",
            "probe_mode": bundle.get("probe_mode", "?"),
            "prefill": bundle.get("prefill"),
            "n_prompts": bundle.get("n_prompts"),
            "control": {"kind": kind, "seed": SEED,
                        "source": SRC.name,
                        "derangement": dict(zip(layers, [layers[p] for p in perm]))
                        if kind == "shuf" else None},
        }
        torch.save(payload, path)
        norms = [float(v.norm()) for v in d.values()]
        print(f"wrote {path.name}: {len(d)} dirs, norm range "
              f"{min(norms):.4f}..{max(norms):.4f}")

    write(D / "dirs-ctrl-random.pt", rand, "random")
    write(D / "dirs-ctrl-shuf.pt", shuf, "shuf")

    (D / "dirs-ctrl-manifest.json").write_text(json.dumps(
        {"seed": SEED, "source": SRC.name,
         "files": ["dirs-ctrl-random.pt", "dirs-ctrl-shuf.pt"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
