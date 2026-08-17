---
type: finding
title: "ROAD-Waymo's Constraint Arrays Are Mis-Indexed (fossils of the original ROAD) — all prior t-norm results tainted"
aliases: ["childs bug", "fossil constraint arrays", "duplex_childs bug", "mis-indexed childs"]
created: 2026-08-17
updated: 2026-08-17
sources:
  - "ROAD_Reason/experiments/exp9_joint_heterogeneous/constraints_verified.json"
  - "ROAD_Reason/tnorm_loss.py"
  - "PedestrianIntent++/ROAD_plus_plus_Baseline/modules/tnorm_loss.py"
  - "ROAD_Reason/experiments/exp2f_flat_head/eval_baseline_compat.py"
  - "ROAD_Reason/experiments/exp1_road_r/train.py"
tags: [finding, road-plusplus, t-norm, neuro-symbolic, data-bug, verified-numbers, exp1, exp1b, exp2f, exp9]
status: complete
---

# ROAD-Waymo's Constraint Arrays Are Mis-Indexed

**Discovered 2026-08-17** while extracting the constraint set for exp9. The
`duplex_childs` (39 entries) and `triplet_childs` (68) arrays shipped inside
`road_waymo_trainval_v1.1.json` do **not** decode against the JSON's own label
lists: only **20/39** and **6/68** entries correspond to valid compositions;
the rest decode to nonsense like `Car-Rev`.

## Root cause — fossils of the original ROAD

Decoded against the **original UK ROAD dataset's** label lists, all 39 duplex
entries resolve to sensible ROAD duplexes *in alphabetical order*
(`Bus-MovAway … Car-Brake … OthTL-Red … TL-Red`) — the signature of a sorted
valid-composition list generated for ROAD and copied verbatim into the
ROAD-Waymo JSON. ROAD-Waymo then changed the label lists (inserted `SmalVeh`
into agents; added `Rev`/`MovRht`/`MovLft` to actions), shifting every index
past the insertion points. `duplex_labels`/`triplet_labels` were regenerated;
the childs arrays were not. Upstream's own `datasets.py` loads these arrays as
the constraint interface (`self.childs`) for road_waymo — so the defect is in
the **dataset release**, not in local processing.

## Blast radius — every prior loss-side t-norm number

Both `tnorm_loss.py` implementations in the lab lineage (ROAD_Reason root and
the copy in the baseline clone — identical md5; lab-written, plugged into
upstream's childs interface) build the **penalty set as the complement of the
fossil arrays**. Consequence: **29/49 valid duplexes and 80/86 valid triplets
were actively penalized as "invalid" during training.** Our own reference doc
assumed the array held "the 49 valid pairs"; it holds 39, and no assertion
checked it.

| Tainted artifact | Number as recorded | Status |
|---|---|---|
| [[findings/exp1-vs-retinanet-baseline\|Exp1]] "constraints inert" (violations 0.02%, L_tnorm ≈ 1e-5) | measured against fossil set | invalid as a constraint measure |
| Exp1/baseline Gödel comparison (agent 17.76→17.01, duplex 13.44→13.62, triplet 9.17→9.37) | trained with fossil penalty set | not a valid t-norm result — gains/losses achieved while penalizing most valid compositions |
| [[findings/exp1b-fcos-detection\|Exp1b]] violation rate 0.29% | fossil set | invalid as a constraint measure |
| [[findings/exp2f-flat-head\|Exp2f]] triplet violation **80.09%** | fossil set: only 6/86 valid triplets recognized | **mostly artifact** — valid co-predictions counted as violations. Duplex stayed low (0.62%) only because the dominant pairs (Car-MovAway/Stop/MovTow…) happen to sit in the 20-pair overlap. The conf>0.9 row (88.51% on N=435) is pure fossil noise. No cached detections survive, so a corrected recompute needs a model re-run |
| ROAD-Waymo paper Table 7 (upstream t-norm baselines) | unverifiable — their logic-training code is not public | *plausibly* affected: the fossil arrays are the only machine-readable constraint source in their release |

## Corrected constraint set

`exp9_joint_heterogeneous/constraints_verified.json` — 49 duplex + 86 triplet
index tuples derived from the `duplex_labels`/`triplet_labels` **strings**
(unambiguous). Per-agent valid-duplex counts: Ped 9, Car 13, Cyc 3, Mobike 1,
SmalVeh **0**, MedVeh 10, LarVeh 4, Bus 5, EmVeh 1, TL 3. Triplets: Ped 22,
Car 31, Cyc 4, MedVeh 19, LarVeh 4, Bus 6; TL/Mobike/SmalVeh/EmVeh **0**.

## Implications

1. **The loss-side t-norm experiment has never been run correctly** in this
   lineage (and possibly not upstream either). Exp9's R2 — Moradi's
   outstanding requested experiment — becomes the first correctly-constrained
   run, now doubly motivated.
2. **Do not quote** the exp1/exp1b violation rates or the exp2f 80.09% as
   evidence about constraint behavior; the "constraints are inert" narrative
   needs re-testing against the corrected set.
3. Any code consuming `duplex_childs`/`triplet_childs` must instead derive
   from the label strings (or assert `len == 49/86` and fail loudly).
4. Mirrors [[datasets/road-plusplus|ROAD++]]'s overcounted paper statistics:
   trust the JSON's *data*, verify its *metadata*. Candidate upstream report
   to the Road-waymo-dataset maintainers (pending Moradi sign-off).

## Related

- [[directions/vlm-reasoning-layer|Approach 8]] — exp9 design context
- [[concepts/neuro-symbolic-constraints|Neuro-Symbolic Constraints]] — the constraint formalism
- [[findings/exp1-vs-retinanet-baseline|Exp1 vs RetinaNet]] | [[findings/exp1b-fcos-detection|Exp1b FCOS]] | [[findings/exp2f-flat-head|Exp2f Flat Head]] — tainted numbers annotated in place
