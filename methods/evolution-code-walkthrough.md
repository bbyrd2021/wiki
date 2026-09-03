---
type: method
title: "Code Walkthrough — the Evolution Pipeline, Stage by Stage"
aliases: [evolution-codewalk, code-walkthrough]
created: 2026-09-03
updated: 2026-09-03
sources:
  - "ROAD_Reason/experiments/exp11_yolo"
  - "ROAD_Reason/experiments/exp12_phrase_head"
tags: [method, code, walkthrough, exp11, exp12, exp13, exp14, hybrid, defense-prep]
status: complete
---

# Code Walkthrough: the Evolution, Stage by Stage

Every load-bearing block from the code that produced the results table, quoted verbatim with commentary. Read top to bottom once, then keep it open next to the source files. Paths are relative to `/data/repos/ROAD_Reason/experiments/`.

---

## 0. The shared spine (read this first)

Three conventions appear in every script. Understand them once and every file gets easier.

**The 184-column layout.** Every score vector is ordered `[agentness 1 | agent 10 | action 22 | loc 16 | duplex 49 | triplet 86]`. Scripts hard-code the boundaries:

```python
offsets = np.cumsum([0] + num_c)   # [0, 1, 11, 33, 49, 98, 184]
```

So `sig[:, :49]` is always "the primitives" (agentness+agent+action+loc), `sig[:, 49:98]` is duplex, `sig[:, 98:184]` is triplet. When you see magic slices like `[:, 49:98]`, this is the map.

**The 840x600 box space.** The 3D-RetinaNet dump stores boxes in an 840x600 pixel convention (its input resolution). YOLO stores normalized `boxes_xyxyn` in 0..1. Every script that mixes the two divides the I3D side down:

```python
BOX_W, BOX_H = 840, 600
ib = irec["boxes"].astype(np.float32) / np.array([BOX_W, BOX_H, BOX_W, BOX_H], np.float32)
```

This is a coordinate convention, not an input size. The frames themselves are 1920x1280; YOLO infers at imgsz 1280.

**The evaluator is the baseline repo's own.** No script computes mAP itself. They all build `gt_all` / `det_all` lists and hand them to the official code, which is what makes every row comparable:

```python
sys.path.insert(0, "/data/repos/PedestrianIntent++/ROAD_plus_plus_Baseline")
import modules.evaluation as baseline_eval
from modules.utils import get_individual_labels
```

Full-candidate protocol everywhere: up to 300 YOLO boxes per frame at conf 0.001, f-mAP at IoU 0.5, the whole val split.

---

## 1. The detector (feeds Stages 1 through 6)

### `exp11_yolo/train_v8x.py`

The whole training run is one call. The docstring records the replication contract and the two documented deviations, which is why this file is short on purpose:

```python
YOLO(a.model).train(
    data="/data/datasets/road_waymo_yolo/data.yaml",
    imgsz=a.imgsz, epochs=a.epochs, batch=a.batch, device=a.device,
    optimizer="SGD", lr0=0.005, lrf=0.1, close_mosaic=5, ...)
```

`lr0=0.005, lrf=0.1` approximates the ECCV report's two-phase schedule (30 epochs at 0.005 then 20 at 0.0005) as one linear decay. `close_mosaic=5` turns augmentation off for the last 5 epochs, matching the report. `imgsz=1280` is our deviation, justified by box-size stats (43% of boxes under 8px at 640).

### `exp11_yolo/eval_yolo_agent.py` (also the dump recipe)

```python
r = model.predict(IMG_DIR / f"{stem}.jpg", imgsz=1280, conf=0.001, iou=0.7,
                  max_det=300, verbose=False)
```

This one line defines "full-candidate": keep almost everything (conf 0.001), NMS at IoU 0.7, cap at 300. The dumps (`dets_v8x_best_val_fullcand.pkl`) store `boxes_xyxyn`, `conf`, and `cls` (the hard argmax class) per frame. Every later stage consumes this pickle instead of re-running YOLO.

Note what YOLO gives each box: 4 coordinates, 1 confidence, 1 hard class. It never produces 10 agent sigmoids. That is why, in every eval, the agent task is scored by masking on `cls`:

```python
for c in range(num_c[1]):
    m = ycls == c
    det_all[1][c].append(np.concatenate([yb[m], yconf[m, None]], 1))
```

A box contributes its conf to exactly one agent column and is absent from the other nine.

---

## 2. Stage 1: score transfer (`exp11_yolo/eval_hybrid_score_transfer.py`)

Zero trained parameters. The whole stage is one matching loop.

**The match.** For each frame, an IoU matrix between YOLO boxes and I3D detections, then a row-wise argmax:

```python
M = iou_matrix(yb, np.clip(ib, 0, 1))
best = M.argmax(1); bv = M[np.arange(n), best]
hit = bv >= IOU_MATCH                     # IoU >= 0.5
trans[hit] = isig[best[hit]]              # copy the matched candidate's 184 sigmoids
n_matched += int(hit.sum())
```

`trans` starts as zeros, so the miss case is implicit: a YOLO box with no I3D partner keeps zero on every transferred head. `n_matched / n_yolo` is the 40.6% match rate that indicts this stage.

**Who supplies what.** After the loop, rows are assembled with YOLO's own conf as agentness and YOLO's hard class for agent, while action/loc/duplex/triplet come from `trans`. That division of labor (boxes and agent identity from YOLO, everything else copied) is the entire stage.

**Why it fails.** Nothing here is wrong; the ceiling is structural. Scores can only flow where I3D happened to detect something, and its detections are sparse and badly localized, which is the reason it was demoted in the first place. The fix is to stop asking for its detections and start reading its features.

---

## 3. Stage 2: RoIAlign + trained head

### `exp11_yolo/cache_roi_feats.py` (the read operation)

**Training rows.** GT boxes with multi-hot targets, plus up to 8 random background boxes per frame with all-zero targets, rejection-sampled to avoid overlapping GT:

```python
w = rng.uniform(0.02, 0.35) * BOX_W
h = rng.uniform(0.02, 0.35) * BOX_H
x1 = rng.uniform(0, BOX_W - w); y1 = rng.uniform(0, BOX_H - h)
...
if (inter / np.clip(union, 1e-9, None)).max() > NEG_IOU:   # 0.3
    continue
```

A separate `--junk` mode pools at YOLO's own train-split false positives (all-zero targets): teaching the head what YOLO's mistakes look like in feature space. Both row types go into training.

**The pooling.** This is the block your Stage 2 figure draws:

```python
p3 = det._sources_cache[-1][0]                    # hook: FPN level P3, [1,256,Tp,75,105]
if Tp != T:
    p3 = F.interpolate(p3, size=(T, Hp, Wp), mode="trilinear", ...)  # time back to 8
sx, sy = Wp / float(BOX_W), Hp / float(BOX_H)     # 105/840, 75/600: pixel -> grid
b = b * torch.tensor([sx, sy, sx, sy], ...)       # scale the box by 1/8
f = roi_align(fmaps[t:t+1], rois, output_size=(1, 1), aligned=True)
```

Read it as: grab the frozen P3 map, scale the box into its stride-8 grid, average what is under the box into one 256-d vector, at the box's own frame `t`. `output_size=(1,1)` is the "one vector per box" in the figure. No matching anywhere; every box gets a vector because every location has features.

### `exp11_yolo/train_head.py` (the only trained part)

```python
head = nn.Linear(256, NUM_CLASSES).to(dev)        # 256 -> 184, that is the whole model
```

Loss is focal with per-class alpha weights (copied verbatim from exp6, the "verbatim copy convention" so numbers stay comparable):

```python
loss = ce * (1.0 - pt).pow(gamma)                 # gamma=2: down-weight easy rows
alpha_t = targets * alpha + (1 - targets) * (1 - alpha)
loss = alpha_t * loss
```

The `--lam` argument adds a Godel t-norm constraint penalty; lam=0 won (constraints are inert through a linear head, a Stage 2 lesson booked in the wiki).

### `exp11_yolo/eval_head.py` (where the gate lives)

```python
sig = torch.sigmoid(head(torch.from_numpy(f).float())).numpy()
sig_raw = sig.copy() ...                          # ungated copy, BEFORE the gate
if args.gate:
    sig = sig * yrec["conf"].astype(np.float32)[:, None]
```

The confidence gate is Stage 2's second lesson: multiply every head sigmoid by YOLO's box conf before ranking. Ungated, 300 candidates of noise collapse the ranking (action 5.3 vs 15.9 gated). Functionally the gate replaces what RetinaNet's background-trained anchors used to do. Note the order: `sig_raw` is captured first; anything that feeds a later model wants raw sigmoids, anything that feeds the evaluator wants gated ones. This ordering matters again in Stages 3, 5, and 6.

The `--derive` blocks in this file are the failed alternative your Stage 3 story needs: composing duplex/triplet by min or product over the valid pairs. Rules discard co-occurrence knowledge; the learned MLP beat both.

---

## 4. Stage 3: the stacked composition MLP (`exp11_yolo/train_comp_mlp.py`)

The pattern (identical in the crop version, so annotated once here, thoroughly, in section 6):

1. Split videos into two folds by sorted order (`i % 2`), video-level so no video leaks across folds.
2. Train a throwaway fold head on fold A, another on fold B.
3. For every row, compute primitive sigmoids using the OPPOSITE fold's head. These are out-of-fold (OOF): the MLP never trains on a prediction made by a head that saw that row.
4. Concatenate `[49 OOF primitive sigmoids | 256 features]` = 305-d, train `Linear(305,512) -> ReLU -> Linear(512,135)` against the 135 composition targets.
5. At eval, the full-data head supplies primitives and the MLP overwrites columns 49:184.

The eval-side wiring in `eval_head.py`:

```python
zin = np.concatenate([sig_raw[:, :49], f.astype(np.float32)], 1)   # RAW primitives + features
comp = torch.sigmoid(comp_mlp(torch.from_numpy(zin))).numpy()
if args.gate:
    comp = comp * yrec["conf"][:, None]                             # gate at emission
sig[:, 49:98] = comp[:, :49]; sig[:, 98:184] = comp[:, 49:135]      # overwrite compositions only
```

Three things to internalize: the MLP eats RAW sigmoids (pre-gate); the gate is applied to its OUTPUT; and it only ever touches columns 49 to 183. Action and location pass through from the head untouched, which is why those columns never change between Stage 2 and Stage 3 in the results table.

Why the output is only valid compositions: the 135 outputs are indexed by the dataset's `duplex_labels`/`triplet_labels` vocabularies, so an invalid agent-action pair has no output column to receive probability. Structural constraint satisfaction, no penalty term needed.

---

## 5. Stage 4: the contrastive phrase head (exp12)

### `exp12_phrase_head/embed_phrases.py` (build-time, runs once)

```python
order, texts = ["agentness"], [ph["agentness"]]
for h, labels in (("agent", d["agent_labels"]), ("action", d["action_labels"]), ...):
    for n in labels:
        order.append(f"{h}:{n}"); texts.append(ph[h][n])
assert len(texts) == 184
```

The 184 phrases are assembled in exactly the 184-column order (the assert is the guarantee), pushed once through the frozen CLIP_S text tower, L2-normalized, and saved as `phrase_embeds.pt`. This is the answer to "where do the phrases come from at inference": they do not; this matrix is a stored constant, about 380KB. `PHRASES` env var swaps vocabularies (that is how the deterministic v2 provenance ablation ran without touching code).

### `exp12_phrase_head/crop_full/cache_crop_feats_h200.py` (the substrate)

**Rows.** Val rows are exactly the YOLO full-candidate boxes. Train rows are GT + 8 random negatives + up to 16 YOLO junk boxes, same philosophy as Stage 2's cache but crop-based:

```python
bx = np.concatenate([gt, negs, junk], 0)
tg = np.zeros((bx.shape[0], N_OUT), np.float16)
if gt.shape[0]:
    tg[: gt.shape[0]] = np.stack(mh_rows)      # only GT rows get positive targets
```

**Crops.** Each box is padded 2x around its center, cut from 8 real frames (t-3..t+4), resized to 224 via `roi_align` on the full frames (one batched op for all boxes at all 8 frames):

```python
rois = torch.cat([batch_idx, boxes_px.repeat(8, 1)], 1)     # [8n, 5]
crops = roi_align(frames8, rois, output_size=(224, 224), ...)
crops = crops.view(8, n, 3, 224, 224).permute(1, 0, 2, 3, 4)  # -> [n, 8, 3, 224, 224]
```

That permute is why the input tensor in the figures reads `[n, 8, 3, 224, 224]` (time before channels). The frozen InternVideo2-S encodes each crop; a hook takes the keyframe token slice, mean-pooled to 1024-d. One 1024-vector per box, cached to disk.

(Post coordinate-probe, the cacher also persists `boxes` per key. The original campaign did not, which is what forced the probe re-cache.)

### `exp12_phrase_head/train_clip_head.py` (both heads live here)

The flat head is `nn.Linear(1024, 184)`: learned classifier columns, the control. The phrase head is the mechanism:

```python
class PhraseHead(nn.Module):
    def __init__(self, embeds, in_dim=1024):
        self.proj = nn.Linear(in_dim, embeds.shape[1])           # the ONLY trained part
        self.register_buffer("P", F.normalize(embeds.float(), dim=-1))  # frozen phrase matrix
        self.log_tau = nn.Parameter(torch.tensor(2.3))           # learned temperature, tau ~ 10
        self.bias = nn.Parameter(torch.zeros(embeds.shape[0]))   # learned per-class bias
    def forward(self, x):
        z = F.normalize(self.proj(x), dim=-1)
        return z @ self.P.t() * self.log_tau.exp() + self.bias   # scaled cosine + bias
```

Read the forward as: project the crop feature into the text space, normalize, cosine against all 184 phrase embeddings at once, scale, shift. `register_buffer` is what "frozen" means in code: `P` is saved with the model but receives no gradients. The trained parameters are the projection, one temperature, and 184 biases. Classification capacity lives in language geometry, not learned columns; that is the whole thesis mechanism in nine lines.

Both heads train with the same focal-plus-alphas loss as everything since exp6. There is no contrastive (InfoNCE) loss anywhere: multi-label boxes want BCE, and the contrastive geometry comes free from InternVideo2's pretraining.

---

## 6. Stage 5: the record (`crop_full/train_comp_mlp_crop.py` + `eval_comb.py`)

### Training with out-of-fold stacking

```python
vids = sorted(set(V.tolist()))
foldA = set(v for i, v in enumerate(vids) if i % 2 == 0)   # video-level 2-fold split
maskA = np.isin(V, list(foldA))
```

Fold membership is by VIDEO, not by row. Rows from the same video always land in the same fold, so temporal near-duplicates can never leak across the boundary.

```python
hA = train_fold(maskA, "A"); hB = train_fold(~maskA, "B")   # throwaway fold heads
for rows, h in ((maskA, hB), (~maskA, hA)):                  # OOF: OPPOSITE fold's head
    Z[b] = torch.sigmoid(h(Xt[b].to(dev)))[:, :N_PRIM].cpu().numpy()
```

The tuple order `(maskA, hB)` is the entire OOF idea in one line: fold A rows are scored by the head trained on fold B. If a committee member asks about leakage in the stack, this is the line to point at.

```python
Zin = torch.from_numpy(np.concatenate([Z, X], 1))            # 49 + 1024 = 1073
mlp = nn.Sequential(nn.Linear(N_PRIM + FEAT_DIM, 512), nn.ReLU(), nn.Linear(512, N_COMP))
```

Same MLP shape as Stage 3, wider input (crop features instead of RoI features). The saved checkpoint records `head_ckpt`, which the eval asserts against, so you can never accidentally pair a MLP with the wrong head.

### `eval_comb.py` (the assembly line, one frame at a time)

```python
sig = torch.sigmoid(head(torch.from_numpy(f32)))...          # flat head, all 184
sig_raw = sig.copy() if comp is not None else None           # capture BEFORE gating
sig = sig * yrec["conf"].astype(np.float32)[:, None]         # gate for emission
zin = torch.from_numpy(np.concatenate([sig_raw[:, :49], f32], 1))
c = torch.sigmoid(comp(zin)).numpy() * yrec["conf"][:, None] # MLP output, then gate
sig[:, 49:98] = c[:, :49]; sig[:, 98:184] = c[:, 49:135]     # overwrite compositions
```

Trace one box through it: features -> 184 raw sigmoids -> raw primitives + features into the MLP -> 135 composition scores -> everything gated by YOLO conf -> compositions overwritten. Action/location reach the evaluator from the gated flat sigmoids; agentness/agent come from YOLO's conf and hard class further down. This is why the record's action/location are bit-identical to the flat head alone: the MLP never writes columns 0 to 48.

---

## 7. Stage 6: the fusion (`crop_full/train_comp_mlp_fusion.py` + `eval_comb.py --phrase-ckpt`)

Training doubles the fold-head work:

```python
fA = train_fold(maskA, "flat", "A");   fB = train_fold(~maskA, "flat", "B")
pA = train_fold(maskA, "phrase", "A"); pB = train_fold(~maskA, "phrase", "B")
for rows, hf, hp in ((maskA, fB, pB), (~maskA, fA, pA)):     # both heads OOF, same folds
    Zp[b] = torch.sigmoid(hf(xb))[:, :N_PRIM]                # flat primitives [49]
    Zc[b] = torch.sigmoid(hp(xb))[:, N_PRIM:184]             # phrase COMPOSITIONS [135]
Zin = np.concatenate([Zp, Zc, X], 1)                         # 49 + 135 + 1024 = 1208
```

Note which slice each head contributes: the flat head gives its primitives (what it is good at), the phrase head gives its compositions (what it is good at, especially on the tail). The MLP learns when to trust which.

Eval side, the fusion branch:

```python
psig_raw = torch.sigmoid(phead(torch.from_numpy(f32))).numpy()      # raw, never gated
zin = np.concatenate([sig_raw[:, :49], psig_raw[:, 49:184], f32], 1)
```

Both sigmoid blocks are pre-gate (`sig_raw`, `psig_raw`), matching training exactly. The consistency asserts above this block check `in_dim == 1208` and that the MLP checkpoint was trained against these exact head checkpoints. Result: best triplet tail at every cutoff, minus 0.15 average triplet.

---

## 8. The coordinate probe (`crop_full/train_head_coord.py`, for completeness)

```python
def coordfeat(bx):
    x1, y1, x2, y2 = bx[:, 0], bx[:, 1], bx[:, 2], bx[:, 3]
    return np.stack([(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1], 1)   # xyxyn -> cx,cy,w,h
```

Two heads, identical rows, identical seed, one difference: 1024 vs 1024+4 input. The paired design is the point; anything that differs between the two runs is attributable to the four coordinates. Verdict booked in the wiki: position lifts action/duplex/triplet, not location.

---

## 9. Cross-cutting reading tips

- **Raw vs gated is the most repeated trap.** Anything feeding a MODEL uses raw sigmoids; anything feeding the EVALUATOR is gated by YOLO conf. Every `*_raw = sig.copy()` line marks the boundary.
- **Nothing computes its own metric.** All roads end in `baseline_eval.evaluate(...)` from the official repo. If a number looks weird, suspect row assembly, never the metric.
- **Frozen means register_buffer or .eval() + no_grad.** Search for those to audit what trains.
- **Every stack is OOF by video.** Search `foldA` to find the split; the opposite-fold tuple `((maskA, hB), (~maskA, hA))` is the leakage guard.
- **Determinism convention:** `torch.manual_seed(0)` before every trained module, `default_rng(42)` (or 0/1) for row sampling. The one place this bit us: cache row RNG plus preemption resume made train-row boxes unreconstructable, hence the probe re-cache.


## Related

- [[findings/hybrid-evolution-narrative]] — the results narrative these scripts produced
- [[findings/exp12-crop-full-record]] — Exp13/Exp14 result booking
- [[methods/stacked-composition-mlp]] — the engine, conceptually
- [[findings/exp12-phrase-head-attribution]] — attribution + coordinate probe
