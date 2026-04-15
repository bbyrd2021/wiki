---
type: method
title: "Experiment 1 Code Explainer: Qwen2.5-VL on ROAD-R"
aliases: ["exp1_road_r explainer", "Qwen ROAD-R code walkthrough", "VLM code explainer"]
created: 2026-04-15
updated: 2026-04-15
sources:
  - "/data/repos/ROAD_Reason/experiments/exp1_road_r/model.py"
  - "/data/repos/ROAD_Reason/experiments/exp1_road_r/train.py"
  - "/data/repos/ROAD_Reason/experiments/exp1_road_r/dataset.py"
  - "/data/repos/ROAD_Reason/experiments/exp1_road_r/losses.py"
  - "/data/repos/ROAD_Reason/experiments/exp1_road_r/config.py"
tags: [method, road-reason, qwen, vlm, lora, training, beginner-friendly]
status: draft
---

# Experiment 1 Code Explainer: Qwen2.5-VL on ROAD-R

This note explains the current `exp1_road_r` code at a beginner-friendly level for someone who has built shallow neural networks before, but has not yet worked deeply with LLMs or VLMs.

The goal is not only to explain what each file does, but also to explain the **abstraction level** of the model: what is frozen, what is trainable, what counts as the backbone, what the loss is doing, and how LoRA would fit in later.

---

## The Big Picture

The current experiment is **not** training a whole VLM from scratch.

It is doing something much more practical:

1. Start from a very large pretrained **vision-language model**: `Qwen2.5-VL`
2. Throw away most of the language-generation part for this experiment
3. Keep the **vision encoder**
4. Use that encoder to turn images into features
5. Add small trainable heads on top of those features
6. Train only those smaller parts for the ROAD-Waymo label prediction task

So the current experiment is best understood as:

**Pretrained visual backbone + lightweight task-specific heads**

This is much closer to transfer learning than to training a foundation model from scratch.

---

## What Is a VLM?

A **VLM** is a **Vision-Language Model**.

At a high level, it combines:

- a **vision encoder**
  - turns images or video frames into vectors
- a **language model**
  - turns token sequences into text predictions
- a **bridge / projector**
  - maps vision features into a representation the language model can use

Very roughly:

```text
image/video -> vision encoder -> visual features -> projector -> language model -> text
```

Examples:

- CLIP
- LLaVA
- Qwen2.5-VL
- SmolVLM

In your current `exp1_road_r` code, you are **not using the full text-generation path**. You are only using the **visual encoder part** of Qwen2.5-VL and putting classification heads on top.

That means this experiment is more like:

```text
frames -> Qwen vision encoder -> visual feature maps -> ROI pooling -> temporal linking -> label heads
```

---

## What Is an LLM?

A **Large Language Model** is usually a Transformer trained to predict the next token.

At a high level:

```text
previous tokens -> transformer blocks -> next-token probabilities
```

Examples:

- GPT models
- Llama
- Qwen text model

The reason VLMs are related is that many VLMs are really:

- a visual encoder
- plus an LLM that has been adapted to consume visual information

So a VLM is often an LLM-based system with extra visual machinery attached.

---

## What the Current Experiment Is Trying to Learn

The experiment in `/data/repos/ROAD_Reason/experiments/exp1_road_r/` is doing:

- input: short clip of ROAD-Waymo frames
- output: multi-label predictions for each annotated agent

The predicted labels are:

- agent
- action
- location
- duplex
- triplet

Important:

- it is **not learning detection boxes**
- it is using **ground-truth boxes**
- it is therefore a **classification on top of GT regions** experiment

That makes this experiment much cheaper and simpler than full detection training.

---

## File-by-File Overview

### `config.py`

This file is the experiment's control panel.

It defines:

- where data lives
- which Qwen model to use
- clip length
- class counts
- learning rate
- number of epochs
- t-norm settings

This is the equivalent of your hyperparameter block in a smaller neural net project.

Most important entries:

- `MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"`
- `FREEZE_VIT = True`
- `CLIP_LEN = 8`
- `N_AGENTS = 10`
- `N_ACTIONS = 22`
- `N_LOCS = 16`
- `N_DUPLEXES = 49`
- `N_TRIPLETS = 86`
- `TNORM_TYPE = "lukasiewicz"`

The most conceptually important one is:

- `FREEZE_VIT = True`

This means the huge pretrained visual encoder is being treated as a fixed feature extractor.

---

### `dataset.py`

This file builds the training samples.

Its job is:

1. open the ROAD-Waymo annotation JSON
2. choose clips of `CLIP_LEN` consecutive annotated frames
3. load each frame image
4. convert frame annotations into tensors

Each sample returned by the dataset is:

```text
([PIL frame 1, ..., PIL frame T], [target dict 1, ..., target dict T])
```

Each target dict contains:

- `boxes`
- `agent`
- `action`
- `loc`
- `duplex`
- `triplet`

The label tensors are **multi-hot**, not one-hot.

Why?

Because a box can have multiple labels simultaneously, especially for:

- actions
- locations
- composite labels

This is why the heads later use **sigmoid + BCE**, not softmax + cross-entropy.

If you come from shallow classification models:

- **softmax** means "exactly one class"
- **sigmoid per output** means "each class is independently on/off"

This experiment uses the second case.

---

### `model.py`

This file defines the model architecture.

It has four conceptual stages:

1. Qwen visual feature extraction
2. ROI pooling over GT boxes
3. cross-frame linking / context
4. classification heads

We can write the whole model like this:

```text
frames
  -> Qwen vision encoder
  -> spatial feature maps per frame
  -> ROI pooling at GT boxes
  -> one feature per annotated agent
  -> temporal attention across frames
  -> 5 classification heads
```

This is the core of the experiment.

---

### `losses.py`

This file defines the training objective.

There are two pieces:

1. `L_cls`
   - standard binary classification loss across all five heads
2. `L_tnorm`
   - logic-constraint violation loss using ROAD-Waymo validity rules

Then:

```text
L_total = L_cls + L_tnorm
```

So training is not only asking:

- "predict the labels correctly"

It is also asking:

- "do not predict logically invalid label combinations"

This is the neuro-symbolic part of the experiment.

---

### `train.py`

This file is the training loop.

Its job is:

1. load config and arguments
2. build dataset + dataloaders
3. load processor
4. load model
5. build optimizer and scheduler
6. run epoch loop
7. compute train loss and val loss
8. save checkpoints and metrics

If you have trained shallow nets before, this file should feel most familiar.

The biggest differences are:

- the feature extractor is massive and pretrained
- the data objects are much more complicated
- the forward pass includes image processing and region pooling
- the loss is multi-part

---

## The Model in Detail

## 1. `QwenViTExtractor`

This class loads the visual part of `Qwen2.5-VL`.

In `model.py`, the key move is:

- load `Qwen2_5_VLForConditionalGeneration`
- keep only `full.model.visual`
- delete the rest

So this experiment is intentionally discarding:

- the LLM head
- most of the text-generation machinery

and keeping:

- the visual transformer encoder

This is important conceptually.

You are **not fine-tuning Qwen as a full chat model** here.
You are using **Qwen's image backbone as a pretrained computer vision encoder**.

That is why this experiment is computationally manageable.

### What comes out of the visual encoder?

For each frame, the extractor returns a spatial token map:

```text
[H', W', D]
```

where:

- `H'` and `W'` are downsampled spatial dimensions
- `D` is the feature dimension

Think of this as a learned image feature grid.

This is similar in spirit to CNN feature maps, except the features come from a transformer-based visual encoder.

---

## 2. `ROIAveragePool`

You already have ground-truth boxes for agents in each frame.

So instead of asking the model to detect objects from scratch, you:

- take the feature map from the visual encoder
- look up the region corresponding to each GT box
- average the features inside that region

This produces one feature vector per annotated agent box.

Very roughly:

```text
feature map + GT box -> one vector for that box
```

This is analogous to classical ROI pooling in detection models, but simplified as region averaging over token features.

Why do this?

Because now each agent gets its own compact learned representation.

---

## 3. `TubeLinkingModule`

This is the temporal piece.

At this point, you have:

- frame 1: feature vectors for its agents
- frame 2: feature vectors for its agents
- ...
- frame T: feature vectors for its agents

The code concatenates all those agent features across the clip, then applies a single **multi-head self-attention** layer across them.

So instead of treating each box independently, the model lets them exchange information across time.

This helps the model learn things like:

- this pedestrian in frame 3 is probably the same agent as in frame 2
- this box's label should depend on neighboring frames
- temporal context helps action/location prediction

You can think of this as a very lightweight temporal reasoning module.

It is much smaller than the Qwen visual encoder, and it is trainable.

---

## 4. `ClassificationHeads`

These are the final label predictors.

There are five linear heads:

- agent
- action
- location
- duplex
- triplet

Each one is:

```text
Linear(D -> num_classes) -> sigmoid
```

This means:

- each class gets an independent probability
- multiple classes can be active at once

This is why the loss uses binary cross-entropy instead of softmax cross-entropy.

---

## What Is Frozen and What Is Trainable?

This is the most important training abstraction in the current code.

### Frozen

- Qwen visual encoder parameters

Because `FREEZE_VIT = True`, the giant pretrained encoder is treated as fixed.

### Trainable

- `TubeLinkingModule`
- `ClassificationHeads`

Potentially later:

- LoRA adapters inside the visual encoder

So the learning burden is currently placed on the smaller task-specific layers.

This is a standard transfer-learning strategy:

- use a strong pretrained backbone
- train only lightweight heads first
- unfreeze or adapt more later if needed

---

## Why This Is Not "Training a VLM from Scratch"

If you have only trained shallow networks before, it helps to place this experiment on a spectrum:

### From scratch

You initialize everything randomly:

- vision encoder
- temporal layers
- output heads

Then train everything on your dataset.

This is usually impossible for a 7B-scale VLM on a normal workstation.

### Transfer learning

You start from a pretrained model and adapt it.

That is what this experiment does.

### Parameter-efficient fine-tuning

You avoid touching most backbone weights and instead add small trainable modules.

That is what LoRA is for.

This experiment is currently somewhere between transfer learning and parameter-efficient fine-tuning:

- frozen large backbone
- small trainable modules on top
- LoRA planned but not yet active

---

## The Training Loop in Familiar Terms

If you have trained a small neural net before, you already know the skeleton:

```text
for epoch:
    for batch:
        y_pred = model(x)
        loss = criterion(y_pred, y_true)
        loss.backward()
        optimizer.step()
```

The current code is the same idea, just with more complex data and a more complex model.

### Step 1: preprocess the images

`preprocess_clip()` uses the Qwen image processor to convert a list of PIL frames into tensors:

- `pixel_values`
- `image_grid_thw`

These are what the Qwen visual module expects.

### Step 2: stack targets

`stack_targets()` combines the per-frame labels into one stacked tensor dictionary.

### Step 3: forward pass

`preds = model(pixel_values, image_grid_thw, boxes_per_frame)`

### Step 4: compute loss

`loss_fn(preds, targets)` returns:

- `L_cls`
- `L_tnorm`
- `L_total`

### Step 5: backward pass

If training:

- zero gradients
- `loss.backward()`
- clip gradients
- `optimizer.step()`

### Step 6: scheduler + logging

After the epoch:

- compute validation loss
- log metrics
- save latest checkpoint
- save best checkpoint

So conceptually, the loop is exactly the same as what you already know.
The model and data pipeline are what changed.

---

## What Is BCE Doing Here?

`ROADClassificationLoss` uses **binary cross-entropy** on each head.

This makes sense because the outputs are multi-label.

Example:

A box might simultaneously have:

- `Ped`
- `Wait2X`
- `RhtPav`

Those are not mutually exclusive in the way "cat vs dog vs bird" would be.

So the model predicts many independent yes/no values instead of one mutually exclusive class.

---

## What Is the T-Norm Loss Doing?

The t-norm loss is the logic layer.

ROAD-Waymo has valid and invalid combinations.

Examples:

- some agent-action pairs are valid
- some are invalid
- some agent-action-location triples are valid
- some are invalid

The t-norm loss penalizes the model when it assigns high probabilities to combinations that violate those rules.

This is why `losses.py` builds a flat vector consisting of:

- `agentness`
- agent probabilities
- action probabilities
- location probabilities

and passes that to `TNormConstraintLoss`.

So:

- `L_cls` says "fit the labels"
- `L_tnorm` says "respect the ontology"

This is the core ROAD-R idea.

---

## What Is `agentness`?

In this experiment, `agentness` is set to `1.0` for all instances in the t-norm loss path.

Why?

Because this experiment is using **ground-truth boxes**.

So every ROI already corresponds to a real annotated agent.

In a full detector, `agentness` would be:

- "is there any object/agent here at all?"

But here that question is already answered by the dataset.

---

## Why the LLM Is Not Used in This Experiment

Even though the backbone model is a VLM, the current experiment is not generating text.

That means:

- no prompts
- no decoding
- no token-level language loss

The code only uses:

- the image processor
- the visual encoder

So it is fair to say:

**Experiment 1 currently treats Qwen2.5-VL more like a pretrained vision backbone than like a full conversational VLM.**

That is not wrong. It is a common and sensible strategy.

---

## What LoRA Is

LoRA stands for **Low-Rank Adaptation**.

It is a parameter-efficient fine-tuning method.

Instead of updating a huge weight matrix `W` directly, LoRA learns a small low-rank update:

```text
W' = W + A B
```

where:

- `W` is the original frozen pretrained weight
- `A` and `B` are small trainable matrices
- `A B` has low rank, so it is much cheaper to train

The core idea:

- keep the big model mostly frozen
- train a small correction on top

Why this is useful:

- much less VRAM than full fine-tuning
- faster training
- less risk of destroying pretrained knowledge
- easy to save/merge adapter weights

---

## Intuition for LoRA If You Know Gradient Descent

Suppose in a shallow net you would normally update all weights:

```text
W <- W - lr * dL/dW
```

With LoRA, you instead say:

- keep `W` fixed
- learn only a small structured delta `ΔW = AB`

So optimization happens in a lower-dimensional subspace.

This is why it is called "low-rank" adaptation.

You are restricting the kind of changes the model can make.

That sounds limiting, but in practice it often works very well.

---

## Where LoRA Would Fit in This Code

Right now in `model.py`, LoRA is **mentioned but not activated**.

The comment says:

- `peft` is not currently installed
- later you could wrap the visual module with `get_peft_model()`

Conceptually, LoRA would go into the pretrained Qwen visual encoder.

So instead of:

- freezing the whole vision encoder completely

you would do:

- freeze the original weights
- add LoRA adapters to selected transformer layers
- train those adapters

This gives you a middle ground between:

- fully frozen backbone
- full fine-tuning

For your workstation, that is exactly the kind of tradeoff you want.

---

## How Full LLM/VLM Training Usually Works

It helps to know where your current code sits compared to full-scale training.

## Full LLM training from scratch

Usually means:

- random initialization
- giant token dataset
- next-token prediction loss
- all transformer weights trainable
- enormous compute

This is far outside normal workstation research.

## Full VLM training from scratch

Usually means:

- train visual encoder
- train projector
- train language model
- often use huge image-text/video-text corpora

Again, extremely expensive.

## Fine-tuning

Usually means:

- load pretrained model
- update some or all weights on your task

## Parameter-efficient fine-tuning

Usually means:

- load pretrained model
- freeze most weights
- add LoRA or small task modules
- train only those

Your current code is closest to:

**pretrained VLM visual encoder + small task-specific modules + future LoRA option**

That is a realistic research workflow.

---

## Why This Design Is Sensible for You

This experiment is a good stepping stone because it teaches the right abstractions:

1. how to reuse a pretrained backbone
2. how to build a dataset pipeline for clips and structured labels
3. how to attach custom heads to a foundation model
4. how to combine standard loss with logic-constrained loss
5. how to later introduce LoRA without rewriting the whole system

So even though it is not full text-generation VLM fine-tuning yet, it is still a very good practical entry point.

---

## Mental Model to Keep in Your Head

If you want one simple mental model for the current code, use this:

```text
Qwen2.5-VL pretrained vision backbone
    +
ROI feature extraction from GT boxes
    +
small temporal attention block
    +
five sigmoid label heads
    +
BCE loss + logical consistency loss
```

Or even shorter:

```text
big frozen feature extractor + small trainable prediction layers
```

That is the right abstraction level for understanding this experiment.

---

## What to Read Next

After this note, the most useful next code files to study are:

1. `/data/repos/ROAD_Reason/experiments/exp1_road_r/model.py`
   - understand the data flow through the model
2. `/data/repos/ROAD_Reason/experiments/exp1_road_r/train.py`
   - understand the training loop
3. `/data/repos/ROAD_Reason/experiments/exp1_road_r/losses.py`
   - understand how standard supervision and symbolic constraints are combined
4. `/data/repos/ROAD_Reason/tnorm_loss.py`
   - understand how the actual logic loss is implemented

---

## Short Summary

- The current experiment is **not** training a VLM from scratch.
- It uses **Qwen2.5-VL's visual encoder** as a pretrained backbone.
- The language model part is mostly bypassed in this experiment.
- GT boxes are used, so this is **classification over known regions**, not full detection.
- The trainable parts are mainly:
  - ROI-pooled temporal linking
  - classification heads
- The loss is:
  - binary cross-entropy for labels
  - plus t-norm logic loss for valid combinations
- LoRA is the natural next step if you want to adapt the backbone cheaply without full fine-tuning.

---

## Related

- [[projects/road-reason|ROAD_Reason]]
- [[directions/qwen-multitask-finetuning|Qwen2.5-VL Multi-Task Fine-Tuning]]
- [[methods/qwen25-vl-multitask|Qwen2.5-VL Multi-Task Architecture]]
- [[concepts/neuro-symbolic-constraints|Neuro-Symbolic Constraints]]
