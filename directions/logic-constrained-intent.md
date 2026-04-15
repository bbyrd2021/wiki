---
type: direction
title: "Logic-Constrained Pedestrian Intent (ROAD-R → JAAD/PIE)"
aliases: ["logic-constrained intent", "constraint loss on JAAD/PIE"]
created: 2026-04-07
updated: 2026-04-07
sources: ["PedestrianIntent++/SYNTHESIS.md"]
tags: [direction, jaad, pie, neuro-symbolic, logic-constraints, novel]
status: complete
novelty: true
feasibility: workstation
datasets_required: [jaad, pie, road-plusplus]
---

# Logic-Constrained Pedestrian Intent

**Direction 4** from SYNTHESIS.md Part 7.

## Motivation

ROAD-R (Marconato et al. 2022) introduces 243 logical requirements for ROAD. Similar constraints exist for pedestrian scenarios:
- "A crossing pedestrian must be walking" (action constraint)
- "A pedestrian cannot transition directly from standing+not-looking to crossing" (sequence constraint)
- "A pedestrian with Wait2X at a Red traffic light should not transition to Xing immediately" (context constraint)

JAAD's label vocabulary supports this. ROAD++'s duplex/triplet structure is specifically designed for constraint satisfaction.

## Approach

1. Define JAAD/PIE-specific logical constraints from domain knowledge + cross-tab analysis
   - e.g., `standing+looking` rarely crosses in PIE → soft constraint
   - e.g., `cross=1` implies `action=walking` in the same frame → hard constraint
2. Add a constraint satisfaction loss at training or inference
3. Use ROAD++'s AV action + pedestrian action + traffic light triplets as constraint supervision signal

## What's Missing

No paper applies neuro-symbolic or logic-constrained learning to JAAD/PIE pedestrian intent. ROAD-R methodology has only been applied to scene-level event detection, not intent prediction.

## Related

- [[concepts/neuro-symbolic-constraints|Neuro-Symbolic Constraints]]
- [[papers/marconato-2022-road-r|ROAD-R Paper]]
- [[datasets/jaad|JAAD]] | [[datasets/pie|PIE]]
