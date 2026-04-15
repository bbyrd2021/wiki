---
type: paper
title: "Diving Deeper Into Pedestrian Behavior Understanding: Intention, Action, and Risk Assessment"
authors: "Rasouli & Kotseruba"
year: 2024
venue: "IV"
arxiv: "2407.00446"
created: 2026-04-07
updated: 2026-04-07
sources: ["PedestrianIntent++/SYNTHESIS.md"]
tags: [paper, pie, intent-prediction, risk-assessment, evaluation]
status: complete
datasets_used: [pie]
---

# Diving Deeper Into Pedestrian Behavior Understanding

**Rasouli & Kotseruba | IV 2024 | arXiv:2407.00446**

Comprehensive evaluation of intent estimation, action prediction, and risk assessment under scenario-stratified metrics on PIE.

## Contribution

- Evaluates three tasks jointly: intent estimation, action prediction, risk assessment
- Scenario-stratified metrics: by pedestrian distance and ego-vehicle speed
- Shows significant degradation for **distant pedestrians** and at **high ego-vehicle speeds**
- Uses PedFormer architecture with 1–3s time-to-event (TTE) windows

## Key Findings

- Intent accuracy degrades substantially for small (distant) pedestrians — consistent with ENCORE
- High ego-vehicle speeds reduce available decision time and hurt all metrics
- Risk assessment (will the pedestrian pose a danger to the AV?) is harder than intent prediction alone

## Related

- [[papers/rasouli-2023-pedformer|PedFormer]] | [[papers/rasouli-2024-encore|ENCORE]]
- [[datasets/pie|PIE]]
