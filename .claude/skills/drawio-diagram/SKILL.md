# drawio-diagram

Generate architecture diagrams as native draw.io (.drawio) XML files in the
lab's house style, editable shape-by-shape in draw.io or diagrams.net.

## When to use

- `/drawio-diagram <name> "<description>"` or any request for an editable
  architecture diagram. Prefer this over SVG/HTML when Brandon will edit the
  figure himself (slides, thesis figures).

## House style (the visual grammar — always include a legend)

draw.io style strings:

- **frozen component**: `rounded=0;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#111111;strokeWidth=1;dashed=1;dashPattern=5 4;fontFamily=Helvetica;fontSize=12;`
- **trained component**: same but `strokeWidth=3;dashed=0;`
- **plain / parameter-free op**: same but `strokeWidth=1;dashed=0;`
- **note / retired**: `strokeColor=#999999;fontColor=#555555;strokeWidth=1;`
- **eyebrow labels**: text cell, `text;html=1;fontSize=10;fontColor=#666666;fontStyle=1;spacing=0;` uppercase with letter-spacing via CSS not available — just uppercase.
- **wires**: `edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;strokeColor=#111111;strokeWidth=1;endArrow=block;endFill=1;` — label edges with what flows (shape + dims).
- **coordinate-only / special wires**: `strokeWidth=2;` and a label; cached wires: `dashed=1;dashPattern=8 3;strokeColor=#555555;`

## Rules (inherited from 6 adversarial review rounds)

1. Every arrowhead terminates ON a box edge; label wires at BOTH ends if long.
2. Dims everywhere: every tensor gets its shape; every layer its in/out sizes.
3. Fan-outs drawn explicitly (distribution bar + per-target arrows), never implied.
4. Dead-ends marked (an X or "discarded" note); nothing "exists but unclear".
5. No lab jargon in visible text (no exp numbers); metric named; baselines defined.
6. NO EM DASHES in any visible text (standing Brandon rule).
7. Legend on every diagram mapping the border grammar.

## Output

Write `<name>.drawio` to ~/Downloads/ (or the asked location), send via
SendUserFile. The file must open cleanly at app.diagrams.net. Keep one
diagram per file, page size ~1100x800.

## Template-grade detail mode (Brandon 2026-08-29: default for slide/thesis figures)

Source vocabulary: ~/Downloads/drawio_examples/drawio-nn-templates/ (mxlibrary
shape libraries; load into draw.io via File > Open Library). Reference
diagrams extracted: template_ref_Yolo_V1.drawio, template_ref_Action_Recognition.drawio.

Conventions to follow at this detail level (~100-300 cells per figure):
- **Feature maps / tensors = isometric cubes**:
  `shape=cube;whiteSpace=wrap;html=1;boundedLbl=1;backgroundOutline=1;darkOpacity=0.05;darkOpacity2=0.1;size=<depth-px>;`
  Cube width/height proportional to spatial dims; channel count as the cube's
  bounded label; spatial dims as small free text along the cube's edges.
- **Every layer drawn**: conv/pool/attention blocks as colored rounded rects
  with kernel/stride/channels in the label. No "backbone" black boxes.
- **Palette semantics** (draw.io standard fills):
  green #d5e8d4 = inputs · blue #dae8fc = conv/compute · orange #ffe6cc
  (stroke #d79b00) / #fad7ac = feature-map cubes · yellow #fff2cc = pooling
  and elementwise ops · red #f8cecc = heads/losses/outputs · purple #e1d5e7 =
  recurrent/temporal modules · gray #f5f5f5 = post-processing (NMS etc.)
- Frozen vs trained still encoded by border: dashed 5 4 = frozen, strokeWidth
  3 = trained, on top of the fill colors.
- Dimension labels between every stage; arrows carry stride/shape changes.

## Edge best practices (learned from Brandon's manual cleanup, 2026-08-29)

1. **Every edge**: `rounded=1;jettySize=16;` in the style (soft corners,
   consistent stubs). Never ship sharp multi-jog orthogonals.
2. **Fan-ins of 3+ sources use a COLLECTOR BUS**: a thin filled bar
   (`fillColor=#111111;height=3`), sources drop onto it with
   `endArrow=none` vertical edges, then ONE arrowed edge per consumer
   leaves the bus. Never draw N parallel orthogonal edges into one target.
3. **Explicit exit/entry anchors on every edge** (`exitX/Y`, `entryX/Y`),
   chains always right-edge → left-edge, taps always bottom → top.
4. **Align rows**: shapes on the same logical row share a vertical center so
   horizontal runs are straight; cubes are vertically centered to the block
   row they sit in.
5. **Column locality**: derive-from relationships (e.g. P6/P7 from C5's lane)
   live in the SAME column as their source; outputs of a block sit directly
   right of or below it. Long cross-canvas edges mean the layout is wrong,
   not the edge.
6. Side-by-side beats stacked when two consumers share one producer (heads
   under a bus), so no through-shape wiring is ever needed.

## v4 corrections (from the second screenshot round)

7. **Bus entries are PROPORTIONAL, never fixed**: each drop's entryX =
   (source_center_x - bus_x) / bus_width, entryY=0 — so every drop is a
   straight vertical at its own lane. A fixed entry point turns the bus into
   a horizontal snake pile.
8. **Compute alignment, don't eyeball**: define column center constants
   (e.g. T3X=703) and derive EVERY x in that column from them (cube x =
   cx - w/2, block x = cx - 55). A 7px offset reads as a jagged elbow.
9. **Row exclusivity**: a row that carries a horizontal edge (the up-chain)
   may contain NOTHING else at that y. Lanes that must coexist go strictly
   right of the last column that the horizontal reaches.
10. **Merging two outputs into one consumer = symmetric funnel**: both exit
    bottom, waypoint at a shared y below both, enter the consumer's top at
    0.25/0.75. Never route one output horizontally past its sibling.

11. **Fan-in grouping: CONTAINER, not bus bar** (Brandon 2026-08-29): when N
    shapes feed one consumer set, put them inside a titled container
    (container=1;pointerEvents=0; light gray fill) and connect the CONTAINER
    to the consumers — 2 edges instead of N drops + bar. Title placement:
    find a band of the container no edge crosses (usually bottom, off-center);
    the top edge is always pierced by feeds. Validate with the drawio-skill's
    validate.py --score until 0; export PNG headlessly and self-inspect
    before delivering. Never deliver a diagram you have not seen rendered.
