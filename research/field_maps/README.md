# Where the field maps came from

Reference material, not source. It is here so the provenance of the mission
model geometry is recorded, and so the studio's collision check can be built
on something with a known origin instead of on a guess.

## The short answer

**FIRST's own field wireframe PDF is the source, and it is public.** Each
project measured the mission models off it by hand:

- <https://firstinspires.blob.core.windows.net/fll/challenge/2026-27/fll-challenge-bioglow-wireframe-grid.pdf>
  — the 2026-27 BIOGLOW wireframe (904 kB, `application/pdf`, public blob).
- <https://firstinspires.blob.core.windows.net/fll/challenge/2025-26/fll-challenge-unearthed-rgr.pdf>
  — the 2025-26 UNEARTHED field guide.

The mat's **artwork** is FIRST's copyright and is not in any of these
repositories. The **coordinates** are facts about a printed mat, and they are
in all of them.

## What is in this directory

### `robovibe_yaml/`

RoboVibe (<https://robovibe.afarago.hu/>) embeds one YAML per season in its
public JavaScript bundle, under the virtual paths `assets/config/<id>.yaml`.
Ten of them were extracted from `assets/js/ui-common-*.js`:

| file | season |
|---|---|
| `fll2026founders.yaml` | FLL 2026-27 BIOGLOW Founders Edition |
| `fll_bioglow_future_editi.yaml` | FLL 2026-27 BIOGLOW Future Edition (preview) |
| `fll_unearthed.yaml` | FLL 2025-26 UNEARTHED |
| `educup2025em.yaml` | EduCup 2025 Master Engineering Challenge |
| `educup_2023_kids_enginee.yaml` | EduCup 2023 Kids Engineering Challenge |
| `wro_2025_robomission_sen.yaml` | WRO 2025-26 RoboMission Senior |
| `wro_2026_robomission_sen.yaml` | WRO 2026 RoboMission Senior |
| `wro_2026_robomission_ele.yaml` | WRO 2026 RoboMission Elementary |
| `wro_2026_robomission_jun.yaml` | WRO 2026 RoboMission Junior |
| `wro_2026_inspired_kids.yaml` | WRO 2026 Inspired Kids |

The `blob*` files are the same YAMLs before they were named.

Each YAML has three blocks:

- `challenge:` — the season, the year, the match duration, and `rules:`,
  the link to the official document the geometry was measured from.
- `map:` — `unit_of_measurement: mm`, the field's `width`/`height`, the
  `imageurl` of the mat picture, the `startareas` (the launch areas as
  quarter-circles with a radius and a quadrant) and the `models`.
- `models:` — per mission: the `areas` (the collision rectangles and circles:
  `x`, `y`, `width`, `length`, `rotation`, `type`, and the `passive` /
  `placeholder` flags), and the `score` rules.

**The frame is the table, not the mat.** The coordinates run over the
2362 × 1143 mm table interior, with the mat centred: the launch areas sit at
x = 181 mm and x = 2181 mm, because (2362 − 2000) / 2 = 181.

### The two readings, in detail

Two things have to be right before RoboVibe's geometry lands on the mat, and
both were found by checking against FIRST's own wireframe:

- **The frame.** RoboVibe's coordinates are in the *table's* (2362 x 1143 mm),
  with the mat centred in it, so x is shifted by 181 mm.
- **The anchor.** A shape's `x` and `y` are the **corner** of its box, and
  the shape rotates **clockwise about that corner**. Reading the anchor as
  the centre puts every shape out by half its size; reading the rotation as
  counter-clockwise, or as being about the centre, puts the turned ones out
  by tens of millimetres.

With both right, the models the two fields share agree to a few millimetres:

| model | RoboVibe | the other field | apart |
|---|---|---|---:|
| Exploding Seeds | 56 x 58, turned 45 | 62 x 62 | 2 mm |
| Flip the Rock | 137 x 65 | 132 x 66 | 4 mm |
| Reaching Roots | 41 x 91 | 42 x 88 | 4 mm |
| Window to the Past | 74 x 81, turned 45 | 95 x 95, turned 45 | 4 mm |

### `field-check.png`

The converted RoboVibe field drawn over the composed wireframe, so the
alignment can be seen rather than argued about. `draw_field_check.py` makes
it: it draws `fields/2026-27-robovibe.json` over the mat picture the studio
composes from FIRST's field guide.

### `martijas/`

<https://github.com/martijas/fll-sim> keeps the same data as JSON, one file
per season:

- `season-2026-27.json` — the table (interior 2362 × 1143, walls 38 × 64),
  the mat (2000 × 1140, with their own note: *"Measured from the wireframe
  PDF: ten 20 cm grid columns = 2000 mm; aspect 1.7537 -> 1140 mm"*),
  the launch areas (quarter-circles of radius 483 at the mat's corners), and
  **15 `missionModels` with a collision footprint each** (a `rect` with
  `cx`, `cy`, `w`, `h`, `rot`, or a `circle` with `cx`, `cy`, `r`).
- `mission-models-2026-27.json` — the **placement pose** of every mission
  model they build from real LEGO parts, with `fitMeanMm`: the mean error of
  the fit that put the model on the mat (1.4 to 8.3 mm).

Their coordinates are in the **mat's** frame (2000 × 1140), not the table's.
Their wireframe PDF is git-ignored, because it is FIRST's; their
`tools/mat-import/import_mat.py` renders it to a PNG and their
`tools/model-build` fits each built model onto the wireframe marks.

## The two readings, compared

Two teams measured the same printed wireframe independently. After putting
them in the same frame (adding the 181 mm mat offset to martijas' numbers),
the **sizes** agree to a few millimetres and the **positions** do not:

| model | RoboVibe's area | martijas' footprint | size difference | apart |
|---|---|---|---|---:|
| M01 Drone Survey | circle d64 | circle d68 | 4 mm | 44 mm |
| M10 habitat A | 40 × 45 | 42 × 47 | 2 mm | 28 mm |
| M10 habitat B | 40 × 45 | 36 × 36 | 4–9 mm | 101 mm |
| M12 Forest Elder | circle d64 | circle d66 | 2 mm | 41 mm |

The sizes agreeing to a few millimetres says both readings are careful. The
positions differing by 28 to 101 mm says they are describing **different
things**: RoboVibe's areas are the *scoring* areas and the printed graphics,
martijas' are the footprints of the *physical LEGO models*.

**Consequence for a collision check:** a check built on this data needs a
tolerance of 50 to 100 mm, or the operator's own tape measurement. A path
that "misses" a mission model by 30 mm has not been shown to hit anything.

## The other two projects

- `jaimeyu/fll_sim` (<https://github.com/jaimeyu/fll_sim>, GPL-3.0):
  `src/missions/season-config.ts` has the same season with an `arenaPosition`
  and a `yawDegrees` per model, and `src/missions/field-layout-manager.ts`
  makes the layout **editable and saves it to the browser**, so a team can
  correct the placements on their own table.
- `hkevin01/FLL-SIM` (<https://github.com/hkevin01/FLL-SIM>): the mat
  **pictures** per season, fetched from a manifest and cached under
  `assets/mats/<season>/mat.png`, and the mat sizes in
  `configs/profiles/defaults.yaml` ([2362.0, 1143.0] for 2024-submerged and
  2023-masterpiece).

## Licence

RoboVibe is closed source with no published licence, and martijas/fll-sim
has no licence file either. These files are kept here as **reference
material** so the provenance is on record. Nothing here should be copied
into the planner as source. If the studio needs the geometry in the tree,
the honest route is to measure it from FIRST's public wireframe (or from the
team's own mat) and ship that, with the operator able to adjust it — which
is what `jaimeyu` and robovibe both let their users do.
