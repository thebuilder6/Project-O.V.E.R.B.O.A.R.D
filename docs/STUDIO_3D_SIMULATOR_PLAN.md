# 3D Simulator Mode and Interactive Mission Elements

**Status:** Far stretch goal / design proposal (2026-10-20). Not started.
**Scope:** Adding an interactive 3D WebGL (Three.js + Rapier3D Wasm) simulation mode into the browser studio (`overboard/viz/studio.html`), importing real LEGO CAD models (LDraw / BrickLink Studio) for robots and competition missions, and simulating articulated mission mechanisms as interactable physical elements.
**Audience:** Anyone extending the browser studio, adding 3D visualization, or modeling physical robot-mission interactions.

> **Read this first.** The core of OVERBOARD is mathematical trajectory planning (CasADi + IPOPT) and formal reachability validation (JAX + Immrax interval arithmetic). A 3D physics simulator does **not** replace interval validation: interval reachability bounds the tracking error over the uncertainty the model declares, whereas rigid-body simulation tests a single nominal run with contact mechanics. Whether that bound covers the real robot is an assumption awaiting the hardware campaign (KI-020), not a guarantee. 3D simulation serves as an intuitive digital twin: validating tool clearances, visualizing physical push/actuation dynamics, and verifying that robot attachments physically trigger mission mechanisms.

---

## Table of contents

- [Decisions](#decisions)
- [1. Motivation & Context](#1-motivation--context)
- [2. Model Sources & Licensing](#2-model-sources--licensing)
  - [2.1 Official season mission models (Komurobo)](#21-official-season-mission-models-komurobo)
  - [2.2 Scripted LDraw models (martijas/fll-sim)](#22-scripted-ldraw-models-martijasfll-sim)
  - [2.3 Community CAD & LDraw parts library](#23-community-cad--ldraw-parts-library)
  - [2.4 Copyright and asset isolation](#24-copyright-and-asset-isolation)
- [3. The Ingestion & Physics Pipeline](#3-the-ingestion--physics-pipeline)
  - [3.1 Topological pin clustering](#31-topological-pin-clustering)
  - [3.2 Joint & mechanism detection](#32-joint--mechanism-detection)
  - [3.3 MissionElement contract](#33-missionelement-contract)
- [4. Studio Integration Architecture](#4-studio-integration-architecture)
- [5. The Phases](#5-the-phases)
- [6. Gotchas](#6-gotchas)
- [7. Open Questions](#7-open-questions)

---

## Decisions

These decisions guide the design and ensure the 3D simulator stays cleanly decoupled from the solver core.

| # | Decision | Why |
|---|---|---|
| D1 | **Client-side WebAssembly execution (Rapier3D + Three.js).** | Keeps the studio lightweight and cross-platform. Physics runs inside the browser tab at 60 Hz without burdening the Python server or requiring native C++ physics bindings on the host. |
| D2 | **Topological pin clustering before physics instantiation.** | Real LEGO models contain hundreds of Technic pins and beams. Simulating each brick as an independent rigid body causes severe lag and joint jitter. Clustering merges rigidly pinned sub-assemblies into single compound colliders. |
| D3 | **Progressive enhancement: 2D canvas remains default.** | The fast 2D canvas remains the primary tool for waypoint layout and trajectory review. The 3D simulator is an opt-in tab/view that lazy-loads Three.js, Rapier Wasm, and model meshes on demand. |
| D4 | **Dual-mode simulation: Kinematic Trajectory Playback vs. Dynamic Physics.** | Mode A (playback) sweeps the robot geometry along the exact solved trajectory ribbon for instant clearance checks. Mode B (physics) drops the robot into the Rapier3D world and executes closed-loop tracking (Ramsete) with contact friction. |
| D5 | **External asset loading for mats and season CAD.** | Respects FIRST copyright policies. Official mat images and mission models are loaded from external URLs or user-supplied files; only open-source part definitions and placement metadata live in the tree. |

---

## 1. Motivation & Context

OVERBOARD currently computes optimal trajectories and audits them against 2D polygonal footprints (`fields/*.json` via `overboard/viz/fields.py`). While 2D bounding boxes detect if the chassis hits a mission base, they cannot answer:
1. **Vertical clearance:** Does an overhanging arm or antenna clip an elevated mission element?
2. **Mechanism interaction:** Does the robot's front bumper or motorized attachment actually flip the rock (M03), push the toggle lever (M02), or drop the restoration keystone (M13)?
3. **Contact reaction forces:** Does pushing a heavy model cause wheel slip or drivebase stall? (This directly connects to the future interaction force profiles described in [Future_Plans.md](../Future_Plans.md)).

---

## 2. Model Sources & Licensing

### 2.1 Official season mission models (Komurobo)
* **Source:** [Komurobo FLL 3D Models](https://komurobo.com/fll/3d-models/)
* **Format:** BrickLink Studio (`.io`).
* **Details:** Komurobo models the complete mission set for each season (*BIOGLOW*, *Unearthed*, *Submerged*, *Masterpiece*, etc.) in BrickLink Studio. Because `.io` files are standard zip containers enclosing `model.ldr` and color metadata, they can be parsed by standard LDraw loaders.

### 2.2 Scripted LDraw models (martijas/fll-sim)
* **Source:** [martijas/fll-sim](https://github.com/martijas/fll-sim) (`apps/desktop/resources/missions/*.ldr`)
* **Details:** Built via automated TypeScript scripts matching FIRST building instruction PDFs and *Bricks for the Blind* specifications. Accompanied by table coordinate metadata (`seasons/2026-27/mission-models.json`) recording millimeter positions (`xMm`, `yMm`, `headingDeg`).

### 2.3 Community CAD & LDraw parts library
* **Source:** [LDraw.org](https://www.ldraw.org/) / [OMR](https://omr.ldraw.org/)
* **Details:** Open-source `.dat` geometries for LEGO Technic primitives (beams, axles, gears, wheels, and SPIKE Prime hubs/motors).

### 2.4 Copyright and asset isolation
* **Mat artwork:** FIRST holds the copyright for table mat graphics. Consistent with [`research/field_maps/README.md`](../research/field_maps/README.md), mat textures are never committed to the git repository; the studio downloads or loads the user's mat file into `mats/`.
* **Model licensing:** CAD files from Komurobo and community authors are loaded dynamically via URL or imported locally via drag-and-drop, preserving third-party attribution.

---

## 3. The Ingestion & Physics Pipeline

### 3.1 Topological pin clustering
A mission model like *M06-07 Leafcutter* contains 232 LEGO pieces. Simulating 232 connected dynamic bodies in Rapier3D would lead to solver instability and low frame rates.

The ingestion pipeline (inspired by the clustering solvers in `jaimeyu/fll_sim` and `martijas/fll-sim`):
1. **Parse LDraw hierarchy:** Read part IDs, position vectors, and rotation matrices.
2. **Graph connectivity:** Identify connection points (pin holes, cross-axle holes, axle bushes).
3. **Rigid clustering:** Any group of parts connected by friction pins (e.g., Technic pins `2780`, `6558`) is merged into a single compound rigid body.
4. **Collider generation:** Approximate clustered geometries using convex decomposition or compound bounding boxes for the physics world, while retaining high-fidelity meshes for visual rendering.

### 3.2 Joint & mechanism detection
Axle connections with rotational clearance and sliding rails are recognized as kinematic degrees of freedom:
* **Revolute Joint ($1\text{ DOF}$ rotation):** Hinges, levers, dials, turnstiles (e.g., M03 Rock lever, M02 toggle riser). Configured with angular limits and joint friction damping.
* **Prismatic Joint ($1\text{ DOF}$ translation):** Sliders, plungers, drawers (e.g., M11 past window). Configured with travel limits and linear restitution.
* **Free Dynamic Bodies:** Loose mission items (e.g., seed balls, restoration keystones, crates) that can be pushed, lifted, or rolled.

### 3.3 MissionElement contract
Each interactable model implements a clean interface:
```typescript
interface MissionElement {
  id: string;
  name: string;
  basePose: { xMm: number; yMm: number; zMm: number; headingDeg: number };
  rigidBodies: Rapier.RigidBody[];
  joints: Rapier.ImpulseJoint[];
  visualMesh: THREE.Group;
  
  // State inspection
  isScored(): boolean;
  getActuationProgress(): number; // 0.0 to 1.0
  reset(): void;
}
```

---

## 4. Studio Integration Architecture

The 3D simulator runs directly in the browser studio without external plugins:

```
[ Studio: left Mission | Robot | Field | Solve   right Result | Waypoint | Plots | Run | 3D Sim ]
                                │
                                ▼
                   [ Three.js WebGL Viewport ]
                   ├── 4x8 ft Competition Table & Walls
                   ├── Mat Texture (Canvas Plane)
                   ├── 3D Mission Elements (Clustered LDraw Meshes)
                   └── Robot Model (Chassis, Wheels, Attachments)
                                │
                                ▼
                 [ Rapier3D (WebAssembly @ 60 Hz) ]
                   ├── Ground & Border Colliders
                   ├── Robot Kinematic / Dynamic Body
                   └── Articulated Mission Element Joints
                                │
                                ▼
       [ Trajectory Replay & Closed-Loop Ramsete Controller ]
```

---

## 5. The Phases

### Phase 1: 3D Field Viewport & Visual Inspection
* Add a `3D View` tab to the right-hand strip of `overboard/viz/studio.html` (beside Result, Waypoint, Plots and Run), lazy-loaded so the rest of the studio pays nothing for it.
* Render the competition table (interior 2362 × 1143 mm), border walls, and mat texture using Three.js with OrbitControls.
* Render static 3D mission models loaded from LDraw (`.ldr`) or BrickLink Studio (`.io`) at their designated coordinates.
* Display the planned trajectory as a 3D ribbon with color-coded speed profiles.

### Phase 2: Trajectory Kinematic Playback & Clearance Checks
* Kinematic ghost drive: Animate the 3D robot model along the planned trajectory over time ($t$).
* Continuous 3D collision detection: Check for clearances between robot attachments and 3D mission structures. Highlight intersections in red on the 3D mesh.
* Provide an interactive scrub bar linked with the trajectory plots and waypoint timeline.

### Phase 3: Rigid-Body Physics Simulation (Robot Dynamics)
* Initialize headless Rapier3D (Wasm) inside the browser worker/thread.
* Simulate the differential-drive robot on the competition mat with calibrated wheel friction, weight transfer, and center of mass.
* Run the digital twin's Ramsete tracking loop driving virtual motors against simulated ground contact.
* Record tracking error and slip; compare the simulated trace against the JAX validator's reachable envelope.

### Phase 4: Articulated & Interactable Mission Elements
* Implement articulated physics joints on key season mission elements (revolute levers, spring toggles, sliding trays).
* Enable dynamic interaction: Robot attachments push levers, lift arms, and manipulate loose game pieces.
* Measure interaction resistance forces ($F_{\text{push}}$) during actuation to feed back into future interaction-aware trajectory optimization.

---

## 6. Gotchas

1. **Polygon count & WebGL performance:** A table with 15 fully modeled LEGO missions contains upwards of 500,000 triangles. The Three.js renderer must use instanced meshes (`InstancedMesh`) for identical LEGO bricks (bushings, 2-stud pins) and simplified collision convex hulls for Rapier physics.
2. **Physics determinism:** Rapier3D must step at a fixed timestep (e.g. $\Delta t = 1/60\text{ s}$ or $1/120\text{ s}$) rather than variable frame delta to maintain repeatable simulation runs across different user machines.
3. **LDraw coordinate system:** LDraw uses an inverted Y-axis convention (1 LDU $\approx 0.4\text{ mm}$, with $+Y$ pointing downward). Coordinates and matrices must be converted to standard right-handed metric space ($+Z$ up, units in meters) upon import.

---

## 7. Open Questions

1. **Pre-converted glTF vs. runtime LDraw parsing:** Should official season models be pre-converted into optimized binary glTF (`.glb`) with embedded joint metadata to avoid browser-side LDraw parsing overhead, while keeping raw `.io`/`.ldr` import for custom user attachments?
2. **Physics solver substeps:** Does contact between thin LEGO beams require sub-stepping ($120\text{ Hz}$ or $240\text{ Hz}$) to prevent high-speed tunneling of thin attachment levers?
