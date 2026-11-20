# PyWaveDude

Python port of [WaveDude Next](https://github.com/wavedudenext/WaveDudeNext) (Alexander Gräf, Huygens Optics),
including its colour schemes and project file format: a 2D wave optics simulator.

## Installation

Requirements:

| Item | Version | Notes |
|---|---|---|
| [uv](https://docs.astral.sh/uv/) | any recent | creates the environment |
| Python | ≥ 3.11 | installed by uv if missing |
| ffmpeg | optional | on `PATH`; enables MP4 and MPEG-1 export |

```bash
git clone <repository-url> PyWaveDude
cd PyWaveDude
uv sync
```

Installed packages: `numpy`, `numba`, `pillow`, `dearpygui`, `pyyaml`, and `mlx` on Apple Silicon only.

## Quick start

```bash
uv run pywavedude                         # new project, blank canvas
uv run pywavedude scene.png               # new project on a scene image
uv run pywavedude project.wdproj          # open a saved project
uv run pywavedude examples/sted.wdproj    # example: STED focus with a phase step plate, true colour
uv run pywavedude --backend cpu           # force a compute backend: auto | cpu | mlx
uv run pywavedude --defaults my.yaml      # start with another defaults file
```

First steps:

1. Nothing is computed at start. The default project has one point source on a 1920 × 1080 canvas.
2. Project tree toolbar: **+ Source**, **+ Part ▼** (a kind from the menu) or **+ Probe ▼** → click the canvas.
3. Click an object on the canvas or its row in the tree; its settings open in the Properties pane; drag it on the
   canvas to move it, its handles to rotate or resize it.
4. **Run** (F5): computes the simulation from step 0; the timeline returns to 0 and shades as frames are computed.
5. **▶ Play** (Space): animates the computed frames; the same button pauses. Click or drag the timeline to scrub.
6. While you edit, the frame at the timeline's position is recomputed and shown; when the edit is finished it
   becomes an undo step and, after a run, a new run starts (switch: *Re-run automatically after edits*).
7. **Stop** (Shift + F5): discards the computed simulation. **File → Export** saves a video.

## Features

### Simulation

| Item | Description |
|---|---|
| Model | scalar 2D wave equation, explicit leapfrog finite differences |
| Laplacian | 9-point stencil, or 4th-order with automatic sub-stepping |
| Units | 1 grid cell = 1 pixel, 1 time step; wave speed in a medium = 1/n |
| Edges | open (absorbing) or reflective; optional damping rim inside or outside the canvas |
| Intensity | running two-stage low-pass of the squared field, time constant set by the slowest source |
| Backends | `cpu`: multithreaded JIT kernels, any OS · `mlx`: Metal GPU, Apple Silicon |
| Determinism | every run from step 0 is reproducible (project seed for all random choices) |
| Separate wavelengths | optional: one simulation per source wavelength on the same scene; field = their sum, intensity kept per wavelength (incoherent between wavelengths, coherent within one) |

### Scene

| Source | Description |
|---|---|
| Blank canvas | width, height, background refractive index (1.00–2.55) |
| Scene image | PNG, BMP, JPEG, TIFF, GIF; embedded in the saved project |

Grid (row under the transport buttons, Display settings or View menu): on/off, step in pixels, opacity; object centres snap to grid lines and nodes
(switchable). Shown on the preview only, never exported.

Scene image channels:

| Channel | Meaning |
|---|---|
| Red | refractive index × 100 (100 = vacuum, 150 = glass); 0 = mirror |
| Green | emitter; value = frequency level (used when *Image sources* is on) |
| Blue | absorption, 0–255 |

### Sources

| Setting | Values |
|---|---|
| Frequency level | 0.01–255 (wavelength shown in pixels) |
| Amplitude, phase | 0–100 %, −180…180° |
| Emission | continuous · pulsed (Gaussian or smoothed square, pulse/off width in cycles) |
| Detach step | stop driving the cell after a given step |
| Composite | grid of columns × rows, spacing, rotation |
| Random spread | per element: position, frequency, phase, off width |

### Optical parts

| Group | Parts |
|---|---|
| Lenses | lens (spherical surfaces, convex/concave), Fresnel lens, GRIN lens |
| Shapes | rectangle, ellipse, polygon, triangle, free outline |
| Waveguides | fibre (Bézier path, core/cladding, absorbing band) |
| Layers | Bragg reflector, interference filter |
| Gratings | simple dot grating, facetted grating (sawtooth, square, sine), diffuser |
| Diffractive | zone plate, meta lens, meta prism, meta axicon |
| Phase plates | phase step plate: zones across the aperture alternate by a set phase delay (thickness from the design wavelength); 2 zones × 180° = 2D counterpart of a vortex plate (dark on-axis focus, STED depletion) |
| Logic | XOR, AND, OR waveguide gates |

Common part settings: position, rotation, mirror or glass, refractive index, anti-reflection coating (layer or
graded, per face), absorption with ramp. Edges are anti-aliased.

Lens read-out: edge thickness, fitted radii, effective and back focal length (thick-lens lensmaker's formula from
the fitted radii, in the blank canvas's background index; Fresnel lens as thin).

### Probes

| Type | Output |
|---|---|
| Point | field amplitude, intensity, or integrated intensity per step |
| Line (A → B) | field or intensity profile per frame, or integrated profile |
| Counting detector | random detections along a line per step (photomultiplier) |
| Imaging detector | accumulated detections per pixel bin (photon-counting array) |

- Live graphs: in the **Probe graphs** column at the right edge of the window (opens when a run with probes
  starts; switch: *Graph column*), and optionally in boxes over the picture of the layer the probe measures (switch:
  *Graphs on the canvas*, off by default; drag a box to move it, its corner grip to resize it).
- Separate wavelengths: intensity probes (point and line, not detectors) show a curve per wavelength in its colour
  beside the total (white); the curves add up to the total (switch: *Probe curves per colour*, Display settings).
- Graph options: running window or full run, auto or fixed value range, linear or logarithmic scale.
- **Export CSV** (File → Export): one row per step (point) or per frame (line), `;`-separated.

### Interface

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ Menu: File · Edit · View · Simulation · Help                                                         │
├──────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Project tree                 │ Field   step N   Brightness [  ] Auto                │ Probe graphs   │
│ + Source  + Part  + Probe    │ picture: grid, handles, graph boxes                  │ (optional)     │
│ Project · Scene · Simulation │                                                      │                │
│ Display · Sources · Parts    │ Intensity   step N   Brightness [  ] Auto            │                │
│ Probes · Colour schemes      │ picture                                              │                │
├──────────────────────────────┤                                                      │                │
│ Properties                   │ timeline: ticks, computed part, in/out, playhead     │                │
│ form of the selection        │ Run · Pause · Stop · Play · ◀ ▶ · In Out · Loop Live │                │
│                              │ Grid · step · Snap · opacity · Graphs · True colour  │                │
├──────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Status bar: last message · run state                                                                 │
└──────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

| Element | Description |
|---|---|
| Project tree | Project (seed, description), Scene, Simulation, Display; groups Sources, Optical parts, Probes, Colour schemes; per row: Enabled, move up/down, remove |
| Selection | click; Ctrl/Cmd + click toggles; Shift + click a range in the tree / adds on the canvas; drag on empty canvas = marquee; Ctrl/Cmd + A all |
| Properties | form of the current object: labels, categories, a hint on every label, kind description for parts; values apply as you type |
| Picture header | layer, step, brightness box, **Auto** (fits the brightness to the frame shown) |
| Picture frame | red: reflective edges; blue: damping rim outside the canvas; grey: open edges |
| Handles | selected object: rotation lever (Shift: 15° steps), corner grip (size), lens surface handles (radii), outline points, fibre nodes and Bézier handles, probe ends, graph box grip |
| Timeline | ticks in frames; computed part shaded; in/out brackets = export range (drag them); playhead |
| Transport | Run, Pause/Resume run, Stop, Play/Pause, frame back/forward, In, Out, Clear, Loop, Live (follow the newest frame), position (frame and step), speed, shown layers, arrangement |
| View row (under the transport) | Grid, grid step, Snap, grid opacity; Graphs on the canvas, Graph column; Separate wavelengths, True colour |
| Undo / redo | unlimited; an edit becomes a step when it has finished (mouse released, field left) |
| Clipboard | cut, copy, paste, duplicate of sources, parts, probes and colour schemes, as XML text |
| Colour schemes | project schemes from any scheme; gradient editor: click adds a stop, drag moves it, drag away or double-click removes it; colour and position fields |
| Memory warning | before a run whose frames exceed half the memory: halve resolution and/or frames, or run anyway |
| Status bar | last message (warnings yellow, errors red), what is being placed, run state and speed |

Menus:

| Menu | Entries |
|---|---|
| File | New, Open, Recent projects (session), Examples, Save, Save as, Open scene image, Blank canvas, Export (field, intensity, both, field frame, intensity frame, probe data as CSV), Export last, Save shown picture as PNG, Exit |
| Edit | Undo, Redo, Cut, Copy, Paste, Duplicate, Delete, Select all |
| View | Panes (project tree, properties, status bar, probe graphs), preview resolution, preview frames (every 1st/2nd/4th), UI zoom, grid, handles, labels, graphs on the canvas |
| Simulation | Run, Pause/resume run, Stop, Play/Pause, compute backend, re-run after edits, separate wavelengths, memory warning |
| Help | Manual (this README), About (versions, GPU, system information to copy) |

### Display

- Layers: field, intensity or both; stacked or side by side; each a panel with its header.
- 23 colour schemes plus project schemes; brightness, black level, auto brightness.
- True colour intensity (*Separate wavelengths* and *True colour*: view row, Simulation menu, Display settings): each wavelength through a black-to-colour ramp, added like
  light; colours from the wavelength (shortest 450 nm … longest 650 nm, or a set nm per pixel); colour key above
  the intensity panel.
- Part outlines, source / part / probe markers with labels; handles of every object or of the selected one only.
- Preview resolution: full, half, quarter; preview frames: every 1st, 2nd or 4th kept.
- Playback at the export frame rate × speed (0.25–8); loop over the export range.

### Export

| Format | Content |
|---|---|
| MP4 (H.264), MPEG-1 | video (needs ffmpeg) |
| AVI | uncompressed video, no size limit |
| QuickTime | Animation (lossless RLE) or PNG codec |
| PNG / JPEG sequence | numbered frames in a folder |
| OpenEXR | 16-bit float data, intensity as I or √I |

Options: field / intensity / both, frame rate, quality, step range (the timeline's in/out points), part outlines,
probe graphs, probe and source markers. **Export shown frame** saves a single full-resolution still. **Export last**
(Ctrl/Cmd + E) repeats the session's last export without the dialog.

### Projects

- File format: `.wdproj` (XML).
- Stored: scene, embedded image, sources, parts, probes, display settings, custom colour schemes, random seed,
  grid, true colour, layer and preview settings.

## Defaults

File: `src/pywavedude/defaults.yaml`, read at start (`--defaults FILE` reads another). Every key must name an
existing setting; an unknown key or a wrong value stops the start with the key's path.

| Section | Contents |
|---|---|
| `app.backend` | compute backend at start (`--backend` overrides it) |
| `app.ui` | window size, left pane width, tree pane height, probe graph column width, font size, UI zoom and its choices, rows per tree group |
| `app.canvas` | pick radius, snap radius, rotation lever length (screen pixels) |
| `app.playback` | speed and its choices, loop, live |
| `app.run` | memory warning on/off, memory budget as a share of the physical memory |
| `app.export` | file name, format (`MP4`, `MPEG1`, `AVI`, `MOV`, `MOV_PNG`, `PNG`, `JPEG`, `EXR`, `EXR_ROOT`), layers, part outlines |
| `new_project.scene` / `.simulation` / `.display` | every setting of a new project |
| `new_project.sources` | the sources a new project starts with (list of source settings with `name`, `x`, `y`) |
| `new_objects.source`, `.probe` | settings of a newly placed source or probe |
| `new_objects.parts.<Kind>` | settings of a newly placed part, per kind (`LensPart`, `RectanglePart`, …) |

Values in a saved project are its own; an attribute a `.wdproj` file leaves out keeps the project format's default.

## Controls

| Key / action | Function |
|---|---|
| F5 / Shift + F5 / F6 | Run / Stop / Pause or resume the run |
| Space, F7 | Play / Pause the animation of the computed frames |
| `,` `.` | one frame back / forward |
| I, O, X | in point, out point, clear the export range |
| L, F | loop, live |
| Ctrl/Cmd + N, O, S, Shift + S | new, open, save, save as |
| Ctrl/Cmd + E | export last |
| Ctrl/Cmd + Z, Y (Shift + Z) | undo, redo |
| Ctrl/Cmd + X, C, V, D, A | cut, copy, paste, duplicate, select all |
| F1 | manual |
| Click / drag on canvas | select / move objects (centres snap to the grid when snapping is on); drag empty space for a marquee |
| Arrows (Shift: ×10), Tab | after a click on the canvas: nudge the selection, select the next object |
| Del, Backspace | after a click on the canvas: remove the selected objects |
| Click after **+ Source / Part / Probe** | place the new object (line probes and detectors: two clicks, A then B) |
| Esc | cancel a drag (everything back), cancel placing, or clear the selection |

## Benchmark

```bash
uv run python -m pywavedude.bench [width height steps]
```

## License

GPL-3.0-or-later. See [LICENSE.md](LICENSE.md). The bundled Inter font is under the SIL Open Font License 1.1
([OFL.txt](src/pywavedude/assets/fonts/OFL.txt)).
