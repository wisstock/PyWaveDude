# PyWaveDude

Python port of [WaveDude Next](https://github.com/wavedudenext/WaveDudeNext) (Alexander Gräf, Huygens Optics),
including its colour schemes and project file format: a 2D wave optics simulator.

## Contents

- [Installation](#installation)
- [Running](#running)
- [Workflow](#workflow)
- [Simulation](#simulation)
- [Scene](#scene)
- [Sources](#sources)
- [Optical parts](#optical-parts)
- [Probes](#probes)
- [Display](#display)
- [Interface](#interface)
- [Controls](#controls)
- [Export](#export)
- [Projects](#projects)
- [Defaults](#defaults)
- [Benchmark](#benchmark)
- [License](#license)

## Installation

| Requirement | Version | Notes |
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

## Running

```bash
uv run pywavedude [FILE] [--backend auto|cpu|mlx] [--defaults FILE]
```

| Argument | Meaning |
|---|---|
| `FILE` | a project (`.wdproj`) or a scene image (PNG, BMP, JPEG, TIFF, GIF); none: a new project on a blank canvas |
| `--backend` | compute backend: `auto` (GPU where available), `cpu`, `mlx` |
| `--defaults` | another defaults file instead of the bundled one |

Example projects: **File → Examples**.

## Workflow

| Step | Action |
|---|---|
| Canvas | **Scene**: width, height, background index; or **File → Open scene image** |
| Wavelength scale | **Simulation → Wavelength at frequency 100 (px)**; shortest wavelength ≥ ~8 px (≥ 20 px for thin-film parts) |
| Objects | **+ Source**, **+ Part ▼**, **+ Probe ▼** in the project tree → click the canvas |
| Editing | click an object (canvas or tree) → **Properties**; drag to move, lever to rotate, corner grip to resize |
| Run | **Run** (F5): computes from step 0; the timeline shades as frames are computed |
| Playback | **▶ Play** (Space); click or drag the timeline to scrub |
| Re-run | an edit stops playback; when finished it becomes an undo step and, after a run, a new run starts from step 0 with the timeline position kept (*Re-run automatically after edits*) |
| Stop | **Stop** (Shift + F5): discards the computed simulation |
| Results | **File → Save** (`.wdproj`), **File → Export** (video, frames, EXR, probe CSV), **Save shown picture as PNG** |

## Simulation

| Item | Description |
|---|---|
| Model | scalar 2D wave equation, explicit leapfrog finite differences |
| Laplacian | 9-point stencil, or 4th-order with automatic sub-stepping |
| Units | 1 grid cell = 1 pixel, 1 time step; wave speed in a medium = 1/n |
| Wavelength scale | *Wavelength at frequency 100 (px)* sets the source frequency scale: every wavelength ∝ 1 / (frequency × scale) |
| Intensity | running two-stage low-pass of the squared field, time constant set by the slowest source |
| Separate wavelengths | one simulation per source wavelength on the same scene; field = their sum, intensity kept per wavelength (incoherent between wavelengths, coherent within one) |
| Backends | `cpu`: multithreaded JIT kernels, any OS · `mlx`: Metal GPU, Apple Silicon |
| Determinism | every run from step 0 is reproducible (project seed for all random choices) |
| Memory warning | before a run whose frames exceed the memory budget: halve resolution and/or frames, or run anyway |

## Scene

| Setting | Description |
|---|---|
| Blank canvas | width, height, background refractive index (1.00–2.55) |
| Scene image | PNG, BMP, JPEG, TIFF, GIF; embedded in the saved project |
| Edges | open (absorbing) or reflective |
| Damping edge | absorbing rim inside or outside the canvas |
| Image sources | green pixels of the image emit (*Image source amplitude*) |

Scene image channels:

| Channel | Meaning |
|---|---|
| Red | refractive index × 100 (100 = vacuum, 150 = glass); 0 = mirror |
| Green | emitter; value = frequency level |
| Blue | absorption, 0–255 |

## Sources

| Type | Setup | Result |
|---|---|---|
| Point | default source | circular wave |
| Line / plane wave | *Composite*: 1 column × N rows, spacing 2–4 px | beam along ±x, width N × spacing |
| Array | *Composite*: columns × rows, spacings, *Rotation* | phased array, rotated beam |
| Random / incoherent | *Composite* + randomisation | partially coherent light, speckle |
| Pulsed | *Mode* Pulsed | wave packets |
| Image sources | green channel of a scene image | emitters of any shape |

| Setting | Values | Meaning |
|---|---|---|
| X, Y | px | position (array centre) |
| Frequency | 0.01–255 | frequency level; *Wavelength (px)* beside it shows and sets the same |
| Amplitude | 0–100 % | drive amplitude |
| Phase | −180…180° | phase offset against other sources of the same frequency |
| Mode | Continuous, Pulsed | emission |
| Pulse shape | Gaussian, SmoothedSquare | envelope of a pulse |
| Pulse width, Off width | cycles | on-span and pause of a pulse train |
| Detach after step | −1 = never | stop driving the cell; it becomes ordinary medium |
| Composite source | on/off | many emitters from one source |
| Columns, Rows | count | array size |
| Horizontal / vertical spacing | px | emitter distance |
| Rotation | degrees | array angle |
| Randomisation: Frequency, Phase, Off width, Position | ± level, °, cycles, px; 0 = off | per-emitter spread (project seed) |

## Optical parts

### Common settings

| Setting | Meaning |
|---|---|
| Name, Enabled | label; a disabled part is not painted |
| X, Y, Angle | centre and rotation (counter-clockwise); a part's axis is its x direction |
| Mirror | the body reflects everything (glass otherwise) |
| Refractive index | 1.00–2.55 |
| Anti-reflection coating | band width in px, profile *Layer* (band of index √n) or *Graded*; front / back / sides on or off |
| Absorption, Absorption ramp | 0–255 absorption of the body, reached over a ramp in px from the surface |

Edges are anti-aliased. Wavelength-designed parts (zone plate, meta surfaces, gates, phase step plate, dichroic
mirror) show a *Design wavelength (px)* beside their design frequency; typing it sets the frequency.

### Catalogue

| Group | Part | Main settings |
|---|---|---|
| Lenses and mirrors | Lens | Diameter, R1, R2 (sign = convex / concave; *Flip* button), Centre thickness |
| | Fresnel lens | lens settings + Zones, Zone division (EqualWidth, EqualDepth) |
| | GRIN lens | Length, Diameter, Centre index, Edge index (parabolic profile) |
| Shapes | Rectangle, Ellipse | Length, Width |
| | Polygon | Number of sides, Side length |
| | Triangle | Base, Type (Equilateral, RightIsosceles, ThirtySixtyNinety, Isosceles, Custom), Angle A, Angle B |
| | Outline | Points (dragged on the canvas) |
| Waveguides | Fibre | Core index, Core diameter, Cladding index, Cladding thickness, Ends (Flat, Round), Nodes with Bézier handles, cladding absorption |
| Layers | Bragg reflector | Layer pairs, Thickness 1/2, Index 1/2, Width |
| | Interference filter | Length, Width, Layers (`thickness@index; …`, front to back) |
| | Dichroic mirror | Type, Cut-off frequency, Design angle, Layer pairs, High / Low index, Half-thickness end layers, Medium index, Substrate thickness, Width |
| Diffraction gratings | Grating | Columns, Rows, Column / Row spacing, Dot diameter |
| | Facetted grating | Length, Width, Pattern (Sawtooth, SquareWave, Sinusoidal), Pitch, Amplitude |
| | Diffuser | Length, Width, Centre wavelength, Amplitude, Sigma (random rough front face) |
| Diffractive optics | Zone plate | Diameter, Thickness, Target frequency, Focal length, Inverse, Mount, Mount width |
| | Meta lens | Length, Width, Target frequency, Pitch, Height, Pattern index, Focal length, Lens type (Converging, Diverging) |
| | Meta prism | meta settings + Deflection (degrees) |
| | Meta axicon | meta settings + Cone angle (degrees) |
| Phase plates | Phase step plate | Aperture, Base thickness, Zones, Phase step, Design frequency, Medium index; 2 zones × 180° = 2D vortex plate (dark on-axis focus, STED depletion) |
| Logic gates | XOR, AND, OR gate | Design frequency, Core index, Cladding index, Port spacing, AR on input / output side, Upper / Lower output |

### Read-outs

| Part | Read-out |
|---|---|
| Lens, Fresnel lens | edge thickness, fitted radii, focal and back focal length (thick-lens formula in the background index; Fresnel lens as thin) |
| Zone plate | wavelength, number of zones |
| Phase step plate | wavelength, step height, zone width |
| Meta surfaces | wavelength, full-turn height, number of cells; axicon: line of focus |
| Interference filter | stack thickness |
| Bragg reflector | length |
| Logic gates | wavelength, core width, coupler gap and length, port spacing, outputs |
| Dichroic mirror | see below |

### Dichroic mirror

| Item | Description |
|---|---|
| Structure | substrate (Substrate thickness, Refractive index) with a quarter-wave stack of High / Low index on its front (−x) face |
| Layer thickness | d = λ0 / (4 · n · cos θ), θ from the Design angle by Snell's law in each layer |
| Reflected band | λ0 / (1 + Δg) … λ0 / (1 − Δg), Δg = (2/π) · arcsin((ηH − ηL) / (ηH + ηL)), η = n · cos θ |
| Longpass | reflects the band just below the cut-off, passes longer waves; periods H/2 L H/2 |
| Shortpass | reflects the band just above the cut-off, passes shorter waves; periods L/2 H L/2 |
| Band | reflects a band centred on the cut-off; periods H L |
| Placement | new mirror rotated 45°: a beam along +x meets the stack at 45°, the reflected beam leaves along +y (down) |
| Read-out | cut-off and band centre, reflected band in px and frequency levels, peak R, layer and stack thickness, R and T for every source (transfer-matrix method, s-polarisation), warning for layers under 2 px |

## Probes

| Type | Placement | Output |
|---|---|---|
| Point (field / intensity) | one click | field amplitude, intensity, or integrated intensity per step |
| Line (A → B) | two clicks | field or intensity profile per frame, or integrated profile |
| Counting detector | two clicks | random detections along a line per step (photomultiplier) |
| Imaging detector | two clicks | accumulated detections per pixel bin (photon-counting array) |

| Setting | Values | Meaning |
|---|---|---|
| Layer | Field, Intensity | what is measured |
| Measurement | Amplitude, Value, IntegratedValue | raw field, value, or running sum |
| Kind | Point, Line | point A, or the line A → B |
| Detector | Standard, Counting, Imaging | plain probe or photon detector |
| A x/y, B x/y | px | ends (dragged on the canvas) |
| Start after step | step | ignore the start-up transient |
| Detector sensitivity, Bin size | value, px | detection rate; imaging bin width |
| Box size, Box x/y/width/height | Tiny … ExtraLarge, Custom | graph box on the canvas |
| Step axis, Window | Running / Full, steps | running window or the whole run |
| Auto-scale steps / values, Value minimum / maximum | on/off, values | graph ranges |
| Value axis, Decades | Linear / Logarithmic, count | value scale |

| Graph | Description |
|---|---|
| Graph column | **Probe graphs** column at the right edge; opens when a run with probes starts (*Graph column*) |
| Graphs on the canvas | boxes over the picture of the measured layer (*Graphs on the canvas*, off by default); drag to move, corner grip to resize |
| Per-wavelength curves | with separate wavelengths, intensity probes (point and line) show each wavelength's share in its colour beside the total (*Probe curves per colour*) |
| CSV | **File → Export → Probe data as CSV**: one row per step (point) or per frame (line), `;`-separated |

## Display

| Item | Description |
|---|---|
| Pictures | field, intensity or both; stacked or side by side; each a panel with a header |
| Colour | 23 colour schemes plus project schemes; brightness, black level, **Auto** brightness |
| True colour | with separate wavelengths: each wavelength through a black-to-colour ramp, added like light; colours from the wavelength (shortest 450 nm … longest 650 nm, or a set nm per px); colour key above the intensity panel |
| Grid | on by default, step 25 px, opacity; object centres snap to lines and nodes (*Snap*); never exported |
| Markers | part outlines, source / part / probe markers and labels; handles of every object or of the selected one only |
| Preview | resolution full, half, quarter; frames kept: every 1st, 2nd or 4th |
| Playback | export frame rate × speed (0.25–8); loop over the export range |

## Interface

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
│                              │ Zoom · Grid · Snap · Graphs · True colour            │                │
├──────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Status bar: last message · run state                                                                 │
└──────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

| Element | Description |
|---|---|
| Project tree | Project (seed, description), Scene, Simulation, Display; groups Sources, Optical parts, Probes, Colour schemes; per row: Enabled, move up/down, remove |
| Properties | form of the current object: categories, a hint on every label, kind description for parts, read-outs; values apply as you type |
| Picture header | layer, step, brightness box, **Auto** |
| Picture frame | red: reflective edges; blue: damping rim outside the canvas; grey: open edges |
| Handles | rotation lever (Shift: 15° steps), corner grip (size), lens surface handles (radii), outline points, fibre nodes and Bézier handles, probe ends, graph box grip |
| Timeline | ticks in frames; computed part shaded; in/out brackets = export range; playhead |
| Transport | Run, Pause/Resume run, Stop, Play/Pause, frame back/forward, In, Out, Clear, Loop, Live (follow the newest frame), position (frame and step), speed, shown layers, arrangement |
| View row | Zoom −, zoom %, +, Fit; Grid, grid step, Snap, grid opacity; Graphs on the canvas, Graph column; Separate wavelengths, True colour |
| Canvas zoom | 100 % (whole canvas) to 3200 %, both pictures together; wheel zooms at the pointer (or pans: `app.canvas.wheel: pan`) |
| Canvas pan | zoomed: drag empty canvas; anywhere: right or middle drag, Alt + drag |
| Selection | click; Ctrl/Cmd + click toggles; Shift + click: range in the tree, add on the canvas; marquee on empty canvas; Ctrl/Cmd + A all |
| Undo / redo | unlimited; an edit becomes a step when finished (mouse released, field left) |
| Clipboard | cut, copy, paste, duplicate of sources, parts, probes and colour schemes, as XML text |
| Colour schemes | project schemes from any scheme; gradient editor: click adds a stop, drag moves it, drag away or double-click removes it |
| Status bar | last message (warnings yellow, errors red), object being placed, run state and speed |

| Menu | Entries |
|---|---|
| File | New, Open, Recent projects, Examples, Save, Save as, Open scene image, Blank canvas, Export (Field, Intensity, Both, Probe data as CSV), Export last, Save shown picture as PNG, Exit |
| Edit | Undo, Redo, Cut, Copy, Paste, Duplicate, Delete, Select all |
| View | Panes (project tree, properties, status bar, probe graphs), preview resolution, preview frames, UI zoom, canvas zoom in / out / whole canvas, grid, handles, labels, graphs on the canvas |
| Simulation | Run, Pause/resume run, Stop, Play/Pause, compute backend, re-run after edits, separate wavelengths, memory warning |
| Help | Manual (this file), About (versions, GPU, system information) |

## Controls

| Key / action | Function |
|---|---|
| F5 / Shift + F5 / F6 | Run / Stop / Pause or resume the run |
| Space, F7 | Play / Pause |
| `,` `.` | one frame back / forward |
| I, O, X | in point, out point, clear the export range |
| L, F | loop, live |
| Ctrl/Cmd + N, O, S, Shift + S | new, open, save, save as |
| Ctrl/Cmd + E | export last |
| Ctrl/Cmd + Z, Y (Shift + Z) | undo, redo |
| Ctrl/Cmd + X, C, V, D, A | cut, copy, paste, duplicate, select all |
| Ctrl/Cmd + =, -, 0 | zoom in, zoom out, whole canvas |
| F1 | manual |
| Wheel / two-finger scroll | zoom at the pointer (wheel mode `pan`: scroll, Shift sideways, Ctrl/Cmd zooms) |
| Drag on empty canvas | marquee; zoomed: pan (Shift or Ctrl/Cmd + drag: marquee) |
| Right or middle drag, Alt + drag | pan |
| Click / drag on an object | select / move (centres snap to the grid when *Snap* is on) |
| Arrows (Shift: ×10), Tab | after a canvas click: nudge the selection, select the next object |
| Del, Backspace | after a canvas click: remove the selected objects |
| Click after **+ Source / Part / Probe** | place the object (lines and detectors: two clicks, A then B) |
| Esc | cancel a drag, cancel placing, or clear the selection |

## Export

| Format | Content |
|---|---|
| MP4 (H.264), MPEG-1 | video (needs ffmpeg) |
| AVI | uncompressed video |
| QuickTime | Animation (lossless RLE) or PNG codec |
| PNG / JPEG sequence | numbered frames in a folder |
| OpenEXR | 16-bit float data, intensity as I or √I |

| Option | Description |
|---|---|
| Layers | field, intensity, or both (one file each) |
| Frame rate, Quality | video timing; compression quality (MP4, MPEG-1, JPEG) |
| From / To step | export range (preset from the timeline's in/out points) |
| Part outlines | glowing scene edges |
| Probe graphs | graph boxes, with per-wavelength curves when *Probe curves per colour* is on |
| Probe markers, Source markers | probe ends with numbers; a ring on every emitter |
| Labels | names of parts and sources; of probes when their graphs are not exported |
| Export shown frame | one full-resolution still of the current step |
| Export last | the session's last export again, without the dialog (Ctrl/Cmd + E) |

## Projects

| Item | Description |
|---|---|
| Format | `.wdproj` (XML) |
| Stored | scene and embedded image, sources, parts, probes, simulation and display settings, colour schemes, random seed |
| Missing attributes | read as the format's defaults |

## Defaults

File: `src/pywavedude/defaults.yaml`, read at start (`--defaults FILE` reads another). Every key must name an
existing setting; an unknown key or a wrong value stops the start with the key's path.

| Section | Contents |
|---|---|
| `app.backend` | compute backend at start (`--backend` overrides it) |
| `app.ui` | window size, left pane width, tree pane height, probe graph column width, font size, UI zoom and its choices, rows per tree group |
| `app.canvas` | pick radius, snap radius, rotation lever length (screen pixels), wheel mode (zoom / pan), pan step, zoom step per wheel notch, largest zoom |
| `app.playback` | speed and its choices, loop, live |
| `app.run` | memory warning on/off, memory budget as a share of the physical memory |
| `app.export` | file name, format (`MP4`, `MPEG1`, `AVI`, `MOV`, `MOV_PNG`, `PNG`, `JPEG`, `EXR`, `EXR_ROOT`), layers, part outlines |
| `new_project.scene` / `.simulation` / `.display` | every setting of a new project |
| `new_project.sources` | sources of a new project (list of source settings with `name`, `x`, `y`) |
| `new_objects.source`, `.probe` | settings of a newly placed source or probe |
| `new_objects.parts.<Kind>` | settings of a newly placed part, per kind (`LensPart`, `DichroicMirrorPart`, …) |

Values in a saved project override these defaults.

## Benchmark

```bash
uv run python -m pywavedude.bench [width height steps]
```

## License

GPL-3.0-or-later. See [LICENSE.md](LICENSE.md). The bundled Inter font is under the SIL Open Font License 1.1
([OFL.txt](src/pywavedude/assets/fonts/OFL.txt)).
