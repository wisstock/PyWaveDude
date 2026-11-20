# SPDX-License-Identifier: GPL-3.0-or-later
# PyWaveDude: Python port of WaveDude Next (C) Alexander Graef, after Jeroen Vleggaar (Huygens Optics) and 0x23.
"""Labels, hints and categories of every editable field, for the properties pane.

``ui_meta.json`` holds the original's texts (display name, description, category per property, the kind
descriptions and the enum members' names); this module maps the Python dataclasses onto them and adds the fields
this port has on its own."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class FieldMeta:
    label: str
    hint: str = ""
    category: str = "General"


# Python class name → the original's class name
CLASS_NAMES = {"PointSource": "Source", "SceneSettings": "Scene", "SimSettings": "Simulation",
               "DisplaySettings": "Display"}
# fields whose original property has another name (or lives in a nested class)
FIELD_NAMES = {
    ("PointSource", "level"): ("Source", "Frequency"),
    ("PointSource", "composite"): ("Source", "IsComposite"),
    ("PointSource", "size_x"): ("SourceArray", "SizeX"), ("PointSource", "size_y"): ("SourceArray", "SizeY"),
    ("PointSource", "spacing_x"): ("SourceArray", "SpacingX"),
    ("PointSource", "spacing_y"): ("SourceArray", "SpacingY"),
    ("PointSource", "array_angle"): ("SourceArray", "Angle"),
    ("PointSource", "random_frequency"): ("SourceArray", "RandomFrequency"),
    ("PointSource", "random_phase"): ("SourceArray", "RandomPhase"),
    ("PointSource", "random_off_width"): ("SourceArray", "RandomOffWidth"),
    ("PointSource", "random_position"): ("SourceArray", "RandomPosition"),
    ("SceneSettings", "open_edges"): ("Scene", "Edges"),
    ("SimSettings", "fourth_order"): ("Simulation", "FourthOrderLaplacian"),
    ("Probe", "ax"): ("Probe", "AX"), ("Probe", "ay"): ("Probe", "AY"),
    ("Probe", "bx"): ("Probe", "BX"), ("Probe", "by"): ("Probe", "BY"),
    ("DisplaySettings", "fps"): ("Video", "Fps"), ("DisplaySettings", "quality"): ("Video", "Quality"),
    ("DisplaySettings", "outline"): ("DisplayLayer", "Overlay"),
    ("DisplaySettings", "outline_gain"): ("DisplayLayer", "OverlayGain"),
}
LAYER_FIELDS = {"scheme": "ColorScheme", "brightness": "Brightness", "black": "BlackLevel"}

# this port's own fields
OWN = {
    ("SceneSettings", "use_image"): FieldMeta("Scene image", "Use the embedded scene image; off: a blank canvas of "
                                              "the width and height.", "Canvas"),
    ("SimSettings", "backend"): FieldMeta("Compute", "auto: the Metal GPU on Apple Silicon, else the multithreaded "
                                          "CPU; or force one.", "Compute"),
    ("SimSettings", "preview"): FieldMeta("Preview resolution", "The preview run computes the full canvas and keeps "
                                          "its frames at this resolution; exports are always full size.", "Compute"),
    ("SimSettings", "rerun_on_edit"): FieldMeta("Re-run after edits", "A finished edit of the scene, a source or a "
                                                "part starts a new run by itself.", "Compute"),
    ("SimSettings", "split_wavelengths"): FieldMeta(
        "Separate wavelengths", "One simulation per source wavelength on the same scene: the field is their sum, the "
        "intensity is kept per wavelength (incoherent between wavelengths), for the true colour display. Costs one "
        "run per wavelength.", "Solver"),
    ("DisplaySettings", "layer"): FieldMeta("Pictures", "Which pictures the preview shows.", "Preview"),
    ("DisplaySettings", "arrangement"): FieldMeta("Arrangement", "Pictures one above the other or side by side.",
                                                  "Preview"),
    ("DisplaySettings", "markers"): FieldMeta("Markers", "Source, part and probe markers over the pictures.",
                                              "Preview"),
    ("DisplaySettings", "auto_brightness"): FieldMeta("Keep auto brightness", "Fit the brightness to every frame "
                                                      "shown.", "Preview"),
    ("DisplaySettings", "graphs_in_preview"): FieldMeta("Graphs on the canvas", "Probe graph boxes over the pictures.",
                                                        "Preview"),
    ("DisplaySettings", "probe_colours"): FieldMeta(
        "Probe curves per colour", "With separate wavelengths: intensity probe graphs show each wavelength's share in "
        "its colour beside the total.", "Graph"),
    ("DisplaySettings", "handles"): FieldMeta("Handles in the preview", "Handles of every object; off: only the "
                                              "selected object's.", "Preview"),
    ("DisplaySettings", "labels"): FieldMeta("Labels in the preview", "Names beside the handles; off: only the "
                                             "selected object's.", "Preview"),
    ("DisplaySettings", "true_colour"): FieldMeta(
        "True colour intensity", "With separate wavelengths: each wavelength's intensity through a black-to-colour "
        "ramp, the colours added like light.", "True colour"),
    ("DisplaySettings", "nm_per_px"): FieldMeta(
        "nm per px of wavelength", "Colour of a wavelength: λ in px × this, in nm; 0 spreads the wavelengths over "
        "450–650 nm.", "True colour"),
    ("DisplaySettings", "labels_in_export"): FieldMeta(
        "Labels in export", "Names of the enabled parts and sources in exported frames, and of the active probes when their graphs "
        "(titled with the name) are not exported.", "Export"),
    ("DisplaySettings", "grid"): FieldMeta("Grid", "Grid lines over the preview; never exported.", "Grid"),
    ("DisplaySettings", "grid_step"): FieldMeta("Grid step", "Distance of the grid lines in canvas pixels.", "Grid"),
    ("DisplaySettings", "grid_opacity"): FieldMeta("Grid opacity", "0 invisible, 1 opaque.", "Grid"),
    ("DisplaySettings", "grid_snap"): FieldMeta("Snap to grid", "Centres of moved and placed objects stick to grid "
                                                "lines and nodes.", "Grid"),
    ("PhaseStepPlatePart", "aperture"): FieldMeta("Aperture", "Width of the plate across the beam in pixels.",
                                                  "Plate"),
    ("PhaseStepPlatePart", "base_thickness"): FieldMeta("Base thickness", "Thickness of the thin zones in pixels.",
                                                        "Plate"),
    ("PhaseStepPlatePart", "zones"): FieldMeta("Zones", "Zones across the aperture, alternating thin and thick; 2 is "
                                               "the 2D vortex plate.", "Plate"),
    ("PhaseStepPlatePart", "phase_step"): FieldMeta("Phase step", "Delay of the thick zones against the thin ones, "
                                                    "in degrees (180: antiphase).", "Design"),
    ("PhaseStepPlatePart", "design_frequency"): FieldMeta("Design frequency", "The source frequency level the step "
                                                          "height is made for.", "Design"),
    ("PhaseStepPlatePart", "medium_index"): FieldMeta("Medium index", "Refractive index around the plate.", "Design"),
    ("DichroicMirrorPart", "length"): FieldMeta("Substrate thickness", "Thickness of the substrate behind the "
                                                "stack, in pixels.", "Plate"),
    ("DichroicMirrorPart", "width"): FieldMeta("Width", "Width of the mirror across its axis, in pixels.", "Plate"),
    ("DichroicMirrorPart", "mode"): FieldMeta(
        "Type", "Longpass: reflects the band just below the cut-off wavelength, passes longer waves. Shortpass: "
        "reflects the band just above it, passes shorter waves. Band: reflects a band centred on it.", "Design"),
    ("DichroicMirrorPart", "design_frequency"): FieldMeta("Cut-off frequency", "The source frequency level at the "
                                                          "band edge (Band: at its centre).", "Design"),
    ("DichroicMirrorPart", "design_angle"): FieldMeta("Design angle", "Angle of incidence the layers are made for, "
                                                      "in degrees from the normal (0–80).", "Design"),
    ("DichroicMirrorPart", "pairs"): FieldMeta("Layer pairs", "High/low index periods; more pairs reflect more and "
                                               "sharpen the edges.", "Design"),
    ("DichroicMirrorPart", "index_high"): FieldMeta("High index", "Refractive index of the high layers.", "Design"),
    ("DichroicMirrorPart", "index_low"): FieldMeta("Low index", "Refractive index of the low layers; the larger the "
                                                   "contrast, the wider the band.", "Design"),
    ("DichroicMirrorPart", "smooth_edges"): FieldMeta("Half-thickness end layers", "Edge-filter periods (L/2 H L/2 "
                                                      "or H/2 L H/2) that flatten the passband ripple.", "Design"),
    ("DichroicMirrorPart", "medium_index"): FieldMeta("Medium index", "Refractive index the light arrives from.",
                                                      "Design"),
}
OWN_KINDS = {"PhaseStepPlatePart": "A phase step plate, the 2D counterpart of a vortex phase plate: the aperture is "
                                   "split across the beam into zones that alternate between two thicknesses, so "
                                   "neighbouring zones delay the wave by the phase step. Two zones at 180° give a "
                                   "focus with a dark zero on the axis between two lobes, as used for STED depletion.",
             "DichroicMirrorPart": "A dichroic mirror: a substrate whose front face carries a quarter-wave stack of "
                                   "two indices, designed for a cut-off wavelength at an angle of incidence. One band "
                                   "of wavelengths is reflected by thin-film interference, the rest passes. The stack "
                                   "faces the part's −x side; rotate the part to the design angle."}
CATEGORY_ORDER = ["General", "Position", "Canvas", "Sources", "Run", "Compute", "Solver", "Wave", "Pulsing", "Array",
                  "Grid", "Randomisation", "Measurement", "Graph", "Axes", "Preview", "Outputs", "Rendering",
                  "True colour", "Scene outline", "Video", "Export"]



@lru_cache(maxsize=1)
def _data() -> dict:
    return json.loads((Path(__file__).parent / "ui_meta.json").read_text(encoding="utf-8"))


def _pascal(name: str) -> str:
    return "".join(p[:1].upper() + p[1:] for p in name.split("_"))


def _lookup(cls_name: str, prop: str):
    classes = _data()["classes"]
    while cls_name in classes:
        p = classes[cls_name]["props"].get(prop)
        if p is not None:
            return p
        cls_name = classes[cls_name]["base"]
    return None


def field_meta(obj, name: str) -> FieldMeta:
    """Label, hint and category of ``obj.name``, the original's where it has one."""
    for cls in type(obj).__mro__:
        own = OWN.get((cls.__name__, name))
        if own is not None:
            return own
    for cls in type(obj).__mro__:
        key = (cls.__name__, name)
        if key in FIELD_NAMES:
            p = _lookup(*FIELD_NAMES[key])
            if p:
                return FieldMeta(p["label"] or _pascal(name), p["hint"] or "", p["category"] or "General")
    if type(obj).__name__ == "DisplaySettings":
        for prefix in ("field_", "intensity_"):
            if name.startswith(prefix) and name[len(prefix):] in LAYER_FIELDS:
                p = _lookup("DisplayLayer", LAYER_FIELDS[name[len(prefix):]])
                if p:
                    return FieldMeta(f"{prefix[:-1].capitalize()} {p['label'].lower()}", p["hint"] or "",
                                     prefix[:-1].capitalize())
    for cls in type(obj).__mro__:
        p = _lookup(CLASS_NAMES.get(cls.__name__, cls.__name__), _pascal(name))
        if p:
            return FieldMeta(p["label"] or _pascal(name), p["hint"] or "", p["category"] or "General")
    return FieldMeta(name.replace("_", " ").capitalize())


def kind_description(obj) -> str:
    name = type(obj).__name__
    return OWN_KINDS.get(name) or _data()["kinds"].get(name, "")


def enum_label(value: Enum) -> str:
    return _data()["enums"].get(f"{type(value).__name__}.{value.value}", str(value.value))


def category_rank(cat: str) -> int:
    return CATEGORY_ORDER.index(cat) if cat in CATEGORY_ORDER else len(CATEGORY_ORDER)
