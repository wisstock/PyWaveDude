# SPDX-License-Identifier: GPL-3.0-or-later
# PyWaveDude: Python port of WaveDude Next (C) Alexander Graef, after Jeroen Vleggaar (Huygens Optics) and 0x23.
"""The project and its ``.wdproj`` file (port of ``Model/Project.cs`` and the XmlSerializer layout it uses).

The file is sparse XML: an attribute per property that differs from its default, named as the C# property
(``RefractiveIndex``, ``AX``), children for lists (``<Node>``, ``<Point>``, ``<Layer>``, ``<Array>``). Attributes this
port does not model (graph boxes, export flags, …) are kept on the object they were read with and written back,
so a round trip through PyWaveDude does not lose them. Version upgrades of older files follow ``Project.Load``.
"""

from __future__ import annotations

import base64
import dataclasses
import io
import math
import random
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np

from .engine.sources import PointSource
from .parts import model as pm
from .probes import Probe, ProbeBoxSize, ProbeKind
from .render import ColorScheme

CURRENT_VERSION = 4
EXTRA = "_xml_extra"          # unknown attributes kept per object: {name: text}
EXTRA_CHILDREN = "_xml_children"


# ====================================================================== the project
@dataclass
class SceneSettings:
    use_image: bool = False            # Mode = Image
    image_name: str = ""               # Image: the file name shown
    image_path: str = ""               # ImagePath: where it came from (relative to the project when possible)
    width: int = 1920
    height: int = 1080
    refractive_index: float = 1.0
    open_edges: bool = True            # Edges = Open / Reflective
    damping_edge: int = -32
    image_sources: bool = False
    image_source_scale: float = 1.0
    object_sources: bool = True


@dataclass
class SimSettings:
    steps: int = 2500
    visualize_every: int = 2
    fourth_order: bool = True
    wave_speed: float = 1.0
    source_frequency_scale: float = 2.0
    time_decay: float = 10.0
    intensity_scale: float = 1.0
    # the app's own, not in the file
    backend: str = "auto"
    preview: str = "Half"
    rerun_on_edit: bool = True
    split_wavelengths: bool = False         # one simulation per wavelength: per-colour intensity
    preview_frames: int = 1                 # the preview run keeps every n-th frame (1, 2, 4)


@dataclass
class DisplaySettings:
    layer: str = "Both"
    arrangement: str = "Stacked"            # the app's own: "Stacked" or "Side by side"
    handles: bool = True                    # handles of every object (off: the selected object's only)
    labels: bool = True                     # names beside the handles
    field_scheme: str = "Default field"
    intensity_scheme: str = "Default intensity"
    field_brightness: float = 30.0
    intensity_brightness: float = 30.0
    field_black: float = -0.05
    intensity_black: float = 0.0
    markers: bool = True
    outline: bool = True
    outline_gain: float = 3.0
    auto_brightness: bool = False
    fps: int = 60
    quality: int = 85
    graphs_in_preview: bool = False         # the app's own (not written to project files) ...
    probe_colours: bool = True              # intensity probe graphs: a curve per wavelength (separate wavelengths)
    grid: bool = True                       # ... grid over the preview, never exported
    grid_step: int = 25
    grid_opacity: float = 0.25
    grid_snap: bool = True                  # element centres stick to grid lines and nodes
    true_colour: bool = True                # split runs: each wavelength's intensity in its own colour
    nm_per_px: float = 0.0                  # wavelength in px → nm for the colours; 0 spreads them over 450–650 nm
    export_from: int = 0
    export_to: int = 0
    probe_graphs_in_export: bool = True
    probe_markers_in_export: bool = True
    source_markers_in_export: bool = False


@dataclass
class Project:
    scene: SceneSettings = field(default_factory=SceneSettings)
    sim: SimSettings = field(default_factory=SimSettings)
    display: DisplaySettings = field(default_factory=DisplaySettings)
    sources: list[PointSource] = field(default_factory=list)
    parts: list[pm.OpticalPart] = field(default_factory=list)
    probes: list[Probe] = field(default_factory=list)
    color_schemes: list[ColorScheme] = field(default_factory=list)
    seed: int = field(default_factory=lambda: random.randint(1, 2 ** 31 - 2))
    description: str = ""
    image_bytes: bytes | None = None   # ImageData: the embedded scene image file, as it was read
    path: str = ""                     # where it was loaded from or saved to (not written)

    def scene_rgb(self) -> np.ndarray | None:
        """The embedded scene image as RGB uint8, or None for a blank canvas."""
        if not self.scene.use_image or not self.image_bytes:
            return None
        from PIL import Image
        with Image.open(io.BytesIO(self.image_bytes)) as im:
            return np.asarray(im.convert("RGB")).copy()

    def embed_image(self, path: str) -> np.ndarray:
        """Embed an image file (its bytes as they are) and switch the scene to it."""
        data = Path(path).read_bytes()
        self.image_bytes = data
        self.scene.use_image = True
        self.scene.image_name = Path(path).name
        self.scene.image_path = str(path)
        rgb = self.scene_rgb()
        self.scene.height, self.scene.width = rgb.shape[:2]
        return rgb


# ====================================================================== names and values
SPECIAL = {"ax": "AX", "ay": "AY", "bx": "BX", "by": "BY", "r1": "R1", "r2": "R2", "y_min": "YMin", "y_max": "YMax",
           "y_scale": "YScale"}
PART_ELEMENTS = {
    "Lens": pm.LensPart, "Polygon": pm.PolygonPart, "Rectangle": pm.RectanglePart, "Ellipse": pm.EllipsePart,
    "Triangle": pm.TrianglePart, "Outline": pm.OutlinePart, "Fiber": pm.FiberPart,
    "BraggReflector": pm.BraggReflectorPart, "GrinLens": pm.GrinLensPart, "XorGate": pm.XorGatePart,
    "AndGate": pm.AndGatePart, "OrGate": pm.OrGatePart, "ZonePlate": pm.ZonePlatePart, "Diffuser": pm.DiffuserPart,
    "InterferenceFilter": pm.InterferenceFilterPart, "SimpleGrating": pm.SimpleGratingPart,
    "FacettedGrating": pm.FacettedGratingPart, "MetaLens": pm.MetaLensPart, "MetaPrism": pm.MetaPrismPart,
    "MetaAxicon": pm.MetaAxiconPart, "FresnelLens": pm.FresnelLensPart,
    "PhaseStepPlate": pm.PhaseStepPlatePart, "DichroicMirror": pm.DichroicMirrorPart,   # this port only
}
ELEMENT_OF = {cls: name for name, cls in PART_ELEMENTS.items()}
# what an absent attribute means in a file: the C# initialisers, where a new part from the tree is set up otherwise
FILE_DEFAULTS = {
    pm.FresnelLensPart: dict(diameter=200, r1=300.0, r2=-300.0, thickness=40),
    pm.SimpleGratingPart: dict(mirror=False),
    pm.DiffuserPart: dict(front_coated=True, sides_coated=True),
    pm.MetaLensPart: dict(sides_coated=True), pm.MetaPrismPart: dict(sides_coated=True),
    pm.MetaAxiconPart: dict(sides_coated=True),
    pm.InterferenceFilterPart: dict(layers=[]), pm.FiberPart: dict(nodes=[]), pm.OutlinePart: dict(points=[]),
}
NOT_IN_FILE = {"project_seed", "frequency_scale", "composite", "size_x", "size_y", "spacing_x", "spacing_y",
               "array_angle", "random_frequency", "random_phase", "random_off_width", "random_position"}
ARRAY_FIELDS = {"size_x": "SizeX", "size_y": "SizeY", "spacing_x": "SpacingX", "spacing_y": "SpacingY",
                "array_angle": "Angle", "random_frequency": "RandomFrequency", "random_phase": "RandomPhase",
                "random_off_width": "RandomOffWidth", "random_position": "RandomPosition"}


def xml_name(name: str, obj=None) -> str:
    if isinstance(obj, PointSource) and name == "level":
        return "Frequency"
    return SPECIAL.get(name) or "".join(p[:1].upper() + p[1:] for p in name.split("_"))


def parse(text: str, like):
    if isinstance(like, bool):
        return text.strip().lower() == "true"
    if isinstance(like, Enum):
        return type(like)(text)
    if isinstance(like, int):
        return int(float(text))
    if isinstance(like, float):
        return float(text)
    return text


def fmt(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, float):
        if math.isfinite(value) and value.is_integer():
            return str(int(value))
        return repr(value).upper()          # .NET writes 5E-05
    return str(value)


def _defaults(cls):
    obj = cls()
    for k, v in FILE_DEFAULTS.get(cls, {}).items():
        setattr(obj, k, v)
    return obj


def _read_attrs(obj, el: ET.Element, skip=()) -> None:
    """Fill ``obj``'s scalar dataclass fields from ``el``'s attributes; keep the unknown ones."""
    known = {}
    for f in dataclasses.fields(obj):
        if f.name in NOT_IN_FILE or isinstance(getattr(obj, f.name), list):
            continue
        known[xml_name(f.name, obj)] = f.name
    extra = {}
    for key, text in el.attrib.items():
        if key in skip:
            continue
        name = known.get(key)
        if name is None:
            extra[key] = text
            continue
        try:
            setattr(obj, name, parse(text, getattr(obj, name)))
        except (ValueError, KeyError):
            extra[key] = text
    if extra:
        setattr(obj, EXTRA, extra)


def _always(obj) -> set[str]:
    """Properties without a [DefaultValue] in the original: written whatever their value."""
    if isinstance(obj, (pm.OpticalPart, PointSource)):
        extra = {"thickness", "coating"} if isinstance(obj, pm.ZonePlatePart) else set()
        return {"name", "x", "y"} | extra
    if isinstance(obj, Probe):
        return {"name", "ax", "ay"}
    if isinstance(obj, pm.FiberNode):
        return {"x", "y"}
    return set()


def _skipped(obj) -> set[str]:
    """The original's ShouldSerialize* rules."""
    if isinstance(obj, Probe):
        out = set()
        if obj.kind != ProbeKind.LINE:
            out |= {"bx", "by"}
        if obj.box_size != ProbeBoxSize.CUSTOM:
            out |= {"box_width", "box_height"}
        return out
    return set()


def _write_attrs(obj, el: ET.Element, default) -> None:
    always = _always(obj)
    skipped = _skipped(obj)
    for f in dataclasses.fields(obj):
        if f.name in NOT_IN_FILE or f.name in skipped:
            continue
        v = getattr(obj, f.name)
        if isinstance(v, list) or (v == getattr(default, f.name) and f.name not in always):
            continue
        el.set(xml_name(f.name, obj), fmt(v))
    for k, v in getattr(obj, EXTRA, {}).items():
        el.attrib.setdefault(k, v)


# ====================================================================== load
def load(path: str | Path) -> Project:
    path = Path(path)
    p = loads(path.read_bytes(), path.parent)
    p.path = str(path)
    return p


def loads(data: bytes | str, folder: Path | None = None, image_bytes: bytes | None = None) -> Project:
    """A project from its XML; ``image_bytes`` stands in for a scene image the XML does not embed."""
    root = ET.fromstring(data)
    if root.tag != "Project":
        raise ValueError("The document is not a WaveDude Next project.")
    p = Project(seed=int(root.get("Seed", "0")))
    version = int(root.get("Version", "1"))
    desc = root.find("Description")
    p.description = desc.text or "" if desc is not None else ""
    p.image_bytes = image_bytes
    _load_scene(p, root.find("Scene"), version, folder or Path.cwd())
    _load_simulation(p, root.find("Simulation"))
    _load_display(p, root.find("Display"), version)
    _load_app(p, root.find(APP_ELEMENT))
    version_items = _load_items(root, version)
    p.sources += version_items["sources"]
    p.parts += version_items["parts"]
    p.probes += version_items["probes"]
    p.color_schemes += version_items["color_schemes"]
    for part in p.parts:
        _attach(p, part)
    return p


def _attach(p: Project, part) -> None:
    if hasattr(part, "frequency_scale"):
        part.frequency_scale = p.sim.source_frequency_scale
    if hasattr(part, "project_seed"):
        part.project_seed = p.seed


def _load_items(root, version: int = CURRENT_VERSION) -> dict:
    out = {"sources": [], "parts": [], "probes": [], "color_schemes": []}
    for el in root:
        tag = el.tag
        if tag == "Source":
            out["sources"].append(_load_source(el))
        elif tag in PART_ELEMENTS:
            out["parts"].append(_load_part(el, PART_ELEMENTS[tag], version))
        elif tag == "Probe":
            pr = Probe()
            _read_attrs(pr, el)
            # the original's box follows point A while the two coincide, so an absent box coordinate is A's
            if "BoxX" not in el.attrib:
                pr.box_x = pr.ax
            if "BoxY" not in el.attrib:
                pr.box_y = pr.ay
            pr.normalize()
            out["probes"].append(pr)
        elif tag == "ColorScheme":
            out["color_schemes"].append(ColorScheme.from_element(el))
    return out


# ====================================================================== objects as text (clipboard)
FRAGMENT = "WaveDudeObjects"


def items_to_xml(items) -> str:
    """Sources, parts, probes and colour schemes as an XML fragment of their file elements."""
    root = ET.Element(FRAGMENT)
    for obj in items:
        root.append(_item_element(obj))
    ET.indent(root, space="\t")
    return ET.tostring(root, encoding="unicode")


def items_from_xml(text: str) -> dict:
    """What ``items_to_xml`` wrote (or a whole project's elements): ``{"sources": [...], "parts": ..., ...}``."""
    root = ET.fromstring(text.strip())
    if root.tag not in (FRAGMENT, "Project"):
        raise ValueError("no project objects")
    return _load_items(root)


def _load_scene(p: Project, el, version: int, folder: Path) -> None:
    s = p.scene
    if el is not None:
        s.use_image = el.get("Mode", "Canvas") == "Image"
        s.image_name = el.get("Image", "")
        s.image_path = el.get("ImagePath", "")
        for attr, name in (("Width", "width"), ("Height", "height"), ("DampingEdge", "damping_edge"),
                           ("RefractiveIndex", "refractive_index"), ("ImageSources", "image_sources"),
                           ("ImageSourceScale", "image_source_scale"), ("ObjectSources", "object_sources")):
            if attr in el.attrib:
                setattr(s, name, parse(el.get(attr), getattr(s, name)))
        s.open_edges = el.get("Edges", "Open") == "Open"
        data = el.find("ImageData")
        if data is not None and data.text:
            p.image_bytes = base64.b64decode(data.text)
        elif s.use_image and s.image_path and not p.image_bytes:
            f = Path(s.image_path)
            f = f if f.is_absolute() else folder / f
            if f.exists():
                p.image_bytes = f.read_bytes()
        if p.image_bytes and s.use_image:
            rgb = p.scene_rgb()
            s.height, s.width = rgb.shape[:2]
    if version < 2:
        s.open_edges = False          # version 1 knew only reflective edges
    if version < 3 and (el is None or "DampingEdge" not in el.attrib):
        s.damping_edge = 0            # until version 3 an absent damping edge meant none


def _load_simulation(p: Project, el) -> None:
    if el is None:
        return
    m = p.sim
    for attr, name in (("Steps", "steps"), ("VisualizeEvery", "visualize_every"),
                       ("FourthOrderLaplacian", "fourth_order"), ("WaveSpeed", "wave_speed"),
                       ("TimeDecay", "time_decay"), ("SourceFrequencyScale", "source_frequency_scale"),
                       ("IntensityScale", "intensity_scale")):
        if attr in el.attrib:
            setattr(m, name, parse(el.get(attr), getattr(m, name)))


def _load_display(p: Project, el, version: int) -> None:
    d = p.display
    if el is None:
        return
    for tag, prefix in (("Field", "field"), ("Intensity", "intensity")):
        layer = el.find(tag)
        if layer is None:
            continue
        if layer.get("ColorScheme"):
            setattr(d, f"{prefix}_scheme", layer.get("ColorScheme"))
        if "Brightness" in layer.attrib:
            setattr(d, f"{prefix}_brightness", float(layer.get("Brightness")))
        if "BlackLevel" in layer.attrib:
            setattr(d, f"{prefix}_black", float(layer.get("BlackLevel")))
        if tag == "Field":
            d.outline = parse(layer.get("Overlay", "true"), True)
            d.outline_gain = float(layer.get("OverlayGain", "3"))
    for attr, name in (("ExportFrom", "export_from"), ("ExportTo", "export_to"),
                       ("ProbeGraphsInExport", "probe_graphs_in_export"),
                       ("ProbeMarkersInExport", "probe_markers_in_export"),
                       ("SourceMarkersInExport", "source_markers_in_export")):
        if attr in el.attrib:
            setattr(d, name, parse(el.get(attr), getattr(d, name)))
    video = el.find("Video")
    if video is not None:
        d.fps = int(video.get("Fps", d.fps))
        d.quality = int(video.get("Quality", d.quality))


APP_ELEMENT = "PyWaveDude"
APP_SETTINGS = (("sim", "split_wavelengths", "SplitWavelengths"), ("display", "graphs_in_preview", "GraphsInPreview"),
                ("display", "probe_colours", "ProbeColours"),
                ("display", "grid", "Grid"), ("display", "grid_step", "GridStep"),
                ("display", "grid_opacity", "GridOpacity"), ("display", "grid_snap", "GridSnap"),
                ("display", "true_colour", "TrueColour"), ("display", "nm_per_px", "NmPerPixel"),
                ("display", "layer", "Layer"), ("display", "arrangement", "Arrangement"),
                ("display", "handles", "HandlesInPreview"), ("display", "labels", "LabelsInPreview"),
                ("sim", "preview", "PreviewResolution"), ("sim", "preview_frames", "PreviewFrames"))


def _load_app(p: Project, el) -> None:
    """Settings of this port only, kept in a ``<PyWaveDude>`` element written only when one differs from its default."""
    if el is None:
        return
    for group, name, attr in APP_SETTINGS:
        if attr in el.attrib:
            obj = getattr(p, group)
            setattr(obj, name, parse(el.get(attr), getattr(obj, name)))


def _save_app(p: Project, root) -> None:
    refs = {"sim": SimSettings(), "display": DisplaySettings()}
    attrs = {attr: fmt(getattr(getattr(p, g), name)) for g, name, attr in APP_SETTINGS
             if getattr(getattr(p, g), name) != getattr(refs[g], name)}
    if attrs:
        ET.SubElement(root, APP_ELEMENT, attrs)


def _load_source(el) -> PointSource:
    s = PointSource(0, 0, name="")
    _read_attrs(s, el)
    arr = el.find("Array")
    if arr is not None:
        s.composite = True
        for name, attr in ARRAY_FIELDS.items():
            if attr in arr.attrib:
                setattr(s, name, parse(arr.get(attr), getattr(s, name)))
    s.normalize()
    return s


def _load_part(el, cls, version: int):
    part = _defaults(cls)
    _read_attrs(part, el)
    if isinstance(part, pm.FiberPart):
        part.nodes = []
        for n in el.findall("Node"):
            node = pm.FiberNode()
            _read_attrs(node, n)
            part.nodes.append(node)
    elif isinstance(part, pm.OutlinePart):
        part.points = [(int(q.get("X", 0)), int(q.get("Y", 0))) for q in el.findall("Point")]
    elif isinstance(part, pm.InterferenceFilterPart):
        part.layers = []
        for q in el.findall("Layer"):
            layer = pm.FilterLayer()
            _read_attrs(layer, q)
            part.layers.append(layer)
    if isinstance(part, pm.ZonePlatePart) and version < 4:
        if "Thickness" not in el.attrib:
            part.thickness = 64
        if "Coating" not in el.attrib:
            part.coating = 32
    return part


# ====================================================================== save
def save(p: Project, path: str | Path, embed_image: bool = True) -> None:
    path = Path(path)
    path.write_bytes(dumps(p, path.parent, embed_image))
    p.path = str(path)


def dumps(p: Project, folder: Path | None = None, embed_image: bool = True) -> bytes:
    """The project file's bytes (``folder``: where relative image paths start)."""
    root = ET.Element("Project", {"Version": str(CURRENT_VERSION), "Seed": str(p.seed)})
    if p.description:
        ET.SubElement(root, "Description").text = p.description
    _save_scene(p, root, folder or Path.cwd(), embed_image)
    _save_simulation(p, root)
    _save_display(p, root)
    _save_app(p, root)
    for obj in [*p.sources, *p.parts, *p.probes]:
        root.append(_item_element(obj))
    for cs in p.color_schemes:
        root.append(cs.to_element())
    ET.indent(root, space="\t")
    return b'<?xml version="1.0" encoding="utf-8"?>\n' + ET.tostring(root, encoding="utf-8")


def _item_element(obj) -> ET.Element:
    if isinstance(obj, ColorScheme):
        return obj.to_element()
    if isinstance(obj, PointSource):
        el = ET.Element("Source")
        _write_attrs(obj, el, PointSource(0, 0, name=""))
        _reorder(el, ["Name", "Enabled", "X", "Y"])
        if obj.composite:
            arr = ET.SubElement(el, "Array")
            ref = PointSource(0, 0)
            for name, attr in ARRAY_FIELDS.items():
                v = getattr(obj, name)
                if v != getattr(ref, name):
                    arr.set(attr, fmt(v))
        return el
    if isinstance(obj, Probe):
        el = ET.Element("Probe")
        _write_attrs(obj, el, Probe())
        _reorder(el, ["Name"])
        return el
    el = ET.Element(ELEMENT_OF[type(obj)])
    _write_attrs(obj, el, _defaults(type(obj)))
    _reorder(el, ["Name", "Enabled", "X", "Y", "Angle"])
    if isinstance(obj, pm.FiberPart):
        for n in obj.nodes:
            _write_attrs(n, ET.SubElement(el, "Node"), pm.FiberNode())
    elif isinstance(obj, pm.OutlinePart):
        for x, y in obj.points:
            ET.SubElement(el, "Point", {"X": str(x), "Y": str(y)})
    elif isinstance(obj, pm.InterferenceFilterPart):
        for layer in obj.layers:
            _write_attrs(layer, ET.SubElement(el, "Layer"), pm.FilterLayer())
    return el


def _reorder(el: ET.Element, first: list[str]) -> None:
    items = list(el.attrib.items())
    el.attrib.clear()
    for k in first:
        for kk, v in items:
            if kk == k:
                el.set(kk, v)
    for kk, v in items:
        if kk not in first:
            el.set(kk, v)


def _save_scene(p: Project, root, folder: Path, embed: bool) -> None:
    s, ref = p.scene, SceneSettings()
    el = ET.Element("Scene")
    if s.use_image:
        el.set("Mode", "Image")
        if s.image_name:
            el.set("Image", s.image_name)
        if s.image_path:
            ip = Path(s.image_path)
            try:
                ip = ip.resolve().relative_to(folder.resolve())
            except ValueError:
                pass
            el.set("ImagePath", str(ip))
    for attr, name in (("Width", "width"), ("Height", "height"), ("DampingEdge", "damping_edge")):
        if getattr(s, name) != getattr(ref, name):
            el.set(attr, str(getattr(s, name)))
    if not s.open_edges:
        el.set("Edges", "Reflective")
    for attr, name in (("RefractiveIndex", "refractive_index"), ("ImageSources", "image_sources"),
                       ("ImageSourceScale", "image_source_scale"), ("ObjectSources", "object_sources")):
        if getattr(s, name) != getattr(ref, name):
            el.set(attr, fmt(getattr(s, name)))
    if s.use_image and p.image_bytes and embed:
        ET.SubElement(el, "ImageData").text = base64.b64encode(p.image_bytes).decode()
    if el.attrib or len(el):
        root.append(el)


def _save_simulation(p: Project, root) -> None:
    m, ref = p.sim, SimSettings()
    el = ET.Element("Simulation")
    for attr, name in (("Steps", "steps"), ("VisualizeEvery", "visualize_every"),
                       ("FourthOrderLaplacian", "fourth_order"), ("WaveSpeed", "wave_speed"),
                       ("TimeDecay", "time_decay"), ("SourceFrequencyScale", "source_frequency_scale"),
                       ("IntensityScale", "intensity_scale")):
        if getattr(m, name) != getattr(ref, name):
            el.set(attr, fmt(getattr(m, name)))
    if el.attrib:
        root.append(el)


def _save_display(p: Project, root) -> None:
    d, ref = p.display, DisplaySettings()
    el = ET.SubElement(root, "Display")
    for attr, name in (("ExportFrom", "export_from"), ("ExportTo", "export_to"),
                       ("ProbeGraphsInExport", "probe_graphs_in_export"),
                       ("ProbeMarkersInExport", "probe_markers_in_export"),
                       ("SourceMarkersInExport", "source_markers_in_export")):
        if getattr(d, name) != getattr(ref, name):
            el.set(attr, fmt(getattr(d, name)))
    for tag, prefix in (("Field", "field"), ("Intensity", "intensity")):
        layer = ET.SubElement(el, tag, {"ColorScheme": getattr(d, f"{prefix}_scheme")})
        if getattr(d, f"{prefix}_brightness") != 30.0:
            layer.set("Brightness", fmt(float(getattr(d, f"{prefix}_brightness"))))
        if getattr(d, f"{prefix}_black") != 0.0:
            layer.set("BlackLevel", fmt(float(getattr(d, f"{prefix}_black"))))
        if not d.outline:
            layer.set("Overlay", "false")
        if d.outline_gain != 3.0:
            layer.set("OverlayGain", fmt(float(d.outline_gain)))
    if d.fps != 60 or d.quality != 85:
        v = ET.SubElement(el, "Video")
        if d.fps != 60:
            v.set("Fps", str(d.fps))
        if d.quality != 85:
            v.set("Quality", str(d.quality))


# ====================================================================== from a project to a run
def paint_scene(p: Project, rgb: np.ndarray | None = None) -> np.ndarray | None:
    """The scene image with every enabled part painted in (None: a blank canvas without parts)."""
    from .parts import paint_parts
    s, m = p.scene, p.sim
    image = rgb if rgb is not None else p.scene_rgb()
    parts = [pt for pt in p.parts if pt.enabled]
    for pt in parts:
        if hasattr(pt, "frequency_scale"):
            pt.frequency_scale = m.source_frequency_scale
        if hasattr(pt, "project_seed"):
            pt.project_seed = p.seed
    if parts:
        image = paint_parts(image, parts, s.width, s.height, s.refractive_index)
    return image


def build_engine(p: Project, backend: str | None = None, rgb: np.ndarray | None = None,
                 painted: np.ndarray | None = None):
    """Scene, options and emitters of the project in an engine (``painted`` skips painting when given)."""
    from .engine import EngineOptions, Emitter, SceneData, create_engine
    s, m = p.scene, p.sim
    image = painted if painted is not None else paint_scene(p, rgb)
    if image is not None:
        scene = SceneData.from_rgb(image, s.image_sources, s.damping_edge, s.open_edges)
    else:
        scene = SceneData.empty(s.width, s.height, s.damping_edge, s.refractive_index, s.open_edges)
    opts = EngineOptions(m.source_frequency_scale, s.image_source_scale, True, m.fourth_order, m.time_decay,
                         m.intensity_scale)
    sources = [src for src in p.sources if s.object_sources]
    if not m.split_wavelengths:
        emitters = [e for src in sources for e in Emitter.from_source(src, p.seed)]
        return create_engine(scene, opts, emitters, backend or m.backend)
    return _split_engine(p, scene, opts, sources, backend or m.backend)


def _split_engine(p: Project, scene, opts, sources, backend):
    """One engine per wavelength (a source's frequency level; image sources by their green level)."""
    from .engine import Emitter, create_engine
    from .engine.split import SplitEngine
    emitters, group = [], {}
    for src in sources:
        for e in Emitter.from_source(src, p.seed):
            emitters.append(e)
            group[id(e)] = float(src.level)
    levels = sorted({lv for lv in group.values() if lv > 0} | {float(lv) for lv in np.unique(scene.source_level)})
    if not levels:
        return create_engine(scene, opts, emitters, backend)
    engines = []
    for lv in levels:
        # every engine holds every source, so a source cell treats passing waves alike in all of them (the sum is
        # exactly the full run); only this wavelength's sources drive
        eng = create_engine(scene, opts, emitters, backend)
        d = eng.driver
        gain = np.concatenate([scene.source_level == lv,
                               [group[id(e)] == lv for e in d.emitters]]).astype(np.float32)
        d.gain = gain
        engines.append(eng)
    return SplitEngine(engines, levels)
