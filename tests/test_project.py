# SPDX-License-Identifier: GPL-3.0-or-later
"""Project files: the original's examples survive a save attribute for attribute; new projects, file defaults,
version upgrades, this port's own settings and the embedded image round-trip."""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from pywavedude import project as pj
from pywavedude.engine import PointSource
from pywavedude.parts import model as pm
from pywavedude.probes import Probe, ProbeKind

EXAMPLES = sorted((Path(__file__).parents[1] / "reference" / "WaveDudeNext" / "Examples").glob("*.wdproj"))


def _elements(path):
    out = []
    for el in ET.parse(path).getroot():
        if el.tag in ("Display", "Description"):
            continue
        a = dict(el.attrib)
        if el.tag == "Probe":   # an absent box coordinate reads as point A's, in the original as here
            a.setdefault("BoxX", a.get("AX", "0"))
            a.setdefault("BoxY", a.get("AY", "0"))
        out.append((el.tag, a, [(c.tag, dict(c.attrib)) for c in el]))
    return out


@pytest.mark.skipif(not EXAMPLES, reason="reference examples not checked out")
def test_examples_round_trip(tmp_path):
    for path in EXAMPLES:
        p = pj.load(path)
        pj.save(p, tmp_path / "copy.wdproj")
        assert _elements(tmp_path / "copy.wdproj") == _elements(path), path.name
        q = pj.load(tmp_path / "copy.wdproj")
        assert (q.display, q.sources, q.parts, q.probes) == (p.display, p.sources, p.parts, p.probes), path.name


def test_new_project_defaults_and_versions(tmp_path):
    p = pj.Project(seed=42, description="test")
    p.sources.append(PointSource(10, 20, name="S", composite=True, size_x=3, random_phase=5))
    p.parts += [pm.FresnelLensPart(name="F", x=5, y=6), pm.SimpleGratingPart(name="G"), pm.OutlinePart(name="O")]
    p.probes.append(Probe("P", kind=ProbeKind.LINE, ax=1, ay=2, bx=3, by=4, box_x=1, box_y=2))
    p.scene.open_edges, p.sim.steps, p.sim.split_wavelengths, p.display.grid_step = False, 123, True, 40
    pj.save(p, tmp_path / "a.wdproj")
    text = (tmp_path / "a.wdproj").read_text()
    assert 'Diameter="500"' in text and 'Mirror="true"' in text      # new-part values differ from the file's
    q = pj.load(tmp_path / "a.wdproj")
    assert (q.sources, q.parts, q.probes, q.scene) == (p.sources, p.parts, p.probes, p.scene)
    assert (q.sim.steps, q.sim.split_wavelengths, q.display.grid_step) == (123, True, 40)
    pj.save(pj.Project(), tmp_path / "b.wdproj")
    assert b"PyWaveDude" not in (tmp_path / "b.wdproj").read_bytes()  # defaults leave the file as the original's
    (tmp_path / "old.wdproj").write_text('<Project Version="1" Seed="3"><ZonePlate Name="Z" /></Project>')
    old = pj.load(tmp_path / "old.wdproj")
    assert old.scene.open_edges is False and old.scene.damping_edge == 0 and old.parts[0].thickness == 64


def test_embedded_image(tmp_path):
    from PIL import Image
    img = np.zeros((20, 30, 3), np.uint8)
    img[..., 0] = 100
    Image.fromarray(img).save(tmp_path / "s.png")
    p = pj.Project()
    p.embed_image(str(tmp_path / "s.png"))
    p.scene.image_path = "no/such/file.png"                               # the project must carry the image itself
    pj.save(p, tmp_path / "i.wdproj")
    q = pj.load(tmp_path / "i.wdproj")
    assert q.scene.use_image and np.array_equal(q.scene_rgb(), img) and (q.scene.width, q.scene.height) == (30, 20)


def test_defaults_file(tmp_path):
    from pywavedude import config
    from pywavedude.parts.model import LensPart
    config.load()                                                     # the bundled file is complete and valid
    p = config.new_project()
    assert p.sources and p.scene.width == 1920
    (tmp_path / "d.yaml").write_text("new_project:\n  scene: {width: 640}\nnew_objects:\n  parts:\n"
                                     "    LensPart: {diameter: 300}\napp: {ui: {font_size: 13}}\n")
    config.load(tmp_path / "d.yaml")
    try:
        assert config.new_project().scene.width == 640 and config.app("ui.font_size") == 13
        assert config.new_object(LensPart, x=5).diameter == 300 and LensPart().diameter == 200   # format unchanged
        (tmp_path / "bad.yaml").write_text("new_project:\n  scene: {widht: 640}\n")
        with pytest.raises(config.ConfigError):
            config.load(tmp_path / "bad.yaml")
    finally:
        config.load()
