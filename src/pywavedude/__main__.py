# SPDX-License-Identifier: GPL-3.0-or-later
"""``pywavedude [project.wdproj | scene.png] [--backend auto|cpu|mlx] [--defaults FILE]``"""

import argparse

from .engine import BACKENDS


def main() -> None:
    ap = argparse.ArgumentParser(prog="pywavedude", description="2D wave optics simulator (WaveDude Next port)")
    ap.add_argument("file", nargs="?", help="a WaveDude Next project (.wdproj), or a scene image: red = index×100 "
                                            "(0 mirror), green = source, blue = absorption")
    ap.add_argument("--backend", choices=BACKENDS, default=None, help="compute backend (default: the defaults file's)")
    ap.add_argument("--defaults", metavar="FILE", help="another defaults file (YAML) instead of the bundled one")
    args = ap.parse_args()
    from . import config
    try:
        config.load(args.defaults)
    except config.ConfigError as exc:
        raise SystemExit(f"pywavedude: defaults file: {exc}")
    from .app import run_app
    run_app(args.file, args.backend or config.app("backend", "auto"))


if __name__ == "__main__":
    main()
