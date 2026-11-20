# SPDX-License-Identifier: GPL-3.0-or-later
"""Test files stay inside the project (AGENTS.md): ``tmp_path`` is ``tests/_tmp/<test>``, git-ignored and
overwritten on the next run instead of being deleted."""

import re
from pathlib import Path

import pytest


@pytest.fixture
def tmp_path(request) -> Path:
    d = Path(__file__).parent / "_tmp" / re.sub(r"[^\w.-]", "_", request.node.name)
    d.mkdir(parents=True, exist_ok=True)
    return d
