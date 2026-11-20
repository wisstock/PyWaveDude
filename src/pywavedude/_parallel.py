# SPDX-License-Identifier: GPL-3.0-or-later
"""One lock around every Numba ``parallel=True`` call.

Numba's default (workqueue) threading layer aborts the process when two Python threads launch parallel regions at
once, which the app does (simulation, export and preview colouring run on separate threads). Each region already
uses every core, so taking turns costs little; the lock keeps the layer portable (no TBB or OpenMP needed).
"""

import threading

PARALLEL_LOCK = threading.RLock()
