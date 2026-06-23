#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

sys.dont_write_bytecode = True

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import advDF.ensemble_test.options  # noqa: F401
import advDF.ensemble_test.TVLoss  # noqa: F401
import pytorch_colors  # noqa: F401

print('smoke imports ok')
