#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["PATH"] = str(Path(sys.executable).resolve().parent) + os.pathsep + os.environ.get("PATH", "")
os.environ.setdefault("LIDRA_SKIP_INIT", "true")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ["SAM3D_GS_RENDER_BACKEND"] = "gsplat"
for _name in ("SAM3D_PM_GATE", "SAM3D_SS_REANCHOR", "SAM3D_SSI_PER_AXIS"):
    os.environ.pop(_name, None)

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from crossview.pipeline import load_pipeline, run_object


def main() -> None:
    pipeline = load_pipeline(ROOT)
    run_object(pipeline, ROOT / "data" / "object1", ROOT / "outputs" / "object1")


if __name__ == "__main__":
    main()
