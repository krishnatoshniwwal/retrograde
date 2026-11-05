"""
examples/generate_samples.py — Write the built-in demo shapes to PNG files.

The shapes themselves live in ``src/samples.py`` (the Streamlit demo uses
them directly).  Run from the project root:

    python examples/generate_samples.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.samples import DEMO_SHAPES, circle_image  # noqa: E402

OUTPUT_DIR = Path(__file__).parent


def main() -> None:
    shapes = dict(DEMO_SHAPES)
    shapes["Circle"] = circle_image
    for name, make in shapes.items():
        out = OUTPUT_DIR / f"{name.lower().replace(' ', '_')}.png"
        make().save(out)
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
