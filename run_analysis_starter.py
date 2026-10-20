#!/usr/bin/env python3
"""Run the bundled starter notebook's Python cells and save figures without Jupyter."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile


def main() -> None:
    root = Path(__file__).resolve().parent
    os.chdir(root)
    notebook = json.loads((root / "notebooks/predictions_starter.ipynb").read_text())
    with tempfile.TemporaryDirectory(prefix="energy-matplotlib-") as cache:
        os.environ.setdefault("MPLCONFIGDIR", cache)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        namespace = {"__name__": "__main__"}
        for index, cell in enumerate(notebook["cells"], start=1):
            if cell["cell_type"] == "code":
                print(f"Running starter notebook cell {index}", flush=True)
                exec(compile("".join(cell["source"]), f"predictions_starter.ipynb:cell{index}", "exec"), namespace)
        plt.close("all")
    print("Starter figures saved in multihorizon_outputs/; no model was retrained.")


if __name__ == "__main__":
    main()
