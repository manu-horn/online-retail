"""Exercise the notebook's import sequence with an older module cached in a kernel."""

import importlib
import json
import sys
from pathlib import Path


def test_rfm_notebook_imports_refresh_stale_modules():
    rfm = importlib.import_module("online_retail.features.rfm")
    notebook = Path(__file__).resolve().parents[1] / "notebooks/02_stats_and_fe.ipynb"
    cells = json.loads(notebook.read_text())["cells"]
    source = next("".join(c["source"]) for c in cells if c.get("id") == "rfm-prepare")
    saved_analysis = sys.modules.pop("online_retail.analysis.rfm", None)
    del rfm.component_values
    try:
        scope = {}
        exec(source.split("with (ROOT /")[0], scope)
        assert callable(scope["cluster_diagnostics"])
        assert callable(scope["RFMSegmenter"])
        assert callable(rfm.component_values)
    finally:
        importlib.reload(rfm)
        if saved_analysis is not None:
            sys.modules["online_retail.analysis.rfm"] = saved_analysis
