"""Render the 3D map: graph export into the page template.

    python3 scripts/export_graph.py <repo> <subsystem> > /tmp/graph.json
    python3 scripts/build_map.py /tmp/graph.json viz/canon_map.html

The payload rides in a <script type="application/json"> block, so the only
sequences that can break out of it are markup characters; escaping them as
\\uXXXX keeps the JSON valid because they only ever occur inside strings.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "viz" / "canon_map_template.html"

data = Path(sys.argv[1]).read_text()
out = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "viz" / "canon_map.html"

safe = (data.replace("<", "\\u003c").replace(">", "\\u003e")
            .replace("&", "\\u0026")
            .replace("\u2028", "\\u2028").replace("\u2029", "\\u2029"))
out.write_text(TEMPLATE.read_text().replace("__GRAPH_DATA__", safe))
print(f"wrote {out} ({out.stat().st_size // 1024} KB)")
