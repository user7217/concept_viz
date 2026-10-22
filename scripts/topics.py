"""Print the topic links between a repo's components, with where each is stated.

    python3 scripts/topics.py <repo>

No model and no network: every link comes from code, a parameter, a launch
argument, or a library's documented default name.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atlas.ingest import ingest_repo
from atlas.topics import endpoints, report

if len(sys.argv) < 2:
    print(__doc__)
    raise SystemExit(1)

repo = Path(sys.argv[1]).resolve()
print(report(endpoints(ingest_repo(repo), repo)))
