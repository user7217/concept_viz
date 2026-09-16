"""Lift verified snippets out of a third-party library, then forget the clone.

    git clone --depth 1 <url> /tmp/robot_localization
    python3 scripts/extract_library.py /tmp/robot_localization
    rm -rf /tmp/robot_localization

The spans below were located by an agent reading the source and then checked
line by line against the files before being written here; every range was
opened and compared. Storing the excerpt rather than the tree is deliberate:
a vendored copy makes this repo a redistributor of someone else's code and
goes stale in silence, while a path plus a pinned commit can be re-verified
by anyone with `git clone`.
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "library_code.json"

SOURCE = {
    "name": "robot_localization",
    "url": "https://github.com/cra-ros-pkg/robot_localization",
    "license": "BSD-3-Clause",
    "license_note": (
        "LICENSE at the repo root is BSD 3-Clause (Charles River Analytics, "
        "2014-2016). package.xml line 12 declares Apache License 2.0, which "
        "contradicts it; the LICENSE file governs."
    ),
    "copyright": "Copyright (c) 2014, 2015, 2016, Charles River Analytics, Inc.",
}

# node id -> (path, first line, last line, enclosing function, what it does)
SPANS = [
    ("jacobian_matrix", "src/ekf.cpp", 371, 386, "Ekf::predict",
     "Copies the transfer function into the Jacobian, then overwrites the 13 "
     "entries that are genuinely nonlinear -- the full df/dx at the current state."),
    ("linearization", "src/ekf.cpp", 302, 323, "Ekf::predict",
     "Evaluates analytically derived partials using the current roll, pitch and "
     "yaw: this is the linearisation of f about the present mean estimate."),
    ("state_space_model", "include/robot_localization/filter_common.hpp", 41, 57,
     "enum StateMembers",
     "The 15-dimensional state itself: position, orientation, their velocities, "
     "and linear acceleration."),
    ("state_space_model", "src/ekf.cpp", 259, 281, "Ekf::predict",
     "Fills F: the rotation matrix times dt that carries body-frame velocity "
     "into world-frame position, with the half-dt-squared acceleration terms."),
    ("marginalization", "src/ekf.cpp", 420, 436, "Ekf::predict",
     "Projects the mean forward and propagates P = J P J' + Q dt -- the prior "
     "obtained by marginalising the joint over the previous state."),
    ("gaussian_conditioning", "src/ekf.cpp", 170, 188, "Ekf::correct",
     "The Kalman gain K = PH'(HPH' + R)^-1 and the innovation z - Hx: the "
     "Gaussian prior conditioned on the measurement."),
    ("covariance_matrix", "src/ekf.cpp", 195, 207, "Ekf::correct",
     "Applies the gain and updates the error covariance in Joseph form, "
     "(I - KH)P(I - KH)' + KRK', which stays symmetric by construction."),
    ("positive_semidefinite_matrix", "src/ekf.cpp", 128, 152, "Ekf::correct",
     "Conditions R before it reaches the gain: negative diagonal entries are "
     "replaced by their absolute value and near-zero variances floored at 1e-9, "
     "so the innovation covariance stays invertible."),
]

# Deliberately absent: matrix_inversion_lemma. The filter calls .inverse()
# directly and the UKF takes an LLT factorisation; neither is the Woodbury
# identity, and grep finds no Woodbury or Sherman-Morrison anywhere in the
# tree. Pointing the node at a plain inverse would be the kind of
# nearly-right evidence this project keeps having to delete.


def main() -> None:
    root = Path(sys.argv[1])
    if not root.exists():
        raise SystemExit(f"no clone at {root}")
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True).stdout.strip()
    tag = subprocess.run(
        ["git", "-C", str(root), "describe", "--tags"],
        capture_output=True, text=True, check=False).stdout.strip()

    out: dict[str, list[dict]] = {}
    for node_id, rel, start, end, function, note in SPANS:
        path = root / rel
        lines = path.read_text(errors="replace").splitlines()
        if not 1 <= start <= end <= len(lines):
            raise SystemExit(f"{rel}:{start}-{end} is outside the file")
        out.setdefault(node_id, []).append({
            "path": rel, "language": "cpp", "start": start, "hit": start,
            "term": function, "strength": "library", "note": note,
            "lines": lines[start - 1:end],
        })

    OUT.write_text(json.dumps({
        "_note": "Verified excerpts from a third-party library. Regenerate "
                 "with scripts/extract_library.py after re-cloning.",
        "source": {**SOURCE, "commit": commit, "tag": tag},
        "spans": out,
    }, indent=2) + "\n")
    print(f"wrote {OUT.name}: {sum(len(v) for v in out.values())} spans "
          f"across {len(out)} nodes, from {tag or commit[:7]}")


if __name__ == "__main__":
    main()
