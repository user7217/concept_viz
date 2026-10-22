# Third-party notices

Derivation Atlas has no runtime dependencies beyond the Python standard
library. It does use other people's work in five ways, listed here with what
each licence asks and how this repository meets it.

This repository's own licences: the code is MIT (`LICENSE`); the content in
`data/` is CC BY-SA 4.0 (`data/LICENSE`), as Wikipedia's ShareAlike term
requires of the restated sheets. The robot_localization excerpts in
`data/library_code.json` stay BSD 3-Clause.

## Graphify

- **Project:** https://github.com/Graphify-Labs/graphify (https://graphify.com)
- **Licence:** Apache-2.0. Portions were contributed under MIT before
  relicensing; Graphify ships both texts (`LICENSE`, `LICENSE-MIT`) and a
  `NOTICE`. Copyright 2026 Safi Shamsi and the Graphify contributors.
- **Version used:** 0.9.90 (PyPI package `graphifyy`)
- **How it is used:** installed separately, then run as a command-line tool
  on the repo being studied (`atlas/graphify_build.py` invokes `graphify
  extract`, `label` and `export` as subprocesses). Atlas reads the
  `graphify-out/graph.json` it leaves behind (`atlas/codegraph.py`).
- **What that means:** using a tool is not redistributing it. No Graphify
  code, prompt or skill text is copied, vendored, modified, imported or
  redistributed here. A test (`tests/test_codegraph.py`) fails if
  anything in `atlas/` or `scripts/` imports it. Apache-2.0's conditions
  attach to redistributing Graphify or works derived from it, and this repo
  does neither, so none apply; it is credited here because the workflow
  depends on it. Install it yourself, under its own licence. Derivation Atlas
  is not affiliated with or endorsed by the Graphify project.

## robot_localization

- **Project:** https://github.com/cra-ros-pkg/robot_localization
- **Version:** 3.10.1, commit `7dfb6aa97b2082185d2fac3420888ae8474bfc1a`
- **Licence:** BSD-3-Clause, per the repository's `LICENSE` file. Its
  `package.xml` declares Apache License 2.0; the two disagree, and the source
  files themselves carry the BSD header reproduced below.
- **How it is used:** `data/library_code.json` holds 8 verbatim excerpts
  (152 lines) from `src/ekf.cpp` and
  `include/robot_localization/filter_common.hpp`, shown on the published map
  beside the mathematics they implement.
- **What that means:** an excerpt is a redistribution of those lines, and the
  licence asks that redistributed source keep the copyright notice, the
  conditions and the disclaimer. Each file's own notice is captured verbatim
  by `scripts/extract_library.py`, stored alongside the excerpts, and shown
  on the map under every excerpt from that file. `tests/test_third_party.py`
  fails if an excerpt arrives without its notice. The project's and its
  contributors' names are used only to attribute the code, never to endorse
  anything here.
- `data/library_topics.json` records the default topic names two of its nodes
  open, with the file and line each was read from. Those are facts about the
  software, not its code.

### Notice: `src/ekf.cpp`

```
Copyright (c) 2014, 2015, 2016 Charles River Analytics, Inc.
Copyright (c) 2017, Locus Robotics, Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above
copyright notice, this list of conditions and the following
disclaimer in the documentation and/or other materials provided
with the distribution.
3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived
from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
```

### Notice: `include/robot_localization/filter_common.hpp`

```
Copyright (c) 2014, 2015, 2016  Charles River Analytics, Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above
copyright notice, this list of conditions and the following
disclaimer in the documentation and/or other materials provided
with the distribution.
3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived
from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
```

## Wikipedia

- **Licence:** text is CC BY-SA 4.0
  (https://creativecommons.org/licenses/by-sa/4.0/).
- **How it is used:** the sheets in `data/derivations/` restate mathematics
  retrieved from Wikipedia articles: paraphrased, condensed and re-ordered by
  a model, then checked against the source. Each sheet records the articles
  it was restated from in `source_ids`.
- **What that means:** attribution. The published map names and links every
  article a statement was restated from, states the licence, and says the
  text was paraphrased. CC BY-SA also asks that adaptations be shared under
  the same or a compatible licence, which is why `data/` is CC BY-SA 4.0
  (`data/LICENSE`).

## Pinned papers

- **Work:** Gyubeom Im, "Notes on Kalman Filter (KF, EKF, ESKF, IEKF,
  IESKF)", 2024, arXiv:2406.06427 (https://arxiv.org/abs/2406.06427).
- **Licence:** CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/),
  read from the paper's own arXiv page on 2026-10-22 and recorded in
  `data/source_documents.json`.
- **How it is used:** the `extended_kalman_filter` sheet restates section 4,
  and `linear_kalman_filter` restates sections 8.1 and 8.2: paraphrased,
  condensed and, for the linear Kalman filter, renamed to this project's
  notation (the source writes R for process noise and Q for measurement
  noise; the sheet says so).
- **What that means:** attribution, a link to the licence, and an indication
  of changes. The published map credits the paper by title, author, year and
  link beside each sheet that restates it, names the licence, and says the
  text was paraphrased; the copied study prompt names it too. CC BY places
  no condition on the licence of the adaptation.

Only openly licensed documents are pinned; each entry in
`data/source_documents.json` records its licence and the date it was read.

## Other retrieved sources

Further-reading links (`data/reading.json`) are links only; nothing from the
linked pages is reproduced. Each link carries its title, and where known its
authors' surnames and year, as bibliographic facts. These come from the
Crossref REST API, the arXiv API and Open Library, all of which publish this
metadata openly (Crossref and Open Library under CC0, arXiv's metadata under
CC0 1.0), or from the linked page's own `<title>`.
