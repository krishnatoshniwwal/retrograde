# RETROGRADE — v2 Plan

*Image/Live-Feed → Functions → Interactive Math*

This supersedes the original brief. Current state (already built, per screenshots):
static reconstruction (Fourier + spline), epicycle tab, equations tab, and an export
tab with PNG / kinetics GIF / symbolic PDF+TeX / Desmos (HTML graph + JSON state).

This plan adds the features needed to take it from "solid math project" to
"hackathon-winning demo" — things that are genuinely novel, live/interactive on
stage, and hard for a judge to dismiss as "just another image filter."

---

## 1. Guiding principle for this revision

A demo wins when a judge can point a camera at *themselves*, and thirty seconds
later see their own face rendered as spinning circles and live equations, with
the tool explaining what just happened in plain English. That loop — **capture →
math → explanation** — is the spine of this plan. Everything else supports it.

---

## 2. New headline features

### 2.1 Live webcam capture → real-time equations
- Add a `[ USE WEBCAM ]` input source alongside `[ Upload ]` / `[ Use demo data ]`.
- Browser captures a frame (or a short burst of frames) via `getUserMedia`, sends
  it into the same pipeline already used for uploaded images — no separate code path.
- Show a live "processing" state (contour count ticking up, terms being fit) so the
  wait itself is part of the show, not dead air.
- **Stretch — live tracking mode:** instead of one frame, keep sampling the webcam
  at low FPS and re-fit the Fourier series continuously, so the epicycle drawing
  visibly follows the person as they move — this is the single most demo-able
  feature on this whole list.

### 2.2 "Explain the math" button
- A button (per tab, or one global one) that opens a side panel or modal walking
  through *what's currently on screen*, not a generic textbook page:
  - On the Reconstruction tab: explains contour extraction and what Canny edges
    picked up on *this specific image*.
  - On the Epicycles tab: explains that each circle is one Fourier term, points at
    the largest/smallest circle currently rendered, and explains frequency vs.
    amplitude using the actual numbers from the current fit.
  - On the Equations tab: click any single equation and get a plain-English
    breakdown of that specific term (this is a DFT coefficient, its magnitude
    controls circle size, its angle controls starting rotation, etc.)
- Two implementation tiers:
  - **Static tier (fast to build):** pre-written explanation templates with
    placeholders filled from the current run's actual numbers (contour count,
    term count, a sampled coefficient) — no external API needed, zero latency,
    zero cost, and still feels "aware" of the current result.
  - **LLM tier (more impressive, needs an API key + network):** send the current
    run's parameters to an LLM with a tight prompt ("explain this specific Fourier
    term to a beginner in 2 sentences") and stream the response into the panel.
    Falls back to the static tier if the call fails or is disabled — never blocks
    the demo on network availability.

### 2.3 Progressive reconstruction scrubber
- A slider (0 → max terms/knots) that live-redraws the static reconstruction as
  you drag it, so you can visibly watch the picture "resolve" from a blob into a
  recognizable face as terms increase.
- Directly visualizes the accuracy/complexity tradeoff from the original Tier 4
  stretch goal, but as an interactive toy instead of a static convergence plot —
  much better for a live demo.
- Pairs naturally with a small live-updating "reconstruction error" readout
  (e.g. RMS distance to original contour) next to the slider.

### 2.4 Sonification — turn the image into sound
- Each Fourier term is a (frequency, amplitude, phase) triple — which is also
  exactly the description of a musical tone. Map the N largest terms to
  oscillators (Web Audio API) and let the reconstruction play as an ambient
  drone or arpeggio while the epicycle animation runs.
- Very cheap to build (a few dozen lines with the Web Audio API) and extremely
  effective in a live room — turns "watch the picture" into "watch and listen,"
  which almost nobody else at a hackathon will have thought of.
- Optional: let the user scrub the term-count slider (2.3) and hear the sound
  get richer/noisier as more terms are added, tying the two features together.

### 2.5 Two-image morph via coefficient interpolation
- Run the pipeline on two images, align their term counts, and linearly
  interpolate the Fourier coefficients between them over time.
- Rendered as an animation, this produces one contour smoothly morphing into a
  completely different one (e.g. your face morphing into a friend's, or into a
  star) — mathematically legitimate (it's literally interpolating a function
  space) and visually striking.

### 2.6 "Battle mode" — minimum viable equation challenge
- A small game layer: given a target image, how few Fourier terms / spline knots
  can you use and still get reconstruction error under some threshold?
- Local leaderboard (in-memory or simple JSON file) ranking attempts by
  `(term_count, error)`. Turns the accuracy/complexity tradeoff into something
  judges can literally compete at during the demo, which is a great way to get
  people to engage with your booth instead of just watching.

### 2.7 Shareable results
- Every run gets a short slug/URL; visiting it re-loads the original image,
  parameters, and all export artifacts (PNG, GIF, PDF, Desmos HTML/JSON) without
  recomputation — just serve the cached outputs.
- Makes the project usable *after* the demo ends: judges/friends can revisit a
  link instead of the moment being gone once you close the laptop.

### 2.8 Print-ready / physical output
- One extra export: a clean SVG/PDF poster layout suitable for a plotter or laser
  cutter — thick title text, the equations in a corner, the reconstructed curve
  as a single-stroke path. This is a cheap addition on top of existing exports
  and gives you something *physical* to hand a judge (if a plotter/laser cutter
  is available at the event) or at least a genuinely poster-quality PDF.

---

## 3. Smaller polish features worth including

- **Colormap-by-data mode:** color each contour segment by local curvature or by
  Fourier term magnitude instead of only decorative palettes — ties the
  "colorful" rendering directly back to the math, which is a nice detail if a
  judge asks "is the color meaningful or just pretty?"
- **Side-by-side compare view:** original image, Fourier reconstruction, and
  spline reconstruction shown together with their respective term counts and
  errors, so the two methods can be judged head-to-head in one screenshot.
- **One-click "random demo" carousel:** cycle through a few preset images (star,
  a face, a logo) so the demo doesn't depend on a working webcam/upload if venue
  wifi or lighting is bad.
- **Session history panel:** last N runs in this browser session, so you can
  quickly flip back to compare an earlier capture without re-uploading.
- **Accessibility pass:** captions on the "explain the math" panel, keyboard
  navigation across tabs, and readable contrast in the dark theme — small effort,
  and something judges do notice if you mention it.

---

## 4. Updated feature tiers (build order)

### Tier A — Already built
- Upload / demo data input, Canny edge + contour extraction, Fourier and spline
  fitting, static reconstruction view, epicycle view, equations view, PNG/GIF/
  symbolic-PDF/Desmos (HTML + JSON) export.

### Tier B — Core v2 additions (build these first)
1. Webcam capture as a new input source (single-frame first, live-tracking later).
2. Progressive reconstruction scrubber (2.3) — cheap, reuses existing fit code.
3. Static-tier "explain the math" panel (2.2) — no external API dependency.
4. Print-ready SVG/PDF export (2.8) — reuses existing render pipeline.

### Tier C — Differentiators (build if Tier B lands early)
5. Sonification (2.4).
6. LLM-backed explain-the-math upgrade (2.2, LLM tier).
7. Shareable result URLs (2.7).
8. Side-by-side compare view + colormap-by-data mode (Section 3).

### Tier D — Stretch (only if way ahead of schedule)
9. Live webcam tracking mode (2.1 stretch).
10. Two-image morph (2.5).
11. Battle mode / leaderboard (2.6).
12. Session history + accessibility pass (Section 3).

---

## 5. Why this set of features wins a hackathon

- **It's interactive on stage**, not just a gallery of pre-rendered outputs —
  webcam capture and the term-count scrubber mean a judge can change the input
  and watch the output react live.
- **It teaches while it performs** — the explain-the-math panel means judges
  who don't know Fourier analysis still walk away understanding what they saw,
  which directly helps in judging Q&A.
- **It appeals to more than one sense** — sonification means the project isn't
  purely visual, which stands out against the wall of "chart/dashboard" projects
  a hackathon usually has.
- **It has a competitive hook** — battle mode gives people a reason to engage
  with your table instead of just watching someone else's turn.
- **It survives bad demo conditions** — the preset carousel and shareable links
  mean a flaky webcam or wifi doesn't sink the whole presentation.

---

## 6. Technical notes / dependencies to add

| Feature | New dependency |
|---|---|
| Webcam capture | Browser `getUserMedia` (frontend only, no new Python dep) |
| Sonification | Web Audio API (frontend only) |
| LLM explain panel | Any LLM API client (e.g. Anthropic API) + a small prompt template |
| Shareable URLs | Lightweight key-value store or flat-file cache keyed by run hash |
| Print-ready export | Existing matplotlib/SVG export path, new layout template only |
| Battle mode leaderboard | Simple JSON/SQLite file, no new service needed |

None of these require replacing the existing architecture — they slot in as new
input sources, new export targets, or new UI panels around the pipeline that
already works.
