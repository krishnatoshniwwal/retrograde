# Image-to-Function Renderer

A project brief covering concept, math, architecture, feature tiers, and execution scope —
written to be understood by a human reviewer or an LLM picking up the project cold.

---

## 1. What This Project Is

This project converts a raster image (e.g. a portrait photo) into a set of mathematical
functions whose plot reproduces the image. Concretely: the program detects edges/contours
in the image, treats each contour as a curve in the plane, and fits that curve with a
mathematical representation — either a truncated **Fourier series** (curve as a sum of
rotating circles / complex exponentials) or a **piecewise function** (splines / polynomial
segments stitched together). The result can be:

- (a) re-plotted with matplotlib to visually reconstruct the original image purely from the functions,
- (b) animated as rotating "epicycles" tracing the drawing in real time, and
- (c) exported as a LaTeX document showing the actual symbolic equations that produce the image, compiled to PDF.

The core idea is inspired by a smaller precedent project (a Grade 11 math summative:
*"Python program to convert images to massive piecewise functions that show the image when
plotted"*). This version is intended to go further in scope, math sophistication, visual
polish, and software engineering practice (tests, CI, modular architecture, possibly a live UI).

## 2. Why This Project (Purpose)

- Built as a portfolio piece to demonstrate applied math + programming ability, specifically
  for admission into a college tech club.
- Meant to show sustained, incremental engineering work via a real GitHub commit history —
  the commit history is itself a submission artifact, not just the code.
- Chosen because it sits at the intersection of several skill areas a tech club would value:
  numerical methods, computer vision basics, symbolic math, data visualization, and
  (optionally) front-end/UI work — all in one coherent project.

## 3. Core Math and CS Concepts Involved

### 3.1 Image processing
- Grayscale conversion and resizing for consistent processing.
- Edge detection (e.g. Canny) to isolate outlines/features in the image.
- Contour extraction (e.g. OpenCV `findContours`) to get ordered sequences of (x, y) points
  describing each outline.
- Contour simplification (e.g. Douglas-Peucker / `approxPolyDP`) to reduce point count while
  preserving shape.

### 3.2 Representing a contour as a function
Two independent mathematical strategies can be used to turn a discrete point sequence into a
continuous function (the project can support one or both):

- **Fourier / epicycle approach:** treat each 2D point (x, y) as a complex number z = x + iy.
  The ordered contour becomes a discrete complex-valued signal. Taking the Discrete Fourier
  Transform (DFT) of that signal gives a set of complex coefficients, each corresponding to a
  circle rotating at a fixed frequency. Summing N rotating circles (epicycles) reconstructs
  the original path; more circles = higher fidelity. This is literally "the image as a sum of
  functions" — specifically a sum of complex exponentials, i.e. a Fourier series.
- **Piecewise function approach:** fit the contour points with piecewise polynomial or spline
  segments (e.g. cubic splines via `scipy.interpolate`), producing a large number of small
  polynomial "pieces" whose union traces the contour — closer to the original precedent
  project's approach, and more directly "giant piecewise function" in flavor.

### 3.3 Symbolic math and LaTeX generation
Numeric coefficients (Fourier or spline) can be converted into actual symbolic expressions
using `sympy`, then auto-typeset into a LaTeX document (one equation per contour segment or
per Fourier term) and compiled to a PDF. This is what produces the "giant LaTeX PDF of the
functions" output.

### 3.4 Visualization
- Static reconstruction plots (matplotlib) showing the image redrawn purely from the fitted functions.
- Animated epicycle rendering (matplotlib.animation) showing the rotating-circles construction
  in motion, exportable as GIF/MP4.
- Aesthetic controls: color gradients along the path, line-weight variation, dark/light themes,
  multiple images composed into a gallery.

## 4. End-to-End Pipeline

| Stage | Input → Output | Key tools |
|---|---|---|
| 1. Load & preprocess | Raw image → grayscale, resized, denoised image | OpenCV / Pillow |
| 2. Edge detection | Preprocessed image → binary edge map | OpenCV (Canny) |
| 3. Contour extraction | Edge map → ordered point sequences | OpenCV (findContours) |
| 4. Simplification | Raw contours → reduced-point contours | OpenCV (approxPolyDP) |
| 5. Function fitting | Point sequences → Fourier coeffs or spline pieces | NumPy / SciPy |
| 6. Symbolic conversion | Numeric fit → symbolic expressions | SymPy |
| 7. Rendering | Functions → static plot / animation | Matplotlib |
| 8. Document export | Symbolic expressions → typeset PDF | LaTeX / PyLaTeX |

## 5. Feature Tiers — How Far This Can Go

The project does not need to hit every tier below. These are presented as a menu, roughly
ordered by increasing effort/sophistication, so scope can be adjusted to whatever timeline is
available without changing the core idea.

### Tier 0 — Baseline (matches the original precedent project)
- Single image in, edge detection, piecewise function fit, matplotlib plot reconstruction.
- Basic LaTeX/PDF export of the piecewise functions.

### Tier 1 — Core upgrade
- Add Fourier/epicycle representation as a second, mathematically distinct method.
- Animated epicycle reconstruction (GIF/MP4 export).
- Symbolic (sympy) expressions instead of raw numeric coefficients in the LaTeX output.
- Clean modular codebase (separate modules for preprocessing, fitting, rendering, export).

### Tier 2 — Polish & robustness
- Aesthetic rendering: gradients, adjustable stroke, theming (dark/light).
- Batch processing across multiple images with a generated gallery page.
- Unit tests for the math core (DFT correctness, spline fit error bounds, etc.).
- Config-driven runs (YAML/JSON config instead of hardcoded parameters).
- GitHub Actions CI running tests on every push.

### Tier 3 — Interactive / advanced
- Streamlit or Gradio front end: upload an image, choose method (Fourier vs piecewise), watch
  the reconstruction render live, download the PDF/animation.
- Adjustable fidelity slider (number of Fourier terms / spline knots) with live preview of the
  accuracy/complexity tradeoff.
- Support for color images (per-channel or luminance-weighted contour extraction) instead of
  just outlines.
- Performance work: vectorized FFT (`numpy.fft`) instead of naive DFT, caching of intermediate results.

### Tier 3.5 — Desmos export and colorful rendering
Two complementary upgrades, both aimed at making the output visually striking and shareable
rather than a plain matplotlib plot:

- **Desmos expression-list export:** convert each fitted contour/segment into a LaTeX
  expression (already produced for the PDF pipeline) and export it as a Desmos-importable
  expression list, with each contour assigned its own color field, so the whole image can be
  pasted into Desmos as a set of colored equations.
- **Embedded Desmos API graph:** generate a small self-contained HTML file that embeds the real
  Desmos `GraphingCalculator` JS API and programmatically injects every expression via
  `calculator.setExpression({id, latex, color})`. Opening the file gives an interactive,
  zoomable, colored Desmos graph of the image with every underlying equation visible in the
  sidebar — a strong, demoable deliverable in its own right.
- **Colorful matplotlib rendering:** move past default single-color plots using per-contour
  palettes (e.g. distinct colors for hair, jaw, eyes), gradient-along-the-path coloring via
  `LineCollection` (color shifts continuously along each curve, matching the look of slick
  epicycle animations), and dark-background/neon themes so static and animated output look
  like generative art rather than a plotted homework function.
- Optional: color mapping driven by data (e.g. local curvature or reconstruction speed) rather
  than purely decorative choice, so color also communicates information.

### Tier 4 — Stretch / research-flavored
- Error-quantified fitting: report reconstruction error (e.g. Hausdorff distance or pointwise
  RMS) as a function of number of terms/knots, with a plotted convergence curve.
- Compression angle: frame the Fourier truncation explicitly as lossy compression, and
  discuss/visualize the accuracy-vs-file-size tradeoff.
- Web deployment of the Streamlit app so anyone can try it without local setup.
- Video input support: apply the same pipeline frame-by-frame to a short video clip.

## 6. Suggested Repository Structure

```
image-to-function/
  ├── src/
  │   ├── preprocessing.py    # grayscale, resize, edge detection
  │   ├── contours.py         # extraction + simplification
  │   ├── fourier.py          # DFT / epicycle math
  │   ├── piecewise.py        # spline fitting
  │   ├── symbolic.py         # sympy expression generation
  │   ├── render.py           # matplotlib static + animation
  │   └── latex_export.py     # LaTeX doc build + PDF compile
  ├── tests/                  # pytest unit tests per module
  ├── app.py                  # Streamlit/Gradio UI (optional / later tier)
  ├── examples/                # sample input images + output gallery
  ├── .github/workflows/ci.yml
  ├── requirements.txt
  └── README.md
```

## 7. Suggested Tech Stack

| Purpose | Library |
|---|---|
| Image I/O & CV | opencv-python, Pillow |
| Numerics | numpy, scipy |
| Symbolic math | sympy |
| Plotting / animation | matplotlib |
| LaTeX generation | pylatex (or manual .tex templating) + a LaTeX engine (e.g. TeX Live) to compile |
| Desmos export | Desmos API (GraphingCalculator) embedded in generated HTML, or expression-list export |
| Testing | pytest |
| Optional UI | streamlit or gradio |
| CI | GitHub Actions |

## 8. Concrete Deliverables

- A GitHub repository with modular, tested, documented source code.
- A README with: problem statement, math explanation, usage instructions, example
  input/output images, and (ideally) an animated GIF demo.
- At least one generated LaTeX/PDF output showing the actual symbolic functions for a
  sample image.
- A commit history reflecting real incremental development.
- Optional: a deployed or locally-runnable interactive demo (Streamlit/Gradio).

## 9. Why This Should Read Well to a Reviewer

- Demonstrates applied mathematics (Fourier analysis, symbolic algebra, numerical
  approximation) tied directly to a visual, tangible output — not abstract or toy.
- Demonstrates software engineering practice: modular code, tests, CI, documentation,
  version control discipline.
- Is self-contained and explainable: every design decision traces back to a clear pipeline
  stage, so the author can speak fluently to any part of it in an interview.
- Scales gracefully: the tiered feature list means the project can be presented honestly at
  whatever level of completeness was actually reached by the deadline.

## 10. Quick Glossary (for readers unfamiliar with the math)

| Term | Meaning |
|---|---|
| Contour | An ordered sequence of points tracing the boundary/outline of a shape in an image. |
| DFT (Discrete Fourier Transform) | A transform that decomposes a discrete signal (here, a sequence of complex-number points) into a sum of rotating circular components (frequencies). |
| Epicycle | A circle rotating on top of another rotating circle; a chain of epicycles can trace complex curves, visually representing a Fourier series. |
| Piecewise function | A function defined by different sub-functions (pieces), each applying over a specific interval — used here to approximate a contour with many small curve segments. |
| Spline | A smooth piecewise-polynomial curve commonly used to interpolate a set of points. |
| Symbolic math | Representing math expressions as exact algebraic objects (via SymPy) rather than only numeric approximations, enabling exact printed equations. |

---

*This document is intended as a standalone brief: a human reviewer or an LLM reading only
this file should be able to understand the project's purpose, math, architecture, and scope
without needing additional context.*
