# Image-to-Function Renderer

> Convert any image into a set of **mathematical functions** whose plot reproduces the image —
> Fourier series (animated epicycles) or piecewise cubic splines — with symbolic LaTeX output.

[![CI](https://github.com/YOUR_USERNAME/retrograde/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/retrograde/actions)

---

## What This Does

This project takes a raster image (portrait, drawing, etc.) and:

1. **Detects edges** using Canny edge detection
2. **Extracts contours** — ordered sequences of (x, y) points tracing each outline
3. **Fits mathematical functions** to each contour using one of two methods:
   - **Fourier / Epicycles**: Decomposes the contour into rotating circles via the DFT.
     The sum of N rotating circles traces the original path with increasing accuracy.
   - **Piecewise Splines**: Fits cubic spline segments that interpolate the contour points,
     producing a "giant piecewise function" that traces the shape.
4. **Reconstructs and renders** the image from pure mathematics
5. **Animates** the epicycle construction in real time (GIF/MP4)
6. **Exports symbolic LaTeX** — the actual equations that draw the image

---

## Math Background

### Fourier / Epicycle Method

Each contour point (x_k, y_k) is encoded as a complex number z_k = x_k + i·y_k.
The DFT decomposes the discrete signal into N rotating phasors:

```
Z_n = Σ_{k=0}^{N-1}  z_k · e^{-2πi·n·k/N}
```

Keeping the M largest-amplitude terms gives an M-epicycle approximation:

```
z(t) ≈ Σ_{n ∈ top-M}  (|Z_n|/N) · e^{i·(n·t + arg Z_n)}
```

More terms → higher fidelity. A single rotating circle reproduces a circle exactly.

### Piecewise Spline Method

Points are parameterised by arc length t ∈ [0, 1].
Two independent cubic splines x(t) and y(t) interpolate all contour points.
Each spline piece over interval [t_k, t_{k+1}] is:

```
x(t) = a + b(t-tₖ) + c(t-tₖ)² + d(t-tₖ)³
```

The union of all pieces forms the "giant piecewise function".

---

## Pipeline

```
Image file
  ↓ load_image + resize + denoise
Preprocessed gray image
  ↓ Canny edge detection
Binary edge map
  ↓ findContours + approxPolyDP
Contour point sequences
  ↓ DFT (numpy.fft) or CubicSpline (scipy)
Fourier coeffs / Spline segments
  ↓ SymPy symbolic conversion
LaTeX expressions
  ↓ matplotlib / animation
Static PNG + Animated GIF + .tex file
```

---

## Project Structure

```
retrograde/
  ├── src/
  │   ├── preprocessing.py    # load, resize, denoise, Canny
  │   ├── contours.py         # findContours, Douglas-Peucker, complex conversion
  │   ├── fourier.py          # DFT, epicycle coefficients, reconstruction
  │   ├── piecewise.py        # CubicSpline fitting, segment extraction
  │   ├── symbolic.py         # SymPy expressions, LaTeX strings
  │   ├── render.py           # matplotlib static + animation (GIF/MP4)
  │   └── latex_export.py     # .tex document builder + pdflatex runner
  ├── tests/                  # pytest unit tests (math correctness)
  ├── app.py                  # Streamlit interactive UI
  ├── config/default.yaml     # pipeline parameters
  ├── examples/               # sample images
  ├── .github/workflows/ci.yml
  ├── requirements.txt
  └── README.md
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** Requires Python ≥ 3.10. OpenCV may need system libraries on Linux:
> `sudo apt-get install libgl1-mesa-glx`

### 2. Run the interactive UI

```bash
streamlit run app.py
```

Upload any image (or use the built-in demo), adjust parameters, and see the reconstruction live.

### 3. Use as a library

```python
from src.preprocessing import preprocess
from src.contours import get_all_contours
from src.fourier import dft_pipeline
from src.render import plot_reconstruction, fig_to_png_bytes
import matplotlib.pyplot as plt

# Process an image
edges, original = preprocess("examples/face.png")
contours = get_all_contours(edges)

# Fit Fourier representation
results = [dft_pipeline(c, config={"fourier": {"n_terms": 100}}) for c in contours]
paths = [r["path"] for r in results]

# Render
fig = plot_reconstruction(paths, config={"render": {"theme": "dark", "colormap": "plasma"}})
fig.savefig("output/reconstruction.png", dpi=150, bbox_inches="tight")
plt.close(fig)
```

### 4. Run tests

```bash
pytest tests/ -v
```

### 5. Export to Desmos

After processing an image in the Streamlit UI, go to the **Downloads** tab and find the `[ DSM ]` card:

- **`[ DL_DESMOS_JSON ]`** — download a JSON expression list and import it into [desmos.com/calculator](https://desmos.com/calculator) via the expression panel's import dialog.
- **`[ DL_DESMOS_HTML ]`** — download a self-contained HTML file. Open it in any browser to get a fully interactive, zoomable Desmos graph with all equations loaded and a per-contour color legend — no login needed.

---

## Configuration

All parameters are controlled via `config/default.yaml`:

```yaml
image:
  max_dim: 512          # resize longest side to this (pixels)
  denoise_kernel: 5     # Gaussian blur kernel (odd integer)
edge:
  canny_low: 50         # Canny lower threshold
  canny_high: 150       # Canny upper threshold
contour:
  min_length: 20        # discard short contours
  epsilon_fraction: 0.002  # Douglas-Peucker tolerance
fourier:
  n_terms: 100          # epicycles to keep
spline:
  n_knots: 50           # knots per contour
render:
  theme: dark           # "dark" or "light"
  colormap: plasma      # matplotlib colormap
```

---

## Feature Tiers

| Tier | Feature | Status |
|------|---------|--------|
| 0 | Edge detection, piecewise splines, static plot | ✅ |
| 1 | Fourier / epicycles, animation, SymPy LaTeX | ✅ |
| 2 | Gradient rendering, themes, unit tests, YAML config, CI | ✅ |
| 3 | Streamlit UI, fidelity slider, live preview, downloads | ✅ |
| 3.5 | Desmos export, neon rendering | ✅ |
| 4 | Error curves, web deployment, video input | 🔜 |

---

## Tech Stack

| Purpose | Library |
|---|---|
| Image I/O & CV | opencv-python, Pillow |
| Numerics | numpy, scipy |
| Symbolic math | sympy |
| Plotting / animation | matplotlib |
| LaTeX generation | manual .tex + pdflatex |
| Testing | pytest |
| UI | streamlit |
| CI | GitHub Actions |

---

## License

MIT
