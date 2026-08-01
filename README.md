# retrograde

[![CI](https://github.com/krishnatoshniwwal/retrograde/actions/workflows/ci.yml/badge.svg)](https://github.com/krishnatoshniwwal/retrograde/actions)

**Retrograde turns any image into the mathematics that draws it.**

Upload a photo or sketch. The app traces the outlines of your image, then figures out a set of equations — real, symbolic, exportable equations — that can reproduce those outlines from scratch. You end up with the actual math behind the picture.

---

## The core idea

When you look at any drawing, you're really looking at a collection of curves. Retrograde asks: *what function would you have to plot to get this curve?*

There are two answers it can give you:

**Fourier series (the spinning circles method)**

Imagine you're drawing a shape with a pen attached to the edge of a spinning wheel, which is itself on the edge of another spinning wheel, which is on another, and so on. If you pick the right sizes and speeds for those wheels, the tip of the pen traces any shape you want. This is called an epicycle construction — the same idea that ancient astronomers used to model planetary motion.

Retrograde decomposes your image outline into exactly these spinning circles using a mathematical operation called the Discrete Fourier Transform. The result is an equation like:

```
z(t) ≈ 3.2·e^(it) + 1.7·e^(3it) + 0.9·e^(-2it) + ...
```

Each term is one spinning circle. More terms means a more faithful reproduction of your image.

**Piecewise cubic splines (the smooth curve method)**

The second method fits smooth polynomial curves through the outline points, one segment at a time. Each segment is described by four numbers, making the full curve a "giant piecewise function" — a different cubic equation for each small stretch of the outline.

Both methods spit out real equations that you can copy, graph, or print.

---

## What you can do with it

**See the reconstruction live** — adjust how many terms/curves to use and watch the image appear from nothing, term by term.

**Watch the wheels spin** — an animated GIF shows the epicycle mechanism in motion: the rotating circles building up the outline in real time.

**Read the equations** — every tab has an "Explain the math" panel that walks through what the equations on screen actually mean, using the numbers from your specific image.

**Export everything:**
- PNG of the reconstruction
- Animated GIF of the epicycles
- LaTeX `.tex` file and compiled PDF with all the equations
- A Desmos graph (JSON or self-contained HTML) you can open in any browser and interact with
- A print-ready SVG/PDF poster — clean white background, equation footer, plotter-ready

---

## Running it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open your browser. Upload an image (or use the webcam, or the built-in demo), pick Fourier or spline mode, and drag the sliders.

**Requirements:** Python ≥ 3.10.

---

## How it works under the hood

```
Your image
  → edge detection (finds outlines)
  → contour extraction (traces each outline as a list of points)
  → Fourier transform or spline fitting (turns points into equations)
  → reconstruction (plots the equations back into an image)
  → export (LaTeX, GIF, Desmos, SVG)
```

The whole pipeline is modular — each step is a separate Python module with its own tests.

---

## Tech

Python · NumPy · SciPy · OpenCV · SymPy · Matplotlib · Streamlit

---

## License

MIT
