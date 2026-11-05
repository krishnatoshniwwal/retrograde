# retrograde

[![CI](https://github.com/krishnatoshniwwal/retrograde/actions/workflows/ci.yml/badge.svg)](https://github.com/krishnatoshniwwal/retrograde/actions)

**Retrograde turns any image into the mathematics that draws it.**

Upload a photo or sketch. The app traces the outlines of your image, then figures out a set of equations — real, symbolic, exportable equations — that can reproduce those outlines from scratch. You end up with the actual math behind the picture.

Try it here!!!!
https://retrograde-etvpburewc6tufyaqbyk5r.streamlit.app/

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

**See the reconstruction live**: put the original, its edge map and the reconstruction side by side, then drag the scrubber to watch the picture resolve one Fourier term at a time, with an RMS-error meter.

**Watch the wheels spin**: the Epicycles tab animates *every* contour at once, each drawn by its own chain of rotating circles. It runs in your browser with no server rendering. You can pause, change the speed, show the circles for one contour or all of them, and move the focus between contours. You can also bake a single contour into a shareable GIF.

**Read the equations**: each contour's x(t), y(t) is typeset with KaTeX, with its LaTeX source ready to copy. Every tab has an "Explain the math" panel that walks through the equations using the numbers from your image.

**Start from a built-in shape**: a planet's retrograde loop (the app's namesake), a star, a heart, Fay's butterfly curve, a rose curve and nested squares are all drawn procedurally, so no image files are needed.

**Export everything** (files are generated only when you click):
- PNG of the reconstruction
- Animated GIF of the epicycles
- LaTeX `.tex` file and a PDF of all the equations
- A Desmos graph (JSON or self-contained HTML) you can open in any browser and interact with
- A print-ready SVG/PDF poster in cream **Paper** or **Night** style, with a custom title and a live preview

---

## Look & feel

Retrograde uses a pixel-art night-sky theme: deep navy skies, cream ink, a lime primary, cyan highlights, a pale moon and sakura pink. It has square corners, hard offset shadows and the *Jersey 10* display face. The hero is a working epicycle machine animated in pure CSS, and it traces a real retrograde loop.

The palette is defined once in [`src/theme.py`](src/theme.py) and shared by the Streamlit CSS ([`assets/style.css`](assets/style.css)), every matplotlib figure, the PDF, the poster and the Desmos export. The four custom colormaps (`aurora`, `sakura`, `dusk`, `moonlight`) each have a darker "ink" twin that is used automatically on paper output.

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

The whole pipeline is modular, and each step is a separate Python module with its own tests:

```
app.py                 Streamlit front-end
src/preprocessing.py   resize, denoise, Canny edges
src/contours.py        contour tracing + Douglas–Peucker simplification
src/fourier.py         DFT → epicycle coefficients, reconstruction, RMS error
src/piecewise.py       parametric cubic splines
src/symbolic.py        SymPy / LaTeX equations
src/render.py          matplotlib figures, GIF animation, poster
src/latex_export.py    .tex document + matplotlib-rendered PDF
src/desmos_export.py   Desmos expressions, JSON state, standalone HTML
src/explainer.py       "Explain the math" text from live numbers
src/theme.py           palette + custom colormaps (single source of truth)
src/ui.py              hero, HUD tiles, cards, live epicycle viewer
src/samples.py         procedurally drawn demo shapes
assets/                stylesheet + epicycle viewer template
```

---

## Tech

Python · NumPy · SciPy · OpenCV · SymPy · Matplotlib · Streamlit

---

## License

MIT
