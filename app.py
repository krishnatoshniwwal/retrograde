"""
app.py — Retrograde: turn any image into the mathematics that draws it.

Streamlit front-end.  The heavy lifting lives in ``src/``; visual building
blocks live in ``src/ui.py`` and ``assets/``.

Features
  - Image upload, camera capture, or procedurally drawn demo shapes
  - Fourier epicycles or piecewise cubic splines
  - Side-by-side source / edge map / reconstruction
  - Progressive Fourier scrubber with RMS error meter
  - Live in-browser epicycle animation (all contours at once) + GIF render
  - Symbolic equations with copyable LaTeX
  - Exports: PNG, GIF, LaTeX, PDF, Desmos (JSON/HTML), SVG/PDF poster
  - "Explain the math" panels filled with numbers from the current run
"""

from __future__ import annotations

import hashlib
import io
import time
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import yaml
from PIL import Image

from src import ui
from src.contours import get_all_contours
from src.desmos_export import (
    build_desmos_expression_list,
    build_desmos_html,
    expression_list_to_desmos_state,
)
from src.explainer import explain_epicycles, explain_equations, explain_reconstruction
from src.fourier import compute_rms_error, dft_pipeline, reconstruct_path
from src.latex_export import build_latex_document, generate_pdf_bytes
from src.piecewise import spline_pipeline
from src.preprocessing import preprocess_array
from src.render import (
    animate_epicycles,
    animation_to_gif_bytes,
    fig_to_pdf_bytes,
    fig_to_png_bytes,
    fig_to_svg_bytes,
    plot_poster,
    plot_reconstruction,
)
from src.samples import DEMO_SHAPES
from src.symbolic import fourier_to_latex_lines, spline_to_latex_lines
from src.theme import COLORMAP_CHOICES, NIGHT

FOURIER = "Fourier (Epicycles)"
SPLINE = "Piecewise (Splines)"
METHODS = {"Fourier": FOURIER, "Splines": SPLINE}
THEME = "night"

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Retrograde · image → function",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="auto",
)
st.markdown(ui.load_css(), unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────
@st.cache_data
def load_default_config() -> dict:
    cfg_path = Path("config/default.yaml")
    if cfg_path.exists():
        with open(cfg_path) as f:
            return yaml.safe_load(f) or {}
    return {}


DEFAULT_CONFIG = load_default_config()


@st.cache_data(show_spinner=False)
def demo_png(name: str) -> bytes:
    buf = io.BytesIO()
    DEMO_SHAPES[name]().save(buf, format="PNG")
    return buf.getvalue()


def memo(name: str, key: Any, fn: Callable[[], Any]) -> Any:
    """Per-session single-slot memo.

    Pipeline outputs hold scipy spline objects that can't be pickled, so
    ``st.cache_data`` is out; this keeps the latest value per ``name`` in
    session state and recomputes only when ``key`` changes.
    """
    store = st.session_state.setdefault("_memo", {})
    hit = store.get(name)
    if hit is not None and hit[0] == key:
        return hit[1]
    value = fn()
    store[name] = (key, value)
    return value


def build_config(n_terms: int, canny_low: int, canny_high: int, max_dim: int,
                 colormap: str) -> dict:
    return {
        "image": {"max_dim": max_dim, "denoise_kernel": 5},
        "edge": {"canny_low": canny_low, "canny_high": canny_high},
        "contour": {"min_length": 15, "epsilon_fraction": 0.002},
        "fourier": {"n_terms": n_terms},
        "spline": {"n_knots": n_terms},
        "render": {
            "theme": THEME,
            "colormap": colormap,
            "line_width": 1.3,
            "figsize": [8, 8],
            "fps": 20,
            "animation_frames": 200,
        },
    }


def run_pipeline(img_bytes: bytes, method: str, config: dict, max_contours: int) -> dict:
    """Edge detection → contours → Fourier or spline fit for every contour."""
    img_arr = np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
    edges, _ = preprocess_array(img_arr, config=config)
    contours = get_all_contours(edges, config=config, simplify=True)[:max_contours]

    results, paths = [], []
    for c in contours:
        try:
            res = dft_pipeline(c, config=config) if method == FOURIER else spline_pipeline(c, config=config)
        except ValueError:
            continue
        results.append(res)
        paths.append(res["path"])

    return {
        "results": results,
        "paths": paths,
        "contours": contours,
        "edges": edges,
        "n_contours": len(contours),
        "n_points": int(sum(len(c) for c in contours)),
    }


def run_spline_for_desmos(img_bytes: bytes, canny_low: int, canny_high: int,
                          max_dim: int, max_contours: int) -> list[dict]:
    """Spline fits for every contour — Desmos plots splines exactly."""
    img_arr = np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
    config = {
        "image": {"max_dim": max_dim, "denoise_kernel": 5},
        "edge": {"canny_low": canny_low, "canny_high": canny_high},
        "contour": {"min_length": 15, "epsilon_fraction": 0.002},
        "spline": {"n_knots": 60},
    }
    edges, _ = preprocess_array(img_arr, config=config)
    contours = get_all_contours(edges, config=config, simplify=True)[:max_contours]
    out = []
    for c in contours:
        try:
            out.append(spline_pipeline(c, config=config))
        except ValueError:
            continue
    return out


def edges_to_rgb(edges: np.ndarray) -> np.ndarray:
    """Colourise a binary edge map in the night palette."""
    bg = np.array([int(NIGHT["bg_deep"][i:i + 2], 16) for i in (1, 3, 5)], dtype=np.uint8)
    fg = np.array([int(NIGHT["accent"][i:i + 2], 16) for i in (1, 3, 5)], dtype=np.uint8)
    out = np.empty((*edges.shape, 3), dtype=np.uint8)
    out[:] = bg
    out[edges > 0] = fg
    return out


def figure_png(paths: list[np.ndarray], config: dict, title: str = "", dpi: int = 130) -> bytes:
    fig = plot_reconstruction(paths, config=config, title=title)
    png = fig_to_png_bytes(fig, dpi=dpi)
    plt.close(fig)
    return png


@st.cache_data(show_spinner=False, max_entries=64)
def render_at_k_terms(pipeline_key: str, k: int, colormap: str, _results: list[dict]) -> bytes:
    """Reconstruct every contour from its top-k Fourier terms (scrubber).

    ``_results`` is excluded from hashing (leading underscore); the pipeline
    key identifies it, so dragging the slider never re-runs edge detection.
    """
    paths_k = [reconstruct_path(r["coeffs"], n_terms=min(k, len(r["coeffs"]))) for r in _results]
    cfg = {"render": {"colormap": colormap, "line_width": 1.3, "theme": THEME, "figsize": [8, 8]}}
    return figure_png(paths_k, cfg, dpi=110)


def set_state(key: str, value: Any) -> None:
    st.session_state[key] = value


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(ui.sidebar_brand(), unsafe_allow_html=True)

    st.markdown(ui.sidebar_step(1, "Source"), unsafe_allow_html=True)
    input_mode = st.segmented_control(
        "Input source", ["Upload", "Camera", "Demo"],
        default="Upload", required=True, key="input_mode", label_visibility="collapsed",
        width="stretch",
    )
    uploaded = webcam_image = None
    demo_name = None
    if input_mode == "Upload":
        uploaded = st.file_uploader(
            "Upload an image", type=["png", "jpg", "jpeg", "bmp", "webp"],
            label_visibility="collapsed",
        )
    elif input_mode == "Camera":
        st.markdown(ui.callout("High-contrast subjects with bold outlines and even lighting "
                               "give the cleanest curves."), unsafe_allow_html=True)
        webcam_image = st.camera_input("Capture frame", label_visibility="collapsed", key="webcam_capture")
    else:
        demo_name = st.selectbox("Demo shape", list(DEMO_SHAPES), key="demo_shape")

    st.markdown(ui.sidebar_step(2, "Method"), unsafe_allow_html=True)
    method_label = st.segmented_control(
        "Representation", list(METHODS), default="Fourier", required=True,
        key="method_label", label_visibility="collapsed", width="stretch",
    )
    method = METHODS[method_label]
    st.caption("Rotating circles from a Fourier transform." if method == FOURIER
               else "Smooth cubic polynomials stitched end to end.")

    st.markdown(ui.sidebar_step(3, "Fidelity"), unsafe_allow_html=True)
    n_terms = st.slider(
        "Circles per contour" if method == FOURIER else "Spline knots",
        min_value=5, max_value=500, step=5,
        value=int(DEFAULT_CONFIG.get("fourier", {}).get("n_terms", 100)),
        help="More terms = finer detail. 200–300 suits complex images.",
    )
    max_contours = st.slider("Max contours", 1, 200, 25,
                             help="Portraits and illustrations need 50–150. More = richer but slower.")
    max_dim = st.slider("Working resolution (px)", 128, 1024, 512, step=64,
                        help="Longest side of the image fed to the pipeline.")

    st.markdown(ui.sidebar_step(4, "Edges"), unsafe_allow_html=True)
    canny_low, canny_high = st.slider(
        "Canny thresholds (low, high)", 10, 400, (50, 150),
        help="Lower values catch fainter edges; higher values keep only bold outlines.",
    )

    st.markdown(ui.sidebar_step(5, "Palette"), unsafe_allow_html=True)
    colormap = st.selectbox("Colormap", COLORMAP_CHOICES, index=0, label_visibility="collapsed")
    st.markdown(ui.colormap_swatch(colormap), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(ui.hero(), unsafe_allow_html=True)

# ── Resolve input image ──
img_bytes: bytes | None = None
img_name = "image"
if input_mode == "Upload" and uploaded is not None:
    img_bytes, img_name = uploaded.getvalue(), Path(uploaded.name).stem
elif input_mode == "Camera" and webcam_image is not None:
    img_bytes, img_name = webcam_image.getvalue(), "camera_capture"
elif input_mode == "Demo":
    img_bytes, img_name = demo_png(demo_name), demo_name.lower().replace(" ", "_")

if img_bytes is None:
    if input_mode == "Camera":
        st.markdown(ui.empty_state("Camera ready",
                                   "Use the camera panel in the sidebar to capture a frame."),
                    unsafe_allow_html=True)
    else:
        st.markdown(ui.empty_state("Drop an image to begin",
                                   "Upload a PNG, JPG, BMP or WebP in the sidebar &mdash; "
                                   "or start with one of the built-in shapes."),
                    unsafe_allow_html=True)
        _, mid, _ = st.columns([1, 1, 1])
        with mid:
            st.button("◆  Try a demo shape", type="primary", width="stretch",
                      on_click=set_state, args=("input_mode", "Demo"))
    st.markdown(ui.footer(), unsafe_allow_html=True)
    st.stop()

img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
img_hash = hashlib.sha1(img_bytes).hexdigest()

# ── Pipeline (memoised per session) ──
config = build_config(n_terms, canny_low, canny_high, max_dim, colormap)
pipeline_key = f"{img_hash}|{method}|{n_terms}|{canny_low}|{canny_high}|{max_dim}|{max_contours}"


def _run() -> dict:
    t0 = time.time()
    out = run_pipeline(img_bytes, method, config, max_contours)
    out["elapsed"] = time.time() - t0
    return out


with st.spinner("Tracing curves…"):
    pipe = memo("pipeline", pipeline_key, _run)

results: list[dict] = pipe["results"]
paths: list[np.ndarray] = pipe["paths"]
n_contours: int = pipe["n_contours"]
is_fourier = method == FOURIER
method_short = "Fourier" if is_fourier else "Spline"

st.markdown(ui.stat_tiles([
    (str(n_contours), "contours traced", "lime"),
    (f"{pipe['n_points']:,}", "points sampled", "cyan"),
    (str(n_terms), "circles / contour" if is_fourier else "knots / contour", "pink"),
    (method_short, "representation", "moon"),
    (f"{pipe['elapsed']:.2f}s", "solve time", "lime"),
]), unsafe_allow_html=True)

tab_recon, tab_anim, tab_eq, tab_export = st.tabs([
    ":material/draw: Reconstruction",
    ":material/cyclone: Epicycles",
    ":material/function: Equations",
    ":material/download: Export",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Reconstruction
# ══════════════════════════════════════════════════════════════════════════════
with tab_recon:
    st.markdown(ui.section_header(
        "Picture → equations", "01",
        "Left: what went in. Right: the same outlines redrawn purely from the fitted functions "
        "&mdash; no pixels, just math evaluated along each curve."),
        unsafe_allow_html=True)

    col_src, col_out = st.columns(2, gap="large")
    with col_src:
        with st.container(key="frame_src"):
            view = st.segmented_control("View", ["Original", "Edge map"], default="Original",
                                        required=True, key="src_view", label_visibility="collapsed")
            if view == "Original":
                st.markdown(ui.frame_label("Source", f"{img_pil.width}×{img_pil.height}"), unsafe_allow_html=True)
                st.image(img_pil, width="stretch")
            else:
                st.markdown(ui.frame_label("Canny edges", f"low {canny_low} · high {canny_high}"), unsafe_allow_html=True)
                st.image(edges_to_rgb(pipe["edges"]), width="stretch")

    with col_out:
        with st.container(key="frame_out"):
            st.markdown(ui.frame_label("Reconstruction", f"{method_short} · {n_terms} terms"), unsafe_allow_html=True)
            if paths:
                png = memo("recon_png", (pipeline_key, colormap),
                           lambda: figure_png(paths, config, dpi=130))
                st.image(png, width="stretch")
            else:
                st.markdown(ui.callout("No contours detected. Try lowering the Canny thresholds "
                                       "or raising the working resolution.", "warn"),
                            unsafe_allow_html=True)
    if paths:
        st.markdown(ui.callout(f"Rendered <b>{len(paths)}</b> contours from "
                               f"<b>{pipe['n_points']:,}</b> traced points.", "ok"),
                    unsafe_allow_html=True)

    # ── Progressive scrubber ──
    if paths and is_fourier and results:
        st.markdown(ui.section_header(
            "Watch it resolve", "02",
            "Drag to add Fourier terms one at a time. A handful of circles gives a blurry blob; "
            "each extra term adds finer detail. RMS error is measured on the first contour "
            "against the full reconstruction."),
            unsafe_allow_html=True)
        scrub_k = st.slider("Terms", 1, n_terms, min(10, n_terms), key="scrubber_k",
                            label_visibility="collapsed")
        rms = 0.0
        if scrub_k < n_terms:
            c0 = results[0]["coeffs"]
            rms = compute_rms_error(reconstruct_path(c0, n_terms=scrub_k), results[0]["path"])
        st.markdown(ui.rms_meter(rms, scrub_k, n_terms), unsafe_allow_html=True)
        sc_l, sc_mid, sc_r = st.columns([1, 2, 1])
        with sc_mid:
            with st.container(key="frame_scrub"):
                st.image(render_at_k_terms(pipeline_key, scrub_k, colormap, results),
                         caption=f"{scrub_k} / {n_terms} terms", width="stretch")

    with st.expander("Explain the math — edges, contours & fitting", icon=":material/school:"):
        st.markdown(explain_reconstruction(
            n_contours=n_contours, method=method, n_terms=n_terms,
            colormap=colormap, canny_low=canny_low, canny_high=canny_high,
        ))


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Epicycles
# ══════════════════════════════════════════════════════════════════════════════
with tab_anim:
    if not is_fourier:
        st.markdown(ui.empty_state("Epicycles need Fourier mode",
                                   "Splines are polynomials, not circles. Switch the method to "
                                   "Fourier to watch the gears turn."),
                    unsafe_allow_html=True)
        _, mid, _ = st.columns([1, 1, 1])
        with mid:
            st.button("Switch to Fourier", type="primary", width="stretch",
                      on_click=set_state, args=("method_label", "Fourier"))
    elif not results:
        st.markdown(ui.callout("No contours to animate — adjust the edge thresholds.", "warn"),
                    unsafe_allow_html=True)
    else:
        st.markdown(ui.section_header(
            "Live epicycles", "01",
            "Every contour, drawn at once by its own chain of rotating circles. "
            "Runs in your browser — press <b>space</b> to pause, <b>Next</b> to move the focus."),
            unsafe_allow_html=True)
        viewer = memo("viewer", (pipeline_key, colormap), lambda: ui.epicycle_viewer_html(
            results, paths, n_terms=n_terms, colormap=colormap, height=560))
        st.iframe(viewer, height=572)

        st.markdown(ui.section_header(
            "Render a GIF", "02",
            "Bake one contour's epicycles into a looping GIF you can share. "
            "Takes 15–40 s on the server."),
            unsafe_allow_html=True)
        g1, g2, g3 = st.columns([3, 2, 2], gap="medium", vertical_alignment="bottom")
        with g1:
            contour_idx = st.selectbox(
                "Contour", options=list(range(len(results))),
                format_func=lambda i: f"Contour {i + 1} · {len(pipe['contours'][i])} points",
                key="anim_contour_idx",
            )
        with g2:
            anim_fps = st.slider("Frame rate", 10, 30, 20, key="anim_fps")
        with g3:
            make_gif = st.button("Render GIF", type="primary", width="stretch", key="gen_gif")

        if make_gif:
            with st.spinner(f"Rendering contour {contour_idx + 1}…"):
                sel = results[contour_idx]
                anim = animate_epicycles(
                    coeffs=sel["coeffs"], n_terms=min(n_terms, 80), trace_path=sel["path"],
                    config={"render": {"theme": THEME, "fps": anim_fps,
                                       "animation_frames": 360, "figsize": [7, 7]}},
                )
                st.session_state["gif_bytes"] = animation_to_gif_bytes(anim, fps=anim_fps)
                st.session_state["gif_label"] = f"Contour {contour_idx + 1} · {min(n_terms, 80)} epicycles"
                plt.close("all")

        if st.session_state.get("gif_bytes"):
            gl, gm, gr = st.columns([1, 2, 1])
            with gm:
                with st.container(key="frame_gif"):
                    st.image(st.session_state["gif_bytes"],
                             caption=st.session_state.get("gif_label", ""), width="stretch")
                st.download_button("Download GIF", st.session_state["gif_bytes"],
                                   file_name=f"{img_name}_epicycles.gif", mime="image/gif",
                                   width="stretch", key="dl_gif_inline")

        with st.expander("Explain the math — Fourier series & epicycles", icon=":material/school:"):
            st.markdown(explain_epicycles([r["coeffs"] for r in results], n_terms=n_terms))


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Equations
# ══════════════════════════════════════════════════════════════════════════════
with tab_eq:
    st.markdown(ui.section_header(
        "The functions", "01",
        "Parametric equations x(t), y(t) for each contour, with t running from 0 to 2π. "
        + ("Each term is one spinning circle." if is_fourier else "Each line is one cubic piece of the spline.")),
        unsafe_allow_html=True)

    if not results:
        st.markdown(ui.callout("No equations yet — no contours were detected.", "warn"), unsafe_allow_html=True)
    else:
        e1, e2 = st.columns(2, gap="large")
        with e1:
            n_show = st.slider("Contours to show", 1, max(2, len(results)), min(len(results), 5),
                               disabled=len(results) < 2)
        with e2:
            n_sym_terms = st.slider("Terms per contour" if is_fourier else "Segments per contour", 1, 20, 4)

        for idx, res in enumerate(results[:n_show]):
            with st.expander(f"Contour {idx + 1}", expanded=(idx == 0), icon=":material/timeline:"):
                if is_fourier:
                    lines = fourier_to_latex_lines(res["coeffs"], n_terms=n_sym_terms, contour_index=idx)
                else:
                    lines = spline_to_latex_lines(res.get("segments", []), contour_index=idx,
                                                  max_segments=n_sym_terms)
                for line in lines:
                    st.latex(line)
                st.code("\n".join(lines), language="latex", wrap_lines=True)

        with st.expander("Explain the math — reading the equations", icon=":material/school:"):
            eq_idx = st.number_input("Contour to explain", min_value=1, max_value=len(results),
                                     value=1, step=1, key="eq_explain_idx") - 1
            st.markdown(explain_equations(results=results, method=method,
                                          contour_idx=int(eq_idx), term_idx=0))


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Export
# ══════════════════════════════════════════════════════════════════════════════
GLYPH = {
    "png": f'<svg width="22" height="22" viewBox="0 0 22 22" shape-rendering="crispEdges"><rect x="2" y="4" width="18" height="14" fill="none" stroke="{NIGHT["accent"]}" stroke-width="2"/><rect x="6" y="8" width="3" height="3" fill="{NIGHT["moon"]}"/><path d="M4 16 L9 12 L12 14 L16 9 L18 11 L18 16 Z" fill="{NIGHT["primary"]}"/></svg>',
    "gif": f'<svg width="22" height="22" viewBox="0 0 22 22"><circle cx="9" cy="11" r="6" fill="none" stroke="{NIGHT["accent"]}" stroke-width="2"/><circle cx="15" cy="11" r="3" fill="none" stroke="{NIGHT["sakura"]}" stroke-width="2"/><rect x="16" y="9" width="4" height="4" fill="{NIGHT["primary"]}"/></svg>',
    "tex": f'<svg width="22" height="22" viewBox="0 0 22 22"><text x="11" y="16" text-anchor="middle" font-family="serif" font-size="15" font-style="italic" fill="{NIGHT["primary"]}">&#931;</text></svg>',
    "dsm": f'<svg width="22" height="22" viewBox="0 0 22 22" shape-rendering="crispEdges"><path d="M3 3 V19 H19" fill="none" stroke="{NIGHT["muted"]}" stroke-width="2"/><path d="M5 16 Q9 2 13 10 T19 5" fill="none" stroke="{NIGHT["accent"]}" stroke-width="2" shape-rendering="auto"/></svg>',
}

with tab_export:
    st.markdown(ui.section_header(
        "Take the math with you", "01",
        "Files are generated when you click, so nothing slows the page down."),
        unsafe_allow_html=True)

    if not results:
        st.markdown(ui.callout("Nothing to export yet — no contours were detected.", "warn"),
                    unsafe_allow_html=True)
    else:
        _method_key = "fourier" if is_fourier else "piecewise"
        # Bind current values for the deferred download callables.
        _paths, _results, _config = paths, results, dict(config)
        _bytes = img_bytes
        _cl, _ch, _md, _mc = canny_low, canny_high, max_dim, min(max_contours, 60)

        def _png() -> bytes:
            return figure_png(_paths, _config, title="", dpi=220)

        def _tex() -> bytes:
            return build_latex_document(_results, method=_method_key, image_name=img_name,
                                        n_symbolic_terms=6, max_contours=15).encode("utf-8")

        def _pdf() -> bytes:
            return generate_pdf_bytes(_results, method=_method_key, image_name=img_name,
                                      n_terms=6, max_contours=min(len(_results), 20))

        def _desmos_exprs() -> list[dict]:
            dsm = run_spline_for_desmos(_bytes, _cl, _ch, _md, _mc)
            return build_desmos_expression_list(dsm, method="piecewise", max_contours=len(dsm),
                                                max_segs_per_contour=30)

        def _desmos_json() -> bytes:
            return expression_list_to_desmos_state(_desmos_exprs()).encode("utf-8")

        def _desmos_html() -> bytes:
            return build_desmos_html(_desmos_exprs(), title=f"Retrograde · {img_name}").encode("utf-8")

        c1, c2, c3, c4 = st.columns(4, gap="medium")
        with c1:
            with st.container(key="card_png"):
                st.markdown(ui.export_card_head(GLYPH["png"], "Image", "High-resolution render of the reconstruction on the night sky.", "PNG"), unsafe_allow_html=True)
                st.download_button("Download PNG", _png, file_name=f"{img_name}_reconstruction.png",
                                   mime="image/png", type="primary", width="stretch", key="dl_png")
        with c2:
            with st.container(key="card_gif"):
                st.markdown(ui.export_card_head(GLYPH["gif"], "Animation", "Looping epicycle GIF of a single contour.", "GIF"), unsafe_allow_html=True)
                if st.session_state.get("gif_bytes"):
                    st.download_button("Download GIF", st.session_state["gif_bytes"],
                                       file_name=f"{img_name}_epicycles.gif", mime="image/gif",
                                       type="primary", width="stretch", key="dl_gif")
                else:
                    st.button("Render first", disabled=True, width="stretch", key="gif_hint",
                              help="Open the Epicycles tab and press Render GIF first.")
        with c3:
            with st.container(key="card_tex"):
                st.markdown(ui.export_card_head(GLYPH["tex"], "Equations", "Typeset document of every contour's x(t), y(t).", "PDF · TEX"), unsafe_allow_html=True)
                st.download_button("Download PDF", _pdf, file_name=f"{img_name}_functions.pdf",
                                   mime="application/pdf", type="primary", width="stretch", key="dl_pdf")
                st.download_button("Download .tex", _tex, file_name=f"{img_name}_functions.tex",
                                   mime="text/plain", width="stretch", key="dl_tex")
        with c4:
            with st.container(key="card_dsm"):
                st.markdown(ui.export_card_head(GLYPH["dsm"], "Desmos", "Spline curves you can pan, zoom and edit in Desmos.", "HTML · JSON"), unsafe_allow_html=True)
                st.download_button("Desmos HTML", _desmos_html, file_name=f"{img_name}_desmos.html",
                                   mime="text/html", type="primary", width="stretch", key="dl_dsm_html")
                st.download_button("Desmos JSON", _desmos_json, file_name=f"{img_name}_desmos.json",
                                   mime="application/json", width="stretch", key="dl_dsm_json",
                                   help="desmos.com → menu → Load graph")

        # ── Poster ──
        st.markdown(ui.section_header(
            "Print-ready poster", "02",
            "A 12×14 in vector poster — ready for a printer, pen plotter or laser cutter."),
            unsafe_allow_html=True)

        eq_snippet = ""
        if is_fourier and results[0].get("coeffs"):
            c = results[0]["coeffs"][0]
            eq_snippet = f"z(t) ≈ Σ {c.amplitude:.1f}·exp(i·{c.freq}t) + …"
        elif results:
            eq_snippet = "x(t) = a₀ + a₁(t−t₀) + a₂(t−t₀)² + a₃(t−t₀)³"
        subtitle = f"{method_short.lower()} · {n_terms} terms · {n_contours} contours"

        p_prev, p_ctrl = st.columns([2, 3], gap="large")
        with p_ctrl:
            poster_style = st.segmented_control("Poster style", ["Paper", "Night"], default="Paper",
                                                required=True, key="poster_style")
            poster_title = st.text_input("Title", value="RETROGRADE", max_chars=28, key="poster_title")
            _pstyle = poster_style.lower()

            def _poster_fig():
                return plot_poster(paths=_paths, title=poster_title, subtitle=subtitle,
                                   equation_snippet=eq_snippet, config=_config, style=_pstyle)

            def _poster(kind: str) -> Callable[[], bytes]:
                def make() -> bytes:
                    fig = _poster_fig()
                    data = fig_to_svg_bytes(fig) if kind == "svg" else fig_to_pdf_bytes(fig)
                    plt.close(fig)
                    return data
                return make

            st.markdown(ui.callout("SVG keeps every curve as a vector path — open it in Inkscape, "
                                   "Illustrator or a plotter tool. PDF is ready to print."),
                        unsafe_allow_html=True)
            b1, b2 = st.columns(2, gap="medium")
            with b1:
                st.download_button("Download SVG", _poster("svg"), file_name=f"{img_name}_poster.svg",
                                   mime="image/svg+xml", type="primary", width="stretch", key="dl_svg")
            with b2:
                st.download_button("Download PDF", _poster("pdf"), file_name=f"{img_name}_poster.pdf",
                                   mime="application/pdf", width="stretch", key="dl_pdf_poster")

        with p_prev:
            def _preview() -> bytes:
                fig = _poster_fig()
                data = fig_to_png_bytes(fig, dpi=48)
                plt.close(fig)
                return data

            with st.container(key="frame_poster"):
                st.image(memo("poster_preview", (pipeline_key, colormap, _pstyle, poster_title), _preview),
                         caption="Preview", width="stretch")


st.markdown(ui.footer(), unsafe_allow_html=True)
