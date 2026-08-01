"""
app.py — Streamlit UI for the Image-to-Function Renderer.

Features (Tier 3 baseline + Tier B additions):
  - Image upload, webcam capture, or built-in examples  [B-1]
  - Method selector: Fourier epicycles vs Piecewise splines
  - Fidelity / complexity slider
  - Canny edge threshold controls
  - Live static reconstruction preview
  - Progressive reconstruction scrubber with RMS error  [B-2]
  - "Explain the math" panel per tab (static tier)      [B-3]
  - Animated epicycle GIF download
  - LaTeX .tex file download + direct PDF generation
  - Print-ready SVG + PDF poster export                  [B-4]
  - Desmos HTML / JSON export
"""

from __future__ import annotations

import io
import time
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import yaml

# ── Project modules ────────────────────────────────────────────────────────
from src.preprocessing import preprocess_array
from src.contours import get_all_contours
from src.fourier import dft_pipeline, reconstruct_path, compute_rms_error
from src.piecewise import spline_pipeline
from src.symbolic import fourier_to_latex_lines, spline_to_latex_lines
from src.render import (
    plot_reconstruction,
    fig_to_png_bytes,
    animate_epicycles,
    animation_to_gif_bytes,
    plot_poster,
    fig_to_svg_bytes,
    fig_to_pdf_bytes,
)
from src.latex_export import build_latex_document, generate_pdf_bytes
from src.desmos_export import (
    build_desmos_expression_list,
    expression_list_to_desmos_state,
    build_desmos_html,
)
from src.explainer import (
    explain_reconstruction,
    explain_epicycles,
    explain_equations,
)


# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RETROGRADE // SYSTEM",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Inter:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #f8f9fb;
    color: #1a1a2e;
}
.stApp { background: #f8f9fb; min-height: 100vh; }

[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e2e8f0;
    box-shadow: 2px 0 12px rgba(0,0,0,0.06);
}
[data-testid="stSidebar"] > div:first-child { padding-top: 1.5rem; }
[data-testid="stSidebarContent"] label,
[data-testid="stSidebarContent"] .stMarkdown p { color: #64748b !important; font-size: 0.85rem; }

.sidebar-section {
    display: flex; align-items: center; gap: 8px;
    padding: 10px 0 6px 0; color: #4f46e5 !important;
    font-size: 0.75rem; font-weight: 700; letter-spacing: 0.12em;
    text-transform: uppercase; border-bottom: 1px solid #e2e8f0; margin-bottom: 10px;
}
.sidebar-section::before { content: '>>'; color: #e11d48; }

.hero-title {
    font-family: 'Share Tech Mono', monospace;
    font-size: 3rem; font-weight: 400; color: #1a1a2e;
    margin-bottom: 0.2rem; letter-spacing: 0.03em; text-transform: uppercase;
}
.hero-subtitle { font-size: 1rem; color: #64748b; margin-bottom: 0; line-height: 1.6; }
.hero-subtitle b { color: #e11d48; }
.hero-badge {
    display: inline-block; padding: 4px 12px; font-size: 0.75rem;
    font-weight: 700; letter-spacing: 0.08em; margin-right: 8px;
    text-transform: uppercase; border: 1px solid; border-radius: 4px;
}
.badge-fourier { color: #4f46e5; border-color: #4f46e5; background: rgba(79,70,229,0.08); }
.badge-spline  { color: #e11d48; border-color: #e11d48; background: rgba(225,29,72,0.08); }
.badge-math    { color: #0891b2; border-color: #0891b2; background: rgba(8,145,178,0.08); }

.stat-row { display: flex; gap: 15px; margin: 1.5rem 0 1rem 0; }
.stat-card {
    flex: 1; background: #ffffff; border: 1px solid #e2e8f0;
    border-radius: 8px; padding: 15px 20px; box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.stat-value { font-size: 1.8rem; font-weight: 700; color: #4f46e5; line-height: 1; margin-bottom: 6px; }
.stat-label { font-size: 0.75rem; color: #94a3b8; letter-spacing: 0.08em; text-transform: uppercase; }
.stat-icon { position: absolute; top: 15px; right: 20px; font-size: 1.2rem; color: #e11d48; font-weight: bold; font-family: 'Share Tech Mono', monospace; }

.card-title {
    font-size: 0.85rem; font-weight: 700; color: #4f46e5;
    text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 1rem;
    display: flex; align-items: center; gap: 8px;
}
.card-title::before { content: '//'; color: #e11d48; }
.card-title::after { content: ''; flex: 1; height: 1px; background: #e2e8f0; }

.info-box {
    background: #f1f5f9; border-left: 3px solid #4f46e5;
    padding: 0.8rem 1rem; margin: 0.6rem 0; font-size: 0.85rem;
    color: #475569; line-height: 1.5; border-radius: 0 4px 4px 0;
}
.info-box code { background: #e0e7ff; color: #4f46e5; padding: 2px 6px; border-radius: 3px; }
.success-box {
    background: #f0fdf4; border-left: 3px solid #16a34a;
    padding: 0.8rem 1rem; margin: 0.6rem 0; font-size: 0.85rem;
    color: #15803d; border-radius: 0 4px 4px 0;
}
.warning-box {
    background: #fffbeb; border-left: 3px solid #d97706;
    padding: 0.8rem 1rem; margin: 0.6rem 0; font-size: 0.85rem;
    color: #92400e; border-radius: 0 4px 4px 0;
}
.explain-box {
    background: #faf5ff; border-left: 3px solid #7c3aed;
    padding: 1rem 1.2rem; margin: 0.6rem 0; font-size: 0.88rem;
    color: #3b0764; line-height: 1.7; border-radius: 0 4px 4px 0;
}
.scrubber-box {
    background: #f0f9ff; border: 1px solid #bae6fd;
    border-radius: 8px; padding: 1rem 1.2rem; margin: 0.8rem 0;
}
.rms-badge {
    display: inline-block; background: #1e40af; color: #dbeafe;
    font-family: 'Share Tech Mono', monospace; font-size: 0.9rem;
    padding: 4px 14px; border-radius: 4px; font-weight: 700;
    letter-spacing: 0.06em;
}
.webcam-note {
    background: #f0fdf4; border: 1px solid #86efac;
    border-radius: 6px; padding: 0.7rem 1rem; font-size: 0.8rem;
    color: #15803d; margin-bottom: 0.5rem;
}

.dl-card {
    background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px;
    padding: 1.5rem; height: 100%; transition: all 0.2s;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}
.dl-card:hover { border-color: #4f46e5; box-shadow: 0 4px 12px rgba(79,70,229,0.12); }
.dl-icon { font-size: 1.1rem; color: #e11d48; margin-bottom: 10px; font-family: 'Share Tech Mono', monospace; }
.dl-title { font-size: 1rem; font-weight: 700; color: #1a1a2e; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.04em; }
.dl-desc  { font-size: 0.8rem; color: #94a3b8; margin-bottom: 16px; line-height: 1.5; }

.stButton > button {
    background: #4f46e5 !important; color: #ffffff !important; border: none !important;
    border-radius: 6px; font-weight: 600; font-size: 0.85rem; padding: 0.6rem 1.2rem;
    transition: all 0.2s ease; letter-spacing: 0.06em; text-transform: uppercase; width: 100%;
}
.stButton > button:hover { background: #4338ca !important; box-shadow: 0 4px 12px rgba(79,70,229,0.3); }

.stDownloadButton > button {
    background: #ffffff !important; border: 1.5px solid #e11d48 !important; color: #e11d48 !important;
    border-radius: 6px; font-weight: 600; font-size: 0.85rem; transition: all 0.2s;
    width: 100%; letter-spacing: 0.06em; text-transform: uppercase;
}
.stDownloadButton > button:hover { background: #fff1f2 !important; box-shadow: 0 4px 12px rgba(225,29,72,0.15); }

.stTabs [data-baseweb="tab-list"] { background: transparent; border-bottom: 2px solid #e2e8f0; gap: 0; }
.stTabs [data-baseweb="tab"] {
    border-radius: 0; color: #94a3b8; font-weight: 600; font-size: 0.85rem;
    padding: 10px 20px; text-transform: uppercase; letter-spacing: 0.08em; border: none;
}
.stTabs [data-baseweb="tab"]:hover { color: #475569; }
.stTabs [aria-selected="true"] {
    background: transparent !important; color: #4f46e5 !important;
    border-bottom: 2px solid #4f46e5 !important; margin-bottom: -2px;
}

.streamlit-expanderHeader {
    background: #f8fafc !important; border: 1px solid #e2e8f0 !important;
    border-radius: 6px !important; color: #1a1a2e !important;
    font-weight: 600; letter-spacing: 0.04em; text-transform: uppercase;
}
.streamlit-expanderContent {
    background: #ffffff !important; border: 1px solid #e2e8f0 !important;
    border-top: none !important; border-radius: 0 0 6px 6px !important;
}

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #f1f5f9; }
::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #94a3b8; }

hr { border-color: #e2e8f0 !important; }
.stImage > img { border: 1px solid #e2e8f0; border-radius: 6px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
.stRadio label { color: #4f46e5 !important; font-weight: 600; text-transform: uppercase; }
.katex { color: #1a1a2e !important; }
</style>
""", unsafe_allow_html=True)


# ── Config loader ──────────────────────────────────────────────────────────
@st.cache_data
def load_default_config() -> dict:
    cfg_path = Path("config/default.yaml")
    if cfg_path.exists():
        with open(cfg_path) as f:
            return yaml.safe_load(f)
    return {}


DEFAULT_CONFIG = load_default_config()


# ── Sample image generator ─────────────────────────────────────────────────
@st.cache_data
def generate_sample_image() -> np.ndarray:
    """Create a synthetic demo image (star shape) for when no file is uploaded."""
    from PIL import ImageDraw
    import math
    img = Image.new("RGB", (400, 400), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    cx, cy, r_outer, r_inner, n_points = 200, 200, 160, 70, 5
    pts = []
    for i in range(n_points * 2):
        r = r_outer if i % 2 == 0 else r_inner
        angle = math.pi / n_points * i - math.pi / 2
        pts.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
    draw.polygon(pts, outline=(255, 255, 255), fill=None)
    for r in [40, 80, 120]:
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(180, 180, 180), width=2)
    return np.array(img)


# ── Core pipeline (cached) ─────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_pipeline(
    img_bytes: bytes,
    method: str,
    n_terms: int,
    canny_low: int,
    canny_high: int,
    max_dim: int,
    max_contours: int,
    theme: str,
    colormap: str,
) -> dict:
    """Run the full pipeline. Cached so re-runs only happen on parameter change."""
    img_arr = np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
    config = {
        "image": {"max_dim": max_dim, "denoise_kernel": 5},
        "edge": {"canny_low": canny_low, "canny_high": canny_high},
        "contour": {"min_length": 15, "epsilon_fraction": 0.002},
        "fourier": {"n_terms": n_terms},
        "spline": {"n_knots": n_terms},
        "render": {
            "theme": theme,
            "colormap": colormap,
            "line_width": 1.2,
            "figsize": [8, 8],
            "fps": 20,
            "animation_frames": 200,
        },
    }
    edges, original_bgr = preprocess_array(img_arr, config=config)
    contours = get_all_contours(edges, config=config, simplify=True)
    contours = contours[:max_contours]

    results, paths = [], []
    for c in contours:
        if method == "Fourier (Epicycles)":
            res = dft_pipeline(c, config=config)
            paths.append(res["path"])
            results.append(res)
        else:
            try:
                res = spline_pipeline(c, config=config)
                paths.append(res["path"])
                results.append(res)
            except ValueError:
                continue

    return {
        "results": results,
        "paths": paths,
        "contours": contours,
        "config": config,
        "n_contours": len(contours),
        "method": method,
    }


# ── Scrubber helper — reconstruct at k terms (fast, cached) ───────────────
@st.cache_data(show_spinner=False)
def _render_at_k_terms(
    coeffs_tuple: tuple,   # hashable: tuple of (amp, phase, freq, real, imag) per coeff per contour
    k: int,
    config_render: tuple,  # hashable: (colormap, line_width, theme)
) -> bytes:
    """Reconstruct all contours using only the top-k Fourier terms and return PNG bytes.

    This is intentionally separate from run_pipeline so that dragging the
    scrubber slider does NOT re-run edge detection — only the cheap path
    reconstruction is repeated.
    """
    from src.fourier import EpicycleCoeff

    colormap, line_width, theme = config_render
    config = {"render": {"colormap": colormap, "line_width": line_width,
                          "theme": theme, "figsize": [8, 8]}}

    paths_k = []
    for contour_coeffs_raw in coeffs_tuple:
        coeffs = [
            EpicycleCoeff(freq=f, amplitude=a, phase=p, real=r, imag=im)
            for (a, p, f, r, im) in contour_coeffs_raw
        ]
        k_clamped = min(k, len(coeffs))
        path = reconstruct_path(coeffs, n_terms=k_clamped)
        paths_k.append(path)

    fig = plot_reconstruction(paths_k, config=config, title=f"{k} terms")
    png = fig_to_png_bytes(fig, dpi=120)
    plt.close(fig)
    return png


def _coeffs_to_hashable(results: list[dict]) -> tuple:
    """Convert results list into a hashable tuple for st.cache_data."""
    out = []
    for res in results:
        coeffs = res.get("coeffs", [])
        out.append(tuple((c.amplitude, c.phase, c.freq, c.real, c.imag) for c in coeffs))
    return tuple(out)


def _paths_to_hashable(paths: list[np.ndarray]) -> tuple:
    """Convert a list of numpy path arrays to a hashable key for st.cache_data."""
    return tuple(p.tobytes() for p in paths)


@st.cache_data(show_spinner=False)
def _build_poster_bytes(
    paths_key: tuple,          # from _paths_to_hashable — used only as cache key
    paths_data: tuple,         # tuple of (N,2) arrays serialised as bytes
    title: str,
    subtitle: str,
    eq_snippet: str,
    colormap: str,
    line_width: float,
) -> tuple[bytes, bytes]:
    """Render the print-ready poster and return (svg_bytes, pdf_bytes).

    Cached so that clicking a download button (which triggers a Streamlit re-run)
    does NOT re-render the 12x14 figure — only runs again when paths or settings
    actually change.
    """
    # Reconstruct numpy arrays from serialised bytes
    paths = [np.frombuffer(b, dtype=np.float64).reshape(-1, 2) for b in paths_data]
    config = {"render": {"colormap": colormap, "line_width": line_width}}
    fig = plot_poster(
        paths=paths,
        title=title,
        subtitle=subtitle,
        equation_snippet=eq_snippet,
        config=config,
    )
    svg = fig_to_svg_bytes(fig)
    pdf = fig_to_pdf_bytes(fig)
    plt.close(fig)
    return svg, pdf


# ── Spline pipeline — always used for Desmos export ───────────────────────
@st.cache_data(show_spinner=False)
def run_spline_for_desmos(
    img_bytes: bytes,
    canny_low: int,
    canny_high: int,
    max_dim: int,
    max_contours: int,
) -> list[dict]:
    """Run the spline pipeline on every contour, cached for Desmos export."""
    img_arr = np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
    config = {
        "image": {"max_dim": max_dim, "denoise_kernel": 5},
        "edge": {"canny_low": canny_low, "canny_high": canny_high},
        "contour": {"min_length": 15, "epsilon_fraction": 0.002},
        "spline": {"n_knots": 60},
    }
    edges, _ = preprocess_array(img_arr, config=config)
    contours = get_all_contours(edges, config=config, simplify=True)
    contours = contours[:max_contours]
    results = []
    for c in contours:
        try:
            results.append(spline_pipeline(c, config=config))
        except ValueError:
            continue
    return results


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 0.5rem 0 1.2rem;">
        <div style="font-family:'Share Tech Mono', monospace; font-size:2.5rem; color:#1a1a2e; margin-bottom:4px; letter-spacing:0.05em;">RETROGRADE</div>
        <div style="font-size:0.7rem; color:#e11d48; font-weight:700; letter-spacing:0.2em; text-transform:uppercase; margin-top:2px;">SYS // V2.0</div>
    </div>
    """, unsafe_allow_html=True)

    # ── INPUT ──────────────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section">INPUT_SOURCE</div>', unsafe_allow_html=True)

    input_mode = st.radio(
        "Input source",
        ["[ UPLOAD ]", "[ WEBCAM ]", "[ DEMO DATA ]"],
        label_visibility="collapsed",
    )

    uploaded      = None
    webcam_image  = None

    if input_mode == "[ UPLOAD ]":
        uploaded = st.file_uploader(
            "Upload an image", type=["png", "jpg", "jpeg", "bmp", "webp"],
            label_visibility="collapsed",
        )

    elif input_mode == "[ WEBCAM ]":
        st.markdown(
            '<div class="webcam-note">Point the camera at a high-contrast subject. '
            'Bold outlines and good lighting produce the best reconstructions.</div>',
            unsafe_allow_html=True,
        )
        webcam_image = st.camera_input(
            "Capture frame",
            label_visibility="collapsed",
            key="webcam_capture",
        )

    # ── METHOD ──
    st.markdown('<div class="sidebar-section">PROCESSING_METHOD</div>', unsafe_allow_html=True)
    method = st.radio(
        "Representation method",
        ["Fourier (Epicycles)", "Piecewise (Splines)"],
        label_visibility="collapsed",
    )

    # ── FIDELITY ──
    st.markdown('<div class="sidebar-section">FIDELITY_PARAMS</div>', unsafe_allow_html=True)
    n_terms = st.slider(
        "TERMS_KNOTS", min_value=5, max_value=500,
        value=DEFAULT_CONFIG.get("fourier", {}).get("n_terms", 100),
        step=5,
        help="Fourier: number of epicycles. Try 200-300 for complex images.",
    )
    max_contours = st.slider(
        "MAX_CONTOURS", min_value=1, max_value=200, value=25,
        help="Portraits/illustrations need 50-150. More = richer but slower.",
    )
    max_dim = st.slider(
        "RESOLUTION_PX", min_value=128, max_value=1024, value=512, step=64,
        help="Longest side of image fed to pipeline. Higher = more detail.",
    )

    # ── EDGE DETECTION ──
    st.markdown('<div class="sidebar-section">EDGE_DETECTION</div>', unsafe_allow_html=True)
    canny_low  = st.slider("CANNY_LOW",  10, 200,  50, help="Lower = catches fainter edges")
    canny_high = st.slider("CANNY_HIGH", 50, 400, 150, help="Upper hysteresis threshold")

    # ── APPEARANCE ──
    st.markdown('<div class="sidebar-section">RENDER_CONFIG</div>', unsafe_allow_html=True)
    theme = "light"  # light mode only
    colormap = st.selectbox(
        "COLORMAP_PRESET",
        ["plasma", "viridis", "inferno", "magma", "cool", "spring", "turbo", "rainbow"],
    )




# ══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ══════════════════════════════════════════════════════════════════════════════

# ── Hero ──
st.markdown("""
<div style="padding: 1.5rem 0 0.5rem;">
    <div class="hero-title">IMG -> FUNC // SYSTEM</div>
    <div class="hero-subtitle">
        Transform visual input into <b>mathematical models</b>.
        [ Fourier Series ] [ Epicycle Kinetics ] [ Piecewise Splines ]
        Initiate extraction. Export symbolic data.
    </div>
    <div style="margin-top: 1rem;">
        <span class="hero-badge badge-fourier">MODE: FOURIER</span>
        <span class="hero-badge badge-spline">MODE: SPLINE</span>
        <span class="hero-badge badge-math">EXP: SYMBOLIC</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr style='margin: 1rem 0;'>", unsafe_allow_html=True)

# ── Resolve input image ──
if input_mode == "[ UPLOAD ]":
    if uploaded is not None:
        img_bytes = uploaded.read()
        img_pil   = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_name  = Path(uploaded.name).stem
    else:
        st.markdown("""
        <div style="text-align:center; padding:3rem 1rem; color:#a0a0a0; border: 1px dashed rgba(79,70,229,0.4); margin: 2rem; border-radius:8px;">
            <div style="font-family:'Share Tech Mono', monospace; font-size:3rem; color:#4f46e5; margin-bottom:1rem;">[ AWAITING_INPUT ]</div>
            <div style="font-size:1.1rem; font-weight:700; color:#1a1a2e; text-transform:uppercase;">Upload an image to begin</div>
            <div style="font-size:0.85rem; margin-top:0.5rem; letter-spacing:0.1em; color:#94a3b8;">Supports PNG · JPG · BMP · WebP</div>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

elif input_mode == "[ WEBCAM ]":
    if webcam_image is not None:
        img_bytes = webcam_image.read()
        img_pil   = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_name  = "webcam_capture"
    else:
        st.markdown("""
        <div style="text-align:center; padding:3rem 1rem; color:#a0a0a0; border: 1px dashed rgba(22,163,74,0.4); margin: 2rem; border-radius:8px;">
            <div style="font-family:'Share Tech Mono', monospace; font-size:3rem; color:#16a34a; margin-bottom:1rem;">[ WEBCAM_READY ]</div>
            <div style="font-size:1.1rem; font-weight:700; color:#1a1a2e; text-transform:uppercase;">Click the camera button in the sidebar</div>
            <div style="font-size:0.85rem; margin-top:0.5rem; letter-spacing:0.1em; color:#94a3b8;">Position subject, then capture a frame</div>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

else:  # DEMO DATA
    sample_arr = generate_sample_image()
    buf = io.BytesIO()
    Image.fromarray(sample_arr).save(buf, format="PNG")
    img_bytes = buf.getvalue()
    img_pil   = Image.fromarray(sample_arr)
    img_name  = "demo_star"

# ── Run pipeline ──
with st.spinner(""):
    t0 = time.time()
    pipeline_out = run_pipeline(
        img_bytes=img_bytes,
        method=method,
        n_terms=n_terms,
        canny_low=canny_low,
        canny_high=canny_high,
        max_dim=max_dim,
        max_contours=max_contours,
        theme=theme,
        colormap=colormap,
    )
    elapsed = time.time() - t0

results    = pipeline_out["results"]
paths      = pipeline_out["paths"]
config     = pipeline_out["config"]
n_contours = pipeline_out["n_contours"]

# ── Stat cards ──
method_short = "FOURIER" if "Fourier" in method else "SPLINE"
input_short  = {"[ UPLOAD ]": "FILE", "[ WEBCAM ]": "CAM", "[ DEMO DATA ]": "DEMO"}[input_mode]
st.markdown(f"""
<div class="stat-row">
    <div class="stat-card">
        <div class="stat-icon">[C]</div>
        <div class="stat-value">{n_contours}</div>
        <div class="stat-label">CONTOURS EXTRACTED</div>
    </div>
    <div class="stat-card">
        <div class="stat-icon">[T]</div>
        <div class="stat-value">{n_terms}</div>
        <div class="stat-label">TERMS / KNOTS</div>
    </div>
    <div class="stat-card">
        <div class="stat-icon">[M]</div>
        <div class="stat-value">{method_short}</div>
        <div class="stat-label">ALGORITHM</div>
    </div>
    <div class="stat-card">
        <div class="stat-icon">[S]</div>
        <div class="stat-value">{input_short}</div>
        <div class="stat-label">INPUT SOURCE</div>
    </div>
    <div class="stat-card">
        <div class="stat-icon">[T]</div>
        <div class="stat-value">{elapsed:.2f}s</div>
        <div class="stat-label">EXECUTION TIME</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──
tab_static, tab_anim, tab_equations, tab_download = st.tabs([
    " [ RECONSTRUCTION ] ",
    " [ EPICYCLES ] ",
    " [ EQUATIONS ] ",
    " [ EXPORT ] ",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Static Reconstruction
# ══════════════════════════════════════════════════════════════════════════════
with tab_static:
    col_orig, col_recon = st.columns(2, gap="large")

    with col_orig:
        st.markdown('<div class="card-title">ORIGINAL_IMAGE</div>', unsafe_allow_html=True)
        st.image(img_pil, use_container_width=True)

    with col_recon:
        st.markdown('<div class="card-title">STATIC_RECONSTRUCTION</div>', unsafe_allow_html=True)
        if paths:
            fig = plot_reconstruction(
                paths, config=config,
                title=f"{method_short} // {n_terms} terms // {n_contours} contours"
            )
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.markdown(
                f'<div class="success-box">[ SUCCESS ] Rendered {len(paths)} contours via {method} </div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown("""
            <div style="display:flex; flex-direction:column; align-items:center;
                 justify-content:center; height:300px; color:#e11d48; text-align:center;
                 border: 1px dashed rgba(225, 29, 72, 0.5); border-radius:8px;">
                <div style="font-family:'Share Tech Mono', monospace; font-size:2.5rem; margin-bottom:0.8rem;">[ ERROR ]</div>
                <div style="font-weight:700; letter-spacing:0.1em; text-transform:uppercase;">No contours detected</div>
                <div style="font-size:0.85rem; margin-top:0.4rem; color:#a0a0a0;">Adjust Canny thresholds in side panel</div>
            </div>
            """, unsafe_allow_html=True)

    # ── B-2: Progressive Reconstruction Scrubber ─────────────────────────
    if paths and method == "Fourier (Epicycles)" and results:
        st.markdown("<hr style='margin: 1.5rem 0 1rem;'>", unsafe_allow_html=True)
        st.markdown('<div class="card-title">PROGRESSIVE_SCRUBBER</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="info-box">Drag the slider to watch the image resolve from a coarse blob '
            'into its full form as more Fourier terms are added. '
            'The RMS error measures how far this approximation is from the full reconstruction.</div>',
            unsafe_allow_html=True,
        )

        scr_col1, scr_col2 = st.columns([4, 1], gap="medium")
        with scr_col1:
            scrub_k = st.slider(
                "TERMS",
                min_value=1,
                max_value=n_terms,
                value=min(10, n_terms),
                step=1,
                key="scrubber_k",
                label_visibility="collapsed",
            )
        with scr_col2:
            # Compute RMS vs full reconstruction for first contour
            if scrub_k < n_terms:
                from src.fourier import EpicycleCoeff
                c0 = results[0]["coeffs"]
                path_k    = reconstruct_path(c0, n_terms=scrub_k)
                path_full = results[0]["path"]
                rms = compute_rms_error(path_k, path_full)
                st.markdown(
                    f'<br><span class="rms-badge">RMS: {rms:.2f} px</span>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<br><span class="rms-badge">RMS: 0.00 px</span>',
                    unsafe_allow_html=True,
                )

        # Render scrubbed reconstruction
        coeffs_hashable = _coeffs_to_hashable(results)
        config_render   = (colormap, config["render"]["line_width"], theme)
        scrub_png = _render_at_k_terms(coeffs_hashable, scrub_k, config_render)
        st.image(scrub_png, caption=f"Reconstruction at {scrub_k} / {n_terms} terms", use_container_width=True)

    # ── B-3: Explain the Math — Reconstruction tab ───────────────────────
    with st.expander("[ EXPLAIN THE MATH ]  — Reconstruction & Contours"):
        explanation_text = explain_reconstruction(
            n_contours=n_contours,
            method=method,
            n_terms=n_terms,
            colormap=colormap,
            canny_low=canny_low,
            canny_high=canny_high,
        )
        st.markdown(
            f'<div class="explain-box">{explanation_text}</div>',
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Epicycle Animation
# ══════════════════════════════════════════════════════════════════════════════
with tab_anim:
    if method != "Fourier (Epicycles)":
        st.markdown("""
        <div style="text-align:center; padding:3rem 1rem; color:#a0a0a0; border: 1px dashed rgba(234,179,8,0.5); border-radius:8px;">
            <div style="font-family:'Share Tech Mono', monospace; font-size:2.5rem; color:#ca8a04; margin-bottom:0.8rem;">[ MODE_LOCKED ]</div>
            <div style="font-size:1rem; font-weight:700; text-transform:uppercase;">Switch to Fourier mode to enable kinetic animation</div>
        </div>
        """, unsafe_allow_html=True)
    elif not results:
        st.warning("No contours to animate.")
    else:
        st.markdown('<div class="card-title">KINETIC_ANIMATION</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="info-box">Animates contour trajectory as a system of rotating epicycles. '
            'Rendering requires 15-40s processing time.</div>',
            unsafe_allow_html=True,
        )

        anim_col1, anim_col2 = st.columns([3, 1], gap="medium")
        with anim_col1:
            contour_idx = st.selectbox(
                "TARGET_CONTOUR",
                options=list(range(len(results))),
                format_func=lambda i: f"CONTOUR {i + 1} // {len(pipeline_out['contours'][i])} PTS",
                key="anim_contour_idx",
            )
        with anim_col2:
            anim_fps = st.slider("FRAMERATE", 10, 30, 20, key="anim_fps")

        if st.button("[ GENERATE_ANIMATION ]", key="gen_gif"):
            with st.spinner(f"Rendering contour {contour_idx + 1} ..."):
                sel_result  = results[contour_idx]
                coeffs      = sel_result["coeffs"]
                trace_path  = sel_result["path"]
                anim_config = {
                    "render": {
                        "theme": theme,
                        "fps": anim_fps,
                        "animation_frames": 360,
                        "figsize": [7, 7],
                    }
                }
                anim = animate_epicycles(
                    coeffs=coeffs,
                    n_terms=min(n_terms, 80),
                    trace_path=trace_path,
                    config=anim_config,
                )
                gif_bytes = animation_to_gif_bytes(anim, fps=anim_fps)
                plt.close("all")
                st.image(
                    gif_bytes,
                    caption=f"CONTOUR {contour_idx + 1} // {min(n_terms, 80)} EPICYCLES",
                    use_container_width=True,
                )
                st.session_state["gif_bytes"] = gif_bytes
                st.markdown(
                    '<div class="success-box">[ SUCCESS ] GIF rendered. Proceed to EXPORT tab.</div>',
                    unsafe_allow_html=True,
                )

    # ── B-3: Explain the Math — Epicycles tab ────────────────────────────
    if method == "Fourier (Epicycles)" and results:
        with st.expander("[ EXPLAIN THE MATH ]  — Fourier Series & Epicycles"):
            coeffs_list = [r["coeffs"] for r in results]
            explanation_text = explain_epicycles(coeffs_list, n_terms=n_terms)
            st.markdown(
                f'<div class="explain-box">{explanation_text}</div>',
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Symbolic Equations
# ══════════════════════════════════════════════════════════════════════════════
with tab_equations:
    st.markdown('<div class="card-title">SYMBOLIC_DATA</div>', unsafe_allow_html=True)

    eq_ctrl1, eq_ctrl2 = st.columns(2, gap="medium")
    with eq_ctrl1:
        n_show = st.slider(
            "DISPLAY_LIMIT", 1, max(1, len(results)), min(len(results), 5),
        )
    with eq_ctrl2:
        n_sym_terms = st.slider("TERMS_PER_CONTOUR", 1, 20, 4)

    if not results:
        st.warning("[ WARN ] No data buffer available.")
    else:
        st.markdown(
            f'<div class="info-box">Extracted symbolic equations for {n_show} contours. '
            f'Parametric x(t) and y(t) data loaded.</div>',
            unsafe_allow_html=True,
        )

        for idx, res in enumerate(results[:n_show]):
            with st.expander(f"CONTOUR_{idx + 1}", expanded=(idx == 0)):
                if method == "Fourier (Epicycles)":
                    lines = fourier_to_latex_lines(
                        res["coeffs"], n_terms=n_sym_terms, contour_index=idx,
                    )
                else:
                    segs = res.get("segments", [])
                    lines = spline_to_latex_lines(
                        segs, contour_index=idx, max_segments=n_sym_terms,
                    )
                for line in lines:
                    st.latex(line)

    # ── B-3: Explain the Math — Equations tab ────────────────────────────
    if results:
        with st.expander("[ EXPLAIN THE MATH ]  — Reading the Equations"):
            eq_explain_idx = st.number_input(
                "Contour to explain", min_value=1,
                max_value=len(results), value=1, step=1, key="eq_explain_idx",
            ) - 1
            explanation_text = explain_equations(
                results=results,
                method=method,
                contour_idx=int(eq_explain_idx),
                term_idx=0,
            )
            st.markdown(
                f'<div class="explain-box">{explanation_text}</div>',
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Downloads / Export
# ══════════════════════════════════════════════════════════════════════════════
with tab_download:
    st.markdown('<div class="card-title">DATA_EXPORT</div>', unsafe_allow_html=True)

    # ── Row 1: four main export cards ────────────────────────────────────────
    dl1, dl2, dl3, dl4 = st.columns(4, gap="large")

    # ── PNG ──────────────────────────────────────────────────────────────────
    with dl1:
        st.markdown("""
        <div class="dl-card">
            <div class="dl-icon">[ IMG ]</div>
            <div class="dl-title">STATIC PNG</div>
            <div class="dl-desc">High-res reconstruction image.</div>
        </div>
        """, unsafe_allow_html=True)
        if paths:
            fig = plot_reconstruction(paths, config=config, title=f"{method_short}")
            png_bytes = fig_to_png_bytes(fig, dpi=200)
            plt.close(fig)
            st.download_button(
                "Download PNG",
                data=png_bytes,
                file_name=f"{img_name}_reconstruction.png",
                mime="image/png",
                key="dl_png",
            )
        else:
            st.markdown('<div class="info-box" style="font-size:0.8rem;">[ AWAITING_PIPELINE ]</div>', unsafe_allow_html=True)

    # ── GIF ──────────────────────────────────────────────────────────────────
    with dl2:
        st.markdown("""
        <div class="dl-card">
            <div class="dl-icon">[ KIN ]</div>
            <div class="dl-title">KINETICS GIF</div>
            <div class="dl-desc">Animated epicycle loop.</div>
        </div>
        """, unsafe_allow_html=True)
        gif_bytes = st.session_state.get("gif_bytes")
        if gif_bytes:
            st.download_button(
                "Download GIF",
                data=gif_bytes,
                file_name=f"{img_name}_epicycles.gif",
                mime="image/gif",
                key="dl_gif",
            )
        else:
            st.markdown(
                '<div class="info-box" style="font-size:0.8rem;">Generate in Epicycles tab first.</div>',
                unsafe_allow_html=True,
            )

    # ── PDF / LaTeX ───────────────────────────────────────────────────────────
    with dl3:
        st.markdown("""
        <div class="dl-card">
            <div class="dl-icon">[ SYM ]</div>
            <div class="dl-title">SYMBOLIC DATA</div>
            <div class="dl-desc">LaTeX equations &amp; compiled PDF.</div>
        </div>
        """, unsafe_allow_html=True)

        if results:
            _method_key = "fourier" if "Fourier" in method else "piecewise"

            if st.button("Build PDF", key="gen_pdf"):
                with st.spinner("Building payload..."):
                    pdf_data = generate_pdf_bytes(
                        results,
                        method=_method_key,
                        image_name=img_name,
                        n_terms=6,
                        max_contours=min(len(results), 20),
                    )
                    st.session_state["pdf_bytes"] = pdf_data
                    st.markdown(
                        '<div class="success-box">[ PDF_READY ]</div>',
                        unsafe_allow_html=True,
                    )

            pdf_data = st.session_state.get("pdf_bytes")
            if pdf_data:
                st.download_button(
                    "Download PDF",
                    data=pdf_data,
                    file_name=f"{img_name}_functions.pdf",
                    mime="application/pdf",
                    key="dl_pdf",
                )

            tex_content = build_latex_document(
                results,
                method=_method_key,
                image_name=img_name,
                n_symbolic_terms=6,
                max_contours=15,
            )
            st.download_button(
                "Download .tex",
                data=tex_content.encode("utf-8"),
                file_name=f"{img_name}_functions.tex",
                mime="text/plain",
                key="dl_tex",
            )
        else:
            st.markdown('<div class="info-box" style="font-size:0.8rem;">[ AWAITING_PIPELINE ]</div>', unsafe_allow_html=True)

    # ── Desmos ───────────────────────────────────────────────────────────────
    with dl4:
        st.markdown("""
        <div class="dl-card">
            <div class="dl-icon">[ DSM ]</div>
            <div class="dl-title">DESMOS GRAPH</div>
            <div class="dl-desc">Spline curves, exact in Desmos.</div>
        </div>
        """, unsafe_allow_html=True)

        if results:
            with st.spinner("Building spline graph..."):
                dsm_results = run_spline_for_desmos(
                    img_bytes=img_bytes,
                    canny_low=canny_low,
                    canny_high=canny_high,
                    max_dim=max_dim,
                    max_contours=min(max_contours, 60),
                )

            if dsm_results:
                desmos_exprs = build_desmos_expression_list(
                    dsm_results,
                    method="piecewise",
                    max_contours=len(dsm_results),
                    max_segs_per_contour=30,
                )

                desmos_state = expression_list_to_desmos_state(desmos_exprs)
                st.download_button(
                    "Desmos JSON",
                    data=desmos_state.encode("utf-8"),
                    file_name=f"{img_name}_desmos.json",
                    mime="application/json",
                    key="dl_dsm_json",
                    help="desmos.com → hamburger menu → Load Graph",
                )

                desmos_html_str = build_desmos_html(
                    desmos_exprs,
                    title=f"retrograde - {img_name}",
                )
                st.download_button(
                    "Desmos HTML",
                    data=desmos_html_str.encode("utf-8"),
                    file_name=f"{img_name}_desmos.html",
                    mime="text/html",
                    key="dl_dsm_html",
                )

                seg_count = sum(len(r.get("segments", [])) for r in dsm_results)
                st.markdown(
                    f'<div class="info-box" style="font-size:0.75rem;">'
                    f'{len(dsm_results)} contours · {seg_count} segments · {len(desmos_exprs)} expr</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown('<div class="info-box" style="font-size:0.8rem;">[ NO_CONTOURS ] Adjust edge thresholds.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="info-box" style="font-size:0.8rem;">[ AWAITING_PIPELINE ]</div>', unsafe_allow_html=True)

    # ── Row 2: Print-ready poster (full-width card + two buttons side by side)
    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
    prt_card, prt_btns = st.columns([2, 3], gap="large")

    with prt_card:
        st.markdown("""
        <div class="dl-card" style="height:100%;">
            <div class="dl-icon">[ PRT ]</div>
            <div class="dl-title">PRINT POSTER</div>
            <div class="dl-desc">Clean vector poster — plotter / laser cutter / print-ready.
            White background, bold title, gradient reconstruction, equation footer.
            12×14 in · vector · SVG + PDF.</div>
        </div>
        """, unsafe_allow_html=True)

    with prt_btns:
        if paths:
            eq_snippet = ""
            if method == "Fourier (Epicycles)" and results:
                c = results[0]["coeffs"]
                if c:
                    eq_snippet = f"z(t) ≈ Σ {c[0].amplitude:.1f}·exp(i·{c[0].freq}t) + ..."
            elif results:
                eq_snippet = "x(t) = a₀ + a₁(t−t₀) + a₂(t−t₀)² + a₃(t−t₀)³"

            _paths_key  = _paths_to_hashable(paths)
            _paths_data = tuple(p.astype(np.float64).tobytes() for p in paths)
            svg_bytes, pdf_poster_bytes = _build_poster_bytes(
                paths_key  = _paths_key,
                paths_data = _paths_data,
                title      = "RETROGRADE",
                subtitle   = f"{method_short} // {n_terms} terms // {n_contours} contours",
                eq_snippet = eq_snippet,
                colormap   = colormap,
                line_width = config["render"]["line_width"],
            )

            st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
            btn_l, btn_r = st.columns(2, gap="medium")
            with btn_l:
                st.download_button(
                    "Download SVG",
                    data=svg_bytes,
                    file_name=f"{img_name}_poster.svg",
                    mime="image/svg+xml",
                    key="dl_svg",
                )
            with btn_r:
                st.download_button(
                    "Download PDF",
                    data=pdf_poster_bytes,
                    file_name=f"{img_name}_poster.pdf",
                    mime="application/pdf",
                    key="dl_pdf_poster",
                )
        else:
            st.markdown('<div class="info-box" style="font-size:0.8rem;">[ AWAITING_PIPELINE ]</div>', unsafe_allow_html=True)





# ── Footer ────────────────────────────────────────────────────────────────
st.markdown("<hr style='margin: 2rem 0 1rem;'>", unsafe_allow_html=True)
st.markdown("""
<div style="display:flex; justify-content:space-between; align-items:center;
     padding: 0 0.5rem 1rem; flex-wrap:wrap; gap:0.5rem;">
    <div style="font-size:0.75rem; color:#1a1a2e; font-weight:700; letter-spacing:0.1em; text-transform:uppercase;">
        [ RETROGRADE ] // IMG_TO_FUNC
    </div>
    <div style="font-size:0.75rem; color:#e11d48; font-family:'Share Tech Mono', monospace;">
        SYS_ONLINE // TIER_B
    </div>
</div>
""", unsafe_allow_html=True)
