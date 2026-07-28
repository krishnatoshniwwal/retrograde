"""
app.py — Streamlit UI for the Image-to-Function Renderer.

Features (Tier 3):
  - Image upload (or built-in examples)
  - Method selector: Fourier epicycles vs Piecewise splines
  - Fidelity / complexity slider
  - Canny edge threshold controls
  - Live static reconstruction preview
  - Animated epicycle GIF download
  - LaTeX .tex file download + direct PDF generation
  - Futuristic / Cyberpunk aesthetic (no emojis)
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
from src.fourier import dft_pipeline
from src.piecewise import spline_pipeline
from src.symbolic import fourier_to_latex_lines, spline_to_latex_lines
from src.render import (
    plot_reconstruction,
    fig_to_png_bytes,
    animate_epicycles,
    animation_to_gif_bytes,
)
from src.latex_export import build_latex_document, generate_pdf_bytes
from src.desmos_export import (
    build_desmos_expression_list,
    expression_list_to_json,
    build_desmos_html,
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
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=JetBrains+Mono:wght@400;700&display=swap');

html, body, [class*="css"] {
    font-family: 'JetBrains Mono', monospace;
    background-color: #030303;
    color: #00f0ff;
}

/* ── Background ── */
.stApp {
    background: #030303;
    background-image: 
        linear-gradient(rgba(0, 240, 255, 0.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0, 240, 255, 0.04) 1px, transparent 1px);
    background-size: 30px 30px;
    min-height: 100vh;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(3, 3, 3, 0.98) !important;
    border-right: 1px solid #00f0ff;
    box-shadow: 2px 0 20px rgba(0, 240, 255, 0.15);
}
[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.5rem;
}
[data-testid="stSidebarContent"] label,
[data-testid="stSidebarContent"] .stMarkdown p {
    color: #a0a0a0 !important;
    font-size: 0.85rem;
}

/* ── Sidebar section headers ── */
.sidebar-section {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 0 6px 0;
    color: #00f0ff !important;
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    border-bottom: 1px solid rgba(0, 240, 255, 0.3);
    margin-bottom: 10px;
}
.sidebar-section::before {
    content: '>>';
    color: #ff003c;
}

/* ── Hero header ── */
.hero-title {
    font-family: 'Share Tech Mono', monospace;
    font-size: 3.5rem;
    font-weight: 400;
    color: #00f0ff;
    text-shadow: 0 0 15px rgba(0, 240, 255, 0.6);
    margin-bottom: 0.2rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.hero-subtitle {
    font-size: 1rem;
    color: #888;
    margin-bottom: 0;
    line-height: 1.6;
}
.hero-subtitle b {
    color: #ff003c;
}
.hero-badge {
    display: inline-block;
    padding: 4px 12px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    margin-right: 8px;
    text-transform: uppercase;
    border: 1px solid;
}
.badge-fourier { color: #00f0ff; border-color: #00f0ff; background: rgba(0, 240, 255, 0.1); }
.badge-spline  { color: #ff003c; border-color: #ff003c; background: rgba(255, 0, 60, 0.1); }
.badge-math    { color: #ffff00; border-color: #ffff00; background: rgba(255, 255, 0, 0.1); }

/* ── Stat cards ── */
.stat-row {
    display: flex;
    gap: 15px;
    margin: 1.5rem 0 1rem 0;
}
.stat-card {
    flex: 1;
    background: rgba(0, 0, 0, 0.8);
    border: 1px solid rgba(0, 240, 255, 0.4);
    padding: 15px 20px;
    position: relative;
    box-shadow: inset 0 0 20px rgba(0, 240, 255, 0.05);
}
.stat-card::before {
    content: ''; position: absolute; top: 0; left: 0;
    width: 20px; height: 20px; border-top: 2px solid #ff003c; border-left: 2px solid #ff003c;
}
.stat-card::after {
    content: ''; position: absolute; bottom: 0; right: 0;
    width: 20px; height: 20px; border-bottom: 2px solid #ff003c; border-right: 2px solid #ff003c;
}
.stat-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #00f0ff;
    line-height: 1;
    margin-bottom: 6px;
    text-shadow: 0 0 8px rgba(0, 240, 255, 0.6);
}
.stat-label {
    font-size: 0.75rem;
    color: #a0a0a0;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.stat-icon {
    position: absolute;
    top: 15px; right: 20px;
    font-size: 1.2rem;
    color: #ff003c;
    font-weight: bold;
    font-family: 'Share Tech Mono', monospace;
}

/* ── Glass cards / Containers ── */
.card-title {
    font-size: 0.9rem;
    font-weight: 700;
    color: #00f0ff;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.card-title::before { content: '//'; color: #ff003c; }
.card-title::after {
    content: ''; flex: 1; height: 1px;
    background: rgba(0, 240, 255, 0.3);
}

/* ── Info / alert boxes ── */
.info-box {
    background: rgba(0, 240, 255, 0.05);
    border-left: 2px solid #00f0ff;
    padding: 0.8rem 1rem;
    margin: 0.6rem 0;
    font-size: 0.85rem;
    color: #a0a0a0;
    line-height: 1.5;
}
.info-box code {
    background: rgba(0, 240, 255, 0.15);
    color: #00f0ff;
    padding: 2px 6px;
    border: 1px solid rgba(0, 240, 255, 0.3);
}
.success-box {
    background: rgba(57, 255, 20, 0.05);
    border-left: 2px solid #39ff14;
    padding: 0.8rem 1rem;
    margin: 0.6rem 0;
    font-size: 0.85rem;
    color: #39ff14;
}
.warning-box {
    background: rgba(255, 255, 0, 0.05);
    border-left: 2px solid #ffff00;
    padding: 0.8rem 1rem;
    margin: 0.6rem 0;
    font-size: 0.85rem;
    color: #ffff00;
}

/* ── Download cards ── */
.dl-card {
    background: rgba(0, 0, 0, 0.8);
    border: 1px solid rgba(0, 240, 255, 0.3);
    padding: 1.5rem;
    height: 100%;
    position: relative;
    transition: all 0.2s;
}
.dl-card:hover { 
    border-color: #00f0ff;
    box-shadow: 0 0 15px rgba(0, 240, 255, 0.2);
}
.dl-icon { font-size: 1.2rem; color: #ff003c; margin-bottom: 10px; font-family: 'Share Tech Mono', monospace; }
.dl-title { font-size: 1rem; font-weight: 700; color: #00f0ff; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.05em; }
.dl-desc  { font-size: 0.8rem; color: #a0a0a0; margin-bottom: 16px; line-height: 1.5; }

/* ── Buttons ── */
.stButton > button {
    background: transparent !important;
    color: #00f0ff !important;
    border: 1px solid #00f0ff !important;
    border-radius: 0;
    font-weight: 700;
    font-size: 0.85rem;
    padding: 0.6rem 1.2rem;
    transition: all 0.2s ease;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    width: 100%;
}
.stButton > button:hover {
    background: rgba(0, 240, 255, 0.1) !important;
    box-shadow: 0 0 15px rgba(0, 240, 255, 0.4);
}

/* Download buttons */
.stDownloadButton > button {
    background: rgba(255, 0, 60, 0.05) !important;
    border: 1px solid #ff003c !important;
    color: #ff003c !important;
    border-radius: 0;
    font-weight: 700;
    font-size: 0.85rem;
    transition: all 0.2s;
    width: 100%;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.stDownloadButton > button:hover {
    background: rgba(255, 0, 60, 0.15) !important;
    box-shadow: 0 0 15px rgba(255, 0, 60, 0.4);
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: 1px solid rgba(0, 240, 255, 0.3);
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 0;
    color: #666;
    font-weight: 700;
    font-size: 0.85rem;
    padding: 10px 20px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    border: 1px solid transparent;
    border-bottom: none;
}
.stTabs [data-baseweb="tab"]:hover { color: #a0a0a0; }
.stTabs [aria-selected="true"] {
    background: rgba(0, 240, 255, 0.05) !important;
    color: #00f0ff !important;
    border: 1px solid #00f0ff !important;
    border-bottom: 1px solid #030303 !important;
    margin-bottom: -1px;
}



/* ── Expanders ── */
.streamlit-expanderHeader {
    background: rgba(0, 0, 0, 0.8) !important;
    border: 1px solid #00f0ff !important;
    border-radius: 0 !important;
    color: #00f0ff !important;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.streamlit-expanderContent {
    background: rgba(0, 0, 0, 0.5) !important;
    border: 1px solid rgba(0, 240, 255, 0.3) !important;
    border-top: none !important;
    border-radius: 0 !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #000; }
::-webkit-scrollbar-thumb { background: #00f0ff; }
::-webkit-scrollbar-thumb:hover { background: #ff003c; }

/* ── Horizontal divider ── */
hr { border-color: rgba(0, 240, 255, 0.2) !important; }

/* ── Image containers ── */
.stImage > img {
    border: 1px solid #00f0ff;
    box-shadow: 0 0 15px rgba(0, 240, 255, 0.15);
}

/* ── Radio & Selectbox ── */
.stRadio label { color: #00f0ff !important; font-weight: 700; text-transform: uppercase; }
.stSelectbox > div > div {
    background: rgba(0, 0, 0, 0.8) !important;
    border: 1px solid #00f0ff !important;
    border-radius: 0 !important;
    color: #00f0ff !important;
}

/* ── LaTeX rendering ── */
.katex { color: #00f0ff !important; }
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


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 0.5rem 0 1.2rem;">
        <div style="font-family:'Share Tech Mono', monospace; font-size:2.5rem; color:#00f0ff; text-shadow:0 0 15px rgba(0,240,255,0.6); margin-bottom:4px; letter-spacing:0.05em;">RETROGRADE</div>
        <div style="font-size:0.7rem; color:#ff003c; font-weight:700; letter-spacing:0.2em; text-transform:uppercase; margin-top:2px;">SYS // V1.0</div>
    </div>
    """, unsafe_allow_html=True)

    # ── INPUT ──
    st.markdown('<div class="sidebar-section">INPUT_SOURCE</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Upload an image", type=["png", "jpg", "jpeg", "bmp", "webp"],
        label_visibility="collapsed",
    )
    use_sample = st.checkbox("[ USE DEMO DATA ]", value=(uploaded is None))

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
    theme = st.selectbox("THEME_PROFILE", ["dark", "light"])
    colormap = st.selectbox(
        "COLORMAP_PRESET",
        ["plasma", "viridis", "inferno", "magma", "cool", "spring", "turbo", "rainbow"],
    )

    st.markdown("""
    <div style="margin-top:2rem; padding: 12px; background:rgba(0,240,255,0.05);
         border:1px solid #00f0ff; text-align:center;">
        <div style="font-size:0.68rem; color:#00f0ff; font-weight:700; letter-spacing:0.1em;">NUMPY // SCIPY // SYMPY</div>
    </div>
    """, unsafe_allow_html=True)


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
if uploaded is not None:
    img_bytes = uploaded.read()
    img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_name = Path(uploaded.name).stem
elif use_sample:
    sample_arr = generate_sample_image()
    buf = io.BytesIO()
    Image.fromarray(sample_arr).save(buf, format="PNG")
    img_bytes = buf.getvalue()
    img_pil = Image.fromarray(sample_arr)
    img_name = "demo_star"
else:
    st.markdown("""
    <div style="text-align:center; padding:3rem 1rem; color:#a0a0a0; border: 1px dashed rgba(0, 240, 255, 0.5); margin: 2rem;">
        <div style="font-family:'Share Tech Mono', monospace; font-size:3rem; color:#ff003c; margin-bottom:1rem;">[ AWAITING_INPUT ]</div>
        <div style="font-size:1.1rem; font-weight:700; color:#00f0ff; text-transform:uppercase;">Upload image or enable demo mode</div>
        <div style="font-size:0.85rem; margin-top:0.5rem; letter-spacing:0.1em;">Configure parameters in sidebar</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

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

results   = pipeline_out["results"]
paths     = pipeline_out["paths"]
config    = pipeline_out["config"]
n_contours = pipeline_out["n_contours"]

# ── Stat cards ──
method_short = "FOURIER" if "Fourier" in method else "SPLINE"
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
                 justify-content:center; height:300px; color:#ff003c; text-align:center;
                 border: 1px dashed rgba(255, 0, 60, 0.5);">
                <div style="font-family:'Share Tech Mono', monospace; font-size:2.5rem; margin-bottom:0.8rem;">[ ERROR ]</div>
                <div style="font-weight:700; letter-spacing:0.1em; text-transform:uppercase;">No contours detected</div>
                <div style="font-size:0.85rem; margin-top:0.4rem; color:#a0a0a0;">Adjust Canny thresholds in side panel</div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Epicycle Animation
# ══════════════════════════════════════════════════════════════════════════════
with tab_anim:
    if method != "Fourier (Epicycles)":
        st.markdown("""
        <div style="text-align:center; padding:3rem 1rem; color:#a0a0a0; border: 1px dashed rgba(255, 255, 0, 0.5);">
            <div style="font-family:'Share Tech Mono', monospace; font-size:2.5rem; color:#ffff00; margin-bottom:0.8rem;">[ MODE_LOCKED ]</div>
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


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Downloads
# ══════════════════════════════════════════════════════════════════════════════
with tab_download:
    st.markdown('<div class="card-title">DATA_EXPORT</div>', unsafe_allow_html=True)

    dl1, dl2, dl3, dl4 = st.columns(4, gap="large")

    # ── PNG ──────────────────────────────────────────────────────────────────
    with dl1:
        st.markdown("""
        <div class="dl-card">
            <div class="dl-icon">[ IMG ]</div>
            <div class="dl-title">STATIC_RENDER.PNG</div>
            <div class="dl-desc">High-res rendering of trajectory map.</div>
        </div>
        """, unsafe_allow_html=True)
        if paths:
            fig = plot_reconstruction(paths, config=config, title=f"{method_short}")
            png_bytes = fig_to_png_bytes(fig, dpi=200)
            plt.close(fig)
            st.download_button(
                "[ DL_PNG ]",
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
            <div class="dl-title">KINETICS.GIF</div>
            <div class="dl-desc">Animated loop of epicycle mechanism.</div>
        </div>
        """, unsafe_allow_html=True)
        gif_bytes = st.session_state.get("gif_bytes")
        if gif_bytes:
            st.download_button(
                "[ DL_GIF ]",
                data=gif_bytes,
                file_name=f"{img_name}_epicycles.gif",
                mime="image/gif",
                key="dl_gif",
            )
        else:
            st.markdown(
                '<div class="info-box" style="font-size:0.8rem;">[ REQUIRES_GENERATION ] Render in EPICYCLES tab.</div>',
                unsafe_allow_html=True,
            )

    # ── PDF / LaTeX ───────────────────────────────────────────────────────────
    with dl3:
        st.markdown("""
        <div class="dl-card">
            <div class="dl-icon">[ SYM ]</div>
            <div class="dl-title">SYMBOLIC_DATA</div>
            <div class="dl-desc">Raw LaTeX equations and compiled PDF payload.</div>
        </div>
        """, unsafe_allow_html=True)

        if results:
            _method_key = "fourier" if "Fourier" in method else "piecewise"

            if st.button("[ RENDER_PDF ]", key="gen_pdf"):
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
                    "[ DL_PDF ]",
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
                "[ DL_TEX ]",
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
            <div class="dl-title">DESMOS_GRAPH</div>
            <div class="dl-desc">Interactive graph + JSON expression list for Desmos.</div>
        </div>
        """, unsafe_allow_html=True)

        if results:
            _method_key_dsm = "fourier" if "Fourier" in method else "piecewise"
            _n_terms_dsm = n_terms if _method_key_dsm == "fourier" else 80

            desmos_exprs = build_desmos_expression_list(
                results,
                method=_method_key_dsm,
                n_terms=_n_terms_dsm,
                max_contours=min(len(results), 60),
            )

            desmos_json = expression_list_to_json(desmos_exprs)
            st.download_button(
                "[ DL_DESMOS_JSON ]",
                data=desmos_json.encode("utf-8"),
                file_name=f"{img_name}_desmos.json",
                mime="application/json",
                key="dl_dsm_json",
            )

            desmos_html_str = build_desmos_html(
                desmos_exprs,
                title=f"retrograde — {img_name}",
            )
            st.download_button(
                "[ DL_DESMOS_HTML ]",
                data=desmos_html_str.encode("utf-8"),
                file_name=f"{img_name}_desmos.html",
                mime="text/html",
                key="dl_dsm_html",
            )

            st.markdown(
                f'<div class="info-box" style="font-size:0.75rem;">'
                f'[ EXPR_COUNT ] {len(desmos_exprs)} expressions · '
                f'{min(len(results), 60)} contours</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown('<div class="info-box" style="font-size:0.8rem;">[ AWAITING_PIPELINE ]</div>', unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────
st.markdown("<hr style='margin: 2rem 0 1rem;'>", unsafe_allow_html=True)
st.markdown("""
<div style="display:flex; justify-content:space-between; align-items:center;
     padding: 0 0.5rem 1rem; flex-wrap:wrap; gap:0.5rem;">
    <div style="font-size:0.75rem; color:#00f0ff; font-weight:700; letter-spacing:0.1em; text-transform:uppercase;">
        [ RETROGRADE ] // IMG_TO_FUNC
    </div>
    <div style="font-size:0.75rem; color:#ff003c; font-family:'Share Tech Mono', monospace;">
        SYS_ONLINE
    </div>
</div>
""", unsafe_allow_html=True)
