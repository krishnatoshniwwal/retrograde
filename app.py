"""
app.py — Streamlit UI for the Image-to-Function Renderer.

Features (Tier 3):
  - Image upload (or built-in examples)
  - Method selector: Fourier epicycles vs Piecewise splines
  - Fidelity / complexity slider
  - Canny edge threshold controls
  - Live static reconstruction preview
  - Animated epicycle GIF download
  - LaTeX .tex file download
  - Batch / gallery output
  - Dark-mode aesthetic with live caching
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
from src.fourier import dft_pipeline, compute_dft, reconstruct_path
from src.piecewise import spline_pipeline
from src.symbolic import fourier_to_latex_lines, spline_to_latex_lines
from src.render import (
    plot_reconstruction,
    fig_to_png_bytes,
    animate_epicycles,
    animation_to_gif_bytes,
)
from src.latex_export import build_latex_document, generate_pdf_bytes


# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Image-to-Function Renderer",
    page_icon="🌀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark gradient background */
.stApp {
    background: linear-gradient(135deg, #0a0a0f 0%, #0d0d1a 50%, #0a0a0f 100%);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0f1e 0%, #13132a 100%);
    border-right: 1px solid #1e1e3a;
}

/* Headers */
h1 { color: #c084fc !important; }
h2 { color: #a78bfa !important; }
h3 { color: #818cf8 !important; }

/* Metric cards */
[data-testid="stMetric"] {
    background: rgba(139, 92, 246, 0.08);
    border: 1px solid rgba(139, 92, 246, 0.2);
    border-radius: 12px;
    padding: 0.75rem 1rem;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #4f46e5);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.2s;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(124, 58, 237, 0.4);
}

/* Info boxes */
.info-box {
    background: rgba(99, 102, 241, 0.08);
    border-left: 3px solid #6366f1;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    margin: 0.5rem 0;
    font-size: 0.9rem;
    color: #c7d2fe;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(15, 15, 30, 0.6);
    border-radius: 10px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #94a3b8;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: rgba(124, 58, 237, 0.3) !important;
    color: #c084fc !important;
}
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
    img = Image.new("RGB", (400, 400), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Draw a star
    import math
    cx, cy, r_outer, r_inner, n_points = 200, 200, 160, 70, 5
    pts = []
    for i in range(n_points * 2):
        r = r_outer if i % 2 == 0 else r_inner
        angle = math.pi / n_points * i - math.pi / 2
        pts.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
    draw.polygon(pts, outline=(255, 255, 255), fill=None)

    # Add inner circles
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

    results = []
    paths = []
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


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌀 Image-to-Function")
    st.markdown("*Convert images to mathematical functions*")
    st.divider()

    st.markdown("### 📁 Input Image")
    uploaded = st.file_uploader(
        "Upload an image", type=["png", "jpg", "jpeg", "bmp", "webp"],
        label_visibility="collapsed",
    )
    use_sample = st.checkbox("Use built-in demo image", value=(uploaded is None))

    st.divider()
    st.markdown("### ⚙️ Method")
    method = st.radio(
        "Representation method",
        ["Fourier (Epicycles)", "Piecewise (Splines)"],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("### 🎛️ Fidelity")
    n_terms = st.slider(
        "Terms / Knots",
        min_value=5, max_value=500,
        value=DEFAULT_CONFIG.get("fourier", {}).get("n_terms", 100),
        step=5,
        help="Fourier: number of epicycles (more = sharper corners). Splines: number of knots. Try 200–300 for complex images.",
    )
    max_contours = st.slider(
        "Max contours", min_value=1, max_value=200, value=25,
        help="Number of contours processed. Complex images (portraits, illustrations) need 50–150. More = richer output but slower.",
    )
    max_dim = st.slider(
        "Image resolution (px)", min_value=128, max_value=1024, value=512, step=64,
        help="Longest side of the image fed into the pipeline.",
    )

    st.divider()
    st.markdown("### 🔬 Edge Detection")
    canny_low = st.slider("Canny low threshold", 10, 200, 50)
    canny_high = st.slider("Canny high threshold", 50, 400, 150)

    st.divider()
    st.markdown("### 🎨 Appearance")
    theme = st.selectbox("Theme", ["dark", "light"])
    colormap = st.selectbox(
        "Colormap",
        ["plasma", "viridis", "inferno", "magma", "cool", "spring", "turbo", "rainbow"],
        index=0,
    )

    st.divider()
    st.caption("Image-to-Function Renderer v1.0")


# ── Main content ───────────────────────────────────────────────────────────
st.markdown("# 🌀 Image-to-Function Renderer")
st.markdown(
    "Converts images into **mathematical functions** — "
    "Fourier series (epicycles) or piecewise splines — "
    "then animates and exports the equations."
)

# Determine input image
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
    st.info("👈 Upload an image or enable the demo to get started.")
    st.stop()

# ── Run pipeline ───────────────────────────────────────────────────────────
with st.spinner("🔄 Running pipeline…"):
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

results = pipeline_out["results"]
paths = pipeline_out["paths"]
config = pipeline_out["config"]
n_contours = pipeline_out["n_contours"]

# ── Stats bar ──────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Contours detected", n_contours)
col2.metric("Method", method.split()[0])
col3.metric("Terms / knots", n_terms)
col4.metric("Pipeline time", f"{elapsed:.2f}s")

st.divider()

# ── Tabs ───────────────────────────────────────────────────────────────────
tab_static, tab_anim, tab_equations, tab_download = st.tabs([
    "📊 Static Reconstruction",
    "🔄 Epicycle Animation",
    "📐 Symbolic Equations",
    "⬇️ Download",
])

# ── Tab 1: Static reconstruction ──────────────────────────────────────────
with tab_static:
    col_orig, col_recon = st.columns(2)

    with col_orig:
        st.markdown("### Original Image")
        st.image(img_pil, use_container_width=True)

    with col_recon:
        st.markdown("### Reconstruction")
        if paths:
            fig = plot_reconstruction(paths, config=config, title=f"{method} Reconstruction")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.markdown(
                f'<div class="info-box">✅ Rendered {len(paths)} contour path(s) '
                f'using the <b>{method}</b> method with <b>{n_terms}</b> terms.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.warning("No contours found. Try lowering the Canny thresholds.")

# ── Tab 2: Epicycle animation ──────────────────────────────────────────────
with tab_anim:
    if method != "Fourier (Epicycles)":
        st.info("Switch to **Fourier (Epicycles)** method to enable animation.")
    elif not results:
        st.warning("No contours to animate.")
    else:
        st.markdown("### Epicycle Animation")
        st.markdown(
            '<div class="info-box">🎬 Animates <b>one contour</b> at a time as '
            'rotating epicycles tracing the path. '
            'Click <b>Generate GIF</b> — rendering takes 15–40 s.</div>',
            unsafe_allow_html=True,
        )

        anim_col1, anim_col2 = st.columns([2, 1])
        with anim_col1:
            contour_idx = st.selectbox(
                "Which contour to animate",
                options=list(range(len(results))),
                format_func=lambda i: f"Contour {i + 1} ({len(pipeline_out['contours'][i])} pts)",
                key="anim_contour_idx",
            )
        with anim_col2:
            anim_fps = st.slider("Animation FPS", 10, 30, 20, key="anim_fps")

        if st.button("🎬 Generate Epicycle GIF", key="gen_gif"):
            with st.spinner(f"Rendering contour {contour_idx + 1} animation ({n_terms} epicycles, 360 frames)…"):
                sel_result = results[contour_idx]
                coeffs = sel_result["coeffs"]
                trace_path = sel_result["path"]
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
                st.image(gif_bytes, caption=f"Contour {contour_idx + 1} — {min(n_terms, 80)} epicycles",
                         use_container_width=True)
                st.session_state["gif_bytes"] = gif_bytes
                st.success("GIF ready! Go to the Download tab.")

# ── Tab 3: Symbolic equations ──────────────────────────────────────────────
with tab_equations:
    st.markdown("### Symbolic Equations")
    n_show = st.slider("Contours to show", 1, max(1, len(results)), min(len(results), 5))
    n_sym_terms = st.slider("Terms per contour", 1, 20, 4)

    if not results:
        st.warning("No results to display.")
    else:
        for idx, res in enumerate(results[:n_show]):
            with st.expander(f"Contour {idx + 1}", expanded=(idx == 0)):
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

# ── Tab 4: Downloads ───────────────────────────────────────────────────────
with tab_download:
    st.markdown("### Download Outputs")

    dl_col1, dl_col2, dl_col3 = st.columns(3)

    # PNG download
    with dl_col1:
        st.markdown("**📸 Reconstruction PNG**")
        if paths:
            fig = plot_reconstruction(paths, config=config, title=f"{method}")
            png_bytes = fig_to_png_bytes(fig, dpi=200)
            plt.close(fig)
            st.download_button(
                "⬇️ Download PNG",
                data=png_bytes,
                file_name=f"{img_name}_reconstruction.png",
                mime="image/png",
                key="dl_png",
            )
        else:
            st.info("Run pipeline first.")

    # GIF download
    with dl_col2:
        st.markdown("**🎬 Animation GIF**")
        gif_bytes = st.session_state.get("gif_bytes")
        if gif_bytes:
            st.download_button(
                "⬇️ Download GIF",
                data=gif_bytes,
                file_name=f"{img_name}_epicycles.gif",
                mime="image/gif",
                key="dl_gif",
            )
        else:
            st.info("Generate GIF in the Animation tab first.")

    # LaTeX / PDF download — split into two sub-columns
    with dl_col3:
        st.markdown("**📄 Equations Export**")
        if results:
            _method_key = "fourier" if "Fourier" in method else "piecewise"

            sub_pdf, sub_tex = st.columns(2)

            with sub_pdf:
                if st.button("🖨️ Generate PDF", key="gen_pdf"):
                    with st.spinner("Building PDF…"):
                        pdf_data = generate_pdf_bytes(
                            results,
                            method=_method_key,
                            image_name=img_name,
                            n_terms=6,
                            max_contours=min(len(results), 20),
                        )
                        st.session_state["pdf_bytes"] = pdf_data
                        st.success("PDF ready!")

                pdf_data = st.session_state.get("pdf_bytes")
                if pdf_data:
                    st.download_button(
                        "⬇️ Download PDF",
                        data=pdf_data,
                        file_name=f"{img_name}_functions.pdf",
                        mime="application/pdf",
                        key="dl_pdf",
                    )

            with sub_tex:
                tex_content = build_latex_document(
                    results,
                    method=_method_key,
                    image_name=img_name,
                    n_symbolic_terms=6,
                    max_contours=15,
                )
                st.download_button(
                    "⬇️ Download .tex",
                    data=tex_content.encode("utf-8"),
                    file_name=f"{img_name}_functions.tex",
                    mime="text/plain",
                    key="dl_tex",
                )
            st.markdown(
                '<div class="info-box">PDF uses matplotlib mathtext (no LaTeX install needed). '
                '.tex can be compiled with <code>pdflatex</code> for full typesetting.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.info("Run pipeline first.")

st.divider()
st.markdown(
    "<center style='color:#4b5563; font-size:0.8rem;'>"
    "Image-to-Function Renderer &nbsp;·&nbsp; "
    "Fourier series &amp; piecewise splines &nbsp;·&nbsp; "
    "Built with NumPy, SciPy, SymPy, Matplotlib &amp; Streamlit"
    "</center>",
    unsafe_allow_html=True,
)
