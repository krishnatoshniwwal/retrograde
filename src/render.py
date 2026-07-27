"""
render.py — Matplotlib-based static reconstruction and epicycle animation.

Supports:
- Static contour plots with gradient colouring (LineCollection)
- Dark/light themes
- Animated epicycle reconstruction (GIF/MP4 via imageio)
- Gallery HTML page for batch outputs
"""

from __future__ import annotations

import io
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, safe for Streamlit & CI
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Theme helpers
# ---------------------------------------------------------------------------

DARK_THEME = {
    "bg": "#0d0d0d",
    "fg": "#f0f0f0",
    "grid": "#1e1e1e",
}
LIGHT_THEME = {
    "bg": "#fafafa",
    "fg": "#111111",
    "grid": "#e0e0e0",
}


def _apply_theme(fig: plt.Figure, ax: plt.Axes, theme: str = "dark") -> None:
    """Apply dark or light aesthetic theme to a figure."""
    t = DARK_THEME if theme == "dark" else LIGHT_THEME
    fig.patch.set_facecolor(t["bg"])
    ax.set_facecolor(t["bg"])
    ax.tick_params(colors=t["fg"])
    for spine in ax.spines.values():
        spine.set_edgecolor(t["grid"])


def _make_figure(config: dict[str, Any]) -> tuple[plt.Figure, plt.Axes]:
    """Create a figure with theme applied."""
    cfg_r = config.get("render", {})
    figsize = tuple(cfg_r.get("figsize", [10, 10]))
    theme = cfg_r.get("theme", "dark")

    fig, ax = plt.subplots(figsize=figsize)
    _apply_theme(fig, ax, theme=theme)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


# ---------------------------------------------------------------------------
# Gradient path drawing
# ---------------------------------------------------------------------------

def _path_to_segments(path: np.ndarray) -> np.ndarray:
    """Convert (N, 2) path to (N-1, 2, 2) segment array for LineCollection."""
    pts = path.reshape(-1, 1, 2)
    return np.concatenate([pts[:-1], pts[1:]], axis=1)


def plot_path_gradient(
    ax: plt.Axes,
    path: np.ndarray,
    colormap: str = "plasma",
    line_width: float = 1.2,
    alpha: float = 0.9,
) -> None:
    """Draw a path with a continuous colour gradient along its length.

    Uses LineCollection for efficient rendering even with thousands of segments.

    Args:
        ax: Target matplotlib Axes.
        path: (N, 2) array of (x, y) points.
        colormap: matplotlib colormap name.
        line_width: Stroke width.
        alpha: Opacity.
    """
    if len(path) < 2:
        return
    segs = _path_to_segments(path)
    colors = np.linspace(0, 1, len(segs))
    lc = LineCollection(segs, cmap=colormap, linewidth=line_width, alpha=alpha)
    lc.set_array(colors)
    ax.add_collection(lc)


# ---------------------------------------------------------------------------
# Static reconstruction plot
# ---------------------------------------------------------------------------

def plot_reconstruction(
    paths: list[np.ndarray],
    config: dict[str, Any] | None = None,
    title: str = "Reconstruction",
) -> plt.Figure:
    """Create a static reconstruction figure from a list of paths.

    Each path is a (N, 2) array; all are drawn with gradient colouring.

    Args:
        paths: List of (N, 2) arrays.
        config: Optional config dict.
        title: Figure title.

    Returns:
        matplotlib Figure (caller owns it; call fig.savefig or plt.close).
    """
    config = config or {}
    cfg_r = config.get("render", {})
    colormap = cfg_r.get("colormap", "plasma")
    line_width = cfg_r.get("line_width", 1.2)
    theme = cfg_r.get("theme", "dark")
    fg = DARK_THEME["fg"] if theme == "dark" else LIGHT_THEME["fg"]

    fig, ax = _make_figure(config)

    # Flip y for image coordinates (origin top-left → bottom-left) BEFORE
    # computing axis limits so limits match the drawn paths.
    flipped_paths = []
    for p in paths:
        if len(p) < 2:
            continue
        fp = p.copy()
        fp[:, 1] = -fp[:, 1]
        flipped_paths.append(fp)

    if not flipped_paths:
        ax.set_title(title, color=fg, pad=12, fontsize=13, fontweight="bold")
        fig.tight_layout(pad=0.5)
        return fig

    # Determine axis limits from flipped paths
    all_pts = np.vstack(flipped_paths)
    x_min, y_min = all_pts.min(axis=0)
    x_max, y_max = all_pts.max(axis=0)
    margin = max(x_max - x_min, y_max - y_min) * 0.05 + 1.0
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)

    for flipped in flipped_paths:
        plot_path_gradient(ax, flipped, colormap=colormap, line_width=line_width)

    ax.set_title(title, color=fg, pad=12, fontsize=13, fontweight="bold")
    fig.tight_layout(pad=0.5)
    return fig


def fig_to_png_bytes(fig: plt.Figure, dpi: int = 150) -> bytes:
    """Render a figure to PNG bytes (for Streamlit / download).

    Args:
        fig: matplotlib Figure.
        dpi: Resolution.

    Returns:
        PNG bytes.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# Epicycle animation
# ---------------------------------------------------------------------------

def _draw_epicycles(
    ax: plt.Axes,
    frame_data: list[tuple[float, float, float]],
    theme: str = "dark",
) -> list:
    """Draw one frame of epicycles: circles + arm lines.

    Args:
        ax: Matplotlib Axes (pre-cleared).
        frame_data: List of (cx, cy, radius) for each epicycle.
        theme: "dark" or "light".

    Returns:
        List of drawn artists (not used externally but kept for reference).
    """
    t = DARK_THEME if theme == "dark" else LIGHT_THEME
    artists = []

    prev_x, prev_y = 0.0, 0.0
    for i, (cx, cy, radius) in enumerate(frame_data):
        # Draw circle outline
        circle = plt.Circle(
            (prev_x, prev_y), radius,
            color="#3a7ca5", fill=False, linewidth=0.5, alpha=0.3,
        )
        ax.add_patch(circle)
        # Draw arm line from previous centre to current
        line, = ax.plot(
            [prev_x, cx], [prev_y, cy],
            color="#7ec8e3", linewidth=0.8, alpha=0.7,
        )
        artists.extend([circle, line])
        prev_x, prev_y = cx, cy

    # Draw tip dot
    tip, = ax.plot(prev_x, prev_y, "o", color="#ff6b6b", markersize=3, alpha=0.9)
    artists.append(tip)
    return artists


def animate_epicycles(
    coeffs: list,
    n_terms: int,
    trace_path: np.ndarray,
    config: dict[str, Any] | None = None,
) -> FuncAnimation:
    """Create a matplotlib FuncAnimation showing the epicycle construction.

    Args:
        coeffs: List of EpicycleCoeff from fourier.compute_dft.
        n_terms: Number of epicycles to animate.
        trace_path: (F, 2) array — the full reconstructed path for tracing.
        config: Optional config dict.

    Returns:
        matplotlib FuncAnimation object.
    """
    from src.fourier import get_epicycle_frames

    config = config or {}
    cfg_r = config.get("render", {})
    theme = cfg_r.get("theme", "dark")
    fps = cfg_r.get("fps", 30)
    n_frames = cfg_r.get("animation_frames", 300)
    figsize = tuple(cfg_r.get("figsize", [10, 10]))

    t_theme = DARK_THEME if theme == "dark" else LIGHT_THEME

    # Pre-compute all frame data
    frames_data = get_epicycle_frames(coeffs, n_terms=n_terms, n_frames=n_frames)

    # Axis limits: use epicycle arm reach (sum of top-N amplitudes) so arms
    # never go off-screen, centered on the trace centroid.
    terms = coeffs[:n_terms]
    max_reach = sum(c.amplitude for c in terms)

    flipped = trace_path.copy()
    flipped[:, 1] = -flipped[:, 1]
    cx = float(flipped[:, 0].mean())
    cy = float(flipped[:, 1].mean())
    pad = max_reach * 1.1

    xlim = (cx - pad, cx + pad)
    ylim = (cy - pad, cy + pad)

    fig, ax = plt.subplots(figsize=figsize)
    _apply_theme(fig, ax, theme=theme)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")

    # The accumulated trace line
    trace_line, = ax.plot([], [], color="#ff6b6b", linewidth=1.0, alpha=0.8)
    trace_x: list[float] = []
    trace_y: list[float] = []
    dynamic_artists: list = []

    def init():
        trace_line.set_data([], [])
        return [trace_line]

    def update(frame_idx: int):
        nonlocal dynamic_artists
        # Remove previous frame's artists
        for artist in dynamic_artists:
            artist.remove()
        dynamic_artists = []

        frame = frames_data[frame_idx]
        # The tip of the last epicycle
        tip_x, tip_y, _ = frame[-1]
        tip_y_flipped = -tip_y

        trace_x.append(tip_x)
        trace_y.append(tip_y_flipped)
        trace_line.set_data(trace_x, trace_y)

        # Build flipped frame for display
        flipped_frame = [(x, -y, r) for (x, y, r) in frame]

        # Draw epicycles in temp list
        prev_x, prev_y = 0.0, 0.0
        for (cx, cy, radius) in flipped_frame:
            circle = plt.Circle(
                (prev_x, prev_y), radius,
                color="#3a7ca5", fill=False, linewidth=0.5, alpha=0.25,
            )
            ax.add_patch(circle)
            line, = ax.plot(
                [prev_x, cx], [prev_y, cy],
                color="#7ec8e3", linewidth=0.8, alpha=0.6,
            )
            dynamic_artists.extend([circle, line])
            prev_x, prev_y = cx, cy

        dot, = ax.plot(prev_x, prev_y, "o", color="#ff6b6b", markersize=4)
        dynamic_artists.append(dot)

        return [trace_line] + dynamic_artists

    anim = FuncAnimation(
        fig, update, frames=n_frames,
        init_func=init, blit=True,
        interval=int(1000 / fps),
    )
    return anim


def save_animation(
    anim: FuncAnimation,
    out_path: str | Path,
    fmt: str = "gif",
    fps: int = 30,
) -> Path:
    """Save a FuncAnimation to disk as GIF or MP4.

    MP4 requires ffmpeg on PATH. Falls back to GIF if MP4 fails.

    Args:
        anim: matplotlib FuncAnimation.
        out_path: Destination path (extension may be overridden).
        fmt: "gif" or "mp4".
        fps: Frames per second.

    Returns:
        Path to the saved file.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "mp4":
        try:
            out_path = out_path.with_suffix(".mp4")
            anim.save(str(out_path), writer="ffmpeg", fps=fps)
            return out_path
        except Exception:
            # Fall back to GIF
            fmt = "gif"

    out_path = out_path.with_suffix(".gif")
    writer = PillowWriter(fps=fps)
    anim.save(str(out_path), writer=writer)
    plt.close(anim._fig)
    return out_path


def animation_to_gif_bytes(anim: FuncAnimation, fps: int = 30) -> bytes:
    """Save animation to in-memory GIF bytes (for Streamlit download).

    Newer matplotlib versions require PillowWriter to write to a real file
    path rather than a BytesIO buffer. We write to a NamedTemporaryFile,
    read the bytes back, then clean up.

    Args:
        anim: matplotlib FuncAnimation.
        fps: Frames per second.

    Returns:
        GIF bytes.
    """
    import tempfile, os
    writer = PillowWriter(fps=fps)
    with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        anim.save(tmp_path, writer=writer)
        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Gallery HTML page (Tier 2)
# ---------------------------------------------------------------------------

def render_gallery(
    results: list[dict[str, Any]],
    output_dir: str | Path,
) -> Path:
    """Generate a simple HTML gallery page from a batch of results.

    Args:
        results: List of dicts, each with:
                 - 'name': str (image label)
                 - 'png_path': str (path to reconstruction PNG)
                 - 'gif_path': str (optional, path to GIF)
        output_dir: Directory to write gallery.html.

    Returns:
        Path to gallery.html.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cards = ""
    for r in results:
        name = r.get("name", "Untitled")
        png = Path(r.get("png_path", "")).name
        gif = r.get("gif_path", "")
        gif_tag = f'<img src="{Path(gif).name}" alt="animation"/>' if gif else ""
        cards += f"""
        <div class="card">
          <h2>{name}</h2>
          <img src="{png}" alt="reconstruction"/>
          {gif_tag}
        </div>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="description" content="Image-to-Function Renderer output gallery"/>
<title>Image-to-Function Gallery</title>
<style>
  body {{ background:#0d0d0d; color:#f0f0f0; font-family: 'Segoe UI', sans-serif;
          margin:0; padding:2rem; }}
  h1 {{ text-align:center; font-size:2rem; margin-bottom:2rem; }}
  .gallery {{ display:flex; flex-wrap:wrap; gap:2rem; justify-content:center; }}
  .card {{ background:#1a1a1a; border-radius:12px; padding:1rem;
           max-width:460px; text-align:center; }}
  .card h2 {{ font-size:1rem; margin-bottom:0.75rem; color:#ccc; }}
  .card img {{ max-width:100%; border-radius:8px; }}
</style>
</head>
<body>
<h1>Image-to-Function Renderer Gallery</h1>
<div class="gallery">{cards}</div>
</body>
</html>"""

    out_file = output_dir / "gallery.html"
    out_file.write_text(html, encoding="utf-8")
    return out_file
