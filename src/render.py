"""
render.py — Matplotlib-based static reconstruction and epicycle animation.

Supports:
- Static contour plots with gradient colouring (LineCollection)
- Night-sky / cream-paper themes (see src/theme.py)
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

from src.theme import NIGHT, PAPER, paper_colormap, register_colormaps

register_colormaps()


# ---------------------------------------------------------------------------
# Theme helpers
# ---------------------------------------------------------------------------

#: Night-sky theme (default) — matches the Streamlit UI.
NIGHT_THEME = {
    "bg": NIGHT["bg_deep"],
    "fg": NIGHT["fg"],
    "muted": NIGHT["muted"],
    "grid": NIGHT["border"],
    "circle": NIGHT["accent"],
    "arm": NIGHT["fg"],
    "trace": NIGHT["primary"],
    "tip": NIGHT["sakura"],
}
#: Cream-paper theme — for print and light backgrounds.
PAPER_THEME = {
    "bg": PAPER["bg"],
    "fg": PAPER["fg"],
    "muted": PAPER["muted"],
    "grid": PAPER["border"],
    "circle": PAPER["accent"],
    "arm": PAPER["fg"],
    "trace": PAPER["accent"],
    "tip": PAPER["sakura"],
}

# Backwards-compatible aliases ("dark" / "light" were the original names).
DARK_THEME = NIGHT_THEME
LIGHT_THEME = PAPER_THEME


def get_theme(theme: str = "night") -> dict[str, str]:
    """Return the palette dict for a theme name.

    ``"night"`` / ``"dark"`` -> night sky; ``"paper"`` / ``"light"`` -> cream paper.
    """
    return PAPER_THEME if theme in ("paper", "light") else NIGHT_THEME


def _apply_theme(fig: plt.Figure, ax: plt.Axes, theme: str = "night") -> None:
    """Apply the night or paper theme to a figure."""
    t = get_theme(theme)
    fig.patch.set_facecolor(t["bg"])
    ax.set_facecolor(t["bg"])
    ax.tick_params(colors=t["fg"])
    for spine in ax.spines.values():
        spine.set_edgecolor(t["grid"])


def _make_figure(config: dict[str, Any]) -> tuple[plt.Figure, plt.Axes]:
    """Create a figure with theme applied."""
    cfg_r = config.get("render", {})
    figsize = tuple(cfg_r.get("figsize", [10, 10]))
    theme = cfg_r.get("theme", "night")

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
    colormap: str = "aurora",
    line_width: float = 1.2,
    alpha: float = 0.95,
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
    colormap = cfg_r.get("colormap", "aurora")
    line_width = cfg_r.get("line_width", 1.2)
    theme = cfg_r.get("theme", "night")
    fg = get_theme(theme)["muted"]

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
        if title:
            ax.set_title(title, color=fg, pad=12, fontsize=11, fontfamily="monospace")
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

    if title:
        ax.set_title(title, color=fg, pad=12, fontsize=11, fontfamily="monospace")
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
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# Epicycle animation
# ---------------------------------------------------------------------------

def _draw_epicycles(
    ax: plt.Axes,
    frame_data: list[tuple[float, float, float]],
    theme: str = "night",
) -> list:
    """Draw one frame of epicycles: circles + arm lines.

    Args:
        ax: Matplotlib Axes (pre-cleared).
        frame_data: List of (cx, cy, radius) for each epicycle.
        theme: "dark" or "light".

    Returns:
        List of drawn artists (not used externally but kept for reference).
    """
    t = get_theme(theme)
    artists = []

    prev_x, prev_y = 0.0, 0.0
    for i, (cx, cy, radius) in enumerate(frame_data):
        # Draw circle outline
        circle = plt.Circle(
            (prev_x, prev_y), radius,
            color=t["circle"], fill=False, linewidth=0.5, alpha=0.3,
        )
        ax.add_patch(circle)
        # Draw arm line from previous centre to current
        line, = ax.plot(
            [prev_x, cx], [prev_y, cy],
            color=t["arm"], linewidth=0.8, alpha=0.7,
        )
        artists.extend([circle, line])
        prev_x, prev_y = cx, cy

    # Draw tip dot
    tip, = ax.plot(prev_x, prev_y, "s", color=t["tip"], markersize=3, alpha=0.9)
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
    theme = cfg_r.get("theme", "night")
    fps = cfg_r.get("fps", 30)
    n_frames = cfg_r.get("animation_frames", 300)
    figsize = tuple(cfg_r.get("figsize", [10, 10]))

    t_theme = get_theme(theme)

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
    trace_line, = ax.plot([], [], color=t_theme["trace"], linewidth=1.4, alpha=0.95)
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
                color=t_theme["circle"], fill=False, linewidth=0.6, alpha=0.28,
            )
            ax.add_patch(circle)
            line, = ax.plot(
                [prev_x, cx], [prev_y, cy],
                color=t_theme["arm"], linewidth=0.8, alpha=0.55,
            )
            dynamic_artists.extend([circle, line])
            prev_x, prev_y = cx, cy

        dot, = ax.plot(prev_x, prev_y, "s", color=t_theme["tip"], markersize=5)
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
<title>Retrograde Gallery</title>
<style>
  body {{ background:linear-gradient(180deg,{NIGHT["bg_deep"]},{NIGHT["bg"]} 40%,{NIGHT["sky_1"]});
          min-height:100vh; color:{NIGHT["fg"]}; font-family:system-ui,sans-serif;
          margin:0; padding:2rem; }}
  h1 {{ text-align:center; font-size:2.4rem; margin-bottom:2rem; letter-spacing:.04em;
        text-shadow:0 3px 0 {NIGHT["bg_deep"]}; }}
  .gallery {{ display:flex; flex-wrap:wrap; gap:2rem; justify-content:center; }}
  .card {{ background:{NIGHT["card"]}; border:2px solid {NIGHT["border"]};
           box-shadow:4px 4px 0 {NIGHT["bg_deep"]}; padding:1rem;
           max-width:460px; text-align:center; }}
  .card h2 {{ font-size:1rem; margin-bottom:0.75rem; color:{NIGHT["primary"]};
              text-transform:uppercase; letter-spacing:.1em; }}
  .card img {{ max-width:100%; border:2px solid {NIGHT["border"]}; }}
</style>
</head>
<body>
<h1>Retrograde Gallery</h1>
<div class="gallery">{cards}</div>
</body>
</html>"""

    out_file = output_dir / "gallery.html"
    out_file.write_text(html, encoding="utf-8")
    return out_file


# ---------------------------------------------------------------------------
# Print-ready poster export (Tier B-4)
# ---------------------------------------------------------------------------

def plot_poster(
    paths: list[np.ndarray],
    title: str = "RETROGRADE",
    subtitle: str = "Image -> Mathematical Functions",
    equation_snippet: str = "",
    config: dict[str, Any] | None = None,
    style: str = "paper",
) -> plt.Figure:
    """Create a poster-layout figure suitable for SVG/PDF export.

    Two styles are available:

    - ``"paper"`` (default): cream background, navy ink — print / plotter friendly.
    - ``"night"``: deep night-sky background, matching the app.

    Args:
        paths:            List of (N, 2) reconstructed path arrays.
        title:            Large title text (top of poster).
        subtitle:         Smaller subtitle text below the title.
        equation_snippet: Short equation string rendered in the bottom-right corner.
        config:           Optional config dict (uses render.colormap, render.line_width).
        style:            ``"paper"`` or ``"night"``.

    Returns:
        matplotlib Figure — call fig_to_svg_bytes or fig_to_pdf_bytes to export.
    """
    config = config or {}
    cfg_r  = config.get("render", {})
    colormap   = cfg_r.get("colormap", "aurora")
    cmap_label = colormap
    line_width = cfg_r.get("line_width", 1.8)

    if style == "night":
        bg, ink, muted, rule, accent = (
            NIGHT["bg_deep"], NIGHT["fg"], NIGHT["muted"], NIGHT["border"], NIGHT["primary"],
        )
    else:
        bg, ink, muted, rule, accent = (
            PAPER["bg"], PAPER["fg"], PAPER["muted"], PAPER["border"], PAPER["accent"],
        )
        colormap = paper_colormap(colormap)

    fig = plt.figure(figsize=(12, 14))

    # Layout: title area (top 12%), plot area (75%), footer (13%)
    ax_title  = fig.add_axes([0.0, 0.88, 1.0, 0.12])
    ax_plot   = fig.add_axes([0.04, 0.13, 0.92, 0.74])
    ax_footer = fig.add_axes([0.0, 0.0,  1.0, 0.13])

    fig.patch.set_facecolor(bg)
    for ax in (ax_title, ax_plot, ax_footer):
        ax.set_facecolor(bg)
        ax.axis("off")

    # ── Title area ──
    ax_title.text(
        0.5, 0.70, title,
        ha="center", va="center", transform=ax_title.transAxes,
        fontsize=38, fontweight="bold", color=ink,
        fontfamily="monospace",
    )
    ax_title.text(
        0.5, 0.26, subtitle,
        ha="center", va="center", transform=ax_title.transAxes,
        fontsize=12, color=muted, fontfamily="monospace",
    )
    # Pixel-style double rule under the title
    ax_title.axhline(y=0.04, color=accent, linewidth=3.0, xmin=0.04, xmax=0.96)
    ax_title.axhline(y=-0.04, color=rule, linewidth=1.0, xmin=0.04, xmax=0.96)

    # ── Reconstruction paths ──
    ax_plot.set_aspect("equal")

    flipped_paths = []
    for p in paths:
        if len(p) < 2:
            continue
        fp = p.copy()
        fp[:, 1] = -fp[:, 1]
        flipped_paths.append(fp)

    if flipped_paths:
        all_pts = np.vstack(flipped_paths)
        x_min, y_min = all_pts.min(axis=0)
        x_max, y_max = all_pts.max(axis=0)
        margin = max(x_max - x_min, y_max - y_min) * 0.06 + 1.0
        ax_plot.set_xlim(x_min - margin, x_max + margin)
        ax_plot.set_ylim(y_min - margin, y_max + margin)

        for fp in flipped_paths:
            segs = _path_to_segments(fp)
            colors = np.linspace(0, 1, len(segs))
            lc = LineCollection(
                segs, cmap=colormap, linewidth=line_width * 1.4, alpha=0.95,
            )
            lc.set_array(colors)
            ax_plot.add_collection(lc)

    # ── Footer ──
    n_contours = len(flipped_paths)
    footer_left = f"contours: {n_contours}  ·  colormap: {cmap_label}"
    ax_footer.axhline(y=0.88, color=rule, linewidth=1.0, xmin=0.04, xmax=0.96)
    ax_footer.text(
        0.04, 0.5, footer_left,
        ha="left", va="center", transform=ax_footer.transAxes,
        fontsize=9, color=muted, fontfamily="monospace",
    )
    if equation_snippet:
        ax_footer.text(
            0.96, 0.5, equation_snippet,
            ha="right", va="center", transform=ax_footer.transAxes,
            fontsize=8, color=accent, fontfamily="monospace",
        )

    return fig


def fig_to_svg_bytes(fig: plt.Figure) -> bytes:
    """Render a matplotlib figure to SVG bytes (vector, plotter-ready).

    Args:
        fig: matplotlib Figure.

    Returns:
        SVG bytes (UTF-8 encoded XML).
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="svg", bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf.read()


def fig_to_pdf_bytes(fig: plt.Figure) -> bytes:
    """Render a matplotlib figure to PDF bytes using the PDF backend.

    Args:
        fig: matplotlib Figure.

    Returns:
        PDF bytes.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="pdf", bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf.read()

