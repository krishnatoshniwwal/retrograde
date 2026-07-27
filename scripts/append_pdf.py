"""append_pdf.py — one-shot script to append generate_pdf_bytes to latex_export.py"""
from pathlib import Path

addition = r'''

# ---------------------------------------------------------------------------
# PDF generation via matplotlib — no pdflatex required
# ---------------------------------------------------------------------------

def _split_eq(eq: str, chunk: int = 65) -> list:
    """Split a long LaTeX equation string at top-level +/- boundaries."""
    if len(eq) <= chunk:
        return [eq]
    lines = []
    current = ""
    depth = 0
    for i, ch in enumerate(eq):
        if ch in "{(":
            depth += 1
        elif ch in "})":
            depth -= 1
        if depth == 0 and ch in "+-" and i > 0 and len(current) >= chunk:
            lines.append(current.rstrip())
            current = "  " + ch
        else:
            current += ch
    if current.strip():
        lines.append(current)
    return lines if lines else [eq]


def generate_pdf_bytes(
    contour_results,
    method: str = "fourier",
    image_name: str = "input",
    n_terms: int = 6,
    max_contours: int = 15,
) -> bytes:
    """Generate a PDF of symbolic equations using matplotlib PdfPages.

    No pdflatex required — uses matplotlib's built-in mathtext renderer.
    Each contour gets its own page with x(t) and y(t) equations.

    Args:
        contour_results: Pipeline result dicts (with 'coeffs' or 'segments').
        method: "fourier" or "piecewise".
        image_name: Label shown on the title page.
        n_terms: Fourier terms / spline segments per contour.
        max_contours: Cap on contours included.

    Returns:
        PDF as bytes.
    """
    import tempfile
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    BG = "#0d0d0d"
    FG = "#e2e8f0"
    ACCENT = "#c084fc"
    SUB = "#94a3b8"
    DIV = "#2d2d4e"

    results = contour_results[:max_contours]

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        with PdfPages(tmp_path) as pdf:

            # ── Title page ───────────────────────────────────────────────
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.patch.set_facecolor(BG)
            ax = fig.add_subplot(111)
            ax.set_facecolor(BG)
            ax.axis("off")
            ax.text(0.5, 0.72, "Image-to-Function Renderer", ha="center",
                    fontsize=22, fontweight="bold", color=ACCENT, transform=ax.transAxes)
            ax.text(0.5, 0.64, "Symbolic Function Output", ha="center",
                    fontsize=14, color="#a78bfa", transform=ax.transAxes)
            ax.axhline(y=0.60, xmin=0.15, xmax=0.85, color=DIV, linewidth=0.8,
                       transform=ax.transAxes)
            ax.text(0.5, 0.54, f"Image: {image_name}", ha="center", fontsize=12,
                    color=SUB, transform=ax.transAxes)
            ax.text(0.5, 0.48, f"Method: {method.title()}", ha="center", fontsize=12,
                    color=SUB, transform=ax.transAxes)
            ax.text(0.5, 0.42,
                    f"Contours: {len(results)}   |   Terms per contour: {n_terms}",
                    ha="center", fontsize=10, color="#64748b", transform=ax.transAxes)
            ax.text(0.5, 0.10,
                    "Generated with numpy.fft \xb7 scipy \xb7 sympy \xb7 matplotlib",
                    ha="center", fontsize=8, color="#374151",
                    transform=ax.transAxes, style="italic")
            pdf.savefig(fig, facecolor=BG)
            plt.close(fig)

            # ── One page per contour ──────────────────────────────────────
            for idx, result in enumerate(results):
                if method == "fourier":
                    eq_lines = fourier_to_latex_lines(
                        result.get("coeffs", []),
                        n_terms=n_terms,
                        contour_index=idx,
                    )
                else:
                    eq_lines = spline_to_latex_lines(
                        result.get("segments", []),
                        contour_index=idx,
                        max_segments=n_terms,
                    )

                if not eq_lines:
                    continue

                fig = plt.figure(figsize=(8.27, 11.69))
                fig.patch.set_facecolor(BG)
                ax = fig.add_subplot(111)
                ax.set_facecolor(BG)
                ax.axis("off")

                ax.text(0.5, 0.97, f"Contour {idx + 1}", ha="center", va="top",
                        fontsize=16, fontweight="bold", color=ACCENT, transform=ax.transAxes)
                ax.axhline(y=0.94, xmin=0.05, xmax=0.95, color=DIV, linewidth=0.6,
                           transform=ax.transAxes)

                y = 0.91
                for raw_eq in eq_lines:
                    for sub in _split_eq(raw_eq, chunk=70):
                        if y < 0.03:
                            break
                        try:
                            ax.text(0.04, y, f"${sub}$", ha="left", va="top",
                                    fontsize=8.5, color=FG, transform=ax.transAxes)
                        except Exception:
                            # mathtext can't render it — show as plain text
                            ax.text(0.04, y, sub, ha="left", va="top",
                                    fontsize=8.5, color=FG, transform=ax.transAxes)
                        y -= 0.055
                    y -= 0.015  # gap between x(t) and y(t)

                pdf.savefig(fig, facecolor=BG)
                plt.close(fig)

        with open(tmp_path, "rb") as f:
            return f.read()

    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
'''

target = Path("src/latex_export.py")
current = target.read_text(encoding="utf-8")
target.write_text(current + addition, encoding="utf-8")
print(f"Appended {len(addition)} chars. Total: {len(current + addition)}")
