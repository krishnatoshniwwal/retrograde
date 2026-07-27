"""
latex_export.py — Generate a LaTeX document containing the symbolic equations
for a processed image, and optionally compile it to PDF.

The output document lists:
  - A title/header section with metadata
  - One section per contour, showing x(t) and y(t) equations
  - Either spline piecewise polynomials or Fourier series terms
"""

from __future__ import annotations

import subprocess
import shutil
from pathlib import Path
from typing import Any

from src.symbolic import spline_to_latex_lines, fourier_to_latex_lines


# ---------------------------------------------------------------------------
# LaTeX document template
# ---------------------------------------------------------------------------

_PREAMBLE = r"""
\documentclass[10pt,a4paper]{article}
\usepackage{amsmath, amssymb, geometry, xcolor, parskip, microtype, hyperref}
\geometry{margin=2cm}
\hypersetup{colorlinks=true, linkcolor=blue, urlcolor=blue}
\definecolor{darkbg}{HTML}{0d0d0d}
\pagecolor{white}

\title{\textbf{Image-to-Function Renderer} \\ \large Symbolic Function Output}
\date{\today}
\author{Generated automatically}
""".strip()

_BEGIN_DOC = r"""
\begin{document}
\maketitle
\tableofcontents
\newpage
""".strip()

_END_DOC = r"\end{document}"


def _wrap_equations(lines: list[str]) -> str:
    """Wrap a list of LaTeX equation strings in an align* block."""
    if not lines:
        return ""
    inner = r" \\" + "\n    ".join(lines)
    return "\\begin{align*}\n    " + inner + "\n\\end{align*}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_latex_document(
    contour_results: list[dict[str, Any]],
    method: str = "fourier",
    image_name: str = "input",
    n_symbolic_terms: int = 8,
    max_contours: int = 20,
) -> str:
    """Build a complete LaTeX document string.

    Args:
        contour_results: List of dicts, each with:
            - 'coeffs' (list[EpicycleCoeff])  — for Fourier mode
            - 'segments' (list[SplineSegment]) — for piecewise mode
        method: "fourier" or "piecewise".
        image_name: Label used in the document title/section headers.
        n_symbolic_terms: How many terms / segments to include per contour.
        max_contours: Cap on number of contours to include (keeps PDF size sane).

    Returns:
        Full .tex document as a string.
    """
    sections = []
    contour_results = contour_results[:max_contours]

    for idx, result in enumerate(contour_results):
        section_lines = [
            f"\\section{{Contour {idx + 1} ({method} representation}})",
        ]

        if method == "fourier":
            coeffs = result.get("coeffs", [])
            if coeffs:
                eq_lines = fourier_to_latex_lines(
                    coeffs,
                    n_terms=n_symbolic_terms,
                    contour_index=idx,
                )
                section_lines.append(_wrap_equations(eq_lines))
            else:
                section_lines.append("\\textit{(no coefficients computed)}")

        else:  # piecewise
            segments = result.get("segments", [])
            if segments:
                eq_lines = spline_to_latex_lines(
                    segments,
                    contour_index=idx,
                    max_segments=n_symbolic_terms,
                )
                section_lines.append(_wrap_equations(eq_lines))
            else:
                section_lines.append("\\textit{(no segments computed)}")

        sections.append("\n".join(section_lines))

    intro = (
        f"\\section*{{Overview}}\n"
        f"This document contains the symbolic mathematical functions generated\n"
        f"from image \\texttt{{{image_name}}} using the "
        f"\\textbf{{{method}}} method.\n"
        f"A total of {len(contour_results)} contour(s) are represented below.\n"
    )

    parts = [_PREAMBLE, _BEGIN_DOC, intro] + sections + [_END_DOC]
    return "\n\n".join(parts)


def save_tex(content: str, out_path: str | Path) -> Path:
    """Write a LaTeX string to a .tex file.

    Args:
        content: Full LaTeX document string.
        out_path: Destination path (.tex extension).

    Returns:
        Path to the written file.
    """
    out_path = Path(out_path).with_suffix(".tex")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
    return out_path


def compile_pdf(tex_path: str | Path) -> Path | None:
    """Attempt to compile a .tex file to PDF using pdflatex.

    Requires pdflatex on the system PATH. Fails gracefully and returns
    None if the tool is not available or compilation fails.

    Args:
        tex_path: Path to the .tex file.

    Returns:
        Path to the PDF if successful, else None.
    """
    tex_path = Path(tex_path)
    if not tex_path.exists():
        return None

    if shutil.which("pdflatex") is None:
        # pdflatex not available — skip silently
        return None

    try:
        result = subprocess.run(
            [
                "pdflatex",
                "-interaction=nonstopmode",
                "-output-directory", str(tex_path.parent),
                str(tex_path),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        pdf_path = tex_path.with_suffix(".pdf")
        if result.returncode == 0 and pdf_path.exists():
            return pdf_path
    except Exception:
        pass

    return None


def export_latex(
    contour_results: list[dict[str, Any]],
    out_dir: str | Path,
    method: str = "fourier",
    image_name: str = "input",
    compile: bool = False,
    n_symbolic_terms: int = 8,
    max_contours: int = 20,
) -> dict[str, Path | None]:
    """Full export: build .tex, save it, optionally compile to PDF.

    Args:
        contour_results: List of pipeline result dicts.
        out_dir: Directory to write outputs.
        method: "fourier" or "piecewise".
        image_name: Label for the document.
        compile: Whether to attempt pdflatex compilation.
        n_symbolic_terms: Terms/segments per contour in the symbolic output.
        max_contours: Cap on contours included.

    Returns:
        Dict with:
          - 'tex': Path to .tex file
          - 'pdf': Path to .pdf (or None)
    """
    doc = build_latex_document(
        contour_results,
        method=method,
        image_name=image_name,
        n_symbolic_terms=n_symbolic_terms,
        max_contours=max_contours,
    )

    out_dir = Path(out_dir)
    tex_path = save_tex(doc, out_dir / f"{image_name}_functions.tex")

    pdf_path = None
    if compile:
        pdf_path = compile_pdf(tex_path)

    return {"tex": tex_path, "pdf": pdf_path}


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
            ax.plot([0.15, 0.85], [0.60, 0.60], color=DIV, linewidth=0.8,
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
                ax.plot([0.05, 0.95], [0.94, 0.94], color=DIV, linewidth=0.6,
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
