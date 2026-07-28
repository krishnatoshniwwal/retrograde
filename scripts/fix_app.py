"""
Apply all changes to the original clean app.py:
1. Light mode CSS (replace dark CSS)
2. Add run_spline_for_desmos function after run_pipeline
3. Replace old Fourier Desmos card with spline-based card
4. Lock theme to "light", remove theme selectbox
5. Fix config default to light
6. Fix sidebar brand colors (remove neon cyan)
7. Fix footer colors

This script operates on the clean original (post git-stash pop).
"""

with open("app.py", encoding="utf-8") as f:
    text = f.read()

print(f"Original: {len(text.splitlines())} lines")

# ===========================================================================
# 1. REPLACE DARK CSS WITH LIGHT CSS
# ===========================================================================
old_css_start = "@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=JetBrains+Mono:wght@400;700&display=swap');"
old_css_end = ".katex { color: #00f0ff !important; }\n</style>"

new_css = """@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Inter:wght@400;600;700&display=swap');

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
</style>"""

i_start = text.find(old_css_start)
i_end = text.find(old_css_end) + len(old_css_end)
assert i_start > 0 and i_end > i_start, "CSS markers not found"
text = text[:i_start] + new_css + text[i_end:]
print("1. CSS replaced.")

# ===========================================================================
# 2. ADD run_spline_for_desmos AFTER run_pipeline
# ===========================================================================
pipeline_end_marker = "    return {\n        \"results\": results,\n        \"paths\": paths,\n        \"contours\": contours,\n        \"config\": config,\n        \"n_contours\": len(contours),\n        \"method\": method,\n    }\n\n\n# \u2500\u2500 SIDEBAR"
# Find the end of run_pipeline
run_pipeline_end = '    return {\n        "results": results,\n        "paths": paths,\n        "contours": contours,\n        "config": config,\n        "n_contours": len(contours),\n        "method": method,\n    }\n'
idx = text.find(run_pipeline_end)
assert idx >= 0, "run_pipeline return block not found"
insert_at = idx + len(run_pipeline_end)

spline_fn = '''

# \u2500\u2500 Spline pipeline \u2014 always used for Desmos export \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
# Splines are piecewise polynomials: Desmos evaluates them exactly with no
# sampling artefacts, regardless of which method the user selected in the UI.
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

'''
text = text[:insert_at] + spline_fn + text[insert_at:]
print("2. run_spline_for_desmos added.")

# ===========================================================================
# 3. FIX SIDEBAR BRAND COLORS (neon cyan -> dark)
# ===========================================================================
text = text.replace(
    "font-size:2.5rem; color:#00f0ff; text-shadow:0 0 15px rgba(0,240,255,0.6);",
    "font-size:2.5rem; color:#1a1a2e;",
)
text = text.replace(
    "font-size:0.7rem; color:#ff003c;",
    "font-size:0.7rem; color:#e11d48;",
)
print("3. Sidebar brand colors fixed.")

# ===========================================================================
# 4. REMOVE THEME SELECTBOX, LOCK TO LIGHT
# ===========================================================================
old_theme_block = '    theme = st.selectbox("THEME_PROFILE", ["dark", "light"])\n'
new_theme_block = '    theme = "light"  # light mode only\n'
if old_theme_block in text:
    text = text.replace(old_theme_block, new_theme_block)
    print("4. Theme selectbox removed.")
else:
    print("4. Theme selectbox not found (may be already removed).")

# ===========================================================================
# 5. FIX IMPORT: replace expression_list_to_json with expression_list_to_desmos_state
# ===========================================================================
text = text.replace(
    "    expression_list_to_json,\n",
    "    expression_list_to_desmos_state,\n",
)
print("5. Import updated.")

# ===========================================================================
# 6. REPLACE OLD FOURIER DESMOS CARD WITH SPLINE-BASED CARD
# ===========================================================================
old_desmos_card = '''    # \u2500\u2500 Desmos \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
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
                title=f"retrograde \u2014 {img_name}",
            )
            st.download_button(
                "[ DL_DESMOS_HTML ]",
                data=desmos_html_str.encode("utf-8"),
                file_name=f"{img_name}_desmos.html",
                mime="text/html",
                key="dl_dsm_html",
            )

            st.markdown(
                f\'<div class="info-box" style="font-size:0.75rem;">\\'
                f\'[ EXPR_COUNT ] {len(desmos_exprs)} expressions \xb7 \\'
                f\'{min(len(results), 60)} contours</div>\',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(\'<div class="info-box" style="font-size:0.8rem;">[ AWAITING_PIPELINE ]</div>\', unsafe_allow_html=True)
'''

new_desmos_card = '''    # \u2500\u2500 Desmos \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    with dl4:
        st.markdown("""
        <div class="dl-card">
            <div class="dl-icon">[ DSM ]</div>
            <div class="dl-title">DESMOS_GRAPH</div>
            <div class="dl-desc">Spline-based export (polynomial curves, exact in Desmos).</div>
        </div>
        """, unsafe_allow_html=True)

        if results:
            with st.spinner("Building spline graph\u2026"):
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
                    "[ DL_DESMOS_STATE ]",
                    data=desmos_state.encode("utf-8"),
                    file_name=f"{img_name}_desmos.json",
                    mime="application/json",
                    key="dl_dsm_json",
                    help="Desmos.com \u2192 hamburger menu \u2192 Load Graph",
                )

                desmos_html_str = build_desmos_html(
                    desmos_exprs,
                    title=f"retrograde \u2014 {img_name}",
                )
                st.download_button(
                    "[ DL_DESMOS_HTML ]",
                    data=desmos_html_str.encode("utf-8"),
                    file_name=f"{img_name}_desmos.html",
                    mime="text/html",
                    key="dl_dsm_html",
                )

                seg_count = sum(len(r.get("segments", [])) for r in dsm_results)
                st.markdown(
                    f\'<div class="info-box" style="font-size:0.75rem;">\\'
                    f\'[ SPLINE ] {len(dsm_results)} contours &middot; \\'
                    f\'{seg_count} segments &middot; {len(desmos_exprs)} expressions</div>\',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(\'<div class="info-box" style="font-size:0.8rem;">[ NO_CONTOURS ] Adjust edge thresholds.</div>\', unsafe_allow_html=True)
        else:
            st.markdown(\'<div class="info-box" style="font-size:0.8rem;">[ AWAITING_PIPELINE ]</div>\', unsafe_allow_html=True)
'''

if old_desmos_card in text:
    text = text.replace(old_desmos_card, new_desmos_card)
    print("6. Desmos card replaced with spline version.")
else:
    # Try a looser match on the key part
    old_key = '            desmos_json = expression_list_to_json(desmos_exprs)'
    if old_key in text:
        print("6. WARN: Exact Desmos card block not found but key line present.")
        print("   Manual review may be needed.")
    else:
        print("6. Desmos card not found -- already updated or different structure.")

# ===========================================================================
# 7. FIX FOOTER COLORS
# ===========================================================================
text = text.replace(
    '"font-size:0.75rem; color:#00f0ff; font-weight:700;',
    '"font-size:0.75rem; color:#1a1a2e; font-weight:700;',
)
text = text.replace(
    'style="font-size:0.75rem; color:#ff003c;',
    'style="font-size:0.75rem; color:#e11d48;',
)
print("7. Footer colors fixed.")

# ===========================================================================
# WRITE + SYNTAX CHECK
# ===========================================================================
with open("app.py", "w", encoding="utf-8") as f:
    f.write(text)

print(f"\nWritten: {len(text.splitlines())} lines")

import subprocess, sys
result = subprocess.run([sys.executable, "-m", "py_compile", "app.py"],
                       capture_output=True, text=True)
if result.returncode == 0:
    print("Syntax check: PASSED")
else:
    print("Syntax check: FAILED")
    print(result.stderr)
