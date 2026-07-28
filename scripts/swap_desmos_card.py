"""Replace the old Fourier Desmos card (lines 731-779) with the spline version."""

with open("app.py", encoding="utf-8") as f:
    lines = f.readlines()

print(f"Total lines: {len(lines)}")

# Find the Desmos section start (with dl4:)
start = None
end = None
for i, l in enumerate(lines):
    if "with dl4:" in l and start is None:
        start = i
    if start and i > start and "# ── Footer" in l:
        end = i
        break

print(f"Desmos card: lines {start+1} to {end}")

new_card = '''    # ── Desmos ───────────────────────────────────────────────────────────────
    with dl4:
        st.markdown("""
        <div class="dl-card">
            <div class="dl-icon">[ DSM ]</div>
            <div class="dl-title">DESMOS_GRAPH</div>
            <div class="dl-desc">Spline-based export (polynomial curves, exact in Desmos).</div>
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
                    "[ DL_DESMOS_STATE ]",
                    data=desmos_state.encode("utf-8"),
                    file_name=f"{img_name}_desmos.json",
                    mime="application/json",
                    key="dl_dsm_json",
                    help="desmos.com -> hamburger menu -> Load Graph",
                )

                desmos_html_str = build_desmos_html(
                    desmos_exprs,
                    title=f"retrograde - {img_name}",
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
                    f\'[ SPLINE ] {len(dsm_results)} contours | \\'
                    f\'{seg_count} segments | {len(desmos_exprs)} expressions</div>\',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(\'<div class="info-box" style="font-size:0.8rem;">[ NO_CONTOURS ] Adjust edge thresholds.</div>\', unsafe_allow_html=True)
        else:
            st.markdown(\'<div class="info-box" style="font-size:0.8rem;">[ AWAITING_PIPELINE ]</div>\', unsafe_allow_html=True)


'''

new_lines = lines[:start] + new_card.splitlines(keepends=True) + lines[end:]
with open("app.py", "w", encoding="utf-8") as f:
    f.writelines(new_lines)

print(f"Written: {len(new_lines)} lines")

import subprocess, sys
r = subprocess.run([sys.executable, "-m", "py_compile", "app.py"], capture_output=True, text=True)
print("Syntax:", "PASSED" if r.returncode == 0 else f"FAILED\n{r.stderr}")
