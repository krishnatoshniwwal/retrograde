"""One-shot script: strip the orphaned old dark-CSS block from app.py."""
import re

with open("app.py", encoding="utf-8") as f:
    text = f.read()

# The orphaned block starts right after the closing triple-quote of the NEW
# css st.markdown call and ends at the next """, unsafe_allow_html=True) pair
# that belongs to the OLD dark-mode st.markdown.
# Strategy: find the second occurrence of '""", unsafe_allow_html=True)'
# and delete everything between the end of the first such line and the end
# of the second such line (inclusive of the second).

marker = '""", unsafe_allow_html=True)'
first = text.index(marker)
second = text.index(marker, first + 1)

# Keep everything up to (and including) the first marker, then jump to
# just after the second marker.
cleaned = text[: first + len(marker)] + text[second + len(marker):]

with open("app.py", "w", encoding="utf-8") as f:
    f.write(cleaned)

print("Done — removed orphaned CSS block.")
print(f"Original length: {len(text)}, new length: {len(cleaned)}, removed: {len(text)-len(cleaned)} chars")
