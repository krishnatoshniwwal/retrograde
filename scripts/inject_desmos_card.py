"""Inject the clean Desmos card snippet into app.py."""

with open("app.py", encoding="utf-8") as f:
    lines = f.readlines()

with open("scripts/desmos_card_snippet.py", encoding="utf-8") as f:
    new_card_lines = f.readlines()

# Find start: the line with "with dl4:"
start = next(i for i, l in enumerate(lines) if "with dl4:" in l)
# Find end: the line with "# -- Footer" or "# ── Footer"
end = next(i for i, l in enumerate(lines) if "Footer" in l and l.strip().startswith("#"))

print(f"Replacing lines {start+1} to {end} with {len(new_card_lines)}-line snippet")

result = lines[:start] + new_card_lines + ["\n\n"] + lines[end:]

with open("app.py", "w", encoding="utf-8") as f:
    f.writelines(result)

print(f"Written: {len(result)} lines")

import subprocess, sys
r = subprocess.run([sys.executable, "-m", "py_compile", "app.py"], capture_output=True, text=True)
print("Syntax:", "PASSED" if r.returncode == 0 else f"FAILED\n{r.stderr}")
