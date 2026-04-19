import re

with open("tmp/generate_report.py", encoding="utf-8") as f:
    content = f.read()

replacements = {
    "\u2014": "--",      # em dash
    "\u2013": "-",       # en dash
    "\u2192": "->",      # right arrow
    "\u2190": "<-",      # left arrow
    "\u00b1": "+/-",     # plus-minus
    "\u00b9": "^1",      # superscript 1
    "\u207b": "-",       # superscript minus
    "\u00b2": "^2",      # superscript 2
    "\u00b3": "^3",      # superscript 3
    "\u00b8": "8",       # cedilla
    "\u2248": "~=",      # approximately equal
    "\u2026": "...",     # ellipsis
    "\u2265": ">=",      # greater than or equal
    "\u2264": "<=",      # less than or equal
    "\u00d7": "x",       # multiplication sign
    "\u221a": "sqrt",    # square root
    "\u03b1": "alpha",   # alpha
    "\u03b2": "beta",    # beta
    "\u2019": "'",       # right single quotation mark
    "\u2018": "'",       # left single quotation mark
    "\u201c": '"',       # left double quotation mark
    "\u201d": '"',       # right double quotation mark
    "\u2022": "*",       # bullet
    "\u00e8": "e",       # e with grave
    "\u00e0": "a",       # a with grave
    "\u00e9": "e",       # e with acute
    "\u2009": " ",       # thin space
    "\u00a0": " ",       # non-breaking space
    "\u00b0": " deg",    # degree sign
    "\u00b5": "u",       # micro sign
    "\u00ae": "(R)",     # registered trademark
    "\u2122": "(TM)",    # trademark
    "\u00a9": "(C)",     # copyright
}

for orig, repl in replacements.items():
    content = content.replace(orig, repl)

with open("tmp/generate_report.py", "w", encoding="utf-8") as f:
    f.write(content)

print("Fixed unicode issues in generate_report.py")
