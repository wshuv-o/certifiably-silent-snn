#!/usr/bin/env bash
# Validate the manuscript's LaTeX independently of iopart.cls, which IOP distributes only from its own
# author site (it is not on CTAN -- confirmed 404 -- and not in Tectonic's bundle). We substitute
# `article` plus stubs for the iopart-specific macros. If this compiles, the only thing standing
# between this source and a PDF is the class file itself.
SRC=/mnt/d/proc/certifiably-silent-snn/research/paper/manuscript_nce.tex
WORK=$HOME/research/texval
TEC=$HOME/.local/bin/tectonic
rm -rf "$WORK"; mkdir -p "$WORK"; cd "$WORK" || exit 1
[ -f "$SRC" ] || { echo "SOURCE MISSING: $SRC"; exit 1; }

cat > stub.tex <<'EOF'
\documentclass[12pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{amssymb}
% stubs approximating iopart-specific commands, for validation only
\newcommand{\ead}[1]{\par\noindent Email: \texttt{#1}}
\providecommand{\address}[1]{\par\noindent #1}
\newcommand{\submitto}[1]{}
\newcommand{\NCE}{Neuromorphic Computing and Engineering}
\newcommand{\ackx}{\section*{Acknowledgements}}
\newcommand{\etal}{\textit{et al}}
\newcommand{\eref}[1]{(\ref{#1})}
\newcommand{\br}{\toprule}
\newcommand{\mr}{\midrule}
\newcommand{\ns}{}
\newcommand{\lineup}{}
\newenvironment{indented}{\begin{quote}}{\end{quote}}
EOF

# take everything after the \documentclass line, drop the iopart-only \usepackage duplicates,
# and rename \ack (iopart macro) to the stub name
sed -n '/^\\documentclass/,$p' "$SRC" | tail -n +2 \
  | grep -vE '^\\usepackage\{(graphicx|amsmath|amssymb)\}' \
  | sed 's/^\\ack$/\\ackx/' > body.tex

cat stub.tex body.tex > test.tex
echo "=== body lines: $(wc -l < body.tex) ==="
"$TEC" -X compile test.tex --keep-logs > tec_stdout.txt 2>&1
rc=$?
echo "=== tectonic exit=$rc ==="
if [ -f test.pdf ]; then
  echo "PDF PRODUCED: $(stat -c%s test.pdf) bytes"
  python3 - <<'PY'
import re
d=open("test.log",encoding="utf8",errors="ignore").read()
errs=[l for l in d.splitlines() if l.startswith("!")]
und=[l for l in d.splitlines() if "Undefined control sequence" in l]
ovf=[l for l in d.splitlines() if "Overfull" in l or "Underfull" in l]
print(f"hard errors        : {len(errs)}")
for e in errs[:10]: print("   ", e)
print(f"undefined commands : {len(und)}")
for e in und[:10]: print("   ", e)
print(f"over/underfull box : {len(ovf)} (cosmetic)")
PY
else
  echo "NO PDF. First errors from tectonic stdout:"
  grep -nE "^error|^!|not found|Undefined" tec_stdout.txt | head -20
  echo "--- log ---"
  grep -nE "^!|^l\.[0-9]+" test.log 2>/dev/null | head -20
fi
