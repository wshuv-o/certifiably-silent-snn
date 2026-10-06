#!/usr/bin/env bash
# Render the manuscript .docx to PDF and page images for visual checking.
set -e
D=/tmp/render; rm -rf $D; mkdir -p $D
cp "$1" $D/m.docx
cd $D && soffice --headless --convert-to pdf m.docx >/dev/null 2>&1
pdftoppm -jpeg -r 60 m.pdf page
ls $D | head -40
mkdir -p "$2" && cp $D/page-*.jpg $D/m.pdf "$2"/
