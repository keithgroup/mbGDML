#!/usr/bin/env bash
# Build sphinx documentation like normal.
cd "${0%/*}"
rm -rf ./html/
sphinx-build -nT ./source/ ./html/
touch ./html/.nojekyll