#!/usr/bin/env bash
cd "${0%/*}"
rm -rf ./html
sphinx-build -nT ./source/ ./html/
touch ./html/.nojekyll
