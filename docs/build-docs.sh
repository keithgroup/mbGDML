#!/usr/bin/env bash
cd "${0%/*}"
rm -rf ./html
sphinx-build -nT ./source/ ./html/ -j auto
touch ./html/.nojekyll
