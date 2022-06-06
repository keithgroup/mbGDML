#!/usr/bin/env bash
cd "${0%/*}"
rm -rf ./source/doc/
sphinx-apidoc --force --separate --private -o ./source/doc/ ../mbgdml/
sphinx-build -nT ./source/ ./html/
touch ./html/.nojekyll
