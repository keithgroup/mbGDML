#!/usr/bin/env bash
cd "${0%/*}"
sphinx-apidoc --force -o ./source/doc/ ../mbgdml/
sphinx-build -nT ./source/ ./html/
touch ./html/.nojekyll